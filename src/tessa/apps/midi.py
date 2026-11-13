"""Encode a MIDI score through the Tessa encoding (two passes).

A MIDI file is decomposed into per-note **pitch** and **duration** lists.  Each list is
optionally quantised and warped, then sent through *its own* encoding pass (two runs:
one for pitch, one for duration).  The reconstructed values are un-warped back to
their original ranges, the pitches are quantised to the nearest valid MIDI note
number, and a new MIDI file is written using the reconstructed pitches and durations
(keeping each note's original onset time, velocity and channel -- those could be run
through the encoder too).

Each pass is :func:`tessa.prepare_field` / :func:`tessa.restore_field` applied to one
field, so the same knobs are available per field: bit or codebook **quantisation**, an
invertible **warp**, and **dynamic-range recovery**.

The quantum round-trip itself is supplied as a callable ``roundtrip(x) -> x'`` mapping
a field to its reconstruction -- typically :func:`tessa.encode` followed by
:func:`tessa.decode`.  This lets the same codec drive a simulator or real hardware.

Requires ``mido`` (the ``audio`` extra): ``pip install "tessa[audio]"``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MidiData",
    "denormalize",
    "encode_decode_midi",
    "normalize",
    "quantize_pitches",
    "read_midi",
    "write_midi",
]


@dataclass
class MidiData:
    """Per-note MIDI content, in event order."""

    onsets: np.ndarray  # note start times (seconds)
    durations: np.ndarray  # note durations (seconds)
    pitches: np.ndarray  # MIDI note numbers (int, 0..127)
    velocities: np.ndarray  # MIDI velocities (int, 0..127)
    channels: np.ndarray  # MIDI channels (int)
    tempo: int  # microseconds per beat
    ticks_per_beat: int

    @property
    def n_notes(self) -> int:
        return len(self.pitches)


def read_midi(path) -> MidiData:
    """Parse a MIDI file into per-note arrays (in note-onset order)."""
    import mido

    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat
    tempo = 500_000
    t = 0.0
    active: dict[tuple[int, int], tuple[float, int]] = {}
    rows: list[tuple[float, float, int, int, int]] = []

    for msg in mido.merge_tracks(mid.tracks):
        t += mido.tick2second(msg.time, tpb, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)] = (t, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active:
                start, vel = active.pop(key)
                rows.append((start, max(t - start, 1e-3), msg.note, vel, msg.channel))

    rows.sort(key=lambda r: r[0])
    arr = np.array(rows, dtype=float) if rows else np.zeros((0, 5))
    return MidiData(
        onsets=arr[:, 0],
        durations=arr[:, 1],
        pitches=arr[:, 2].astype(int),
        velocities=arr[:, 3].astype(int),
        channels=arr[:, 4].astype(int),
        tempo=tempo,
        ticks_per_beat=tpb,
    )


def write_midi(path, data: MidiData) -> None:
    """Write a :class:`MidiData` back out to a standard MIDI file."""
    import mido

    mid = mido.MidiFile(ticks_per_beat=data.ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=int(data.tempo), time=0))

    events: list[tuple[float, int, mido.Message]] = []
    for on, du, p, v, ch in zip(
        data.onsets,
        data.durations,
        data.pitches,
        data.velocities,
        data.channels,
        strict=True,
    ):
        p = int(np.clip(round(p), 0, 127))
        v = int(np.clip(round(v), 1, 127))
        ch = int(ch)
        events.append(
            (float(on), 1, mido.Message("note_on", note=p, velocity=v, channel=ch))
        )
        events.append(
            (
                float(on + du),
                0,
                mido.Message("note_off", note=p, velocity=0, channel=ch),
            )
        )

    events.sort(key=lambda e: (e[0], e[1]))  # note_off before note_on at equal time
    prev = 0.0
    for tsec, _kind, msg in events:
        dt = tsec - prev
        prev = tsec
        msg.time = max(0, round(mido.second2tick(dt, data.ticks_per_beat, data.tempo)))
        track.append(msg)
    mid.save(str(path))


def normalize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Map ``values`` to ``[0, 1]``; returns ``(normed, vmin, vmax)``."""
    values = np.asarray(values, dtype=float)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax == vmin:
        return np.zeros_like(values), vmin, vmax
    return (values - vmin) / (vmax - vmin), vmin, vmax


def denormalize(normed: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Inverse of :func:`normalize`."""
    return np.asarray(normed, dtype=float) * (vmax - vmin) + vmin


def quantize_pitches(values: np.ndarray) -> np.ndarray:
    """Round to the nearest valid MIDI note number (0..127)."""
    return np.clip(np.rint(np.asarray(values, dtype=float)), 0, 127).astype(int)


def _roundtrip_field(
    values: np.ndarray,
    roundtrip: Callable[[np.ndarray], np.ndarray],
    warp="linear",
    dynamic_range: str = "none",
    pct: float = 2.0,
    quant_bits: int | None = None,
    quant_levels=None,
) -> tuple[np.ndarray, object]:
    """One field through quantise -> warp -> ``roundtrip`` -> dynamic range -> un-warp.

    Returns ``(reconstructed_field, prep)``; ``prep`` carries the codebook when
    ``quant_levels`` is used.
    """
    from ..transforms import prepare_field, restore_field

    prepared, prep = prepare_field(
        np.asarray(values, dtype=float),
        warp=warp,
        quant_bits=quant_bits,
        quant_levels=quant_levels,
    )
    recon = np.asarray(roundtrip(prepared), dtype=float)
    out = restore_field(recon, prep, dynamic_range=dynamic_range, pct=pct)
    return np.asarray(out, dtype=float).ravel(), prep


def encode_decode_midi(
    data: MidiData,
    roundtrip: Callable[[np.ndarray], np.ndarray],
    out_path=None,
    dynamic_range: str = "none",
    pitch_warp: str = "linear",
    duration_warp: str = "linear",
    pitch_quant_bits: int | None = None,
    duration_quant_bits: int | None = None,
    pitch_quant_levels=None,
    duration_quant_levels=None,
    pct: float = 2.0,
    dynamic_range_recovery: bool | None = None,
) -> tuple[MidiData, dict]:
    """Two-pass Tessa encode/decode of a MIDI score.

    Pass 1 encodes the pitch list, pass 2 the duration list. ``roundtrip`` is the
    quantum round trip (simulator or hardware) mapping a field to its
    reconstruction.

    Parameters
    ----------
    dynamic_range:
        Recovery applied to each noisy reconstruction, in the warped space, from a
        few summary scalars of the input field: ``"none"``, ``"min_max"``,
        ``"percentile"``, ``"mean_std"``, ``"abs_max"`` or ``"min_avg_max"`` (see
        :func:`tessa.transforms.apply_dynamic_range`).
    pitch_warp, duration_warp:
        Invertible pre-encoding warp per field (``"linear"``, ``"log"``, ``"sqrt"``).
        ``"log"`` on durations compresses the long-note tail so the encoder resolves
        the many short notes better.
    pitch_quant_bits, duration_quant_bits:
        Uniform quantisation of that field to ``2**bits`` levels over its own range;
        ``None`` keeps full float precision.
    pitch_quant_levels, duration_quant_levels:
        Codebook quantisation of that field: ``"auto"`` (every distinct value), an
        int ``L`` (``L`` data-derived levels) or an explicit list. Levels are encoded
        by rank at equal spacing in ``[0, 1]`` and snapped back afterwards, which
        suits pitches (already a discrete set) and is far more noise-tolerant.
        Overrides the matching ``*_quant_bits``.
    pct:
        Percentile used by ``dynamic_range="percentile"``.
    dynamic_range_recovery:
        Legacy flag: ``True`` means ``"min_max"``, ``False`` means ``"none"``.

    Returns ``(reconstructed_MidiData, report)``, and writes the reconstructed MIDI
    to ``out_path`` if given. ``report`` holds the note count, the settings used, and pitch/duration accuracy (exact pitch matches and mean absolute
    errors).
    """
    if dynamic_range_recovery is not None:
        dynamic_range = "min_max" if dynamic_range_recovery else "none"

    rec_pitch_f, pitch_prep = _roundtrip_field(
        data.pitches.astype(float),
        roundtrip,
        warp=pitch_warp,
        dynamic_range=dynamic_range,
        pct=pct,
        quant_bits=pitch_quant_bits,
        quant_levels=pitch_quant_levels,
    )
    rec_dur, dur_prep = _roundtrip_field(
        data.durations,
        roundtrip,
        warp=duration_warp,
        dynamic_range=dynamic_range,
        pct=pct,
        quant_bits=duration_quant_bits,
        quant_levels=duration_quant_levels,
    )

    rec_pitches = quantize_pitches(rec_pitch_f)
    rec_dur = np.clip(rec_dur, 1e-3, None)  # durations must stay positive

    recon = MidiData(
        onsets=data.onsets.copy(),
        durations=rec_dur,
        pitches=rec_pitches,
        velocities=data.velocities.copy(),
        channels=data.channels.copy(),
        tempo=data.tempo,
        ticks_per_beat=data.ticks_per_beat,
    )
    if out_path is not None:
        write_midi(out_path, recon)

    def n_levels(prep) -> int | None:
        return None if prep.codebook is None else int(prep.codebook.size)

    exact = int(np.sum(rec_pitches == data.pitches))
    report = {
        "n_notes": data.n_notes,
        "dynamic_range": dynamic_range,
        "pitch_warp": pitch_warp,
        "duration_warp": duration_warp,
        "pitch_quant_bits": pitch_quant_bits,
        "duration_quant_bits": duration_quant_bits,
        "pitch_n_levels": n_levels(pitch_prep),
        "duration_n_levels": n_levels(dur_prep),
        "pitch_exact_matches": exact,
        "pitch_exact_fraction": exact / max(data.n_notes, 1),
        "pitch_mae_semitones": float(np.mean(np.abs(rec_pitches - data.pitches))),
        "duration_mae_s": float(np.mean(np.abs(rec_dur - data.durations))),
    }
    return recon, report
