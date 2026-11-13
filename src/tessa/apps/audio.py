"""Minimal polyphonic wavetable synthesiser driven by quantum-reconstructed tables.

A *wavetable* is a single cycle of a periodic waveform stored as an array of samples.
To synthesise a note of frequency ``f`` we read the table with a phase accumulator
advancing ``f / sample_rate`` cycles per output sample and wrapping around the table
(with linear interpolation between table entries).  Every MIDI note becomes one such
oscillator, shaped by an ADSR amplitude envelope; all the notes are summed to produce
the final signal.

The point of this module is to be fed a wavetable that has been round-tripped through
the quantum encoder, so that the *timbre* of every note is literally the waveform
reconstructed from the quantum computer.

Requires ``mido`` and ``scipy`` (the ``audio`` extra): ``pip install "tessa[audio]"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Note",
    "adsr",
    "midi_note_to_freq",
    "parse_midi",
    "render",
    "render_note",
    "synthesize_midi",
    "write_wav",
]


# ----------------------------------------------------------------------
# MIDI parsing
# ----------------------------------------------------------------------
@dataclass
class Note:
    start: float  # seconds
    duration: float  # seconds
    freq: float  # Hz
    velocity: float  # 0..1


def midi_note_to_freq(note: int) -> float:
    """Equal-tempered frequency of a MIDI note number (A4 = 69 = 440 Hz)."""
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def parse_midi(path) -> tuple[list[Note], float]:
    """Parse a MIDI file into absolute-time :class:`Note` events.

    Returns ``(notes, total_seconds)``.  Requires the ``mido`` package.
    """
    import mido

    mid = mido.MidiFile(str(path))
    notes: list[Note] = []
    # Merge all tracks onto one timeline; mido gives per-message delta ticks.
    tempo = 500_000  # default 120 bpm (us per beat)
    tpb = mid.ticks_per_beat
    abs_time = 0.0
    active: dict[tuple[int, int], tuple[float, int]] = {}  # (chan,note)->(start,vel)

    for msg in mido.merge_tracks(mid.tracks):
        abs_time += mido.tick2second(msg.time, tpb, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)] = (abs_time, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active:
                start, vel = active.pop(key)
                notes.append(
                    Note(
                        start,
                        max(abs_time - start, 1e-3),
                        midi_note_to_freq(msg.note),
                        vel / 127.0,
                    )
                )
    total = max((n.start + n.duration for n in notes), default=0.0)
    return notes, total


# ----------------------------------------------------------------------
# Envelope + oscillator
# ----------------------------------------------------------------------
def adsr(
    n: int, sr: int, attack=0.01, decay=0.06, sustain=0.7, release=0.12
) -> np.ndarray:
    """Linear ADSR envelope of length ``n`` samples (note held for ``n/sr`` s)."""
    a = int(attack * sr)
    d = int(decay * sr)
    r = int(release * sr)
    env = np.full(n, sustain, dtype=float)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
    if d > 0:
        end = min(a + d, n)
        env[a:end] = np.linspace(1.0, sustain, end - a, endpoint=False)
    if r > 0:
        r = min(r, n)
        env[-r:] = np.linspace(env[-r], 0.0, r)
    return env


def render_note(note: Note, wavetable: np.ndarray, sr: int, release: float) -> np.ndarray:
    """Render one note through the wavetable oscillator with an ADSR envelope."""
    length = len(wavetable)
    n = int((note.duration + release) * sr) + 1
    t = np.arange(n)
    phase = (note.freq / sr) * t  # cycles
    idx = (phase % 1.0) * length
    i0 = np.floor(idx).astype(int) % length
    i1 = (i0 + 1) % length
    frac = idx - np.floor(idx)
    wave = wavetable[i0] * (1.0 - frac) + wavetable[i1] * frac
    env = adsr(n, sr, release=release)
    return wave * env * note.velocity


def render(
    notes: list[Note],
    wavetable: np.ndarray,
    sr: int = 44_100,
    release: float = 0.12,
    total: float | None = None,
    normalize: float = 0.9,
) -> np.ndarray:
    """Render all notes with ``wavetable`` and mix down to a mono signal."""
    wavetable = np.asarray(wavetable, dtype=float)
    if total is None:
        total = max((n.start + n.duration for n in notes), default=0.0)
    length = int((total + release + 0.2) * sr) + 1
    buf = np.zeros(length, dtype=float)
    for note in notes:
        s = int(note.start * sr)
        voice = render_note(note, wavetable, sr, release)
        e = min(s + len(voice), length)
        buf[s:e] += voice[: e - s]
    peak = np.max(np.abs(buf))
    if peak > 0 and normalize:
        buf *= normalize / peak
    return buf


def write_wav(path, signal: np.ndarray, sr: int = 44_100) -> None:
    """Write a mono float signal to a 16-bit PCM WAV file."""
    from scipy.io import wavfile

    pcm = np.clip(signal, -1.0, 1.0)
    wavfile.write(str(path), sr, (pcm * 32767).astype(np.int16))


def synthesize_midi(
    midi_path,
    wavetable: np.ndarray,
    out_wav,
    sr: int = 44_100,
    release: float = 0.12,
) -> tuple[np.ndarray, int]:
    """End-to-end: parse ``midi_path``, render with ``wavetable``, write ``out_wav``.

    Returns ``(signal, sample_rate)``.
    """
    notes, total = parse_midi(midi_path)
    signal = render(notes, wavetable, sr=sr, release=release, total=total)
    write_wav(out_wav, signal, sr)
    return signal, sr
