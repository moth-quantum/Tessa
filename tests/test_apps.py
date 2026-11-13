"""The example applications: MIDI codec and wavetable synthesiser."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from tessa import TessaGraph, decode_from_counts, encode


def ideal_roundtrip(graph):
    """An exact ``values -> values`` round trip: encode, then decode the statevector."""

    def run(values):
        circuit, enc = encode(graph, values)
        probs = Statevector(circuit).probabilities_dict()
        return decode_from_counts({"z": probs}, enc)

    return run


mido = pytest.importorskip("mido", reason="needs the [audio] extra")

from tessa.apps.midi import (
    MidiData,
    denormalize,
    encode_decode_midi,
    normalize,
    quantize_pitches,
    read_midi,
    write_midi,
)


@pytest.fixture
def tiny_score():
    """A short synthetic score: 8 notes, distinct pitches and durations."""
    n = 8
    return MidiData(
        onsets=np.arange(n) * 0.5,
        durations=np.linspace(0.2, 1.6, n),
        pitches=np.array([60, 62, 64, 65, 67, 69, 71, 72]),
        velocities=np.full(n, 80),
        channels=np.zeros(n, dtype=int),
        tempo=500_000,
        ticks_per_beat=480,
    )


@pytest.fixture(scope="module")
def midi_graph():
    """Capacity 16: 1 data qubit + 4 address qubits, big enough for a tiny score."""
    return TessaGraph.from_groups(5, [(0, [1, 2, 3, 4])])


# ----------------------------------------------------------------------
# MIDI I/O
# ----------------------------------------------------------------------
def test_normalize_denormalize_roundtrip():
    x = np.array([3.0, 7.0, 11.0])
    normed, lo, hi = normalize(x)
    assert normed.min() == 0.0 and normed.max() == 1.0
    assert denormalize(normed, lo, hi) == pytest.approx(x)


def test_normalize_constant_field():
    normed, lo, hi = normalize(np.full(3, 4.0))
    assert np.all(normed == 0.0) and lo == hi == 4.0


def test_quantize_pitches_clips_to_midi_range():
    assert list(quantize_pitches(np.array([-5.4, 60.4, 60.6, 999.0]))) == [0, 60, 61, 127]


def test_write_then_read_preserves_note_count(tmp_path, tiny_score):
    path = tmp_path / "tiny.mid"
    write_midi(path, tiny_score)
    back = read_midi(path)
    assert back.n_notes == tiny_score.n_notes
    assert list(back.pitches) == list(tiny_score.pitches)
    assert back.durations == pytest.approx(tiny_score.durations, abs=2e-3)


def test_read_bundled_score(data_file):
    score = read_midi(data_file("telemann_small.mid"))
    assert score.n_notes == 347  # fits the 448 heavy-hex capacity
    assert 0 <= score.pitches.min() <= score.pitches.max() <= 127
    assert np.all(score.durations > 0)


# ----------------------------------------------------------------------
# Two-pass encode/decode
# ----------------------------------------------------------------------
def test_ideal_two_pass_recovers_every_pitch(tiny_score, midi_graph):
    rec_ideal = ideal_roundtrip(midi_graph)
    out, report = encode_decode_midi(tiny_score, rec_ideal)
    assert list(out.pitches) == list(tiny_score.pitches)
    assert report["pitch_exact_fraction"] == 1.0
    assert report["pitch_mae_semitones"] == 0.0
    assert report["duration_mae_s"] < 1e-9
    # Untouched fields pass straight through.
    assert out.onsets == pytest.approx(tiny_score.onsets)
    assert list(out.velocities) == list(tiny_score.velocities)


@pytest.mark.parametrize("warp", ["linear", "log", "sqrt"])
def test_ideal_two_pass_is_warp_invariant(tiny_score, midi_graph, warp):
    rec_ideal = ideal_roundtrip(midi_graph)
    _out, report = encode_decode_midi(
        tiny_score, rec_ideal, duration_warp=warp, pitch_warp=warp
    )
    assert report["pitch_exact_fraction"] == 1.0
    assert report["duration_mae_s"] < 1e-9


def test_legacy_dynamic_range_recovery_flag(tiny_score, midi_graph):
    rec_ideal = ideal_roundtrip(midi_graph)
    _out, report = encode_decode_midi(tiny_score, rec_ideal, dynamic_range_recovery=True)
    assert report["dynamic_range"] == "min_max"


def test_encode_decode_writes_a_file(tmp_path, tiny_score, midi_graph):
    rec_ideal = ideal_roundtrip(midi_graph)
    out_path = tmp_path / "rec.mid"
    encode_decode_midi(tiny_score, rec_ideal, out_path=out_path)
    assert out_path.is_file()
    assert read_midi(out_path).n_notes == tiny_score.n_notes


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_bit_quantised_pass_stays_within_step(tiny_score, midi_graph, bits):
    """Bit-quantising a field costs at most half a level, exactly as designed."""
    rec_ideal = ideal_roundtrip(midi_graph)
    out, report = encode_decode_midi(
        tiny_score,
        rec_ideal,
        duration_quant_bits=bits,
        pitch_quant_bits=bits,
    )
    assert report["duration_quant_bits"] == bits
    assert report["duration_n_levels"] is None
    span = tiny_score.durations.max() - tiny_score.durations.min()
    assert report["duration_mae_s"] <= span / (2 * ((1 << bits) - 1)) + 1e-9
    # 8 bits over 8 distinct pitches is lossless.
    if bits == 8:
        assert list(out.pitches) == list(tiny_score.pitches)


def test_codebook_pass_recovers_exact_values(tiny_score, midi_graph):
    """Codebook mode encodes by rank and snaps back to the exact input values."""
    rec_ideal = ideal_roundtrip(midi_graph)
    out, report = encode_decode_midi(
        tiny_score,
        rec_ideal,
        pitch_quant_levels="auto",
        duration_quant_levels="auto",
    )
    assert report["pitch_n_levels"] == 8  # 8 distinct pitches
    assert report["duration_n_levels"] == 8
    assert list(out.pitches) == list(tiny_score.pitches)
    assert out.durations == pytest.approx(tiny_score.durations)


def test_codebook_snaps_durations_to_the_level_set(tiny_score, midi_graph):
    """With L levels, every reconstructed duration is one of the L codebook values."""
    rec_ideal = ideal_roundtrip(midi_graph)
    out, report = encode_decode_midi(tiny_score, rec_ideal, duration_quant_levels=4)
    assert report["duration_n_levels"] == 4
    assert np.unique(out.durations).size <= 4


def test_quant_levels_overrides_quant_bits_per_field(tiny_score, midi_graph):
    rec_ideal = ideal_roundtrip(midi_graph)
    _out, report = encode_decode_midi(
        tiny_score,
        rec_ideal,
        duration_quant_bits=3,
        duration_quant_levels="auto",
    )
    assert report["duration_n_levels"] == 8


def test_unquantised_pass_is_unchanged_by_the_pipeline_refactor(tiny_score, midi_graph):
    """The default path still reproduces the field to machine precision."""
    rec_ideal = ideal_roundtrip(midi_graph)
    out, report = encode_decode_midi(tiny_score, rec_ideal)
    assert report["pitch_n_levels"] is None
    assert out.durations == pytest.approx(tiny_score.durations, abs=1e-9)


def test_codebook_beats_continuous_under_noise(tiny_score, midi_graph):
    """Snapping to a small level set is what makes a noisy pass survive."""

    exact = ideal_roundtrip(midi_graph)

    def rec_lossy(v):
        # A deterministic stand-in for hardware: shrink the dynamic range (as noise
        # does) and add a small ripple.
        out = exact(v)
        mid = 0.5 * (out.min() + out.max())
        span = max(out.max() - out.min(), 1e-12)
        return mid + 0.8 * (out - mid) + 0.02 * span * np.sin(np.arange(out.size))

    _o1, plain = encode_decode_midi(tiny_score, rec_lossy)
    _o2, coded = encode_decode_midi(
        tiny_score,
        rec_lossy,
        pitch_quant_levels="auto",
        dynamic_range="min_max",
    )
    assert coded["pitch_mae_semitones"] < plain["pitch_mae_semitones"]


def test_too_many_notes_for_capacity(tiny_score):
    small = TessaGraph.from_groups(2, [(0, [1])])  # capacity 2
    with pytest.raises(ValueError, match="exceed capacity"):
        encode_decode_midi(tiny_score, ideal_roundtrip(small))


# ----------------------------------------------------------------------
# Wavetable synthesiser
# ----------------------------------------------------------------------
def test_wavetable_render_and_write(tmp_path, data_file):
    pytest.importorskip("scipy", reason="needs the [audio] extra")
    from tessa.apps.audio import midi_note_to_freq, parse_midi, render, write_wav

    assert midi_note_to_freq(69) == pytest.approx(440.0)
    notes, total = parse_midi(data_file("telemann_small.mid"))
    assert len(notes) == 347 and total > 0
    table = np.sin(2 * np.pi * np.arange(64) / 64)
    signal = render(notes[:20], table, sr=8_000, total=5.0)
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) == pytest.approx(0.9, abs=1e-6)  # normalised
    wav = tmp_path / "out.wav"
    write_wav(wav, signal, sr=8_000)
    assert wav.stat().st_size > 0


def test_adsr_envelope_shape():
    from tessa.apps.audio import adsr

    env = adsr(1000, 1000, attack=0.1, decay=0.1, sustain=0.5, release=0.1)
    assert env[0] == pytest.approx(0.0)
    assert env[-1] == pytest.approx(0.0)
    assert env.max() <= 1.0 + 1e-12
    assert env[500] == pytest.approx(0.5)  # sustain plateau


def test_wavetable_encoded_through_tessa_is_recovered(midi_graph):
    """The timbre really is the quantum-reconstructed waveform."""
    table = np.sin(2 * np.pi * np.arange(midi_graph.capacity) / midi_graph.capacity)
    rec = ideal_roundtrip(midi_graph)(table)
    assert np.abs(rec - table).max() < 1e-9
