"""App. "Other applications" and "Practical considerations": Table midi, the pitch columns
of Table drr, and Fig. sound - all as SIMULATIONS.

Provenance (see ``results/simulation/midi_audio/README.md``): the paper labels Table midi
and the pitch columns of Table drr as hardware ("Fez score"), but every number in them
is the output of a FakeFez noise-model simulation in the original analysis notebook
(an earlier code base of the author, not published): uniform machine-average noise
from the FakeFez calibration (``backend_rates``), each data qubit simulated alone with
its address qubits, 8000 shots, per-cell simulator seeds drawn from
``default_rng(1)``. The one real ibm_fez MIDI run saved no counts. Fig. sound shows an
ibm_fez wavetable whose data were not saved.

This module re-runs that simulation with the :mod:`tessa` API (same circuits, same
noise model, same transpilation and the same seed stream; the reconstructions agree
with the original code to 2e-16) and reproduces every number of Table midi and of the
pitch columns of Table drr. For Fig. sound it draws a re-creation from the same
FakeFez simulation, clearly labelled; the paper's image is the reference.

Inputs: ``data/midi/telemann_small.mid`` (347 notes). Cached outputs:
``results/simulation/midi_audio/{midi_fakefez.json,wavetable_fakefez.npz}``; recomputing
takes about 15 s.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from qiskit.quantum_info import Statevector

from tessa import (
    NATIVE_BASIS,
    TessaGraph,
    apply_dynamic_range,
    backend_rates,
    build_average_noise_model,
    build_tessa_circuit,
    decode_cells,
    encoding_coupling_map,
    graph_from_backend,
)
from tessa_paper._check import kind_of, with_kinds
from tessa_paper.paths import DATA, RESULTS

__all__ = [
    "CACHE_DIR",
    "KNOWN_DISCREPANCIES",
    "PAPER_VALUES",
    "check_against_paper",
    "compute_all",
    "fez_graph",
    "fig_sound",
    "midi_results",
    "midi_roundtrip",
    "reconstruct_exact",
    "reconstruct_noisy",
    "table_drr_pitch",
    "table_midi",
    "wavetable",
]

CACHE_DIR = RESULTS / "simulation" / "midi_audio"
MIDI_FILE = DATA / "midi" / "telemann_small.mid"
SHOTS = 8000
SEED = 1
#: The pitch DRR methods of Table drr (``abs_max`` is printed as "---").
PITCH_METHODS = ("none", "abs_max", "min_max", "percentile", "mean_std", "min_avg_max")
DURATION_WARPS = ("linear", "log", "sqrt")

#: Numbers as printed: Table 7 (tab:midi) and the pitch columns of Table 6 (tab:drr),
#: as (exact %, pitch MAE st[, duration MAE s]).
PAPER_VALUES: dict[str, object] = {
    "tab:midi": {
        "noiseless simulation": ("100.0", "0.00", "0.000"),
        "hardware, raw": ("18.7", "2.13", "0.491"),
        "hardware, linear": ("49.3", "0.66", "0.208"),
        "hardware, log": ("49.3", "0.66", "0.145"),
        "hardware, square root": ("49.3", "0.66", "0.117"),
    },
    "tab:drr pitch": {
        "none": ("18.7", "2.13"),
        "abs_max": ("---", "---"),
        "min_max": ("8.4", "1.83"),
        "percentile": ("12.4", "1.67"),
        "mean_std": ("49.3", "0.66"),
        "min_avg_max": ("23.3", "1.13"),
    },
    "fig:sound": {"samples": 448, "shots": 4096, "drr": "min_max", "device": "ibm_fez"},
}

#: Paper-vs-data differences. Keys that are items of :func:`check_against_paper` mark
#: rows that do not match; the others concern labels and captions.
KNOWN_DISCREPANCIES: dict[str, str] = {
    "tab:midi / tab:drr pitch: source": (
        "The rows are labelled hardware ('hardware, raw', 'Fez score') but every number is "
        "the FakeFez noise-model simulation reproduced here; the one real ibm_fez MIDI run "
        "(2026-07-29, 8192 shots, min_max: 39.5 % exact, 1.62 st, 0.469 s) saved no counts "
        "and is not in the paper."
    ),
    "tab:drr pitch abs_max exact (%)": (
        "Printed as '---' (the original notebook did not run abs_max); the same simulation "
        "gives 0.3 %, since MIDI note numbers 62-88 are not anchored at zero."
    ),
    "tab:drr pitch abs_max MAE (st)": (
        "Printed as '---'; the same simulation gives 7.56 semitones."
    ),
    "fig:sound: data": (
        "The ibm_fez wavetable of the paper saved no counts, job id or table, so the figure "
        "here is a FakeFez-simulation re-creation."
    ),
    "fig:sound: caption": (
        "The caption says 4096 shots and min_max DRR, but the notebook that made the "
        "figure used 8192 shots and divided by the maximum (wt / |max(wt)|)."
    ),
}


# ---------------------------------------------------------------------------
# The simulation of the original notebook, with the tessa API
# ---------------------------------------------------------------------------
def fez_graph() -> tuple[TessaGraph, dict]:
    """The Tessa graph of FakeFez (156 qubits, capacity 448) and its average rates."""
    from qiskit_ibm_runtime.fake_provider import FakeFez

    fez = FakeFez()
    return graph_from_backend(fez), backend_rates(fez)


def _cell(degree: int) -> TessaGraph:
    return TessaGraph.from_groups(degree + 1, [(0, list(range(1, degree + 1)))])


def reconstruct_noisy(
    graph: TessaGraph,
    rates: dict,
    values01: np.ndarray,
    shots: int = SHOTS,
    seed: int = SEED,
) -> np.ndarray:
    """Per-cell noisy round trip of a capacity vector in [0, 1] (theta = pi * v).

    Port of ``qcrank_graph.reconstruct_noisy_by_groups``: each data qubit and its
    address qubits are encoded alone, transpiled to the native basis
    (``optimization_level=1``, ``seed_transpiler=0``), simulated with
    :func:`tessa.build_average_noise_model` on ``rates`` and Aer seed
    ``default_rng(seed).integers(1e9)`` drawn per cell in group order, then decoded.
    Returns the reconstruction (capacity,).
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    packed = graph.pack(np.asarray(values01, dtype=float))
    rng = np.random.default_rng(seed)
    out = {}
    for g in graph.groups:
        sub = _cell(g.degree)
        qc = build_tessa_circuit(sub, np.asarray(packed[g.site], float))
        qc.measure_all()
        tqc = transpile(
            qc,
            basis_gates=NATIVE_BASIS,
            coupling_map=encoding_coupling_map(sub),
            optimization_level=1,
            seed_transpiler=0,
        )
        sim = AerSimulator(noise_model=build_average_noise_model(sub, rates))
        counts = (
            sim.run(tqc, shots=shots, seed_simulator=int(rng.integers(1_000_000_000)))
            .result()
            .get_counts()
        )
        cell = decode_cells({"z": counts}, sub.groups, sub.n_qubits, "single", 0.0)
        out[g.site] = cell[sub.groups[0].site]
    return graph.unpack(out)


def reconstruct_exact(graph: TessaGraph, values01: np.ndarray) -> np.ndarray:
    """Noiseless per-cell round trip from exact statevector probabilities."""
    packed = graph.pack(np.asarray(values01, dtype=float))
    out = {}
    for g in graph.groups:
        sub = _cell(g.degree)
        probs = Statevector(build_tessa_circuit(sub, packed[g.site])).probabilities_dict()
        cell = decode_cells({"z": probs}, sub.groups, sub.n_qubits, "single", 0.0)
        out[g.site] = cell[sub.groups[0].site]
    return graph.unpack(out)


def _field(
    values: np.ndarray,
    capacity: int,
    reconstruct: Callable[[np.ndarray], np.ndarray],
    warp: str,
    dynamic_range: str,
) -> np.ndarray:
    """warp -> normalise -> pad -> round trip -> clip [0,1] -> denormalise -> DRR -> unwarp."""
    from tessa.apps.midi import denormalize, normalize
    from tessa.transforms import get_warp

    w = get_warp(warp)
    warped = w.forward(np.asarray(values, dtype=float))
    normed, vmin, vmax = normalize(warped)
    padded = np.zeros(capacity)
    padded[: len(values)] = normed
    rec = np.clip(np.asarray(reconstruct(padded), float)[: len(values)], 0.0, 1.0)
    rec = apply_dynamic_range(denormalize(rec, vmin, vmax), warped, method=dynamic_range)
    return w.inverse(rec)


def midi_roundtrip(
    reconstruct: Callable[[np.ndarray], np.ndarray],
    capacity: int,
    dynamic_range: str = "none",
    duration_warp: str = "linear",
    path: Path = MIDI_FILE,
) -> dict[str, float]:
    """Pitch and duration lists of a MIDI file through two round trips (Table midi).

    Port of the MIDI round trip of the original notebook (not published): each list is warped,
    normalised to [0, 1], padded to ``capacity`` and sent through ``reconstruct``; the
    result is de-normalised, DRR-corrected (``dynamic_range``, in the warped space, both
    fields), un-warped; pitches are rounded to MIDI notes, durations kept >= 1 ms.
    Returns {n_notes, pitch_exact_fraction, pitch_mae_semitones, duration_mae_s}.
    """
    from tessa.apps.midi import quantize_pitches, read_midi

    score = read_midi(path)
    pitch = _field(
        score.pitches.astype(float), capacity, reconstruct, "linear", dynamic_range
    )
    dur = _field(score.durations, capacity, reconstruct, duration_warp, dynamic_range)
    pitches = quantize_pitches(pitch)
    dur = np.clip(dur, 1e-3, None)
    return {
        "n_notes": int(score.n_notes),
        "pitch_exact_fraction": float(np.mean(pitches == score.pitches)),
        "pitch_mae_semitones": float(np.mean(np.abs(pitches - score.pitches))),
        "duration_mae_s": float(np.mean(np.abs(dur - score.durations))),
    }


def _compute_midi() -> dict:
    graph, rates = fez_graph()
    cache: dict[bytes, np.ndarray] = {}

    def noisy(v: np.ndarray) -> np.ndarray:
        key = np.asarray(v, float).tobytes()
        if key not in cache:
            cache[key] = reconstruct_noisy(graph, rates, v)
        return cache[key]

    cap = graph.capacity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = {"noiseless": midi_roundtrip(lambda v: reconstruct_exact(graph, v), cap)}
        for m in PITCH_METHODS:
            runs[f"pitch_drr={m}"] = midi_roundtrip(noisy, cap, dynamic_range=m)
        for w in DURATION_WARPS:
            runs[f"mean_std, duration_warp={w}"] = midi_roundtrip(
                noisy, cap, dynamic_range="mean_std", duration_warp=w
            )
    return {
        "kind": "SIMULATION (FakeFez machine-average noise), not hardware",
        "input": "data/midi/telemann_small.mid",
        "device_model": "FakeFez (qiskit-ibm-runtime), backend_rates averages",
        "capacity": cap,
        "shots": SHOTS,
        "seed": SEED,
        "rates": rates,
        "runs": runs,
    }


def midi_results(recompute: bool = False) -> dict:
    """All MIDI round trips of Table midi and Table drr (pitch), FakeFez simulation.

    Reads ``results/simulation/midi_audio/midi_fakefez.json``; ``recompute=True`` re-runs
    the simulation (about 12 s). Returns {kind, input, capacity, shots, seed, rates,
    runs: {name: midi_roundtrip report}} with run names ``noiseless``,
    ``pitch_drr=<method>`` and ``mean_std, duration_warp=<warp>``.
    """
    path = CACHE_DIR / "midi_fakefez.json"
    if recompute or not path.exists():
        return _compute_midi()
    return json.loads(path.read_text())


def table_midi(recompute: bool = False) -> pd.DataFrame:
    """Table midi (SIMULATION): exact-pitch %, pitch MAE (st), duration MAE (s).

    Rows as in the paper; the "hardware" rows of the paper are the FakeFez simulation.
    Input: :func:`midi_results`. Values unrounded.
    """
    runs = midi_results(recompute)["runs"]
    rows = {
        "noiseless simulation": runs["noiseless"],
        "FakeFez sim, raw (paper: hardware, raw)": runs["pitch_drr=none"],
        "FakeFez sim, linear (paper: hardware, linear)": runs[
            "mean_std, duration_warp=linear"
        ],
        "FakeFez sim, log (paper: hardware, log)": runs["mean_std, duration_warp=log"],
        "FakeFez sim, square root (paper: hardware, square root)": runs[
            "mean_std, duration_warp=sqrt"
        ],
    }
    return pd.DataFrame(
        {
            name: {
                "exact (%)": 100 * r["pitch_exact_fraction"],
                "pitch (st)": r["pitch_mae_semitones"],
                "dur. (s)": r["duration_mae_s"],
            }
            for name, r in rows.items()
        }
    ).T


def table_drr_pitch(recompute: bool = False) -> pd.DataFrame:
    """Pitch columns of Table drr (SIMULATION): exact % and pitch MAE per DRR method.

    Input: :func:`midi_results` (runs ``pitch_drr=<method>``, linear duration warp).
    """
    runs = midi_results(recompute)["runs"]
    return pd.DataFrame(
        {
            m: {
                "exact (%)": 100 * runs[f"pitch_drr={m}"]["pitch_exact_fraction"],
                "MAE (st)": runs[f"pitch_drr={m}"]["pitch_mae_semitones"],
            }
            for m in PITCH_METHODS
        }
    ).T


# ---------------------------------------------------------------------------
# Fig. sound (re-creation)
# ---------------------------------------------------------------------------
def _compute_wavetable() -> dict[str, np.ndarray]:
    graph, rates = fez_graph()
    n = graph.capacity
    sine = np.sin(2 * np.pi * np.arange(n) / n)
    x01 = (sine + 1) / 2  # value range (-1, 1) -> theta = pi * (v + 1) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ideal = 2 * reconstruct_exact(graph, x01) - 1
        noisy = 2 * reconstruct_noisy(graph, rates, x01) - 1
    return {"sine": sine, "noiseless": ideal, "fakefez": noisy}


def wavetable(recompute: bool = False) -> dict[str, np.ndarray]:
    """The 448-sample sine through Tessa on FakeFez (Fig. sound re-creation, SIMULATION).

    Reads ``results/simulation/midi_audio/wavetable_fakefez.npz`` (``recompute=True``:
    about 3 s). Returns ``sine`` (input), ``noiseless`` (exact) and ``fakefez``
    (machine-average FakeFez noise, 8000 shots, seed 1, raw decode), all in [-1, 1].
    The original notebook printed a mean absolute error of 0.289 for ``fakefez``.
    """
    path = CACHE_DIR / "wavetable_fakefez.npz"
    if recompute or not path.exists():
        return _compute_wavetable()
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def fig_sound(save: Path | None = None, recompute: bool = False) -> Figure:
    """Fig. sound, RE-CREATED from a FakeFez simulation (the paper's hardware data are lost).

    Input sine (dashed), the noiseless reconstruction and the FakeFez-noise
    reconstruction after ``min_max`` DRR (as the caption states; the original notebook
    divided the table by its maximum instead, see REPRODUCIBILITY.md). The title names
    this choice. Returns the figure;
    saves it if ``save`` is given (do not overwrite the paper's ``sound_example.png``;
    the notebook writes ``sound_example_recreation.png``).
    """
    d = wavetable(recompute)
    rec = apply_dynamic_range(d["fakefez"], d["sine"], method="min_max")
    n = len(d["sine"])
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(d["sine"], "k--", lw=1, label="input sine")
    ax.plot(
        d["noiseless"],
        color="#1f77b4",
        lw=1.2,
        alpha=0.6,
        label="noiseless reconstruction",
    )
    ax.plot(
        rec, color="#d66e14", lw=1, alpha=0.9, label="FakeFez noise model, min_max DRR"
    )
    ax.set_xlabel(f"wavetable index (0..{n - 1})")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Re-creation (FakeFez simulation, {SHOTS} shots, min_max DRR): "
        f"{n}-sample sine wavetable.\nThe paper's ibm_fez data were not saved.",
        fontsize=9,
    )
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=150)
    return fig


def compute_all(write: bool = True) -> dict:
    """Recompute both caches (about 15 s); if ``write``, store them in ``CACHE_DIR``.

    Writes ``midi_fakefez.json`` and ``wavetable_fakefez.npz``. Returns both results.
    """
    midi, wt = _compute_midi(), _compute_wavetable()
    if write:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "midi_fakefez.json").write_text(json.dumps(midi, indent=1) + "\n")
        np.savez_compressed(CACHE_DIR / "wavetable_fakefez.npz", **wt)
    return {"midi": midi, "wavetable": wt}


# ---------------------------------------------------------------------------
# Check against the paper
# ---------------------------------------------------------------------------
def check_against_paper(recompute: bool = False) -> pd.DataFrame:
    """Every number of Table midi and of the pitch columns of Table drr, at printed precision.

    The reproduction is the FakeFez SIMULATION (the paper's rows say hardware; see
    :data:`KNOWN_DISCREPANCIES`). ``abs_max`` ("---" in the paper) is listed with the
    simulated value and ``match`` False. Returns [item, paper, reproduced, match,
    kind] (see :mod:`tessa_paper._check`).
    """
    midi = table_midi(recompute)
    drr = table_drr_pitch(recompute)
    rows = []
    fmts = ("{:.1f}", "{:.2f}", "{:.3f}")
    for (label, printed), (_, got) in zip(
        PAPER_VALUES["tab:midi"].items(), midi.iterrows(), strict=True
    ):
        for col, p, f in zip(midi.columns, printed, fmts, strict=True):
            s = f.format(got[col])
            rows.append((f"tab:midi {label} {col}", p, s, s == p, kind_of(p)))
    for method, printed in PAPER_VALUES["tab:drr pitch"].items():
        for col, p, f in zip(drr.columns, printed, fmts[:2], strict=True):
            s = f.format(drr.loc[method, col])
            rows.append((f"tab:drr pitch {method} {col}", p, s, s == p, kind_of(p)))
    wt = wavetable(recompute)
    mae = float(np.abs(wt["fakefez"] - wt["sine"]).mean())
    rows.append(
        (
            "fig:sound re-creation: FakeFez mean |err| (original notebook printed 0.289)",
            "not in paper",
            f"{mae:.3f}",
            f"{mae:.3f}" == "0.289",
            "not printed",
        )
    )
    return with_kinds(
        pd.DataFrame(rows, columns=["item", "paper", "reproduced", "match", "kind"])
    )
