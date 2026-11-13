"""App. "Encoding DNA and protein sequences": Fig. genetic-results, Table genetic, macros,
Fig. constellation and the shot-count claim.

Inputs:

* ``data/sequences/genetic_sequences.txt`` - the seven sequences (symbols -> alphabet
  index: DNA ``ACGT``, amino acids in Kyte-Doolittle order ``IVLFCMAGTSWYPHEQDNKR``).
* ``results/hardware/genetic/<run>.json|.npz`` - the two ibm_phoenix runs (tickets and
  packed shots). Every hardware cell of Table genetic is re-derived here from the raw
  shots: the ticket's graph (``edges``) and ``slot_order`` re-create the encoding, the
  shots are turned into counts with ``qiskit.primitives.BitArray`` and decoded with
  :func:`tessa.symbols_from_counts` for every DRR method.
* ``results/hardware/genetic/constellation_points.npz`` - the constellation point sets
  (4, 16, 20 and 400 points) with which the stored shots decode to the paper's
  table. :func:`tessa.constellation` computes them with a short repulsion iteration
  whose last digits depend on the platform, and two hardware cells of Table genetic
  change if the start points move by a relative 1e-10, so the decode reads the stored
  sets by default (:func:`constellation_points`).
* ``results/simulation/genetic/metrics.json`` - the "no noise" and "simulated noise"
  columns (Aer, per-cell, calibration snapshot of ibm_phoenix of 2026-10-09);
  :func:`rerun_simulation` regenerates them.
* ``results/simulation/genetic/shot_sweep.json`` - noiseless constellation errors vs
  shots (claim: 32768 is the smallest power of two with no error).
* ``figures/paper/results_{table,macros}.tex`` - the files the paper
  ``\\input``s, to compare the regenerated LaTeX with.

Ported from the original analysis notebook and scripts (not published).
"""

from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from tessa import (
    DYNAMIC_RANGE_METHODS,
    aer_sampler,
    calibrated_sampler,
    constellation,
    decode_symbols,
    encode_symbols,
    graph_from_backend,
    graph_from_coupling,
    probe_qubits,
    slot_order,
    symbols_from_counts,
)
from tessa_paper._check import with_kinds
from tessa_paper.paths import DATA, FIGURES, RESULTS

__all__ = [
    "KNOWN_DISCREPANCIES",
    "PAPER_VALUES",
    "SEQUENCE_KEYS",
    "check_against_paper",
    "constellation_points",
    "constellation_spacing",
    "constellation_tex",
    "fig_genetic_results",
    "hardware_results",
    "load_sequences",
    "macros_tex",
    "rerun_shot_sweep",
    "rerun_simulation",
    "shot_sweep",
    "simulation_results",
    "simulation_slot_order",
    "table_genetic",
    "table_genetic_tex",
]

SEQUENCE_FILE = DATA / "sequences" / "genetic_sequences.txt"
HW_DIR = RESULTS / "hardware" / "genetic"
SIM_DIR = RESULTS / "simulation" / "genetic"
PAPER_TEX = FIGURES / "paper"
#: The constellation point sets of the hardware runs (``points_<n>``, n = 4, 16, 20, 400).
CONSTELLATION_FILE = HW_DIR / "constellation_points.npz"
#: The two hardware runs (ticket/npz stems), in the order of the notebook.
RUNS = ("genetic_sequences_db91e1694fic73eua10g", "genetic_second_db980g0c77vs73811r9g")

SEQUENCE_KEYS = (
    "POLG2_mRNA",
    "SUR1_ABCC8_iso2",
    "MT_CO3",
    "RRM1_HUMAN",
    "PIKACHURIN_HUMAN",
    "SHH_HUMAN",
    "P53_HUMAN",
)
SCHEMES = ("constellation", "levels")
DNA_ORDER = "ACGT"
KYTE_DOOLITTLE = "IVLFCMAGTSWYPHEQDNKR"
SHOTS = 32768
SEED = 7
PROBE_SEED = 1234
DRR_METHODS = tuple(m for m in DYNAMIC_RANGE_METHODS if m != "none")

NAMES = {
    "POLG2_mRNA": "POLG2 mRNA",
    "SUR1_ABCC8_iso2": "SUR1 isoform 2",
    "MT_CO3": "MT-CO3",
    "RRM1_HUMAN": "RRM1",
    "PIKACHURIN_HUMAN": "pikachurin",
    "SHH_HUMAN": "Sonic hedgehog",
    "P53_HUMAN": "p53",
}
SHORT = {
    "POLG2_mRNA": "POLG2",
    "SUR1_ABCC8_iso2": "SUR1",
    "MT_CO3": "MT-CO3",
    "RRM1_HUMAN": "RRM1",
    "PIKACHURIN_HUMAN": "pikachurin",
    "SHH_HUMAN": "SHH",
    "P53_HUMAN": "p53",
}
SCHEME_NAMES = {"constellation": "constellation", "levels": "angle"}
METHOD_CODES = {
    "min_max": "mm",
    "percentile": "pc",
    "mean_std": "ms",
    "abs_max": "am",
    "min_avg_max": "mam",
}

#: Printed in App. G (text, Table 8 and the two .tex inputs).
PAPER_VALUES: dict[str, object] = {
    "tab:genetic": (PAPER_TEX / "results_table.tex"),
    "macros": (PAPER_TEX / "results_macros.tex"),
    "constellation min spacing, 400 points (rad)": "0.156",
    "angle spacing, 20 levels, clamp 0.2 pi (rad)": "0.099",
    "smallest power of two with zero noiseless constellation errors": 32768,
    "fig:constellation": "figures/paper/constellation.tex",
}

#: Paper-vs-data differences (item of :func:`check_against_paper` -> explanation).
KNOWN_DISCREPANCIES: dict[str, str] = {
    "constellation 400 points, min spacing (rad)": (
        "The text quotes 0.156 rad; the stored 400-point set, with which every hardware "
        "cell of Table 8 is reproduced from the raw shots, has 0.163 rad. The 500-step "
        "repulsion that spreads the points is sensitive to rounding (a 1e-10 change of "
        "the start moves the minimum to 0.159), so the printed value may come from "
        "another platform or an earlier version of the code."
    ),
}


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------
def load_sequences(path: Path = SEQUENCE_FILE) -> dict[str, dict]:
    """The FASTA-like sequence file: ``{key: {type, length, source, title, seq}}``."""
    records: dict[str, dict] = {}
    key = None
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        if line.startswith(">"):
            head, _, title = line[1:].partition(" | ")
            key, *fields = head.split()
            records[key] = {
                **dict(f.split("=", 1) for f in fields),
                "title": title,
                "seq": "",
            }
        else:
            records[key]["seq"] += line.strip().upper()
    return records


def _alphabet(record: dict) -> str:
    return DNA_ORDER if record["type"] == "dna" else KYTE_DOOLITTLE


def _symbols(record: dict, alphabet: str) -> np.ndarray:
    return np.array([alphabet.index(s) for s in record["seq"]])


def _metrics(got: np.ndarray, truth: np.ndarray, slots: int) -> dict[str, float]:
    err = got - truth
    return {
        "accuracy": float(np.mean(err == 0)),
        "errors": int(np.count_nonzero(err)),
        "first": float(np.mean(err[:slots] == 0)),
        "second": float(np.mean(err[slots:] == 0)) if err.size > slots else float("nan"),
        "within 1": float(np.mean(np.abs(err) <= 1)),
        "mean dist": float(np.mean(np.abs(err))),
    }


def _evaluate(counts: dict, enc, truth: np.ndarray) -> dict:
    """``{"best_drr", "metrics": {method: metrics}}``; best = highest accuracy, then
    lowest mean distance, first in :data:`DRR_METHODS` order on ties (as the notebook)."""
    runs = {
        m: _metrics(symbols_from_counts(counts, enc, drr=m), truth, enc.slots)
        for m in ("none", *DRR_METHODS)
    }
    best = max(DRR_METHODS, key=lambda m: (runs[m]["accuracy"], -runs[m]["mean dist"]))
    return {"best_drr": best, "metrics": runs}


# ---------------------------------------------------------------------------
# Hardware: decode the two ibm_phoenix runs from the raw shots
# ---------------------------------------------------------------------------
def _load_counts(stem: str, n_circuits: int) -> list[dict[str, int]]:
    from qiskit.primitives import BitArray

    with np.load(HW_DIR / f"{stem}.npz") as z:
        return [
            BitArray(z[f"circuit_{i}"], int(z[f"num_bits_{i}"])).get_counts()
            for i in range(n_circuits)
        ]


def constellation_points(regenerate: bool = False) -> dict[int, np.ndarray]:
    """The constellation point sets of the genetic runs: ``{n_points: (n, 3) array}``.

    By default read from ``results/hardware/genetic/constellation_points.npz``
    (n = 4, 16, 20, 400: DNA and amino acids, one or two per slot). These are the
    sets with which the stored shots decode to every hardware cell of Table 8; they
    were computed once with :func:`tessa.constellation` (Linux x86_64, numpy 2.5.4);
    the sets computed on Linux aarch64 decode the shots to the same table.
    ``regenerate=True`` recomputes them with
    :func:`tessa.constellation` on this machine instead (the result can differ in the
    last digits between platforms, see the module docstring).
    """
    with np.load(CONSTELLATION_FILE) as z:
        sizes = sorted(int(k.split("_")[1]) for k in z.files)
        if regenerate:
            return {n: constellation(n) for n in sizes}
        return {n: z[f"points_{n}"] for n in sizes}


def _points_for(
    points: dict[int, np.ndarray] | None, scheme: str, n: int, k: int, cap: int
):
    """The point set ``encode_symbols`` needs for ``n`` symbols of alphabet size ``k``
    (one per slot if they fit in ``cap`` slots, two otherwise), or None."""
    if scheme != "constellation" or points is None:
        return None
    return points[k ** (1 if n <= cap else 2)]


def _decode_run(
    stem: str, records: dict, points: dict[int, np.ndarray] | None = None
) -> tuple[dict, dict]:
    ticket = json.loads((HW_DIR / f"{stem}.json").read_text())
    graph = graph_from_coupling(ticket["edges"], ticket["n_qubits"])
    items = (
        [tuple(i) for i in ticket["items"]]
        if "items" in ticket
        else list(product(ticket["keys"], ticket["schemes"]))
    )
    counts = _load_counts(stem, len(ticket["circuits"]))
    labels: list[list[str]] = []
    by_label = {}
    found: dict[str, dict] = {}
    for key, scheme in items:
        alphabet = ticket["alphabets"][key]
        truth = _symbols(records[key], alphabet)[: 2 * graph.capacity]
        _, enc = encode_symbols(
            graph,
            truth,
            len(alphabet),
            scheme=scheme,
            slot_order=ticket["slot_order"],
            points=_points_for(points, scheme, truth.size, len(alphabet), graph.capacity),
        )
        for basis in enc.bases:
            labels.append([key, scheme, basis])
        found.setdefault(key, {})[scheme] = (enc, truth)
    if labels != ticket["circuits"]:
        raise ValueError(f"{stem}: circuit order does not match the re-created encoding")
    by_label = {tuple(lb): c for lb, c in zip(labels, counts, strict=True)}
    out: dict[str, dict] = {}
    for key, by_scheme in found.items():
        for scheme, (enc, truth) in by_scheme.items():
            c = {b: by_label[(key, scheme, b)] for b in enc.bases}
            out.setdefault(key, {})[scheme] = {
                **_evaluate(c, enc, truth),
                "job_id": ticket["job_id"],
            }
    meta = {k: ticket.get(k, "") for k in ("job_id", "backend", "shots", "finished")}
    return meta, out


def hardware_results(regenerate_constellation: bool = False) -> dict:
    """Hardware columns of Table genetic, decoded from the raw shots of both runs.

    Reads ``results/hardware/genetic/*.json|.npz`` (about 10 s) and the stored
    constellation point sets (:func:`constellation_points`;
    ``regenerate_constellation=True`` computes them on this machine instead). Returns
    ``{"runs": [meta of run 1, meta of run 2], "sequences": {key: {scheme: {best_drr,
    metrics: {method: {accuracy, errors, first, second, within 1, mean dist}}, job_id}}}}``.
    """
    records = load_sequences()
    points = constellation_points(regenerate=regenerate_constellation)
    metas, merged = [], {}
    for stem in RUNS:
        meta, found = _decode_run(stem, records, points)
        metas.append(meta)
        for key, by_scheme in found.items():
            merged.setdefault(key, {}).update(by_scheme)
    return {"runs": metas, "sequences": merged}


def simulation_results(path: Path = SIM_DIR / "metrics.json") -> dict:
    """Stored simulation metrics (``"<scheme> | noiseless"`` / ``"<scheme> | phoenix noise"``)."""
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Table genetic and the macros
# ---------------------------------------------------------------------------
def _pct(m: dict) -> str:
    return f"{100 * m['accuracy']:.1f}"


def table_genetic(hw: dict | None = None, sim: dict | None = None) -> pd.DataFrame:
    """Table genetic: percentage of correctly decoded symbols.

    Simulated cells from ``metrics.json`` (or ``sim``), hardware cells from
    :func:`hardware_results` (or ``hw``). Returns one row per (sequence, encoding) with
    columns per_slot, sim_noiseless_raw, sim_noise_raw, sim_noise_min_max,
    sim_noise_best, sim_noise_best_method, hw_raw, hw_min_max, hw_best,
    hw_best_method (percent, unrounded).
    """
    hw = hardware_results() if hw is None else hw
    sim = simulation_results() if sim is None else sim
    records = load_sequences()
    rows = []
    for scheme in SCHEMES:
        for key in SEQUENCE_KEYS:
            s = sim["sequences"][key]
            quiet = s[f"{scheme} | noiseless"]["metrics"]
            noisy = s[f"{scheme} | phoenix noise"]
            h = hw["sequences"][key][scheme]
            n = len(records[key]["seq"])
            rows.append(
                {
                    "sequence": NAMES[key],
                    "encoding": SCHEME_NAMES[scheme],
                    "per_slot": 1 if n <= 792 else 2,
                    "sim_noiseless_raw": 100 * quiet["none"]["accuracy"],
                    "sim_noise_raw": 100 * noisy["metrics"]["none"]["accuracy"],
                    "sim_noise_min_max": 100 * noisy["metrics"]["min_max"]["accuracy"],
                    "sim_noise_best": 100
                    * noisy["metrics"][noisy["best_drr"]]["accuracy"],
                    "sim_noise_best_method": noisy["best_drr"],
                    "hw_raw": 100 * h["metrics"]["none"]["accuracy"],
                    "hw_min_max": 100 * h["metrics"]["min_max"]["accuracy"],
                    "hw_best": 100 * h["metrics"][h["best_drr"]]["accuracy"],
                    "hw_best_method": h["best_drr"],
                }
            )
    return pd.DataFrame(rows).set_index(["sequence", "encoding"])


def table_genetic_tex(table: pd.DataFrame | None = None) -> str:
    """The rows of Table genetic as LaTeX, in the format of ``results_table.tex``."""
    t = table_genetic() if table is None else table
    lines = []
    for i, scheme in enumerate(SCHEMES):
        sub = t[t.index.get_level_values("encoding") == SCHEME_NAMES[scheme]]
        for (name, enc_name), r in sub.iterrows():
            cells = [name, enc_name, str(r["per_slot"]), f"{r['sim_noiseless_raw']:.1f}"]
            cells += [
                f"{r['sim_noise_raw']:.1f}",
                f"{r['sim_noise_min_max']:.1f}",
                f"{r['sim_noise_best']:.1f} ({METHOD_CODES[r['sim_noise_best_method']]})",
                f"{r['hw_raw']:.1f}",
                f"{r['hw_min_max']:.1f}",
                f"{r['hw_best']:.1f} ({METHOD_CODES[r['hw_best_method']]})",
            ]
            lines.append(" & ".join(cells) + r" \\")
        if i < len(SCHEMES) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


#: First line of the published ``results_macros.tex`` (it names the original notebook,
#: which is not published); kept so that the regenerated file is byte-identical.
_MACROS_HEADER = "% Written by notebooks/genetic_sequence_demo.ipynb, section 11.\n"
#: First line of the published ``constellation.tex`` (names the original script, not
#: published); kept so that the regenerated file is byte-identical.
_TIKZ_HEADER = "% Written by appendix_genetic/make_tikz.py. Do not edit by hand."


def macros_tex(hw: dict | None = None, sim: dict | None = None) -> str:
    """``results_macros.tex``: the simulation shots (from ``metrics.json``) and the
    shots, backend, job id and date of both hardware runs (from the tickets)."""
    hw = hardware_results() if hw is None else hw
    sim = simulation_results() if sim is None else sim

    def shots(n: int) -> str:
        return f"{n:,}".replace(",", "{,}")

    macros = {"SimShots": shots(int(sim["shots"]))}
    for run, suffix in zip(hw["runs"], ("", "B"), strict=True):
        macros[f"HwShots{suffix}"] = shots(run["shots"])
        macros[f"HwBackend{suffix}"] = run["backend"].replace("_", r"\_")
        macros[f"HwJob{suffix}"] = run["job_id"]
        macros[f"HwDate{suffix}"] = run["finished"][:10]
    return _MACROS_HEADER + "".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()
    )


# ---------------------------------------------------------------------------
# Fig. genetic-results
# ---------------------------------------------------------------------------
def fig_genetic_results(
    save: Path | None = None, table: pd.DataFrame | None = None
) -> Figure:
    """Fig. genetic-results: fraction correct, simulated noise and hardware, raw and best DRR.

    Input: :func:`table_genetic` (or ``table``). Returns the figure; if ``save`` is given
    it is written there (the paper uses ``genetic_results.pdf``; a ``.png`` suffix gives
    200 dpi).
    """
    t = table_genetic() if table is None else table
    sim_c, hw_c = "#2a78d6", "#eb6834"
    bars = [
        ("simulated, raw", "sim_noise_raw", sim_c, 0.45),
        ("simulated, best DRR", "sim_noise_best", sim_c, 1.0),
        ("hardware, raw", "hw_raw", hw_c, 0.45),
        ("hardware, best DRR", "hw_best", hw_c, 1.0),
    ]
    names = [NAMES[k] for k in SEQUENCE_KEYS]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=True)
    x = np.arange(len(names))
    w = 0.2
    for ax, scheme, letter in zip(axes, SCHEMES, "ab", strict=True):
        sub = t.xs(SCHEME_NAMES[scheme], level="encoding").loc[names]
        for i, (label, col, colour, alpha) in enumerate(bars):
            pos = x + (i - 1.5) * w
            ax.bar(
                pos,
                sub[col].to_numpy() / 100,
                w * 0.92,
                color=colour,
                alpha=alpha,
                label=label,
            )
        ax.set_ylim(0, 1.18)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_ylabel("Ratio of Correct Symbols")
        ax.set_title(
            f"({letter}) {SCHEME_NAMES[scheme]} encoding", loc="left", fontsize=10
        )
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, ncol=4, loc="upper right")
    axes[-1].set_xticks(x, [SHORT[k] for k in SEQUENCE_KEYS], fontsize=8)
    fig.tight_layout()
    if save is not None:
        save = Path(save)
        fig.savefig(save, dpi=200 if save.suffix == ".png" else "figure")
    return fig


# ---------------------------------------------------------------------------
# Shot sweep, constellation spacing, Fig. constellation
# ---------------------------------------------------------------------------
def shot_sweep(path: Path = SIM_DIR / "shot_sweep.json") -> pd.DataFrame:
    """Noiseless constellation errors per shot count (``shot_sweep.json``).

    Returns a DataFrame indexed by shots with one column per sequence holding the
    largest error count over the five seeds.
    """
    d = json.loads(Path(path).read_text())
    cols = {
        key: {int(s): max(v) for s, v in entry["errors"]["constellation"].items()}
        for key, entry in d.items()
    }
    return pd.DataFrame(cols).sort_index()


def _smallest_errorless_shots(sweep: pd.DataFrame) -> int:
    ok = (sweep == 0).all(axis=1)
    # smallest shot count from which every larger count is also error-free
    for shots in sweep.index:
        if ok.loc[shots:].all():
            return int(shots)
    return -1


def _min_angle(points: np.ndarray) -> float:
    cos = np.clip(points @ points.T, -1.0, 1.0)
    np.fill_diagonal(cos, -1.0)
    return float(np.arccos(cos.max()))


def constellation_spacing() -> dict[str, float]:
    """Spacings quoted in App. genetic, in rad.

    Returns the minimum angular distance of the stored 400-point constellation (two
    amino acids per slot) and of the 20-point one (:func:`constellation_points`), and
    the spacing of 20 angle levels on the clamped amplitude, ``(pi - 2 * 0.2 pi) / 19``.
    """
    points = constellation_points()
    return {
        "constellation_400": _min_angle(points[400]),
        "constellation_20": _min_angle(points[20]),
        "angle_20_clamped": float((np.pi - 2 * 0.2 * np.pi) / 19),
    }


def constellation_tex(save: Path | None = None) -> str:
    """Fig. constellation: the TikZ source (port of the original script, not published).

    The 64 points of ``tessa.constellation(64)`` (k = 8) in an orthographic view, an
    illustrative noise-shrunk Bloch vector and the decoding pipeline. Returns the text
    and writes it to ``save`` if given (paper file ``figures/paper/constellation.tex``).
    """
    radius, k = 1.9, 8
    az, el = np.radians(-55.0), np.radians(18.0)

    def project(v):
        x, y, z = v
        u = -np.sin(az) * x + np.cos(az) * y
        w = -np.sin(el) * np.cos(az) * x - np.sin(el) * np.sin(az) * y + np.cos(el) * z
        depth = np.cos(el) * np.cos(az) * x + np.cos(el) * np.sin(az) * y + np.sin(el) * z
        return radius * u, radius * w, depth

    pts = constellation(k * k)
    proj = np.array([project(v) for v in pts])
    score = np.where(proj[:, 2] > 0.15, np.hypot(proj[:, 0], proj[:, 1]), 0)
    score[(proj[:, 0] > 0) | (proj[:, 1] < 0)] = 0
    highlight = int(score.argmax())
    p = pts[highlight]
    r = 0.72 * (p + 0.12 * np.cross(p, [0, 0, 1]))
    out = [
        _TIKZ_HEADER,
        r"\begin{tikzpicture}[>=stealth, font=\small]",
        r"\begin{scope}",
        rf"\shade[inner color=white, outer color=blue!14] (0,0) circle ({radius});",
        rf"\draw[gray!70] (0,0) circle ({radius});",
        rf"\draw[gray!50, dashed] ({radius},0) arc (0:180:{radius} and {radius * np.sin(el):.3f});",
        rf"\draw[gray!70] ({-radius},0) arc (180:360:{radius} and {radius * np.sin(el):.3f});",
    ]
    for i, v in enumerate(pts):
        u, w, d = project(v)
        if i == highlight:
            continue
        style = "fill=black!75" if d >= 0 else "fill=black!20"
        out.append(rf"\fill[{style}] ({u:.3f},{w:.3f}) circle (1.1pt);")
    for name, axis in [("z", (0, 0, 1)), ("x", (1, 0, 0)), ("y", (0, 1, 0))]:
        u, w, _ = project(np.array(axis) * 1.35)
        out.append(
            rf"\draw[->, black!55] (0,0) -- ({u:.3f},{w:.3f}) "
            rf"node[anchor=south west, inner sep=1pt, fill=white, fill opacity=0.8, text opacity=1] {{${name}$}};"
        )
    pu, pw, _ = project(p)
    ru, rw, _ = project(r)
    out += [
        rf"\draw[orange!80!black, thick, dashed] ({pu:.3f},{pw:.3f}) circle (0.36);",
        rf"\draw[->, thick, orange!80!black] (0,0) -- ({ru:.3f},{rw:.3f});",
        rf"\node[orange!80!black, anchor=north east, inner sep=1pt] at ({ru / 2:.3f},{rw / 2:.3f}) {{$\mathbf{{r}}$}};",
        rf"\fill[blue!70!black] ({pu:.3f},{pw:.3f}) circle (2pt);",
        rf"\node[blue!70!black, anchor=south east, inner sep=2pt] at ({pu:.3f},{pw:.3f}) {{$\mathbf{{p}}_j$}};",
        r"\end{scope}",
    ]
    steps = [
        r"symbols $(s_1, s_2)$, $s_i \in \{0,\dots,k-1\}$",
        r"point $j = s_1 k + s_2$ of $k^2$",
        r"$(\theta_j, \varphi_j)$ of $\mathbf{p}_j$",
        r"$R_Y(\theta_j)$, $R_Z(\varphi_j)$ multiplexers",
        r"measure $Z$, $X$, $Y$: $\mathbf{r} = (\langle X\rangle, \langle Y\rangle, \langle Z\rangle)$",
        r"$\hat{\jmath} = \arg\max_j \, \mathbf{p}_j \cdot \mathbf{r} / \lVert \mathbf{r} \rVert$",
    ]
    x0, y0, dy = radius + 4.9, 1.75, 0.7
    for i, text in enumerate(steps):
        out.append(
            rf"\node[draw, rounded corners=2pt, fill=gray!6, minimum width=6.4cm, "
            rf"inner sep=3pt] (s{i}) at ({x0},{y0 - i * dy:.2f}) {{{text}}};"
        )
        if i:
            out.append(rf"\draw[->] (s{i - 1}) -- (s{i});")
    out.append(r"\end{tikzpicture}")
    text = "\n".join(out) + "\n"
    if save is not None:
        Path(save).write_text(text)
    return text


# ---------------------------------------------------------------------------
# Re-running the simulations
# ---------------------------------------------------------------------------
def _phoenix():
    from tessa.hardware.devices import snapshot_backend

    return snapshot_backend("phoenix")


def simulation_slot_order(shots: int = SHOTS, seed: int = PROBE_SEED) -> np.ndarray:
    """Slot order of the simulations: data qubits ranked by a probe on the snapshot noise.

    ``probe_qubits`` on ``calibrated_sampler(phoenix snapshot, seed=1234)`` with 32768
    shots, as in the original notebook and shot-sweep script. Returns the 792 slot
    indices.
    """
    backend = _phoenix()
    graph = graph_from_backend(backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quality = probe_qubits(
            graph, calibrated_sampler(backend, seed=seed), shots=shots, seed=seed
        )
    return slot_order(graph, quality)


def rerun_simulation(
    keys: tuple[str, ...] = SEQUENCE_KEYS,
    schemes: tuple[str, ...] = SCHEMES,
    samplers: tuple[str, ...] = ("noiseless", "phoenix noise"),
    order: np.ndarray | None = None,
) -> dict:
    """Re-run the simulated columns of Table genetic (the notebook's sections 1-5).

    Encodes each sequence on the ibm_phoenix snapshot graph (best slots first, see
    :func:`simulation_slot_order`; pass ``order`` to reuse it), decodes with
    ``aer_sampler(graph, seed=7)`` ("noiseless") and ``calibrated_sampler(phoenix,
    seed=7)`` ("phoenix noise"), 32768 shots, per cell with the library's default
    traced halo (as the notebook did), with the stored constellation point sets
    (:func:`constellation_points`). Returns ``{key: {"<scheme> | <sampler>":
    {best_drr, metrics}}}`` in the schema of ``metrics.json``. Cost: see the notes; run
    it in chunks to stay within a time budget. Measured on a 4-core VM: probe 12 s;
    all 7 sequences x 2 schemes x 2 samplers about 3.5 min (10-30 s per sequence and
    scheme). Result (``results/simulation/genetic/metrics_rerun.json``): identical to
    ``metrics.json`` except MT-CO3, constellation, phoenix noise, where one symbol of
    784 differs for the methods ``none`` and ``mean_std``.
    """
    backend = _phoenix()
    graph = graph_from_backend(backend)
    order = simulation_slot_order() if order is None else order
    records = load_sequences()
    points = constellation_points()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        make = {
            "noiseless": lambda: aer_sampler(graph, seed=SEED),
            "phoenix noise": lambda: calibrated_sampler(backend, seed=SEED),
        }
        out: dict[str, dict] = {}
        for key in keys:
            alphabet = _alphabet(records[key])
            truth = _symbols(records[key], alphabet)
            for scheme in schemes:
                circuit, enc = encode_symbols(
                    graph,
                    truth,
                    len(alphabet),
                    scheme=scheme,
                    slot_order=order,
                    points=_points_for(
                        points, scheme, truth.size, len(alphabet), graph.capacity
                    ),
                )
                for name in samplers:
                    _, counts = decode_symbols(
                        circuit, enc, make[name](), shots=SHOTS, return_counts=True
                    )
                    out.setdefault(key, {})[f"{scheme} | {name}"] = _evaluate(
                        counts, enc, truth
                    )
    return out


def rerun_shot_sweep(
    keys: tuple[str, ...] = (
        "POLG2_mRNA",
        "SUR1_ABCC8_iso2",
        "P53_HUMAN",
        "PIKACHURIN_HUMAN",
        "SHH_HUMAN",
    ),
    shots: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536),
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    order: np.ndarray | None = None,
) -> dict:
    """Re-run ``shot_sweep.json`` (port of the original shot-sweep script).

    Noiseless constellation decoding on the snapshot graph with the probe slot order
    and the stored constellation point sets,
    ``aer_sampler(graph, seed=seed)``; one symbol per slot if the sequence fits, two
    otherwise. Returns ``{key: {n, alphabet, per_slot, errors: {"constellation":
    {shots (str): [errors per seed]}}}}``. A few seconds per decode; chunk over keys,
    shots and seeds.
    """
    backend = _phoenix()
    graph = graph_from_backend(backend)
    order = simulation_slot_order() if order is None else order
    records = load_sequences()
    points = constellation_points()
    out: dict[str, dict] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key in keys:
            alphabet = _alphabet(records[key])
            truth = _symbols(records[key], alphabet)
            circuit, enc = encode_symbols(
                graph,
                truth,
                len(alphabet),
                scheme="constellation",
                slot_order=order,
                points=_points_for(
                    points, "constellation", truth.size, len(alphabet), graph.capacity
                ),
            )
            rows = {
                str(n): [
                    int(
                        np.count_nonzero(
                            decode_symbols(
                                circuit, enc, aer_sampler(graph, seed=sd), shots=n
                            )
                            != truth
                        )
                    )
                    for sd in seeds
                ]
                for n in shots
            }
            out[key] = {
                "n": int(truth.size),
                "alphabet": alphabet,
                "per_slot": enc.per_slot,
                "errors": {"constellation": rows},
            }
    return out


# ---------------------------------------------------------------------------
# Check against the paper
# ---------------------------------------------------------------------------
def check_against_paper(hw: dict | None = None) -> pd.DataFrame:
    """Every cell of Table genetic, the macros, the spacing and shot claims, and Fig.
    constellation, against the paper's files.

    Returns a DataFrame [item, paper, reproduced, match, kind]; table cells are compared
    as printed (one decimal, method code). ``hw`` reuses a :func:`hardware_results`
    output. The macros (shots, backend, job ids, dates) count as labels.
    """
    hw = hardware_results() if hw is None else hw
    table = table_genetic(hw=hw)
    rows = []
    paper_lines = PAPER_VALUES["tab:genetic"].read_text().splitlines()
    ours = table_genetic_tex(table).splitlines()
    heads = [
        "per slot",
        "no noise raw",
        "sim raw",
        "sim min-max",
        "sim best",
        "hw raw",
        "hw min-max",
        "hw best",
    ]
    for p_line, o_line in zip(paper_lines, ours, strict=True):
        if p_line.startswith(r"\midrule"):
            continue
        p_cells = [c.strip() for c in p_line.rstrip(" \\").split("&")]
        o_cells = [c.strip() for c in o_line.rstrip(" \\").split("&")]
        for head, pc, oc in zip(heads, p_cells[2:], o_cells[2:], strict=True):
            rows.append(
                (
                    f"tab:genetic {p_cells[0]} / {p_cells[1]} / {head}",
                    pc,
                    oc,
                    pc == oc,
                    "number",
                )
            )
    paper_macros = dict(
        line[len("\\newcommand{\\") :].rstrip("}").split("}{", 1)
        for line in PAPER_VALUES["macros"].read_text().splitlines()
        if line.startswith("\\newcommand")
    )
    our_macros = dict(
        line[len("\\newcommand{\\") :].rstrip("}").split("}{", 1)
        for line in macros_tex(hw).splitlines()
        if line.startswith("\\newcommand")
    )
    for k, v in paper_macros.items():
        rows.append(
            (f"macro \\{k}", v, our_macros.get(k, ""), v == our_macros.get(k), "label")
        )
    sp = constellation_spacing()
    rows.append(
        (
            "constellation 400 points, min spacing (rad)",
            "0.156",
            f"{sp['constellation_400']:.3f}",
            f"{sp['constellation_400']:.3f}" == "0.156",
            "number",
        )
    )
    rows.append(
        (
            "20 angles on the clamped amplitude, spacing (rad)",
            "0.099",
            f"{sp['angle_20_clamped']:.3f}",
            f"{sp['angle_20_clamped']:.3f}" == "0.099",
            "number",
        )
    )
    n = _smallest_errorless_shots(shot_sweep())
    rows.append(
        (
            "smallest power of two with zero noiseless constellation errors",
            "32768",
            str(n),
            n == 32768,
            "number",
        )
    )
    paper_tikz = (PAPER_TEX / "constellation.tex").read_text()
    same = constellation_tex() == paper_tikz
    rows.append(
        (
            "fig:constellation TikZ source",
            "constellation.tex",
            "identical" if same else "differs",
            same,
            "label",
        )
    )
    return with_kinds(
        pd.DataFrame(rows, columns=["item", "paper", "reproduced", "match", "kind"])
    )
