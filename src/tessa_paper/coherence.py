"""App. ``sec:downstream`` ("What do the coherences mean?"): Fig. ``fig:coherences``
and Table ``tab:kcompare``.

**Fig. coherences** is the hardware run of 2026-10-02 on ``ibm_phoenix`` (job
``davnlrlj371s73dn89o0``, 8192 shots per circuit), *experiment B*: every
horizontally and vertically adjacent pixel pair (i, j) of the 28x28 QCAT image is
encoded as an isolated degree-one super-pixel (one data and one address qubit, values
v_i, v_j at the two addresses) on a coupler of a maximum matching of 60 disjoint
low-error couplers (1512 pairs in 26 circuits, plus one reference circuit), then a
Hadamard is applied to the address qubit and K = 2 P(address = 0) - 1 is measured;
the exact value is cos(pi (v_i - v_j) / 2). The figure shows the edge strength
1 - K. This is the degree-one special case of Eq. (Ka), measured pair by pair. The
appendix text and caption describe Eq. (Ka) for the encoded full-lattice state
(Hadamards on every address qubit of the whole-chip encoding: *experiment A*,
circuit ``k_image`` of the same job), which is **not** what the figure shows.

Reads ``results/hardware/coherence/tables/phoenix_B_pairs.csv`` (per-pair exact and
measured K) and ``data/images/qcat/qcat28.png``; :func:`rederive_pairs_from_counts`
recomputes the measured K from ``counts/phoenix.json.gz`` and ``design.json``.
Port of the figure cell (and its helpers) of the original analysis notebook (not
published).

**Table kcompare**: no code or data survive. The paper's description fixes the
lattice (3x4), the address qubit (the single degree-four address qubit), the
sampling (noiseless), the number of repetitions (200) and the shot counts, but not
the payload, which sets both K_a and the bias of the classical route; see
:data:`KCOMPARE_UNSPECIFIED`. :func:`kcompare_undefined_fraction` reproduces the one
payload-independent column; :func:`kcompare_simulation` is a re-implementation from
the paper's description (not the original code) that needs the payload as input.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from qiskit.quantum_info import Statevector

from tessa import TessaGraph, decode_from_counts, encode, graph_from_coupling
from tessa.graph import square_lattice_edges

from ._check import compare_with_paper
from .paths import DATA, RESULTS

__all__ = [
    "KCOMPARE_SHOTS",
    "KCOMPARE_UNSPECIFIED",
    "KNOWN_DISCREPANCIES",
    "PAPER_VALUES",
    "check_against_paper",
    "coherence_figure",
    "edge_maps",
    "edge_metrics",
    "k_exact",
    "kcompare_interference_rms",
    "kcompare_lattice",
    "kcompare_simulation",
    "kcompare_table",
    "kcompare_undefined_fraction",
    "load_pairs",
    "rederive_pairs_from_counts",
]

COHERENCE = RESULTS / "hardware" / "coherence"
PARTS = (
    ("horizontal", "$1-K$, horizontal pairs"),
    ("vertical", "$1-K$, vertical pairs"),
    ("combined", "combined"),
)
#: Shot counts of Table kcompare.
KCOMPARE_SHOTS = (64, 256, 1024, 4096, 65536)
#: Rows of Table 4 (tab:kcompare) as printed:
#: interference RMS, classical RMS, classical bias, undefined.
_KCOMPARE_PRINTED = {
    64: ("0.113", "0.169", "-0.158", "26%"),
    256: ("0.054", "0.056", "-0.044", "0%"),
    1024: ("0.029", "0.022", "-0.015", "0%"),
    4096: ("0.015", "0.010", "-0.006", "0%"),
    65536: ("0.004", "0.003", "-0.001", "0%"),
}

# Numbers printed in Fig. 9 (figures/paper/coherences.png) and its caption, and the
# payload-independent column of Table 4 (tab:kcompare).
PAPER_VALUES: dict[str, str] = {
    "fig:coherences shots": "8192",
    "fig:coherences horizontal MAE": "0.022",
    "fig:coherences horizontal rho": "0.98",
    "fig:coherences vertical MAE": "0.019",
    "fig:coherences vertical rho": "0.99",
    "fig:coherences combined MAE": "0.020",
    "fig:coherences combined rho": "0.99",
    "fig:coherences horizontal bias": "+0.000",
    "fig:coherences horizontal max |err|": "0.97",
    "fig:coherences vertical bias": "+0.005",
    "fig:coherences vertical max |err|": "0.96",
    "fig:coherences combined bias": "+0.002",
    "fig:coherences combined max |err|": "0.68",
    **{f"tab:kcompare N={n} undefined": row[3] for n, row in _KCOMPARE_PRINTED.items()},
}

KNOWN_DISCREPANCIES: dict[str, str] = {
    "tab:kcompare N=64 undefined": (
        "a Monte-Carlo fraction of 200 repetitions: seed 0 gives 22%, 20000 "
        "repetitions give 23.5% (standard error of a 200-repetition estimate 3.0 "
        "points), so the paper's 26% is consistent with it but its seed is unknown."
    ),
}

#: What the paper's description of Table 4 (App. D) leaves open.
KCOMPARE_UNSPECIFIED = (
    "the payload: which values were encoded on the 3x4 lattice (an image, a random "
    "draw and from which distribution, a fixed payload or a new one per repetition); "
    "K_a and the bias of the classical route depend on it",
    "the value-to-angle map (presumably theta = pi * value, value range (0, 1))",
    "how repetitions in which an address configuration received no shots enter the "
    "classical RMS error and bias (left out, or counted with some default angle)",
    "which address configurations count for 'undefined' (those of the data qubits "
    "next to the address qubit, which K_a needs, or those of every data qubit)",
    "the random seeds",
)


def load_pairs(device: str = "phoenix") -> pd.DataFrame:
    """Per-pair table of experiment B, ``results/hardware/coherence/tables/
    <device>_B_pairs.csv``: one row per pixel pair in pair order (all horizontal
    pairs row-major, then all vertical pairs row-major), with the exact K, the
    hardware and local-noise-simulation K and the coupler references."""
    return pd.read_csv(COHERENCE / "tables" / f"{device}_B_pairs.csv")


def edge_maps(
    k: Sequence[float] | np.ndarray, shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    """Per-pair K -> maps of the edge strength 1 - K: ``horizontal`` (R, C - 1),
    ``vertical`` (R - 1, C) and ``combined`` = sqrt(h**2 + v**2) / sqrt(2) on
    (R - 1, C - 1), as in the original notebook."""
    rows, cols = shape
    nh = rows * (cols - 1)
    k = np.asarray(k, float)
    eh = 1 - k[:nh].reshape(rows, cols - 1)
    ev = 1 - k[nh:].reshape(rows - 1, cols)
    combined = np.sqrt(eh[:-1, :] ** 2 + ev[:, :-1] ** 2) / np.sqrt(2)
    return {"horizontal": eh, "vertical": ev, "combined": combined}


def _compare(est: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    """n, MAE, RMSE, bias, Pearson rho of ``est`` against ``exact`` (finite entries)."""
    est, exact = np.ravel(est).astype(float), np.ravel(exact).astype(float)
    ok = np.isfinite(est) & np.isfinite(exact)
    e, x = est[ok], exact[ok]
    err = e - x
    return {
        "n": int(e.size),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "bias": float(err.mean()),
        "max |err|": float(np.abs(err).max()),
        "rho": float(np.corrcoef(e, x)[0, 1]),
    }


def _image(device: str = "phoenix") -> np.ndarray:
    name = {"phoenix": "qcat28.png", "fez": "qcat.png"}[device]
    return np.asarray(Image.open(DATA / "images" / "qcat" / name).convert("L")) / 255.0


def edge_metrics(device: str = "phoenix") -> pd.DataFrame:
    """Hardware edge maps of experiment B against the exact maps: n, MAE, RMSE, bias,
    largest absolute error and Pearson rho of 1 - K for the horizontal, vertical and
    combined maps (the numbers printed in Fig. coherences). Reads the pairs table.

    On the 28x28 image there are 28 x 27 = 756 horizontal and 27 x 28 = 756 vertical
    pairs; the combined map needs both the horizontal and the vertical pair that start
    at a pixel, so it covers the 27 x 27 = 729 pixels not in the last row or column
    (:func:`edge_maps`).
    """
    pairs = load_pairs(device)
    shape = _image(device).shape
    exact = edge_maps(pairs["K_exact"], shape)
    hardware = edge_maps(pairs["hardware_measured"], shape)
    rows = {part: _compare(hardware[part], exact[part]) for part, _ in PARTS}
    return pd.DataFrame(rows).T


def coherence_figure(device: str = "phoenix", save: Path | None = None) -> Figure:
    """Fig. ``fig:coherences``: experiment B on ``device`` (the paper shows Phoenix).

    Rows: exact edge strength 1 - K, hardware, hardware minus exact; columns: the
    input image, horizontal pairs, vertical pairs, combined. Under the hardware maps
    the MAE and rho, under the error maps the mean (bias) and the largest absolute
    error. Reads ``tables/<device>_B_pairs.csv`` and the input image. With ``save``
    the figure is written at 100 dpi, the resolution of the published PNG (an inline
    notebook render). Returns the figure.
    """
    pairs = load_pairs(device)
    img = _image(device)
    exact = edge_maps(pairs["K_exact"], img.shape)
    hardware = edge_maps(pairs["hardware_measured"], img.shape)
    metrics = edge_metrics(device)
    err = {p: hardware[p] - exact[p] for p, _ in PARTS}
    every = np.concatenate([np.abs(np.ravel(e)) for e in err.values()])
    lim = max(0.05, float(np.nanpercentile(every, 99)))
    rows = [("exact", "exact"), ("hardware", "hardware"), ("error", "hardware $-$ exact")]
    fig = Figure(figsize=(7.2, 5.4), layout="constrained")
    axes = fig.subplots(3, 4)
    for r, (kind, label) in enumerate(rows):
        ax0 = axes[r, 0]
        ax0.axis("off")
        if r == 0:
            ax0.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax0.set_title("input", fontsize=8)
        ax0.text(
            -0.08,
            0.5,
            label,
            transform=ax0.transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )
        for c, (part, title) in enumerate(PARTS, start=1):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8)
            if kind == "exact":
                im_k = ax.imshow(
                    np.clip(exact[part], 0, 1),
                    cmap="magma",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
            elif kind == "hardware":
                ax.imshow(
                    np.clip(hardware[part], 0, 1),
                    cmap="magma",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                m = metrics.loc[part]
                ax.set_xlabel(
                    f"MAE {m['MAE']:.3f}, $\\rho$ {m['rho']:.2f}", fontsize=7, labelpad=2
                )
            else:
                e = err[part]
                im_e = ax.imshow(
                    e, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest"
                )
                note = (
                    f"bias {np.nanmean(e):+.3f}\nmax $|$err$|$ {np.nanmax(np.abs(e)):.2f}"
                )
                ax.set_xlabel(note, fontsize=7, labelpad=2)
    fig.colorbar(im_k, ax=axes[:2, 1:], shrink=0.7, label="edge strength $1-K$")
    fig.colorbar(im_e, ax=axes[2, 1:], shrink=0.9, extend="both", label="error in $1-K$")
    backend = {"phoenix": "ibm_phoenix", "fez": "ibm_fez"}[device]
    fig.suptitle(
        f"{backend}: Coherences (edge detection) of QCAT at {_job_shots(device)} shots",
        fontsize=9,
    )
    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=100, bbox_inches="tight")
    return fig


def _p0_all(counts: dict[str, int]) -> np.ndarray:
    """P(bit q = 0) for every classical bit q (bit q of ``measure_all`` is qubit q)."""
    keys = list(counts)
    w = np.array([counts[k] for k in keys], float)
    bits = (
        np.frombuffer("".join(keys).encode(), dtype=np.uint8).reshape(len(keys), -1) - 48
    )
    return 1.0 - (w[:, None] * bits[:, ::-1]).sum(0) / w.sum()


def rederive_pairs_from_counts(device: str = "phoenix") -> pd.DataFrame:
    """Recompute the measured K of every pair of experiment B from the raw counts.

    Reads ``counts/<device>.json.gz`` (circuits ``pairs_0`` ...) and ``design.json``
    (matching of couplers, pairs, assignment of pairs to circuits and couplers):
    K = 2 P(address qubit = 0) - 1 of the pair's coupler in its circuit. Returns the
    pair table with columns ``K_rederived`` and ``|diff|`` (against the stored
    ``hardware_measured``).
    """
    with gzip.open(COHERENCE / "counts" / f"{device}.json.gz", "rt") as f:
        counts = json.load(f)
    design = json.loads((COHERENCE / "design.json").read_text())["devices"][device]["B"]
    address = np.array([a for _, a in design["matching"]])
    p0 = np.full(len(design["pairs"]), np.nan)
    for b, batch in enumerate(design["batches"]):
        measured = _p0_all(counts[f"pairs_{b}"])
        for p, e in batch:
            p0[p] = measured[address[e]]
    table = load_pairs(device)
    table["K_rederived"] = 2 * p0 - 1
    table["|diff|"] = np.abs(table["K_rederived"] - table["hardware_measured"])
    return table


# ---------------------------------------------------------------------------
# Table kcompare: re-implementation from the paper's description
# ---------------------------------------------------------------------------
def kcompare_lattice() -> tuple[TessaGraph, int]:
    """The 3x4 square lattice with its Tessa colouring and its single degree-four
    address qubit (qubit 6, row 1, column 2)."""
    edges, n = square_lattice_edges(3, 4)
    graph = graph_from_coupling(edges, n)
    data = set(graph.data_qubits)
    (address,) = [q for q in range(n) if q not in data and len(graph.neighbours(q)) == 4]
    return graph, address


def k_exact(graph: TessaGraph, theta: dict[int, np.ndarray]) -> dict[int, float]:
    """Eq. (Ka) for every address qubit, from the angles ``theta[site]`` of each cell
    (indexed by local address configuration, bit j = ``address_qubits[j]``).
    Port of ``k_exact`` of the original coherence notebook."""
    adjacent: dict[int, list] = defaultdict(list)
    for g in graph.groups:
        for j, a in enumerate(g.address_qubits):
            adjacent[a].append((g, j))
    out = {}
    for a, cells in adjacent.items():
        others = sorted({q for g, _ in cells for q in g.address_qubits if q != a})
        pos = {q: i for i, q in enumerate(others)}
        bits = (np.arange(2 ** len(others))[:, None] >> np.arange(len(others))) & 1
        prod = np.ones(2 ** len(others))
        for g, j in cells:
            x0 = np.zeros(len(prod), dtype=int)
            for jj, q in enumerate(g.address_qubits):
                if q != a:
                    x0 |= bits[:, pos[q]] << jj
            th = np.asarray(theta[g.site])
            prod *= np.cos((th[x0] - th[x0 | (1 << j)]) / 2)
        out[a] = float(prod.mean())
    return out


def _cells_needed(graph: TessaGraph, address: int, scope: str) -> list:
    if scope == "neighbours":
        return [g for g in graph.groups if address in g.address_qubits]
    if scope == "lattice":
        return list(graph.groups)
    raise ValueError("scope must be 'neighbours' or 'lattice'")


def kcompare_undefined_fraction(
    shots: Sequence[int] = KCOMPARE_SHOTS,
    repetitions: int = 200,
    seed: int = 0,
    scope: str = "neighbours",
) -> pd.DataFrame:
    """Column "undefined" of Table ``tab:kcompare``: the fraction of repetitions in
    which some address configuration of a cell received no shots.

    In noiseless sampling the address bits are uniform and independent whatever the
    payload, so this column does not depend on the payload: each shot draws the 6
    address bits of the 3x4 lattice uniformly. ``scope="neighbours"`` counts the
    cells next to the degree-four address qubit (the cells Eq. (Ka) needs);
    ``"lattice"`` every cell (on this lattice both include the only degree-four data
    qubit, which dominates). Returns, per shot count, the fraction in ``repetitions``
    repetitions and in 20000 repetitions (a precise estimate).
    """
    graph, address = kcompare_lattice()
    cells = _cells_needed(graph, address, scope)
    addresses = sorted({a for g in graph.groups for a in g.address_qubits})
    col = {a: i for i, a in enumerate(addresses)}
    rng = np.random.default_rng(seed)
    configs = np.arange(2 ** len(addresses))
    bits = (configs[:, None] >> np.arange(len(addresses))) & 1

    def undefined(n_shots: int, reps: int) -> float:
        hit = rng.multinomial(n_shots, np.full(len(configs), 1 / len(configs)), size=reps)
        bad = np.zeros(reps, bool)
        for g in cells:
            local = np.zeros(len(configs), int)
            for j, a in enumerate(g.address_qubits):
                local |= bits[:, col[a]] << j
            onehot = np.eye(2 ** len(g.address_qubits), dtype=int)[local]
            bad |= (hit @ onehot == 0).any(axis=1)
        return float(bad.mean())

    rows = [
        {
            "shots": n,
            f"undefined ({repetitions} repetitions)": undefined(n, repetitions),
            "undefined (20000 repetitions)": undefined(n, 20_000),
        }
        for n in shots
    ]
    return pd.DataFrame(rows).set_index("shots")


def kcompare_interference_rms(shots: Sequence[int] = KCOMPARE_SHOTS) -> pd.DataFrame:
    """The interference route's standard error 2 sqrt(P(1 - P) / N) =
    sqrt((1 - K**2) / N): its bound 1 / sqrt(N), the paper's RMS error, and the K_a
    that the paper's RMS error implies (|K| = sqrt(1 - N RMS**2); NaN when the
    printed RMS exceeds the bound). Shows what the printed column says about the
    unknown payload."""
    rows = []
    for n in shots:
        rms = float(_KCOMPARE_PRINTED[n][0])
        implied = 1 - n * rms**2
        rows.append(
            {
                "shots": n,
                "bound 1/sqrt(N)": 1 / np.sqrt(n),
                "paper RMS": rms,
                "implied |K_a|": np.sqrt(implied) if implied >= 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("shots")


def kcompare_simulation(
    payload: np.ndarray,
    shots: Sequence[int] = KCOMPARE_SHOTS,
    repetitions: int = 200,
    seed: int = 0,
    scope: str = "neighbours",
) -> pd.DataFrame:
    """Re-implementation of Table ``tab:kcompare`` from the paper's description (not
    the original code, which is lost), for a given ``payload``.

    ``payload``: the 48 values in [0, 1] of the 3x4 lattice (theta = pi * value);
    the paper does not say which payload it used (:data:`KCOMPARE_UNSPECIFIED`), so
    the numbers depend on this choice and are not a reproduction of the table.
    Per shot count N and repetition, noiseless sampling from the exact state:

    * interference route: H on every address qubit, N shots of the degree-four
      address qubit a, K = 2 P(z_a = 0) - 1 (binomial sampling of the exact marginal);
    * classical route: N shots of the Z readout of every qubit, decoded with
      :func:`tessa.decode_from_counts`, then Eq. (Ka) on the decoded angles; a
      repetition is "undefined" when an address configuration of a needed cell
      (``scope``) got no shots, and is left out of the classical RMS and bias.

    Returns per N: exact K_a, interference RMS error, classical RMS error, classical
    bias and the undefined fraction.
    """
    graph, address = kcompare_lattice()
    payload = np.asarray(payload, float)
    circuit, enc = encode(graph, payload, value_range=(0, 1))
    exact = k_exact(graph, graph.pack(np.pi * payload))[address]
    with_h = circuit.copy()
    with_h.h(sorted({a for g in graph.groups for a in g.address_qubits}))
    p0 = float(Statevector(with_h).probabilities([address])[0])
    probs = Statevector(circuit).probabilities()
    probs = probs / probs.sum()
    keys = [format(i, f"0{graph.n_qubits}b") for i in range(len(probs))]
    cells = _cells_needed(graph, address, scope)
    rng = np.random.default_rng(seed)
    rows = []
    for n in shots:
        k_int = 2 * rng.binomial(n, p0, size=repetitions) / n - 1
        k_cls, undefined = [], 0
        for hits in rng.multinomial(n, probs, size=repetitions):
            counts = {keys[i]: int(hits[i]) for i in np.flatnonzero(hits)}
            if _some_configuration_empty(counts, cells, graph.n_qubits):
                undefined += 1
                continue
            values = np.ravel(decode_from_counts({"z": counts}, enc))
            slots = np.zeros(graph.capacity)
            slots[: values.size] = values
            k_cls.append(k_exact(graph, graph.pack(np.pi * slots))[address])
        err_cls = np.asarray(k_cls) - exact
        rows.append(
            {
                "shots": n,
                "exact K_a": exact,
                "interference RMS": float(np.sqrt(np.mean((k_int - exact) ** 2))),
                "classical RMS": float(np.sqrt(np.mean(err_cls**2)))
                if err_cls.size
                else np.nan,
                "classical bias": float(err_cls.mean()) if err_cls.size else np.nan,
                "undefined": undefined / repetitions,
            }
        )
    return pd.DataFrame(rows).set_index("shots")


def _some_configuration_empty(counts: dict[str, int], cells: list, n_qubits: int) -> bool:
    keys = list(counts)
    w = np.array([counts[k] for k in keys])
    bits = (np.frombuffer("".join(keys).encode(), np.uint8).reshape(len(keys), -1) - 48)[
        :, ::-1
    ]
    for g in cells:
        local = np.zeros(len(keys), int)
        for j, a in enumerate(g.address_qubits):
            local |= bits[:, a].astype(int) << j
        if (
            np.bincount(local, weights=w, minlength=2 ** len(g.address_qubits)) == 0
        ).any():
            return True
    return False


def kcompare_table() -> pd.DataFrame:
    """Table ``tab:kcompare`` as printed in the paper, next to the reproduced
    payload-independent "undefined" column (200 repetitions, seed 0, cells next to
    the address qubit). The other columns cannot be reproduced without the payload."""
    undefined = kcompare_undefined_fraction()
    rows = []
    for n, (rms_i, rms_c, bias, und) in _KCOMPARE_PRINTED.items():
        rows.append(
            {
                "shots": n,
                "interference RMS (paper)": float(rms_i),
                "classical RMS (paper)": float(rms_c),
                "classical bias (paper)": float(bias),
                "undefined (paper)": float(und.rstrip("%")) / 100,
                "undefined (reproduced, 200 rep.)": undefined.loc[
                    n, "undefined (200 repetitions)"
                ],
                "undefined (20000 rep.)": undefined.loc[
                    n, "undefined (20000 repetitions)"
                ],
            }
        )
    return pd.DataFrame(rows).set_index("shots")


def _job_shots(device: str) -> int:
    """Shots per circuit of the coherence job on ``device``, from ``jobs.json``."""
    return int(json.loads((COHERENCE / "jobs.json").read_text())[device]["shots"])


def check_against_paper() -> pd.DataFrame:
    """``[item, paper, reproduced, match]`` for the numbers of Fig. coherences and the
    "undefined" column of Table kcompare (200 repetitions, seed 0)."""
    metrics = edge_metrics("phoenix")
    reproduced: dict[str, float] = {"fig:coherences shots": _job_shots("phoenix")}
    for part, _ in PARTS:
        m = metrics.loc[part]
        reproduced[f"fig:coherences {part} MAE"] = m["MAE"]
        reproduced[f"fig:coherences {part} rho"] = m["rho"]
        reproduced[f"fig:coherences {part} bias"] = m["bias"]
        reproduced[f"fig:coherences {part} max |err|"] = m["max |err|"]
    undefined = kcompare_undefined_fraction()
    for n in KCOMPARE_SHOTS:
        reproduced[f"tab:kcompare N={n} undefined"] = undefined.loc[
            n, "undefined (200 repetitions)"
        ]
    return compare_with_paper(PAPER_VALUES, reproduced)
