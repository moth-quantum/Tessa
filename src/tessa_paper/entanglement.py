"""App. "Entanglement of the encoded state": Fig. entmap, Fig. arealaw, Table pairs.

Exact statevector simulation of noiseless Tessa encodings on small square lattices
(at most 20 qubits). Port of the original appendix script (not published), which used
the predecessor of :func:`tessa.build_tessa_circuit`. Both produce the same circuit and
bit-identical statevectors (checked for all payloads used here), and every array
below equals the one of the original script exactly when run on the same machine
(on another CPU architecture some values differ in the last bits, about 1e-16).

Inputs: ``data/images/qcat/qcat.png`` (Fig. entmap) and seeded random payloads
``np.random.default_rng(seed).random(capacity)`` (Fig. arealaw: seeds 0-11; Table
pairs: seeds 100-139). Cached outputs: ``results/simulation/entanglement/``
(``entanglement_map.npz``, ``area_law.npz``, ``pairs.npz``, ``summary.json``);
:func:`compute_all` rewrites them (about 30 s on four cores, almost all of it the area law).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from qiskit.quantum_info import Statevector, entropy, partial_trace

from tessa import Lattice, TessaGraph, build_tessa_circuit
from tessa_paper._check import kind_of, with_kinds
from tessa_paper.paths import DATA, RESULTS

__all__ = [
    "CACHE_DIR",
    "KNOWN_DISCREPANCIES",
    "PAPER_VALUES",
    "address_pair_example",
    "area_law",
    "binary_entropy",
    "check_against_paper",
    "compute_all",
    "cut_entropy",
    "encoded_state",
    "fig_area_law",
    "fig_entanglement_map",
    "identity_check",
    "negativity",
    "pair_coherence",
    "pair_statistics",
    "qcat_entanglement_map",
    "table_pairs",
]

CACHE_DIR = RESULTS / "simulation" / "entanglement"
QCAT_IMAGE = DATA / "images" / "qcat" / "qcat.png"

#: Area law: R x 4 lattices cut after column 1, and 4 x C lattices cut after column 0.
AREA_ROWS = (2, 3, 4, 5)
AREA_COLS = (2, 3, 4, 5)
AREA_SEEDS = tuple(range(12))
#: Table pairs: 3 x 4 lattice, 40 payloads.
PAIR_SHAPE = (3, 4)
PAIR_SEEDS = tuple(range(100, 140))
#: Identity check S(rho_a) = h((1 + K_a) / 2): 3 x 4 lattice, seed 7.
IDENTITY_SEED = 7

#: Table 3 (tab:pairs) as printed. ``"<1e-4"`` is printed as $<10^{-4}$.
PAPER_VALUES: dict[str, object] = {
    "tab:pairs": {
        ("DA", 1): ("0.031", "0.334", "0.189", "1.329"),
        ("AA", 2): ("<1e-4", "0.009", "0.052", "0.396"),
        ("DD", 2): ("0", "0", "0.018", "0.290"),
        ("DA", 3): ("0", "0", "0.002", "0.070"),
        ("AA", 4): ("0", "0", "0.001", "0.010"),
        ("DD", 4): ("0", "0", "0", "0"),
        ("DA", 5): ("0", "0", "0", "0"),
    },
    "n_payloads": 40,
    "area_law_reps": 12,
}

#: No paper-vs-data discrepancy is known for this appendix.
KNOWN_DISCREPANCIES: dict[str, str] = {}

_TABLE_COLUMNS = ("mean N", "max N", "mean I", "max I")
_ZERO = 1e-9


# ---------------------------------------------------------------------------
# Quantum-information helpers
# ---------------------------------------------------------------------------
def encoded_state(geometry: Lattice | TessaGraph, values01: np.ndarray) -> Statevector:
    """Noiseless Tessa state of ``values01`` (single mode, theta = pi * v, clamp 0)."""
    return Statevector(build_tessa_circuit(geometry, np.asarray(values01, dtype=float)))


def binary_entropy(k: float) -> float:
    """Entropy in bits of the qubit state ``[[1, k], [k, 1]] / 2``, i.e. h((1+k)/2)."""
    k = min(abs(float(k)), 1.0)
    lam = np.array([(1 + k) / 2, (1 - k) / 2])
    lam = lam[lam > 1e-15]
    return float(-(lam * np.log2(lam)).sum())


def negativity(rho4: np.ndarray) -> float:
    """Negativity of a two-qubit density matrix (sum of negative partial-transpose eigenvalues)."""
    r = np.asarray(rho4).reshape(2, 2, 2, 2)
    ev = np.linalg.eigvalsh(r.transpose(0, 3, 2, 1).reshape(4, 4))
    return float(-ev[ev < 0].sum()) + 0.0


def _others(n: int, qubits: list[int]) -> list[int]:
    return [q for q in range(n) if q not in qubits]


def cut_entropy(rows: int, cols: int, last_left_col: int, seed: int) -> float:
    """Entanglement entropy (bits) between columns ``<= last_left_col`` and the rest.

    Fig. arealaw. The payload is ``default_rng(seed).random(capacity)`` on a
    ``rows x cols`` lattice. Computed as in the original script (reduced density
    matrix with ``qiskit.quantum_info``), so the figure is reproduced bit for bit.
    """
    lat = Lattice(rows, cols)
    sv = encoded_state(lat, np.random.default_rng(seed).random(lat.capacity))
    right = [q for q in range(lat.n_qubits) if (q % cols) > last_left_col]
    return float(entropy(partial_trace(sv, right), base=2))


def pair_coherence(a: float, b: float) -> float:
    """K of the address qubit of a degree-one super-pixel storing the values (a, b).

    Fig. entmap. Equals ``cos(pi (a - b) / 2)``; computed here from the circuit.
    """
    graph = TessaGraph.from_groups(2, [(0, [1])])
    sv = encoded_state(graph, np.array([a, b], dtype=float))
    rho_a = partial_trace(sv, [0]).data
    return 2.0 * float(np.real(rho_a[0, 1]))


def identity_check(seed: int = IDENTITY_SEED) -> float:
    """Max |S(rho_a) - h((1+K_a)/2)| over the address qubits of a 3x4 lattice.

    Text claim of App. entanglement ("computing S_a independently ... gives the same
    answer"): S from the reduced density matrix, K_a from the Z marginal after a
    Hadamard layer on the address register. Returns the largest difference (bits).
    """
    lat = Lattice(3, 4)
    qc = build_tessa_circuit(lat, np.random.default_rng(seed).random(lat.capacity))
    sv = Statevector(qc)
    addr = sorted({q for g in lat.groups for q in g.address_qubits})
    qh = qc.copy()
    qh.h(addr)
    probs = Statevector(qh).probabilities_dict()
    worst = 0.0
    for a in addr:
        p0 = sum(p for bits, p in probs.items() if bits[::-1][a] == "0")
        s_a = entropy(partial_trace(sv, _others(lat.n_qubits, [a])), base=2)
        worst = max(worst, abs(s_a - binary_entropy(2 * p0 - 1)))
    return worst


def address_pair_example() -> float:
    """Negativity of the address pair of one data qubit with payload (0, 1, 0, 0).

    App. entanglement, "Range of two-qubit entanglement": the example of an entangled
    address-address pair (the paper prints no number; the original script gave 0.1830).
    """
    graph = TessaGraph.from_groups(3, [(0, [1, 2])])
    sv = encoded_state(graph, np.array([0.0, 1.0, 0.0, 0.0]))
    return negativity(partial_trace(sv, [0]).data)


# ---------------------------------------------------------------------------
# The three computations, cached in results/simulation/entanglement/
# ---------------------------------------------------------------------------
def _compute_entanglement_map() -> dict[str, np.ndarray]:
    img = np.asarray(Image.open(QCAT_IMAGE).convert("L"), dtype=float) / 255.0
    h, w = img.shape
    cache: dict[tuple[float, float], float] = {}

    def k_of(a: float, b: float) -> float:
        key = (round(float(a), 5), round(float(b), 5))
        if key not in cache:
            cache[key] = pair_coherence(a, b)
        return cache[key]

    k_h, k_v = np.ones((h, w)), np.ones((h, w))
    for r in range(h):
        for c in range(w - 1):
            k_h[r, c] = k_of(img[r, c], img[r, c + 1])
    for r in range(h - 1):
        for c in range(w):
            k_v[r, c] = k_of(img[r, c], img[r + 1, c])
    s_h = np.vectorize(binary_entropy)(k_h)
    s_v = np.vectorize(binary_entropy)(k_v)
    return {"image": img, "K_h": k_h, "K_v": k_v, "S": np.maximum(s_h, s_v)}


def _compute_area_law() -> dict[str, np.ndarray]:
    s_rows = np.array([[cut_entropy(r, 4, 1, s) for s in AREA_SEEDS] for r in AREA_ROWS])
    s_cols = np.array([[cut_entropy(4, c, 0, s) for s in AREA_SEEDS] for c in AREA_COLS])
    return {
        "rows": np.array(AREA_ROWS),
        "cols": np.array(AREA_COLS),
        "seeds": np.array(AREA_SEEDS),
        "S_rows": s_rows,
        "S_cols": s_cols,
    }


def _compute_pairs() -> dict[str, np.ndarray]:
    lat = Lattice(*PAIR_SHAPE)
    n, cols = lat.n_qubits, lat.cols
    pairs = list(itertools.combinations(range(n), 2))
    neg = np.zeros((len(PAIR_SEEDS), len(pairs)))
    mi = np.zeros_like(neg)
    for i, seed in enumerate(PAIR_SEEDS):
        sv = encoded_state(lat, np.random.default_rng(seed).random(lat.capacity))
        s1 = [entropy(partial_trace(sv, _others(n, [q])), base=2) for q in range(n)]
        for j, (u, w) in enumerate(pairs):
            rho = partial_trace(sv, _others(n, [u, w]))
            neg[i, j] = negativity(rho.data)
            mi[i, j] = s1[u] + s1[w] - entropy(rho, base=2)
    is_data = np.array([((q // cols) + (q % cols)) % 2 == 0 for q in range(n)])
    return {
        "pairs": np.array(pairs),
        "seeds": np.array(PAIR_SEEDS),
        "is_data": is_data,
        "distance": np.array(
            [abs(u // cols - w // cols) + abs(u % cols - w % cols) for u, w in pairs]
        ),
        "negativity": neg,
        "mutual_information": mi,
    }


_COMPUTE = {
    "entanglement_map": _compute_entanglement_map,
    "area_law": _compute_area_law,
    "pairs": _compute_pairs,
}


def _load_or_compute(name: str, recompute: bool) -> dict[str, np.ndarray]:
    path = CACHE_DIR / f"{name}.npz"
    if recompute or not path.exists():
        return _COMPUTE[name]()
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def qcat_entanglement_map(recompute: bool = False) -> dict[str, np.ndarray]:
    """Arrays behind Fig. entmap.

    Reads ``results/simulation/entanglement/entanglement_map.npz`` (or, with
    ``recompute=True``, ``data/images/qcat/qcat.png``; < 1 s). Returns ``image``
    (21x21 gray in [0,1]), ``K_h`` / ``K_v`` (coherence of the horizontal / vertical
    pixel-pair bond at each pixel, 1 where there is no bond) and ``S`` (bits, the larger
    of the two bond entropies).
    """
    return _load_or_compute("entanglement_map", recompute)


def area_law(recompute: bool = False) -> dict[str, np.ndarray]:
    """Arrays behind Fig. arealaw.

    Reads ``results/simulation/entanglement/area_law.npz`` (``recompute=True``: about
    30 s on four cores). Returns ``rows``, ``cols``, ``seeds`` and the entropies ``S_rows`` (4 x 12,
    R x 4 lattices cut after column 1) and ``S_cols`` (4 x 12, 4 x C lattices, region
    = first column), in bits.
    """
    return _load_or_compute("area_law", recompute)


def pair_statistics(recompute: bool = False) -> dict[str, np.ndarray]:
    """Per-payload negativity and mutual information of every qubit pair (Table pairs).

    Reads ``results/simulation/entanglement/pairs.npz`` (``recompute=True``: about
    1 s). Returns ``pairs`` (66 x 2), ``is_data`` (12), ``distance`` (66),
    ``negativity`` and ``mutual_information`` (40 payloads x 66 pairs, bits).
    """
    return _load_or_compute("pairs", recompute)


def compute_all(write: bool = True) -> dict[str, dict[str, np.ndarray]]:
    """Recompute all three caches (about 30 s) and, if ``write``, store them.

    Writes ``entanglement_map.npz``, ``area_law.npz``, ``pairs.npz`` and
    ``summary.json`` (Table pairs, area-law means, identity check, address-pair
    example) to ``results/simulation/entanglement/``. Returns the arrays.
    """
    out = {name: fn() for name, fn in _COMPUTE.items()}
    if write:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for name, arrays in out.items():
            np.savez_compressed(CACHE_DIR / f"{name}.npz", **arrays)
        table = _table_from(out["pairs"])
        summary = {
            "table_pairs": [
                {
                    "pair": t,
                    "distance": int(d),
                    **{c: float(row[c]) for c in _TABLE_COLUMNS},
                }
                for (t, d), row in table.iterrows()
            ],
            "area_law": {
                "S_rows_mean": out["area_law"]["S_rows"].mean(1).tolist(),
                "S_rows_std": out["area_law"]["S_rows"].std(1).tolist(),
                "S_cols_mean": out["area_law"]["S_cols"].mean(1).tolist(),
                "S_cols_std": out["area_law"]["S_cols"].std(1).tolist(),
            },
            "identity_check_max_abs_diff": identity_check(),
            "address_pair_example_negativity": address_pair_example(),
        }
        (CACHE_DIR / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return out


# ---------------------------------------------------------------------------
# Figures and table
# ---------------------------------------------------------------------------
_RC = {"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9}


def fig_entanglement_map(save: Path | None = None, recompute: bool = False) -> Figure:
    """Fig. entmap: QCAT input and the address-qubit entanglement entropy map.

    Input: :func:`qcat_entanglement_map`. Returns the figure; saves it (300 dpi) if
    ``save`` is given (paper file name ``fig_entanglement_map.png``).
    """
    d = qcat_entanglement_map(recompute)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.75))
        ax[0].imshow(d["image"], cmap="gray", vmin=0, vmax=1)
        ax[0].set_title("(a) input")
        ax[0].axis("off")
        m = ax[1].imshow(d["S"], cmap="inferno", vmin=0, vmax=1)
        ax[1].set_title(r"(b) $S$ (bits)")
        ax[1].axis("off")
        fig.colorbar(m, ax=ax[1], fraction=0.046, pad=0.03)
        if save is not None:
            fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


def fig_area_law(save: Path | None = None, recompute: bool = False) -> Figure:
    """Fig. arealaw: entropy across a cut, mean and std over 12 payloads.

    Input: :func:`area_law`. Returns the figure; saves it (300 dpi) if ``save`` is
    given (paper file name ``fig_area_law.png``).
    """
    d = area_law(recompute)
    rows, cols = list(d["rows"]), list(d["cols"])
    s_a, s_b = d["S_rows"], d["S_cols"]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.3))
        ax[0].errorbar(
            rows,
            s_a.mean(1),
            yerr=s_a.std(1),
            fmt="o-",
            color="#1f77b4",
            capsize=3,
            label=r"$S$ across cut",
        )
        ax[0].plot(rows, rows, "k--", lw=1, label=r"bound $|\partial R|$")
        ax[0].set_xlabel(r"edges crossing the cut, $|\partial R|$")
        ax[0].set_ylabel(r"$S$ (bits)")
        ax[0].set_title("(a) lengthen the cut")
        ax[0].legend(frameon=False)
        ax[0].grid(alpha=0.3)
        ax[0].set_xticks(rows)
        nq = [4 * c for c in cols]
        ax[1].errorbar(
            nq,
            s_b.mean(1),
            yerr=s_b.std(1),
            fmt="s-",
            color="#d66e14",
            capsize=3,
            label=r"$S$ of first column",
        )
        ax[1].axhline(4, ls="--", color="k", lw=1, label=r"bound $|\partial R|=4$")
        ax[1].set_xlabel("total qubits (added away from the cut)")
        ax[1].set_ylabel(r"$S$ (bits)")
        ax[1].set_title("(b) fixed cut, growing volume")
        ax[1].set_ylim(0, 4.5)
        ax[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.84))
        ax[1].grid(alpha=0.3)
        ax[1].set_xticks(nq)
        fig.tight_layout()
        if save is not None:
            fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


def _table_from(d: dict[str, np.ndarray]) -> pd.DataFrame:
    is_data = d["is_data"]
    kinds = []
    for u, w in d["pairs"]:
        if is_data[u] and is_data[w]:
            kinds.append("DD")
        elif not is_data[u] and not is_data[w]:
            kinds.append("AA")
        else:
            kinds.append("DA")
    rows = []
    for kind, dist in sorted(
        set(zip(kinds, d["distance"].tolist(), strict=True)), key=lambda k: (k[1], k[0])
    ):
        sel = np.array([k == kind for k in kinds]) & (d["distance"] == dist)
        n, i = d["negativity"][:, sel], d["mutual_information"][:, sel]
        rows.append(
            {
                "pair": kind,
                "distance": dist,
                "n_pairs": int(sel.sum()),
                "mean N": n.mean(),
                "max N": max(n.max(), 0.0),
                "mean I": i.mean(),
                "max I": i.max(),
            }
        )
    return pd.DataFrame(rows).set_index(["pair", "distance"])


def table_pairs(recompute: bool = False) -> pd.DataFrame:
    """Table pairs: negativity N and mutual information I by pair type and distance.

    Input: :func:`pair_statistics`. Returns a DataFrame indexed by (pair, distance)
    with columns ``n_pairs`` (pairs per payload), ``mean N``, ``max N``, ``mean I``,
    ``max I`` (bits), over 40 payloads, unrounded.
    """
    return _table_from(pair_statistics(recompute))


def _format(x: float, printed: str) -> tuple[str, bool]:
    if printed == "0":
        return ("0" if abs(x) < _ZERO else f"{x:.2e}"), abs(x) < _ZERO
    if printed == "<1e-4":
        return ("<1e-4" if abs(x) < 1e-4 else f"{x:.4f}"), abs(x) < 1e-4
    decimals = len(printed.split(".")[1])
    s = f"{x:.{decimals}f}"
    return s, s == printed


def check_against_paper(recompute: bool = False) -> pd.DataFrame:
    """Every number of Table pairs and the qualitative claims, against the paper.

    Returns a DataFrame [item, paper, reproduced, match, kind] at the printed precision
    ("0" entries must vanish to 1e-9). Claim rows: the identity check, the area-law
    bound S <= |dR| and the entangled address-pair example (N > 0).
    """
    table = table_pairs(recompute)
    rows = []
    for (kind, dist), printed in PAPER_VALUES["tab:pairs"].items():
        for col, p in zip(_TABLE_COLUMNS, printed, strict=True):
            s, ok = _format(float(table.loc[(kind, dist), col]), p)
            rows.append(
                {
                    "item": f"tab:pairs {kind} d={dist} {col}",
                    "paper": p,
                    "reproduced": s,
                    "match": ok,
                    "kind": kind_of(p),
                }
            )
    diff = identity_check()
    rows.append(
        {
            "item": "S_a = h((1+K_a)/2) (max |diff|)",
            "paper": "equal",
            "reproduced": f"{diff:.1e}",
            "match": diff < 1e-10,
            "kind": "claim",
        }
    )
    al = area_law(recompute)
    below = bool(
        np.all(al["S_rows"] <= al["rows"][:, None]) and np.all(al["S_cols"] <= 4)
    )
    rows.append(
        {
            "item": "fig:arealaw S <= |dR| for every payload",
            "paper": "True",
            "reproduced": str(below),
            "match": below,
            "kind": "claim",
        }
    )
    neg = address_pair_example()
    rows.append(
        {
            "item": "address pair example entangled (N > 0)",
            "paper": "entangled",
            "reproduced": f"N = {neg:.4f}",
            "match": neg > 1e-6,
            "kind": "claim",
        }
    )
    return with_kinds(pd.DataFrame(rows))
