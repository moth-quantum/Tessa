"""Fig. ``fig:hh``: the coupling graphs of the two devices with their
maximum-capacity 2-colouring (data qubits blue, address qubits orange).

(a) heavy hex, ``ibm_fez`` (156 qubits, capacity 448); (b) square lattice,
``ibm_phoenix`` (120 qubits, 12 x 10, capacity 792). The coupling maps come from the
calibration snapshots shipped with the library (``tessa.hardware.devices``), so no IBM
account is needed; the colouring is the one the encoder uses
(:func:`tessa.graph_from_backend`), and it is checked to equal the graphs stored with
the QCAT hardware runs (``results/hardware/qcat/encodings.json``).

(b) ports ``scripts/square_lattice_figure.py`` of the paper sources (which used
``FakeMiami``, the same 218-coupler lattice). The script that drew the published
``fig_heavyhex.png`` is lost; :func:`heavy_hex_figure` is a new drawing in the same
style (colours sampled from that figure), with a regular-hexagon layout instead of
the published figure's rotated, hand-arranged one.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from tessa import TessaGraph, graph_from_backend, load_encoding
from tessa.hardware.devices import snapshot_backend

from ._check import compare_with_paper
from .paths import RESULTS
from .style import ADDRESS_COLOUR, COUPLER_COLOUR, DATA_COLOUR, save_figure

__all__ = [
    "EXPECTED_COUNTS",
    "KNOWN_DISCREPANCIES",
    "PAPER_VALUES",
    "capacity_table",
    "check_against_paper",
    "device_graph",
    "heavy_hex_figure",
    "heavy_hex_positions",
    "square_lattice_figure",
    "square_lattice_positions",
]

DEVICES = {"fez": "ibm_fez", "phoenix": "ibm_phoenix"}

# Capacities printed in Sec. 3 and the caption of Fig. 5, and the qubit count of
# Table 2.
PAPER_VALUES: dict[str, str] = {
    "fig:hh Fez capacity": "448",
    "fig:hh Phoenix capacity": "792",
    "tab:prior Phoenix qubits": "120",
}
KNOWN_DISCREPANCIES: dict[str, str] = {}

#: Counts of the maximum-capacity colouring (not printed in the paper; asserted).
EXPECTED_COUNTS = {
    "fez": {"qubits": 156, "couplers": 176, "data": 64, "address": 92, "capacity": 448},
    "phoenix": {
        "qubits": 120,
        "couplers": 218,
        "data": 60,
        "address": 60,
        "capacity": 792,
    },
}


def device_graph(device: str) -> TessaGraph:
    """The Tessa graph (maximum-capacity colouring) of ``"fez"`` or ``"phoenix"``,
    from the device snapshot in ``tessa.hardware``."""
    return graph_from_backend(snapshot_backend(device))


def _edges(graph: TessaGraph) -> list[tuple[int, int]]:
    return sorted({tuple(sorted(e)) for e in graph.edges})


def heavy_hex_positions(graph: TessaGraph) -> dict[int, tuple[float, float]]:
    """Regular-hexagon drawing positions of a 156-qubit Heron heavy-hex chip.

    The chip has 8 rows of 16 qubits (``q % 20 < 16``, row ``q // 20``) joined by
    4 bridge qubits between neighbouring rows. Row qubits with a bridge are the
    hexagon corners; they sit a quarter of a hexagon side above or below their row
    (corners of the outer rows without a bridge alternate with their neighbours),
    the other row qubits and the bridges sit half-way between their neighbours. The
    drawing is turned by 90 degrees (rows run vertically) to fit a landscape figure.
    Every coupler then has length 1/2.
    """
    if graph.n_qubits != 156:
        raise ValueError("heavy_hex_positions knows only the 156-qubit Heron layout")
    side = 1.0
    dx, d, row_h = side * math.sqrt(3) / 4, side / 4, 1.5 * side
    row_qubits = [q for q in range(graph.n_qubits) if q % 20 < 16]
    shift: dict[int, float] = {}
    for q in row_qubits:
        row = q // 20
        for n in graph.neighbours(q):
            if n % 20 >= 16:  # a bridge: above (smaller row) or below
                other = next(m for m in graph.neighbours(n) if m != q)
                shift[q] = d if other // 20 < row else -d
    for q in row_qubits:  # outer-row corners without a bridge
        if q % 2 == 1 and q not in shift:
            for p in (q - 2, q + 2):
                if p // 20 == q // 20 and p % 20 < 16 and p in shift:
                    shift[q] = -shift[p]
                    break
    pos = {q: ((q % 20) * dx, -(q // 20) * row_h + shift.get(q, 0.0)) for q in row_qubits}
    for q in range(graph.n_qubits):
        if q not in pos:
            (xa, ya), (xb, yb) = (pos[n] for n in graph.neighbours(q))
            pos[q] = ((xa + xb) / 2, (ya + yb) / 2)
    return {q: (-y, -x) for q, (x, y) in pos.items()}


def square_lattice_positions(
    graph: TessaGraph, cols: int = 10
) -> dict[int, tuple[int, int]]:
    """Positions of the 120-qubit square lattice, drawn landscape: qubit ``q`` sits in
    device row ``q // cols`` and column ``q % cols``; device rows run left to right."""
    return {q: (q // cols, -(q % cols)) for q in range(graph.n_qubits)}


def _draw(fig: Figure, graph: TessaGraph, pos: dict, size: float) -> None:
    ax = fig.add_subplot()
    data = set(graph.data_qubits)
    for a, b in _edges(graph):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.plot([x0, x1], [y0, y1], color=COUPLER_COLOUR, lw=1.0, zorder=1)
    xy = np.array([pos[q] for q in range(graph.n_qubits)])
    colours = [
        DATA_COLOUR if q in data else ADDRESS_COLOUR for q in range(graph.n_qubits)
    ]
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=colours, lw=0, zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")


def heavy_hex_figure(save: Path | None = None) -> Figure:
    """Fig. ``fig:hh`` (a): the ``ibm_fez`` coupling graph (snapshot in
    ``tessa.hardware``) with its maximum-capacity colouring, data qubits blue,
    address qubits orange. A new drawing (the original script is lost) in the style
    of ``figures/paper/fig_heavyhex.png``. Saved at 140 dpi (as the published PNG).
    Returns the figure."""
    graph = device_graph("fez")
    fig = Figure(figsize=(6.6, 4.2))
    _draw(fig, graph, heavy_hex_positions(graph), size=60)
    save_figure(fig, save, dpi=140, bbox_inches="tight", pad_inches=0.05)
    return fig


def square_lattice_figure(save: Path | None = None) -> Figure:
    """Fig. ``fig:hh`` (b): the ``ibm_phoenix`` 12 x 10 square lattice (snapshot in
    ``tessa.hardware``; the original script used ``FakeMiami``, which has the same
    couplers) with its maximum-capacity colouring. Port of
    ``scripts/square_lattice_figure.py``; saved at 200 dpi. Returns the figure."""
    graph = device_graph("phoenix")
    fig = Figure(figsize=(4.4, 3.7))
    _draw(fig, graph, square_lattice_positions(graph), size=70)
    save_figure(fig, save, dpi=200, bbox_inches="tight", pad_inches=0.05)
    return fig


def capacity_table() -> pd.DataFrame:
    """Qubits, couplers, data and address qubits, capacity and the degrees of the data
    qubits of both devices' maximum-capacity colouring.

    Also reports whether the graph equals the one stored with the QCAT hardware runs
    (``results/hardware/qcat/encodings.json``). Index: device.
    """
    stored = load_encoding(RESULTS / "hardware" / "qcat" / "encodings.json")
    rows = []
    for dev, backend in DEVICES.items():
        graph = device_graph(dev)
        degrees = Counter(len(g.address_qubits) for g in graph.groups)
        same = stored[f"{dev}/single"].graph
        rows.append(
            {
                "device": dev,
                "backend": backend,
                "qubits": graph.n_qubits,
                "couplers": len(_edges(graph)),
                "data": len(graph.data_qubits),
                "address": graph.n_qubits - len(graph.data_qubits),
                "capacity": graph.capacity,
                "data qubits by degree": dict(sorted(degrees.items())),
                "same graph as the QCAT runs": _edges(graph) == _edges(same)
                and sorted(graph.data_qubits) == sorted(same.data_qubits),
            }
        )
    return pd.DataFrame(rows).set_index("device")


def check_against_paper() -> pd.DataFrame:
    """``[item, paper, reproduced, match]`` for the capacities of Fig. hh (and the
    Phoenix qubit count)."""
    table = capacity_table()
    reproduced = {
        "fig:hh Fez capacity": table.loc["fez", "capacity"],
        "fig:hh Phoenix capacity": table.loc["phoenix", "capacity"],
        "tab:prior Phoenix qubits": table.loc["phoenix", "qubits"],
    }
    return compare_with_paper(PAPER_VALUES, reproduced)
