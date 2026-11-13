"""Which pixels each data qubit holds.

A mapping is an order of the opaque pixels. They fill the slots in that order,
one data qubit after another, ``pixels_per_slot`` pixels to a slot:

``lines``       row by row, data qubits in index order: each holds a strip.
``squares``     tiles as square as the largest data qubit allows (4 x 4 for
                16 values), largest data qubits first, so each one fills a whole
                tile, half a tile or a quarter.
``rectangles``  the same with flat tiles (2 x 8 for 16 values).
``patches``     puzzle pieces: each data qubit gets a compact region grown at
                random from the first free pixel. Seeded, so it is repeatable.
``random``      a seeded random permutation: neighbouring pixels land on
                unrelated qubits.

Transparent pixels are skipped, so a tile or a patch that meets one carries on
into the next pixels in order.
"""

from __future__ import annotations

import numpy as np

MAPPINGS = ("lines", "squares", "rectangles", "patches", "random")


def _tile(unit: int, flat: bool) -> tuple[int, int]:
    """``(height, width)`` of a tile of ``unit`` pixels (a power of two)."""
    height = 1 << (int(np.log2(unit)) // 2)
    if flat:
        height = max(1, height // 2)
    return height, unit // height


def _tiled(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Raster indices of the opaque pixels, tile by tile, row by row in each tile."""
    y, x = np.nonzero(mask)
    order = np.lexsort((x % width, y % height, x // width, y // height))
    return (y * mask.shape[1] + x)[order]


def _patches(mask: np.ndarray, sizes: list[int], rng) -> np.ndarray:
    """Grow one compact region per size, each from the first free pixel in raster order."""
    h, w = mask.shape
    free = mask.ravel().copy()
    seeds = iter(np.flatnonzero(free))
    out: list[int] = []

    def neighbours(r):
        y, x = divmod(r, w)
        if y > 0:
            yield r - w
        if y < h - 1:
            yield r + w
        if x > 0:
            yield r - 1
        if x < w - 1:
            yield r + 1

    for size in sizes:
        region = 0
        frontier: dict[int, int] = {}  # free pixel -> how many region pixels touch it
        while region < size:
            if not frontier:  # a new piece, or one boxed in: start from the next seed
                seed = next((r for r in seeds if free[r]), None)
                if seed is None:
                    return np.array(out, dtype=int)
                frontier[seed] = 1
            cand = np.fromiter(frontier, dtype=int)
            touch = np.fromiter(frontier.values(), dtype=float)
            # Favour pixels the piece already wraps (compact pieces), and pixels with
            # few free neighbours, so no small pocket is left for a later piece.
            free_nbrs = np.array([sum(free[nb] for nb in neighbours(c)) for c in cand])
            weight = touch**3 / (1.0 + free_nbrs) ** 2
            r = int(rng.choice(cand, p=weight / weight.sum()))
            del frontier[r]
            free[r] = False
            out.append(r)
            region += 1
            for nb in neighbours(r):
                if free[nb]:
                    frontier[nb] = frontier.get(nb, 0) + 1
        # Pixels next to a finished piece wait for a later seed.
    return np.array(out, dtype=int)


def assign(mapping: str, mask: np.ndarray, graph, pixels_per_slot: int, seed: int = 0):
    """Capacity slot and position in the slot (``0`` or ``1``) of each opaque pixel.

    Pixels are numbered in raster order over the opaque ones, the same order as
    ``image[mask]``. Slots are numbered as :meth:`TessaGraph.pack` numbers
    them: data qubits in ``graph.groups`` order, ``2**degree`` slots each.
    """
    if mapping not in MAPPINGS:
        raise ValueError(f"unknown mapping {mapping!r}; choose from {MAPPINGS}")
    sizes = [g.n_values for g in graph.groups]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    groups = list(range(len(sizes)))
    if mapping in ("squares", "rectangles", "patches"):
        groups.sort(key=lambda k: -sizes[k])  # stable: aligns every piece to its tile
    slot_order = np.concatenate([starts[k] + np.arange(sizes[k]) for k in groups])
    units = [sizes[k] * pixels_per_slot for k in groups]

    rng = np.random.default_rng(seed)
    if mapping == "lines":
        raster = np.flatnonzero(mask.ravel())
    elif mapping in ("squares", "rectangles"):
        raster = _tiled(mask, *_tile(max(units), flat=mapping == "rectangles"))
    elif mapping == "patches":
        raster = _patches(mask, units, rng)
    else:
        raster = rng.permutation(np.flatnonzero(mask.ravel()))

    rank = np.cumsum(mask.ravel()) - 1  # raster index -> opaque-pixel number
    n = int(mask.sum())
    slot = np.empty(n, dtype=int)
    part = np.empty(n, dtype=int)
    j = np.arange(n)
    slot[rank[raster]] = slot_order[j // pixels_per_slot]
    part[rank[raster]] = j % pixels_per_slot
    return slot, part
