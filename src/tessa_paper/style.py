"""Shared plotting helpers for the paper figures.

Nothing here changes matplotlib's global state: figure functions apply their font
sizes with :func:`matplotlib.pyplot.rc_context` and close nothing they return.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import hsv_to_rgb
from matplotlib.figure import Figure

__all__ = [
    "ADDRESS_COLOUR",
    "COUPLER_COLOUR",
    "DATA_COLOUR",
    "TRANSPARENT_GRAY",
    "bare_axes",
    "font_sizes",
    "render_value_hue",
    "save_figure",
]

#: Data qubits in the coupling-graph figures (sampled from the published fig_heavyhex.png).
DATA_COLOUR = "#1f77b4"
#: Address qubits in the coupling-graph figures (sampled from fig_heavyhex.png).
ADDRESS_COLOUR = "#d66e14"
#: Couplers in the coupling-graph figures (sampled from fig_heavyhex.png).
COUPLER_COLOUR = "#799eff"
#: Fill colour of transparent (not encoded) pixels in the pixel-art figures.
TRANSPARENT_GRAY = np.array([0.86, 0.86, 0.86])


@contextmanager
def font_sizes(size: float, title: float | None = None) -> Iterator[None]:
    """Temporarily set the base font size (and the axes title size) of new figures."""
    with plt.rc_context({"font.size": size, "axes.titlesize": title or size}):
        yield


def render_value_hue(value: np.ndarray, hue: np.ndarray) -> np.ndarray:
    """RGB image of a (value, hue) pair at full saturation, as in Fig. qcat (b).

    ``value`` is clipped to [0, 1] and ``hue`` (a fraction of the circle) is taken
    modulo 1. Returns an array of shape ``value.shape + (3,)``.
    """
    v = np.clip(np.asarray(value, float), 0, 1)
    h = np.mod(np.asarray(hue, float), 1.0)
    return hsv_to_rgb(np.stack([h, np.ones_like(h), v], -1))


def bare_axes(ax: Axes, spine_width: float | None = None) -> None:
    """Remove the ticks of an image axes (optionally set the spine line width)."""
    ax.set_xticks([])
    ax.set_yticks([])
    if spine_width is not None:
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)


def save_figure(fig: Figure, path: Path | None, **kwargs) -> None:
    """Save ``fig`` to ``path`` (creating its folder) if a path is given."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **kwargs)
