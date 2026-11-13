"""Pixel art on ``ibm_phoenix``: Fig. ``fig:pixelart`` (a) single images and
(b) transitions, and the City-pop cat row of Fig. ``fig:teaser``.

Reads the two hardware packages ``results/hardware/pixel_art/{PixelArt,Pixels2}``:
``requests/<label>.json`` (the encoded input, base64 RGBA) and
``results/<label>/result.json`` (decoded images and metrics). The raw shots are in
``job/result.json``; :func:`redecode_check` decodes them again with
:func:`tessa.hardware.decode_folder`.

Port of the original figure script (not published; its compact "raw" variant is the
one from which the published ``fig_pixelart.pdf`` was made). The published PDF
was edited by hand afterwards (see :data:`KNOWN_DISCREPANCIES` and
:func:`pixelart_figure`).
"""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._check import compare_with_paper
from .paths import RESULTS
from .style import TRANSPARENT_GRAY, bare_axes, save_figure

__all__ = [
    "KNOWN_DISCREPANCIES",
    "PAPER_VALUES",
    "SINGLE",
    "TRANSITION",
    "check_against_paper",
    "crop_box",
    "flatten",
    "load_run",
    "metrics_table",
    "pixelart_figure",
    "redecode_check",
    "reproduced_values",
    "rgba",
]

PIXEL_ART = RESULTS / "hardware" / "pixel_art"
TEXTWIDTH = 6.57  # inches: a4paper, margin 0.85 in
FS_HEAD = 7.0
FS_TEXT = 6.3

#: Panel (a): (package, run label, paper name, header line), in column order.
SINGLE = [
    ("PixelArt", "spaceship", "Spaceship", "palette, $Z$"),
    ("PixelArt", "statue", "Statue", "gray, $Z,X,Y$"),
    ("PixelArt", "sleepingCat", "Sleeping cat", "gray, $Z,X,Y$"),
    ("Pixels2", "forest", "Forest", "gray, $Z,X,Y$"),
    ("PixelArt", "citypopcat", "City-pop cat", "hue/value, $Z,X,Y$"),
    ("PixelArt", "anime", "Portrait", "RGB, $Z,X,Y$"),
    ("Pixels2", "ramen", "Ramen", "HSV, $Z,X,Y$"),
]
#: Panel (b): (run label, title), in row order (all in the PixelArt package).
TRANSITION = [
    ("cat_transition", "QCAT $\\to$ dead QCAT"),
    ("apple_moth_gray", "Apple $\\to$ moth"),
    ("apple_moth_color", "Apple $\\to$ moth"),
]
_NAMES = {label: name for _, label, name, _ in SINGLE} | {
    "cat_transition": "QCAT -> dead QCAT",
    "apple_moth_gray": "Apple -> moth (gray)",
    "apple_moth_color": "Apple -> moth (palette)",
}

# Numbers printed in Fig. 6 (figures/paper/fig_pixelart.pdf, read from the rendered
# PDF) and its caption, and in the caption of Fig. 1 (teaser).
PAPER_VALUES: dict[str, str] = {
    "fig:pixelart Spaceship values": "354",
    "fig:pixelart Spaceship raw exact": "16%",
    "fig:pixelart Spaceship raw idx": "1.71",
    "fig:pixelart Spaceship DRR exact": "43%",
    "fig:pixelart Spaceship DRR idx": "1.00",
    "fig:pixelart Spaceship palette colours": "13",
    "fig:pixelart Statue values": "940",
    "fig:pixelart Statue raw MAE": "0.136",
    "fig:pixelart Statue DRR MAE": "0.128",
    "fig:pixelart Sleeping cat values": "1506",
    "fig:pixelart Sleeping cat raw MAE": "0.175",
    "fig:pixelart Sleeping cat DRR MAE": "0.154",
    "fig:pixelart Forest values": "1521",
    "fig:pixelart Forest raw MAE": "0.129",
    "fig:pixelart Forest DRR MAE": "0.115",
    "fig:pixelart City-pop cat values": "1568",
    "fig:pixelart City-pop cat raw V": "0.160",
    "fig:pixelart City-pop cat raw H": "0.050",
    "fig:pixelart City-pop cat DRR V": "0.114",
    "fig:pixelart City-pop cat DRR H": "0.050",
    "fig:pixelart Portrait values": "1257",
    "fig:pixelart Portrait raw MAE": "0.123",
    "fig:pixelart Portrait DRR MAE": "0.118",
    "fig:pixelart Ramen values": "1104",
    "fig:pixelart Ramen raw MAE": "0.127",
    "fig:pixelart Ramen DRR MAE": "0.120",
    "fig:pixelart caption DRR Spaceship": "mean_std",
    "fig:pixelart caption DRR Sleeping cat": "mean_std",
    "fig:pixelart caption DRR City-pop cat": "mean_std",
    "fig:pixelart caption DRR Statue": "percentile",
    "fig:pixelart caption DRR Portrait": "percentile",
    "fig:pixelart caption DRR Forest": "min_avg_max",
    "fig:pixelart caption DRR Ramen": "min_avg_max",
    "fig:pixelart caption shots (a)": "8192",
    "fig:pixelart QCAT -> dead QCAT values": "882",
    "fig:pixelart QCAT -> dead QCAT shots": "8192",
    "fig:pixelart QCAT -> dead QCAT raw A at 0": "0.198",
    "fig:pixelart QCAT -> dead QCAT raw all angles": "0.189",
    "fig:pixelart QCAT -> dead QCAT raw B at 90": "0.211",
    "fig:pixelart QCAT -> dead QCAT DRR A at 0": "0.145",
    "fig:pixelart QCAT -> dead QCAT DRR all angles": "0.126",
    "fig:pixelart QCAT -> dead QCAT DRR B at 90": "0.139",
    "fig:pixelart QCAT -> dead QCAT DRR method": "mean_std",
    "fig:pixelart Apple -> moth (gray) values": "1568",
    "fig:pixelart Apple -> moth (gray) shots": "8192",
    "fig:pixelart Apple -> moth (gray) raw A at 0": "0.135",
    "fig:pixelart Apple -> moth (gray) raw all angles": "0.096",
    "fig:pixelart Apple -> moth (gray) raw B at 90": "0.098",
    "fig:pixelart Apple -> moth (gray) DRR A at 0": "0.132",
    "fig:pixelart Apple -> moth (gray) DRR all angles": "0.093",
    "fig:pixelart Apple -> moth (gray) DRR B at 90": "0.085",
    "fig:pixelart Apple -> moth (gray) DRR method": "mean_std",
    "fig:pixelart Apple -> moth (palette) values": "1568",
    "fig:pixelart Apple -> moth (palette) shots": "8192",
    "fig:pixelart Apple -> moth (palette) raw A at 0": "18%",
    "fig:pixelart Apple -> moth (palette) raw B at 90": "17%",
    "fig:pixelart Apple -> moth (palette) DRR A at 0": "28%",
    "fig:pixelart Apple -> moth (palette) DRR B at 90": "35%",
    "fig:pixelart Apple -> moth (palette) DRR method": "mean_std",
    "fig:pixelart Apple -> moth (palette) palette colours": "16",
    "fig:teaser City-pop cat values": "1568",
    "fig:teaser City-pop cat shots": "8192",
}

KNOWN_DISCREPANCIES: dict[str, str] = {
    "fig:pixelart Apple -> moth (palette) shots": (
        "additional value of idx error given in the figure."
    ),
}

PUBLISHED_FIGURE_EDITS = (
    "Some manual cropping and movement of legends was performed by hand for typography",
)


def load_run(package: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(request, result)`` of one pixel-art run: ``requests/<label>.json`` and
    ``results/<label>/result.json`` of the package folder."""
    folder = PIXEL_ART / package
    request = json.loads((folder / "requests" / f"{label}.json").read_text())
    result = json.loads((folder / "results" / label / "result.json").read_text())
    return request, result


def rgba(b64: str, width: int, height: int) -> np.ndarray:
    """Decode a base64 RGBA image of the hardware packages to ``uint8 (h, w, 4)``."""
    return np.frombuffer(base64.b64decode(b64), np.uint8).reshape(height, width, 4)


def luminance(px: np.ndarray) -> np.ndarray:
    """Gray value in [0, 1] of RGBA pixels, as encoded by gray runs
    (0.299 R + 0.587 G + 0.114 B)."""
    return (px[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])) / 255.0


def flatten(px: np.ndarray, gray: bool = False) -> np.ndarray:
    """RGBA ``uint8`` -> RGB in [0, 1]; transparent pixels (alpha < 128) light gray."""
    rgb = px[..., :3] / 255.0
    if gray:
        rgb = np.repeat(luminance(px)[..., None], 3, axis=2)
    return np.where(px[..., 3:] >= 128, rgb, TRANSPARENT_GRAY)


def crop_box(*images: np.ndarray) -> tuple[slice, slice]:
    """Bounding box of the pixels opaque in any of ``images``."""
    mask = np.any([im[..., 3] >= 128 for im in images], axis=0)
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def n_values(result: dict[str, Any]) -> int:
    """Numbers stored by a run: opaque pixels times numbers per pixel."""
    per_pixel = {"palette": 1, "gray": 1, "color": 3, "huevalue": 2, "transition": 2}
    return int(result["n_pixels"]) * per_pixel[result["representation"]]


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _single_lines(result: dict[str, Any]) -> tuple[str, str]:
    """(raw line, DRR line) printed under a single-image run, errors on [0, 1]."""
    rep = result["representation"]

    def line(m: dict[str, float]) -> str:
        if rep == "palette":
            return f"exact {_pct(m['pixels_exact'])}, idx {m['mean_index_error']:.2f}"
        if rep == "gray":
            return f"MAE {m['mean_abs_error_255'] / 255:.3f}"
        if rep == "huevalue":
            return (
                f"V {m['mean_abs_error_value_255'] / 255:.3f}, "
                f"H {m['mean_abs_error_hue_degrees'] / 360:.3f}"
            )
        return f"MAE {m['mean_abs_error_rgb_255'] / 255:.3f}"

    return line(result["metrics"]), line(result["drr"]["metrics"])


def _transition_numbers(
    metrics: dict[str, float], palette: bool, index_error: bool
) -> tuple[str, str, str]:
    """Numbers under alpha = 0, 45 and 90 degrees (the middle one may be empty)."""
    if palette:
        middle = f"{metrics['mean_index_error_all_angles']:.2f}" if index_error else ""
        return (
            _pct(metrics["pixels_exact_A_at_0"]),
            middle,
            _pct(metrics["pixels_exact_B_at_90"]),
        )
    a, b, c = (
        f"{metrics[f'mean_abs_error_{k}_255'] / 255:.3f}"
        for k in ("A_at_0", "all_angles", "B_at_90")
    )
    return a, b, c


def _transition_inputs(a: np.ndarray, b: np.ndarray, gray: bool) -> list[np.ndarray]:
    """What is encoded: luminance (zero where transparent), or the colours."""
    if gray:
        return [
            np.repeat((luminance(p) * (p[..., 3] >= 128))[..., None], 3, axis=2)
            for p in (a, b)
        ]
    return [flatten(a), flatten(b)]


def _show(ax: Axes, img: np.ndarray) -> None:
    ax.imshow(img, interpolation="nearest")
    bare_axes(ax, spine_width=0.6)


def pixelart_figure(
    save: Path | None = None, palette_index_error: bool = False
) -> Figure:
    """Fig. ``fig:pixelart``: (a) seven single images (input, raw hardware, best DRR)
    and (b) three transitions (image A, five raw frames, five DRR frames, image B).

    Reads ``requests/<label>.json`` and ``results/<label>/result.json`` of
    ``results/hardware/pixel_art/{PixelArt,Pixels2}``. Every printed number and the
    shot count of each transition header come from ``result.json``; in particular the
    palette transition is labelled with its stored 16384 shots (the published PDF says
    8192). ``palette_index_error=False`` leaves out the palette transition's mean
    index error, as in the published figure; ``True`` prints it as the original
    script did. With ``save`` the figure is written there at 600 dpi (use a ``.pdf``
    or ``.png`` suffix). Returns the figure (6.57 in wide).
    """
    left, right, gap = 0.17, 0.01, 0.05  # inches
    cell = (TEXTWIDTH - left - right - 6 * gap) / 7
    head_a, text_h, run_h = 0.40, 0.13, 0.16
    rows_a = 3 * cell + 2 * text_h + 0.04
    small, gap_b, group_gap, head_b = cell / 2, 0.03, 0.12, 0.30
    rows_b = 3 * (run_h + small + text_h)
    height = 0.12 + head_a + rows_a + 0.30 + head_b + rows_b + 0.02
    fig = Figure(figsize=(TEXTWIDTH, height))

    def square(x: float, top: float, size: float) -> Axes:
        """An image cell ``size`` inches wide, ``x`` inches from the left, its top
        edge ``top`` inches below the top of the figure."""
        return fig.add_axes(
            (x / TEXTWIDTH, 1 - (top + size) / height, size / TEXTWIDTH, size / height)
        )

    def text(x: float, y: float, s: str, **kw: Any) -> None:
        kw.setdefault("fontsize", FS_TEXT)
        fig.text(x / TEXTWIDTH, 1 - y / height, s, **kw)

    def col_centre(col: int) -> float:
        return left + col * (cell + gap) + cell / 2

    # (a) single images
    y = 0.12
    text(0.0, y, "(a)", fontsize=FS_HEAD + 1, fontweight="bold", va="top")
    y += head_a
    for col, (package, label, title, mode) in enumerate(SINGLE):
        request, result = load_run(package, label)
        w, h = result["width"], result["height"]
        src = request["images"][0]
        inp = rgba(src["rgba"], src["width"], src["height"])
        raw, drr = rgba(result["image"], w, h), rgba(result["drr"]["image"], w, h)
        box = crop_box(inp)
        gray = result["representation"] == "gray"
        cx = col_centre(col)
        text(
            cx,
            y - 0.27,
            title,
            fontsize=FS_HEAD,
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        text(cx, y - 0.145, mode, ha="center", va="bottom")
        text(cx, y - 0.03, f"{n_values(result)} values", ha="center", va="bottom")
        lines = _single_lines(result)
        top = y
        for row, img in enumerate([inp, raw, drr]):
            _show(
                square(left + col * (cell + gap), top, cell),
                flatten(img[box], gray=gray and row == 0),
            )
            top += cell
            if row > 0:
                text(cx, top + 0.015, lines[row - 1], ha="center", va="top")
                top += text_h
            else:
                top += 0.04
    for row, name in enumerate(["input", "hardware", "hardware, DRR"]):
        mid = y + (0.5 * cell, 1.5 * cell + 0.04, 2.5 * cell + 0.04 + text_h)[row]
        text(
            left - 0.04, mid, name, fontsize=FS_HEAD, rotation=90, ha="right", va="center"
        )
    y += rows_a + 0.30

    # (b) transitions: one row of 12 half-size cells per run
    xs, x = [], 0.0
    for col in range(12):
        xs.append(x)
        x += small + (group_gap if col in (0, 5, 10) else gap_b)
    shift = (TEXTWIDTH - (x - group_gap)) / 2
    xs = [x + shift for x in xs]
    centre = [x + small / 2 for x in xs]
    text(0.0, y - 0.05, "(b)", fontsize=FS_HEAD + 1, fontweight="bold", va="bottom")
    for first, name in ((1, "hardware"), (6, "hardware, DRR")):
        mid = (xs[first] + xs[first + 4] + small) / 2
        text(mid, y - 0.05, name, fontsize=FS_HEAD, ha="center", va="bottom")
    for col, s in ((0, "image A"), (11, "image B")):
        text(centre[col], y + 0.10, s, fontsize=FS_HEAD, ha="center", va="bottom")
    for k, a in enumerate((0, 22.5, 45, 67.5, 90)):
        for col in (1 + k, 6 + k):
            text(centre[col], y + 0.10, f"{a:g}°", ha="center", va="bottom")
    y += 0.30 - 0.13
    for label, title in TRANSITION:
        request, result = load_run("PixelArt", label)
        w, h = result["width"], result["height"]
        a, b = (rgba(im["rgba"], w, h) for im in request["images"])
        raw_frames = [rgba(f, w, h) for f in result["frames"]]
        drr_frames = [rgba(f, w, h) for f in result["drr"]["frames"]]
        box = crop_box(a, b)
        palette = result["transition_values"] == "palette"
        kind = (
            f"palette of {len(result['transition_palette'])} colours"
            if palette
            else "gray"
        )
        text(
            xs[0],
            y + run_h - 0.025,
            f"{title}: {kind}, {n_values(result)} values, "
            f"{result['run']['shots']} shots per angle",
            fontsize=FS_HEAD,
            va="bottom",
        )
        y += run_h
        a_img, b_img = _transition_inputs(a, b, not palette)
        imgs = [a_img] + [flatten(f) for f in raw_frames + drr_frames] + [b_img]
        for col, img in enumerate(imgs):
            _show(square(xs[col], y, small), img[box])
        for first, metrics in ((1, result["metrics"]), (6, result["drr"]["metrics"])):
            numbers = _transition_numbers(metrics, palette, palette_index_error)
            for col, s in zip((first, first + 2, first + 4), numbers, strict=True):
                if s:
                    text(centre[col], y + small + 0.015, s, ha="center", va="top")
        y += small + text_h
    save_figure(fig, save, dpi=600)
    return fig


def _metric_rows(result: dict[str, Any]) -> list[tuple[str, float, float]]:
    """(metric, raw, DRR) on the figure's scale: errors on [0, 1], fractions."""
    raw, drr = result["metrics"], result["drr"]["metrics"]
    rep = result["representation"]
    if rep == "transition":
        if result["transition_values"] == "palette":
            keys = [
                ("exact A at 0", "pixels_exact_A_at_0", 1),
                ("idx all angles", "mean_index_error_all_angles", 1),
                ("exact B at 90", "pixels_exact_B_at_90", 1),
            ]
        else:
            keys = [
                ("A at 0", "mean_abs_error_A_at_0_255", 255),
                ("all angles", "mean_abs_error_all_angles_255", 255),
                ("B at 90", "mean_abs_error_B_at_90_255", 255),
            ]
    elif rep == "palette":
        keys = [("exact", "pixels_exact", 1), ("idx", "mean_index_error", 1)]
    elif rep == "gray":
        keys = [("MAE", "mean_abs_error_255", 255)]
    elif rep == "huevalue":
        keys = [
            ("V", "mean_abs_error_value_255", 255),
            ("H", "mean_abs_error_hue_degrees", 360),
        ]
    else:
        keys = [("MAE", "mean_abs_error_rgb_255", 255)]
    return [(name, raw[k] / scale, drr[k] / scale) for name, k, scale in keys]


def _representation(result: dict[str, Any]) -> str:
    if result["representation"] == "transition":
        return f"transition ({result['transition_values']})"
    return result["representation"]


def _all_runs() -> list[tuple[str, str, str]]:
    """(package, label, panel) of every run in the figure."""
    return [(p, label, "a") for p, label, _, _ in SINGLE] + [
        ("PixelArt", label, "b") for label, _ in TRANSITION
    ]


def metrics_table() -> pd.DataFrame:
    """Per-run metrics of Fig. pixelart, one row per (run, metric).

    Columns: package, label, name (paper name), panel, representation, values,
    shots, DRR method (the best DRR stored in ``result.json``), metric, raw, DRR.
    Errors are mean absolute errors on [0, 1] (hue: fraction of the circle);
    ``exact`` metrics are fractions of pixels. Reads ``results/<label>/result.json``.
    """
    rows = []
    for package, label, panel in _all_runs():
        _, result = load_run(package, label)
        for metric, raw, drr in _metric_rows(result):
            rows.append(
                {
                    "package": package,
                    "label": label,
                    "name": _NAMES[label],
                    "panel": panel,
                    "representation": _representation(result),
                    "values": n_values(result),
                    "shots": result["run"]["shots"],
                    "DRR method": result["drr"]["method"],
                    "metric": metric,
                    "raw": raw,
                    "DRR": drr,
                }
            )
    return pd.DataFrame(rows)


def reproduced_values() -> dict[str, float | str]:
    """Every item of :data:`PAPER_VALUES` computed from the stored results."""
    out: dict[str, float | str] = {}
    names = {
        "exact": "exact",
        "idx": "idx",
        "MAE": "MAE",
        "V": "V",
        "H": "H",
        "A at 0": "A at 0",
        "all angles": "all angles",
        "B at 90": "B at 90",
        "exact A at 0": "A at 0",
        "exact B at 90": "B at 90",
        "idx all angles": "idx all angles",
    }
    for package, label, panel in _all_runs():
        _, result = load_run(package, label)
        name = _NAMES[label]
        out[f"fig:pixelart {name} values"] = n_values(result)
        for metric, raw, drr in _metric_rows(result):
            out[f"fig:pixelart {name} raw {names[metric]}"] = raw
            out[f"fig:pixelart {name} DRR {names[metric]}"] = drr
        if panel == "a":
            out[f"fig:pixelart caption DRR {name}"] = result["drr"]["method"]
        else:
            out[f"fig:pixelart {name} shots"] = result["run"]["shots"]
            out[f"fig:pixelart {name} DRR method"] = result["drr"]["method"]
        if "palette" in result:
            out[f"fig:pixelart {name} palette colours"] = len(result["palette"])
        if "transition_palette" in result:
            out[f"fig:pixelart {name} palette colours"] = len(
                result["transition_palette"]
            )
    shots_a = {load_run(p, label)[1]["run"]["shots"] for p, label, _, _ in SINGLE}
    out["fig:pixelart caption shots (a)"] = shots_a.pop() if len(shots_a) == 1 else -1
    _, cat = load_run("PixelArt", "citypopcat")
    out["fig:teaser City-pop cat values"] = n_values(cat)
    out["fig:teaser City-pop cat shots"] = cat["run"]["shots"]
    return out


def check_against_paper() -> pd.DataFrame:
    """``[item, paper, reproduced, match]`` for every number printed in Fig. pixelart
    (and the City-pop cat numbers of the teaser caption)."""
    labels = {
        item: "label"
        for item in PAPER_VALUES
        if item.startswith("fig:pixelart caption DRR") or item.endswith("DRR method")
    }
    return compare_with_paper(PAPER_VALUES, reproduced_values(), labels)


def _differences(got: Any, want: Any, where: str, out: list[tuple[str, float]]) -> None:
    """Collect (path, size of difference) of two decoded result trees: structure and
    strings must agree exactly; floats contribute their relative difference."""
    if isinstance(want, dict):
        if got.keys() != want.keys():
            out.append((f"{where}: keys differ", np.inf))
            return
        for key in want:
            _differences(got[key], want[key], f"{where}/{key}", out)
    elif isinstance(want, list):
        if len(got) != len(want):
            out.append((f"{where}: lengths differ", np.inf))
            return
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _differences(g, w, f"{where}/{i}", out)
    elif isinstance(want, float):
        out.append((where, abs(got - want) / max(abs(want), 1e-300)))
    elif got != want:
        out.append((f"{where}: values differ", np.inf))


def redecode_check(out: Path | None = None) -> pd.DataFrame:
    """Decode both pixel-art packages again from their raw shots and compare.

    Runs :func:`tessa.hardware.decode_folder` on
    ``results/hardware/pixel_art/{PixelArt,Pixels2}`` (reads ``manifest.json``,
    ``requests/`` and ``job/result.json``; needs ``qiskit-ibm-runtime``) into ``out``
    (default: a temporary folder, deleted afterwards), then compares every new
    ``results/<label>/result.json`` with the stored one. Returns one row per run:
    package, label, identical structure/strings/images, number of floats that differ
    at all, largest relative float difference.
    """
    from tessa.hardware import decode_folder

    def quiet(message: str) -> None:
        """Discard the progress messages of the decode tool."""

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(out) if out is not None else Path(tmp)
        for package in ("PixelArt", "Pixels2"):
            target = base / package
            decode_folder(PIXEL_ART / package, out=target, log=quiet)
            for stored in sorted((PIXEL_ART / package / "results").glob("*/result.json")):
                label = stored.parent.name
                got = json.loads((target / "results" / label / "result.json").read_text())
                diffs: list[tuple[str, float]] = []
                _differences(got, json.loads(stored.read_text()), label, diffs)
                exact = [d for _, d in diffs if np.isinf(d)]
                floats = [d for _, d in diffs if np.isfinite(d)]
                rows.append(
                    {
                        "package": package,
                        "label": label,
                        "structure, strings and images identical": not exact,
                        "floats differing": int(sum(d > 0 for d in floats)),
                        "max relative float difference": max(floats, default=0.0),
                    }
                )
    return pd.DataFrame(rows)
