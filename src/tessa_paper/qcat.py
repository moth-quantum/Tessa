"""QCAT hardware runs: Fig. ``fig:qcat``, Fig. ``fig:teaser``, Table ``tab:hw`` and the
image columns of Table ``tab:drr``.

The QCAT image was encoded on the whole of ``ibm_phoenix`` (28x28, 120 qubits) and
``ibm_fez`` (21x21, 156 qubits), once in the single (Z) encoding and once in the pair
(Z, X) encoding, with 8192 shots per basis (jobs of 2026-09-26). Everything here is
read from ``results/hardware/qcat/`` and ``results/simulation/qcat/``:

* ``counts/*.json.gz`` + ``encodings.json``: the raw hardware counts and the decode
  data, from which :func:`rederive_from_counts` recomputes the hardware images;
* ``results/simulation/qcat/sim.npz``: the raw decodes of the local-noise and
  noiseless simulations;
* ``stats.json``: the paper's analysis (value-only ``mean_std`` DRR, no clipping),
  used as a cross-check (:func:`stats_crosscheck`) and for the circuit counts;
* ``reconstructions.npz``: the images of that analysis, drawn in the figures.

Table 1 and the image columns of Table 6 are computed by :func:`image_metrics` from
the re-derived hardware images, the stored simulation decodes and the input images in
``data/images/qcat/``, with :func:`tessa.apply_dynamic_range` for the DRR columns.

Port of the original figure script of the paper (not published).
"""

from __future__ import annotations

import gzip
import json
from functools import cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv
from matplotlib.figure import Figure
from PIL import Image

from tessa import apply_dynamic_range, decode_from_counts, load_encoding

from ._check import compare_with_paper
from .paths import DATA, RESULTS
from .pixel_art import crop_box, flatten, load_run, rgba
from .style import bare_axes, font_sizes, render_value_hue, save_figure

__all__ = [
    "KNOWN_DISCREPANCIES",
    "PAPER_VALUES",
    "check_against_paper",
    "drr_image_table",
    "hardware_table",
    "image_metrics",
    "input_images",
    "load_reconstructions",
    "load_stats",
    "qcat_figure",
    "rederivation_check",
    "rederive_from_counts",
    "reproduced_values",
    "stats_crosscheck",
    "teaser_figure",
]

QCAT = RESULTS / "hardware" / "qcat"
SIM = RESULTS / "simulation" / "qcat"
#: Input images (gray for the single encoding, colour for the pair encoding).
INPUTS = {
    "phoenix": ("qcat28.png", "qcat28_color.png"),
    "fez": ("qcat.png", "qcat_color.png"),
}
DRR = "mean_std"
DEVICES = {"phoenix": "ibm_phoenix", "fez": "ibm_fez"}
DEVICE_NAMES = {"phoenix": "Phoenix", "fez": "Fez"}
#: Runs in the order of Table hw.
RUNS = ("phoenix/single", "fez/single", "phoenix/pair", "fez/pair")
SOURCES = {"hardware": "hardware", "local_sim": "local sim.", "noiseless": "noiseless"}
#: DRR methods in the order of Table drr, with the number of input scalars each uses.
DRR_SCALARS = {
    "none": 0,
    "abs_max": 1,
    "min_max": 2,
    "percentile": 2,
    "mean_std": 2,
    "min_avg_max": 3,
}

_COLUMNS = {
    False: [
        ("input", None, "input"),
        ("hardware", "none", "hardware"),
        ("hardware", DRR, "hardware, DRR"),
        ("local_sim", "none", "local-noise sim."),
        ("local_sim", DRR, "local-noise sim., DRR"),
    ],
    True: [
        ("input", None, "Input"),
        ("hardware", "none", "Hardware"),
        ("hardware", DRR, "Hardware,\nDRR"),
        ("local_sim", "none", "Local-noise\nsim."),
    ],
}

# Numbers printed in the paper: Table 1 (tab:hw), the CZ and value counts in its run
# headers, the image columns of Table 6 (tab:drr), the slot counts of Sec. 3 and the
# abstract.
_HW_PRINTED = {
    "Phoenix, Z": {
        "hardware": ("0.130", "0.075", "0.96", None),
        "local sim.": ("0.104", "0.059", "0.97", None),
        "noiseless": ("0.004", "0.004", "1.00", None),
    },
    "Fez, Z": {
        "hardware": ("0.131", "0.078", "0.95", None),
        "local sim.": ("0.121", "0.081", "0.94", None),
        "noiseless": ("0.004", "0.004", "1.00", None),
    },
    "Phoenix, Z,X": {
        "hardware": ("0.152", "0.154", "0.69", "0.092"),
        "local sim.": ("0.059", "0.048", "0.94", "0.072"),
        "noiseless": ("0.018", "0.017", "1.00", "0.025"),
    },
    "Fez, Z,X": {
        "hardware": ("0.144", "0.134", "0.73", "0.084"),
        "local sim.": ("0.071", "0.057", "0.93", "0.083"),
        "noiseless": ("0.013", "0.012", "1.00", "0.015"),
    },
}
_HW_COLUMNS = ("value MAE", "DRR", "value rho", "hue MAE")
_DRR_PRINTED = {
    "none": ("0.130", "0.131"),
    "abs_max": ("0.120", "0.121"),
    "min_max": ("0.114", "0.086"),
    "percentile": ("0.085", "0.076"),
    "mean_std": ("0.075", "0.078"),
    "min_avg_max": ("0.112", "0.090"),
}

PAPER_VALUES: dict[str, str] = {
    **{
        f"tab:hw {run} {source} {col}": printed
        for run, rows in _HW_PRINTED.items()
        for source, values in rows.items()
        for col, printed in zip(_HW_COLUMNS, values, strict=True)
        if printed is not None
    },
    "tab:hw Phoenix, Z values": "784",
    "tab:hw Phoenix, Z CZ": "728",
    "tab:hw Fez, Z values": "441",
    "tab:hw Fez, Z CZ": "428",
    "tab:hw Phoenix, Z,X values": "1568",
    "tab:hw Phoenix, Z,X CZ": "1456",
    "tab:hw Fez, Z,X values": "882",
    "tab:hw Fez, Z,X CZ": "852",
    **{
        f"tab:drr {method} {dev}": printed
        for method, pair in _DRR_PRINTED.items()
        for dev, printed in zip(("Phoenix", "Fez"), pair, strict=True)
    },
    "sec:hwrun Phoenix slots": "792",
    "sec:hwrun Fez slots": "448",
    "abstract largest number of values encoded and decoded": "1584",
}

KNOWN_DISCREPANCIES: dict[str, str] = {
    "abstract largest number of values encoded and decoded": (
        "1584 is the capacity of the Phoenix pair encoding; the largest payload "
        "actually encoded and decoded is 1568 values (QCAT Z,X and City-pop cat)."
    ),
}


def load_stats(path: Path | None = None) -> dict[str, Any]:
    """The paper's QCAT analysis, ``results/hardware/qcat/stats.json``, as a dict."""
    return json.loads(Path(path or QCAT / "stats.json").read_text())


def load_reconstructions(path: Path | None = None) -> dict[str, np.ndarray]:
    """All arrays of ``results/hardware/qcat/reconstructions.npz`` (76 float arrays).

    Keys ``<dev>/<mode>/<source>/<drr method>`` (single: (H, W); pair: (2, H, W) =
    value, hue), ``<dev>/input_gray`` and ``<dev>/input_value_hue``.
    """
    with np.load(Path(path or QCAT / "reconstructions.npz")) as z:
        return {k: z[k] for k in z.files}


def _value_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    """The value-field metrics of one stats entry (single: gray_image, pair: value_Z)."""
    return entry["gray_image"] if "gray_image" in entry else entry["value_Z"]


def _run_label(run: str, stats: dict[str, Any]) -> str:
    dev, mode = run.split("/")
    return f"{DEVICE_NAMES[dev]}, {'Z' if mode == 'single' else 'Z,X'}"


def _n_values(run: str, stats: dict[str, Any]) -> int:
    info = stats["circuits"][run]
    return int(info["pixels"]) * len(info["fields"])


def qcat_figure(
    mode: Literal["single", "pair"],
    compact: bool = True,
    save: Path | None = None,
) -> Figure:
    """Fig. ``fig:qcat`` (a) (``mode="single"``, Z readout) or (b) (``"pair"``, Z,X).

    ``compact=True`` is the version printed in the paper
    (``fig_qcat_hardware_{z,zx}_compact.png``: input, hardware, hardware with DRR,
    local-noise simulation); ``compact=False`` is the full version
    (``fig_qcat_hardware_{z,zx}.png``, which adds the local-noise simulation with
    DRR; the paper's teaser crops its top row). Reads ``stats.json`` and
    ``reconstructions.npz`` of ``results/hardware/qcat/``. Returns the figure; with
    ``save`` it is also written there (600 dpi compact, 220 dpi full, as the
    original script).
    """
    stats, recon = load_stats(), load_reconstructions()
    columns = _COLUMNS[compact]
    fs, fs_note = (6, 5.5) if compact else (8, 7)
    sep = "\n" if compact else ", "
    with font_sizes(fs):
        fig = Figure(figsize=(3.0, 2.25) if compact else (7.2, 3.35))
        axes = fig.subplots(2, len(columns))
        for r, dev in enumerate(DEVICES):
            gray = recon[f"{dev}/input_gray"]
            run = f"{dev}/{mode}"
            for c, (source, method, title) in enumerate(columns):
                ax = axes[r, c]
                bare_axes(ax)
                if source == "input":
                    pic = (
                        gray
                        if mode == "single"
                        else render_value_hue(*recon[f"{dev}/input_value_hue"])
                    )
                    note = f"{gray.shape[0]}$\\times${gray.shape[1]}"
                else:
                    pic = recon[f"{run}/{source}/{method}"]
                    entry = stats["stats"][run][source][method]
                    if mode == "single":
                        m = entry["gray_image"]
                        note = f"MAE {m['mae']:.3f}{sep}$\\rho$ {m['pearson_rho']:.2f}"
                    else:
                        note = (
                            f"value {entry['value_Z']['mae']:.3f}{sep}"
                            f"hue {entry['hue_X']['mae']:.3f}"
                        )
                        pic = render_value_hue(*pic)
                if mode == "single":
                    ax.imshow(pic, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                else:
                    ax.imshow(pic, interpolation="nearest")
                ax.set_xlabel(note, fontsize=fs_note, labelpad=2)
                if r == 0:
                    ax.set_title(title)
                if c == 0:
                    values = _n_values(run, stats)
                    if compact:
                        label = f"IBM {DEVICE_NAMES[dev]}\n{values} values"
                    else:
                        n_q = stats["calibration"][dev]["num_qubits"]
                        label = f"{DEVICES[dev]}\n{values} values, {n_q} qubits"
                    ax.set_ylabel(label, fontsize=fs_note)
        if compact:
            fig.tight_layout(pad=0.15, h_pad=0.4, w_pad=0.2)
        else:
            fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    save_figure(fig, save, dpi=600 if compact else 220)
    return fig


def teaser_figure(save: Path | None = None) -> Figure:
    """Fig. ``fig:teaser``: input and raw hardware reconstruction of QCAT (Phoenix, Z,
    784 values) and of City-pop cat (Phoenix, hue/value, 1568 values).

    Built from the data rather than cropped from other figures: QCAT from
    ``results/hardware/qcat/reconstructions.npz`` (``phoenix/input_gray``,
    ``phoenix/single/hardware/none``), City-pop cat from
    ``results/hardware/pixel_art/PixelArt/{requests,results}/citypopcat``. The
    City-pop cat panels are cropped to the opaque pixels of the input, as in
    Fig. pixelart. Returns the figure.
    """
    recon = load_reconstructions()
    request, result = load_run("PixelArt", "citypopcat")
    src = request["images"][0]
    inp = rgba(src["rgba"], src["width"], src["height"])
    raw = rgba(result["image"], result["width"], result["height"])
    box = crop_box(inp)
    stats = load_stats()
    n_cat = int(result["n_pixels"]) * 2
    rows = [
        (
            f"QCAT\n{_n_values('phoenix/single', stats)} values",
            recon["phoenix/input_gray"],
            recon["phoenix/single/hardware/none"],
            True,
        ),
        (f"City-pop cat\n{n_cat} values", flatten(inp[box]), flatten(raw[box]), False),
    ]
    with font_sizes(7):
        fig = Figure(figsize=(2.4, 2.35))
        axes = fig.subplots(2, 2)
        for r, (label, left, right, gray) in enumerate(rows):
            for c, pic in enumerate((left, right)):
                ax = axes[r, c]
                bare_axes(ax)
                if gray:
                    ax.imshow(pic, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                else:
                    ax.imshow(pic, interpolation="nearest")
                if r == 0:
                    ax.set_title(("Input", "Hardware")[c])
            axes[r, 0].set_ylabel(label, fontsize=6.5)
        fig.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.3)
    save_figure(fig, save, dpi=600)
    return fig


def input_images() -> dict[str, dict[str, np.ndarray]]:
    """The encoded inputs from ``data/images/qcat/``: ``{dev: {"gray", "value", "hue"}}``.

    ``gray`` = 8-bit gray / 255 (single encoding); ``value`` and ``hue`` =
    ``matplotlib.colors.rgb_to_hsv`` of the colour image / 255 (pair encoding; hue as
    a fraction of the circle). Equal to ``<dev>/input_gray`` and
    ``<dev>/input_value_hue`` of ``reconstructions.npz`` (asserted in the tests).
    """
    folder = DATA / "images" / "qcat"
    out = {}
    for dev, (gray, colour) in INPUTS.items():
        g = np.asarray(Image.open(folder / gray).convert("L"), dtype=float) / 255
        rgb = np.asarray(Image.open(folder / colour).convert("RGB"), dtype=float) / 255
        hsv = rgb_to_hsv(rgb)
        out[dev] = {"gray": g, "value": hsv[..., 2], "hue": hsv[..., 0]}
    return out


@cache
def _raw_images() -> dict[tuple[str, str], np.ndarray]:
    """Raw decodes (no DRR) per (run, source): hardware re-derived from the counts,
    simulations from ``results/simulation/qcat/sim.npz``. Single: (H, W); pair:
    (2, H, W) = (value, hue)."""
    hardware = rederive_from_counts()
    out = {}
    with np.load(SIM / "sim.npz") as z:
        for run in RUNS:
            out[(run, "hardware")] = hardware[run]["none"]
            fields = 1 if run.endswith("single") else 2
            for source in ("local_sim", "noiseless"):
                arr = np.stack([z[f"{run}/{source}/{j}"] for j in range(fields)])
                out[(run, source)] = arr[0] if fields == 1 else arr
    return out


def _hue_error(hue: np.ndarray, reference: np.ndarray) -> np.ndarray:
    d = np.mod(np.abs(np.asarray(hue) - np.asarray(reference)), 1.0)
    return np.minimum(d, 1.0 - d)


def image_metrics() -> pd.DataFrame:
    """Value MAE (raw and after each DRR method), Pearson rho and hue MAE of every QCAT
    run and source, computed from the images.

    For each run of Table 1 and each source (``hardware``: decoded from the raw counts;
    ``local_sim``, ``noiseless``: the stored raw simulation decodes), the value field
    is compared with the input (:func:`input_images`): MAE without DRR and after
    ``tessa.apply_dynamic_range(value, input, method)`` for the five methods (no
    clipping; the hue is never processed), Pearson rho of the raw value field, and the
    hue MAE on the circle (fraction of a turn; pair runs only). Index (run, source);
    one column per DRR method (``none`` = raw MAE), ``rho`` and ``hue MAE``.
    """
    inputs = input_images()
    rows = []
    for (run, source), raw in _raw_images().items():
        dev, mode = run.split("/")
        ref = inputs[dev]["gray" if mode == "single" else "value"]
        value = raw if mode == "single" else raw[0]
        row: dict[str, Any] = {"run": run, "source": source}
        for method in DRR_SCALARS:
            img = value if method == "none" else apply_dynamic_range(value, ref, method)
            row[method] = float(np.abs(img - ref).mean())
        row["rho"] = float(np.corrcoef(value.ravel(), ref.ravel())[0, 1])
        row["hue MAE"] = (
            float(_hue_error(raw[1], inputs[dev]["hue"]).mean())
            if mode == "pair"
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index(["run", "source"])


def stats_crosscheck() -> pd.DataFrame:
    """:func:`image_metrics` against the stored analysis ``stats.json``.

    One row per (run, source, metric) with the computed value, the stored one and
    their absolute difference (all below 1e-14 on the machines tested).
    """
    stats = load_stats()
    rows = []
    for (run, source), m in image_metrics().iterrows():
        entry = stats["stats"][run][source]
        stored = {meth: _value_metrics(entry[meth])["mae"] for meth in DRR_SCALARS}
        stored["rho"] = _value_metrics(entry["none"])["pearson_rho"]
        if "hue_X" in entry["none"]:
            stored["hue MAE"] = entry["none"]["hue_X"]["mae"]
        for metric, value in stored.items():
            rows.append(
                {
                    "run": run,
                    "source": source,
                    "metric": metric,
                    "computed": m[metric],
                    "stats.json": value,
                    "abs diff": abs(m[metric] - value),
                }
            )
    return pd.DataFrame(rows)


def hardware_table() -> pd.DataFrame:
    """Table ``tab:hw``: value MAE, value MAE after ``mean_std`` DRR, Pearson rho of
    the value and hue MAE, for hardware, local-noise and noiseless sources of the
    four runs, at full precision.

    Index (run, source) with run labels as in the paper's row headers
    (``"Phoenix, Z"`` ...); columns also give the number of values and the CZ count
    per circuit (from ``stats.json``). The metrics are computed by
    :func:`image_metrics` (hardware from the raw counts); :func:`stats_crosscheck`
    compares them with the stored analysis.
    """
    stats = load_stats()
    metrics = image_metrics()
    rows = []
    for run in RUNS:
        circuits = stats["circuits"][run]
        cz = sorted(set(circuits["transpiled_two_qubit_gates"].values()))
        for source, name in SOURCES.items():
            m = metrics.loc[(run, source)]
            rows.append(
                {
                    "run": _run_label(run, stats),
                    "source": name,
                    "values": _n_values(run, stats),
                    "CZ": cz[0] if len(cz) == 1 else cz,
                    "value MAE": m["none"],
                    "DRR": m[DRR],
                    "value rho": m["rho"],
                    "hue MAE": m["hue MAE"],
                }
            )
    return pd.DataFrame(rows).set_index(["run", "source"])


def drr_image_table() -> pd.DataFrame:
    """Image columns of Table ``tab:drr``: MAE of the raw hardware Z images of QCAT on
    Phoenix and Fez after each dynamic-range-recovery method (value field, no clip).

    Index: method (paper order); columns ``scalars`` (input statistics used),
    ``Phoenix``, ``Fez``. Computed by :func:`image_metrics` from the hardware images
    decoded from the raw counts.
    """
    metrics = image_metrics()
    rows = []
    for method, scalars in DRR_SCALARS.items():
        row = {"method": method, "scalars": scalars}
        for dev, name in DEVICE_NAMES.items():
            row[name] = metrics.loc[(f"{dev}/single", "hardware"), method]
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def _counts(dev: str, mode: str, basis: str) -> dict[str, int]:
    with gzip.open(QCAT / "counts" / f"{dev}_{mode}_{basis}.json.gz", "rt") as f:
        return json.load(f)


def rederive_from_counts() -> dict[str, dict[str, np.ndarray]]:
    """Decode the raw hardware counts again and apply the paper's DRR.

    For each run of Table hw: ``decode_from_counts`` on
    ``results/hardware/qcat/counts/<dev>_<mode>_<basis>.json.gz`` with the encoding
    in ``encodings.json``, then ``apply_dynamic_range(value, input, "mean_std")`` on
    the value field only, without clipping (the hue is left as decoded). Returns
    ``{run: {"none": array, "mean_std": array}}`` with the shapes of
    ``reconstructions.npz``.
    """
    encodings = load_encoding(QCAT / "encodings.json")
    recon = load_reconstructions()
    out: dict[str, dict[str, np.ndarray]] = {}
    for run in RUNS:
        dev, mode = run.split("/")
        bases = ["z"] if mode == "single" else ["z", "x"]
        raw = np.asarray(
            decode_from_counts({b: _counts(dev, mode, b) for b in bases}, encodings[run])
        )
        if mode == "single":
            drr = apply_dynamic_range(raw, recon[f"{dev}/input_gray"], method=DRR)
        else:
            value = apply_dynamic_range(raw[0], recon[f"{dev}/input_value_hue"][0], DRR)
            drr = np.stack([value, raw[1]])
        out[run] = {"none": raw, DRR: np.asarray(drr)}
    return out


def rederivation_check() -> pd.DataFrame:
    """Compare :func:`rederive_from_counts` with the stored analysis.

    One row per run and method (``none``, ``mean_std``): the largest absolute
    difference from ``reconstructions.npz``, and the value (and hue) MAE recomputed
    from the re-derived images next to the stored ``stats.json`` numbers.
    """
    stats, recon = load_stats(), load_reconstructions()
    rows = []
    for run, images in rederive_from_counts().items():
        dev, mode = run.split("/")
        for method, img in images.items():
            stored = recon[f"{run}/hardware/{method}"]
            entry = stats["stats"][run]["hardware"][method]
            if mode == "single":
                mae = np.abs(img - recon[f"{dev}/input_gray"]).mean()
                hue = hue_stored = np.nan
            else:
                ref = recon[f"{dev}/input_value_hue"]
                mae = np.abs(img[0] - ref[0]).mean()
                hue = _hue_error(img[1], ref[1]).mean()
                hue_stored = entry["hue_X"]["mae"]
            rows.append(
                {
                    "run": run,
                    "method": method,
                    "max |diff| image": float(np.abs(img - stored).max()),
                    "value MAE": float(mae),
                    "value MAE stored": _value_metrics(entry)["mae"],
                    "hue MAE": float(hue),
                    "hue MAE stored": hue_stored,
                }
            )
    return pd.DataFrame(rows)


def reproduced_values() -> dict[str, float]:
    """Every item of :data:`PAPER_VALUES` computed from the stored data."""
    stats = load_stats()
    out: dict[str, float] = {}
    hw = hardware_table()
    for (run, source), row in hw.iterrows():
        for col in _HW_COLUMNS:
            out[f"tab:hw {run} {source} {col}"] = row[col]
        out[f"tab:hw {run} values"] = row["values"]
        out[f"tab:hw {run} CZ"] = row["CZ"]
    for method, row in drr_image_table().iterrows():
        for dev in ("Phoenix", "Fez"):
            out[f"tab:drr {method} {dev}"] = row[dev]
    out["sec:hwrun Phoenix slots"] = stats["circuits"]["phoenix/single"]["capacity"]
    out["sec:hwrun Fez slots"] = stats["circuits"]["fez/single"]["capacity"]
    out["abstract largest number of values encoded and decoded"] = max(
        _n_values(run, stats) for run in RUNS
    )
    return out


def check_against_paper() -> pd.DataFrame:
    """``[item, paper, reproduced, match]`` for Table hw, the CZ and value counts,
    Table drr (image columns), the slot counts and the abstract's value count."""
    return compare_with_paper(PAPER_VALUES, reproduced_values())
