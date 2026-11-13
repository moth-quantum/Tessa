"""Regenerate the figures of the paper from the stored data and check every number.

Writes the figures to ``figures/`` (or ``--out``), then compares every number, label
and claim the paper prints with the value computed from the stored data
(:mod:`tessa_paper.checks`) and prints one summary line per module. Exits with status
1 if something differs from the paper and the difference is not one of the documented
ones (``KNOWN_DISCREPANCIES`` of each module, listed in REPRODUCIBILITY.md).

Usage, from the repository checkout (after ``pip install -e .``)::

    python scripts/make_all_figures.py
    python scripts/make_all_figures.py --out my_figures --list-differences
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from tessa_paper import (
    audio_midi,
    coherence,
    entanglement,
    genetic,
    lattices,
    pair_encoding,
    pixel_art,
    qcat,
)
from tessa_paper.checks import all_checks, differences, overview
from tessa_paper.paths import FIGURES


def figure_makers(hw: dict) -> list[tuple[str, Callable[[Path], object]]]:
    """(file name, function writing that file) for every regenerated figure."""
    table = genetic.table_genetic(hw=hw)
    return [
        (
            "fig_qcat_hardware_z_compact.png",
            lambda p: qcat.qcat_figure("single", True, p),
        ),
        ("fig_qcat_hardware_zx_compact.png", lambda p: qcat.qcat_figure("pair", True, p)),
        ("fig_qcat_hardware_z.png", lambda p: qcat.qcat_figure("single", False, p)),
        ("fig_qcat_hardware_zx.png", lambda p: qcat.qcat_figure("pair", False, p)),
        ("fig_teaser.png", lambda p: qcat.teaser_figure(save=p)),
        ("fig_heavyhex.png", lambda p: lattices.heavy_hex_figure(save=p)),
        ("fig_square_lattice.png", lambda p: lattices.square_lattice_figure(save=p)),
        ("fig_pixelart.pdf", lambda p: pixel_art.pixelart_figure(save=p)),
        ("fig_entanglement_map.png", lambda p: entanglement.fig_entanglement_map(save=p)),
        ("fig_area_law.png", lambda p: entanglement.fig_area_law(save=p)),
        ("coherences.png", lambda p: coherence.coherence_figure(save=p)),
        ("fig_pair_clamp.png", lambda p: pair_encoding.fig_pair_clamp(save=p)),
        ("sound_example_recreation.png", lambda p: audio_midi.fig_sound(save=p)),
        (
            "genetic_results.pdf",
            lambda p: genetic.fig_genetic_results(save=p, table=table),
        ),
        ("constellation.tex", lambda p: genetic.constellation_tex(save=p)),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, default=FIGURES, help="output folder (default: figures/)"
    )
    parser.add_argument(
        "--list-differences",
        action="store_true",
        help="also list every number that differs",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)

    start = time.perf_counter()
    print("Decoding the genetic hardware runs from the raw shots ...", flush=True)
    hw = genetic.hardware_results()
    for name, make in figure_makers(hw):
        t = time.perf_counter()
        result = make(args.out / name)
        if isinstance(result, matplotlib.figure.Figure):
            plt.close(result)
        print(f"wrote {name:36s} {time.perf_counter() - t:5.1f} s", flush=True)

    checks = all_checks(hw)
    summary = overview(checks)
    total = summary.iloc[-1]
    print()
    print(summary.to_string(index=False))
    print(
        f"\n{total['numbers']} numbers printed in the paper compared: {total['match']} "
        f"match, {total['differ']} differ (documented). Also compared: "
        f"{total['labels']} labels and macros ({total['labels match']} match), "
        f"{total['claims']} qualitative claims ({total['claims hold']} hold), "
        f"{total['not printed']} values not printed in the paper."
    )
    if args.list_differences:
        print()
        print(differences(checks).to_string(index=False))
    print(
        f"\nfigures in {args.out.name}/, total time {time.perf_counter() - start:.0f} s"
    )
    unexpected = int(total["UNEXPECTED"])
    if unexpected:
        print(
            f"{unexpected} comparisons differ from the paper without a documented reason"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
