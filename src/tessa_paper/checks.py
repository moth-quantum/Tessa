"""All comparisons with the paper in one table, and the summary that the notebook and
``scripts/make_all_figures.py`` print.

Every module of :mod:`tessa_paper` has a ``check_against_paper()`` that returns one row
per compared item with a ``kind`` (see :mod:`tessa_paper._check`): a printed number, a
label or macro, a qualitative claim, or a value the paper does not print. Each row
also gets a ``status``: ``"match"``, ``"known difference"`` (listed in the module's
``KNOWN_DISCREPANCIES`` and explained in REPRODUCIBILITY.md) or ``"UNEXPECTED"``.
"""

from __future__ import annotations

from functools import partial
from types import ModuleType

import numpy as np
import pandas as pd

from . import (
    audio_midi,
    circuits,
    coherence,
    entanglement,
    genetic,
    lattices,
    pair_encoding,
    pixel_art,
    qcat,
)

__all__ = ["ITEMS", "MODULES", "all_checks", "check_table", "differences", "overview"]

#: Module name -> the paper items it covers (figure and table numbers of the paper).
ITEMS = {
    "qcat": "Fig. 4, Table 1, Table 6 (images), abstract",
    "pixel_art": "Fig. 1, Fig. 6",
    "lattices": "Fig. 5, Table 2 (qubits)",
    "circuits": "Table 2",
    "entanglement": "Fig. 7, Fig. 8, Table 3",
    "coherence": "Fig. 9, Table 4",
    "pair_encoding": "Fig. 11, Table 5",
    "audio_midi": "Fig. 12, Table 6 (pitch), Table 7",
    "genetic": "Fig. 13, Fig. 14, Table 8",
}
MODULES: dict[str, ModuleType] = {
    "qcat": qcat,
    "pixel_art": pixel_art,
    "lattices": lattices,
    "circuits": circuits,
    "entanglement": entanglement,
    "coherence": coherence,
    "pair_encoding": pair_encoding,
    "audio_midi": audio_midi,
    "genetic": genetic,
}


def check_table(name: str, hw: dict | None = None) -> pd.DataFrame:
    """``check_against_paper()`` of one module, with ``module`` and ``status`` columns.

    ``hw`` reuses a :func:`tessa_paper.genetic.hardware_results` output (genetic only).
    """
    module = MODULES[name]
    df = (
        genetic.check_against_paper(hw=hw)
        if name == "genetic"
        else module.check_against_paper()
    )
    known = set(getattr(module, "KNOWN_DISCREPANCIES", {}))
    status = np.where(
        df["match"],
        "match",
        np.where(df["item"].isin(known), "known difference", "UNEXPECTED"),
    )
    return df.assign(module=name, status=status)


def all_checks(hw: dict | None = None) -> pd.DataFrame:
    """:func:`check_table` of every module, concatenated (one row per comparison)."""
    return pd.concat([check_table(name, hw) for name in ITEMS], ignore_index=True)


def _count(df: pd.DataFrame, kind: str, status: str | None = None) -> int:
    sel = df["kind"] == kind
    if status is not None:
        sel &= df["status"] == status
    return int(sel.sum())


def overview(checks: pd.DataFrame) -> pd.DataFrame:
    """Per module and in total: how many printed numbers were compared and how many
    match or differ (documented), then the labels and macros, the qualitative claims,
    the values not printed in the paper, and any UNEXPECTED difference.
    """
    rows = []
    for name, items in [*ITEMS.items(), ("total", "")]:
        df = checks if name == "total" else checks[checks["module"] == name]
        count = partial(_count, df)
        rows.append(
            {
                "module": name,
                "paper items": items,
                "numbers": count("number"),
                "match": count("number", "match"),
                "differ": count("number", "known difference"),
                "labels": count("label"),
                "labels match": count("label", "match"),
                "claims": count("claim"),
                "claims hold": count("claim", "match"),
                "not printed": count("not printed"),
                "UNEXPECTED": int((df["status"] == "UNEXPECTED").sum()),
            }
        )
    return pd.DataFrame(rows)


def differences(checks: pd.DataFrame) -> pd.DataFrame:
    """The rows that do not match: known differences (and any UNEXPECTED one)."""
    cols = ["module", "item", "kind", "paper", "reproduced", "status"]
    return checks.loc[~checks["match"], cols].reset_index(drop=True)
