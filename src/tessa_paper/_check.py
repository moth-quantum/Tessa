"""Comparison of reproduced numbers with the numbers printed in the paper.

Every comparison row carries a ``kind``, so that a summary can report printed numbers
separately from the other checks:

``"number"``
    a number printed in the paper (also a table cell such as ``92.3 (ms)``, a number
    with the code of the method that gave it);
``"label"``
    a printed label, name or LaTeX macro (a DRR method name, a job id, a date, a
    lattice size, a shot count set by a macro) or the identity of a generated file;
``"claim"``
    a qualitative statement of the text ("about 3", "agree", an inequality);
``"not printed"``
    a value the paper does not print (a "---" cell, or a value printed only by the
    original code), shown for completeness.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd

__all__ = ["KINDS", "as_printed", "compare_with_paper", "kind_of", "with_kinds"]

KINDS = ("number", "label", "claim", "not printed")

#: Printed forms that are numbers: ``0.130``, ``+0.005``, ``16%``, ``<1e-4``, and a
#: table cell with a method code, ``92.3 (ms)``.
_NUMBER = re.compile(r"^[<+-]?\d+(\.\d+)?(e-?\d+)?%?( \([a-z]+\))?$")
_NOT_PRINTED = ("---", "not in paper")


def kind_of(printed: str) -> str:
    """``"number"`` or ``"not printed"`` from the printed form; anything else must be
    classified explicitly (``compare_with_paper(kinds=...)``)."""
    text = printed.strip()
    if text in _NOT_PRINTED:
        return "not printed"
    if _NUMBER.match(text):
        return "number"
    raise ValueError(f"cannot classify the printed value {printed!r}; pass its kind")


def as_printed(value: float, printed: str) -> str:
    """Format ``value`` the way ``printed`` is written in the paper.

    The precision, an explicit ``+`` sign and a trailing ``%`` (``value`` is then a
    fraction) are copied from ``printed``; integers are written without decimals.
    """
    text = printed.strip()
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
        value = 100 * value
    sign = "+" if text.startswith("+") else ""
    decimals = len(text.split(".")[1]) if "." in text else 0
    out = f"{value:{sign}.{decimals}f}"
    if out in ("-0", "-0.0", "-0.00", "-0.000"):
        out = out[1:]
    return out + ("%" if percent else "")


def compare_with_paper(
    paper: Mapping[str, str],
    reproduced: Mapping[str, float | str],
    kinds: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Table ``[item, paper, reproduced, match, kind]`` at the paper's printed precision.

    ``paper`` maps an item to the string printed in the paper, ``reproduced`` maps
    the same items to the numbers (or, for names such as a DRR method, the strings)
    computed from the stored data. ``kinds`` gives the kind (see :data:`KINDS`) of
    items whose printed form is not a number; the others are classified by
    :func:`kind_of`.
    """
    kinds = dict(kinds or {})
    rows = []
    for item, printed in paper.items():
        value = reproduced[item]
        got = value if isinstance(value, str) else as_printed(value, printed)
        rows.append(
            {
                "item": item,
                "paper": printed,
                "reproduced": got,
                "match": got == printed,
                "kind": kinds.pop(item) if item in kinds else kind_of(printed),
            }
        )
    if kinds:
        raise KeyError(f"kinds given for unknown items: {sorted(kinds)}")
    return with_kinds(pd.DataFrame(rows))


def with_kinds(df: pd.DataFrame) -> pd.DataFrame:
    """Check that every row of a comparison table has a valid ``kind``."""
    bad = sorted(set(df["kind"]) - set(KINDS))
    if bad:
        raise ValueError(f"unknown kinds {bad}; choose from {KINDS}")
    return df[["item", "paper", "reproduced", "match", "kind"]]
