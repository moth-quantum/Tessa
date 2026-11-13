"""The classification of the comparisons with the paper and the summary table."""

from __future__ import annotations

import pandas as pd
import pytest

from tessa_paper._check import KINDS, compare_with_paper, kind_of
from tessa_paper.checks import ITEMS, differences, overview


@pytest.mark.parametrize(
    ("printed", "kind"),
    [
        ("0.130", "number"),
        ("+0.005", "number"),
        ("16%", "number"),
        ("<1e-4", "number"),
        ("1584", "number"),
        ("92.3 (ms)", "number"),
        ("---", "not printed"),
        ("not in paper", "not printed"),
    ],
)
def test_kind_of_printed_forms(printed, kind):
    assert kind_of(printed) == kind


@pytest.mark.parametrize("printed", ["mean_std", "about 3", "True", "4x4"])
def test_words_must_be_classified_explicitly(printed):
    with pytest.raises(ValueError, match="cannot classify"):
        kind_of(printed)


def test_compare_with_paper_takes_explicit_kinds():
    df = compare_with_paper(
        {"a": "0.12", "b": "mean_std"}, {"a": 0.1234, "b": "mean_std"}, {"b": "label"}
    )
    assert list(df["kind"]) == ["number", "label"]
    assert df["match"].all()
    with pytest.raises(KeyError):
        compare_with_paper({"a": "1"}, {"a": 1}, {"x": "label"})


def test_overview_reports_the_kinds_separately():
    rows = [
        ("qcat", "x1", "number", True, "match"),
        ("qcat", "x2", "number", False, "known difference"),
        ("qcat", "x3", "label", True, "match"),
        ("genetic", "x4", "claim", True, "match"),
        ("genetic", "x5", "not printed", False, "known difference"),
    ]
    checks = pd.DataFrame(rows, columns=["module", "item", "kind", "match", "status"])
    checks = checks.assign(paper="", reproduced="")
    table = overview(checks)
    assert list(table["module"]) == [*ITEMS, "total"]
    total = table.iloc[-1]
    assert (total["numbers"], total["match"], total["differ"]) == (2, 1, 1)
    assert (total["labels"], total["claims"], total["not printed"]) == (1, 1, 1)
    assert total["UNEXPECTED"] == 0
    assert list(differences(checks)["item"]) == ["x2", "x5"]
    assert set(KINDS) == {"number", "label", "claim", "not printed"}
