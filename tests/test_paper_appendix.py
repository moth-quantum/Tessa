"""Appendix items: reproduced numbers against the paper.

Every number the appendix prints for these items must equal the reproduced one at the
paper's printed precision, except the known discrepancies, which are asserted to equal
the value the data give (so this file documents them). Tests marked ``slow`` recompute
cached simulations and compare them with the stored copies.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tessa_paper import audio_midi, entanglement, genetic, pair_encoding
from tessa_paper.paths import RESULTS

pytestmark = pytest.mark.skipif(
    not (RESULTS / "simulation" / "entanglement" / "area_law.npz").exists(),
    reason="needs the stored results",
)

#: What the data give for each known discrepancy that is a row of a check table.
DATA_VALUES = {
    "tab:drr pitch abs_max exact (%)": "0.3",
    "tab:drr pitch abs_max MAE (st)": "7.56",
    "constellation 400 points, min spacing (rad)": "0.163",
}


def _assert_rows(check, known):
    assert len(check) > 0
    for row in check.itertuples():
        if row.item in known:
            assert not row.match, f"{row.item} now matches; remove it from the list"
            assert row.reproduced == DATA_VALUES[row.item], row.item
        else:
            assert row.match, (
                f"{row.item}: paper {row.paper}, reproduced {row.reproduced}"
            )


# ---------------------------------------------------------------------------
# Entanglement (Fig. entmap, Fig. arealaw, Table pairs)
# ---------------------------------------------------------------------------
def test_entanglement_matches_the_paper():
    check = entanglement.check_against_paper()
    assert check["item"].str.startswith("tab:pairs").sum() == 28
    _assert_rows(check, entanglement.KNOWN_DISCREPANCIES)


def test_entanglement_pairs_and_map_recompute_to_rounding():
    # Bit-identical on the aarch64 machine that wrote the files; on x86_64 some
    # negativities differ in the last bits (about 2.5e-16 absolute), from the
    # eigenvalue routines of the linear-algebra library.
    for name, fn in (
        ("pairs", entanglement.pair_statistics),
        ("entanglement_map", entanglement.qcat_entanglement_map),
    ):
        stored, fresh = fn(), fn(recompute=True)
        assert stored.keys() == fresh.keys()
        for key in stored:
            np.testing.assert_allclose(
                fresh[key], stored[key], rtol=0, atol=1e-13, err_msg=f"{name}/{key}"
            )


def test_entanglement_figures_draw():
    for fig in (entanglement.fig_entanglement_map(), entanglement.fig_area_law()):
        assert len(fig.axes) >= 2


@pytest.mark.slow
def test_entanglement_area_law_recomputes_to_rounding():
    # Bit-identical on aarch64; on x86_64 up to about 1.3e-14 (eigenvalue routines).
    stored, fresh = entanglement.area_law(), entanglement.area_law(recompute=True)
    np.testing.assert_allclose(fresh["S_rows"], stored["S_rows"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fresh["S_cols"], stored["S_cols"], rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Pair encoding (Fig. pairclamp, Table modes)
# ---------------------------------------------------------------------------
def test_pair_encoding_matches_the_paper():
    _assert_rows(pair_encoding.check_against_paper(), pair_encoding.KNOWN_DISCREPANCIES)


def test_pair_encoding_figure_draws():
    assert len(pair_encoding.fig_pair_clamp().axes) == 2


# The re-runs are bit-identical on aarch64; on x86_64 a few numbers differ in the
# last bit (largest difference 2.8e-17), so the tests allow rounding.
@pytest.mark.slow
def test_pair_encoding_e2_rerun_is_identical_to_rounding():
    diff = pair_encoding.compare_rerun("e2_clamp", pair_encoding.rerun_e2())
    assert len(diff) > 100
    assert diff["abs_diff"].max() < 1e-14


@pytest.mark.slow
def test_pair_encoding_e4_rerun_is_identical_to_rounding():
    diff = pair_encoding.compare_rerun("e4_sign_bit", pair_encoding.rerun_e4())
    assert diff["abs_diff"].max() < 1e-14


# ---------------------------------------------------------------------------
# MIDI, wavetable, Table drr pitch columns (FakeFez simulation)
# ---------------------------------------------------------------------------
def test_midi_matches_the_paper_as_simulation():
    check = audio_midi.check_against_paper()
    assert check["item"].str.startswith("tab:midi").sum() == 15
    assert check["item"].str.startswith("tab:drr pitch").sum() == 12
    _assert_rows(check, audio_midi.KNOWN_DISCREPANCIES)


def test_midi_cache_is_labelled_as_simulation():
    stored = json.loads((audio_midi.CACHE_DIR / "midi_fakefez.json").read_text())
    assert stored["kind"].startswith("SIMULATION")
    assert stored["shots"] == 8000 and stored["seed"] == 1


@pytest.mark.slow
def test_midi_simulation_recomputes_exactly():
    stored = audio_midi.midi_results()
    fresh = audio_midi.midi_results(recompute=True)
    for name, report in stored["runs"].items():
        for key, value in report.items():
            assert fresh["runs"][name][key] == pytest.approx(value, abs=1e-12), (
                name,
                key,
            )
    wt_stored, wt_fresh = audio_midi.wavetable(), audio_midi.wavetable(recompute=True)
    for key in wt_stored:
        np.testing.assert_allclose(wt_fresh[key], wt_stored[key], atol=1e-12)


# ---------------------------------------------------------------------------
# Genetic sequences (Table genetic, Fig. genetic-results, macros, Fig. constellation)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def genetic_hw():
    return genetic.hardware_results()


def test_genetic_matches_the_paper(genetic_hw):
    check = genetic.check_against_paper(hw=genetic_hw)
    assert check["item"].str.startswith("tab:genetic").sum() == 14 * 8
    assert check["item"].str.startswith("macro").sum() == 9
    _assert_rows(check, genetic.KNOWN_DISCREPANCIES)


def test_genetic_decode_uses_the_stored_constellation(monkeypatch):
    points = genetic.constellation_points()
    assert sorted(points) == [4, 16, 20, 400]
    for n, p in points.items():
        assert p.shape == (n, 3)
        assert np.allclose(np.linalg.norm(p, axis=1), 1.0, atol=1e-12)
    assert genetic.constellation_spacing()["constellation_400"] == pytest.approx(
        0.1635, abs=1e-4
    )

    def refuse(*args, **kwargs):
        raise AssertionError("the hardware decode must not recompute the constellation")

    monkeypatch.setattr(genetic, "constellation", refuse)
    import tessa.symbols

    monkeypatch.setattr(tessa.symbols, "constellation", refuse)
    hw = genetic.hardware_results()
    ours = genetic.table_genetic_tex(genetic.table_genetic(hw=hw))
    assert ours == (genetic.PAPER_TEX / "results_table.tex").read_text()


def test_genetic_constellation_can_be_regenerated():
    stored = genetic.constellation_points()
    fresh = genetic.constellation_points(regenerate=True)
    assert sorted(fresh) == sorted(stored)
    for n in stored:
        assert fresh[n].shape == stored[n].shape
        # Equal to rounding on the machines tested (x86_64, aarch64); not asserted
        # bit for bit because the repulsion iteration is platform-sensitive.
        assert np.abs(fresh[n] - stored[n]).max() < 1e-3


def test_genetic_table_tex_is_the_papers(genetic_hw):
    ours = genetic.table_genetic_tex(genetic.table_genetic(hw=genetic_hw))
    assert ours == (genetic.PAPER_TEX / "results_table.tex").read_text()
    assert (
        genetic.macros_tex(genetic_hw)
        == (genetic.PAPER_TEX / "results_macros.tex").read_text()
    )


def test_genetic_figure_draws(genetic_hw):
    fig = genetic.fig_genetic_results(table=genetic.table_genetic(hw=genetic_hw))
    assert len(fig.axes) == 2


@pytest.mark.slow
def test_genetic_simulation_rerun_one_sequence():
    stored = genetic.simulation_results()["sequences"]["P53_HUMAN"]
    fresh = genetic.rerun_simulation(keys=("P53_HUMAN",))["P53_HUMAN"]
    for label, entry in fresh.items():
        assert entry["best_drr"] == stored[label]["best_drr"]
        # json.dumps: "second" is NaN for one symbol per slot, and NaN != NaN
        assert json.dumps(entry["metrics"], sort_keys=True) == json.dumps(
            stored[label]["metrics"], sort_keys=True
        )
