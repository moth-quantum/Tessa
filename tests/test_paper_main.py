"""Main-text items and the coherence appendix: reproduced numbers against the paper.

Every number the paper prints for these items must equal the reproduced one at the
paper's printed precision, except the known discrepancies, which are asserted to
equal the value the data give (so this file documents them).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tessa_paper import circuits, coherence, lattices, pixel_art, qcat
from tessa_paper._check import KINDS
from tessa_paper.paths import RESULTS

pytestmark = pytest.mark.skipif(
    not (RESULTS / "hardware" / "qcat" / "stats.json").exists(),
    reason="needs the stored results",
)

#: What the data give for each known discrepancy (the paper's value is in PAPER_VALUES).
DATA_VALUES = {
    "abstract largest number of values encoded and decoded": "1568",
    "fig:pixelart Apple -> moth (palette) shots": "16384",
    "tab:prior Tessa Z 2Q depth": "23",
    "tab:prior Tessa Z,X 2Q depth": "43",
    "tab:kcompare N=64 undefined": "22%",
}


@pytest.mark.parametrize("module", [qcat, pixel_art, lattices, circuits, coherence])
def test_numbers_match_the_paper_or_are_known_discrepancies(module):
    check = module.check_against_paper()
    assert set(check["item"]) == set(module.PAPER_VALUES)
    assert set(check["kind"]) <= set(KINDS)
    for row in check.itertuples():
        if row.item in module.KNOWN_DISCREPANCIES:
            assert not row.match, f"{row.item} now matches; remove it from the list"
            assert row.reproduced == DATA_VALUES[row.item], row.item
        else:
            assert row.match, f"{row.item}: paper {row.paper}, data {row.reproduced}"
    assert set(module.KNOWN_DISCREPANCIES) <= set(module.PAPER_VALUES)


# --- QCAT -------------------------------------------------------------------------
def test_qcat_hardware_table_values():
    table = qcat.hardware_table()
    assert table.shape == (12, 6)
    assert table.loc[("Phoenix, Z", "hardware"), "value MAE"] == pytest.approx(
        0.1299, abs=1e-4
    )
    assert table.loc[("Fez, Z,X", "local sim."), "hue MAE"] == pytest.approx(
        0.0827, abs=1e-4
    )
    assert list(table.xs("hardware", level="source")["CZ"]) == [728, 428, 1456, 852]


def test_qcat_tables_are_computed_from_the_images_and_agree_with_stats_json():
    recon = qcat.load_reconstructions()
    for dev, images in qcat.input_images().items():
        assert np.array_equal(images["gray"], recon[f"{dev}/input_gray"])
        assert np.array_equal(images["value"], recon[f"{dev}/input_value_hue"][0])
        assert np.array_equal(images["hue"], recon[f"{dev}/input_value_hue"][1])
    cross = qcat.stats_crosscheck()
    # 4 runs x 3 sources x (6 DRR methods + rho) + 2 pair runs x 3 sources (hue)
    assert len(cross) == 4 * 3 * 7 + 2 * 3
    assert cross["abs diff"].max() < 1e-12


def test_qcat_rederived_from_counts_equals_the_stored_analysis():
    check = qcat.rederivation_check()
    assert (check["max |diff| image"] < 1e-12).all()
    assert np.allclose(check["value MAE"], check["value MAE stored"], rtol=0, atol=1e-12)
    pair = check.dropna(subset=["hue MAE stored"])
    assert np.allclose(pair["hue MAE"], pair["hue MAE stored"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("compact", [True, False])
def test_qcat_figure_shows_the_stored_images(compact):
    recon = qcat.load_reconstructions()
    fig = qcat.qcat_figure("single", compact=compact)
    n_cols = 4 if compact else 5
    assert len(fig.axes) == 2 * n_cols
    for r, dev in enumerate(("phoenix", "fez")):
        shown = [
            ax.images[0].get_array() for ax in fig.axes[r * n_cols : (r + 1) * n_cols]
        ]
        assert np.array_equal(shown[0], recon[f"{dev}/input_gray"])
        assert np.array_equal(shown[1], recon[f"{dev}/single/hardware/none"])
        assert np.array_equal(shown[2], recon[f"{dev}/single/hardware/mean_std"])
        assert np.array_equal(shown[3], recon[f"{dev}/single/local_sim/none"])
    sep = "\n" if compact else ", "
    labels = [ax.get_xlabel() for ax in fig.axes]
    assert f"MAE 0.130{sep}$\\rho$ 0.96" in labels
    assert f"MAE 0.075{sep}$\\rho$ 0.96" in labels


def test_qcat_pair_figure_labels_are_the_paper_numbers():
    fig = qcat.qcat_figure("pair", compact=True)
    labels = [ax.get_xlabel() for ax in fig.axes]
    assert labels == [
        "28$\\times$28",
        "value 0.152\nhue 0.092",
        "value 0.154\nhue 0.092",
        "value 0.059\nhue 0.072",
        "21$\\times$21",
        "value 0.144\nhue 0.084",
        "value 0.134\nhue 0.084",
        "value 0.071\nhue 0.083",
    ]


def test_teaser_shows_qcat_and_citypop_cat():
    recon = qcat.load_reconstructions()
    fig = qcat.teaser_figure()
    shown = [ax.images[0].get_array() for ax in fig.axes]
    assert np.array_equal(shown[0], recon["phoenix/input_gray"])
    assert np.array_equal(shown[1], recon["phoenix/single/hardware/none"])
    assert shown[2].shape == shown[3].shape and shown[2].shape[-1] == 3
    assert [ax.get_ylabel() for ax in fig.axes[::2]] == [
        "QCAT\n784 values",
        "City-pop cat\n1568 values",
    ]


def test_drr_table_order_and_best_methods():
    table = qcat.drr_image_table()
    assert list(table.index) == list(qcat.DRR_SCALARS)
    assert table["Phoenix"].idxmin() == "mean_std"
    assert table["Fez"].idxmin() == "percentile"


# --- pixel art --------------------------------------------------------------------
def test_pixelart_figure_draws_the_stored_shot_counts():
    texts = [t.get_text() for t in pixel_art.pixelart_figure().texts]
    assert any("16384 shots per angle" in t for t in texts)
    assert sum("8192 shots per angle" in t for t in texts) == 2
    assert "1.80" not in texts and "1.51" not in texts
    with_idx = [
        t.get_text() for t in pixel_art.pixelart_figure(palette_index_error=True).texts
    ]
    assert "1.80" in with_idx and "1.51" in with_idx
    for line in ("exact 16%, idx 1.71", "MAE 0.136", "V 0.160, H 0.050", "0.198", "35%"):
        assert line in texts


def test_pixelart_figure_has_all_panels():
    fig = pixel_art.pixelart_figure()
    assert len(fig.axes) == 7 * 3 + 3 * 12


def test_pixelart_redecode_equals_the_stored_results():
    pytest.importorskip("qiskit_ibm_runtime")
    check = pixel_art.redecode_check()
    assert len(check) == 10
    assert check["structure, strings and images identical"].all()
    assert (check["max relative float difference"] <= 1e-12).all()


# --- lattices ---------------------------------------------------------------------
def test_lattice_counts():
    table = lattices.capacity_table()
    for dev, expected in lattices.EXPECTED_COUNTS.items():
        for key, value in expected.items():
            assert table.loc[dev, key] == value, (dev, key)
    assert table["same graph as the QCAT runs"].all()


def test_heavy_hex_drawing_is_regular():
    graph = lattices.device_graph("fez")
    pos = lattices.heavy_hex_positions(graph)
    lengths = [np.hypot(*np.subtract(pos[a], pos[b])) for a, b in graph.edges]
    assert np.allclose(lengths, 0.5)
    fig = lattices.heavy_hex_figure()
    assert len(fig.axes[0].lines) == 176


# --- circuits ---------------------------------------------------------------------
def test_circuit_table():
    table = circuits.circuit_table().set_index(["device", "mode", "basis"])
    assert table.loc[("phoenix", "single", "z"), "2Q depth"] == 23
    assert table.loc[("phoenix", "pair", "x"), "2Q depth"] == 43
    assert table.loc[("fez", "single", "z"), "2Q depth"] == 11
    assert table.loc[("fez", "single", "z"), "logical total depth"] == 19
    assert (table["2Q depth"] == table["logical 2Q depth"]).all()
    assert (table["CZ"] == table["logical CX"]).all()
    stats = qcat.load_stats()["circuits"]
    for (dev, mode, basis), row in table.iterrows():
        assert row["total depth"] == stats[f"{dev}/{mode}"]["transpiled_depth"][basis]
        assert row["logical total depth"] == stats[f"{dev}/{mode}"]["logical_depth"]


# --- coherence --------------------------------------------------------------------
def test_coherence_metrics_equal_the_stored_statistics():
    stats = json.loads((RESULTS / "hardware" / "coherence" / "stats.json").read_text())
    stored = stats["devices"]["phoenix"]["B"]["hardware"]["metrics"]
    metrics = coherence.edge_metrics("phoenix")
    for part in ("horizontal", "vertical", "combined"):
        assert metrics.loc[part, "MAE"] == pytest.approx(stored[part]["mae"], abs=1e-15)
        assert metrics.loc[part, "rho"] == pytest.approx(
            stored[part]["pearson_rho"], abs=1e-12
        )
        assert metrics.loc[part, "bias"] == pytest.approx(stored[part]["bias"], abs=1e-15)


def test_coherence_pairs_rederive_from_counts():
    table = coherence.rederive_pairs_from_counts("phoenix")
    assert len(table) == 1512
    assert table["|diff|"].max() == 0.0


def test_coherence_figure_panels():
    fig = coherence.coherence_figure()
    labels = [ax.get_xlabel() for ax in fig.axes if ax.get_xlabel()]
    assert labels[:3] == [
        "MAE 0.022, $\\rho$ 0.98",
        "MAE 0.019, $\\rho$ 0.99",
        "MAE 0.020, $\\rho$ 0.99",
    ]
    assert labels[3].startswith("bias +0.000")


def test_kcompare_lattice_and_undefined_column():
    graph, address = coherence.kcompare_lattice()
    assert graph.capacity == 48 and address == 6
    undefined = coherence.kcompare_undefined_fraction()
    precise = undefined["undefined (20000 repetitions)"]
    assert precise.loc[64] == pytest.approx(0.235, abs=0.01)
    assert (precise.loc[[256, 1024, 4096, 65536]] == 0).all()
    # the paper's 26% (200 repetitions) is within two standard errors of the precise value
    se = np.sqrt(precise.loc[64] * (1 - precise.loc[64]) / 200)
    assert abs(0.26 - precise.loc[64]) < 2 * se


def test_kcompare_simulation_runs_and_matches_the_interference_formula():
    payload = np.random.default_rng(1).random(48)
    table = coherence.kcompare_simulation(payload, shots=(256, 4096), repetitions=50)
    k = table["exact K_a"].iloc[0]
    bound = np.sqrt((1 - k**2) / np.array([256, 4096]))
    assert np.allclose(table["interference RMS"], bound, rtol=0.35)
    assert (table["undefined"] == 0).all()
