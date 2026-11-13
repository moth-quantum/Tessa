"""The library against the stored hardware results under results/.

Each test is skipped when the files it needs are absent (the library can be tested
without the data). These checks tie the stored numbers to the library: decoding
the raw hardware counts with the stored encodings gives back the stored
reconstructions.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from tessa import Encoding, apply_dynamic_range, decode_from_counts, load_encoding

RUNS = ["phoenix/single", "phoenix/pair", "fez/single", "fez/pair"]


def _read_json_gz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _hardware_counts(results_file, dev, mode, basis):
    """Raw hardware counts of one QCAT run and basis (not the local simulation)."""
    hits = [
        p
        for p in results_file(f"hardware/qcat/counts/{dev}_{mode}*{basis}.json.gz")
        if "sim" not in p.name
    ]
    assert len(hits) == 1, hits
    return _read_json_gz(hits[0])


@pytest.mark.parametrize("run", RUNS)
def test_qcat_hardware_counts_decode_to_the_stored_reconstructions(results_file, run):
    """Raw counts + encodings.json -> the paper's raw hardware reconstruction (DRR none)."""
    (enc_file,) = results_file("hardware/qcat/encodings.json")
    (npz,) = results_file("hardware/qcat/reconstructions.npz")
    enc = load_encoding(enc_file)[run]
    dev, mode = run.split("/")
    bases = ["z"] if mode == "single" else ["z", "x"]
    counts = {b: _hardware_counts(results_file, dev, mode, b) for b in bases}
    got = np.asarray(decode_from_counts(counts, enc))
    with np.load(npz) as z:
        want = z[f"{run}/hardware/none"]
    assert got.shape == want.shape
    assert np.abs(got - want).max() < 1e-12


@pytest.mark.parametrize("run", RUNS)
def test_qcat_value_drr_is_mean_std_without_clipping(results_file, run):
    """The paper's DRR: mean_std on the value field only, no clip; hue untouched."""
    (npz,) = results_file("hardware/qcat/reconstructions.npz")
    dev = run.split("/")[0]
    with np.load(npz) as z:
        raw = z[f"{run}/hardware/none"]
        drr = z[f"{run}/hardware/mean_std"]
        ref = z[f"{dev}/input_gray"] if raw.ndim == 2 else z[f"{dev}/input_value_hue"][0]
    value = raw if raw.ndim == 2 else raw[0]
    want = drr if raw.ndim == 2 else drr[0]
    got = apply_dynamic_range(value, ref, method="mean_std")
    assert np.abs(got - want).max() < 1e-12
    if raw.ndim == 3:
        assert np.array_equal(drr[1], raw[1])


def _coherence(results_file):
    (design_file,) = results_file("hardware/coherence/design.json")
    (stats_file,) = results_file("hardware/coherence/stats.json")
    design = json.loads(design_file.read_text())["devices"]
    stats = json.loads(stats_file.read_text())["devices"]
    return design, stats


def _p0_all(counts):
    """P(bit q = 0) for every classical bit q (bit q of measure_all is qubit q)."""
    keys = list(counts)
    w = np.array([counts[k] for k in keys], float)
    bits = np.frombuffer("".join(keys).encode(), dtype=np.uint8).reshape(len(keys), -1)
    return 1.0 - (w[:, None] * (bits[:, ::-1] - 48)).sum(0) / w.sum()


@pytest.mark.slow
@pytest.mark.parametrize("dev", ["phoenix", "fez"])
def test_coherence_design_reproduces_the_stored_statistics(results_file, dev):
    """design.json + raw counts -> z-image errors (A) and pair edge errors (B)."""
    design, stats = _coherence(results_file)
    hits = results_file(f"hardware/coherence/counts/{dev}*.json.gz")
    assert len(hits) == 1, hits
    counts = _read_json_gz(hits[0])
    d = design[dev]
    assert sorted(counts) == sorted(d["names"])

    enc = Encoding.from_dict(d["A"]["enc"])
    B = d["B"]
    image = np.full(enc.shape, np.nan)
    for (_, p, q), (a, b) in zip(B["pairs"], B["values"], strict=True):
        image[tuple(p)], image[tuple(q)] = a, b
    assert np.isfinite(image).all()
    v = np.ravel(decode_from_counts({"z": counts["z_image"]}, enc))
    v_drr = apply_dynamic_range(v, image.ravel(), method="mean_std", clip=True)
    a_stats = stats[dev]["A"]["hardware"]
    assert np.abs(v - image.ravel()).mean() == pytest.approx(
        a_stats["z_image_mae"], abs=1e-14
    )
    assert np.abs(v_drr - image.ravel()).mean() == pytest.approx(
        a_stats["z_image_mae_drr"], abs=1e-14
    )

    edge_address = np.array([a for _, a in B["matching"]])
    p0 = np.full(len(B["pairs"]), np.nan)
    for b, batch in enumerate(B["batches"]):
        measured = _p0_all(counts[f"pairs_{b}"])
        for p, e in batch:
            p0[p] = measured[edge_address[e]]
    k = 2 * p0 - 1
    mae = np.abs(k - np.asarray(B["K_exact"])).mean()
    want = stats[dev]["B"]["hardware"]["metrics"]["all_pairs"]["mae"]
    assert mae == pytest.approx(want, abs=1e-14)


@pytest.mark.slow
@pytest.mark.parametrize("package", ["Pixels2", "PixelArt"])
def test_pixel_art_package_decodes_to_the_stored_results(results_file, tmp_path, package):
    """``python -m tessa.hardware decode`` regenerates every stored result.json."""
    pytest.importorskip("qiskit_ibm_runtime", reason="needs the [ibm] extra")
    from tessa.hardware.package import main

    (folder,) = results_file(f"hardware/pixel_art/{package}")
    stored = sorted(folder.glob("results/*/result.json"))
    if not stored or not (folder / "manifest.json").exists():
        pytest.skip(f"{package} is incomplete")
    main(["decode", str(folder), "--out", str(tmp_path)])
    for path in stored:
        label = path.parent.name
        got = json.loads((tmp_path / "results" / label / "result.json").read_text())
        want = json.loads(path.read_text())
        _assert_close(got, want, label)


def _assert_close(got, want, where):
    """Equal structure and strings; floats equal to 1e-12 relative (the stored
    results were written on another platform, so the last bit may differ)."""
    if isinstance(want, dict):
        assert got.keys() == want.keys(), where
        for key in want:
            _assert_close(got[key], want[key], f"{where}/{key}")
    elif isinstance(want, list):
        assert len(got) == len(want), where
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _assert_close(g, w, f"{where}/{i}")
    elif isinstance(want, float):
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12), where
    else:
        assert got == want, where


QCAT_IMAGES = {
    "phoenix": ("qcat28.png", "qcat28_color.png"),
    "fez": ("qcat.png", "qcat_color.png"),
}


@pytest.mark.slow
@pytest.mark.parametrize("run", RUNS)
def test_qcat_encoding_and_noiseless_simulation_regenerate(results_file, data_file, run):
    """The input images re-encode to the stored encodings, and the paper's noiseless
    simulation (bare cells, ``halo=None``, Aer seed 1, 8192 shots) regenerates."""
    from matplotlib.colors import rgb_to_hsv
    from PIL import Image

    from tessa import aer_sampler, decode, encode

    (enc_file,) = results_file("hardware/qcat/encodings.json")
    (sim_file,) = results_file("simulation/qcat/sim.npz")
    enc = load_encoding(enc_file)[run]
    dev, mode = run.split("/")
    gray, colour = QCAT_IMAGES[dev]
    if mode == "single":
        img = np.asarray(Image.open(data_file(gray)).convert("L"), float) / 255.0
        circuit, again = encode(enc.graph, img, value_range=(0, 1))
    else:
        rgb = np.asarray(Image.open(data_file(colour)).convert("RGB"), float) / 255.0
        hsv = rgb_to_hsv(rgb)
        circuit, again = encode(
            enc.graph,
            hsv[..., 2],
            hsv[..., 0],
            mode="pair",
            clamp=0.2 * np.pi,
            clamp_b=0.0,
            value_range=(0, 1),
            value_range_b=(0, 1),
        )
    assert again.to_dict() == enc.to_dict()
    out = decode(circuit, enc, aer_sampler(enc.graph, seed=1), shots=8192, halo=None)
    fields = [out] if mode == "single" else list(out)
    with np.load(sim_file) as z:
        for j, field in enumerate(fields):
            assert np.abs(field - z[f"{run}/noiseless/{j}"]).max() < 1e-12


@pytest.mark.parametrize("run", RUNS)
def test_qcat_local_sim_counts_decode_to_the_stored_simulation(results_file, run):
    """Per-cell local-noise counts -> the paper's local-noise reconstructions."""
    from tessa import decode_from_cell_counts

    (enc_file,) = results_file("hardware/qcat/encodings.json")
    (sim_file,) = results_file("simulation/qcat/sim.npz")
    dev, mode = run.split("/")
    (counts_file,) = results_file(
        f"simulation/qcat/counts/{dev}_{mode}_local_sim.json.gz"
    )
    enc = load_encoding(enc_file)[run]
    out = decode_from_cell_counts(_read_json_gz(counts_file), enc)
    fields = [out] if mode == "single" else list(out)
    with np.load(sim_file) as z:
        for j, field in enumerate(fields):
            assert np.abs(field - z[f"{run}/local_sim/{j}"]).max() < 1e-12
