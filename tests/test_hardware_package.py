"""The hardware-package tool against a fake IBM service: one job for the whole
package, a dropped connection and a restart cost nothing, and each run decodes."""

from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("qiskit_ibm_runtime", reason="needs the [ibm] extra")
requests = pytest.importorskip("requests")

from tessa.hardware import package as hp


class FakeJob:
    def __init__(self, job_id, result, tags):
        self._id, self._result, self.tags = job_id, result, tags
        self.checks = 0

    def job_id(self):
        return self._id

    def status(self):
        self.checks += 1
        return "QUEUED" if self.checks == 1 else "DONE"

    def result(self):
        return self._result

    def metrics(self):
        return {"usage": {"quantum_seconds": 1}}

    def properties(self):
        raise RuntimeError("a fake chip has no calibration")

    def error_message(self):
        return None

    usage_estimation: ClassVar[dict] = {"quantum_seconds": 1}


class FakeService:
    """A 3x4 grid chip. Jobs run on Aer. ``drops`` status checks fail as if the
    connection were down."""

    store: ClassVar[dict] = {}
    submitted: ClassVar[list] = []
    drops = 0

    def __init__(self):
        from tessa import Lattice
        from tessa.backends.aer import topology_device

        self._backend = topology_device(Lattice(3, 4))

    def backend(self, name):
        return self._backend

    def job(self, job_id):
        if FakeService.drops:
            FakeService.drops -= 1
            raise requests.exceptions.ConnectionError("network is unreachable")
        return self.store[job_id]

    def jobs(self, job_tags=None, backend_name=None):
        return [j for j in self.store.values() if set(job_tags) <= set(j.tags)]


@pytest.fixture
def fake_ibm(monkeypatch):
    import qiskit_ibm_runtime
    from qiskit_aer.primitives import SamplerV2 as AerSampler

    FakeService.store.clear()
    FakeService.submitted.clear()

    class FakeSampler:
        def __init__(self, mode, options=None):
            self.tags = options["environment"]["job_tags"]

        def run(self, pubs):
            FakeService.submitted.append(pubs)
            job_id = f"job{len(FakeService.store)}"
            job = FakeJob(job_id, AerSampler(seed=0).run(pubs).result(), self.tags)
            FakeService.store[job_id] = job
            return job

    service = FakeService()

    def make_service(token=None, instance=None):
        FakeService.credentials = (token, instance)
        return service

    monkeypatch.setattr(hp, "make_service", make_service)
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", FakeSampler)
    monkeypatch.setattr(hp.time, "sleep", lambda s: None)
    return service


def _png(path, w, h, seed):
    rng = np.random.default_rng(seed)
    rgba = np.concatenate(
        [rng.integers(0, 256, (h, w, 3)), np.full((h, w, 1), 255)], axis=2
    ).astype(np.uint8)
    rgba[0, 0, 3] = 0  # one transparent corner
    Image.fromarray(rgba, "RGBA").save(path)


def _package(tmp_path):
    for i in range(2):
        _png(tmp_path / f"a{i}.png", 5, 4, i)
    (tmp_path / "pkg.toml").write_text(
        """
name = "test"
device = "fez"
shots = 20000

[[runs]]
label = "gray_zx"
image = "a0.png"
representation = "gray"
basis = "zx"

[[runs]]
label = "palette_z"
image = "a1.png"
representation = "palette"
basis = "z"
shots = 4000

[[runs]]
label = "fade"
images = ["a0.png", "a1.png"]
representation = "transition"
alphas = 3
"""
    )
    return tmp_path / "pkg.toml"


def test_a_package_is_one_job_and_survives_a_dropped_connection(fake_ibm, tmp_path):
    out = tmp_path / "out"
    FakeService.drops = 2
    hp.main(["run", str(_package(tmp_path)), "--out", str(out), "--poll", "0"])

    assert len(FakeService.submitted) == 1  # one job for every run
    pubs = FakeService.submitted[0]
    assert [p[2] for p in pubs] == [20000, 20000, 4000] + [20000] * 3
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["state"] == "finished"
    assert [h["status"] for h in manifest["history"]] == ["QUEUED", "DONE"]
    log = (out / "log.txt").read_text()
    assert log.count("status check failed") == 2

    summary = json.loads((out / "summary.json").read_text())["runs"]
    assert [r["label"] for r in summary] == ["gray_zx", "palette_z", "fade"]
    assert not any("error" in r for r in summary)
    assert summary[0]["raw.mean_abs_error_255"] < 10
    assert (out / "results" / "gray_zx" / "raw_x8.png").exists()
    assert (out / "results" / "fade" / "drr_090.0deg.png").exists()
    assert (out / "summary.csv").read_text().startswith("label,")


def test_a_restart_finds_a_job_whose_reply_was_lost(fake_ibm, tmp_path):
    package = _package(tmp_path)
    out = tmp_path / "out"
    # Submit, then lose the reply: the job exists at IBM, the manifest has no id.
    pkg, entries = hp.load_package(package)
    out.mkdir()
    log = hp.Log(out / "log.txt")
    manifest = hp.prepare(out, package, pkg, entries, log)
    hp.submit(out, manifest, fake_ibm, log)
    manifest.update(state="submitting", job_id=None)
    hp.write_json(out / "manifest.json", manifest)

    hp.main(["run", str(package), "--out", str(out), "--poll", "0"])
    assert len(FakeService.submitted) == 1  # found by its tag, not sent again
    assert json.loads((out / "manifest.json").read_text())["state"] == "finished"

    # Decoding again needs nothing but the folder.
    (out / "summary.csv").unlink()
    hp.main(["decode", str(out)])
    assert (out / "summary.csv").exists()


def test_a_changed_package_never_reuses_a_folder(fake_ibm, tmp_path):
    package = _package(tmp_path)
    out = tmp_path / "out"
    hp.main(["run", str(package), "--out", str(out), "--poll", "0"])
    package.write_text(package.read_text().replace("shots = 4000", "shots = 5000"))
    with pytest.raises(SystemExit, match="different package"):
        hp.main(["run", str(package), "--out", str(out), "--poll", "0"])


def test_a_token_in_the_package_is_refused(fake_ibm, tmp_path):
    package = _package(tmp_path)
    package.write_text(package.read_text() + '\n[ibm]\ntoken = "secret-key-123"\n')
    with pytest.raises(ValueError, match="QISKIT_IBM_TOKEN"):
        hp.main(["run", str(package), "--out", str(tmp_path / "out"), "--poll", "0"])


def test_the_instance_reaches_ibm_and_no_credentials_reach_the_folder(fake_ibm, tmp_path):
    package = _package(tmp_path)
    package.write_text(package.read_text() + '\n[ibm]\ninstance = "test-instance"\n')
    out = tmp_path / "out"
    hp.main(["run", str(package), "--out", str(out), "--poll", "0"])
    assert FakeService.credentials == (None, "test-instance")
    assert "instance" not in json.loads((out / "manifest.json").read_text())


def test_decode_can_write_elsewhere(fake_ibm, tmp_path):
    out = tmp_path / "out"
    hp.main(["run", str(_package(tmp_path)), "--out", str(out), "--poll", "0"])
    again = tmp_path / "again"
    hp.main(["decode", str(out), "--out", str(again)])
    want = json.loads((out / "results" / "gray_zx" / "result.json").read_text())
    got = json.loads((again / "results" / "gray_zx" / "result.json").read_text())
    assert got == want
