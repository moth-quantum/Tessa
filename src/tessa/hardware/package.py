"""Hardware packages: many images in one IBM job, saved so that nothing is lost.

A package file (TOML) lists the images and their settings (see
:mod:`tessa.hardware.pipeline` for the options). All circuits of all runs go into
one IBM job. Everything the run needs is saved in an output folder as it happens,
so a lost connection or a restart costs nothing: run the same command again and
it picks up where it stopped. The pixel-art hardware results of the paper are two
such folders.

Command line (``python -m tessa.hardware <command> ...``)::

    check  <package.toml>             build every plan on the stored calibration
                                      snapshot; contacts nobody
    run    <package.toml> [--out DIR] submit once, wait, fetch and decode
    resume <folder>                   keep waiting on a submitted package
    decode <folder> [--out DIR]       decode the saved raw result again, offline
    snapshot <ibm_backend> <file>     save a chip's calibration as a snapshot

``decode`` needs only the folder (``manifest.json``, ``requests/`` and
``job/result.json``) and ``qiskit-ibm-runtime`` to read the stored result.

Credentials never appear in a package file or an output folder. The API token
comes from ``$QISKIT_IBM_TOKEN`` or from the account saved with
``QiskitRuntimeService.save_account``; the instance CRN from ``$QISKIT_IBM_INSTANCE``,
the saved account, or the optional ``[ibm] instance`` key of the package.

The output folder holds::

    manifest.json          state, job id, job tag, the chip graph, which circuits are whose
    package.toml           the package file as submitted; inputs/ holds its images
    requests/<label>.json  each image's full request
    circuits.qpy           the transpiled circuits, in job order
    calibration_submit.json  the chip's calibration when the job was submitted
    log.txt                every step and every status change, with times
    job/result.json        the raw job result (qiskit_ibm_runtime.RuntimeDecoder reads it)
    job/metrics.json       IBM's timestamps and usage; job/calibration_run.json, the
                           calibration at execution time; job/error.json if it failed
    results/<label>/       result.json (decoded image(s) and metrics), raw and DRR PNGs
    summary.csv, summary.json  one row per image: settings and metrics, raw and DRR
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import shutil
import time
import tomllib
import uuid
from pathlib import Path

import numpy as np

from ..backends.base import counts_of
from ..backends.ibm import make_service
from ..graph import graph_from_coupling
from . import devices
from .pipeline import build_plan, shots_of
from .runners import decode_hardware, hardware_circuits

#: Settings a run may give; any of them may also sit at the top of the package.
RUN_KEYS = {
    "representation",
    "basis",
    "shots",
    "alphas",
    "colorspace",
    "transition_values",
    "mapping",
    "seed",
}
PACKAGE_KEYS = RUN_KEYS | {"name", "device", "runs", "ibm"}
IBM_KEYS = {"instance"}
FINAL = ("DONE", "ERROR", "CANCELLED")


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def write_json(path: Path, data) -> None:
    """Write through a temporary file, so a power cut never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    os.replace(tmp, path)


class Log:
    """Print timestamped lines, and append them to ``path`` if one is given."""

    def __init__(self, path: Path | None):
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"{now()}  {message}"
        print(line, flush=True)
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(line + "\n")


def transient(exc: BaseException) -> bool:
    """A network drop, a timeout or a server-side hiccup: worth trying again.
    A refused request (bad key, bad circuit) is not."""
    import requests

    seen = exc
    while seen is not None:
        if isinstance(
            seen, requests.exceptions.ConnectionError | requests.exceptions.Timeout
        ):
            return True
        if isinstance(seen, ConnectionError | TimeoutError):
            return True
        code = getattr(seen, "status_code", None)
        if code is not None and (code == -1 or code >= 500 or code in (408, 429)):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def patiently(what: str, fn, log: Log, first: float = 30, cap: float = 600):
    """``fn()``, tried again with growing pauses for as long as the failure is
    transient. Any other failure is raised."""
    delay = first
    while True:
        try:
            return fn()
        except Exception as exc:
            if not transient(exc):
                raise
            log(
                f"{what} failed ({type(exc).__name__}: {exc}); trying again in {delay:.0f} s"
            )
            time.sleep(delay)
            delay = min(cap, delay * 2)


# ----------------------------------------------------------------------
# The package file
# ----------------------------------------------------------------------
def image_of(path: Path) -> dict:
    """A PNG as a request holds it: width, height and base64 RGBA bytes."""
    from PIL import Image

    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    h, w = rgba.shape[:2]
    return {"width": w, "height": h, "rgba": base64.b64encode(rgba.tobytes()).decode()}


def load_package(path: Path) -> tuple[dict, list[dict]]:
    """The package file -> (its top level, one entry per run with its request)."""
    package = tomllib.loads(path.read_text())
    unknown = set(package) - PACKAGE_KEYS
    if unknown:
        raise ValueError(
            f"unknown package keys {sorted(unknown)}; use {sorted(PACKAGE_KEYS)}"
        )
    if "name" not in package:
        raise ValueError("give the package a name")
    if "token" in package.get("ibm", {}):
        raise ValueError(
            "do not put an API token in a package file; export QISKIT_IBM_TOKEN or "
            "save an account with QiskitRuntimeService.save_account"
        )
    unknown = set(package.get("ibm", {})) - IBM_KEYS
    if unknown:
        raise ValueError(f"unknown [ibm] keys {sorted(unknown)}; use {sorted(IBM_KEYS)}")
    device = package.get("device", "phoenix")
    if device not in devices.DEVICES:
        raise ValueError(
            f"unknown device {device!r}; choose from {sorted(devices.DEVICES)}"
        )
    runs = package.get("runs", [])
    if not runs:
        raise ValueError("the package has no [[runs]]")
    defaults = {k: v for k, v in package.items() if k in RUN_KEYS}

    entries, labels = [], set()
    for i, run in enumerate(runs):
        unknown = set(run) - RUN_KEYS - {"label", "image", "images"}
        if unknown:
            raise ValueError(f"run {i}: unknown keys {sorted(unknown)}")
        label = run.get("label") or f"run{i:02d}"
        if label in labels or not label.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"run {i}: labels must be unique, letters, digits, _ and -")
        labels.add(label)
        files = run.get("images") or ([run["image"]] if "image" in run else [])
        if not files:
            raise ValueError(
                f"run {label!r}: give an image (or images, for a transition)"
            )
        files = [(path.parent / f).resolve() for f in files]
        request = {**defaults, **{k: v for k, v in run.items() if k in RUN_KEYS}}
        request.update(device=device, images=[image_of(f) for f in files])
        entries.append(
            {"label": label, "files": [str(f) for f in files], "request": request}
        )
    return package, entries


def connect(package: dict | None):
    """A function that opens the IBM service; the token comes from the environment
    or the saved account, the instance from the package's ``[ibm]`` table if given."""
    instance = (package or {}).get("ibm", {}).get("instance")
    return lambda: make_service(instance=instance)


def plans_on(graph, entries: list[dict]) -> list:
    out = []
    for e in entries:
        try:
            out.append(build_plan(e["request"], graph))
        except ValueError as exc:
            raise ValueError(f"run {e['label']!r}: {exc}") from exc
    return out


# ----------------------------------------------------------------------
# check
# ----------------------------------------------------------------------
def check(args) -> None:
    package, entries = load_package(Path(args.package))
    device = package.get("device", "phoenix")
    graph = devices.graph(device)
    plans = plans_on(graph, entries)
    print(
        f"{package['name']}: {len(entries)} runs on {devices.DEVICES[device]} "
        f"(calibration snapshot, {graph.capacity} slots)"
    )
    print(
        f"{'label':24} {'encode':22} {'pixels':>6} {'slots':>5} {'circuits':>8} {'shots':>6}"
    )
    total = 0
    for e, plan in zip(entries, plans, strict=True):
        r = e["request"]
        what = f"{plan.representation} {plan.basis}"
        if plan.representation == "transition":
            what += f" {r.get('transition_values', 'gray')}"
        n = len(hardware_circuits(plan))
        total += n
        print(
            f"{e['label']:24} {what:22} {int(plan.mask.sum()):>6} "
            f"{np.unique(plan.slot).size:>5} {n:>8} {shots_of(r):>6}"
        )
    print(f"{total} circuits in one job. Nothing was sent.")


# ----------------------------------------------------------------------
# run: prepare, submit, wait, fetch, decode
# ----------------------------------------------------------------------
def fingerprint(entries: list[dict]) -> str:
    requests = [e["request"] for e in entries]
    return hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()


def prepare(
    folder: Path, package_path: Path, package: dict, entries: list[dict], log: Log
):
    """Build and transpile every circuit on the live chip, and save it all."""
    from qiskit import qpy, transpile

    device = package.get("device", "phoenix")
    service = patiently("connecting to IBM", connect(package), log)
    backend = patiently(
        "fetching the backend", lambda: service.backend(devices.DEVICES[device]), log
    )
    graph = graph_from_coupling(backend.coupling_map.get_edges(), backend.num_qubits)
    plans = plans_on(graph, entries)

    circuits, runs = [], []
    for e, plan in zip(entries, plans, strict=True):
        own = hardware_circuits(plan)
        request_file = f"requests/{e['label']}.json"
        write_json(folder / request_file, e["request"])
        runs.append(
            {
                "label": e["label"],
                "images": [Path(f).name for f in e["files"]],
                "request_file": request_file,
                "first": len(circuits),
                "n_circuits": len(own),
                "shots": shots_of(e["request"]),
            }
        )
        circuits += own
    log(f"transpiling {len(circuits)} circuits for {backend.name}")
    transpiled = transpile(
        circuits,
        backend,
        initial_layout=list(range(graph.n_qubits)),
        optimization_level=1,
        seed_transpiler=0,
        num_processes=1,
    )
    with (folder / "circuits.qpy").open("wb") as f:
        qpy.dump(transpiled, f)

    shutil.copy(package_path, folder / "package.toml")
    for e in entries:
        for f in e["files"]:
            (folder / "inputs").mkdir(exist_ok=True)
            shutil.copy(f, folder / "inputs" / Path(f).name)
    try:
        write_json(folder / "calibration_submit.json", backend.properties().to_dict())
    except Exception as exc:  # the calibration is a record, never a reason to stop
        log(f"could not save the calibration ({type(exc).__name__}: {exc})")

    manifest = {
        "package": package["name"],
        "fingerprint": fingerprint(entries),
        "tag": f"tessa-{package['name']}-{uuid.uuid4().hex[:8]}",
        "created": now(),
        "state": "prepared",
        "device": device,
        "backend": backend.name,
        "job_id": None,
        "submitted": None,
        "finished": None,
        "status": None,
        "n_circuits": len(transpiled),
        "depths": [c.depth() for c in transpiled],
        "edges": [list(e) for e in sorted(graph.edges)],
        "n_qubits": graph.n_qubits,
        "runs": runs,
        "history": [],
    }
    write_json(folder / "manifest.json", manifest)
    log(f"prepared {len(runs)} runs, {len(transpiled)} circuits, tag {manifest['tag']}")
    return manifest


def submit(folder: Path, manifest: dict, service, log: Log) -> None:
    """Send the saved circuits as one job, tagged so a lost reply can be found."""
    from qiskit import qpy
    from qiskit_ibm_runtime import SamplerV2

    if manifest["state"] == "submitting":
        # The last attempt may have reached IBM before the connection dropped.
        found = patiently(
            "looking for the job by its tag",
            lambda: service.jobs(
                job_tags=[manifest["tag"]], backend_name=manifest["backend"]
            ),
            log,
        )
        if found:
            manifest.update(job_id=found[0].job_id(), state="submitted")
            write_json(folder / "manifest.json", manifest)
            log(f"found job {manifest['job_id']} from the last attempt")
            return

    with (folder / "circuits.qpy").open("rb") as f:
        circuits = qpy.load(f)
    pubs = []
    for run in manifest["runs"]:
        for c in circuits[run["first"] : run["first"] + run["n_circuits"]]:
            pubs.append((c, None, run["shots"]))
    manifest["state"] = "submitting"
    write_json(folder / "manifest.json", manifest)

    backend = patiently(
        "fetching the backend", lambda: service.backend(manifest["backend"]), log
    )
    sampler = SamplerV2(
        mode=backend, options={"environment": {"job_tags": [manifest["tag"]]}}
    )
    job = patiently("submitting", lambda: sampler.run(pubs), log)
    manifest.update(job_id=job.job_id(), submitted=now(), state="submitted")
    write_json(folder / "manifest.json", manifest)
    log(f"submitted job {job.job_id()} to {manifest['backend']}")
    try:
        manifest["usage_estimation"] = job.usage_estimation
        write_json(folder / "manifest.json", manifest)
        log(f"IBM estimates {job.usage_estimation}")
    except Exception:
        pass


def wait(folder: Path, manifest: dict, service, log: Log, every: float) -> str:
    """Ask for the job's status every ``every`` seconds until it is final.
    A failed check is logged and tried again; it never ends the wait."""
    last = manifest["status"]
    delay = every
    while True:
        try:
            status = str(service.job(manifest["job_id"]).status())
            delay = every
        except Exception as exc:
            if not transient(exc):
                raise
            log(
                f"status check failed ({type(exc).__name__}: {exc}); "
                f"trying again in {delay:.0f} s"
            )
            time.sleep(delay)
            delay = min(600, delay * 2)
            continue
        if status != last:
            manifest["status"] = status
            manifest["history"].append({"time": now(), "status": status})
            write_json(folder / "manifest.json", manifest)
            log(f"job {manifest['job_id']}: {status}")
            last = status
        if status in FINAL:
            return status
        time.sleep(every)


def fetch(folder: Path, manifest: dict, service, log: Log, status: str) -> None:
    """Save the raw result (or the error), IBM's metrics and the run's calibration."""
    from qiskit_ibm_runtime import RuntimeEncoder

    job = patiently("fetching the job", lambda: service.job(manifest["job_id"]), log)
    for name, get in [
        ("metrics", job.metrics),
        ("calibration_run", lambda: job.properties().to_dict()),
    ]:
        try:
            write_json(folder / "job" / f"{name}.json", patiently(name, get, log))
        except Exception as exc:
            log(f"could not save {name} ({type(exc).__name__}: {exc})")
    if status != "DONE":
        error = {"status": status, "error": job.error_message(), "time": now()}
        write_json(folder / "job" / "error.json", error)
        manifest.update(state="failed", finished=now(), error=error["error"])
        write_json(folder / "manifest.json", manifest)
        log(f"job {manifest['job_id']} ended {status}: {error['error']}")
        return
    result = patiently("downloading the result", job.result, log)
    path = folder / "job" / "result.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, cls=RuntimeEncoder))
    os.replace(tmp, path)
    manifest.update(state="finished", finished=now())
    write_json(folder / "manifest.json", manifest)
    log(f"saved the result of job {manifest['job_id']}")


def run(args) -> None:
    package_path = Path(args.package).resolve()
    package, entries = load_package(package_path)
    folder = Path(args.out or package["name"]).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    log = Log(folder / "log.txt")

    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint(entries):
            raise SystemExit(
                f"{folder} holds a different package (the images or settings changed). "
                "Give a new name, or --out a new folder."
            )
        log(f"resuming: state {manifest['state']}, job {manifest['job_id']}")
    else:
        plans_on(devices.graph(package.get("device", "phoenix")), entries)  # fail early
        manifest = prepare(folder, package_path, package, entries, log)
    follow(folder, manifest, log, args.poll, package)


def resume(args) -> None:
    folder = Path(args.folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    log = Log(folder / "log.txt")
    log(f"resuming: state {manifest['state']}, job {manifest['job_id']}")
    package_file = folder / "package.toml"
    package = tomllib.loads(package_file.read_text()) if package_file.exists() else None
    follow(folder, manifest, log, args.poll, package)


def follow(
    folder: Path, manifest: dict, log: Log, every: float, package: dict | None
) -> None:
    """Take a prepared package as far as it goes: submit, wait, fetch, decode.
    ``package`` may name the IBM instance; the token comes from the environment."""
    if manifest["state"] in ("prepared", "submitting", "submitted"):
        service = patiently("connecting to IBM", connect(package), log)
        if manifest["state"] != "submitted":
            submit(folder, manifest, service, log)
        try:
            status = wait(folder, manifest, service, log, every)
        except KeyboardInterrupt:
            log("stopped waiting; the job keeps running. Run the same command to resume.")
            raise SystemExit(130) from None
        except Exception as exc:  # e.g. an expired key: the job is fine, so is the folder
            log(
                f"stopped waiting ({type(exc).__name__}: {exc}). "
                "Fix the cause and run the same command to resume."
            )
            raise SystemExit(1) from exc
        fetch(folder, manifest, service, log, status)
    if manifest["state"] == "finished":
        decode_folder(folder, log=log)
    elif manifest["state"] == "failed":
        raise SystemExit(f"job {manifest['job_id']} failed: {manifest.get('error')}")


# ----------------------------------------------------------------------
# decode: saved raw result -> images and metrics
# ----------------------------------------------------------------------
def _read_text(path: Path) -> str:
    """``path``, or its gzipped copy ``path.gz`` if only that exists."""
    if path.exists():
        return path.read_text()
    with gzip.open(path.with_name(path.name + ".gz"), "rt") as f:
        return f.read()


def save_png(path: Path, b64: str, width: int, height: int, scale: int) -> None:
    from PIL import Image

    rgba = np.frombuffer(base64.b64decode(b64), np.uint8).reshape(height, width, 4)
    im = Image.fromarray(rgba, "RGBA")
    im.save(path)
    if scale > 1:
        im.resize((width * scale, height * scale), Image.NEAREST).save(
            path.with_name(f"{path.stem}_x{scale}.png")
        )


def save_images(where: Path, out: dict, scale: int) -> None:
    w, h = out["width"], out["height"]
    for kind, images in [("raw", out), ("drr", out["drr"])]:
        if "image" in images:
            save_png(where / f"{kind}.png", images["image"], w, h, scale)
        for frame, angle in zip(
            images.get("frames", []), out.get("alphas_degrees", []), strict=True
        ):
            save_png(where / f"{kind}_{angle:05.1f}deg.png", frame, w, h, scale)


def decode_folder(
    folder: str | Path,
    out: str | Path | None = None,
    scale: int = 8,
    log: Log | None = None,
) -> list[dict]:
    """Decode the saved raw result of a finished package folder, offline.

    Writes ``results/<label>/`` (``result.json`` and PNGs, also ``scale`` times
    larger), ``summary.json`` and ``summary.csv`` into ``out`` (default: ``folder``
    itself) and returns the summary rows. Progress goes to ``log`` (default: print).
    """
    from qiskit_ibm_runtime import RuntimeDecoder

    folder = Path(folder)
    out = folder if out is None else Path(out)
    log = Log(None) if log is None else log
    manifest = json.loads((folder / "manifest.json").read_text())
    result = json.loads(_read_text(folder / "job" / "result.json"), cls=RuntimeDecoder)
    graph = graph_from_coupling(manifest["edges"], manifest["n_qubits"])
    rows = []
    for run in manifest["runs"]:
        request = json.loads((folder / run["request_file"]).read_text())
        settings = {k: v for k, v in request.items() if k != "images"}
        row = {
            "label": run["label"],
            "images": run["images"],
            **settings,
            "shots": run["shots"],
            "n_circuits": run["n_circuits"],
        }
        where = out / "results" / run["label"]
        where.mkdir(parents=True, exist_ok=True)
        try:
            plan = build_plan(request, graph)
            counts = [
                counts_of(result, run["first"] + i) for i in range(run["n_circuits"])
            ]
            decoded = decode_hardware(plan, counts)
        except Exception as exc:  # one bad run never hides the others
            row["error"] = f"{type(exc).__name__}: {exc}"
            write_json(where / "error.json", row)
            log(f"{run['label']}: could not decode ({row['error']})")
            rows.append(row)
            continue
        decoded["run"] = {
            "kind": "hardware",
            "device": manifest["backend"],
            "job_id": manifest["job_id"],
            "shots": run["shots"],
            "first_circuit": run["first"],
            "n_circuits": run["n_circuits"],
        }
        decoded["settings"] = settings
        write_json(where / "result.json", decoded)
        save_images(where, decoded, scale)
        row.update(
            n_pixels=decoded["n_pixels"],
            slots_used=decoded["slots_used"],
            capacity=decoded["capacity"],
            drr_method=decoded["drr"]["method"],
        )
        row.update({f"raw.{k}": v for k, v in decoded["metrics"].items()})
        row.update({f"drr.{k}": v for k, v in decoded["drr"]["metrics"].items()})
        rows.append(row)
        log(
            f"{run['label']}: decoded, raw {decoded['metrics']}, "
            f"best DRR {decoded['drr']['method']} {decoded['drr']['metrics']}"
        )

    write_json(
        out / "summary.json",
        {"job_id": manifest["job_id"], "backend": manifest["backend"], "runs": rows},
    )
    columns = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, columns)
    writer.writeheader()
    writer.writerows(rows)
    (out / "summary.csv").write_text(buffer.getvalue())
    log(f"wrote {out / 'summary.csv'}")
    return rows


def decode(args) -> None:
    folder = Path(args.folder).resolve()
    out = Path(args.out).resolve() if args.out else None
    decode_folder(folder, out, args.scale)


def save_snapshot(args) -> None:
    service = make_service()
    snap = devices.snapshot(service.backend(args.backend))
    write_json(Path(args.file), snap)
    print(
        f"{snap['name']}: {snap['num_qubits']} qubits, calibrated {snap['calibrated_at']}"
    )


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(
        prog="python -m tessa.hardware", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="build every plan offline; send nothing")
    p.add_argument("package")
    p.set_defaults(fn=check)
    p = sub.add_parser("run", help="submit the package once, then wait for it")
    p.add_argument("package")
    p.add_argument("--out", help="output folder (default ./<package name>)")
    p.add_argument("--poll", type=float, default=60, help="seconds between checks")
    p.set_defaults(fn=run)
    p = sub.add_parser("resume", help="keep waiting on a submitted package folder")
    p.add_argument("folder")
    p.add_argument("--poll", type=float, default=60, help="seconds between checks")
    p.set_defaults(fn=resume)
    p = sub.add_parser("decode", help="decode a finished package folder again, offline")
    p.add_argument("folder")
    p.add_argument("--out", help="write results/ and summary.* here (default: folder)")
    p.add_argument(
        "--scale", type=int, default=8, help="also save PNGs this many times larger"
    )
    p.set_defaults(fn=decode)
    p = sub.add_parser("snapshot", help="save an IBM chip's calibration snapshot")
    p.add_argument("backend", help="IBM backend name, e.g. ibm_phoenix")
    p.add_argument("file", help="JSON file to write")
    p.set_defaults(fn=save_snapshot)
    args = parser.parse_args(argv)
    args.fn(args)
