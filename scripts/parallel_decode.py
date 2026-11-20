"""Noiseless per-cell decode on a process pool: the workflow for a live loop.

Create one pool at startup, pass it to ``decode(..., executor=pool)`` every round,
shut it down at the end. One round: ``ibm_phoenix``, pair mode, 1568 values.

Usage, from the repository checkout::

    uv run python scripts/parallel_decode.py                 # one round
    uv run python scripts/parallel_decode.py --workers 4
    uv run python scripts/parallel_decode.py --benchmark 5   # time vs sequential
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from tessa import aer_sampler, decode, encode
from tessa.hardware import devices


def performance_cores() -> int:
    """Fast physical cores on Apple Silicon; logical CPUs elsewhere."""
    try:
        cmd = ["sysctl", "-n", "hw.perflevel0.physicalcpu"]
        return int(subprocess.check_output(cmd, stderr=subprocess.DEVNULL))
    except Exception:  # Linux/other: logical CPUs; pass --workers if 2 per core
        return os.cpu_count() or 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, performance_cores() - 1),
        help="pool size (default: performance cores - 1, leaving one for the app)",
    )
    p.add_argument("--shots", type=int, default=8192)
    p.add_argument(
        "--benchmark",
        type=int,
        metavar="N",
        help="also time N rounds sequential vs pool and print s/round",
    )
    args = p.parse_args()

    # 1. Startup: once
    graph = devices.graph("phoenix")
    sampler = aer_sampler(graph)  # no seed: chunks draw independent shots
    pool = ProcessPoolExecutor(args.workers)

    # 2. Every round (one per bar in a live loop)
    def quantum_round(x, y, executor=pool):
        circuit, enc = encode(graph, x, y)
        return decode(
            circuit, enc, sampler, shots=args.shots, halo=None, executor=executor
        )

    try:
        rng = np.random.default_rng(0)
        x, y = rng.random(784), rng.random(784)
        a, b = quantum_round(x, y)
        print(
            f"{args.workers} workers; |err| a = {np.abs(a - x).mean():.4f}  "
            f"b = {np.abs(b - y).mean():.4f}"
        )
        if args.benchmark:
            for label, ex in [("sequential", None), (f"pool x{args.workers}", pool)]:
                t = time.perf_counter()
                for _ in range(args.benchmark):
                    quantum_round(x, y, ex)
                dt = (time.perf_counter() - t) / args.benchmark
                print(f"{label:12s} {dt:.2f} s/round")
    finally:
        # 3. Shutdown
        pool.shutdown()


if __name__ == "__main__":  # required for processes on macOS/Windows
    main()
