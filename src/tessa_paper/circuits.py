"""Table ``tab:prior``, Tessa rows: qubits, values, two-qubit gates and two-qubit depth
of the QCAT circuits run on ``ibm_phoenix``.

Computed from the transpiled circuits as submitted,
``results/hardware/qcat/circuits/<dev>_<mode>.qpy.gz`` (single: one Z circuit; pair:
a Z and an X circuit), and from the logical (untranspiled) circuits rebuilt with
:func:`tessa.encode` from the input images in ``data/images/qcat/``.

Definitions used here:

* **2Q depth**: the depth of the circuit counting only two-qubit gates (CZ after
  transpilation, CX before), i.e. the number of two-qubit-gate layers:
  ``QuantumCircuit.depth(filter_function=lambda i: i.operation.num_qubits == 2)``.
* **total depth**: ``QuantumCircuit.depth()`` of the whole circuit (all gates,
  including the final measurement layer and the barrier before it).
* **2Q depth lower bound**: ``2**m_max`` CNOT layers per multiplexer field (App.
  ``sec:depth``): every data qubit of degree ``m`` is the target of ``2**m``
  sequential CNOTs, and the pair encoding applies its RZ multiplexers after its RY
  multiplexers.

The paper prints a 2Q depth of 19 (Z) and 39 (Z,X); the stored circuits give 23 and
43 (see :data:`KNOWN_DISCREPANCIES`).
"""

from __future__ import annotations

import gzip
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from qiskit import QuantumCircuit, qpy

from tessa import encode, load_encoding

from ._check import compare_with_paper
from .paths import DATA, RESULTS

__all__ = [
    "KNOWN_DISCREPANCIES",
    "PAPER_VALUES",
    "check_against_paper",
    "circuit_table",
    "logical_circuit",
    "prior_table",
    "two_qubit_depth",
]

QCAT = RESULTS / "hardware" / "qcat"
RUNS = ("phoenix/single", "phoenix/pair", "fez/single", "fez/pair")
IMAGES = {
    "phoenix": ("qcat28.png", "qcat28_color.png"),
    "fez": ("qcat.png", "qcat_color.png"),
}

# Table 2 (tab:prior), Tessa rows.
PAPER_VALUES: dict[str, str] = {
    "tab:prior Tessa Z qubits": "120",
    "tab:prior Tessa Z values": "784",
    "tab:prior Tessa Z 2Q gates": "728",
    "tab:prior Tessa Z 2Q depth": "19",
    "tab:prior Tessa Z,X qubits": "120",
    "tab:prior Tessa Z,X values": "1568",
    "tab:prior Tessa Z,X 2Q gates": "1456",
    "tab:prior Tessa Z,X 2Q depth": "39",
}

KNOWN_DISCREPANCIES: dict[str, str] = {
    "tab:prior Tessa Z 2Q depth": (
        "the submitted Phoenix Z circuit has 23 CZ layers (the logical circuit 23 CX "
        "layers); 19 matches no Phoenix quantity and equals the total logical depth "
        "of the Fez Z circuit."
    ),
    "tab:prior Tessa Z,X 2Q depth": (
        "both submitted Phoenix Z,X circuits (Z and X readout) have 43 CZ layers (the "
        "logical circuit 43 CX layers); no stored circuit has 39."
    ),
}


def two_qubit_depth(circuit: QuantumCircuit) -> int:
    """Number of two-qubit-gate layers: the depth counting only two-qubit gates."""
    return circuit.depth(filter_function=lambda ins: ins.operation.num_qubits == 2)


def _two_qubit_count(circuit: QuantumCircuit) -> int:
    return sum(1 for ins in circuit.data if ins.operation.num_qubits == 2)


def load_circuits(run: str) -> list[QuantumCircuit]:
    """The transpiled circuits of a QCAT run as submitted (``results/hardware/qcat/
    circuits/<dev>_<mode>.qpy.gz``): ``[z]`` (single) or ``[z, x]`` (pair)."""
    dev, mode = run.split("/")
    with gzip.open(QCAT / "circuits" / f"{dev}_{mode}.qpy.gz", "rb") as f:
        return list(qpy.load(f))


def logical_circuit(run: str) -> QuantumCircuit:
    """The logical encoding circuit of a QCAT run, rebuilt with :func:`tessa.encode`
    from ``data/images/qcat/`` and the stored graph (``encodings.json``) with the
    run's settings (single: theta = pi * gray; pair: HSV value in theta with clamp
    0.2 pi, hue in phi). The rebuilt encoding is checked to equal the stored one."""
    stored = load_encoding(QCAT / "encodings.json")[run]
    dev, mode = run.split("/")
    gray_file, colour_file = IMAGES[dev]
    if mode == "single":
        img = np.asarray(Image.open(DATA / "images" / "qcat" / gray_file).convert("L"))
        circuit, enc = encode(stored.graph, img / 255.0, value_range=(0, 1))
    else:
        rgb = np.asarray(
            Image.open(DATA / "images" / "qcat" / colour_file).convert("RGB")
        )
        hsv = rgb_to_hsv(rgb / 255.0)
        circuit, enc = encode(
            stored.graph,
            hsv[..., 2],
            hsv[..., 0],
            mode="pair",
            clamp=0.2 * np.pi,
            clamp_b=0.0,
            value_range=(0, 1),
            value_range_b=(0, 1),
        )
    if enc.to_dict() != stored.to_dict():
        raise RuntimeError(f"re-encoding {run} does not give the stored encoding")
    return circuit


def circuit_table() -> pd.DataFrame:
    """Gate counts and depths of every stored QCAT circuit (Phoenix and Fez).

    One row per run and readout basis: device, mode, basis, qubits, values, CZ count,
    2Q depth (CZ layers), total depth of the transpiled circuit; CX count, 2Q depth
    and total depth of the logical circuit; and the 2Q-depth lower bound
    ``n_fields * 2**m_max``. Reads ``results/hardware/qcat/circuits/*.qpy.gz``,
    ``encodings.json`` and ``data/images/qcat/``.
    """
    encodings = load_encoding(QCAT / "encodings.json")
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        dev, mode = run.split("/")
        enc = encodings[run]
        logical = logical_circuit(run)
        m_max = max(len(g.address_qubits) for g in enc.graph.groups)
        for basis, circuit in zip(("z", "x"), load_circuits(run), strict=False):
            rows.append(
                {
                    "device": dev,
                    "mode": mode,
                    "readout": "Z" if mode == "single" else "Z,X",
                    "basis": basis,
                    "qubits": circuit.num_qubits,
                    "values": int(np.prod(enc.shape)) * enc.n_fields,
                    "CZ": circuit.count_ops().get("cz", 0),
                    "2Q depth": two_qubit_depth(circuit),
                    "total depth": circuit.depth(),
                    "logical CX": _two_qubit_count(logical),
                    "logical 2Q depth": two_qubit_depth(logical),
                    "logical total depth": logical.depth(),
                    "2Q depth lower bound": enc.n_fields * 2**m_max,
                }
            )
    return pd.DataFrame(rows)


def prior_table() -> pd.DataFrame:
    """Table ``tab:prior``, Tessa rows, from the stored Phoenix circuits: work,
    device, qubits, values, 2Q gates (per circuit), 2Q depth (largest over the
    readout bases), next to the values printed in the paper."""
    table = circuit_table()
    rows = []
    for readout, (gates, depth) in {"Z": ("728", "19"), "Z,X": ("1456", "39")}.items():
        sel = table[(table.device == "phoenix") & (table.readout == readout)]
        rows.append(
            {
                "work": f"Tessa, {readout}",
                "device": "Phoenix",
                "qubits": int(sel.qubits.max()),
                "values": int(sel["values"].max()),
                "2Q gates": int(sel.CZ.max()),
                "2Q depth": int(sel["2Q depth"].max()),
                "2Q gates (paper)": int(gates),
                "2Q depth (paper)": int(depth),
            }
        )
    return pd.DataFrame(rows).set_index("work")


def check_against_paper() -> pd.DataFrame:
    """``[item, paper, reproduced, match]`` for the Tessa rows of Table prior."""
    table = prior_table()
    reproduced = {}
    for readout in ("Z", "Z,X"):
        row = table.loc[f"Tessa, {readout}"]
        reproduced[f"tab:prior Tessa {readout} qubits"] = row["qubits"]
        reproduced[f"tab:prior Tessa {readout} values"] = row["values"]
        reproduced[f"tab:prior Tessa {readout} 2Q gates"] = row["2Q gates"]
        reproduced[f"tab:prior Tessa {readout} 2Q depth"] = row["2Q depth"]
    return compare_with_paper(PAPER_VALUES, reproduced)
