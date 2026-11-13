"""JSON round trip of encodings: what decode_from_counts reads must come back exactly."""

from __future__ import annotations

import json

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from tessa import (
    Encoding,
    Lattice,
    TessaGraph,
    decode_from_counts,
    encode,
    graph_from_coupling,
    load_encoding,
    measurement_circuits,
    save_encoding,
    square_lattice_edges,
)
from tessa.serialize import graph_from_dict, graph_to_dict

MODES = ["single", "pair", "pair_full"]


def _graphs():
    edges, n = square_lattice_edges(3, 4)
    return [
        Lattice(3, 4),
        graph_from_coupling(edges, n),
        TessaGraph.from_groups(5, [(0, [1, 2]), (3, [2, 4])]),
        TessaGraph.from_groups(1, [(0, [])]),
    ]


def _same_groups(a, b):
    return [(g.data_qubit, g.address_qubits) for g in a.groups] == [
        (g.data_qubit, g.address_qubits) for g in b.groups
    ]


@pytest.mark.parametrize("graph", _graphs(), ids=["lattice", "grid", "groups", "single"])
def test_a_graph_round_trips(graph):
    back = graph_from_dict(json.loads(json.dumps(graph_to_dict(graph))))
    assert type(back) is type(graph)
    assert back.n_qubits == graph.n_qubits
    assert back.capacity == graph.capacity
    assert _same_groups(back, graph)


def test_a_graph_with_other_groups_is_refused():
    data = graph_to_dict(_graphs()[1])
    data["groups"][0]["addresses"] = data["groups"][0]["addresses"][::-1]
    with pytest.raises(ValueError, match="groups"):
        graph_from_dict(data)


def _fields(mode, n, seed):
    rng = np.random.default_rng(seed)
    a = rng.random(n) * 3 - 1
    return (a,) if mode == "single" else (a, rng.random(n) * 2 + 5)


def _exact_counts(circuit, enc):
    return {
        basis: Statevector(
            qc.remove_final_measurements(inplace=False)
        ).probabilities_dict()
        for basis, qc in measurement_circuits(circuit, enc).items()
    }


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("graph", _graphs()[:3], ids=["lattice", "grid", "groups"])
def test_an_encoding_round_trips_and_decodes_identically(tmp_path, graph, mode):
    values = _fields(mode, graph.capacity - 2, seed=1)
    circuit, enc = encode(
        graph, *values, mode=mode, clamp=0.1 if mode != "single" else 0.0
    )
    path = save_encoding(tmp_path / "enc.json", enc)
    back = load_encoding(path)
    assert isinstance(back, Encoding)
    assert back.to_dict() == enc.to_dict()
    assert (back.mode, back.clamps, back.ranges, back.shape, back.n) == (
        enc.mode,
        enc.clamps,
        enc.ranges,
        enc.shape,
        enc.n,
    )
    assert _same_groups(back.graph, enc.graph)
    counts = _exact_counts(circuit, enc)
    want = np.asarray(decode_from_counts(counts, enc))
    got = np.asarray(decode_from_counts(counts, back))
    assert np.array_equal(got, want)


def test_a_mapping_of_encodings_round_trips(tmp_path):
    graph = Lattice(3, 4)
    encs = {
        "a/single": encode(graph, np.arange(12.0).reshape(3, 4))[1],
        "b/pair": encode(graph, np.ones(5), np.zeros(5), clamp=0.2 * np.pi, clamp_b=0)[1],
    }
    back = load_encoding(save_encoding(tmp_path / "many.json", encs))
    assert list(back) == list(encs)
    for k, enc in encs.items():
        assert back[k].to_dict() == enc.to_dict()


def test_a_foreign_file_is_refused(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(ValueError, match="not a Tessa encoding"):
        load_encoding(path)
