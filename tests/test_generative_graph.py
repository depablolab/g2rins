# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import copy
import json

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

import g2rins

# graph_validation.json holds nx.adjacency_data of get_generative_graph(include_bond_connectors=False)
# for every tests/smi.json g2rins string; regenerate it whenever the generative-graph
# schema deliberately changes (see dev-clean 2026-07 regeneration).


def node_match(node1_attrs, node2_attrs):
    return node1_attrs == node2_attrs


# Define a custom edge matcher
def edge_match(edge1_attrs, edge2_attrs):
    return edge1_attrs == edge2_attrs


def test_generative_graph_generation(graph_validation_dict):
    for g2rins_string in graph_validation_dict:
        print(g2rins_string)
        graph_creator = g2rins.G2rins.make(g2rins_string).get_graph_creator()
        generative_graph = graph_creator.get_generative_graph(include_bond_connectors=False)
        assert nx.is_isomorphic(generative_graph, graph_validation_dict[g2rins_string], node_match=node_match, edge_match=edge_match)

        dot_string_A = graph_creator.get_dot_string(include_bond_connectors=True)
        dot_string_B = graph_creator.get_dot_string(include_bond_connectors=False, node_prefix="bc-")
        dot_string_A = dot_string_A[: dot_string_A.rfind("}")]
        dot_string_B = dot_string_B[len("digraph{") :]
        dot_string = dot_string_A + dot_string_B
        assert len(dot_string) > 0


def test_generative_graph_json_data_format_block():
    smi = "{[] [<]CC([>])c1ccccc1; [>][H]; [<][H] []}|gauss(1000, 45)|"
    generative_graph = g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    data = g2rins.generative_graph_json_data(generative_graph)

    assert data["format"]["version"] == 1
    declared = set(data["format"]["derived_node_fields"])
    stored_attrs = {key for _node, attrs in generative_graph.nodes(data=True) for key in attrs}
    injected = {key for node_dict in data["graph"]["nodes"] for key in node_dict} - stored_attrs - {"id"}
    # The declared derived fields are exactly what the writer injects, and they
    # are annotations only -- never stored on the graph itself.
    assert injected == declared
    assert not declared & stored_attrs

    labels = g2rins.derive_unit_labels(generative_graph)
    for node_dict in data["graph"]["nodes"]:
        assert node_dict["unit_id"] == labels.unit_id[node_dict["id"]]
        assert node_dict.get("bond_id") == labels.bond_id.get(node_dict["id"])


@pytest.mark.parametrize(("text", "atomic_num"), [("c1cc[se]c1", 34), ("c1cc[as]cc1", 33)])
def test_aromatic_two_letter_elements_keep_their_atomic_number(text, atomic_num):
    creator = g2rins.G2rins.make(text).get_graph_creator().get_ensemble_creator()
    atoms = [data["atomic_num"] for _, data in creator.generative_graph.nodes(data=True)]
    assert atoms.count(atomic_num) == 1
    assert all(number > 0 for number in atoms)
    molecule = creator.create_ensemble(1, output_format="mol", seed=0)[0]
    Chem.SanitizeMol(molecule)
    assert sum(atom.GetAtomicNum() == atomic_num for atom in molecule.GetAtoms()) == 1


def test_export_payload_is_detached_from_nested_graph_attributes():
    graph = g2rins.G2rins.make("{[] [<]CCO[>]; CO[>]; [<][H] []}|poisson(300)|").get_graph_creator().get_generative_graph()
    u, v, key = next(iter(graph.edges(keys=True)))
    graph.edges[u, v, key]["metadata"] = {"values": [1, 2]}
    before = copy.deepcopy(graph)
    payload = g2rins.generative_graph_json_data(graph)["graph"]
    payload["graph"]["unit_g2rins"].clear()
    payload["graph"].pop("g2rins_string")
    payload["nodes"][0]["unit_molar_amounts"][0] = 99
    next(edge for edge in payload["edges"] if "metadata" in edge)["metadata"]["values"].append(3)
    assert nx.utils.graphs_equal(graph, before)


@pytest.mark.parametrize("location", ["node", "edge", "graph"])
def test_export_normalizes_nested_numpy_data_without_mutation(location):
    graph = g2rins.G2rins.make("CC").get_graph_creator().get_generative_graph()
    integer_types = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
    float_types = (np.float16, np.float32, np.float64, np.longdouble)
    metadata = {
        np.str_("integers"): [dtype(7) for dtype in integer_types],
        "floats": tuple(dtype(0.25) for dtype in float_types),
        "booleans": [np.bool_(False), np.bool_(True)],
        "text": np.str_("carbon"),
        "zero_dimensional": np.array(9, dtype=np.int32),
        "matrix": np.array([[1, 2], [3, 4]], dtype=np.uint16),
        "objects": np.array([{np.int64(7): [np.float32(0.5), np.bool_(True)]}], dtype=object),
        "structured": np.array([(4, 0.5)], dtype=[("count", "i4"), ("weight", "f4")])[0],
        "structured_array": np.array([(4, (0.5,)), (5, (0.25,))], dtype=[("count", "i4"), ("stats", [("weight", "f4")])]),
        "structured_subarray": np.array([([1, 2],)], dtype=[("counts", "i4", (2,))]),
    }
    if location == "node":
        attrs = graph.nodes[next(iter(graph))]
    elif location == "edge":
        attrs = graph.edges[next(iter(graph.edges(keys=True)))]
    else:
        attrs = graph.graph
    attrs["metadata"] = metadata

    payload = g2rins.generative_graph_json_data(graph)
    # Serializing the entire payload detects NumPy scalars left inside arrays
    # of objects, dictionaries, keys, or structured scalar fields.
    restored = nx.node_link_graph(json.loads(json.dumps(payload, allow_nan=False))["graph"], edges="edges")
    if location == "node":
        exported = restored.nodes[next(iter(graph))]["metadata"]
        detached = payload["graph"]["nodes"][0]["metadata"]
    elif location == "edge":
        exported = restored.edges[next(iter(graph.edges(keys=True)))]["metadata"]
        detached = payload["graph"]["edges"][0]["metadata"]
    else:
        exported = restored.graph["metadata"]
        detached = payload["graph"]["graph"]["metadata"]
    assert exported == {
        "integers": [7] * len(integer_types),
        "floats": [0.25] * len(float_types),
        "booleans": [False, True],
        "text": "carbon",
        "zero_dimensional": 9,
        "matrix": [[1, 2], [3, 4]],
        "objects": [{"7": [0.5, True]}],
        "structured": [4, 0.5],
        "structured_array": [[4, [0.5]], [5, [0.25]]],
        "structured_subarray": [[[1, 2]]],
    }
    detached["objects"][0][7][0] = 99
    detached["matrix"][0][0] = 99
    assert metadata["objects"][0][np.int64(7)][0] == np.float32(0.5)
    np.testing.assert_array_equal(metadata["matrix"], [[1, 2], [3, 4]])
    assert type(metadata["integers"][0]) is np.int8
    assert type(metadata["objects"][0][np.int64(7)][1]) is np.bool_


def test_export_derives_labels_before_converting_tuple_node_ids():
    graph = g2rins.G2rins.make("CC").get_graph_creator().get_generative_graph()
    graph = nx.relabel_nodes(graph, {node: ("atom", np.int64(index)) for index, node in enumerate(graph)})
    expected = g2rins.derive_unit_labels(graph)
    payload = json.loads(json.dumps(g2rins.generative_graph_json_data(graph)))
    restored = nx.node_link_graph(payload["graph"], edges="edges")
    assert set(restored) == set(graph)
    assert {node: data["unit_id"] for node, data in restored.nodes(data=True)} == expected.unit_id


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.complex64(1 + 2j), id="complex64"),
        pytest.param(np.clongdouble(1 + 2j), id="clongdouble"),
        pytest.param(np.bytes_(b"carbon"), id="bytes"),
        # .item() resolves a datetime by precision: day gives date, second gives
        # datetime, but nanosecond gives int and NaT gives None, so rejection has
        # to key on dtype or high-precision values pass through as bare numbers.
        pytest.param(np.datetime64("2026-09-09"), id="datetime64-day"),
        pytest.param(np.datetime64("2026-09-09", "s"), id="datetime64-second"),
        pytest.param(np.datetime64("2026-09-09", "ns"), id="datetime64-nanosecond"),
        pytest.param(np.datetime64("NaT", "ns"), id="datetime64-nat"),
        pytest.param(np.timedelta64(5, "ns"), id="timedelta64"),
        pytest.param(np.array(["2026-09-09"], dtype="datetime64[ns]"), id="datetime64-array"),
    ],
)
def test_export_reports_values_without_json_equivalents(value):
    graph = g2rins.G2rins.make("C").get_graph_creator().get_generative_graph()
    graph.graph["metadata"] = {"unsupported": value}
    with pytest.raises(TypeError, match=r"\['graph'\]\['metadata'\]\['unsupported'\]"):
        g2rins.generative_graph_json_data(graph)


@pytest.mark.parametrize(
    "dtype",
    [
        [("when", "datetime64[ns]")],
        [("when", "timedelta64[ns]")],
        [("event", [("when", "datetime64[ns]")])],
        [("when", "datetime64[ns]", (2,))],
        [("events", [("when", "datetime64[ns]")], (2,))],
    ],
    ids=["timestamp", "duration", "nested", "subarray", "nested-subarray"],
)
@pytest.mark.parametrize("container", ["scalar", "array", "zero-dimensional", "empty-array"])
def test_export_rejects_datetime_fields_in_structured_records(dtype, container):
    records = np.zeros(1, dtype=dtype)
    if container == "scalar":
        value = records[0]
    elif container == "zero-dimensional":
        value = records.reshape(())
    elif container == "empty-array":
        value = records[:0]
    else:
        value = records
    graph = g2rins.G2rins.make("C").get_graph_creator().get_generative_graph()
    graph.graph["metadata"] = {"unsupported": value}
    with pytest.raises(TypeError, match=r"Unsupported JSON value.*\['graph'\]\['metadata'\]\['unsupported'\].*\['when'\]"):
        g2rins.generative_graph_json_data(graph)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), np.float32("nan"), np.float64("inf"), np.longdouble("-inf")])
@pytest.mark.parametrize("location", ["node", "edge", "graph"])
def test_export_rejects_nested_non_finite_values(value, location):
    graph = g2rins.G2rins.make("CC").get_graph_creator().get_generative_graph()
    if location == "node":
        attrs = graph.nodes[next(iter(graph))]
    elif location == "edge":
        attrs = graph.edges[next(iter(graph.edges(keys=True)))]
    else:
        attrs = graph.graph
    attrs["metadata"] = {"values": np.array([value])}
    with pytest.raises(ValueError, match=r"Non-finite JSON value.*\['metadata'\]\['values'\]\[0\]"):
        g2rins.generative_graph_json_data(graph)


def test_connector_graph_exports_unknown_charges_as_null():
    graph = g2rins.G2rins.make("{[] [<]CCN([>])[>]; [<][H]; O[>], [<][H] []}|poisson(200)|").get_graph_creator().get_generative_graph(include_bond_connectors=True)
    descriptors = {node for node, data in graph.nodes(data=True) if data["atomic_num"] < 0}
    assert descriptors
    assert all(np.isnan(graph.nodes[node]["charge"]) for node in descriptors)
    payload = json.loads(json.dumps(g2rins.generative_graph_json_data(graph), allow_nan=False))
    exported = {data["id"]: data for data in payload["graph"]["nodes"]}
    assert all(exported[node]["charge"] is None for node in descriptors)
    assert all(exported[node]["atomic_num"] == graph.nodes[node]["atomic_num"] for node in descriptors)
    assert all(np.isnan(graph.nodes[node]["charge"]) for node in descriptors)


@pytest.mark.parametrize("key", [1, 1.5, True, False, None, np.int64(7), np.float32(0.5), np.bool_(True)])
def test_export_rejects_keys_that_collide_when_encoded(key):
    native_key = key.item() if isinstance(key, np.generic) else key
    encoded_key = json.dumps(native_key)
    graph = g2rins.G2rins.make("C").get_graph_creator().get_generative_graph()
    graph.graph["metadata"] = {key: "first", encoded_key: "second"}
    with pytest.raises(ValueError, match=r"Duplicate JSON key.*\['metadata'\]"):
        g2rins.generative_graph_json_data(graph)
    assert len(graph.graph["metadata"]) == 2


@pytest.mark.parametrize("key", [float("nan"), float("inf"), np.float32("-inf")])
def test_export_rejects_non_finite_keys(key):
    graph = g2rins.G2rins.make("C").get_graph_creator().get_generative_graph()
    graph.graph["metadata"] = {key: "value"}
    with pytest.raises(ValueError, match=r"Non-finite JSON value.*\['metadata'\]"):
        g2rins.generative_graph_json_data(graph)
