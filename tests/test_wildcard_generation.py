# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import copy
import json
import pickle
import warnings

import networkx as nx
import numpy as np
import pytest

import g2rins
from g2rins.exception import (
    G2RINSError,
    G2RINSWarning,
    GenerationError,
    IncompatibleGenerativeGraphSchema,
    UnsupportedWildcardGeneration,
)

PEI = "{[] [<]CCN([>])[>]; [<][H]; O[>], [<][H] []}|poisson(200)|"
LINEAR = "{[] [<]CCO[>]; CO[>]; [<][H] []}|poisson(300)|"
WILDCARD_INPUTS = [
    pytest.param("{[] [<]CC(*)O[>]; CO[>]; [<][H] []}|poisson(300.0)|", id="A1-pendant"),
    pytest.param("{[] [<]CCO[>]; CO[>]; [<]* []}|poisson(300.0)|", id="A2-terminator"),
    pytest.param("{[] [<]C*C[>]; CO[>]; [<][H] []}|poisson(300.0)|", id="A4-backbone"),
    pytest.param("{[] [<]CC(*)N([>])[>]; [<][H]; O[>], [<][H] []}|poisson(200.0)|", id="mixed-wildcard-and-placeholders"),
]


@pytest.mark.parametrize("text", WILDCARD_INPUTS)
@pytest.mark.parametrize("wildcard", ["*", "[*]"], ids=["bare", "bracketed"])
def test_user_wildcards_parse_and_export_but_cannot_generate(text, wildcard, tmp_path, monkeypatch):
    text = text.replace("*", wildcard)
    parsed = g2rins.G2rins.make(text)
    assert str(parsed) == text
    assert str(g2rins.G2rins.make(str(parsed))) == str(parsed)
    graph_creator = parsed.get_graph_creator()
    graph = graph_creator.get_generative_graph(include_bond_connectors=False)
    user_wildcards = [node for node, data in graph.nodes(data=True) if data["atomic_num"] == 0 and not data["is_connector_placeholder"]]
    assert len(user_wildcards) == 1
    if "N([>])[>]" in text:
        assert sum(data["is_connector_placeholder"] for _, data in graph.nodes(data=True)) == 2

    path = tmp_path / "wildcard-graph.json"
    graph_creator.write_generative_graph_json(str(path))
    exported = json.loads(path.read_text())
    restored = nx.node_link_graph(exported["graph"], edges="edges")
    assert restored.graph["g2rins_string"] == text
    assert sum(data["atomic_num"] == 0 and not data["is_connector_placeholder"] for _, data in restored.nodes(data=True)) == 1

    def unexpected_setup(*args, **kwargs):
        pytest.fail("Wildcard rejection must precede sampling setup")

    monkeypatch.setattr(g2rins.EnsembleCreator, "_create_static_graph", unexpected_setup)
    unit_id = g2rins.derive_unit_labels(graph).unit_id[user_wildcards[0]]
    for build in (graph_creator.get_ensemble_creator, lambda: g2rins.EnsembleCreator(restored)):
        with pytest.raises(UnsupportedWildcardGeneration) as caught:
            build()
        assert isinstance(caught.value, (GenerationError, G2RINSError))
        assert "Parsing and graph export remain supported" in str(caught.value)
        assert "[<][H]" in str(caught.value)
        assert "report" not in str(caught.value).lower()
        error = caught.value
        assert error.node_id == user_wildcards[0]
        assert error.unit_id == unit_id
        assert error.unit_text == graph.graph["unit_g2rins"][unit_id]
        assert wildcard in error.unit_text
        assert error.args == (error.node_id, error.unit_id, error.unit_text)
        for restored_error in (pickle.loads(pickle.dumps(error)), copy.deepcopy(error)):
            assert vars(restored_error) == vars(error)
            assert str(restored_error) == str(error)
        for field in ("node_id", "unit_id", "unit_text"):
            assert f"{field}={getattr(error, field)!r}" in str(error)


def test_placeholder_identity_survives_export_and_sampled_snapshots():
    graph = g2rins.G2rins.make(PEI).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    payload = json.loads(json.dumps(g2rins.generative_graph_json_data(graph)))
    restored = nx.node_link_graph(payload["graph"], edges="edges")
    assert nx.utils.graphs_equal(graph, _without_derived_labels(restored, payload["format"]["derived_node_fields"]))
    creator = g2rins.EnsembleCreator(restored)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chain, units, _, sequences, _, _ = creator.sample_mol_graph(molecule_info=True, rng=np.random.default_rng(0))
    assert all(not data["is_connector_placeholder"] for _, data in chain.nodes(data=True))
    placeholders = [data for unit in units for _, data in unit.nodes(data=True) if data["atomic_num"] == 0]
    assert placeholders and all(data["is_connector_placeholder"] for data in placeholders)
    # A sequence node keeps the placeholder flag only when it is a retained,
    # unmapped split site; every real atom and every mapped stub clears it.
    for sequence in sequences:
        for unit in sequence:
            for _, data in unit.nodes(data=True):
                if data["is_connector_placeholder"]:
                    assert data["atomic_num"] == 0
                    assert "connection" not in data
                else:
                    assert data["atomic_num"] > 0 or "connection" in data
    stubs = [data for sequence in sequences for unit in sequence for _, data in unit.nodes(data=True) if "connection" in data]
    assert stubs and all(data["atomic_num"] == 0 for data in stubs)
    placeholder_origins = {str(node) for node, data in graph.nodes(data=True) if data["is_connector_placeholder"]}
    assert any(data["origin_idx"] in placeholder_origins for data in stubs)


def test_numpy_placeholder_flags_are_normalized_only_in_creator_copy():
    graph = g2rins.G2rins.make(PEI).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    expected = g2rins.EnsembleCreator(graph).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    for _, data in graph.nodes(data=True):
        data["is_connector_placeholder"] = np.bool_(data["is_connector_placeholder"])
    creator = g2rins.EnsembleCreator(graph)
    assert all(isinstance(data["is_connector_placeholder"], np.bool_) for _, data in graph.nodes(data=True))
    assert all(type(data["is_connector_placeholder"]) is bool for _, data in creator.generative_graph.nodes(data=True))
    assert creator.create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0) == expected
    json.dumps(g2rins.generative_graph_json_data(creator.generative_graph))


def test_numpy_false_still_identifies_a_user_wildcard():
    graph = g2rins.G2rins.make(LINEAR.replace("CCO", "CC(*)O")).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    for _, data in graph.nodes(data=True):
        data["is_connector_placeholder"] = np.bool_(False)
    with pytest.raises(UnsupportedWildcardGeneration):
        g2rins.EnsembleCreator(graph)


def _without_derived_labels(graph, fields):
    stripped = graph.copy()
    for _, data in stripped.nodes(data=True):
        for field in fields:
            data.pop(field, None)
    return stripped


def test_ordinary_legacy_graph_remains_generable():
    graph = g2rins.G2rins.make(LINEAR).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    expected = g2rins.EnsembleCreator(graph).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    for _, data in graph.nodes(data=True):
        del data["is_connector_placeholder"]
    actual = g2rins.EnsembleCreator(graph).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    assert actual == expected
    payload = g2rins.generative_graph_json_data(graph)
    assert payload["format"]["version"] == 2
    assert all("is_connector_placeholder" not in data for data in payload["graph"]["nodes"])


@pytest.mark.parametrize("text", [PEI, LINEAR.replace("CCO", "CC(*)O")], ids=["legacy-placeholder", "legacy-wildcard"])
@pytest.mark.parametrize("consume", [g2rins.EnsembleCreator, g2rins.generative_graph_json_data], ids=["creator", "export"])
def test_legacy_zero_number_nodes_require_explicit_identity(text, consume):
    graph = g2rins.G2rins.make(text).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    for _, data in graph.nodes(data=True):
        data.pop("is_connector_placeholder")
    with pytest.raises(IncompatibleGenerativeGraphSchema) as caught:
        consume(graph)
    assert caught.value.missing_attribute == "is_connector_placeholder"
    assert caught.value.reason == "missing"
    assert graph.nodes[caught.value.node_id]["atomic_num"] == 0
    assert "zero-number nodes" in str(caught.value)
    assert "original G2RINS string" in str(caught.value)
    assert str(pickle.loads(pickle.dumps(caught.value))) == str(caught.value)


@pytest.mark.parametrize("flag", [None, 0, 1, "false", "true"], ids=["null", "zero", "one", "false-string", "true-string"])
@pytest.mark.parametrize("consume", [g2rins.EnsembleCreator, g2rins.generative_graph_json_data], ids=["creator", "export"])
def test_invalid_placeholder_flag_rejected(flag, consume):
    graph = g2rins.G2rins.make(PEI).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    node = next(node for node, data in graph.nodes(data=True) if data["atomic_num"] == 0)
    graph.nodes[node]["is_connector_placeholder"] = flag
    with pytest.raises(IncompatibleGenerativeGraphSchema, match="is_connector_placeholder") as caught:
        consume(graph)
    error = caught.value
    assert error.node_id == node
    assert error.reason == "invalid"
    assert repr(node) in str(error)
    assert "rebuild" not in str(error)
    assert "boolean" in str(error)
    for restored_error in (pickle.loads(pickle.dumps(error)), copy.deepcopy(error)):
        assert restored_error.args == error.args
        assert vars(restored_error) == vars(error)
        assert str(restored_error) == str(error)


@pytest.mark.parametrize("consume", [g2rins.EnsembleCreator, g2rins.generative_graph_json_data], ids=["creator", "export"])
def test_real_atom_cannot_be_marked_as_placeholder(consume):
    graph = g2rins.G2rins.make(LINEAR).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    node = next(iter(graph))
    graph.nodes[node]["is_connector_placeholder"] = True
    with pytest.raises(IncompatibleGenerativeGraphSchema, match="atomic_num is zero") as caught:
        consume(graph)
    assert caught.value.node_id == node
    assert caught.value.reason == "invalid"
    assert "rebuild" not in str(caught.value)


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("consume", [g2rins.EnsembleCreator, g2rins.generative_graph_json_data], ids=["creator", "export"])
def test_missing_atomic_number_reports_node(flag, consume):
    graph = g2rins.G2rins.make(PEI).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    node = next(iter(graph))
    del graph.nodes[node]["atomic_num"]
    graph.nodes[node]["is_connector_placeholder"] = flag
    with pytest.raises(IncompatibleGenerativeGraphSchema) as caught:
        consume(graph)
    assert caught.value.missing_attribute == "atomic_num"
    assert caught.value.reason == "missing"
    assert caught.value.node_id == node


def test_numpy_placeholder_flags_export_as_booleans_without_mutating_graph():
    graph = g2rins.G2rins.make(PEI).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    expected = g2rins.generative_graph_json_data(graph)
    for _, data in graph.nodes(data=True):
        data["is_connector_placeholder"] = np.bool_(data["is_connector_placeholder"])
    actual = json.loads(json.dumps(g2rins.generative_graph_json_data(graph)))
    assert actual == expected
    assert all(isinstance(data["is_connector_placeholder"], np.bool_) for _, data in graph.nodes(data=True))
    assert all(type(data["is_connector_placeholder"]) is bool for data in actual["graph"]["nodes"])


def test_wildcard_diagnostics_do_not_require_parser_provenance():
    graph = g2rins.G2rins.make(LINEAR.replace("CCO", "CC(*)O")).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    graph.graph.clear()
    node = next(node for node, data in graph.nodes(data=True) if data["atomic_num"] == 0)
    with pytest.raises(UnsupportedWildcardGeneration) as caught:
        g2rins.EnsembleCreator(graph)
    assert caught.value.node_id == node
    assert caught.value.unit_id == "R0"
    assert caught.value.unit_text is None


def test_incomplete_wildcard_graph_still_reports_node():
    graph = nx.MultiDiGraph()
    graph.add_node("wildcard", atomic_num=0, is_connector_placeholder=False)
    with pytest.raises(UnsupportedWildcardGeneration) as caught:
        g2rins.EnsembleCreator(graph)
    assert caught.value.args == ("wildcard", None, None)


@pytest.mark.parametrize("graph_type", [nx.MultiDiGraph, nx.DiGraph, nx.Graph])
@pytest.mark.parametrize("weight", [None, 1])
def test_wildcard_diagnostics_survive_unusable_label_inputs(graph_type, weight):
    graph = graph_type()
    graph.add_node("wildcard", atomic_num=0, is_connector_placeholder=False, init_weight=weight)
    with pytest.raises(UnsupportedWildcardGeneration) as caught:
        g2rins.EnsembleCreator(graph)
    assert caught.value.node_id == "wildcard"
    assert caught.value.unit_text is None
    assert caught.value.unit_id == ("I0" if graph_type is nx.MultiDiGraph and weight == 1 else None)


def test_legacy_placeholder_migration_is_explicit_and_preserves_generation():
    graph = g2rins.G2rins.make(PEI).get_graph_creator().get_generative_graph()
    expected = g2rins.EnsembleCreator(graph).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    graph.graph.clear()
    for _, attrs in graph.nodes(data=True):
        del attrs["is_connector_placeholder"]
    before = copy.deepcopy(graph)
    with pytest.raises(IncompatibleGenerativeGraphSchema):
        g2rins.EnsembleCreator(graph)
    with pytest.warns(G2RINSWarning, match="unmarked zero-number nodes"):
        migrated = g2rins.mark_legacy_connector_placeholders(graph)
    assert nx.utils.graphs_equal(graph, before)
    assert all(attrs["is_connector_placeholder"] is True for _, attrs in migrated.nodes(data=True) if attrs["atomic_num"] == 0)
    payload = g2rins.generative_graph_json_data(migrated)
    assert payload["format"]["version"] == 2
    restored = nx.node_link_graph(json.loads(json.dumps(payload))["graph"], edges="edges")
    actual = g2rins.EnsembleCreator(restored).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    for info in expected.units.values():
        info["g2rins"] = ""  # The legacy dataset has no parser provenance.
    assert actual == expected
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        again = g2rins.mark_legacy_connector_placeholders(migrated)
    assert not caught
    assert nx.utils.graphs_equal(again, migrated)


def test_legacy_migration_preserves_known_wildcards_and_nested_attributes():
    graph = g2rins.G2rins.make(PEI.replace("CCN", "CC(*)N")).get_graph_creator().get_generative_graph()
    for _, attrs in graph.nodes(data=True):
        if attrs["is_connector_placeholder"]:
            del attrs["is_connector_placeholder"]
    before = copy.deepcopy(graph)
    with pytest.warns(G2RINSWarning):
        migrated = g2rins.mark_legacy_connector_placeholders(graph)
    with pytest.raises(UnsupportedWildcardGeneration):
        g2rins.EnsembleCreator(migrated)
    migrated.graph["unit_g2rins"].clear()
    migrated.nodes[next(iter(migrated))]["unit_molar_amounts"][0] = 99
    assert nx.utils.graphs_equal(graph, before)


def test_legacy_migration_rejects_invalid_existing_flags():
    graph = nx.MultiDiGraph()
    graph.add_node("bad", atomic_num=0, is_connector_placeholder="true")
    with pytest.raises(IncompatibleGenerativeGraphSchema):
        g2rins.mark_legacy_connector_placeholders(graph)


def test_negative_atomic_numbers_rejected_at_construction():
    graph = g2rins.G2rins.make(PEI).get_graph_creator().get_generative_graph(include_bond_connectors=True)
    with pytest.raises(IncompatibleGenerativeGraphSchema, match="include_bond_connectors=False") as caught:
        g2rins.EnsembleCreator(graph)
    assert caught.value.missing_attribute == "atomic_num"
    assert caught.value.reason == "invalid"
    assert graph.nodes[caught.value.node_id]["atomic_num"] < 0
    assert "rebuild" not in str(caught.value)


@pytest.mark.parametrize("static_degree", [0, 2])
def test_placeholder_static_degree_is_validated_at_construction(static_degree):
    graph = g2rins.G2rins.make(PEI).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    placeholder = next(node for node, data in graph.nodes(data=True) if data["is_connector_placeholder"])
    static_edges = [(u, v, k) for u, v, k, d in graph.edges(keys=True, data=True) if d["static"] and placeholder in (u, v) and u != v]
    assert static_edges
    if static_degree == 0:
        graph.remove_edges_from(static_edges)
    else:
        anchor = next(v if u == placeholder else u for u, v, _k in static_edges)
        other = next(node for node, data in graph.nodes(data=True) if data["atomic_num"] > 0 and node != anchor)
        template = copy.deepcopy(graph.edges[static_edges[0]])
        graph.add_edge(placeholder, other, **copy.deepcopy(template))
        graph.add_edge(other, placeholder, **copy.deepcopy(template))
    with pytest.raises(IncompatibleGenerativeGraphSchema, match="exactly one static neighbor") as caught:
        g2rins.EnsembleCreator(graph)
    assert caught.value.node_id == placeholder
    assert caught.value.missing_attribute == "is_connector_placeholder"
    assert caught.value.reason == "invalid"
    assert f"found {static_degree}" in str(caught.value)


@pytest.mark.parametrize("json_round_trip", [False, True], ids=["memory", "json"])
def test_placeholders_cannot_use_each_other_as_static_anchors(json_round_trip):
    graph = g2rins.G2rins.make(PEI).get_graph_creator().get_generative_graph()
    placeholders = {node for node, data in graph.nodes(data=True) if data["is_connector_placeholder"]}
    first, second = placeholders
    static_edges = [(u, v, key) for u, v, key, data in graph.edges(keys=True, data=True) if data["static"] and placeholders.intersection((u, v)) and u != v]
    edge_data = copy.deepcopy(graph.edges[static_edges[0]])
    graph.remove_edges_from(static_edges)
    graph.add_edge(first, second, **copy.deepcopy(edge_data))
    graph.add_edge(second, first, **copy.deepcopy(edge_data))
    if json_round_trip:
        payload = json.loads(json.dumps(g2rins.generative_graph_json_data(graph)))
        graph = nx.node_link_graph(payload["graph"], edges="edges")
    with pytest.raises(IncompatibleGenerativeGraphSchema, match="real atom") as caught:
        g2rins.EnsembleCreator(graph)
    assert caught.value.node_id in placeholders
    assert caught.value.reason == "invalid"
    assert caught.value.missing_attribute == "is_connector_placeholder"


def test_wildcard_diagnostics_omit_stale_unit_text_after_relabeling():
    text = "{[] [<]CCO[>], [<]CC(*)O[>]; CO[>]; [<][H] []}|poisson(300.0)|"
    graph = g2rins.G2rins.make(text).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    labels = g2rins.derive_unit_labels(graph)
    assert "*" not in graph.graph["unit_g2rins"]["R0"]
    assert "*" in graph.graph["unit_g2rins"]["R1"]
    graph.remove_nodes_from(labels.unit_nodes["R0"])
    with pytest.raises(UnsupportedWildcardGeneration) as caught:
        g2rins.EnsembleCreator(graph)
    assert caught.value.unit_id == "R0"
    assert caught.value.unit_text is None


def _reorder_repeat_units_through_json(graph):
    labels = g2rins.derive_unit_labels(graph)
    first_unit = set(labels.unit_nodes["R1"])
    payload = json.loads(json.dumps(g2rins.generative_graph_json_data(graph)))
    payload["graph"]["nodes"].sort(key=lambda node: node["id"] not in first_unit)
    reordered = nx.node_link_graph(payload["graph"], edges="edges")
    assert set(g2rins.derive_unit_labels(reordered).unit_nodes["R0"]) == first_unit
    return reordered


def test_wildcard_diagnostics_omit_stale_unit_text_after_node_reordering():
    text = "{[] [<]CCO[>], [<]CC(*)O[>]; CO[>]; [<][H] []}|poisson(300.0)|"
    graph = g2rins.G2rins.make(text).get_graph_creator().get_generative_graph()
    graph = _reorder_repeat_units_through_json(graph)
    with pytest.raises(UnsupportedWildcardGeneration) as caught:
        g2rins.EnsembleCreator(graph)
    assert caught.value.unit_id == "R0"
    assert caught.value.unit_text is None


@pytest.mark.parametrize("membership", [{}, [], {"R0": None}, {"R0": [{}]}], ids=["empty", "wrong-shape", "null-nodes", "invalid-node"])
def test_wildcard_diagnostics_omit_unverified_unit_text(membership):
    """A malformed membership map is not evidence, so the text is withheld."""
    graph = g2rins.G2rins.make(LINEAR.replace("CCO", "CC(*)O")).get_graph_creator().get_generative_graph()
    graph.graph["unit_node_ids"] = membership
    with pytest.raises(UnsupportedWildcardGeneration) as caught:
        g2rins.EnsembleCreator(graph)
    assert caught.value.unit_id == "R0"
    assert caught.value.unit_text is None


def test_wildcard_diagnostics_keep_unit_text_from_graphs_predating_membership():
    """An absent map means a pre-membership graph, not a mutated one."""
    graph = g2rins.G2rins.make(LINEAR.replace("CCO", "CC(*)O")).get_graph_creator().get_generative_graph()
    graph.graph.pop("unit_node_ids", None)
    with pytest.raises(UnsupportedWildcardGeneration) as caught:
        g2rins.EnsembleCreator(graph)
    assert caught.value.unit_id == "R0"
    assert caught.value.unit_text == "[<]CC(*)O[>]"


def test_ensemble_metadata_omits_stale_unit_text_after_node_reordering():
    text = "{[] [<]CCO[>], [<]CC(N)O[>]; CO[>]; [<][H] []}|poisson(300.0)|"
    graph = g2rins.G2rins.make(text).get_graph_creator().get_generative_graph()
    graph = _reorder_repeat_units_through_json(graph)
    result = g2rins.EnsembleCreator(graph).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    repeat_units = [unit for unit_id, unit in result.units.items() if unit_id.startswith("R")]
    assert repeat_units
    assert all(unit["g2rins"] == "" for unit in repeat_units)
    assert result.units["I0"]["g2rins"] == "CO[>]"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("{[] [<]CCO[>]; CO[>]; [<]* []}|poisson(300.0)|", id="A2-terminator"),
        pytest.param("{[] [<]C*C[>]; CO[>]; [<][H] []}|poisson(300.0)|", id="A4-backbone"),
    ],
)
def test_legacy_migration_marks_provable_wildcards(text):
    graph = g2rins.G2rins.make(text).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    for _, attrs in graph.nodes(data=True):
        del attrs["is_connector_placeholder"]
    with pytest.warns(G2RINSWarning, match="static degree other than one"):
        migrated = g2rins.mark_legacy_connector_placeholders(graph)
    wildcards = [node for node, attrs in migrated.nodes(data=True) if attrs["atomic_num"] == 0 and attrs["is_connector_placeholder"] is False]
    assert len(wildcards) == 1
    with pytest.raises(UnsupportedWildcardGeneration) as caught:
        g2rins.EnsembleCreator(migrated)
    assert caught.value.node_id == wildcards[0]


def test_legacy_migration_cannot_distinguish_pendant_wildcards():
    """Documented limitation: a pendant wildcard has the static degree of a placeholder."""
    graph = g2rins.G2rins.make(LINEAR.replace("CCO", "CC(*)O")).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    for _, attrs in graph.nodes(data=True):
        del attrs["is_connector_placeholder"]
    with pytest.warns(G2RINSWarning, match="pendant"):
        migrated = g2rins.mark_legacy_connector_placeholders(graph)
    assert all(attrs["is_connector_placeholder"] is True for _, attrs in migrated.nodes(data=True) if attrs["atomic_num"] == 0)
    g2rins.EnsembleCreator(migrated)


def _round_trip(graph, drop_unit_node_ids=False):
    payload = json.loads(json.dumps(g2rins.generative_graph_json_data(graph)))
    if drop_unit_node_ids:
        payload["graph"]["graph"].pop("unit_node_ids", None)
    return nx.node_link_graph(payload["graph"], edges="edges")


def test_unit_texts_survive_graphs_written_before_unit_node_ids():
    """A previous-release export carries unit_g2rins but no membership map."""
    graph = g2rins.G2rins.make(PEI).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    expected = g2rins.EnsembleCreator(graph).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    legacy = _round_trip(graph, drop_unit_node_ids=True)
    assert "unit_node_ids" not in legacy.graph
    actual = g2rins.EnsembleCreator(legacy).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    assert {unit_id: info["g2rins"] for unit_id, info in actual.units.items()} == {unit_id: info["g2rins"] for unit_id, info in expected.units.items()}
    assert all(info["g2rins"] for info in actual.units.values())


def test_legacy_unit_texts_are_dropped_when_units_no_longer_match():
    """Without a membership map the weaker guard still catches a removed unit."""
    text = "{[] [<]CCO[>], [<]CC(C)O[>]; CO[>]; [<][H] []}|poisson(300.0)|"
    graph = g2rins.G2rins.make(text).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    legacy = _round_trip(graph, drop_unit_node_ids=True)
    labels = g2rins.derive_unit_labels(legacy)
    legacy.remove_nodes_from(labels.unit_nodes["R1"])
    result = g2rins.EnsembleCreator(legacy).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    assert all(info["g2rins"] == "" for info in result.units.values())


def test_numpy_atomic_numbers_are_normalized_in_the_creator_copy():
    graph = g2rins.G2rins.make(LINEAR).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    expected = g2rins.EnsembleCreator(graph).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    for _, data in graph.nodes(data=True):
        data["atomic_num"] = np.int64(data["atomic_num"])
    creator = g2rins.EnsembleCreator(graph)
    # Chem.Atom rejects numpy.int64, so the copy must hold Python ints.
    assert all(type(data["atomic_num"]) is int for _, data in creator.generative_graph.nodes(data=True))
    assert all(isinstance(data["atomic_num"], np.int64) for _, data in graph.nodes(data=True))
    assert creator.create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0) == expected


@pytest.mark.parametrize(
    "atomic_num",
    [True, False, 6.0, "6", None, np.bool_(True), np.float32(6), np.array(6), np.array([6, 7])],
    ids=["true", "false", "float", "string", "none", "numpy-bool", "numpy-float", "zero-dimensional-array", "array"],
)
def test_non_integer_atomic_numbers_are_rejected(atomic_num):
    graph = g2rins.G2rins.make(LINEAR).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    node = next(n for n, data in graph.nodes(data=True) if data["atomic_num"] == 6)
    graph.nodes[node]["atomic_num"] = atomic_num
    with pytest.raises(IncompatibleGenerativeGraphSchema, match="Expected an integer") as caught:
        g2rins.EnsembleCreator(graph)
    assert caught.value.node_id == node
    assert caught.value.reason == "invalid"
    # Bond-connector advice cannot help any of these values.
    assert "include_bond_connectors" not in str(caught.value)


def test_migration_warning_claims_no_placeholder_assumption_when_none_was_made():
    graph = g2rins.G2rins.make(LINEAR.replace("[<][H]", "[<]*")).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    for _, attrs in graph.nodes(data=True):
        del attrs["is_connector_placeholder"]
    with pytest.warns(G2RINSWarning) as caught:
        g2rins.mark_legacy_connector_placeholders(graph)
    message = str(caught[0].message)
    assert "static degree other than one" in message
    assert "Assuming" not in message
    assert "pendant" not in message


@pytest.mark.parametrize(
    ("attribute", "numpy_type"),
    [
        pytest.param("atomic_num", np.int64, id="atomic-num-int64"),
        pytest.param("charge", np.int64, id="charge-int64"),
        pytest.param("num_explicit_h", np.int32, id="explicit-h-int32"),
        pytest.param("gen_weight", np.float32, id="gen-weight-float32"),
    ],
)
def test_export_payload_is_json_serializable_with_numpy_attributes(attribute, numpy_type):
    """A producer emitting NumPy attributes must still get a serializable payload."""
    graph = g2rins.G2rins.make(LINEAR).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    expected = json.dumps(g2rins.generative_graph_json_data(graph), sort_keys=True)
    for _, data in graph.nodes(data=True):
        if attribute in data:
            data[attribute] = numpy_type(data[attribute])
    payload = g2rins.generative_graph_json_data(graph)
    assert json.dumps(payload, sort_keys=True) == expected
    assert all(not isinstance(node[attribute], np.generic) for node in payload["graph"]["nodes"] if attribute in node)
    # The caller's graph keeps its own types.
    assert any(isinstance(data.get(attribute), np.generic) for _, data in graph.nodes(data=True))


def test_export_payload_is_json_serializable_with_numpy_edge_attributes():
    graph = g2rins.G2rins.make(LINEAR).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    expected = json.dumps(g2rins.generative_graph_json_data(graph), sort_keys=True)
    # int -> int64 preserves the value while staying unserializable, so byte
    # identity is a fair check. A float cast would legitimately turn 0 into 0.0,
    # and `type(value) is int` skips bools, which would turn true into 1.
    cast = 0
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        for key, value in list(data.items()):
            if type(value) is int:
                data[key] = np.int64(value)
                cast += 1
    assert cast, "no integer edge attribute to cast"
    assert json.dumps(g2rins.generative_graph_json_data(graph), sort_keys=True) == expected


def test_export_preserves_negative_atomic_numbers_of_connector_graphs():
    """Coercion must not become validation: connector graphs export legitimately."""
    graph = g2rins.G2rins.make(PEI).get_graph_creator().get_generative_graph(include_bond_connectors=True)
    for _, data in graph.nodes(data=True):
        data["atomic_num"] = np.int64(data["atomic_num"])
    payload = g2rins.generative_graph_json_data(graph)
    json.dumps(payload)
    descriptors = [node["atomic_num"] for node in payload["graph"]["nodes"] if node["atomic_num"] < 0]
    assert descriptors
    assert all(type(value) is int for value in descriptors)


@pytest.mark.parametrize("clear_membership", [False, True], ids=["saved-membership", "empty-membership"])
def test_reordered_wildcard_never_quotes_another_units_text(clear_membership):
    text = "{[] [<]CCO[>], [<]CC(*)O[>]; CO[>]; [<][H] []}|poisson(100)|"
    graph = g2rins.G2rins.make(text).get_graph_creator().get_generative_graph()
    graph = _reorder_repeat_units_through_json(graph)
    if clear_membership:
        graph.graph["unit_node_ids"] = {}
    with pytest.raises(UnsupportedWildcardGeneration) as caught:
        g2rins.EnsembleCreator(graph)
    assert caught.value.unit_id == "R0"
    assert caught.value.unit_text is None


def test_empty_membership_omits_ensemble_text_without_changing_generation():
    graph = g2rins.G2rins.make(LINEAR).get_graph_creator().get_generative_graph()
    expected = g2rins.EnsembleCreator(graph).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    graph.graph["unit_node_ids"] = {}
    actual = g2rins.EnsembleCreator(graph).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    for info in expected.units.values():
        info["g2rins"] = ""
    assert actual == expected


@pytest.mark.parametrize("static_direction", ["incoming", "outgoing"])
@pytest.mark.parametrize("reverse_nonstatic", [False, True])
def test_migration_and_constructor_agree_on_asymmetric_static_neighbors(static_direction, reverse_nonstatic):
    graph = g2rins.G2rins.make(PEI).get_graph_creator().get_generative_graph()
    placeholder = next(node for node, data in graph.nodes(data=True) if data["is_connector_placeholder"])
    anchor = next(v for _, v, data in graph.out_edges(placeholder, data=True) if data["static"])
    removed_direction = (placeholder, anchor) if static_direction == "incoming" else (anchor, placeholder)
    graph.remove_edges_from([(u, v, key) for u, v, key, data in graph.edges(keys=True, data=True) if (u, v) == removed_direction and data["static"]])
    if reverse_nonstatic:
        template = next(copy.deepcopy(data) for _, _, data in graph.edges(data=True) if not data["static"])
        graph.add_edge(*removed_direction, key=0, **template)
    # The original explicitly marked graph is accepted. Migration must agree,
    # even when a reverse non-static edge shadows the sole static anchor edge.
    g2rins.EnsembleCreator(graph)
    del graph.nodes[placeholder]["is_connector_placeholder"]
    with pytest.warns(G2RINSWarning, match="Assuming 1"):
        migrated = g2rins.mark_legacy_connector_placeholders(graph)
    assert migrated.nodes[placeholder]["is_connector_placeholder"] is True
    assert "is_connector_placeholder" not in graph.nodes[placeholder]
    g2rins.EnsembleCreator(migrated)
