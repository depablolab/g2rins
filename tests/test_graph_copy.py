# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Correctness contracts for conservative unit-prototype copies (no timings)."""

from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

import g2rins
import g2rins.ensemble_creator as ensemble_module
from g2rins._graph_copy import _copy_unit_graph
from tests._support import _reset_rngs


class _DictSubclass(dict):
    pass


class _IntSubclass(int):
    pass


class _StrSubclass(str):
    pass


class _GraphSubclass(nx.Graph):
    pass


@pytest.mark.parametrize("location", ["node", "edge"])
def test_unhashable_extension_type_respects_custom_deepcopy_hook(location):
    class UnhashableType(type):
        __hash__ = None

    class Extension(metaclass=UnhashableType):
        pass

    graph = _prototype()
    attributes = graph._node[0] if location == "node" else graph._adj[0][1]
    attributes["extension"] = Extension()
    calls = []
    sentinel = object()

    def fallback(value):
        calls.append(value)
        # A caller may supply an extension-aware deepcopy implementation.
        return sentinel

    copied = _copy_unit_graph(graph, deepcopy=fallback)
    assert calls == [graph]
    assert copied is sentinel


def _prototype():
    # Graph.copy inserts edges in node order, changing node 1's [2, 0, 1]
    # neighbor order to [0, 2, 1]. Access private dicts to avoid caching views.
    graph = nx.Graph()
    graph.add_nodes_from((0, 1, 2, -7, 10**30))
    graph.add_edge(2, 1, bond_type=1, aromatic=False)
    graph.add_edge(0, 1, bond_type=2, aromatic=False)
    graph.add_edge(1, 1, bond_type=1)
    for node, attributes in graph._node.items():
        attributes.update(origin_idx=str(node), atomic_num=6, charge=0)
    return graph


def _ordered_graph(graph):
    """Compare actual IDs, ordered attributes, and every neighbor, not isomorphism."""
    return (
        tuple(graph.graph.items()),
        tuple((node, tuple(data.items())) for node, data in graph._node.items()),
        tuple(
            (node, tuple((other, tuple(data.items())) for other, data in row.items()))
            for node, row in graph._adj.items()
        ),
    )


def _no_fallback(_graph):
    pytest.fail("plain scalar prototype unexpectedly fell back to deepcopy")


@pytest.mark.parametrize("shape", ["empty", "isolates", "ordered-self-loop"])
def test_plain_copy_exactly_matches_deepcopy_order_and_detaches_dicts(shape):
    graph = _prototype()
    if shape == "empty":
        graph = nx.Graph()
    elif shape == "isolates":
        graph = nx.Graph()
        graph.add_nodes_from((10**30, -7, 0))
    expected = deepcopy(graph)
    actual = _copy_unit_graph(graph, deepcopy=_no_fallback)

    assert type(actual) is nx.Graph
    assert actual is not graph
    assert tuple(actual.__dict__) == tuple(graph.__dict__)
    assert _ordered_graph(actual) == _ordered_graph(expected)
    for field in graph.__dict__:
        assert actual.__dict__[field] is not graph.__dict__[field]
    assert all(a is b for a, b in zip(actual._node, graph._node))
    for node in graph._node:
        assert actual._node[node] is not graph._node[node]
        assert actual._adj[node] is not graph._adj[node]
        assert tuple(actual._adj[node]) == tuple(graph._adj[node])
        for other in graph._adj[node]:
            assert actual._adj[node][other] is actual._adj[other][node]
            assert actual._adj[node][other] is not graph._adj[node][other]
    if shape == "ordered-self-loop":
        assert tuple(graph._adj[1]) == (2, 0, 1)
        assert tuple(graph.copy()._adj[1]) != tuple(graph._adj[1])
        assert actual._adj[1][1] is not graph._adj[1][1]


def test_copy_preserves_field_attribute_and_adjacency_field_order():
    graph = _prototype()
    graph._node[0].update({b"bytes": b"value", None: None, 3.5: 2.5, False: True})
    # Both outer structural dictionaries may have their own insertion order.
    graph._adj = dict(reversed(tuple(graph._adj.items())))
    state = graph.__dict__
    graph.__dict__ = {key: state[key] for key in reversed(tuple(state))}
    actual = _copy_unit_graph(graph, deepcopy=_no_fallback)
    assert tuple(actual.__dict__) == tuple(graph.__dict__)
    assert _ordered_graph(actual) == _ordered_graph(deepcopy(graph))
    assert tuple(actual._node[0]) == tuple(graph._node[0])


@pytest.mark.parametrize("cache_present", [False, True])
def test_plain_immutable_attribute_keys_and_values_need_no_recursive_copy(cache_present):
    graph = _prototype()
    graph.__dict__.pop("__networkx_cache__", None)
    if cache_present:
        graph.__dict__["__networkx_cache__"] = {}
    attributes = {
        "long-key-" * 40: "long-value-" * 40,
        10**30: 10**40,
        3.5: 2.5,
        False: True,
        b"bytes": b"value",
        None: None,
    }
    graph._node[0] = attributes
    actual = _copy_unit_graph(graph, deepcopy=_no_fallback)
    assert tuple(actual.__dict__) == tuple(graph.__dict__)
    assert _ordered_graph(actual) == _ordered_graph(deepcopy(graph))
    for (key, value), (copied_key, copied_value) in zip(
        attributes.items(), actual._node[0].items(), strict=True,
    ):
        assert copied_key is key
        assert copied_value is value


def test_node_and_edge_dicts_are_independently_mutable_across_copies():
    graph = _prototype()
    before = _ordered_graph(graph)
    left = _copy_unit_graph(graph, deepcopy=_no_fallback)
    right = _copy_unit_graph(graph, deepcopy=_no_fallback)
    left._node[0]["charge"] = 9
    left._adj[0][1]["bond_type"] = 3
    left.remove_node(2)
    assert left._node[1]["charge"] == 0
    assert left._adj[1][1]["bond_type"] == 1
    assert _ordered_graph(graph) == _ordered_graph(right) == before
    graph._node[0]["origin_idx"] = "changed source"
    graph._adj[1][1]["bond_type"] = 4
    assert right._node[0]["origin_idx"] == "0"
    assert right._adj[1][1]["bond_type"] == 1


def test_shared_node_edge_attribute_aliases_preserved_but_detached():
    graph = _prototype()
    shared = {"first": 1, "second": None}
    graph._node[0] = graph._node[2] = shared
    graph._adj[0][1] = graph._adj[1][0] = shared
    graph._adj[1][1] = shared
    actual = _copy_unit_graph(graph, deepcopy=_no_fallback)
    assert _ordered_graph(actual) == _ordered_graph(deepcopy(graph))
    assert actual._node[0] is actual._node[2]
    assert actual._node[0] is actual._adj[0][1] is actual._adj[1][0]
    assert actual._node[0] is actual._adj[1][1]
    assert actual._node[0] is not shared
    actual._node[0]["first"] = 42
    assert actual._adj[1][1]["first"] == 42
    assert shared["first"] == 1
    assert actual._node[1]["charge"] == 0


_GUARDS = (
    "directed", "multigraph", "graph-subclass", "view", "instance-dict-subclass",
    "extra-attribute", "nonstring-field", "string-subclass-field",
    "missing-graph", "missing-node", "missing-adj",
    "graph-list", "node-list", "adj-list", "cache-list",
    "graph-dict-subclass", "node-dict-subclass", "adj-dict-subclass",
    "cache-dict-subclass", "graph-metadata", "backend-cache",
    "cached-nodes", "cached-edges", "cached-adj", "cached-degree",
    "mismatched-node-sets", "string-node", "bool-node", "integer-subclass-node",
    "noninteger-adjacency-node", "noninteger-node-only", "neighbor-list", "neighbor-dict-subclass",
    "structural-alias", "neighbor-structural-alias", "shared-neighbor-dict",
    "node-structural-alias", "edge-structural-alias",
    "node-attribute-list", "edge-attribute-list",
    "node-attribute-subclass", "edge-attribute-subclass",
    "nested-node", "nested-edge", "node-cycle", "edge-cycle", "graph-cycle",
    "structure-value", "node-scalar-subclass", "edge-scalar-subclass",
    "node-key-subclass", "edge-key-subclass", "tuple-key", "tuple-value",
    "noninteger-neighbor", "dangling-neighbor", "missing-reverse",
    "different-reverse-attributes",
)


def _unsupported(case, graph=None):
    graph = _prototype() if graph is None else graph
    if case == "directed":
        return nx.DiGraph(graph)
    if case == "multigraph":
        return nx.MultiGraph(graph)
    if case == "graph-subclass":
        return _GraphSubclass(graph)
    if case == "view":
        return graph.copy(as_view=True)
    if case == "instance-dict-subclass":
        graph.__dict__ = _DictSubclass(graph.__dict__)
    elif case == "extra-attribute":
        graph.custom = None
    elif case == "nonstring-field":
        graph.__dict__[17] = {}
    elif case == "string-subclass-field":
        graph.__dict__ = {
            (_StrSubclass(key) if key == "graph" else key): value
            for key, value in graph.__dict__.items()
        }
    elif case in ("missing-graph", "missing-node", "missing-adj"):
        del graph.__dict__[{"graph": "graph", "node": "_node", "adj": "_adj"}[case[8:]]]
    elif case in ("graph-list", "node-list", "adj-list", "cache-list"):
        graph.__dict__[
            {"graph-list": "graph", "node-list": "_node", "adj-list": "_adj",
             "cache-list": "__networkx_cache__"}[case]
        ] = []
    elif case in ("graph-dict-subclass", "node-dict-subclass", "adj-dict-subclass",
                  "cache-dict-subclass"):
        field = {"graph-dict-subclass": "graph", "node-dict-subclass": "_node",
                 "adj-dict-subclass": "_adj", "cache-dict-subclass": "__networkx_cache__"}[case]
        graph.__dict__[field] = _DictSubclass(graph.__dict__.get(field, {}))
    elif case == "graph-metadata":
        graph.graph["name"] = "prototype"
    elif case == "backend-cache":
        graph.__dict__["__networkx_cache__"] = {"backend": object()}
    elif case.startswith("cached-"):
        getattr(graph, case[7:])
    elif case == "mismatched-node-sets":
        graph._node[99] = {}
    elif case in ("string-node", "bool-node", "integer-subclass-node"):
        node = {"string-node": "99", "bool-node": True,
                "integer-subclass-node": _IntSubclass(99)}[case]
        # True would otherwise collide with existing node 1.
        if case == "bool-node":
            graph = nx.Graph()
        graph.add_node(node)
    elif case == "noninteger-adjacency-node":
        graph._adj = {(_IntSubclass(node) if node == 2 else node): row
                      for node, row in graph._adj.items()}
    elif case == "noninteger-node-only":
        # Equality of dict key sets still holds; only the _node key type differs.
        graph._node = {(_IntSubclass(node) if node == 2 else node): data
                       for node, data in graph._node.items()}
    elif case == "neighbor-list":
        graph._adj[0] = []
    elif case == "neighbor-dict-subclass":
        graph._adj[0] = _DictSubclass(graph._adj[0])
    elif case == "structural-alias":
        graph.__dict__["__networkx_cache__"] = graph.graph
    elif case == "neighbor-structural-alias":
        graph._adj[-7] = graph.graph
    elif case == "shared-neighbor-dict":
        graph._adj[10**30] = graph._adj[-7]
    elif case == "node-structural-alias":
        graph._node[0] = graph._adj[-7]
    elif case == "edge-structural-alias":
        graph._adj[0][1] = graph._adj[1][0] = graph._adj[-7]
    elif case in ("node-attribute-list", "node-attribute-subclass"):
        graph._node[0] = [] if case.endswith("list") else _DictSubclass(graph._node[0])
    elif case in ("edge-attribute-list", "edge-attribute-subclass"):
        data = [] if case.endswith("list") else _DictSubclass(graph._adj[0][1])
        graph._adj[0][1] = graph._adj[1][0] = data
    elif case == "nested-node":
        graph._node[0]["nested"] = {"items": [1, 2]}
    elif case == "nested-edge":
        graph._adj[0][1]["nested"] = [1, {"two": 2}]
    elif case == "node-cycle":
        graph._node[0]["self"] = graph._node[0]
    elif case == "edge-cycle":
        graph._adj[0][1]["self"] = graph._adj[0][1]
    elif case == "graph-cycle":
        graph._node[0]["owner"] = graph
    elif case == "structure-value":
        graph._adj[0][1]["structure"] = graph._node
    elif case in ("node-scalar-subclass", "edge-scalar-subclass",
                  "node-key-subclass", "edge-key-subclass", "tuple-key", "tuple-value"):
        data = graph._adj[0][1] if case.startswith("edge-") else graph._node[0]
        if "key-subclass" in case:
            data[_StrSubclass("subclass")] = 1
        elif case == "tuple-key":
            data[(1, 2)] = 1
        else:
            data["unsupported"] = (1, 2) if case == "tuple-value" else _IntSubclass(7)
    elif case == "noninteger-neighbor":
        graph._adj[0] = {_IntSubclass(1): graph._adj[0][1]}
    elif case == "dangling-neighbor":
        graph._adj[0][99] = {}
    elif case == "missing-reverse":
        del graph._adj[1][0]
    elif case == "different-reverse-attributes":
        graph._adj[1][0] = dict(graph._adj[0][1])
    else:
        raise AssertionError(case)
    return graph


@pytest.mark.parametrize("case", _GUARDS)
def test_every_guard_delegates_original_object_and_returns_hook_identity(case):
    graph = _unsupported(case)
    calls = []
    sentinel = object()

    def fallback(value):
        calls.append(value)
        return sentinel

    assert _copy_unit_graph(graph, deepcopy=fallback) is sentinel
    assert len(calls) == 1
    assert calls[0] is graph


@pytest.mark.parametrize("case", [
    "nested-node", "nested-edge", "node-cycle", "edge-cycle", "graph-cycle",
    "structure-value", "node-structural-alias", "edge-structural-alias",
    "graph-metadata", "backend-cache", "extra-attribute",
])
def test_successful_copy_does_not_cache_eligibility_after_mutation(case):
    graph = _prototype()
    first = _copy_unit_graph(graph, deepcopy=_no_fallback)
    before = _ordered_graph(first)
    assert _unsupported(case, graph) is graph
    calls = []

    def fallback(value):
        calls.append(value)
        return deepcopy(value)

    second = _copy_unit_graph(graph, deepcopy=fallback)
    assert len(calls) == 1 and calls[0] is graph
    assert second is not graph
    assert _ordered_graph(first) == before


@pytest.mark.parametrize("injected", [False, True])
def test_real_deepcopy_fallback_preserves_cycles_and_structural_aliases(injected):
    graph = _prototype()
    shared = {"nested": [1, {"two": 2}]}
    shared["self"] = shared
    shared["owner"] = graph
    shared["nodes"] = graph._node
    shared["adjacency"] = graph._adj
    shared["neighbors"] = graph._adj[1]
    graph._node[0] = graph._node[2] = shared
    graph._adj[0][1] = graph._adj[1][0] = shared
    graph.graph["self"] = graph
    graph.custom = shared
    calls = []

    def fallback(value):
        calls.append(value)
        return deepcopy(value)

    actual = _copy_unit_graph(graph, deepcopy=fallback) if injected else _copy_unit_graph(graph)
    if injected:
        assert len(calls) == 1 and calls[0] is graph
    assert actual is not graph
    assert actual.graph["self"] is actual
    copied = actual._node[0]
    assert copied is actual._node[2] is actual._adj[0][1] is actual._adj[1][0]
    assert copied is actual.custom
    assert copied["self"] is copied
    assert copied["owner"] is actual
    assert copied["nodes"] is actual._node
    assert copied["adjacency"] is actual._adj
    assert copied["neighbors"] is actual._adj[1]
    assert copied is not shared
    assert copied["nested"] == shared["nested"]
    copied["nested"][1]["two"] = 99
    assert shared["nested"][1]["two"] == 2


@pytest.mark.parametrize("case", ["directed", "multigraph", "graph-subclass", "view",
                                      "cached-nodes", "cached-edges", "cached-adj", "cached-degree"])
def test_real_fallback_retains_networkx_type_and_cached_view_owner(case):
    graph = _unsupported(case)
    actual = _copy_unit_graph(graph)
    expected = deepcopy(graph)
    assert type(actual) is type(expected) is type(graph)
    assert nx.utils.graphs_equal(actual, expected)
    if case.startswith("cached-"):
        name = case[7:]
        assert getattr(actual, name) is not getattr(graph, name)
        # A copied cached view must follow the copied graph, not the source.
        actual.add_edge(2, 99)
        if name in ("nodes", "adj"):
            assert 99 in getattr(actual, name) and 99 not in getattr(graph, name)
        elif name == "edges":
            assert (2, 99) in actual.edges and (2, 99) not in graph.edges
        else:
            assert actual.degree[2] == graph.degree[2] + 1


_RECIPES = (
    "C{[>][<]CC(C)[>];;[<]}|uniform(180,180)|[H]",
    "{[] [<]CCN([>])[>]; [<][H]; O[>], [<][H] []}|uniform(240,240)|",
)


def _creator(recipe):
    return g2rins.G2rins.make(recipe).get_graph_creator().get_ensemble_creator()


def _sequence_signature(sequences):
    return [[_ordered_graph(unit) for unit in sequence] for sequence in sequences]


def _assert_detached_materializations(graphs, prototypes):
    """No occurrence/representative shares graph or mutable dicts with another."""
    assert graphs
    assert len({id(graph) for graph in graphs}) == len(graphs)
    seen = set()
    for graph in [*prototypes, *graphs]:
        # Mirrored edges within one graph intentionally share attribute dicts.
        owned = {id(graph.graph), id(graph._node), id(graph._adj)}
        owned.update(id(data) for data in graph._node.values())
        owned.update(id(row) for row in graph._adj.values())
        owned.update(id(data) for row in graph._adj.values() for data in row.values())
        assert not seen.intersection(owned)
        seen.update(owned)


@pytest.mark.parametrize("recipe", _RECIPES, ids=["linear", "branched"])
@pytest.mark.parametrize("boundary", ["compact", "public-legacy"])
def test_seeded_integrations_match_forced_deepcopy_and_leave_prototypes_unchanged(
    monkeypatch, recipe, boundary,
):
    original_compact = ensemble_module._PartialAtomGraph.compact_metadata
    _reset_rngs(7)
    # Reuse one parsed model: reparsing generates new origin IDs independently
    # of the sampling RNG, which would obscure an exact metadata comparison.
    graph_creator = g2rins.G2rins.make(recipe).get_graph_creator()
    summaries = []
    call_counts = []
    for force_deepcopy in (False, True):
        _reset_rngs(7)
        creator = graph_creator.get_ensemble_creator()
        captured = []
        calls = []
        fast_calls = []
        fallback_calls = []
        real_hook = ensemble_module.deepcopy

        def hook(value):
            fallback_calls.append(value)
            return real_hook(value)

        def copy_unit(graph, *, deepcopy):
            assert deepcopy is hook  # All three call sites preserve the coordinator hook.
            calls.append(graph)
            if force_deepcopy:
                return deepcopy(graph)
            before = len(fallback_calls)
            result = _copy_unit_graph(graph, deepcopy=deepcopy)
            if len(fallback_calls) == before:
                fast_calls.append(graph)
            return result

        def capture(partial, *args, **kwargs):
            compact = original_compact(partial, *args, **kwargs)
            prototypes = tuple(partial._unit_prototypes.values())
            captured.append((compact, prototypes, [_ordered_graph(g) for g in prototypes]))
            return compact

        with monkeypatch.context() as patch:
            patch.setattr(ensemble_module, "deepcopy", hook)
            patch.setattr(ensemble_module, "_copy_unit_graph", copy_unit)
            patch.setattr(ensemble_module._PartialAtomGraph, "compact_metadata", capture)
            if boundary == "compact":
                sample = creator.sample_mol_graph(
                    rng=np.random.default_rng(7),
                    _metadata_mode=ensemble_module._MetadataLevel.COMPACT_SEQUENCES,
                )
                molecule = sample.graph
                units = {}
                sequences = sample.metadata.materialize_sequences()
                # Repeated materialization must not reuse mutable occurrence graphs.
                repeated = sample.metadata.materialize_sequences()
                assert _sequence_signature(repeated) == _sequence_signature(sequences)
                weights, distributions = sample.mol_weights, sample.distributions
            else:
                molecule, units, bonds, sequences, weights, distributions = creator.sample_mol_graph(
                    rng=np.random.default_rng(7), molecule_info=True,
                )
                repeated = []

        assert len(captured) == 1
        compact, prototypes, before = captured[0]
        assert prototypes and calls
        assert [_ordered_graph(g) for g in prototypes] == before
        occurrence_count = sum(map(len, sequences))
        assert occurrence_count > len(prototypes)  # Exercise repeated prototype reuse.
        assert len(calls) == (2 * occurrence_count if boundary == "compact"
                              else occurrence_count + len(units))
        if force_deepcopy:
            assert all(any(value is graph for value in fallback_calls) for graph in calls)
            assert not fast_calls
        else:
            assert len(fast_calls) == len(calls)
        graphs = [*units, *(g for sequence in sequences + repeated for g in sequence)]
        _assert_detached_materializations(graphs, prototypes)
        if boundary == "public-legacy":
            assert bonds == compact.bond_counts
            assert sorted(units.values()) == sorted(compact.unit_counts.values())
        summaries.append((
            _ordered_graph(molecule),
            g2rins.mol_graph_to_smiles(molecule, smiles_policy="canonical"),
            [(_ordered_graph(graph), count) for graph, count in units.items()],
            _sequence_signature(sequences), weights, distributions,
            compact.unit_counts, compact.bond_counts, compact.labeled_bond_counts,
            compact.occurrences, compact.sequences, compact.inter_occurrence_edges,
            compact.origin_bond_id, before,
        ))
        call_counts.append(len(calls))
        # Demonstrate independent mutation after all exact parity snapshots.
        first = graphs[0]
        node = next(iter(first._node))
        first._node[node]["test-only"] = "changed"
        first.add_node("test-only")
        assert all("test-only" not in graph._node for graph in graphs[1:])
        assert [_ordered_graph(g) for g in prototypes] == before
    assert call_counts[0] == call_counts[1] > 0
    assert summaries[0] == summaries[1]


@pytest.mark.parametrize("mode", ["public-disabled", "internal-none", "counts"])
def test_disabled_or_counts_only_metadata_never_calls_unit_copy(monkeypatch, mode):
    _reset_rngs(7)
    creator = _creator(_RECIPES[0])

    def forbidden(*_args, **_kwargs):
        pytest.fail("unit copy helper called without graph-valued metadata")

    monkeypatch.setattr(ensemble_module, "_copy_unit_graph", forbidden)
    kwargs = {"molecule_info": False}
    if mode != "public-disabled":
        kwargs["_metadata_mode"] = (
            ensemble_module._MetadataLevel.COUNTS if mode == "counts"
            else ensemble_module._MetadataLevel.NONE
        )
    sample = creator.sample_mol_graph(rng=np.random.default_rng(7), **kwargs)
    if mode == "counts":
        assert sample.metadata.unit_counts
        assert sample.metadata.unit_prototypes == {}
        assert sample.metadata.sequences == ()
        assert sample.legacy_units is None and sample.legacy_sequences is None
    else:
        assert len(sample) > 0