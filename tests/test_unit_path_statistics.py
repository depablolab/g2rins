"""Exact occurrence-graph path counting regressions."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pytest

from g2rins.ensemble_creator import (
    _UnitOccurrence,
    _public_unit_path_statistics,
    _unit_path_counts,
)


def _metadata(graph: nx.Graph) -> SimpleNamespace:
    occurrence_id = {node: index for index, node in enumerate(graph.nodes)}
    return SimpleNamespace(
        occurrences=tuple(
            _UnitOccurrence(
                unit_id="R0",
                prototype_key=("R0", (str(node),)),
                nodes=(node,),
                incoming_connection=None,
                incoming_edge_token=None,
                connections=[],
            )
            for node in graph.nodes
        ),
        inter_occurrence_edges=tuple(
            (occurrence_id[left], occurrence_id[right], ("1", False))
            for left, right in graph.edges
        ),
    )


def _totals(graph: nx.Graph) -> dict[int, int]:
    counts = _unit_path_counts(_metadata(graph), graph)
    return {k: sum(counter.values()) for k, counter in counts.items()}


def test_linear_occurrence_graph_counts_each_physical_path_once() -> None:
    graph = nx.path_graph(4)
    assert _totals(graph) == {2: 3, 3: 2, 4: 1}


def test_compact_incoming_edges_count_without_extra_edge_scan() -> None:
    graph = nx.path_graph(4)
    metadata = SimpleNamespace(
        occurrences=tuple(
            _UnitOccurrence(
                unit_id="R0",
                prototype_key=("R0", (str(node),)),
                nodes=(node,),
                incoming_connection=(node - 1, node) if node else None,
                incoming_edge_token=("1", False) if node else None,
                connections=[],
            )
            for node in graph.nodes
        ),
        inter_occurrence_edges=(),
    )
    counts = _unit_path_counts(metadata, graph)
    assert {k: sum(counter.values()) for k, counter in counts.items()} == {
        2: 3,
        3: 2,
        4: 1,
    }


def test_star_occurrence_graph_counts_shared_center_once() -> None:
    graph = nx.Graph()
    graph.add_edges_from((0, leaf) for leaf in (1, 2, 3))
    assert _totals(graph) == {2: 3, 3: 3, 4: 0}


def test_ring_closure_is_enumerated_from_compact_closure_metadata() -> None:
    graph = nx.cycle_graph(3)
    assert _totals(graph) == {2: 3, 3: 3, 4: 0}


def test_graft_occurrence_graph_counts_backbone_and_arm_paths() -> None:
    graph = nx.Graph(((0, 1), (1, 2), (1, 3), (3, 4)))
    assert _totals(graph) == {2: 4, 3: 4, 4: 2}


def test_occurrence_id_order_does_not_change_motif_counts() -> None:
    graph = nx.Graph(((0, 1), (1, 2), (1, 3), (3, 4)))
    relabeled = nx.relabel_nodes(graph, {0: 13, 1: 8, 2: 21, 3: 3, 4: 5})
    original = _unit_path_counts(_metadata(graph), graph)
    reordered = _unit_path_counts(_metadata(relabeled), relabeled)
    assert original == reordered


def test_path_count_limit_fails_fast() -> None:
    graph = nx.path_graph(4)
    with pytest.raises(ValueError, match="motif limit"):
        _unit_path_counts(_metadata(graph), graph, max_motifs=1)


def test_overlapping_occurrence_membership_fails() -> None:
    graph = nx.path_graph(2)
    metadata = SimpleNamespace(
        occurrences=(
            _UnitOccurrence("R0", ("R0", ("0",)), (0,), None, None, []),
            _UnitOccurrence("R0", ("R0", ("0",)), (0, 1), None, None, []),
        )
    )
    with pytest.raises(ValueError, match="multiple"):
        _unit_path_counts(metadata, graph)


def test_parallel_occurrence_edges_are_counted_without_collapse() -> None:
    graph = nx.path_graph(2)
    metadata = _metadata(graph)
    metadata.inter_occurrence_edges = (
        (0, 1, ("1", False)),
        (0, 1, ("1", False)),
    )
    counts = _unit_path_counts(metadata, graph)
    assert {k: sum(counter.values()) for k, counter in counts.items()} == {
        2: 2,
        3: 0,
        4: 0,
    }


def test_public_statistics_keep_local_display_tokens_with_canonical_tokens() -> None:
    counts = {
        2: {(("N", "R0", 1), ("E", "1", False, "1", "2"), ("N", "R1", 2)): 3},
        3: {},
        4: {},
    }
    units = {"R0": {"psmiles": "unit-a"}, "R1": {"psmiles": "unit-b"}}

    statistics = _public_unit_path_statistics(counts, units, accepted_chains=1)

    record = statistics["counts"]["2"][0]
    assert statistics["schema"] == "unit-graph-simple-paths/v2"
    assert record["token"] == [
        ["N", "unit-a", 1],
        ["E", "1", False, "1", "2"],
        ["N", "unit-b", 2],
    ]
    assert record["display_tokens"] == [
        {
            "token": [
                ["N", "R0", 1],
                ["E", "1", False, "1", "2"],
                ["N", "R1", 2],
            ],
            "count": 3,
        }
    ]
