# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Pure occurrence-view tests: no sampler, molecular graph, or RDKit fixtures."""

from collections import defaultdict
from copy import deepcopy
from itertools import permutations
from types import SimpleNamespace

import pytest

from g2rins._building_block_statistics import _BuildingBlockStatistics


def _sample(labels, edges=()):
    metadata = SimpleNamespace(
        occurrences=tuple(SimpleNamespace(unit_id=label) for label in labels)
    )
    # Deliberately opaque unused slots: no graph/atom/adjacency access is needed.
    return metadata, (object(), tuple(edges), object())


def _edge(left, right, source="1", target="2", bond="1", aromatic=False):
    return left, right, (bond, aromatic, source, target)


def _result(labels, edges=()):
    statistics = _BuildingBlockStatistics()
    statistics.update(*_sample(labels, edges))
    return statistics.finalize()


def _centers(result, unit):
    return [row for row in result["neighborhoods"] if row["center_unit"] == unit]


def _assert_json_safe(value):
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            _assert_json_safe(item)
    elif type(value) is list:
        for item in value:
            _assert_json_safe(item)
    else:
        assert type(value) in (str, int, bool, float, type(None))


def test_two_chains_three_motif_occurrences_but_two_chain_presences():
    statistics = _BuildingBlockStatistics()
    statistics.update(*_sample(["A", "B", "A", "B"], [_edge(0, 1), _edge(2, 3)]))
    statistics.update(*_sample(["A", "B"], [_edge(0, 1)]))
    result = statistics.finalize()
    assert result["accepted_count"] == statistics.chain_count == 2
    assert result["building_blocks"] == [
        {"unit_id": unit, "occurrence_count": 3, "chain_presence_count": 2}
        for unit in ("A", "B")
    ]
    assert len(result["neighborhoods"]) == 2
    for row in result["neighborhoods"]:
        assert row["occurrence_count"] == 3
        assert row["chain_presence_count"] == 2
    assert _centers(result, "A")[0]["neighbors"] == [{
        "unit_id": "B", "is_center": False,
        "connections": [{"source_site": "1", "target_site": "2",
                         "bond_type": "1", "aromatic": False}],
    }]
    for row in result["site_connections"]:
        assert row["endpoint_count"] == row["occupied_endpoint_count"] == 3
        assert row["chain_presence_count"] == 2
        assert row["probability"] == 1.0


def test_branch_is_one_complete_star_not_independent_pair_motifs():
    star = _result(["A", "B", "C"], [_edge(0, 1), _edge(0, 2, "3", "4")])
    pairs = _result(["A", "B", "A", "C"], [_edge(0, 1), _edge(2, 3, "3", "4")])
    assert star["site_connections"] == pairs["site_connections"]
    assert len(_centers(star, "A")) == 1
    assert len(_centers(star, "A")[0]["neighbors"]) == 2
    assert len(_centers(pairs, "A")) == 2
    assert all(len(row["neighbors"]) == 1 for row in _centers(pairs, "A"))


def test_neighbor_neighbor_edges_are_not_part_of_root_incident_signature():
    edges = [_edge(0, 1), _edge(0, 2, "3", "4")]
    star = _result(["A", "B", "C"], edges)
    triangle = _result(["A", "B", "C"], edges + [_edge(1, 2, "5", "6")])
    left, right = _centers(star, "A")[0], _centers(triangle, "A")[0]
    assert {key: value for key, value in left.items() if key != "motif_id"} == {
        key: value for key, value in right.items() if key != "motif_id"
    }
    assert star["neighborhood_definition"] == "root-incident-covalent-multistar/v1"


def test_isolates_and_associations_preserve_unit_occurrences_without_endpoints():
    result = _result(["A", "B", "A"], [_edge(0, 1, bond="0")])
    assert result == _result(["A", "B", "A"])
    assert result["site_connections"] == []
    assert [row["neighbors"] for row in result["neighborhoods"]] == [[], []]
    assert _centers(result, "A")[0]["occurrence_count"] == 2
    assert _centers(result, "A")[0]["chain_presence_count"] == 1


def test_self_link_emits_two_oriented_arms_and_endpoints():
    result = _result(["A"], [_edge(0, 0)])
    assert _centers(result, "A")[0]["neighbors"] == [{
        "unit_id": "A", "is_center": True,
        "connections": [
            {"source_site": "1", "target_site": "2", "bond_type": "1", "aromatic": False},
            {"source_site": "2", "target_site": "1", "bond_type": "1", "aromatic": False},
        ],
    }]
    assert sum(row["endpoint_count"] for row in result["site_connections"]) == 2
    same_site = _result(["A"], [_edge(0, 0, "1", "1")])
    row = same_site["site_connections"][0]
    assert row["endpoint_count"] == row["occupied_endpoint_count"] == 2
    assert row["chain_presence_count"] == 1
    assert len(_centers(same_site, "A")[0]["neighbors"][0]["connections"]) == 2


def test_parallel_edges_to_one_neighbor_differ_from_separate_same_type_neighbors():
    parallel = _result(["A", "B"], [_edge(0, 1), _edge(0, 1)])
    separate = _result(["A", "B", "B"], [_edge(0, 1), _edge(0, 2)])
    assert parallel["site_connections"] == separate["site_connections"]
    parallel_groups = _centers(parallel, "A")[0]["neighbors"]
    separate_groups = _centers(separate, "A")[0]["neighbors"]
    assert len(parallel_groups) == 1
    assert len(parallel_groups[0]["connections"]) == 2
    assert len(separate_groups) == 2
    assert separate_groups[0] == separate_groups[1]
    assert separate_groups[0]["is_center"] is False


def test_same_type_other_occurrence_is_not_a_self_link():
    result = _result(["A", "A"], [_edge(0, 1, "1", "1")])
    assert len(result["neighborhoods"]) == 1
    assert result["neighborhoods"][0]["occurrence_count"] == 2
    assert result["neighborhoods"][0]["neighbors"][0]["is_center"] is False
    assert result["site_connections"][0]["endpoint_count"] == 2


def test_central_unit_labels_are_never_merged():
    result = _result(["R0", "R1", "B", "B"], [_edge(0, 2), _edge(1, 3)])
    assert len(_centers(result, "R0")) == len(_centers(result, "R1")) == 1
    assert len(result["neighborhoods"]) == 4


def test_orientation_occurrence_and_edge_permutations_are_invariant():
    labels = ["A", "B", "B", "A"]
    edges = [_edge(0, 1), _edge(0, 1, "3", "4", "2"),
             _edge(0, 2, "5", "6", "4", True), _edge(3, 3, "7", "8", "3")]
    expected = _result(labels, edges)
    for order in permutations(range(len(labels))):
        remap = {old: new for new, old in enumerate(order)}
        for edge_order in (edges, list(reversed(edges))):
            for reverse in (False, True):
                changed = []
                for left, right, (bond, aromatic, source, target) in edge_order:
                    if reverse:
                        left, right, source, target = right, left, target, source
                    changed.append(_edge(remap[left], remap[right], source, target, bond, aromatic))
                assert _result([labels[index] for index in order], changed) == expected


def test_site_denominator_is_all_occupied_endpoints_not_chains_or_targets():
    statistics = _BuildingBlockStatistics()
    statistics.update(*_sample(["A", "A", "B", "C"], [
        _edge(0, 1, "1", "1"), _edge(0, 2, "1", "2"),
        _edge(0, 3, "1", "3", "2"), _edge(0, 2, "1", "4", "4", True),
    ]))
    statistics.update(*_sample(["A"]))  # Unoccupied sites add no denominator.
    rows = statistics.finalize()["site_connections"]
    source_rows = [row for row in rows if (row["source_unit"], row["source_site"]) == ("A", "1")]
    assert len(source_rows) == 4
    assert all(row["occupied_endpoint_count"] == 5 for row in source_rows)
    homotypic = next(row for row in source_rows if row["target_unit"] == "A")
    assert homotypic["endpoint_count"] == 2
    assert homotypic["probability"] == pytest.approx(2 / 5)
    sums = defaultdict(float)
    for row in rows:
        sums[row["source_unit"], row["source_site"]] += row["probability"]
        assert row["chain_presence_count"] == 1
    assert all(value == pytest.approx(1) for value in sums.values())


def test_empty_samples_count_and_empty_finalization_schema():
    statistics = _BuildingBlockStatistics()
    empty = statistics.finalize()
    assert empty == {
        "schema": "building-block-statistics/v1", "population": "all-accepted-samples",
        "accepted_count": 0,
        "neighborhood_definition": "root-incident-covalent-multistar/v1",
        "probability_definition": "conditional-on-occupied-source-site-endpoints",
        "building_blocks": [], "neighborhoods": [], "site_connections": [],
    }
    statistics.update(*_sample([]))
    statistics.update(*_sample([]))
    assert statistics.chain_count == 2
    assert statistics.finalize() == {**empty, "accepted_count": 2}


def test_finalize_is_deterministic_fresh_detached_and_json_safe():
    statistics = _BuildingBlockStatistics()
    metadata, view = _sample(["B", "A"], [_edge(0, 1)])
    statistics.update(metadata, view)
    expected = statistics.finalize()
    snapshot = statistics.finalize()
    assert snapshot == expected and snapshot is not expected
    _assert_json_safe(snapshot)
    snapshot["building_blocks"][0]["occurrence_count"] = -99
    snapshot["neighborhoods"][0]["neighbors"][0]["connections"][0]["source_site"] = "999"
    snapshot["site_connections"][0]["probability"] = -1
    metadata.occurrences[0].unit_id = "CHANGED"
    assert statistics.finalize() == expected
    statistics.update(*_sample([]))
    assert expected["accepted_count"] == 1
    assert statistics.finalize()["accepted_count"] == 2
    assert all(type(value) is int or type(value).__name__ == "Counter"
               for value in vars(statistics).values())


@pytest.mark.parametrize("index", [True, False, -1, 2, 1.0, "1", None])
@pytest.mark.parametrize("side", [0, 1])
def test_invalid_indices_are_transactional(index, side):
    edge = list(_edge(0, 1))
    edge[side] = index
    _assert_bad_update(tuple(edge))


@pytest.mark.parametrize("site", ["", "0", "00", "01", "-1", "+1", "1.0", " 1", "1 ", "١", "１", 1, True, None])
@pytest.mark.parametrize("side", [2, 3])
def test_invalid_sites_are_transactional(site, side):
    token = list(_edge(0, 1)[2])
    token[side] = site
    _assert_bad_update((0, 1, tuple(token)))


@pytest.mark.parametrize("token", [
    (), ("1", False), ("1", False, "1", "2", "extra"), None,
    (1, False, "1", "2"), (True, False, "1", "2"),
    ("01", False, "1", "2"), ("5", False, "1", "2"),
    ("1", 0, "1", "2"), ("1", "false", "1", "2"),
    ("0", False, "0", "2"),  # Association tokens are still validated.
])
def test_bad_tokens_are_transactional(token):
    _assert_bad_update((0, 1, token))


@pytest.mark.parametrize("edge", [None, (), (0, 1), (0, 1, (), "extra"), "bad"])
def test_bad_edge_shapes_are_transactional(edge):
    _assert_bad_update(edge)


def _assert_bad_update(edge):
    statistics = _BuildingBlockStatistics()
    statistics.update(*_sample(["saved"]))
    before = deepcopy(vars(statistics))
    # The valid prefix must not leak into any counter before failure.
    with pytest.raises(ValueError):
        statistics.update(*_sample(["A", "B"], [_edge(0, 1), edge]))
    assert vars(statistics) == before
    statistics.update(*_sample(["after"]))
    assert statistics.finalize()["accepted_count"] == 2


@pytest.mark.parametrize("label", ["", None, 1, True])
def test_invalid_unit_labels_are_transactional(label):
    statistics = _BuildingBlockStatistics()
    statistics.update(*_sample(["saved"]))
    before = deepcopy(vars(statistics))
    with pytest.raises(ValueError):
        statistics.update(*_sample(["A", label]))
    assert vars(statistics) == before


@pytest.mark.parametrize("view", [None, (), ([], []), ([], None, {}), ([], [], {}, "extra")])
def test_bad_views_are_transactional(view):
    statistics = _BuildingBlockStatistics()
    statistics.update(*_sample(["saved"]))
    before = deepcopy(vars(statistics))
    with pytest.raises(ValueError):
        statistics.update(_sample(["A"])[0], view)
    assert vars(statistics) == before


@pytest.mark.parametrize("bond", ["1", "2", "3", "4"])
def test_all_covalent_bond_types_and_canonical_large_sites_are_preserved(bond):
    site = "123456789012345678901234567890"
    result = _result(["A", "B"], [_edge(0, 1, site, "10", bond, True)])
    assert len(result["site_connections"]) == 2
    assert all(row["bond_type"] == bond and row["aromatic"] is True
               for row in result["site_connections"])
    assert _centers(result, "A")[0]["neighbors"][0]["connections"][0]["source_site"] == site


def test_chain_update_order_does_not_affect_final_rows_or_ids():
    first = _sample(["B", "C"], [_edge(0, 1)])
    second = _sample(["A", "B"], [_edge(0, 1, "3", "4")])
    left, right = _BuildingBlockStatistics(), _BuildingBlockStatistics()
    for sample in (first, second, first):
        left.update(*sample)
    for sample in (first, first, second):
        right.update(*sample)
    assert left.finalize() == right.finalize()