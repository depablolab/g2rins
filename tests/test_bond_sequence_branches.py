"""Sequence-like branch rendering must preserve every occupied site once."""

from collections import Counter
import warnings

import pytest

import g2rins
from g2rins.bond_sequences import bond_sequence_branch_metadata


def topology(units, edges, components=None):
    return {
        "schema": "bond-sequence-topology/v1",
        "occurrences": list(enumerate(units)),
        "edges": edges,
        "components": components if components is not None else [
            {"root": 0, "occurrence_ids": list(range(len(units)))}
        ],
    }


def assert_connectivity_matches(record, segments, parents):
    """Decode references independently; compare rooted, site-labelled trees."""
    identities, decoded_units, decoded_edges, roots = {}, {}, [], []
    for segment_index, (segment, parent) in enumerate(zip(segments, parents, strict=True)):
        if parent is None:
            assert len(segment) == 1
            roots.append((segment_index, 0))
        else:
            assert parent[0] < segment_index
            assert 0 <= parent[1] < len(segments[parent[0]])
        for position, token in enumerate(segment):
            incoming, unit, outgoing = token.split(".")
            address = (segment_index, position)
            if position == 0 and parent is not None:
                identities[address] = identities[tuple(parent)]
                assert decoded_units[identities[address]] == unit
            else:
                identities[address] = address
                decoded_units[address] = unit
            if position:
                previous = segment[position - 1].split(".")
                decoded_edges.append((identities[(segment_index, position - 1)], previous[2], identities[address], incoming))
    assert Counter(decoded_units.values()) == Counter(unit for _, unit in record["occurrences"])
    assert len(decoded_edges) == len(record["edges"])

    def signature(units, edges, roots):
        neighbors = {node: [] for node in units}
        for left, left_site, right, right_site in edges:
            neighbors[left].append((right, left_site, right_site))
            neighbors[right].append((left, right_site, left_site))
        result = []
        for root in roots:
            stack, parent, order = [root], {root: None}, []
            while stack:
                node = stack.pop()
                order.append(node)
                for neighbor, _, _ in neighbors[node]:
                    if neighbor != parent[node]:
                        parent[neighbor] = node
                        stack.append(neighbor)
            labels = {}
            for node in reversed(order):
                labels[node] = (units[node], tuple(sorted(
                    (source, target, labels[neighbor])
                    for neighbor, source, target in neighbors[node] if neighbor != parent[node]
                )))
            result.append(labels[root])
        return sorted(result)

    assert signature(decoded_units, decoded_edges, roots) == signature(
        dict(record["occurrences"]), [edge[:4] for edge in record["edges"]],
        [component["root"] for component in record["components"]],
    )


def test_grafts_start_at_existing_r1_junction_not_initiator():
    record = topology(["I0", "R0", "R1", "R1", "T1", "R2", "R3"], [
        [0, "1", 1, "1", "1", False], [1, "2", 2, "1", "1", False],
        [2, "3", 3, "1", "1", False], [3, "3", 4, "1", "1", False],
        [2, "2", 5, "1", "1", False], [5, "2", 6, "1", "1", False],
    ])
    segments, parents = g2rins.render_bond_sequence_branches(record)
    assert segments == [
        ["null.I0.null"],
        ["null.I0.1", "1.R0.2", "1.R1.3", "1.R1.3", "1.T1.null"],
        ["1.R1.2", "1.R2.2", "1.R3.null"],
    ]
    assert parents == [None, [0, 0], [1, 2]]
    assert_connectivity_matches(record, segments, parents)


def test_multiple_identical_junctions_and_nested_branches_are_unambiguous():
    record = topology(["I0", "R1", "R1", "T1", "R1", "R1", "R2", "R2"], [
        [0, "1", 1, "1", "1", False], [1, "3", 2, "1", "1", False],
        [2, "3", 3, "1", "1", False], [1, "2", 4, "1", "1", False],
        [2, "2", 5, "1", "1", False], [4, "3", 6, "1", "1", False],
        [4, "2", 7, "1", "1", False],
    ])
    segments, parents = g2rins.render_bond_sequence_branches(record)
    assert_connectivity_matches(record, segments, parents)
    assert sum(len(segment) - (parent is not None) for segment, parent in zip(segments, parents)) == 8
    assert sum(token.count("I0") for segment in segments for token in segment) == 2
    assert [1, 1] in parents and [1, 2] in parents


def test_star_forest_and_isolated_root():
    record = topology(["I0"] + ["R0"] * 6 + ["T0"], [
        [0, str(site), site, "1", "1", False] for site in range(1, 7)
    ], [{"root": 0, "occurrence_ids": list(range(7))}, {"root": 7, "occurrence_ids": [7]}])
    segments, parents = g2rins.render_bond_sequence_branches(record)
    assert segments == [["null.I0.null"]] + [
        [f"null.I0.{site}", "1.R0.null"] for site in range(1, 7)
    ] + [["null.T0.null"]]
    assert parents == [None] + [[0, 0]] * 6 + [None]
    assert_connectivity_matches(record, segments, parents)


def test_reverse_root_swaps_connection_orientation():
    record = topology(["I0", "R0", "T0"], [
        [0, "2", 1, "1", "1", False], [1, "3", 2, "1", "1", False],
    ], [{"root": 2, "occurrence_ids": [0, 1, 2]}])
    segments, parents = g2rins.render_bond_sequence_branches(record)
    assert segments == [["null.T0.null"], ["null.T0.1", "3.R0.1", "2.I0.null"]]
    assert_connectivity_matches(record, segments, parents)


@pytest.mark.parametrize("edges", [
    [[0, "1", 0, "2", "1", False]],
    [[0, "1", 1, "1", "1", False], [0, "2", 1, "2", "1", False]],
    [[0, "1", 1, "1", "1", False], [1, "2", 2, "1", "1", False], [0, "2", 2, "2", "1", False]],
])
def test_closures_require_compact(edges):
    count = 1 + max(max(edge[0], edge[2]) for edge in edges)
    with pytest.raises(ValueError, match="Cyclic.*compact"):
        g2rins.render_bond_sequence_branches(topology(["R0"] * count, edges))


def test_long_chain_budget_and_empty_forest():
    record = topology(["R0"] * 2000, [
        [index, "2", index + 1, "1", "1", False] for index in range(1999)
    ])
    segments, parents = g2rins.render_bond_sequence_branches(record)
    assert list(map(len, segments)) == [1, 2000]
    with pytest.raises(ValueError, match="max_tokens"):
        g2rins.render_bond_sequence_branches(record, max_tokens=2000)
    assert g2rins.render_bond_sequence_branches(topology([], [], [])) == ([], [])


@pytest.mark.parametrize("selection", [{}, {"representative_distance": 0.15}, {"reservoir_size": 1}, {"reservoir_size": 0}])
def test_native_branches_preserve_sampling_retention_and_connectivity(selection):
    source = "{[] [<]CC([>])c1ccccc1; CO[>]; [<][H] []}|gauss(1000, 45)|"
    kwargs = dict(batch_size=3, max_samples=3, window=10, output_format="smiles",
                  metadata=True, retain_bond_sequences=True, seed=7, **selection)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(source).get_graph_creator().get_ensemble_creator()
        compact = creator.create_ensemble_until_converged(**kwargs)
        branches = creator.create_ensemble_until_converged(**kwargs, bond_sequence_format="branches")
    assert branches.chains == compact.chains
    assert branches.molecular_weights == compact.molecular_weights
    assert branches.bond_sequence_indices == compact.bond_sequence_indices
    assert branches.bond_sequence_weights == compact.bond_sequence_weights
    assert compact.bond_sequence_parents is None
    assert branches.bond_sequence_metadata == bond_sequence_branch_metadata(compact.bond_sequence_metadata)
    for record, segments, parents in zip(compact.bond_sequences, branches.bond_sequences, branches.bond_sequence_parents, strict=True):
        assert (segments, parents) == g2rins.render_bond_sequence_branches(record)
        assert_connectivity_matches(record, segments, parents)