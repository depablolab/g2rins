import warnings

import pytest

import g2rins

from g2rins import render_bond_sequence_paths
from g2rins.bond_sequences import bond_sequence_path_metadata


def topology(units, edges, components=None):
    return {
        "schema": "bond-sequence-topology/v1",
        "occurrences": list(enumerate(units)),
        "edges": edges,
        "components": components if components is not None else [
            {"root": 0, "occurrence_ids": list(range(len(units)))}
        ],
    }


def test_linear_paths_and_reverse_orientation():
    record = topology(["I0", "R0", "T0"], [
        [0, "2", 1, "1", "1", False], [1, "3", 2, "1", "1", False],
    ])
    assert render_bond_sequence_paths(record) == [["null.I0.2", "1.R0.3", "1.T0.null"]]
    record["components"][0]["root"] = 2
    assert render_bond_sequence_paths(record) == [["null.T0.1", "3.R0.1", "2.I0.null"]]


def test_graft_paths_repeat_shared_trunks_with_distinct_outgoing_sites():
    record = topology(["I0", "R0", "T0", "R1"], [
        [0, "1", 1, "1", "1", False], [1, "3", 2, "1", "1", False],
        [1, "2", 3, "1", "1", False],
    ])
    assert render_bond_sequence_paths(record) == [
        ["null.I0.1", "1.R0.2", "1.R1.null"],
        ["null.I0.1", "1.R0.3", "1.T0.null"],
    ]


def test_six_arm_initiator_and_disconnected_singleton():
    record = topology(["I0"] + ["R0"] * 6 + ["T0"], [
        [0, str(site), site, "1", "1", False] for site in range(1, 7)
    ], [{"root": 0, "occurrence_ids": list(range(7))}, {"root": 7, "occurrence_ids": [7]}])
    assert render_bond_sequence_paths(record) == [
        [f"null.I0.{site}", "1.R0.null"] for site in range(1, 7)
    ] + [["null.T0.null"]]


@pytest.mark.parametrize("edges", [
    [[0, "1", 0, "2", "1", False]],
    [[0, "1", 1, "1", "1", False], [0, "2", 1, "2", "1", False]],
    [[0, "1", 1, "1", "1", False], [1, "2", 2, "1", "1", False], [0, "2", 2, "2", "1", False]],
])
def test_cycles_are_not_silently_linearized(edges):
    count = 1 + max(max(edge[0], edge[2]) for edge in edges)
    with pytest.raises(ValueError, match="Cyclic.*compact"):
        render_bond_sequence_paths(topology(["R0"] * count, edges))


def test_iterative_long_chain_and_expansion_budget():
    record = topology(["R0"] * 2000, [
        [index, "2", index + 1, "1", "1", False] for index in range(1999)
    ])
    assert len(render_bond_sequence_paths(record)[0]) == 2000
    with pytest.raises(ValueError, match="max_tokens"):
        render_bond_sequence_paths(record, max_tokens=1999)


def test_path_metadata_preserves_weights_and_model_identity():
    metadata = {"format": "compact", "model_fingerprint": "abc", "accepted_count": 12}
    rendered = bond_sequence_path_metadata(metadata)
    assert rendered["format"] == "paths"
    assert rendered["accepted_count"] == 12
    assert rendered["model_fingerprint"] == "abc"
    assert rendered["topology_preserved"] is False
    assert metadata["format"] == "compact"


@pytest.mark.parametrize("selection", [{}, {"representative_distance": 0.15}, {"reservoir_size": 1}])
def test_native_path_format_preserves_sampling_and_retention(selection):
    source = "{[] [<]CC([>])c1ccccc1; CO[>]; [<][H] []}|gauss(1000, 45)|"
    kwargs = dict(
        batch_size=3, max_samples=3, window=10, output_format="smiles",
        metadata=True, retain_bond_sequences=True, seed=7, **selection,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(source).get_graph_creator().get_ensemble_creator()
        compact = creator.create_ensemble_until_converged(**kwargs)
        paths = creator.create_ensemble_until_converged(**kwargs, bond_sequence_format="paths")
    assert paths.chains == compact.chains
    assert paths.molecular_weights == compact.molecular_weights
    assert paths.bond_sequence_indices == compact.bond_sequence_indices
    assert paths.bond_sequence_weights == compact.bond_sequence_weights
    assert paths.bond_sequences == [render_bond_sequence_paths(record) for record in compact.bond_sequences]
    assert paths.bond_sequence_metadata == bond_sequence_path_metadata(compact.bond_sequence_metadata)