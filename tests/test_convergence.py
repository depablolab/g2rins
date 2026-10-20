# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for opt-in ensemble sampling until statistical convergence."""

import copy
import pickle
import warnings
from collections import Counter

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

import g2rins
import g2rins.ensemble_creator as ensemble_creator_module
from g2rins.convergence import ConvergenceTracker
from g2rins.ensemble_creator import (
    EnsembleCreator,
    _CompactMetadata,
    _DeferredChainRecord,
    _PartialAtomGraph,
    _SampledMolecule,
    _UnitOccurrence,
    _defer_chain_conversion,
    _materialize_deferred_chain,
)


FAST_SMI = "{[] [<]CC([>])c1ccccc1; CO[>]; [<][H] []}|gauss(1000, 45)|"


def _deferred(index, molecular_weight=100.0):
    pair = (("R0.1", "left"), ("R0.2", "right"))
    sample = _SampledMolecule(
        graph=nx.Graph(),
        metadata=_CompactMetadata(
            unit_counts={"R0": 1},
            bond_counts={},
            labeled_bond_counts={pair: 1},
            occurrences=(),
            sequences=(),
            unit_prototypes={},
        ),
        mol_weights={0: [molecular_weight]},
        distributions={0: "uniform(100,100)"},
    )
    return _DeferredChainRecord(sample, molecular_weight, index, None, None)


def test_deferred_conversion_uses_reconciled_sampler_weight(monkeypatch):
    deferred = _deferred(3, molecular_weight=123.5)
    deferred.sample.molecular_weight = 123.5

    def unexpected_rdkit_conversion(*_args, **_kwargs):
        raise AssertionError("tracked mass should avoid RDKit conversion")

    monkeypatch.setattr(
        ensemble_creator_module,
        "mol_graph_to_rdkit_mol",
        unexpected_rdkit_conversion,
    )

    result = _defer_chain_conversion(deferred.sample, collect_info=True)

    assert result.molecular_weight == 123.5


def test_worker_materialized_deferred_output_is_reused(monkeypatch):
    deferred = _deferred(4, molecular_weight=91.0)
    deferred.sample.molecular_weight = 91.0
    conversions = []

    def convert_once(*args, **kwargs):
        conversions.append((args, kwargs))
        return "materialized"

    monkeypatch.setattr(ensemble_creator_module, "_convert_chain", convert_once)
    result = _defer_chain_conversion(
        deferred.sample,
        collect_info=True,
        molecule_format="smiles",
        materialize_output=True,
    )

    assert _materialize_deferred_chain(result, "smiles", True, False) == "materialized"
    assert len(conversions) == 1


def test_tracker_requires_complete_stable_window():
    tracker = ConvergenceTracker(window=2, mass_tolerance=0.01, contact_tolerance=0.01)
    tracker.record(2, 100.0, 110.0, {"R0.1|R0.2": 1.0})
    tracker.record(4, 100.5, 110.5, {"R0.1|R0.2": 0.995, "R0.2|T0.1": 0.005})
    assert not tracker.converged()
    tracker.record(6, 100.6, 110.6, {"R0.1|R0.2": 0.996, "R0.2|T0.1": 0.004})
    assert tracker.converged()


@pytest.mark.parametrize(
    "kwargs, match",
    (
        ({"batch_size": 0}, "batch_size"),
        ({"max_samples": 0}, "max_samples"),
        ({"window": 0}, "window"),
        ({"mass_tolerance": -0.1}, "mass_tolerance"),
        ({"contact_tolerance": -0.1}, "contact_tolerance"),
        ({"checkpoint_policy": "compact"}, "checkpoint_policy"),
        ({"representative_distance": -0.1}, "representative_distance"),
        (
            {"reservoir_size": 1, "representative_distance": 0.15},
            "mutually exclusive",
        ),
    ),
)
def test_converged_ensemble_validates_settings(kwargs, match):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    with pytest.raises(ValueError, match=match):
        creator.create_ensemble_until_converged(**kwargs)


@pytest.mark.parametrize(
    "kwargs, exc, match",
    (
        ({"count_bond_sequences": True}, NotImplementedError, "Bond sequence export"),
        ({"bond_sequence_format": "wrong"}, ValueError, "bond_sequence_format"),
        (
            {"retain_bond_sequences": True, "metadata": False},
            ValueError,
            "metadata=True",
        ),
        (
            {"retain_bond_sequences": True, "parallel": True},
            NotImplementedError,
            "parallel mode",
        ),
        (
            {
                "retain_bond_sequences": True,
                "checkpoint_callback": lambda _checkpoint: None,
                "seed": 7,
            },
            NotImplementedError,
            "checkpoints or resume",
        ),
    ),
)
def test_bond_sequence_options_are_guarded(kwargs, exc, match, monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    sample_calls = []
    monkeypatch.setattr(
        creator,
        "_iter_chain_records",
        lambda **_kwargs: sample_calls.append(True),
        raising=False,
    )
    with pytest.raises(exc, match=match):
        creator.create_ensemble_until_converged(**kwargs)
    assert sample_calls == []


def test_bond_sequence_resume_checkpoint_is_rejected_before_sampling(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    sample_calls = []
    monkeypatch.setattr(
        creator,
        "_iter_chain_records",
        lambda **_kwargs: sample_calls.append(True),
        raising=False,
    )
    checkpoint = object.__new__(ensemble_creator_module.ConvergenceCheckpoint)

    with pytest.raises(NotImplementedError, match="checkpoints or resume"):
        creator.create_ensemble_until_converged(
            batch_size=1,
            max_samples=1,
            window=1,
            seed=7,
            checkpoint=checkpoint,
            retain_bond_sequences=True,
        )
    assert sample_calls == []


def test_serial_retained_motif_membership_is_materialized(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)

    def iter_chain_records(**kwargs):
        sample = _SampledMolecule(
            graph=nx.path_graph(2),
            metadata=_CompactMetadata(
                unit_counts={"R0": 2},
                bond_counts={},
                labeled_bond_counts={},
                occurrences=(
                    _UnitOccurrence(
                        unit_id="R0",
                        prototype_key=("R0", ("0",)),
                        nodes=(0,),
                        incoming_connection=None,
                        incoming_edge_token=None,
                        connections=[],
                    ),
                    _UnitOccurrence(
                        unit_id="R0",
                        prototype_key=("R0", ("1",)),
                        nodes=(1,),
                        incoming_connection=(0, 1),
                        incoming_edge_token=("1", False, "1", "1"),
                        connections=[],
                    ),
                ),
                sequences=(),
                unit_prototypes={},
                inter_occurrence_edges=(),
            ),
            mol_weights={0: [250.0]},
            distributions={0: "uniform(250,250)"},
        )
        yield {
            "chain_index": 0,
            "record": _DeferredChainRecord(sample, 250.0, 0, None, None),
            "discards": 0,
            "reasons": (),
            "first_cause": None,
            "warnings": [],
        }

    star_graph = nx.Graph()
    star_graph.add_node(0, atomic_num=6, charge=0, aromatic=False)
    creator._canonical_unit_info = {
        "R0": {
            "psmiles": "unit-a",
            "expected_maps": (1,),
            "expected_real_atom_count": 1,
            "_star_graph": star_graph,
        }
    }
    creator._canonical_unit_prototypes = {"R0": star_graph}
    creator._unit_graph_with_stars = staticmethod(lambda _graph, _origin_bond_id: star_graph)
    creator._validate_unit_psmiles = lambda *args, **kwargs: None
    creator._origin_bond_id = {}
    creator._generative_graph = nx.Graph()
    creator._unit_id_by_origin = {}
    creator._unit_subgraph_cache = {"R0": nx.Graph()}
    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    monkeypatch.setattr(
        ensemble_creator_module,
        "_materialize_deferred_chain",
        lambda *_args, **_kwargs: pytest.fail(
            "motif-only retention must not materialize chain or sequence output"
        ),
    )
    result = creator.create_ensemble_until_converged(
        batch_size=1,
        max_samples=1,
        window=1,
        metadata=True,
        retain_motif_membership=True,
        retain_chains=False,
        retain_sequences=False,
        seed=7,
    )

    assert result.motif_population_statistics["schema"] == "motif-population-statistics/v1"
    assert result.motif_population_statistics["accepted_count"] == 1
    assert result.chain_motif_index["population"] == "retained-samples"
    assert result.chain_motif_index["rows"][0]["chain_index"] == 0
    assert result.chain_motif_index["rows"][0]["molar_mass"] == 250.0
    assert result.chain_motif_index["rows"][0]["motif_counts"]


def test_serial_retained_motif_membership_reuses_budgeted_path_counts(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)

    def iter_chain_records(**kwargs):
        sample = _SampledMolecule(
            graph=nx.path_graph(2),
            metadata=_CompactMetadata(
                unit_counts={"R0": 2},
                bond_counts={},
                labeled_bond_counts={},
                occurrences=(
                    _UnitOccurrence(
                        unit_id="R0",
                        prototype_key=("R0", ("0",)),
                        nodes=(0,),
                        incoming_connection=None,
                        incoming_edge_token=None,
                        connections=[],
                    ),
                    _UnitOccurrence(
                        unit_id="R0",
                        prototype_key=("R0", ("1",)),
                        nodes=(1,),
                        incoming_connection=(0, 1),
                        incoming_edge_token=("1", False, "1", "1"),
                        connections=[],
                    ),
                ),
                sequences=(),
                unit_prototypes={},
                inter_occurrence_edges=(),
            ),
            mol_weights={0: [250.0]},
            distributions={0: "uniform(250,250)"},
        )
        yield {
            "chain_index": 0,
            "record": _DeferredChainRecord(sample, 250.0, 0, None, None),
            "discards": 0,
            "reasons": (),
            "first_cause": None,
            "warnings": [],
        }

    calls = []
    original = ensemble_creator_module._unit_path_counts

    def tracked_counts(
        metadata,
        graph,
        max_motifs=None,
        junction_counts=None,
        *,
        occurrence_view=None,
    ):
        calls.append(max_motifs)
        return original(
            metadata,
            graph,
            max_motifs=max_motifs,
            junction_counts=junction_counts,
            occurrence_view=occurrence_view,
        )

    star_graph = nx.Graph()
    star_graph.add_node(0, atomic_num=6, charge=0, aromatic=False)
    creator._canonical_unit_info = {
        "R0": {
            "psmiles": "unit-a",
            "expected_maps": (1,),
            "expected_real_atom_count": 1,
            "_star_graph": star_graph,
        }
    }
    creator._canonical_unit_prototypes = {"R0": star_graph}
    creator._unit_graph_with_stars = staticmethod(lambda _graph, _origin_bond_id: star_graph)
    creator._validate_unit_psmiles = lambda *args, **kwargs: None
    creator._origin_bond_id = {}
    creator._generative_graph = nx.Graph()
    creator._unit_id_by_origin = {}
    creator._unit_subgraph_cache = {"R0": nx.Graph()}
    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    monkeypatch.setattr(ensemble_creator_module, "_unit_path_counts", tracked_counts)

    creator.create_ensemble_until_converged(
        batch_size=1,
        max_samples=1,
        window=1,
        metadata=True,
        retain_motif_membership=True,
        retain_chains=False,
        retain_sequences=False,
        unit_path_max_motifs=2,
        seed=7,
    )

    assert calls
    assert all(call == 2 for call in calls)


@pytest.mark.parametrize("max_samples", [1, 2])
def test_serial_retained_bond_sequences_are_materialized(
    monkeypatch, max_samples
):
    creator = EnsembleCreator.__new__(EnsembleCreator)

    def unexpected_size_accounting(_payload):
        pytest.fail("Bond-sequence retention must not serialize records for byte budgets")

    monkeypatch.setattr(
        ensemble_creator_module,
        "_bond_sequence_record_encoded_size",
        unexpected_size_accounting,
    )

    def iter_chain_records(**kwargs):
        sample = _SampledMolecule(
            graph=nx.Graph([(0, 1)]),
            metadata=_CompactMetadata(
                unit_counts={"R0": 2},
                bond_counts={},
                labeled_bond_counts={},
                occurrences=(
                    _UnitOccurrence(
                        unit_id="R0",
                        prototype_key=("R0", ("0",)),
                        nodes=(0,),
                        incoming_connection=None,
                        incoming_edge_token=None,
                        connections=[],
                    ),
                    _UnitOccurrence(
                        unit_id="R0",
                        prototype_key=("R0", ("1",)),
                        nodes=(1,),
                        incoming_connection=(0, 1),
                        incoming_edge_token=("1", False, "1", "1"),
                        connections=[],
                    ),
                ),
                sequences=(),
                unit_prototypes={},
                inter_occurrence_edges=(),
            ),
            mol_weights={0: [250.0]},
            distributions={0: "uniform(250,250)"},
        )
        for chain_index in range(kwargs["start_index"], kwargs["start_index"] + kwargs["n_samples"]):
            yield {
                "chain_index": chain_index,
                "record": _DeferredChainRecord(sample, 250.0, chain_index, None, None),
                "discards": 0,
                "reasons": (),
                "first_cause": None,
                "warnings": [],
            }

    star_graph = nx.Graph()
    star_graph.add_node(0, atomic_num=6, charge=0, aromatic=False)
    creator._canonical_unit_info = {
        "R0": {
            "psmiles": "unit-a",
            "expected_maps": (1,),
            "expected_real_atom_count": 1,
            "_star_graph": star_graph,
        }
    }
    creator._canonical_unit_prototypes = {"R0": star_graph}
    creator._unit_graph_with_stars = staticmethod(lambda _graph, _origin_bond_id: star_graph)
    creator._validate_unit_psmiles = lambda *args, **kwargs: None
    creator._origin_bond_id = {}
    creator._generative_graph = nx.Graph()
    creator._unit_id_by_origin = {}
    creator._unit_subgraph_cache = {"R0": nx.Graph()}
    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    occurrence_views = []
    build_occurrence_adjacency = ensemble_creator_module._build_occurrence_adjacency

    def tracked_occurrence_view(*args, **kwargs):
        view = build_occurrence_adjacency(*args, **kwargs)
        occurrence_views.append(view)
        return view

    monkeypatch.setattr(
        ensemble_creator_module,
        "_build_occurrence_adjacency",
        tracked_occurrence_view,
    )
    result = creator.create_ensemble_until_converged(
        batch_size=max_samples,
        max_samples=max_samples,
        window=1,
        metadata=True,
        retain_bond_sequences=True,
        seed=7,
    )
    assert len(occurrence_views) == max_samples

    assert result.bond_sequence_metadata["schema"] == "bond-sequence-topology/v1"
    assert result.bond_sequence_metadata["retention"] == "all"
    assert result.bond_sequence_metadata["accepted_count"] == max_samples
    assert result.bond_sequence_indices == list(range(max_samples))
    assert result.bond_sequence_weights == [1] * max_samples
    assert len(result.bond_sequences) == max_samples
    for record in result.bond_sequences:
        assert record["schema"] == "bond-sequence-topology/v1"
        assert set(record) == {"schema", "occurrences", "edges", "components"}
        # Occurrence IDs identify local instances of the same building-block type.
        assert record["occurrences"] == [[0, "R0"], [1, "R0"]]
        assert record["edges"] == [[0, "1", 1, "1", "1", False]]
        assert record["components"] == [{"root": 0, "occurrence_ids": [0, 1]}]


def test_bond_only_retention_works_without_legacy_sequence_flags(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)

    def iter_chain_records(**kwargs):
        sample = _SampledMolecule(
            graph=nx.Graph([(0, 1)]),
            metadata=_CompactMetadata(
                unit_counts={"R0": 2},
                bond_counts={},
                labeled_bond_counts={},
                occurrences=(
                    _UnitOccurrence(
                        unit_id="R0",
                        prototype_key=("R0", ("0",)),
                        nodes=(0,),
                        incoming_connection=None,
                        incoming_edge_token=None,
                        connections=[],
                    ),
                    _UnitOccurrence(
                        unit_id="R0",
                        prototype_key=("R0", ("1",)),
                        nodes=(1,),
                        incoming_connection=(0, 1),
                        incoming_edge_token=("1", False, "1", "1"),
                        connections=[],
                    ),
                ),
                sequences=(),
                unit_prototypes={},
                inter_occurrence_edges=(),
            ),
            mol_weights={0: [250.0]},
            distributions={0: "uniform(250,250)"},
        )
        yield {
            "chain_index": 0,
            "record": _DeferredChainRecord(sample, 250.0, 0, None, None),
            "discards": 0,
            "reasons": (),
            "first_cause": None,
            "warnings": [],
        }

    star_graph = nx.Graph()
    star_graph.add_node(0, atomic_num=6, charge=0, aromatic=False)
    creator._canonical_unit_info = {
        "R0": {
            "psmiles": "unit-a",
            "expected_maps": (1,),
            "expected_real_atom_count": 1,
            "_star_graph": star_graph,
        }
    }
    creator._canonical_unit_prototypes = {"R0": star_graph}
    creator._unit_graph_with_stars = staticmethod(lambda _graph, _origin_bond_id: star_graph)
    creator._validate_unit_psmiles = lambda *args, **kwargs: None
    creator._origin_bond_id = {}
    creator._generative_graph = nx.Graph()
    creator._unit_id_by_origin = {}
    creator._unit_subgraph_cache = {"R0": nx.Graph()}
    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    monkeypatch.setattr(
        ensemble_creator_module,
        "_materialize_deferred_chain",
        lambda *_args, **_kwargs: pytest.fail(
            "bond-only retention must not materialize chain or sequence output"
        ),
    )
    result = creator.create_ensemble_until_converged(
        batch_size=1,
        max_samples=1,
        window=1,
        metadata=True,
        retain_chains=False,
        retain_sequences=False,
        retain_bond_sequences=True,
        seed=7,
    )

    assert result.bond_sequence_metadata["schema"] == "bond-sequence-topology/v1"
    assert result.bond_sequence_indices == [0]
    assert result.bond_sequence_weights == [1]
    assert len(result.bond_sequences) == 1
    assert result.chains == []
    assert result.sequences == []
    assert result.molecular_weights == [250.0]


def _topology_occurrence(unit_id, nodes, incoming_connection=None, edge_token=None):
    return _UnitOccurrence(
        unit_id=unit_id,
        prototype_key=(unit_id, tuple(map(str, nodes))),
        nodes=tuple(nodes),
        incoming_connection=incoming_connection,
        incoming_edge_token=edge_token,
        connections=[],
    )


def test_compact_topology_normalizes_reversed_edges_and_preserves_multiplicity():
    occurrences = (
        _topology_occurrence("R0", (0,)),
        _topology_occurrence("R0", (1,)),
    )
    unit_info = {"R0": {"expected_maps": (1, 2), "psmiles": "[C:1][C:2]"}}

    def make_record(edges):
        metadata = _CompactMetadata(
            unit_counts={"R0": 2},
            bond_counts={},
            labeled_bond_counts={},
            occurrences=occurrences,
            sequences=(),
            unit_prototypes={},
            inter_occurrence_edges=edges,
        )
        return ensemble_creator_module._compact_bond_sequence_record(
            metadata,
            nx.Graph([(0, 1)]),
            unit_info=unit_info,
        )

    forward = make_record(((0, 1, ("1", False, "1", "2")),))
    reversed_edge = make_record(((1, 0, ("1", False, "2", "1")),))
    assert forward["edges"] == reversed_edge["edges"] == [
        [0, "1", 1, "2", "1", False]
    ]

    multigraph_like = make_record(
        (
            (1, 0, ("1", False, "2", "1")),
            (0, 1, ("2", False, "2", "1")),
            (0, 0, ("1", False, "1", "2")),
        )
    )
    assert multigraph_like["edges"] == [
        [0, "1", 0, "2", "1", False],
        [0, "1", 1, "2", "1", False],
        [0, "2", 1, "1", "2", False],
    ]
    assert multigraph_like["components"] == [
        {"root": 0, "occurrence_ids": [0, 1]}
    ]


def test_bond_sequence_fingerprint_tracks_model_and_unit_site_identity():
    payloads = []

    def fingerprint(unit_info, model_identity="model"):
        return ensemble_creator_module._bond_sequence_export_metadata(
            payloads,
            retention="all",
            weight_semantics="retained-observation",
            retained_payload_status="available",
            accepted_count=0,
            unit_info=unit_info,
            model_identity=model_identity,
        )["model_fingerprint"]

    original = {"R0": {"expected_maps": (1, 2), "psmiles": "[C:1][C:2]"}}
    assert fingerprint(original) == fingerprint(
        {"R0": {"expected_maps": (2, 1), "psmiles": "[C:1][C:2]"}}
    )
    assert fingerprint(original) != fingerprint(original, model_identity="other model")
    assert fingerprint(original) != fingerprint(
        {"R0": {"expected_maps": (1,), "psmiles": "[C:1][C:2]"}}
    )
    assert fingerprint(original) != fingerprint(
        {"R0": {"expected_maps": (1, 2), "psmiles": "[C:1]=[C:2]"}}
    )


def _independent_phantom_port_rows(creator, sample, pre_collapse_graph):
    """Derive cross-occurrence ports from finalized ownership and raw phantom endpoints."""
    owner_by_atom = {
        atom: occurrence_id
        for occurrence_id, occurrence in enumerate(sample.metadata.occurrences)
        for atom in occurrence.nodes
    }
    phantom_nodes = {
        node
        for node, data in pre_collapse_graph.nodes(data=True)
        if data.get("atomic_num") == 0
    }
    rows = []
    for phantom_component in nx.connected_components(
        pre_collapse_graph.subgraph(phantom_nodes)
    ):
        endpoints = []
        for phantom in phantom_component:
            for atom in pre_collapse_graph.neighbors(phantom):
                if atom not in phantom_nodes:
                    endpoints.append((phantom, atom))
        if len(endpoints) < 2:
            continue

        endpoint_rows = []
        for phantom, atom in endpoints:
            occurrence_id = owner_by_atom[atom]
            unit_id = sample.metadata.occurrences[occurrence_id].unit_id
            expected_maps = creator._canonical_unit_info[unit_id]["expected_maps"]
            candidate_sites = set()
            for candidate in (atom, phantom):
                origin = str(pre_collapse_graph.nodes[candidate].get("origin_idx"))
                if creator._origin_unit_id.get(origin) != unit_id:
                    continue
                site = creator._origin_bond_id.get(origin)
                if site in expected_maps:
                    candidate_sites.add(site)
            if len(candidate_sites) != 1:
                raise AssertionError(
                    f"Expected one independently attributable port for {unit_id}: {candidate_sites}"
                )
            endpoint_rows.append((occurrence_id, str(candidate_sites.pop())))

        phantom_edges = [
            data
            for left, right, data in pre_collapse_graph.subgraph(
                phantom_component
            ).edges(data=True)
        ]
        if phantom_edges:
            bond_attributes = phantom_edges[0]
            assert all(data == bond_attributes for data in phantom_edges)
        else:
            external_edges = [
                pre_collapse_graph[phantom][atom]
                for phantom, atom in endpoints
            ]
            bond_attributes = external_edges[0]
            assert all(data == bond_attributes for data in external_edges)

        for left_index in range(len(endpoint_rows)):
            for right_index in range(left_index + 1, len(endpoint_rows)):
                left_id, left_site = endpoint_rows[left_index]
                right_id, right_site = endpoint_rows[right_index]
                if left_id == right_id:
                    continue
                if (left_id, left_site) > (right_id, right_site):
                    left_id, right_id = right_id, left_id
                    left_site, right_site = right_site, left_site
                rows.append(
                    [
                        left_id,
                        left_site,
                        right_id,
                        right_site,
                        str(bond_attributes["bond_type"]),
                        bool(bond_attributes["aromatic"]),
                    ]
                )
    return sorted(rows)


def test_native_branched_topology_matches_independent_phantom_oracle(monkeypatch):
    source = "{[] [<][CH]([>])[>]; [<][H]; [>][H] []}|uniform(65,65)|"
    creator = g2rins.G2rins.make(source).get_graph_creator().get_ensemble_creator()
    snapshots = []
    collapse = ensemble_creator_module._collapse_phantom_nodes

    def capture_before_collapse(graph):
        snapshots.append(copy.deepcopy(graph))
        return collapse(graph)

    monkeypatch.setattr(
        ensemble_creator_module, "_collapse_phantom_nodes", capture_before_collapse
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample = creator.sample_mol_graph(
            rng=np.random.default_rng(0),
            termination_flag=1,
            _metadata_mode=ensemble_creator_module._MetadataLevel.COUNTS,
        )

    assert len(snapshots) == 1
    pre_collapse_graph = snapshots[0]
    assert sum(
        data.get("atomic_num") == 0
        for _node, data in pre_collapse_graph.nodes(data=True)
    ) == 12
    assert all(data.get("atomic_num") != 0 for _node, data in sample.graph.nodes(data=True))
    assert sum(
        data.get("atomic_num") == 1
        for _node, data in sample.graph.nodes(data=True)
    ) == 2

    oracle_rows = _independent_phantom_port_rows(
        creator, sample, pre_collapse_graph
    )
    expected_rows = [
        [0, "1", 1, "1", "1", False],
        [1, "2", 2, "3", "1", False],
        [1, "3", 5, "1", "1", False],
        [2, "2", 3, "3", "1", False],
        [3, "1", 4, "3", "1", False],
    ]
    assert oracle_rows == expected_rows

    owner_by_atom = {
        atom: occurrence_id
        for occurrence_id, occurrence in enumerate(sample.metadata.occurrences)
        for atom in occurrence.nodes
    }
    final_cross_edges = sorted(
        (
            min(owner_by_atom[left], owner_by_atom[right]),
            max(owner_by_atom[left], owner_by_atom[right]),
            str(data["bond_type"]),
            bool(data["aromatic"]),
        )
        for left, right, data in sample.graph.edges(data=True)
        if owner_by_atom[left] != owner_by_atom[right]
    )
    oracle_cross_edges = sorted(
        (row[0], row[2], row[4], row[5]) for row in oracle_rows
    )
    assert final_cross_edges == oracle_cross_edges

    record = ensemble_creator_module._compact_bond_sequence_record(
        sample.metadata,
        sample.graph,
        unit_info=creator._canonical_unit_info,
    )
    assert set(record) == {"schema", "occurrences", "edges", "components"}
    assert record["edges"] == expected_rows
    assert record["components"] == [{"root": 0, "occurrence_ids": [0, 1, 2, 3, 4, 5]}]
    envelope = ensemble_creator_module._bond_sequence_export_metadata(
        [record],
        retention="all",
        weight_semantics="retained-observation",
        retained_payload_status="available",
        accepted_count=1,
        unit_info=creator._canonical_unit_info,
        model_identity=source,
    )
    assert envelope["unit_sites"] == {"I0": [1], "R0": [1, 2, 3], "T0": [1]}
    assert len(envelope["model_fingerprint"]) == 64
    reparsed_creator = (
        g2rins.G2rins.make(source).get_graph_creator().get_ensemble_creator()
    )
    reparsed_envelope = ensemble_creator_module._bond_sequence_export_metadata(
        [],
        retention="all",
        weight_semantics="retained-observation",
        retained_payload_status="available",
        accepted_count=1,
        unit_info=reparsed_creator._canonical_unit_info,
        model_identity=source,
    )
    assert envelope["model_fingerprint"] == reparsed_envelope["model_fingerprint"]
    for unit_id, sites in envelope["unit_sites"].items():
        psmiles = creator._canonical_unit_info[unit_id]["psmiles"]
        mol = Chem.MolFromSmiles(psmiles)
        assert mol is not None
        mapped_sites = sorted(
            atom.GetAtomMapNum()
            for atom in mol.GetAtoms()
            if atom.GetAtomMapNum() > 0
        )
        assert mapped_sites == sites


def test_native_intrinsic_ring_stays_inside_cyclic_unit_topology():
    source = "{[] [<]C1CC1[>]; [<][H]; [>][H] []}|uniform(65,65)|"
    creator = g2rins.G2rins.make(source).get_graph_creator().get_ensemble_creator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample = creator.sample_mol_graph(
            rng=np.random.default_rng(0),
            termination_flag=1,
            _metadata_mode=ensemble_creator_module._MetadataLevel.COUNTS,
        )

    owner_by_atom = {
        atom: occurrence_id
        for occurrence_id, occurrence in enumerate(sample.metadata.occurrences)
        for atom in occurrence.nodes
    }
    expected_rows = []
    for left, right, attributes in sample.graph.edges(data=True):
        left_owner = owner_by_atom[left]
        right_owner = owner_by_atom[right]
        if left_owner == right_owner:
            continue
        left_origin = str(sample.graph.nodes[left]["origin_idx"])
        right_origin = str(sample.graph.nodes[right]["origin_idx"])
        left_site = str(creator._origin_bond_id[left_origin])
        right_site = str(creator._origin_bond_id[right_origin])
        if (left_owner, left_site) > (right_owner, right_site):
            left_owner, right_owner = right_owner, left_owner
            left_site, right_site = right_site, left_site
        expected_rows.append(
            [
                left_owner,
                left_site,
                right_owner,
                right_site,
                str(attributes["bond_type"]),
                bool(attributes["aromatic"]),
            ]
        )
    expected_rows.sort()
    assert expected_rows == [
        [0, "1", 1, "2", "1", False],
        [1, "1", 2, "1", "1", False],
    ]
    assert len(nx.cycle_basis(sample.graph)) == 1

    record = ensemble_creator_module._compact_bond_sequence_record(
        sample.metadata,
        sample.graph,
        unit_info=creator._canonical_unit_info,
    )
    assert record["edges"] == expected_rows
    assert all(left != right for left, _left_site, right, *_rest in record["edges"])
    repeat_unit = Chem.MolFromSmiles(creator._canonical_unit_info["R0"]["psmiles"])
    assert repeat_unit is not None
    assert repeat_unit.GetRingInfo().NumRings() == 1
    assert record["components"] == [{"root": 0, "occurrence_ids": [0, 1, 2]}]


def test_native_counterion_association_is_not_a_covalent_export_edge():
    source = "{[] [<]CC(C[NH3+])[>].[Cl-]; C[>]; [<][H] []}|uniform(200,200)|"
    creator = g2rins.G2rins.make(source).get_graph_creator().get_ensemble_creator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample = creator.sample_mol_graph(
            rng=np.random.default_rng(0),
            termination_flag=1,
            _metadata_mode=ensemble_creator_module._MetadataLevel.COUNTS,
        )

    owner_by_atom = {
        atom: occurrence_id
        for occurrence_id, occurrence in enumerate(sample.metadata.occurrences)
        for atom in occurrence.nodes
    }
    association_edges = [
        (left, right, attrs)
        for left, right, attrs in sample.graph.edges(data=True)
        if attrs["bond_type"] == 0
    ]
    assert len(association_edges) == 1
    left, right, _attrs = association_edges[0]
    assert owner_by_atom[left] == owner_by_atom[right] == 1
    assert {sample.graph.nodes[left]["atomic_num"], sample.graph.nodes[right]["atomic_num"]} == {7, 17}

    record = ensemble_creator_module._compact_bond_sequence_record(
        sample.metadata,
        sample.graph,
        unit_info=creator._canonical_unit_info,
    )
    assert record["occurrences"] == [[0, "I0"], [1, "R0"], [2, "T0"]]
    assert record["edges"] == [
        [0, "1", 1, "1", "1", False],
        [1, "2", 2, "1", "1", False],
    ]
    assert all(row[4] != "0" for row in record["edges"])


def test_bond_sequence_component_uses_surviving_growth_root():
    metadata = _CompactMetadata(
        unit_counts={"R0": 2},
        bond_counts={},
        labeled_bond_counts={},
        occurrences=(
            _topology_occurrence("R0", (0,), (1, 0), ("1", False, "1", "1")),
            _topology_occurrence("R0", (1,)),
        ),
        sequences=(),
        unit_prototypes={},
        inter_occurrence_edges=(),
    )
    unit_info = {"R0": {"expected_maps": (1,)}}

    payload = ensemble_creator_module._compact_bond_sequence_record(
        metadata,
        nx.path_graph(2),
        unit_info=unit_info,
    )

    assert payload["components"] == [{"root": 1, "occurrence_ids": [0, 1]}]


def test_bond_sequence_category_zero_is_not_a_covalent_edge():
    metadata = _CompactMetadata(
        unit_counts={"R0": 2},
        bond_counts={},
        labeled_bond_counts={},
        occurrences=(
            _topology_occurrence("R0", (0,)),
            _topology_occurrence("R0", (1,), (0, 1), ("0", False, "1", "1")),
        ),
        sequences=(),
        unit_prototypes={},
        inter_occurrence_edges=(),
    )

    payload = ensemble_creator_module._compact_bond_sequence_record(
        metadata,
        nx.path_graph(2),
        unit_info={"R0": {"expected_maps": (1,)}},
    )

    assert payload["edges"] == []
    assert payload["components"] == [
        {"root": 0, "occurrence_ids": [0]},
        {"root": 1, "occurrence_ids": [1]},
    ]


def test_bond_sequence_rejects_site_missing_from_unit_dictionary():
    metadata = _CompactMetadata(
        unit_counts={"R0": 2},
        bond_counts={},
        labeled_bond_counts={},
        occurrences=(
            _topology_occurrence("R0", (0,)),
            _topology_occurrence("R0", (1,), (0, 1), ("1", False, "2", "1")),
        ),
        sequences=(),
        unit_prototypes={},
        inter_occurrence_edges=(),
    )

    with pytest.raises(ValueError, match="site absent from the unit map dictionary"):
        ensemble_creator_module._compact_bond_sequence_record(
            metadata,
            nx.path_graph(2),
            unit_info={"R0": {"expected_maps": (1,)}},
        )


def test_empty_bond_sequence_record_uses_exact_v1_wire_keys():
    metadata = _CompactMetadata(
        unit_counts={},
        bond_counts={},
        labeled_bond_counts={},
        occurrences=(),
        sequences=(),
        unit_prototypes={},
    )

    payload = ensemble_creator_module._compact_bond_sequence_record(
        metadata,
        nx.Graph(),
        unit_info={},
    )

    assert payload["occurrences"] == []
    assert payload["edges"] == []
    assert payload["components"] == []
    assert set(payload) == {"schema", "occurrences", "edges", "components"}


def test_bond_sequence_metadata_distinguishes_accepted_and_retained_counts():
    metadata = ensemble_creator_module._bond_sequence_export_metadata(
        [{"schema": "bond-sequence-topology/v1", "occurrences": [], "edges": [], "components": []}],
        retention="cover",
        weight_semantics="accepted-coverage",
        retained_payload_status="available",
        accepted_count=9,
        unit_info={"R0": {"expected_maps": (1,), "psmiles": "[C:1]"}},
    )

    assert metadata["accepted_count"] == 9


@pytest.mark.parametrize(
    "export_options, match",
    (
        ({"retain_bond_sequences": True}, "fixed-size create_ensemble"),
        ({"count_bond_sequences": True}, "exact counting is not implemented"),
    ),
)
def test_fixed_size_bond_sequence_export_is_rejected_before_sampling(
    export_options, match, monkeypatch
):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    sample_calls = []
    monkeypatch.setattr(
        creator,
        "_iter_chain_records",
        lambda **_kwargs: sample_calls.append(True),
        raising=False,
    )

    with pytest.raises(NotImplementedError, match=match):
        creator.create_ensemble(
            1,
            ensemble_info=True,
            seed=1,
            **export_options,
        )
    assert sample_calls == []


def test_retained_bond_sequences_succeed_without_budget_options():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        result = creator.create_ensemble_until_converged(
            batch_size=1,
            max_samples=2,
            window=10,
            output_format="smiles",
            metadata=True,
            retain_bond_sequences=True,
            seed=7,
        )

    assert result.bond_sequence_metadata["retention"] == "all"
    assert result.bond_sequence_metadata["accepted_count"] == 2
    assert result.bond_sequence_indices == [0, 1]
    assert result.bond_sequence_weights == [1, 1]
    assert len(result.bond_sequences) == len(result.chains) == 2
    assert len(result.sequences) == len(result.molecular_weights) == 2
    assert all(record["occurrences"] for record in result.bond_sequences)


def test_converged_ensemble_data_supports_bond_sequence_outputs():
    result = ensemble_creator_module.ConvergedEnsembleData(
        chains=[],
        units={},
        bonds=[],
        sequences=[],
        mol_weights={},
        distributions={},
        molecular_weights=[],
        converged=False,
        n_batches=0,
        convergence_settings={},
        convergence_trace=[],
        number_average_molecular_weight=0.0,
        weight_average_molecular_weight=0.0,
        dispersity=0.0,
        bond_sequence_metadata={"schema": "bond-sequence-topology/v1"},
        bond_sequences=[],
        bond_sequence_indices=[],
        bond_sequence_weights=[],
        bond_sequence_counts={},
    )

    assert result.bond_sequence_metadata["schema"] == "bond-sequence-topology/v1"
    assert result.bond_sequence_counts == {}


def test_create_ensemble_until_converged_stops_after_stable_window(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    calls = []

    def iter_chain_records(**kwargs):
        calls.append(kwargs)
        for index in range(kwargs["n_samples"]):
            yield {
                "chain_index": kwargs["start_index"] + index,
                "record": _deferred(
                    kwargs["start_index"] + index,
                    molecular_weight=17446.7398,
                ),
                "discards": 0,
                "reasons": (),
                "first_cause": None,
                "warnings": [],
            }

    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    progress = []
    result = creator.create_ensemble_until_converged(
        batch_size=2,
        max_samples=20,
        window=2,
        seed=10,
        progress_callback=progress.append,
    )

    assert result.converged
    assert result.n_batches == 3
    assert len(result.chains) == 6
    assert result.molecular_weights == [17446.7398] * 6
    assert result.units["R0"]["count"] == 6
    assert result.bonds[0]["count"] == 6
    assert [kwargs["seed"] for kwargs in calls] == [10, 12, 14]
    assert [kwargs["start_index"] for kwargs in calls] == [0, 2, 4]
    assert len(progress) == 3
    assert progress[0] == (
        "Batch | Samples |         Mn |         Mw | Status\n"
        "----- | ------- | ---------- | ---------- | ------\n"
        "    1 |       2 |    17446.7 |    17446.7 | "
        "warming up window (2 more batch(es) before convergence can be checked)"
    )
    assert progress[-1] == (
        "    3 |       6 |    17446.7 |    17446.7 | "
        "mass_delta=0.0000/0.0100 contact_delta=0.0000/0.0100"
    )


def test_create_ensemble_until_converged_honors_max_samples(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    requested_sizes = []

    def iter_chain_records(**kwargs):
        requested_sizes.append(kwargs["n_samples"])
        for index in range(kwargs["n_samples"]):
            yield {
                "chain_index": kwargs["start_index"] + index,
                "record": _deferred(kwargs["start_index"] + index),
                "discards": 0,
                "reasons": (),
                "first_cause": None,
                "warnings": [],
            }

    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    result = creator.create_ensemble_until_converged(
        batch_size=2,
        max_samples=5,
        window=10,
    )

    assert not result.converged
    assert len(result.chains) == 5
    assert result.n_batches == 3
    assert requested_sizes == [2, 2, 1]


def test_create_ensemble_until_converged_defaults(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)

    def iter_chain_records(**kwargs):
        for index in range(kwargs["n_samples"]):
            yield {
                "chain_index": kwargs["start_index"] + index,
                "record": _deferred(kwargs["start_index"] + index),
                "discards": 0,
                "reasons": (),
                "first_cause": None,
                "warnings": [],
            }

    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    result = creator.create_ensemble_until_converged(window=10_000)

    assert not result.converged
    assert len(result.chains) == 1500
    assert result.convergence_trace[-1]["n_samples"] == 1500
    assert result.convergence_settings["max_samples"] == 1500
    assert result.convergence_settings["mass_tolerance"] == 0.01


def test_create_ensemble_until_converged_uses_repeat_units_as_sources(
    monkeypatch,
):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    source_modes = []

    def iter_chain_records(**kwargs):
        source_modes.append(kwargs["use_repeat_units_as_source"])
        yield {
            "chain_index": 0,
            "record": _deferred(0),
            "discards": 0,
            "reasons": (),
            "first_cause": None,
            "warnings": [],
        }

    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    result = creator.create_ensemble_until_converged(
        batch_size=1,
        max_samples=1,
        window=1,
        use_repeat_units_as_source=True,
    )

    assert result is not None
    assert source_modes == [True]
    assert result.convergence_settings["use_repeat_units_as_source"] is True


def test_converged_ensemble_can_generate_without_an_initiator():
    smi = "{[] [<]CCC[>], [<]NNN[>]|2|; ; [<]Cl []}|poisson(400)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
        result = creator.create_ensemble_until_converged(
            batch_size=1,
            max_samples=1,
            window=1,
            output_format="smiles",
            seed=7,
            use_repeat_units_as_source=True,
        )

    assert result is not None
    assert len(result.chains) == 1
    assert result.chains[0]


def test_native_convergence_handles_branched_polythioester_without_initiator():
    smi = (
        "{[] [>]SCCSSCCS[>]|18|, [>]SCCS(=O)CCS[>]|36|, "
        "O=C(c1cc(C(=O)[<])cc(C(=O)[<])c1)[<]|15|, "
        "O=C(c1ccc(cc1)C(=O)[<])[<]|31|; ; [>]O, [<][H] []}"
        "|flory_schulz(0.000190656)|"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
        result = creator.create_ensemble_until_converged(
            batch_size=1,
            max_samples=1,
            window=1,
            output_format="smiles",
            seed=5,
            use_repeat_units_as_source=True,
            unit_junctions=True,
        )

    assert result is not None
    assert len(result.chains) == 1
    assert result.chains[0]
    assert result.unit_junction_statistics["diagnostics"]["motifs"] > 0


def test_ensemble_info_reports_final_chain_molecular_weights():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        result = creator.create_ensemble(2, ensemble_info=True, seed=7)

    expected = [
        Descriptors.MolWt(g2rins.mol_graph_to_rdkit_mol(chain))
        for chain in result.chains
    ]
    assert result.molecular_weights == pytest.approx(expected)


def test_real_ensemble_converges_and_returns_requested_format():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        result = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=1,
            mass_tolerance=1.0,
            contact_tolerance=1.0,
            output_format="smiles",
            seed=7,
        )

    assert result.converged
    assert result.n_batches == 2
    assert len(result.chains) == 4
    assert all(isinstance(chain, str) for chain in result.chains)
    assert result.convergence_trace[-1]["n_samples"] == 4


def test_convergence_can_stream_without_retaining_samples():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        full = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=19,
        )
        streamed = []
        bounded = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=19,
            retain_chains=False,
            retain_sequences=False,
            sample_callback=lambda index, record: streamed.append(
                (
                    index,
                    record["molecule"],
                    record["sequences"],
                    record["molecule_units"],
                )
            ),
        )

    assert [index for index, _chain, _sequences, _units in streamed] == list(range(6))
    assert [chain for _index, chain, _sequences, _units in streamed] == full.chains
    assert [
        sequences for _index, _chain, sequences, _units in streamed
    ] == full.sequences
    assert all(units for _index, _chain, _sequences, units in streamed)
    assert bounded.chains == []
    assert bounded.sequences == []
    assert bounded.molecular_weights == []
    assert {
        unit_id: data["count"] for unit_id, data in bounded.units.items()
    } == {
        unit_id: data["count"] for unit_id, data in full.units.items()
    }
    assert bounded.bonds == full.bonds
    assert bounded.convergence_trace == full.convergence_trace
    assert bounded.number_average_molecular_weight == pytest.approx(
        full.number_average_molecular_weight
    )
    assert bounded.weight_average_molecular_weight == pytest.approx(
        full.weight_average_molecular_weight
    )


def test_streamed_convergence_skips_batch_and_unretained_conversion(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    materialized = []

    def iter_chain_records(**kwargs):
        for index in range(kwargs["n_samples"]):
            assert not materialized
            chain_index = kwargs["start_index"] + index
            yield {
                "chain_index": chain_index,
                "record": _deferred(chain_index),
                "discards": 0,
                "reasons": (),
                "first_cause": None,
                "warnings": [],
            }

    def fail_batch_conversion(*_args, **_kwargs):
        raise AssertionError("legacy batch conversion must not run")

    def track_materialization(*args, **kwargs):
        materialized.append(args[0].chain_index)
        return original_materialize(*args, **kwargs)

    original_materialize = ensemble_creator_module._materialize_deferred_chain
    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    monkeypatch.setattr(
        creator,
        "_records_to_ensemble_data",
        fail_batch_conversion,
    )
    monkeypatch.setattr(
        ensemble_creator_module,
        "_materialize_deferred_chain",
        track_materialization,
    )

    result = creator.create_ensemble_until_converged(
        batch_size=3,
        max_samples=3,
        window=10,
        retain_chains=False,
        retain_sequences=False,
    )

    assert result.convergence_trace[-1]["n_samples"] == 3
    assert result.units["R0"]["count"] == 3
    assert result.bonds[0]["count"] == 3
    assert materialized == []


def test_convergence_callback_materializes_every_deferred_record(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    materialized = []
    callbacks = []

    def iter_chain_records(**kwargs):
        for index in range(kwargs["n_samples"]):
            chain_index = kwargs["start_index"] + index
            yield {
                "chain_index": chain_index,
                "record": _deferred(chain_index),
                "discards": 0,
                "reasons": (),
                "first_cause": None,
                "warnings": [],
            }

    def track_materialization(*args, **kwargs):
        materialized.append(args[0].chain_index)
        return original_materialize(*args, **kwargs)

    original_materialize = ensemble_creator_module._materialize_deferred_chain
    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    monkeypatch.setattr(
        ensemble_creator_module,
        "_materialize_deferred_chain",
        track_materialization,
    )

    result = creator.create_ensemble_until_converged(
        batch_size=3,
        max_samples=3,
        window=10,
        retain_chains=False,
        retain_sequences=False,
        sample_callback=lambda index, record: callbacks.append((index, record)),
    )

    assert result.convergence_trace[-1]["n_samples"] == 3
    assert materialized == [0, 1, 2]
    assert [index for index, _record in callbacks] == [0, 1, 2]
    assert all("sequences" in record for _index, record in callbacks)


def test_counts_only_convergence_skips_legacy_metadata(monkeypatch):
    creator = (
        g2rins.G2rins.make(FAST_SMI)
        .get_graph_creator()
        .get_ensemble_creator()
    )

    def fail_legacy_materialization(_graph):
        raise AssertionError("counts-only convergence materialized legacy metadata")

    monkeypatch.setattr(
        _PartialAtomGraph,
        "materialize_legacy_metadata",
        fail_legacy_materialization,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=4,
            window=10,
            seed=29,
            retain_chains=False,
            retain_sequences=False,
        )

    assert result.units
    assert result.bonds
    assert result.chains == []
    assert result.sequences == []


def test_convergence_reservoir_is_bounded_and_generation_independent():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        full = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=23,
        )
        reservoir = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=23,
            reservoir_size=2,
        )
        repeated = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=23,
            reservoir_size=2,
        )

    assert len(reservoir.chains) == 2
    assert len(reservoir.sequences) == 2
    assert len(reservoir.molecular_weights) == 2
    assert reservoir.chains == repeated.chains
    assert {
        unit_id: data["count"] for unit_id, data in reservoir.units.items()
    } == {
        unit_id: data["count"] for unit_id, data in full.units.items()
    }
    assert reservoir.bonds == full.bonds
    assert reservoir.convergence_trace == full.convergence_trace
    assert reservoir.number_average_molecular_weight == pytest.approx(
        full.number_average_molecular_weight
    )


def test_bond_sequences_follow_representative_cover_and_keep_coverage_weights():
    callback_molecules = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        without_export = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=43,
            representative_distance=0.15,
        )
        with_export = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=43,
            representative_distance=0.15,
            retain_bond_sequences=True,
            sample_callback=lambda index, record: callback_molecules.__setitem__(
                index, record["molecule"]
            ),
        )

    assert with_export.bond_sequence_metadata["retention"] == "cover"
    assert with_export.bond_sequence_metadata["weight_semantics"] == "accepted-coverage"
    assert len(with_export.bond_sequences) == len(with_export.chains)
    assert with_export.bond_sequence_indices == sorted(with_export.bond_sequence_indices)
    assert with_export.chains == [
        callback_molecules[index] for index in with_export.bond_sequence_indices
    ]
    assert with_export.bond_sequence_weights == with_export.representative_counts
    accepted_count = with_export.convergence_trace[-1]["n_samples"]
    assert with_export.bond_sequence_metadata["accepted_count"] == accepted_count
    assert sum(with_export.bond_sequence_weights) == accepted_count
    assert with_export.chains == without_export.chains
    assert with_export.sequences == without_export.sequences
    assert with_export.representative_counts == without_export.representative_counts
    assert with_export.convergence_trace == without_export.convergence_trace
    assert with_export.bonds == without_export.bonds
    assert {
        unit_id: (data["count"], data["psmiles"], data["g2rins"])
        for unit_id, data in with_export.units.items()
    } == {
        unit_id: (data["count"], data["psmiles"], data["g2rins"])
        for unit_id, data in without_export.units.items()
    }


def test_bond_sequences_follow_reservoir_without_fabricated_coverage():
    callback_molecules = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        without_export = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=47,
            reservoir_size=2,
        )
        with_export = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=47,
            reservoir_size=2,
            retain_bond_sequences=True,
            sample_callback=lambda index, record: callback_molecules.__setitem__(
                index, record["molecule"]
            ),
        )

    assert with_export.bond_sequence_metadata["retention"] == "reservoir"
    assert with_export.bond_sequence_metadata["weight_semantics"] == "not-applicable"
    assert with_export.bond_sequence_metadata["retained_payload_status"] == "available"
    assert with_export.bond_sequence_weights is None
    assert len(with_export.bond_sequences) == len(with_export.chains) == 2
    assert with_export.bond_sequence_indices == sorted(with_export.bond_sequence_indices)
    assert with_export.chains == [
        callback_molecules[index] for index in with_export.bond_sequence_indices
    ]
    accepted_count = with_export.convergence_trace[-1]["n_samples"]
    assert with_export.bond_sequence_metadata["accepted_count"] == accepted_count
    assert all(index < accepted_count for index in with_export.bond_sequence_indices)
    assert with_export.chains == without_export.chains
    assert with_export.sequences == without_export.sequences
    assert with_export.convergence_trace == without_export.convergence_trace
    assert with_export.bonds == without_export.bonds
    assert {
        unit_id: (data["count"], data["psmiles"], data["g2rins"])
        for unit_id, data in with_export.units.items()
    } == {
        unit_id: (data["count"], data["psmiles"], data["g2rins"])
        for unit_id, data in without_export.units.items()
    }


def test_bond_sequence_reservoir_zero_returns_available_empty_payload():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        result = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=4,
            window=10,
            output_format="smiles",
            seed=53,
            reservoir_size=0,
            retain_bond_sequences=True,
        )

    assert result.bond_sequence_metadata["retention"] == "reservoir"
    assert result.bond_sequence_metadata["retained_payload_status"] == "available"
    assert result.bond_sequences == []
    assert result.bond_sequence_indices == []
    assert result.bond_sequence_weights is None


def test_disabled_bond_sequence_export_does_no_topology_serialization(monkeypatch):
    creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()

    def unexpected_projection(*_args, **_kwargs):
        pytest.fail("disabled bond-sequence export serialized a topology record")

    monkeypatch.setattr(
        ensemble_creator_module,
        "_compact_bond_sequence_record",
        unexpected_projection,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=2,
            window=10,
            output_format="smiles",
            seed=59,
            retain_bond_sequences=False,
            count_bond_sequences=False,
        )

    assert result.bond_sequence_metadata is None
    assert result.bond_sequences is None
    assert result.bond_sequence_indices is None
    assert result.bond_sequence_weights is None
    assert result.bond_sequence_counts is None


def test_convergence_representative_cover_counts_assigned_chains(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    molecular_weights = [100.0, 105.0, 130.0]
    materialized_indices = []

    def materialize_representative(deferred, *args, **kwargs):
        materialized_indices.append(deferred.chain_index)
        return _materialize_deferred_chain(deferred, *args, **kwargs)

    def iter_chain_records(**kwargs):
        for index in range(kwargs["n_samples"]):
            chain_index = kwargs["start_index"] + index
            yield {
                "chain_index": chain_index,
                "record": _deferred(
                    chain_index,
                    molecular_weight=molecular_weights[chain_index],
                ),
                "discards": 0,
                "reasons": (),
                "first_cause": None,
                "warnings": [],
            }

    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    monkeypatch.setattr(
        ensemble_creator_module,
        "_materialize_deferred_chain",
        materialize_representative,
    )
    result = creator.create_ensemble_until_converged(
        batch_size=3,
        max_samples=3,
        window=10,
        representative_distance=0.15,
    )

    assert len(result.chains) == 2
    assert result.molecular_weights == [100.0, 130.0]
    assert result.representative_counts == [2, 1]
    assert sum(result.representative_counts) == 3
    assert materialized_indices == [0, 2]
    assert result.convergence_settings["representative_distance"] == 0.15


def test_representative_cover_distinguishes_composition_and_contacts(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    alternate_pair = (("R0.1", "left"), ("T0.1", "right"))

    def iter_chain_records(**kwargs):
        for index in range(kwargs["n_samples"]):
            chain_index = kwargs["start_index"] + index
            deferred = _deferred(chain_index)
            if chain_index == 1:
                deferred.sample.metadata.unit_counts = {"R1": 1}
            elif chain_index == 2:
                deferred.sample.metadata.labeled_bond_counts = {
                    alternate_pair: 1
                }
            yield {
                "chain_index": chain_index,
                "record": deferred,
                "discards": 0,
                "reasons": (),
                "first_cause": None,
                "warnings": [],
            }

    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    result = creator.create_ensemble_until_converged(
        batch_size=3,
        max_samples=3,
        window=10,
        metadata=False,
        representative_distance=0.15,
    )

    assert len(result.chains) == 3
    assert result.representative_counts == [1, 1, 1]


def test_seeded_representative_cover_checkpoint_resumes_exactly():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        checkpoints = []
        creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=2,
            window=10,
            output_format="smiles",
            seed=53,
            representative_distance=0.15,
            checkpoint_callback=checkpoints.append,
        )
        checkpoint = pickle.loads(pickle.dumps(checkpoints[-1]))
        resumed = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=53,
            representative_distance=0.15,
            checkpoint=checkpoint,
        )
        uninterrupted = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=53,
            representative_distance=0.15,
        )

    assert resumed.chains == uninterrupted.chains
    assert resumed.molecular_weights == uninterrupted.molecular_weights
    assert resumed.representative_counts == uninterrupted.representative_counts
    assert sum(resumed.representative_counts) == 6


def test_convergence_can_omit_returned_metadata():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        result = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=4,
            window=10,
            output_format="smiles",
            seed=29,
            metadata=False,
            reservoir_size=1,
        )

    assert len(result.chains) == 1
    assert result.units == {}
    assert result.bonds == []
    assert result.mol_weights == {}
    assert result.distributions == {}
    assert result.convergence_trace[-1]["n_samples"] == 4


def test_seeded_convergence_checkpoint_resumes_exactly():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        checkpoints = []
        creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=2,
            window=10,
            output_format="smiles",
            seed=31,
            reservoir_size=2,
            checkpoint_callback=checkpoints.append,
        )
        checkpoint = pickle.loads(pickle.dumps(checkpoints[-1]))
        resumed = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=31,
            reservoir_size=2,
            checkpoint=checkpoint,
        )
        uninterrupted = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=31,
            reservoir_size=2,
        )

    assert resumed.chains == uninterrupted.chains
    assert resumed.sequences == uninterrupted.sequences
    assert resumed.molecular_weights == uninterrupted.molecular_weights
    assert resumed.bonds == uninterrupted.bonds
    assert {
        unit_id: data["count"] for unit_id, data in resumed.units.items()
    } == {
        unit_id: data["count"]
        for unit_id, data in uninterrupted.units.items()
    }
    assert resumed.convergence_trace == uninterrupted.convergence_trace
    assert resumed.number_average_molecular_weight == pytest.approx(
        uninterrupted.number_average_molecular_weight
    )
    assert resumed.weight_average_molecular_weight == pytest.approx(
        uninterrupted.weight_average_molecular_weight
    )
    assert resumed.unit_path_statistics == uninterrupted.unit_path_statistics
    assert resumed.unit_junction_statistics == uninterrupted.unit_junction_statistics


def test_statistics_checkpoint_is_compact_and_omits_retained_payloads():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = (
            g2rins.G2rins.make(FAST_SMI)
            .get_graph_creator()
            .get_ensemble_creator()
        )
        full_checkpoints = []
        statistics_checkpoints = []
        creator.create_ensemble_until_converged(
            batch_size=4,
            max_samples=4,
            window=10,
            seed=37,
            checkpoint_callback=full_checkpoints.append,
        )
        creator.create_ensemble_until_converged(
            batch_size=4,
            max_samples=4,
            window=10,
            seed=37,
            checkpoint_policy="statistics",
            checkpoint_callback=statistics_checkpoints.append,
        )

    full = full_checkpoints[-1]
    statistics = statistics_checkpoints[-1]
    assert len(full.retained_records) == 4
    assert statistics.retained_records == ()
    assert statistics.policy == "statistics"
    assert len(pickle.dumps(statistics)) < len(pickle.dumps(full)) / 4


def test_statistics_checkpoint_resume_preserves_statistics_without_payloads():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = (
            g2rins.G2rins.make(FAST_SMI)
            .get_graph_creator()
            .get_ensemble_creator()
        )
        checkpoints = []
        creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=2,
            window=10,
            output_format="smiles",
            seed=41,
            reservoir_size=2,
            checkpoint_policy="statistics",
            unit_junctions=True,
            checkpoint_callback=checkpoints.append,
        )
        checkpoint = pickle.loads(pickle.dumps(checkpoints[-1]))
        resumed_checkpoints = []
        resumed = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=41,
            reservoir_size=2,
            checkpoint_policy="statistics",
            unit_junctions=True,
            checkpoint=checkpoint,
            checkpoint_callback=resumed_checkpoints.append,
        )
        uninterrupted_checkpoints = []
        uninterrupted = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=6,
            window=10,
            output_format="smiles",
            seed=41,
            reservoir_size=2,
            checkpoint_policy="statistics",
            unit_junctions=True,
            checkpoint_callback=uninterrupted_checkpoints.append,
        )

    assert resumed.chains == []
    assert resumed.sequences == []
    assert resumed.molecular_weights == []
    assert resumed.bonds == uninterrupted.bonds
    assert {
        unit_id: data["count"] for unit_id, data in resumed.units.items()
    } == {
        unit_id: data["count"]
        for unit_id, data in uninterrupted.units.items()
    }
    assert resumed.convergence_trace == uninterrupted.convergence_trace
    assert resumed.number_average_molecular_weight == pytest.approx(
        uninterrupted.number_average_molecular_weight
    )
    assert resumed.weight_average_molecular_weight == pytest.approx(
        uninterrupted.weight_average_molecular_weight
    )
    assert resumed_checkpoints[-1].retained_seen == 6
    assert (
        resumed_checkpoints[-1].reservoir_rng_state
        == uninterrupted_checkpoints[-1].reservoir_rng_state
    )
    assert resumed.unit_path_statistics == uninterrupted.unit_path_statistics
    assert resumed.unit_junction_statistics == uninterrupted.unit_junction_statistics
    assert resumed_checkpoints[-1].unit_path_counts == uninterrupted_checkpoints[-1].unit_path_counts
    assert resumed_checkpoints[-1].unit_junction_counts == uninterrupted_checkpoints[-1].unit_junction_counts


def test_serial_and_parallel_unit_path_statistics_match():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        serial = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=4,
            window=10,
            output_format="mol_graph",
            seed=43,
            parallel=False,
        )
        parallel = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=4,
            window=10,
            output_format="mol_graph",
            seed=43,
            parallel=True,
            n_workers=2,
        )

    assert serial.unit_path_statistics == parallel.unit_path_statistics
    assert serial.number_average_molecular_weight == pytest.approx(
        parallel.number_average_molecular_weight
    )
    assert serial.weight_average_molecular_weight == pytest.approx(
        parallel.weight_average_molecular_weight
    )


def test_checkpoint_policy_mismatch_is_rejected():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = (
            g2rins.G2rins.make(FAST_SMI)
            .get_graph_creator()
            .get_ensemble_creator()
        )
        checkpoints = []
        creator.create_ensemble_until_converged(
            batch_size=1,
            max_samples=1,
            window=2,
            seed=43,
            checkpoint_callback=checkpoints.append,
        )

    with pytest.raises(ValueError, match="settings"):
        creator.create_ensemble_until_converged(
            batch_size=1,
            max_samples=2,
            window=2,
            seed=43,
            checkpoint_policy="statistics",
            checkpoint=checkpoints[-1],
        )


def test_legacy_full_checkpoint_without_new_settings_remains_resumable():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = (
            g2rins.G2rins.make(FAST_SMI)
            .get_graph_creator()
            .get_ensemble_creator()
        )
        checkpoints = []
        creator.create_ensemble_until_converged(
            batch_size=1,
            max_samples=1,
            window=2,
            seed=47,
            checkpoint_callback=checkpoints.append,
        )
        checkpoint = checkpoints[-1]
        checkpoint.settings.pop("checkpoint_policy")
        checkpoint.settings.pop("fallback_on_worker_crash")
        checkpoint.settings.pop("use_repeat_units_as_source")
        checkpoint.settings.pop("smiles_policy")
        del checkpoint.policy
        resumed = creator.create_ensemble_until_converged(
            batch_size=1,
            max_samples=2,
            window=2,
            seed=47,
            checkpoint=checkpoint,
        )

    assert len(resumed.chains) == 2
    assert resumed.convergence_trace[-1]["n_samples"] == 2


def test_convergence_checkpoint_requires_seed():
    creator = EnsembleCreator.__new__(EnsembleCreator)
    with pytest.raises(ValueError, match="seed"):
        creator.create_ensemble_until_converged(
            checkpoint_callback=lambda _checkpoint: None
        )