# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Behavioral guardrails for performance-focused sampler rewrites."""

import json

import networkx as nx
import numpy as np
import pytest

import g2rins
import g2rins.ensemble_creator as ensemble_module
from g2rins.distribution import StochasticDistribution
from g2rins.ensemble_creator import (
    _GrowthTransaction,
    _MetadataLevel,
    _PartialAtomGraph,
    _StochasticObjectTracker,
)
from benchmarks.cases import CASES, REFERENCE_CASES
from benchmarks.reference import REFERENCE_PATH, _graph_signature, build_reference


with REFERENCE_PATH.open(encoding="utf-8") as reference_file:
    EXPECTED = json.load(reference_file)


@pytest.mark.parametrize("case_name", REFERENCE_CASES)
def test_seeded_sampling_reference(case_name):
    """Canonical chemistry, compact metadata, and rounding choices stay fixed."""
    assert build_reference(case_name) == EXPECTED[case_name]


def test_reference_fixture_covers_declared_cases():
    assert set(EXPECTED) == set(REFERENCE_CASES)


def test_live_metadata_uses_compact_occurrences(monkeypatch):
    """Repeated units are IDs/node tuples, not per-occurrence graph copies."""
    inspected = False
    original = _PartialAtomGraph.compact_metadata

    def inspect_then_compact(partial_graph, include_sequences=False):
        nonlocal inspected
        inspected = True
        assert sum(partial_graph._unit_counts.values()) == len(partial_graph._unit_occurrences)
        assert len(partial_graph._unit_prototypes) < len(partial_graph._unit_occurrences)
        for occurrence in partial_graph._unit_occurrences:
            assert isinstance(occurrence.nodes, tuple)
            assert not any(isinstance(value, nx.Graph) for value in vars(occurrence).values())
        return original(partial_graph, include_sequences)

    monkeypatch.setattr(
        _PartialAtomGraph,
        "compact_metadata",
        inspect_then_compact,
    )
    creator = g2rins.G2rins.make(
        "C{[>][<]CC(C)[>];;[<]}|poisson(900)|[H]"
    ).get_graph_creator().get_ensemble_creator()
    result = creator.create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)

    assert inspected
    assert result is not None


def test_deferred_canonical_unit_conversion_is_cached(monkeypatch):
    fragment_calls = 0
    original = ensemble_module.mol_graph_to_rdkit_mol

    def fail_first_fragment(*args, **kwargs):
        nonlocal fragment_calls
        if kwargs.get("kekulize") is False:
            fragment_calls += 1
            if fragment_calls == 1:
                raise ValueError("speculative fragment conversion failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ensemble_module,
        "mol_graph_to_rdkit_mol",
        fail_first_fragment,
    )
    creator = g2rins.G2rins.make(
        "C{[>][<]CC(C)[>];;[<]}|poisson(900)|[H]"
    ).get_graph_creator().get_ensemble_creator()
    calls_after_construction = fragment_calls

    first = creator.create_ensemble(1, ensemble_info=True, seed=0)
    calls_after_materialization = fragment_calls
    second = creator.create_ensemble(1, ensemble_info=True, seed=0)

    assert {
        unit_id: (info["psmiles"], info["count"])
        for unit_id, info in first.units.items()
    } == {
        unit_id: (info["psmiles"], info["count"])
        for unit_id, info in second.units.items()
    }
    assert calls_after_materialization == calls_after_construction + 1
    assert fragment_calls == calls_after_materialization


def test_creator_prepares_distributions_once(monkeypatch):
    """Sampling trackers reuse creator-level distribution templates."""
    calls = 0
    original = StochasticDistribution.from_serial_vector.__func__

    def counted_from_serial_vector(cls, vector):
        nonlocal calls
        calls += 1
        return original(cls, vector)

    monkeypatch.setattr(
        StochasticDistribution,
        "from_serial_vector",
        classmethod(counted_from_serial_vector),
    )
    creator = g2rins.G2rins.make(
        "C{[>][<]CC(C)[>];;[<]}|poisson(900)|[H]"
    ).get_graph_creator().get_ensemble_creator()
    construction_calls = calls

    first = _StochasticObjectTracker(
        creator._generative_graph,
        np.random.default_rng(1),
        prepared_distributions=creator._prepared_distributions,
    )
    second = _StochasticObjectTracker(
        creator._generative_graph,
        np.random.default_rng(2),
        prepared_distributions=creator._prepared_distributions,
    )
    creator.sample_mol_graph(rng=np.random.default_rng(3))

    assert construction_calls > 0
    assert calls == construction_calls
    assert first._sto_gen_id_distribution.keys() == second._sto_gen_id_distribution.keys()
    for sto_gen_id in first._sto_gen_id_distribution:
        assert (
            first._sto_gen_id_distribution[sto_gen_id]
            is second._sto_gen_id_distribution[sto_gen_id]
            is creator._prepared_distributions[sto_gen_id]
        )

    sto_gen_id = next(iter(first._sto_gen_id_distribution))
    sto_atom_id = first.register_new_atom_instance(sto_gen_id, -1)
    before = repr(first._rng.bit_generator.state)
    assert first.draw_mw(None, sto_atom_id=sto_atom_id) is not None
    assert repr(first._rng.bit_generator.state) != before


@pytest.mark.parametrize("case_name", REFERENCE_CASES)
def test_termination_fragment_cache_matches_legacy_estimator(case_name, monkeypatch):
    """Cached cap chemistry agrees with the temporary-fragment oracle."""
    monkeypatch.setattr(ensemble_module, "_VERIFY_TERMINATION_MW_CACHE", True)
    assert build_reference(case_name) == EXPECTED[case_name]


@pytest.mark.parametrize("case_name", REFERENCE_CASES)
def test_static_templates_and_direct_merge_match_fallback(case_name, monkeypatch):
    """Prepared construction and direct merge preserve the prior code path."""
    optimized = build_reference(case_name)
    monkeypatch.setattr(ensemble_module, "_USE_STATIC_SOURCE_TEMPLATES", False)
    monkeypatch.setattr(ensemble_module, "_USE_DIRECT_GRAPH_MERGE", False)
    fallback = build_reference(case_name)
    assert optimized == fallback == EXPECTED[case_name]


def test_sampling_does_not_repeat_static_edge_dfs(monkeypatch):
    """Creator preparation is the only static DFS needed for repeated units."""
    creator = g2rins.G2rins.make(
        "C{[>][<]CC(C)[>];;[<]}|poisson(900)|[H]"
    ).get_graph_creator().get_ensemble_creator()

    def unexpected_edge_dfs(*_args, **_kwargs):
        raise AssertionError("sampling repeated nx.edge_dfs")

    monkeypatch.setattr(nx, "edge_dfs", unexpected_edge_dfs)
    graph = creator.sample_mol_graph(rng=np.random.default_rng(0))
    assert graph.number_of_nodes() > 0


def test_half_bond_templates_create_mutable_instance_maps():
    """Promotion of one instantiated half-bond cannot alter its siblings."""
    creator = g2rins.G2rins.make(
        "C{[>][<]CC(C)[>];;[<]}|poisson(900)|[H]"
    ).get_graph_creator().get_ensemble_creator()
    node_template = next(
        node
        for source_template in creator._static_source_templates.values()
        for node in source_template.nodes
        if node.half_bond.mode_entries
    )
    tracker = _StochasticObjectTracker(
        creator._generative_graph,
        np.random.default_rng(1),
        prepared_distributions=creator._prepared_distributions,
    )
    first = ensemble_module._HalfAtomBond(
        0,
        node_template.origin_idx,
        creator._generative_graph,
        tracker,
        np.random.default_rng(2),
        template=node_template.half_bond,
    )
    second = ensemble_module._HalfAtomBond(
        1,
        node_template.origin_idx,
        creator._generative_graph,
        tracker,
        np.random.default_rng(2),
        template=node_template.half_bond,
    )
    mode = next(iter(first._mode_target_map))
    second_targets = list(second._mode_target_map[mode])
    first._mode_target_map[mode].clear()
    first._mode_attr_map = {}
    first.parent = 123

    assert second._mode_target_map[mode] == second_targets
    assert second._mode_attr_map
    assert second.parent != first.parent


def test_merged_metadata_nodes_match_contiguous_watermark(monkeypatch):
    """Every merged unit occupies exactly the atom-ID interval just allocated."""
    observed = []
    original = _PartialAtomGraph.add_new_unit_and_bond

    def inspect_watermark(partial_graph, pre_merge_watermark):
        occurrence_id = original(partial_graph, pre_merge_watermark)
        if occurrence_id is not None:
            nodes = partial_graph._unit_occurrences[occurrence_id].nodes
            expected = tuple(range(pre_merge_watermark, partial_graph._atom_id))
            observed.append((nodes, expected))
        return occurrence_id

    monkeypatch.setattr(
        _PartialAtomGraph,
        "add_new_unit_and_bond",
        inspect_watermark,
    )
    creator = g2rins.G2rins.make(
        "C{[>][<]CC(C)[>];;[<]}|poisson(900)|[H]"
    ).get_graph_creator().get_ensemble_creator()
    creator.sample_mol_graph(rng=np.random.default_rng(0), molecule_info=True)

    assert observed
    assert all(nodes == expected for nodes, expected in observed)


@pytest.mark.parametrize("case_name", REFERENCE_CASES)
def test_transaction_matches_legacy_checkpoint(case_name, monkeypatch):
    """Rollback-heavy architectures agree with the temporary snapshot path."""
    journal = build_reference(case_name)
    monkeypatch.setattr(ensemble_module, "_USE_LEGACY_CHECKPOINTS", True)
    legacy = build_reference(case_name)
    assert journal == legacy


@pytest.mark.parametrize("termination_flag", [0, 1])
@pytest.mark.parametrize("case_name", REFERENCE_CASES)
def test_sparse_transaction_matches_legacy_forced_modes(
    case_name,
    termination_flag,
    monkeypatch,
):
    """Forced boundary modes restore exactly the legacy snapshot topology."""
    case = CASES[case_name]

    def sample():
        creator = (
            g2rins.G2rins.make(case.g2rins)
            .get_graph_creator()
            .get_ensemble_creator()
        )
        return creator.sample_mol_graph(
            rng=np.random.default_rng(case.seed),
            termination_flag=termination_flag,
        )

    journal = sample()
    monkeypatch.setattr(ensemble_module, "_USE_LEGACY_CHECKPOINTS", True)
    legacy = sample()
    assert _graph_signature(journal) == _graph_signature(legacy)


@pytest.mark.parametrize("case_name", REFERENCE_CASES)
def test_compact_and_full_internal_metadata_match(case_name):
    case = CASES[case_name]
    creator = (
        g2rins.G2rins.make(case.g2rins)
        .get_graph_creator()
        .get_ensemble_creator()
    )

    counts = creator.sample_mol_graph(
        rng=np.random.default_rng(case.seed),
        _metadata_mode=_MetadataLevel.COUNTS,
    )
    compact = creator.sample_mol_graph(
        rng=np.random.default_rng(case.seed),
        _metadata_mode=_MetadataLevel.COMPACT_SEQUENCES,
    )
    full = creator.sample_mol_graph(
        rng=np.random.default_rng(case.seed),
        _metadata_mode=_MetadataLevel.FULL_LEGACY,
    )

    assert counts.metadata.unit_counts == compact.metadata.unit_counts == full.metadata.unit_counts
    assert counts.metadata.bond_counts == compact.metadata.bond_counts == full.metadata.bond_counts
    assert counts.metadata.labeled_bond_counts == compact.metadata.labeled_bond_counts == full.metadata.labeled_bond_counts
    assert counts.mol_weights == compact.mol_weights == full.mol_weights
    assert counts.distributions == compact.distributions == full.distributions
    assert [
        [_graph_signature(unit) for unit in sequence]
        for sequence in compact.metadata.materialize_sequences()
    ] == [
        [_graph_signature(unit) for unit in sequence]
        for sequence in full.legacy_sequences
    ]


def test_transaction_rollback_does_not_rewind_rng(monkeypatch):
    """A rejected over-step stays consumed instead of replaying forever."""
    observed = []
    original = _GrowthTransaction.rollback

    def observe_rollback(transaction, rng):
        before = repr(rng.bit_generator.state)
        graph = original(transaction, rng)
        observed.append((before, repr(rng.bit_generator.state)))
        return graph

    monkeypatch.setattr(_GrowthTransaction, "rollback", observe_rollback)
    creator = g2rins.G2rins.make(
        "C{[>][<]CC[>];;[<]}|uniform(400,400)|[H]"
    ).get_graph_creator().get_ensemble_creator()
    graph = creator.sample_mol_graph(
        rng=np.random.default_rng(0),
        termination_flag=1,
    )

    assert graph.number_of_nodes() > 0
    assert observed
    assert all(before == after for before, after in observed)


def test_sparse_transaction_capture_defers_mutation_state(monkeypatch):
    """Checkpoint capture stores watermarks; touched state is journaled later."""
    captures = []
    original = _GrowthTransaction.__init__

    def inspect_capture(transaction, graph, owner, epoch):
        original(transaction, graph, owner, epoch)
        captures.append(
            (
                transaction.atom_watermark,
                len(graph.atom_graph),
                dict(transaction.node_runtime),
                dict(transaction.frontier_buckets),
                dict(transaction.tracker_mapping_keys),
                dict(transaction.metadata_counters),
            )
        )

    monkeypatch.setattr(_GrowthTransaction, "__init__", inspect_capture)
    creator = g2rins.G2rins.make(
        "C{[>][<]CC[>];;[<]}|uniform(400,400)|[H]"
    ).get_graph_creator().get_ensemble_creator()
    creator.sample_mol_graph(
        rng=np.random.default_rng(0),
        termination_flag=1,
        molecule_info=True,
    )

    assert captures
    for watermark, node_count, runtime, frontier, tracker, metadata in captures:
        assert watermark == node_count
        assert runtime == frontier == tracker == metadata == {}


def test_sparse_rollback_removes_discarded_instance_references(monkeypatch):
    """Rollback leaves no tracker, frontier, or metadata references to suffix state."""
    rollbacks = 0
    original = _GrowthTransaction.rollback

    def inspect_rollback(transaction, rng):
        nonlocal rollbacks
        graph = original(transaction, rng)
        rollbacks += 1
        tracker = graph.stochastic_tracker
        live_instance_ids = set(tracker._stochastic_atom_id_to_gen_id)
        assert set(tracker._sto_atom_id_actual_molw) == live_instance_ids
        assert set(tracker._sto_atom_id_expected_molw) == live_instance_ids
        assert set(tracker.parent_map) <= live_instance_ids
        assert all(
            set(parents) <= live_instance_ids
            for parents in tracker.parent_map.values()
        )
        assert tracker._terminated_sto_atom_ids <= live_instance_ids
        assert set(graph._open_half_bond_map) <= live_instance_ids
        assert set(graph._atom_to_unit_occurrence) <= set(graph.atom_graph)
        assert all(
            occurrence_id < len(graph._unit_occurrences)
            for sequence in graph._sequences
            for occurrence_id in sequence
        )
        return graph

    monkeypatch.setattr(_GrowthTransaction, "rollback", inspect_rollback)
    creator = g2rins.G2rins.make(
        "C{[>][<]CC[>];;[<]}|uniform(400,400)|[H]"
    ).get_graph_creator().get_ensemble_creator()
    creator.sample_mol_graph(
        rng=np.random.default_rng(0),
        termination_flag=1,
        molecule_info=True,
    )

    assert rollbacks > 0


def test_nested_growth_fans_mutations_to_compatible_transactions(monkeypatch):
    """Descendant writes are recorded in every live ancestor transaction."""
    largest_transaction_fanout = 0
    original = _GrowthTransaction.record_tracker_mapping_key

    def observe_fanout(transaction, name, key, mapping):
        nonlocal largest_transaction_fanout
        largest_transaction_fanout = max(
            largest_transaction_fanout,
            len(transaction.graph._active_transactions),
        )
        return original(transaction, name, key, mapping)

    monkeypatch.setattr(
        _GrowthTransaction,
        "record_tracker_mapping_key",
        observe_fanout,
    )
    assert build_reference("nested") == EXPECTED["nested"]
    assert largest_transaction_fanout >= 2
