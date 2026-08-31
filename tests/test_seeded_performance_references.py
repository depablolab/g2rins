# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Behavioral guardrails for performance-focused sampler rewrites."""

import json

import networkx as nx
import numpy as np
import pytest

import g2rins
import g2rins.ensemble_creator as ensemble_module
from g2rins.ensemble_creator import _GrowthTransaction, _PartialAtomGraph
from benchmarks.cases import REFERENCE_CASES
from benchmarks.reference import REFERENCE_PATH, build_reference


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
    original = _PartialAtomGraph.materialize_legacy_metadata

    def inspect_then_materialize(partial_graph):
        nonlocal inspected
        inspected = True
        assert sum(partial_graph._unit_counts.values()) == len(partial_graph._unit_occurrences)
        assert len(partial_graph._unit_prototypes) < len(partial_graph._unit_occurrences)
        for occurrence in partial_graph._unit_occurrences:
            assert isinstance(occurrence.nodes, tuple)
            assert not any(isinstance(value, nx.Graph) for value in vars(occurrence).values())
        return original(partial_graph)

    monkeypatch.setattr(_PartialAtomGraph, "materialize_legacy_metadata", inspect_then_materialize)
    creator = g2rins.G2rins.make(
        "C{[>][<]CC(C)[>];;[<]}|poisson(900)|[H]"
    ).get_graph_creator().get_ensemble_creator()
    result = creator.create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)

    assert inspected
    assert result is not None


@pytest.mark.parametrize("case_name", REFERENCE_CASES)
def test_transaction_matches_legacy_checkpoint(case_name, monkeypatch):
    """Rollback-heavy architectures agree with the temporary snapshot path."""
    journal = build_reference(case_name)
    monkeypatch.setattr(ensemble_module, "_USE_LEGACY_CHECKPOINTS", True)
    legacy = build_reference(case_name)
    assert journal == legacy


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
