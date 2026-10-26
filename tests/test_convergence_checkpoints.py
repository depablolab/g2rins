# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for convergence checkpoint retention, resume, and compatibility."""

import pickle
import warnings

import pytest

import g2rins
from g2rins.ensemble_creator import EnsembleCreator
from tests._support import FAST_SMI


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