# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for opt-in ensemble sampling until statistical convergence."""

import pickle
import warnings

import pytest
from rdkit.Chem import Descriptors

import g2rins
from g2rins.convergence import ConvergenceTracker
from g2rins.ensemble_creator import EnsembleCreator, EnsembleData


FAST_SMI = "{[] [<]CC([>])c1ccccc1; CO[>]; [<][H] []}|gauss(1000, 45)|"


def _batch(n_samples, molecular_weight=100.0):
    return EnsembleData(
        chains=[f"chain-{i}" for i in range(n_samples)],
        units={"R0": {"count": n_samples}},
        bonds=[
            {
                "labels": ["R0.1", "R0.2"],
                "nodes": ["left", "right"],
                "count": n_samples,
            }
        ],
        sequences=[["R0"] for _ in range(n_samples)],
        mol_weights={0: [molecular_weight] * n_samples},
        distributions={0: "uniform(100,100)"},
        molecular_weights=[molecular_weight] * n_samples,
    )


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
    ),
)
def test_converged_ensemble_validates_settings(kwargs, match):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    with pytest.raises(ValueError, match=match):
        creator.create_ensemble_until_converged(**kwargs)


def test_create_ensemble_until_converged_stops_after_stable_window(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    calls = []

    def iter_chain_records(**kwargs):
        calls.append(kwargs)
        for index in range(kwargs["n_samples"]):
            yield {
                "chain_index": kwargs["start_index"] + index,
                "record": object(),
                "discards": 0,
                "reasons": (),
                "first_cause": None,
                "warnings": [],
            }

    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    monkeypatch.setattr(
        creator,
        "_records_to_ensemble_data",
        lambda records: _batch(len(records)),
    )
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
    assert result.molecular_weights == [100.0] * 6
    assert result.units["R0"]["count"] == 6
    assert result.bonds[0]["count"] == 6
    assert [kwargs["seed"] for kwargs in calls] == [10, 12, 14]
    assert [kwargs["start_index"] for kwargs in calls] == [0, 2, 4]
    assert len(progress) == 3
    assert "convergence:" in progress[-1]


def test_create_ensemble_until_converged_honors_max_samples(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    requested_sizes = []

    def iter_chain_records(**kwargs):
        requested_sizes.append(kwargs["n_samples"])
        for index in range(kwargs["n_samples"]):
            yield {
                "chain_index": kwargs["start_index"] + index,
                "record": object(),
                "discards": 0,
                "reasons": (),
                "first_cause": None,
                "warnings": [],
            }

    monkeypatch.setattr(creator, "_iter_chain_records", iter_chain_records)
    monkeypatch.setattr(
        creator,
        "_records_to_ensemble_data",
        lambda records: _batch(len(records)),
    )
    result = creator.create_ensemble_until_converged(
        batch_size=2,
        max_samples=5,
        window=10,
    )

    assert not result.converged
    assert len(result.chains) == 5
    assert result.n_batches == 3
    assert requested_sizes == [2, 2, 1]


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
                (index, record["molecule"], record["sequences"])
            ),
        )

    assert [index for index, _chain, _sequences in streamed] == list(range(6))
    assert [chain for _index, chain, _sequences in streamed] == full.chains
    assert all(sequences is None for _index, _chain, sequences in streamed)
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


def test_convergence_checkpoint_requires_seed():
    creator = EnsembleCreator.__new__(EnsembleCreator)
    with pytest.raises(ValueError, match="seed"):
        creator.create_ensemble_until_converged(
            checkpoint_callback=lambda _checkpoint: None
        )