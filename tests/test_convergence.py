# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for opt-in ensemble sampling until statistical convergence."""

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

    def create_ensemble(n_samples, **kwargs):
        calls.append((n_samples, kwargs))
        return _batch(n_samples)

    monkeypatch.setattr(creator, "create_ensemble", create_ensemble)
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
    assert [kwargs["seed"] for _size, kwargs in calls] == [10, 12, 14]
    assert len(progress) == 3
    assert "convergence:" in progress[-1]


def test_create_ensemble_until_converged_honors_max_samples(monkeypatch):
    creator = EnsembleCreator.__new__(EnsembleCreator)
    requested_sizes = []

    def create_ensemble(n_samples, **_kwargs):
        requested_sizes.append(n_samples)
        return _batch(n_samples)

    monkeypatch.setattr(creator, "create_ensemble", create_ensemble)
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