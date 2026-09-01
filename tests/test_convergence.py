# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for opt-in ensemble sampling until statistical convergence."""

import pickle
import warnings

import networkx as nx
import pytest
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
        "mass_delta=0.0000/0.0020 contact_delta=0.0000/0.0100"
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


def test_create_ensemble_until_converged_defaults_to_1500_max_samples(monkeypatch):
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
        )

    assert result is not None
    assert len(result.chains) == 1
    assert result.chains[0]


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


def test_legacy_full_checkpoint_without_policy_remains_resumable():
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