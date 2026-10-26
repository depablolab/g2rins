# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Cheap contracts for convergence finalization, independent of sampling."""

from copy import deepcopy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import g2rins
from g2rins import bond_sequences
from g2rins import ensemble_creator as coordinator
from g2rins._convergence_assembly import (
    _assemble_convergence_checkpoint,
    _assemble_convergence_payload,
)


def _explode(*args, **kwargs):
    raise AssertionError("disabled hook was evaluated")


def _record(label, **overrides):
    fields = dict(
        molecule=label,
        sequences=[label],
        molecular_weight=float(len(label)),
        mol_weights={},
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _creator():
    return SimpleNamespace(
        _canonical_unit_info={"unit": "prototype"},
        _generative_graph=SimpleNamespace(graph={"g2rins_string": "model"}),
    )


def _payload(creator=None, **overrides):
    fields = dict(
        retain_motif_membership=False,
        metadata=False,
        reservoir_size=None,
        representative_distance=None,
        aggregate_motif_occurrences=None,
        retained_motif_by_index=None,
        accepted_count=9,
        aggregate_motif_presence=None,
        representative_counts=[],
        retained_records=[],
        retain_chains=False,
        retain_sequences=False,
        retain_samples=False,
        retain_bond_sequences=False,
        retained_topology_by_index=None,
        bond_sequence_format="topology",
        motif_dictionary_from_counts=_explode,
        motif_population_statistics_from_counts=_explode,
        bond_sequence_export_metadata=_explode,
    )
    fields.update(overrides)
    return _assemble_convergence_payload(
        object() if creator is None else creator, **fields
    )


def _checkpoint_inputs(**overrides):
    fields = dict(
        next_chain_index=4,
        accepted_count=3,
        batch_index=2,
        seed=17,
        mass_sum=12.0,
        mass_square_sum=50.0,
        contact_counts={"contact": 2},
        contact_total=2,
        aggregate_unit_counts={"unit": 5},
        aggregate_bond_counts={},
        aggregate_distributions={"distribution": 7},
        tracker=SimpleNamespace(history=[{"samples": [3]}]),
        retained_records=[(2, _record("chain", mol_weights={"unit": [4.0]}))],
        checkpoint_policy="full",
        retained_seen=3,
        reservoir_rng=None,
        checkpoint_settings={"window": 4},
        representative_features=[{"coordinates": [1.0]}],
        representative_counts=[3],
        aggregate_unit_path_counts={2: {"path": 6}},
        aggregate_unit_path_diagnostics={"truncated": 0},
        aggregate_unit_junction_counts={"junction": 8},
        unit_junctions=True,
        checkpoint_type=coordinator.ConvergenceCheckpoint,
        result_type=coordinator.EnsembleData,
        bond_records_from_labeled=lambda counts, origins: [],
        deepcopy=deepcopy,
    )
    fields.update(overrides)
    return fields


def _checkpoint(creator=None, **overrides):
    return _assemble_convergence_checkpoint(
        object() if creator is None else creator,
        **_checkpoint_inputs(**overrides),
    )


@pytest.mark.parametrize("metadata,enabled", [(False, True), (True, False)])
def test_disabled_metadata_hooks_and_record_attributes_are_lazy(metadata, enabled):
    # The creator has no metadata attributes; records lack disabled outputs.
    result = _payload(
        metadata=metadata,
        retain_motif_membership=enabled,
        retain_bond_sequences=enabled,
        retained_records=[(1, SimpleNamespace(mol_weights={}) if metadata else object())],
    )
    assert len(result.retained) == 1
    assert result.chains == result.sequences == result.masses == []
    assert result.chain_motif_index is None
    assert result.motif_population_statistics is None
    assert result.bond_sequence_metadata is None
    assert result.retained_bond_sequence_payloads == []
    assert result.retained_mol_weights == {}


@pytest.mark.parametrize("representative,counts,expected_counts,mutates", [
    (None, [], None, True),
    (0.2, [7], None, True),
    (0.2, [7, 2], [2, 7], False),
])
def test_retained_sort_mutation_and_paired_representative_order(
    representative, counts, expected_counts, mutates
):
    late, early = _record("late"), _record("early")
    records = [(8, late), (2, early)]
    original = list(records)
    result = _payload(
        retained_records=records,
        representative_distance=representative,
        representative_counts=counts,
        retain_chains=True,
        retain_sequences=True,
        retain_samples=True,
    )
    assert result.retained == [early, late]
    assert result.chains == ["early", "late"]
    assert result.sequences == [["early"], ["late"]]
    assert result.masses == [5.0, 4.0]
    assert result.returned_representative_counts == expected_counts
    assert records == (sorted(original) if mutates else original)
    assert counts == ([] if representative is None else [7] if mutates else [7, 2])


def test_retained_mol_weights_merge_in_sorted_record_and_mapping_order():
    early = _record("early", mol_weights={"b": [1, 2], "a": [3]})
    late = _record("late", mol_weights={"a": [4, 5], "c": [6], "b": [7]})
    result = _payload(metadata=True, retained_records=[(9, late), (1, early)])
    assert list(result.retained_mol_weights) == ["b", "a", "c"]
    assert result.retained_mol_weights == {"b": [1, 2, 7], "a": [3, 4, 5], "c": [6]}
    result.retained_mol_weights["b"].append(99)
    assert early.mol_weights["b"] == [1, 2]
    assert late.mol_weights["b"] == [7]


@pytest.mark.parametrize("reservoir,cover,retention,semantics,weights", [
    (None, None, "all", "retained-observation", [1, 1]),
    (2, None, "reservoir", "not-applicable", None),
    (None, 0.2, "cover", "accepted-coverage", [2, 7]),
    (2, 0.2, "reservoir", "not-applicable", None),
])
def test_topology_indices_payloads_and_weights_align(
    reservoir, cover, retention, semantics, weights
):
    creator = _creator()
    topologies = {8: {"chain": 8}, 2: {"chain": 2}}
    calls = []

    def export(payloads, **kwargs):
        calls.append((payloads, kwargs))
        return {"retention": kwargs["retention"]}

    result = _payload(
        creator,
        metadata=True,
        retain_bond_sequences=True,
        retained_records=[(8, _record("late")), (2, _record("early"))],
        retained_topology_by_index=topologies,
        reservoir_size=reservoir,
        representative_distance=cover,
        representative_counts=[7, 2],
        bond_sequence_export_metadata=export,
    )
    assert result.retained_bond_sequence_indices == [2, 8]
    assert result.retained_bond_sequence_payloads == [topologies[2], topologies[8]]
    assert result.retained_bond_sequence_weights == weights
    assert result.retained_bond_sequence_parents is None
    assert result.bond_sequence_metadata == {"retention": retention}
    assert calls == [([topologies[2], topologies[8]], dict(
        retention=retention,
        weight_semantics=semantics,
        retained_payload_status="available",
        accepted_count=9,
        unit_info=creator._canonical_unit_info,
        model_identity="model",
    ))]
    if cover is not None and reservoir is None:
        assert result.retained_bond_sequence_weights is not result.returned_representative_counts


@pytest.mark.parametrize("topologies,cover,counts,message", [
    ({}, None, [], "A selected chain is missing its retained bond-sequence topology"),
    ({1: None}, None, [], "A selected chain is missing its retained bond-sequence topology"),
    ({1: {}}, 0.2, [], "Retained bond-sequence topology is not aligned with coverage weights"),
])
def test_topology_errors_precede_export(topologies, cover, counts, message):
    with pytest.raises(RuntimeError) as error:
        _payload(
            metadata=True,
            retain_bond_sequences=True,
            retained_records=[(1, _record("chain"))],
            retained_topology_by_index=topologies,
            representative_distance=cover,
            representative_counts=counts,
        )
    assert str(error.value) == message


@pytest.mark.parametrize("reservoir,cover,retention", [
    (None, None, "all"), (2, None, "reservoir"),
    (None, 0.2, "cover"), (2, 0.2, "reservoir"),
])
def test_motif_rows_sorted_positive_and_retention_preference(reservoir, cover, retention):
    creator = _creator()
    occurrences, presence = {"a": 9}, {"a": 3}
    lookup = {"a": 2, "z": 0, "zero": 3, "negative": 1}
    entries = {
        8: {"_counts": {"a": 3, "zero": 0, "negative": -2, "z": 1},
            "molar_mass": 12, "motif_exposure": 4.0},
        2: {"_counts": {}, "molar_mass": 7, "motif_exposure": 0},
    }
    before = deepcopy(entries)
    calls = []
    statistics = {"exact": True}

    def dictionary(counts, info):
        assert counts is occurrences and info is creator._canonical_unit_info
        calls.append("dictionary")
        return [{"id": "unused dictionary"}], lookup

    def population(accepted, counts, seen, info):
        assert accepted == 9
        assert counts is occurrences and seen is presence
        assert info is creator._canonical_unit_info
        calls.append("population")
        return statistics

    result = _payload(
        creator, metadata=True, retain_motif_membership=True,
        reservoir_size=reservoir, representative_distance=cover,
        aggregate_motif_occurrences=occurrences,
        aggregate_motif_presence=presence,
        retained_motif_by_index=entries,
        motif_dictionary_from_counts=dictionary,
        motif_population_statistics_from_counts=population,
    )
    assert calls == ["dictionary", "population"]
    assert entries == before
    assert result.motif_population_statistics is statistics
    assert result.chain_motif_index == dict(
        schema="chain-motif-membership/v1", population="retained-samples",
        accepted_count=9, retained_count=2, retention=retention,
        retained_payload_status="available", mass_unit="g/mol",
        motif_dictionary_id="unit-graph-simple-paths/v2",
        rows=[
            dict(chain_index=2, molar_mass=7.0, motif_counts=[], motif_exposure=0),
            dict(chain_index=8, molar_mass=12.0, motif_counts=[[0, 1], [2, 3]],
                 motif_exposure=4),
        ],
    )


@pytest.mark.parametrize("format", ["topology", "paths", "branches"])
@pytest.mark.parametrize("enabled", [False, True])
def test_topology_format_hooks_are_conditional(monkeypatch, format, enabled):
    calls = []

    def renderer(kind, payload):
        calls.append((kind, payload))
        return [kind, payload] if kind == "paths" else (["segment", payload], [-1])

    def metadata(kind, value):
        calls.append((kind + "-metadata", value))
        return {"format": kind}

    monkeypatch.setattr(bond_sequences, "render_bond_sequence_paths",
                        lambda payload: renderer("paths", payload))
    monkeypatch.setattr(bond_sequences, "render_bond_sequence_branches",
                        lambda payload: renderer("branches", payload))
    monkeypatch.setattr(bond_sequences, "bond_sequence_path_metadata",
                        lambda value: metadata("paths", value))
    monkeypatch.setattr(bond_sequences, "bond_sequence_branch_metadata",
                        lambda value: metadata("branches", value))
    result = _payload(
        _creator() if enabled else object(), metadata=enabled,
        retain_bond_sequences=True, bond_sequence_format=format,
        retained_records=[(1, _record("chain"))],
        retained_topology_by_index={1: "raw"},
        bond_sequence_export_metadata=lambda *args, **kwargs: {"base": True},
    )
    if not enabled:
        assert calls == []
        assert result.retained_bond_sequence_payloads == []
        assert result.bond_sequence_metadata is None
    elif format == "topology":
        assert calls == []
        assert result.retained_bond_sequence_payloads == ["raw"]
        assert result.bond_sequence_metadata == {"base": True}
    else:
        assert calls == [(format, "raw"), (format + "-metadata", {"base": True})]
        assert result.bond_sequence_metadata == {"format": format}
        if format == "paths":
            assert result.retained_bond_sequence_payloads == [["paths", "raw"]]
            assert result.retained_bond_sequence_parents is None
        else:
            assert result.retained_bond_sequence_payloads == [["segment", "raw"]]
            assert result.retained_bond_sequence_parents == [[-1]]


@pytest.mark.parametrize("policy", ["full", "statistics"])
def test_checkpoint_copies_mutable_state_and_preserves_pickle_classes(policy):
    inputs = _checkpoint_inputs(checkpoint_policy=policy)
    checkpoint = _assemble_convergence_checkpoint(object(), **inputs)
    assert type(checkpoint) is coordinator.ConvergenceCheckpoint
    assert type(checkpoint.aggregate) is coordinator.EnsembleData
    assert type(checkpoint).__module__ == "g2rins.ensemble_creator"
    assert type(checkpoint.aggregate).__module__ == "g2rins.ensemble_creator"
    restored = pickle.loads(pickle.dumps(checkpoint))
    assert type(restored) is type(checkpoint)
    assert type(restored.aggregate) is type(checkpoint.aggregate)
    assert restored == checkpoint
    assert checkpoint.aggregate.units == {"unit": {"count": 5}}
    assert checkpoint.aggregate.chains == checkpoint.aggregate.sequences == []
    assert checkpoint.aggregate.mol_weights == {}
    assert checkpoint.policy == policy
    if policy == "full":
        assert checkpoint.retained_records == tuple(inputs["retained_records"])
        assert checkpoint.retained_records[0][1] is not inputs["retained_records"][0][1]
        inputs["retained_records"][0][1].mol_weights["unit"].append(99)
        assert checkpoint.retained_records[0][1].mol_weights == {"unit": [4.0]}
    else:
        assert checkpoint.retained_records == ()
    inputs["tracker"].history[0]["samples"].append(99)
    inputs["representative_features"][0]["coordinates"].append(99)
    inputs["representative_counts"].append(99)
    inputs["contact_counts"]["contact"] = 99
    inputs["aggregate_unit_counts"]["unit"] = 99
    inputs["aggregate_distributions"]["distribution"] = 99
    inputs["aggregate_unit_path_counts"][2]["path"] = 99
    inputs["aggregate_unit_path_diagnostics"]["truncated"] = 99
    inputs["aggregate_unit_junction_counts"]["junction"] = 99
    inputs["checkpoint_settings"]["window"] = 99
    assert checkpoint.convergence_history == [{"samples": [3]}]
    assert checkpoint.representative_features == ({"coordinates": [1.0]},)
    assert checkpoint.representative_counts == (3,)
    assert checkpoint.contact_counts == {"contact": 2}
    assert checkpoint.aggregate.units == {"unit": {"count": 5}}
    assert checkpoint.aggregate.distributions == {"distribution": 7}
    assert checkpoint.unit_path_counts == {2: {"path": 6}}
    assert checkpoint.unit_path_diagnostics == {"truncated": 0}
    assert checkpoint.unit_junction_counts == {"junction": 8}
    assert checkpoint.settings == {"window": 4}
    checkpoint.settings["new"] = True
    assert "new" not in inputs["checkpoint_settings"]


def test_statistics_checkpoint_never_copies_or_iterates_omitted_records():
    class OmittedRecords:
        def __deepcopy__(self, memo):
            return _explode()

        def __iter__(self):
            return _explode()

    checkpoint = _checkpoint(
        retained_records=OmittedRecords(), checkpoint_policy="statistics",
        unit_junctions=False, aggregate_unit_junction_counts=None,
    )
    assert checkpoint.retained_records == ()
    assert checkpoint.unit_junction_counts is None
    assert checkpoint.reservoir_rng_state is None


def test_checkpoint_rng_snapshot_is_detached_without_advancing_generator():
    rng = np.random.default_rng(31)
    before = deepcopy(rng.bit_generator.state)
    checkpoint = _checkpoint(reservoir_rng=rng)
    assert rng.bit_generator.state == before
    assert checkpoint.reservoir_rng_state == before
    assert checkpoint.reservoir_rng_state is not rng.bit_generator.state
    checkpoint.reservoir_rng_state["state"]["state"] = 0
    assert rng.bit_generator.state == before
    checkpoint = _checkpoint(reservoir_rng=rng)
    rng.random()
    assert checkpoint.reservoir_rng_state == before


@pytest.mark.parametrize("policy", ["full", "statistics"])
def test_checkpoint_hook_evaluation_and_constructor_field_order(policy):
    events = []
    inputs = _checkpoint_inputs(checkpoint_policy=policy)
    rng_state = {"state": {"value": 1}}

    class BitGenerator:
        @property
        def state(self):
            events.append("rng-state")
            return rng_state

    copy_targets = [
        (inputs["tracker"].history, "history"),
        (inputs["retained_records"], "records"),
        (rng_state, "rng-copy"),
        (inputs["representative_features"], "features"),
    ]

    def copy_hook(value):
        events.append(next(name for target, name in copy_targets if value is target))
        return deepcopy(value)

    origins = {"origin": 17}

    def serialize(counts, origin):
        assert counts is inputs["aggregate_bond_counts"]
        assert origin is origins
        events.append("serialize")
        return ["serialized"]

    def result_type(*args, **kwargs):
        events.append("result")
        assert args == ([], {"unit": {"count": 5}}, ["serialized"], [], {},
                        {"distribution": 7})
        assert kwargs == {}
        return coordinator.EnsembleData(*args)

    def checkpoint_type(**kwargs):
        events.append("checkpoint")
        assert list(kwargs) == [
            "next_chain_index", "accepted_count", "batch_index", "seed",
            "mass_sum", "mass_square_sum", "contact_counts", "contact_total",
            "aggregate", "convergence_history", "retained_records", "retained_seen",
            "reservoir_rng_state", "settings", "policy", "representative_features",
            "representative_counts", "unit_path_counts", "unit_path_diagnostics",
            "unit_junction_counts",
        ]
        assert events == ["serialize", "result", "history"] + (
            ["records"] if policy == "full" else []
        ) + ["rng-state", "rng-copy", "features", "checkpoint"]
        return SimpleNamespace(**kwargs)

    inputs.update(
        reservoir_rng=SimpleNamespace(bit_generator=BitGenerator()),
        bond_records_from_labeled=serialize, result_type=result_type,
        checkpoint_type=checkpoint_type, deepcopy=copy_hook,
    )
    result = _assemble_convergence_checkpoint(
        SimpleNamespace(_origin_node_by_text=origins), **inputs
    )
    assert result.policy == policy


def test_coordinator_still_supplies_legacy_module_hooks(monkeypatch):
    calls = []
    original_serializer = coordinator._bond_records_from_labeled
    original_copy = coordinator.deepcopy

    def serialize(*args, **kwargs):
        calls.append("serialize")
        return original_serializer(*args, **kwargs)

    def copy_hook(value):
        calls.append("deepcopy")
        return original_copy(value)

    monkeypatch.setattr(coordinator, "_bond_records_from_labeled", serialize)
    monkeypatch.setattr(coordinator, "deepcopy", copy_hook)
    creator = g2rins.G2rins.make("C").get_graph_creator().get_ensemble_creator()
    checkpoints = []
    result = creator.create_ensemble_until_converged(
        batch_size=1, max_samples=1, window=2, seed=7,
        output_format="smiles", metadata=True,
        checkpoint_callback=checkpoints.append,
    )
    assert len(checkpoints) == 1
    assert checkpoints[0].accepted_count == 1
    assert result.chains == ["C"]
    # Checkpoint aggregate and final output both consult the coordinator hook.
    assert calls.count("serialize") == 2
    assert "deepcopy" in calls