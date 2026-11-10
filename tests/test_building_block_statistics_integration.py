# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Native API coverage for all-accepted building-block statistics."""

import json
from collections import Counter, defaultdict
from unittest.mock import patch

import pytest
from rdkit import Chem

import g2rins.bond_sequences as bond_sequences
import g2rins.ensemble_creator as module
from benchmarks.cases import CASES
from tests._support import (
    CONVERGENCE, FAST_SMI, SOURCES, _bond_record_oracle, _creator, _ordinary_fields,
)


def _reference(source=FAST_SMI, **options):
    """Capture pre-projection rows: raw statistics deliberately retain H caps."""
    raw_records = []
    normalize = bond_sequences.normalize_hydrogen_termini

    def capture(record, unit_info):
        raw_records.append(record)
        return normalize(record, unit_info)

    with patch.object(bond_sequences, "normalize_hydrogen_termini", capture):
        result = _creator(source).create_ensemble_until_converged(**options)
    result._raw_bond_sequences = raw_records
    assert len(raw_records) == len(result.bond_sequences)
    return result


def _unexpected(*_args, **_kwargs):
    pytest.fail("unexpected statistics, retention, path enumeration, or scheduler work")


def _chemistry(chains):
    """Compare fresh creators chemically, never by parser UUID/origin nodes."""
    result = []
    for chain in chains:
        molecule = (
            Chem.MolFromSmiles(chain) if isinstance(chain, str)
            else module.mol_graph_to_rdkit_mol(chain)
        )
        assert molecule is not None
        result.append(Chem.MolToSmiles(molecule, canonical=True))
    return result


def _assert_population(statistics, accepted_count, unit_counts):
    assert statistics["accepted_count"] == accepted_count
    assert {
        row["unit_id"]: row["occurrence_count"]
        for row in statistics["building_blocks"]
    } == unit_counts
    assert sum(row["occurrence_count"] for row in statistics["building_blocks"]) == sum(unit_counts.values())
    centers = Counter()
    for row in statistics["neighborhoods"]:
        centers[row["center_unit"]] += row["occurrence_count"]
    assert centers == unit_counts
    for collection in ("building_blocks", "neighborhoods", "site_connections"):
        for row in statistics[collection]:
            count = row.get("occurrence_count", row.get("endpoint_count"))
            assert 1 <= row["chain_presence_count"] <= min(accepted_count, count)
    probabilities = defaultdict(float)
    for row in statistics["site_connections"]:
        assert row["probability"] == row["endpoint_count"] / row["occupied_endpoint_count"]
        probabilities[row["source_unit"], row["source_site"]] += row["probability"]
    assert all(value == pytest.approx(1.0) for value in probabilities.values())
    assert json.loads(json.dumps(statistics, allow_nan=False)) == statistics


@pytest.mark.parametrize("source", SOURCES + [
    pytest.param(CASES["small-linear"].g2rins, id="small-linear"),
])
@pytest.mark.parametrize("output_format", ["mol_graph", "smiles"])
def test_fixed_outputs_and_json_only_add_statistics(source, output_format, tmp_path):
    creator = _creator(source)
    paths = [tmp_path / f"ensemble-{index}.json" for index in range(3)]
    options = {"output_format": output_format, "seed": 7, "json_max_chains": 1}
    baseline = creator.create_ensemble(
        2, ensemble_info=True, json_file=str(paths[0]), **options,
    )
    enabled = creator.create_ensemble(
        2, ensemble_info=True, json_file=str(paths[1]),
        building_block_statistics=True, **options,
    )
    json_only = creator.create_ensemble(
        2, json_file=str(paths[2]), building_block_statistics=True, **options,
    )
    assert baseline.building_block_statistics is None
    assert module._metadata_equal(_ordinary_fields(enabled), _ordinary_fields(baseline))
    assert module._metadata_equal(json_only, enabled.chains)
    reference = _reference(source,
        batch_size=2, max_samples=2, window=10, seed=7,
        output_format=output_format, retain_bond_sequences=True,
    )
    assert enabled.building_block_statistics == _bond_record_oracle(reference._raw_bond_sequences)
    assert _chemistry(enabled.chains) == _chemistry(reference.chains)
    payloads = [json.loads(path.read_text()) for path in paths]
    assert "building_block_statistics" not in payloads[0]["ensemble"]
    for payload in payloads[1:]:
        assert payload["ensemble"].pop("building_block_statistics") == enabled.building_block_statistics
        # Same parsed creator makes even provenance fields directly comparable.
        assert payload == payloads[0]
        assert len(payload["ensemble"]["chains"]) == 1
    _assert_population(
        enabled.building_block_statistics, 2,
        {unit: info["count"] for unit, info in enabled.units.items()},
    )


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("output_format", ["mol_graph", "smiles"])
def test_native_counts_match_independent_retained_record_oracle(source, output_format):
    options = {**CONVERGENCE, "output_format": output_format, "retain_bond_sequences": True}
    # Two fresh parses must preserve native labels, not parser UUID identities.
    reference = _reference(source, **options)
    creator = _creator(source)
    baseline = creator.create_ensemble_until_converged(**options)
    enabled = creator.create_ensemble_until_converged(
        building_block_statistics=True, **options,
    )
    assert module._metadata_equal(_ordinary_fields(enabled), _ordinary_fields(baseline))
    assert len(reference.bond_sequences) == 4
    expected = _bond_record_oracle(reference._raw_bond_sequences)
    assert enabled.building_block_statistics == expected
    assert enabled.bond_sequences == reference.bond_sequences
    assert _chemistry(enabled.chains) == _chemistry(reference.chains)
    assert enabled.molecular_weights == reference.molecular_weights
    assert enabled.convergence_trace == reference.convergence_trace
    unit_counts = {unit: info["count"] for unit, info in reference.units.items()}
    assert set(enabled.units) == set(reference.units)
    _assert_population(expected, 4, unit_counts)
    if source == CASES["branched"].g2rins:
        assert any(len(row["neighbors"]) >= 3 for row in expected["neighborhoods"])


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("selection, retained", [
    pytest.param({}, 4, id="all"),
    pytest.param({"reservoir_size": 0}, 0, id="reservoir-zero"),
    pytest.param({"reservoir_size": 1}, 1, id="reservoir-one"),
    pytest.param({"representative_distance": 100.0}, 1, id="representative-cover"),
])
def test_selection_never_reweights_or_truncates_population(source, selection, retained):
    reference = _reference(source,
        retain_bond_sequences=True, **CONVERGENCE,
    )
    creator = _creator(source)
    baseline = creator.create_ensemble_until_converged(**CONVERGENCE, **selection)
    enabled = creator.create_ensemble_until_converged(
        building_block_statistics=True, **CONVERGENCE, **selection,
    )
    assert len(enabled.chains) == len(enabled.sequences) == retained
    assert module._metadata_equal(_ordinary_fields(enabled), _ordinary_fields(baseline))
    assert enabled.convergence_trace == reference.convergence_trace
    assert enabled.convergence_trace[-1]["n_samples"] == 4
    assert enabled.building_block_statistics == _bond_record_oracle(reference._raw_bond_sequences)
    _assert_population(
        enabled.building_block_statistics, 4,
        {unit: info["count"] for unit, info in reference.units.items()},
    )
    if "representative_distance" in selection:
        assert sum(enabled.representative_counts) == 4


@pytest.mark.parametrize("source", SOURCES)
def test_statistics_only_avoids_paths_and_retained_materialization(source, monkeypatch):
    reference = _reference(source,
        retain_bond_sequences=True, **CONVERGENCE,
    )
    creator = _creator(source)
    options = {
        **CONVERGENCE, "metadata": False,
        "retain_chains": False, "retain_sequences": False,
    }
    baseline = creator.create_ensemble_until_converged(**options)
    for name in (
        "_unit_path_counts", "_compact_bond_sequence_record",
        "_materialize_deferred_chain", "_bond_sequence_record_encoded_size",
    ):
        monkeypatch.setattr(module, name, _unexpected)
    # Molecular-weight work remains required: do not sentinel RDKit conversion.
    result = creator.create_ensemble_until_converged(
        building_block_statistics=True, **options,
    )
    assert module._metadata_equal(_ordinary_fields(result), _ordinary_fields(baseline))
    assert result.chains == result.sequences == result.molecular_weights == []
    assert result.units == {} and result.bonds == []
    assert result.bond_sequences is None and result.chain_motif_index is None
    assert result.building_block_statistics == _bond_record_oracle(reference._raw_bond_sequences)
    assert result.convergence_trace == reference.convergence_trace


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("api", ["fixed", "convergence"])
def test_combined_flags_share_one_occurrence_view_per_sample(source, api, monkeypatch):
    reference = (
        _reference(source, retain_bond_sequences=True, **CONVERGENCE)
        if api == "convergence" else None
    )
    creator = _creator(source)
    original = module._build_occurrence_adjacency
    calls = []

    def occurrence_view(*args, **kwargs):
        calls.append(kwargs.get("preserve_self_edges", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_build_occurrence_adjacency", occurrence_view)
    if api == "fixed":
        result = creator.create_ensemble(
            4, ensemble_info=True, output_format="smiles", seed=7,
            retain_motif_membership=True, building_block_statistics=True,
        )
    else:
        result = creator.create_ensemble_until_converged(
            metadata=True, retain_bond_sequences=True,
            retain_motif_membership=True, building_block_statistics=True,
            **CONVERGENCE,
        )
        assert len(result.bond_sequences) == 4
        assert result.building_block_statistics == _bond_record_oracle(reference._raw_bond_sequences)
    assert calls == [True] * 4
    assert result.chain_motif_index["accepted_count"] == 4
    assert len(result.chain_motif_index["rows"]) == 4
    assert result.motif_population_statistics["entries"]
    _assert_population(
        result.building_block_statistics, 4,
        {unit: info["count"] for unit, info in result.units.items()},
    )


@pytest.mark.parametrize("api", ["fixed", "convergence"])
@pytest.mark.parametrize("output_format", ["mol_graph", "smiles"])
@pytest.mark.parametrize("flag", [{}, {"building_block_statistics": False}], ids=["default", "explicit-false"])
def test_disabled_never_constructs_accumulator(api, output_format, flag, monkeypatch, tmp_path):
    creator = _creator()
    monkeypatch.setattr(module, "_BuildingBlockStatistics", _unexpected)
    if api == "fixed":
        path = tmp_path / "disabled.json"
        result = creator.create_ensemble(
            2, ensemble_info=True, output_format=output_format, seed=7,
            json_file=str(path), **flag,
        )
        assert "building_block_statistics" not in json.loads(path.read_text())["ensemble"]
    else:
        result = creator.create_ensemble_until_converged(
            **{**CONVERGENCE, "output_format": output_format}, **flag,
        )
    assert result.building_block_statistics is None


def test_fixed_statistics_count_only_the_accepted_prefix(monkeypatch):
    reference = _reference(
        batch_size=2, max_samples=2, window=10, seed=7,
        retain_bond_sequences=True,
    )
    creator = _creator()
    original = creator._iter_chain_records

    def fail_second(**kwargs):
        iterator = original(**kwargs)
        yield next(iterator)
        yield {
            "chain_index": 1, "record": None, "discards": 1,
            "reasons": (), "first_cause": None, "warnings": [],
        }

    monkeypatch.setattr(creator, "_iter_chain_records", fail_second)
    with pytest.warns(Warning):
        result = creator.create_ensemble(
            2, ensemble_info=True, seed=7, building_block_statistics=True,
        )
    assert len(result.chains) == 1
    assert result.building_block_statistics == _bond_record_oracle(reference._raw_bond_sequences[:1])
    _assert_population(
        result.building_block_statistics, 1,
        {unit: info["count"] for unit, info in result.units.items()},
    )


@pytest.mark.parametrize("api, options, error, match", [
    pytest.param("fixed", {}, ValueError, "ensemble_info=True or json_file", id="fixed-needs-output"),
    pytest.param("convergence", {"checkpoint_callback": lambda _checkpoint: None},
                 NotImplementedError, "checkpoint/resume", id="checkpoint-callback"),
    pytest.param("convergence", {"parallel": True, "n_workers": 2, "checkpoint_callback": lambda _checkpoint: None},
                 NotImplementedError, "checkpoint/resume", id="parallel-checkpoint-callback"),
    pytest.param("convergence", {"checkpoint_policy": "statistics", "checkpoint_callback": lambda _checkpoint: None},
                 NotImplementedError, "checkpoint/resume", id="statistics-checkpoint-callback"),
    pytest.param("convergence", {"checkpoint": object.__new__(module.ConvergenceCheckpoint)},
                 NotImplementedError, "checkpoint/resume", id="resume"),
])
def test_unsupported_options_reject_before_sampling_or_pool(api, options, error, match, monkeypatch):
    creator = module.EnsembleCreator.__new__(module.EnsembleCreator)
    monkeypatch.setattr(creator, "_iter_chain_records", _unexpected)
    monkeypatch.setattr(module, "_ParallelChainScheduler", _unexpected)
    monkeypatch.setattr(module, "_BuildingBlockStatistics", _unexpected)
    with pytest.raises(error, match=match):
        if api == "fixed":
            creator.create_ensemble(1, seed=7, building_block_statistics=True, **options)
        else:
            creator.create_ensemble_until_converged(
                **CONVERGENCE, building_block_statistics=True, **options,
            )


@pytest.mark.parametrize("n_samples", [0, 2])
def test_fixed_empty_and_isolated_samples_have_statistics(n_samples):
    result = _creator("C").create_ensemble(
        n_samples, ensemble_info=True, seed=7, building_block_statistics=True,
    )
    statistics = result.building_block_statistics
    _assert_population(
        statistics, n_samples,
        {unit: info["count"] for unit, info in result.units.items()},
    )
    assert statistics["site_connections"] == []
    assert all(row["neighbors"] == [] for row in statistics["neighborhoods"])
    if n_samples == 0:
        assert statistics == _bond_record_oracle([])