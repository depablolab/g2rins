# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Fixed-size motif analytics and strict opt-in regressions."""

import json
import warnings
from collections import Counter

import pytest

import g2rins
import g2rins.ensemble_creator as module


SOURCE = "{[] [<]CC([>])c1ccccc1; CO[>]; [<][H] []}|gauss(1000, 45)|"


def _creator(source=SOURCE):
    return g2rins.G2rins.make(source).get_graph_creator().get_ensemble_creator()


@pytest.mark.parametrize("output_format", ["smiles", "mol_graph"])
def test_fixed_size_motif_counts_match_rows_and_preserve_sampling(output_format, tmp_path):
    creator = _creator()
    baseline = creator.create_ensemble(2, output_format=output_format, ensemble_info=True, seed=7)
    path = tmp_path / "ensemble.json"
    result = creator.create_ensemble(
        2, output_format=output_format, ensemble_info=True, seed=7,
        retain_motif_membership=True, json_file=str(path),
    )
    assert module._metadata_equal(result.chains, baseline.chains)
    assert module._metadata_equal(result.sequences, baseline.sequences)
    assert result.molecular_weights == baseline.molecular_weights
    rows = result.chain_motif_index["rows"]
    assert [row["chain_index"] for row in rows] == [0, 1]
    assert [row["molar_mass"] for row in rows] == result.molecular_weights
    occurrences, presence = Counter(), Counter()
    for row in rows:
        assert row["motif_counts"]
        assert row["motif_exposure"] == sum(count for _, count in row["motif_counts"])
        occurrences.update(dict(row["motif_counts"]))
        presence.update(motif_id for motif_id, _ in row["motif_counts"])
    totals = result.motif_population_statistics
    assert totals["accepted_count"] == 2
    assert {entry["motif_id"]: entry["occurrence_count"] for entry in totals["entries"]} == occurrences
    assert {entry["motif_id"]: entry["chain_presence_count"] for entry in totals["entries"]} == presence
    saved = json.loads(path.read_text())["ensemble"]
    assert saved["chain_motif_index"] == result.chain_motif_index
    assert saved["motif_population_statistics"] == totals


def test_fixed_size_json_only_contains_motif_analytics(tmp_path):
    path = tmp_path / "ensemble.json"
    chains = _creator().create_ensemble(
        2, output_format="smiles", seed=7, json_file=str(path),
        retain_motif_membership=True,
    )
    saved = json.loads(path.read_text())["ensemble"]
    assert saved["chains"] == chains
    assert len(saved["chain_motif_index"]["rows"]) == 2
    assert saved["motif_population_statistics"]["entries"]


def test_fixed_size_single_unit_has_membership_without_path_motifs():
    result = _creator("C").create_ensemble(
        2, ensemble_info=True, seed=7, retain_motif_membership=True,
    )
    assert len(result.chain_motif_index["rows"]) == 2
    assert all(row["motif_counts"] == [] for row in result.chain_motif_index["rows"])
    assert result.motif_population_statistics["accepted_count"] == 2
    assert result.motif_population_statistics["entries"] == []


def test_fixed_size_empty_ensemble_has_available_empty_membership():
    result = _creator().create_ensemble(
        0, ensemble_info=True, seed=7, retain_motif_membership=True,
    )
    assert result.chain_motif_index["rows"] == []
    assert result.chain_motif_index["accepted_count"] == 0
    assert result.motif_population_statistics["entries"] == []


@pytest.mark.parametrize("api", ["fixed", "convergence"])
@pytest.mark.parametrize("output_format", ["smiles", "mol_graph"])
@pytest.mark.parametrize("selection", [{}, {"reservoir_size": 1}, {"representative_distance": 0.15}])
def test_disabled_feature_does_no_feature_specific_work(api, output_format, selection, monkeypatch, tmp_path):
    creator = _creator()

    def unexpected(*args, **kwargs):
        pytest.fail("disabled feature performed topology export or motif analytics")

    for name in (
        "_compact_bond_sequence_record", "_bond_sequence_export_metadata",
        "_canonical_bond_sequence_json_bytes", "_canonical_motif_key",
        "_motif_dictionary_from_counts", "_motif_population_statistics_from_counts",
    ):
        monkeypatch.setattr(module, name, unexpected)
    if api == "fixed":
        monkeypatch.setattr(module, "_unit_path_counts", unexpected)
        monkeypatch.setattr(module, "_defer_chain_conversion", unexpected)
        path = tmp_path / "ensemble.json"
        result = creator.create_ensemble(
            2, ensemble_info=True, output_format=output_format, seed=7,
            json_file=str(path),
        )
        saved = json.loads(path.read_text())["ensemble"]
        assert "chain_motif_index" not in saved
        assert "motif_population_statistics" not in saved
    else:
        # Existing convergence path statistics must still run, independent of
        # topology/motif export. Neither opt-in feature is requested here.
        result = creator.create_ensemble_until_converged(
            batch_size=4, max_samples=4, output_format=output_format, seed=7,
            **selection,
        )
    assert result.chain_motif_index is None
    assert result.motif_population_statistics is None
    assert result.bond_sequences is None


def test_fixed_size_motif_membership_requires_metadata():
    with pytest.raises(ValueError, match="metadata"):
        _creator().create_ensemble(1, retain_motif_membership=True)


def test_minimal_convergence_does_not_build_occurrence_views(monkeypatch):
    creator = _creator()

    def unexpected(*args, **kwargs):
        pytest.fail("metadata-disabled sampling built occurrence/path analytics")

    monkeypatch.setattr(module, "_unit_path_counts", unexpected)
    monkeypatch.setattr(module, "_build_occurrence_adjacency", unexpected)
    result = creator.create_ensemble_until_converged(
        batch_size=2, max_samples=2, seed=7, metadata=False,
        retain_chains=False, retain_sequences=False,
    )
    assert result.chains == []
    assert result.bond_sequences is None
    assert result.chain_motif_index is None
    assert result.motif_population_statistics is None


def test_fixed_size_motif_converts_once_and_aggregates_warnings(monkeypatch):
    original = module._convert_chain
    calls = []

    def convert(*args, **kwargs):
        calls.append(True)
        warnings.warn(
            "Incomplete double-bond directional markers near atoms 1 and 2",
            RuntimeWarning,
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_convert_chain", convert)
    with pytest.warns(RuntimeWarning, match="2 warning event") as caught:
        result = _creator().create_ensemble(
            2, ensemble_info=True, seed=7, retain_motif_membership=True,
        )
    assert len(calls) == 2
    assert len(caught) == 1
    assert len(result.chain_motif_index["rows"]) == 2


def test_fixed_size_motif_totals_follow_accepted_prefix(monkeypatch):
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = creator.create_ensemble(
            2, ensemble_info=True, seed=7, retain_motif_membership=True,
        )
    assert len(result.chains) == 1
    assert result.chain_motif_index["accepted_count"] == 1
    assert result.chain_motif_index["rows"][0]["chain_index"] == 0
    assert result.motif_population_statistics["accepted_count"] == 1
    assert all(entry["chain_presence_count"] == 1 for entry in result.motif_population_statistics["entries"])