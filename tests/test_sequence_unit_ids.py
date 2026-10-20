"""Stable unit identities must survive output conversion and retention."""

from collections import Counter
import json

import pytest

from g2rins import G2rins


SOURCE = "{[] [>]OCC(C[<])O[>]; [>]OCCO[>]; [<][H] []}|schulz_zimm(650, 450)|"


@pytest.mark.parametrize("output_format", ["smiles", "mol_graph"])
@pytest.mark.parametrize("parallel", [False, True])
def test_fixed_ensemble_preserves_unit_ids(output_format, parallel):
    creator = G2rins.make(SOURCE).get_graph_creator().get_ensemble_creator()
    result = creator.create_ensemble(4, output_format=output_format, ensemble_info=True,
                                     seed=7, parallel=parallel, n_workers=2 if parallel else None)
    totals = Counter()
    for raw, labels in zip(result.sequences, result.sequence_unit_ids, strict=True):
        assert list(map(len, raw)) == list(map(len, labels))
        counts = Counter(unit for segment in labels for unit in segment)
        assert counts["I0"] == 1
        assert counts["T0"] > 0
        totals.update(counts)
    assert totals == Counter({unit: info["count"] for unit, info in result.units.items()})


@pytest.mark.parametrize("retain_sequences", [False, True])
def test_convergence_identity_retention_matches_sequences(retain_sequences):
    creator = G2rins.make(SOURCE).get_graph_creator().get_ensemble_creator()
    result = creator.create_ensemble_until_converged(
        batch_size=4, max_samples=8, window=3, seed=7,
        representative_distance=1e9, retain_sequences=retain_sequences,
    )
    if retain_sequences:
        assert len(result.sequence_unit_ids) == len(result.sequences) == 1
        assert sum(segment.count("I0") for segment in result.sequence_unit_ids[0]) == 1
    else:
        assert result.sequences == []
        assert result.sequence_unit_ids is None


@pytest.mark.parametrize("output_format", ["smiles", "mol_graph"])
def test_json_exports_keep_unit_identity_sidecar(output_format, tmp_path):
    creator = G2rins.make(SOURCE).get_graph_creator().get_ensemble_creator()
    path = tmp_path / "ensemble.json"
    result = creator.create_ensemble(2, output_format=output_format, ensemble_info=True,
                                     seed=7, json_file=str(path))
    saved = json.loads(path.read_text())["ensemble"]
    assert saved["sequence_unit_ids"] == result.sequence_unit_ids
    assert [list(map(len, chain)) for chain in saved["sequences"]] == [
        list(map(len, chain)) for chain in saved["sequence_unit_ids"]
    ]