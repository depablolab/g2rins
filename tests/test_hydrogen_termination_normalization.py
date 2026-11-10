"""Hydrogen caps and implicit valence H have identical structural exports."""

from copy import deepcopy
import warnings

import networkx as nx
import pytest
from rdkit import Chem

import g2rins
from g2rins.bond_sequences import normalize_hydrogen_termini
from g2rins.neighbor_motifs import CenteredMotifStatistics


def topology(units, edges, components=None):
    return {
        "schema": "bond-sequence-topology/v1",
        "occurrences": [[oid, unit] for oid, unit in enumerate(units)],
        "edges": [[a, str(sa), b, str(sb), "1", False] for a, sa, b, sb in edges],
        "components": components if components is not None else [
            {"root": 0, "occurrence_ids": list(range(len(units)))}
        ],
    }


def test_branch_and_outer_occupied_sites_are_normalized_before_counting():
    implicit = topology(["I0", "R0", "R1", "T1"],
                        [(0, 1, 1, 1), (1, 2, 2, 1), (1, 3, 3, 1)])
    explicit = topology(["I0", "R0", "T0", "R1", "T1", "T0"],
                        [(0, 1, 1, 1), (1, 4, 2, 1), (1, 2, 3, 1),
                         (1, 3, 4, 1), (3, 2, 5, 1)])
    original = deepcopy(explicit)
    normalized = normalize_hydrogen_termini(explicit, {
        "T0": {"psmiles": "[H][*:1]"}, "T1": {"psmiles": "C[*:1]"},
    })
    assert normalized == implicit
    assert explicit == original
    assert normalize_hydrogen_termini(normalized, {"T0": {"psmiles": "[H][*:1]"}}) is normalized
    stats = CenteredMotifStatistics(hydrogen_termination="implicit/v1")
    stats.update(normalized)
    stats.update(implicit)
    rows = stats.finalize()["rows"]
    assert all(row["count"] == row["chain_count"] == 2 for row in rows)
    assert sum(row["count"] for row in rows) == 8
    center = next(row for row in rows if row["center_unit"] == "R0")
    assert center["motif"] == "R0{1:1.I0.null;2:1.R1.null;3:1.T1.null}"
    assert center["topology"]["boundary"] == []
    for renderer in (g2rins.render_bond_sequence_paths, g2rins.render_bond_sequence_branches):
        assert renderer(normalized) == renderer(implicit)


@pytest.mark.parametrize("unit,psmiles", [
    ("T0", "[2H][*:1]"), ("T0", "[3H][*:1]"), ("T0", "[1H][*:1]"),
    ("T0", "[H-][*:1]"), ("T0", "O[*:1]"), ("T0", "[HH]"),
    ("I0", "[H][*:1]"), ("R0", "[H][*:1]"),
])
def test_isotopes_other_end_groups_and_nonterminal_units_remain(unit, psmiles):
    record = topology(["R1", unit], [(0, 2, 1, 1)])
    assert normalize_hydrogen_termini(record, {unit: {"psmiles": psmiles}}) is record


def test_isolated_hydrogen_hydrogen_molecules_and_non_single_edges_remain():
    units = {"T0": {"psmiles": "[H][*:1]"}, "I0": {"psmiles": "[H][*:1]"}}
    for record in (topology(["T0"], []),
                   topology(["I0", "T0"], [(0, 1, 1, 1)])):
        assert normalize_hydrogen_termini(record, units) is record
    record = topology(["R0", "T0"], [(0, 1, 1, 1)])
    record["edges"][0][4] = "2"
    assert normalize_hydrogen_termini(record, units) is record


def test_removed_root_remaps_forest_and_preserves_cycles_and_other_roots():
    record = topology(["T0", "R0", "R0", "R0", "I0"],
                      [(0, 1, 1, 3), (1, 2, 2, 1), (2, 2, 3, 1), (1, 1, 3, 2)],
                      [{"root": 0, "occurrence_ids": [0, 1, 2, 3]},
                       {"root": 4, "occurrence_ids": [4]}])
    result = normalize_hydrogen_termini(record, {"T0": {"psmiles": "[H][*:1]"}})
    assert result["components"] == [{"root": 0, "occurrence_ids": [0, 1, 2]},
                                     {"root": 3, "occurrence_ids": [3]}]
    assert len(result["edges"]) == 3
    assert [row[0] for row in result["occurrences"]] == list(range(4))


@pytest.mark.parametrize("output_format", ["compact", "branches", "paths"])
@pytest.mark.parametrize("selection", [{}, {"representative_distance": 1e9}])
def test_native_explicit_and_implicit_hydrogen_have_identical_exports(output_format, selection):
    results = []
    for terminal in ("[<][H]", ""):
        source = "{[] [<]CC[>]; C[>]; " + terminal + " []}|gauss(250, 1)|"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            creator = g2rins.G2rins.make(source).get_graph_creator().get_ensemble_creator()
            result = creator.create_ensemble_until_converged(
                batch_size=3, max_samples=3, window=10, output_format="smiles",
                metadata=True, retain_bond_sequences=True, retain_motif_membership=True,
                bond_sequence_format=output_format, seed=7, **selection,
            )
            # Representation opt-ins must not change generation or raw statistics.
            baseline = creator.create_ensemble_until_converged(
                batch_size=3, max_samples=3, window=10, output_format="smiles",
                metadata=True, seed=7, **selection,
            )
        for attribute in ("chains", "molecular_weights", "bonds", "sequences",
                          "representative_counts", "sequence_unit_ids"):
            assert getattr(result, attribute) == getattr(baseline, attribute)
        assert result.units.keys() == baseline.units.keys()
        for unit, info in result.units.items():
            other = baseline.units[unit]
            assert {key: value for key, value in info.items() if key != "subgraph"} == {
                key: value for key, value in other.items() if key != "subgraph"
            }
            assert nx.utils.graphs_equal(info["subgraph"], other["subgraph"])
        assert result.bond_sequence_metadata["hydrogen_termination"] == "implicit/v1"
        assert result.next_neighbor_motif_statistics["hydrogen_termination"] == "implicit/v1"
        assert all(row["center_unit"] != "T0" for row in result.next_neighbor_motif_statistics["rows"])
        results.append(result)
    explicit, implicit = results
    for attribute in ("bond_sequences", "bond_sequence_parents", "bond_sequence_weights",
                      "next_neighbor_motif_statistics"):
        assert getattr(explicit, attribute) == getattr(implicit, attribute)
    def chemistry(result):
        return [Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smiles)))
                for smiles in result.chains]
    assert chemistry(explicit) == chemistry(implicit)