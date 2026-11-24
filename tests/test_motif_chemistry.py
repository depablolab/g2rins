import pytest
from rdkit import Chem

from g2rins.motif_chemistry import motif_chemical_strings


def canonical(smiles):
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smiles)))


def test_complete_motif_assembles_real_bonds_and_explicit_hydrogen():
    result = motif_chemical_strings(
        {"units": ["R0", "I0", "T1"],
         "edges": [[0, 1, 1, 1, "1", False], [0, 2, 2, 1, "1", False]], "boundary": []},
        {"R0": {"psmiles": "[*:1]CO[*:2]"}, "I0": {"psmiles": "C[*:1]"},
         "T1": {"psmiles": "[H][*:1]"}},
    )
    assert canonical(result["psmiles"]) == canonical("CCO")
    assert result["bigsmiles"] == result["psmiles"]
    assert "*" not in result["psmiles"]


def test_branch_boundary_ports_are_unique_local_descriptors():
    from g2rins.smiles import Smiles

    result = motif_chemical_strings(
        {"units": ["R0", "R0"], "edges": [[0, 1, 1, 1, "1", False]],
         "boundary": [[1, 2, "1", False], [1, 3, "1", False]]},
        {"R0": {"psmiles": "C([*:1])([*:2])[*:3]"}},
    )
    mol = Chem.MolFromSmiles(result["psmiles"])
    assert sorted(atom.GetAtomMapNum() for atom in mol.GetAtoms()
                  if atom.GetAtomicNum() == 0) == [1, 2]
    assert "[$1]" in result["bigsmiles"] and "[$2]" in result["bigsmiles"]
    assert str(Smiles.make(result["bigsmiles"])) == result["bigsmiles"]


def test_boundary_preserves_original_direction_and_index():
    result = motif_chemical_strings(
        {"units": ["R0"], "edges": [],
         "boundary": [[0, 1, "1", False], [0, 3, "1", False]]},
        {"R0": {"psmiles": "COCC(C[*:1])(O[*:3])[*:2]",
                "bigsmiles": "[<]CC([<1|0.1|])(COC)O[>]"}},
    )
    assert "[<]" in result["bigsmiles"]
    assert "[>]" in result["bigsmiles"]
    assert "[$" not in result["bigsmiles"]
    assert "*" not in result["bigsmiles"]
    assert canonical(replace_ports_with_h(result["psmiles"])) == canonical("COCC(C)O")


def replace_ports_with_h(smiles):
    import re
    return re.sub(r"\[\*:\d+\]", "[H]", smiles)


def test_boundary_preserves_numbered_direction_not_local_map_number():
    result = motif_chemical_strings(
        {"units": ["R0"], "edges": [], "boundary": [[0, 2, "1", False]]},
        {"R0": {"psmiles": "FC(F)(F)COC(C[*:1])[*:2]",
                "bigsmiles": "[>7]CC([<7])OCC(F)(F)F|15.0|"}},
    )
    assert "[*:1]" in result["psmiles"]
    assert "[<7]" in result["bigsmiles"]
    assert "*" not in result["bigsmiles"]


def test_absent_port_removal_preserves_chirality():
    result = motif_chemical_strings(
        {"units": ["R0"], "edges": [], "boundary": []},
        {"R0": {"psmiles": "N[C@](F)(Cl)[*:1]"}},
    )
    assert canonical(result["psmiles"]) == canonical("N[C@](F)(Cl)[H]")
    assert "*" not in result["psmiles"]


def test_mismatched_source_site_mapping_is_rejected():
    with pytest.raises(ValueError, match="connectors do not match"):
        motif_chemical_strings(
            {"units": ["R0"], "edges": [], "boundary": []},
            {"R0": {"psmiles": "[*:1]CO[*:2]", "bigsmiles": "[<]CC[>]"}},
        )


def test_reordered_native_sites_recover_original_descriptor_identity():
    from g2rins.motif_chemistry import _source_descriptors

    assert _source_descriptors("[*:1]CO[*:2]", "[<]OC[>]") == {1: "[>]", 2: "[<]"}
    assert _source_descriptors(
        "C(CN([*:2])[*:3])O[*:1]", "[>7|10|]N([>7|5|])CCO[<2]",
    ) == {1: "[<2]", 2: "[>7]", 3: "[>7]"}


def test_symmetry_with_different_descriptor_identities_is_rejected():
    from g2rins.motif_chemistry import _source_descriptors

    with pytest.raises(ValueError, match="ambiguous connector identity"):
        _source_descriptors("C(CN([*:2])[*:3])O[*:1]", "[>7]N([<8])CCO[>]")


@pytest.mark.parametrize("psmiles,bigsmiles,ports", [
    ("C(CN([*:2])[*:3])O[*:1]", "[>|10|]N([>|5|])CCO[>]", 3),
    ("C(CN([*:2])[*:3])N(CCN([*:4])[*:5])[*:1]",
     "[>|10|]N([>|5|])CCN([>|5|])CCN([>|5|])[>|10|]", 5),
    ("C(CCN(CCCN([*:5])[*:6])[*:2])CN(CCCN([*:3])[*:4])[*:1]",
     "[>|10|]N([>|5|])CCCN([>|5|])CCCCN([>|5|])CCCN([>|5|])[>|10|]", 6),
    ("C(O[*:1])[C@@H]1O[C@H](O[C@]2(CO[*:2])O[C@H](CO[*:3])"
     "[C@@H](O[*:4])[C@H]2O[*:5])[C@H](O[*:6])[C@@H](O[*:7])[C@H]1O[*:8]",
     "[>]OC[C@H]1O[C@H](O[C@]2(CO[>])O[C@H](CO[>])[C@@H](O[>])"
     "[C@@H]2O[>])[C@H](O[>])[C@@H](O[>])[C@@H]1O[>]", 8),
])
def test_failed_catalog_initiator_ports_preserve_native_chemistry(psmiles, bigsmiles, ports):
    from g2rins.motif_chemistry import _source_descriptors

    assert _source_descriptors(psmiles, bigsmiles) == {
        site: "[>]" for site in range(1, ports + 1)
    }
    result = motif_chemical_strings(
        {"units": ["I0"], "edges": [],
         "boundary": [[0, site, "1", False] for site in range(1, ports + 1)]},
        {"I0": {"psmiles": psmiles, "bigsmiles": bigsmiles}},
    )
    assert canonical(result["psmiles"]) == canonical(psmiles)
    assert result["bigsmiles"].count("[>]") == ports
    assert "*" not in result["bigsmiles"]


def test_inactive_source_connector_does_not_shift_active_site_descriptors():
    result = motif_chemical_strings(
        {"units": ["R0"], "edges": [],
         "boundary": [[0, 1, "1", False], [0, 2, "1", False]]},
        {"R0": {"psmiles": "CCCCC(C[*:1])O[*:2]",
                "bigsmiles": "[<]CC(CCCC)([<1|0.1|])O[>]"}},
    )
    assert "[<]" in result["bigsmiles"] and "[>]" in result["bigsmiles"]
    assert "[<1]" not in result["bigsmiles"]
    assert "*" not in result["bigsmiles"]


@pytest.mark.parametrize("bond,unit,expected", [("2", "C=[*:1]", "C=C"),
                                             ("3", "C#[*:1]", "C#C")])
def test_internal_bond_orders_are_preserved(bond, unit, expected):
    result = motif_chemical_strings(
        {"units": ["R0", "R0"], "edges": [[0, 1, 1, 1, bond, False]], "boundary": []},
        {"R0": {"psmiles": unit}},
    )
    assert canonical(result["psmiles"]) == canonical(expected)


def test_join_preserves_tetrahedral_chirality():
    units = {"R0": {"psmiles": "N[C@](F)(Cl)[*:1]"}, "R1": {"psmiles": "C[*:1]"}}
    result = motif_chemical_strings(
        {"units": ["R0", "R1"], "edges": [[0, 1, 1, 1, "1", False]], "boundary": []}, units,
    )
    assert canonical(result["psmiles"]) == canonical("N[C@](F)(Cl)C")


def test_self_link_closes_ring_without_duplicating_center():
    result = motif_chemical_strings(
        {"units": ["R0"], "edges": [[0, 1, 0, 2, "1", False]], "boundary": []},
        {"R0": {"psmiles": "[*:1]CCC[*:2]"}},
    )
    assert canonical(result["psmiles"]) == canonical("C1CC1")