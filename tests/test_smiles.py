# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import json

import numpy as np
import pytest

import g2rins


def test_smiles_parsing(chembl_smi_list):
    for smi in chembl_smi_list:
        if len(smi) > 0:
            smiles_instance = g2rins.G2rins.make(smi)
            assert smi == smiles_instance.generate_string(True)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_smiles_weight(n, chembl_smi_list):
    rng = np.random.default_rng()
    no_dot_smi = []
    for smi in chembl_smi_list:
        if "." not in smi and len(smi) > 0:
            no_dot_smi.append(smi)

    for i in range(len(no_dot_smi) // n - 1):
        smis = no_dot_smi[i * n : (i + 1) * n]
        system_string = ""
        total_mw = 0.0
        for smi in smis:
            molw = np.round(rng.uniform(1.0, 1e5), 1)
            system_string += f"{smi}.|{molw}|"
            total_mw += molw
        print(system_string)
        g2rins_object = g2rins.G2rins.make(system_string)
        for mol in g2rins_object.mol_molecular_weight_map:
            print("x", mol, g2rins_object.mol_molecular_weight_map[mol])
        assert abs(total_mw - g2rins_object.total_molecular_weight) < 1e-6


def _rdkit_mol_from_g2rins(text, seed):
    Chem = pytest.importorskip("rdkit.Chem")
    ensemble_creator = g2rins.G2rins.make(text).get_graph_creator().get_ensemble_creator()
    mol_graph = ensemble_creator.sample_mol_graph(rng=np.random.default_rng(seed))
    return Chem, g2rins.mol_graph_to_rdkit_mol(mol_graph)


@pytest.mark.filterwarnings("ignore")  # the backbone alkene triggers a generation warning that is not relevant here
@pytest.mark.parametrize(
    "text, expected",
    [
        ("[H]{[>][<]C=CC[>];;[<]}|uniform(100,100)|[H]", ("STEREONONE", "STEREOANY")),
        ("[H]{[>][<]C/C=C/C[>];;[<]}|uniform(100,100)|[H]", ("STEREOE",)),
        ("[H]{[>][<]C\\C=C\\C[>];;[<]}|uniform(100,100)|[H]", ("STEREOE",)),
        ("[H]{[>][<]C\\C=C/C[>];;[<]}|uniform(100,100)|[H]", ("STEREOZ",)),
        ("[H]{[>][<]C/C=C\\C[>];;[<]}|uniform(100,100)|[H]", ("STEREOZ",)),
    ],
    ids=["unspecified", "trans", "trans_reverse", "cis", "cis_reverse"],
)
def test_polymer_alkene_stereo(text, expected):
    """Alkene E/Z written with / and \\ in a repeat unit survives into the sampled RDKit molecule."""
    Chem, mol = _rdkit_mol_from_g2rins(text, seed=0)
    doubles = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    assert len(doubles) >= 1
    expected = {getattr(Chem.BondStereo, name) for name in expected}
    if expected == {Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY}:
        assert all(db.GetStereo() in expected for db in doubles)
    else:
        assert any(db.GetStereo() in expected for db in doubles)


def test_polymer_chiral_center_isotactic_polypropylene():
    """Atom-level chirality ([C@H]) is preserved in isotactic polypropylene."""
    text = "[H]{[>][<]C[C@H](C)[>];;[<]}|uniform(100,100)|[H]"
    Chem, mol = _rdkit_mol_from_g2rins(text, seed=0)
    chiral_atoms = [a for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
    assert len(chiral_atoms) >= 1, "Expected at least one chiral center in isotactic polypropylene"
    chiral_tags = [a.GetChiralTag() for a in chiral_atoms]
    assert all(tag == chiral_tags[0] for tag in chiral_tags), "All stereocenters should have same configuration (isotactic)"


def test_polymer_chiral_center_syndiotactic_polypropylene():
    """Atom-level chirality ([C@H] and [C@@H]) is preserved and alternates in syndiotactic polypropylene."""
    text = "C{[>][<|0 0 0 1|]C[C@H](C)[>|0 0 1 0|], [<|0 1 0 0|]C[C@@H](C)[>|1 0 0 0|];;[<]}|uniform(100,100)|[H]"
    Chem, mol = _rdkit_mol_from_g2rins(text, seed=0)
    chiral_atoms = [a for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
    assert len(chiral_atoms) >= 1, "Expected at least one chiral center in syndiotactic polypropylene"
    for i in range(len(chiral_atoms) - 1):
        assert chiral_atoms[i].GetChiralTag() != chiral_atoms[i + 1].GetChiralTag(), "Stereocenters should alternate configuration (syndiotactic)"


def _sampled_canonical_smiles(text, seed=0):
    Chem, mol = _rdkit_mol_from_g2rins(text, seed)
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def test_stereo_round_trip(chembl_smi_list):
    """A sampled molecule keeps the tetrahedral and double-bond stereo of its SMILES, whatever order the sampler visits the atoms."""
    Chem = pytest.importorskip("rdkit.Chem")
    for smi in chembl_smi_list:
        # Bare hydrogens and radicals are not reproduced by hydrogen inference and are out of scope here.
        if len(smi) == 0 or "." in smi or "[H]" in smi or "[CH2]" in smi:
            continue
        assert _sampled_canonical_smiles(smi) == Chem.MolToSmiles(Chem.MolFromSmiles(smi)), smi


def test_polymer_chirality_independent_of_entry_side():
    """The same chiral repeat unit entered from its left or its right connector yields the same molecule, not its enantiomer."""
    left_entry = _sampled_canonical_smiles("[H]{[>][<]C[C@H](C)[>];;[<]}|uniform(300,300)|[H]")
    right_entry = _sampled_canonical_smiles("[H]{[<][<]C[C@H](C)[>];;[>]}|uniform(300,300)|[H]")
    assert "@" in left_entry
    assert left_entry == right_entry


def test_unit_psmiles_keeps_chirality(tmp_path):
    """Per-unit P-SMILES in the JSON export keep the configuration of a chiral atom, also when it is the connection atom."""
    Chem = pytest.importorskip("rdkit.Chem")
    cases = {
        "[H]{[>][<]C[C@H](C)[>];;[<]}|uniform(300,300)|[H]": "[*:1]C[C@H](C)[*:2]",
        "[H]{[>][<][C@H](C)C[>];;[<]}|uniform(300,300)|[H]": "[*:1][C@H](C)C[*:2]",
    }
    for text, expected in cases.items():
        json_file = tmp_path / "ensemble.json"
        ensemble_creator = g2rins.G2rins.make(text).get_graph_creator().get_ensemble_creator()
        ensemble_creator.create_ensemble(1, output_format="smiles", json_file=str(json_file), seed=0)
        with open(json_file) as file_handle:
            psmiles = json.load(file_handle)["ensemble"]["units"]["R0"]["psmiles"]
        assert Chem.MolToSmiles(Chem.MolFromSmiles(psmiles)) == Chem.MolToSmiles(Chem.MolFromSmiles(expected)), text
