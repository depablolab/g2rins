# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import warnings

import networkx as nx
import numpy as np
import pytest

import g2rins

pytest.importorskip("rdkit", reason="RDKit is an optional dependency")
from rdkit import Chem


def _sample_mol_graph(g2rins_text, seed=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(g2rins_text).get_graph_creator().get_ensemble_creator()
        return creator.sample_mol_graph(rng=np.random.default_rng(seed))


def test_stochastic_double_bond_keeps_ez_stereo_in_sampled_molecule():
    text = "C{[>][<]C/C=C/C[>];;[<]}|uniform(2,2)|[H]"
    mol_graph = _sample_mol_graph(text)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)

    assert not [w for w in caught if "double-bond directional markers" in str(w.message)]
    double_bonds = [bond for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE]
    assert double_bonds
    assert any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in double_bonds)


def test_stochastic_chiral_center_keeps_chirality_in_sampled_molecule():
    text = "{[] [>]C[C@H](F)O[<]; [H][>]; [<]Cl []}|uniform(1,1)|"
    mol_graph = _sample_mol_graph(text)

    smiles = g2rins.mol_graph_to_smiles(mol_graph)
    mol = Chem.MolFromSmiles(smiles)

    assert "@" in smiles
    assert any(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms())


def test_sampled_graph_keeps_stereo_metadata_fields():
    directional = _sample_mol_graph("C{[>][<]C/C=C/C[>];;[<]}|uniform(2,2)|[H]")
    chiral = _sample_mol_graph("{[] [>]C[C@H](F)O[<]; [H][>]; [<]Cl []}|uniform(1,1)|")

    assert any(data.get("bond_symbol_raw") in {"/", "\\"} for _u, _v, data in directional.edges(data=True))
    assert any(data.get("atom_chiral_token") is not None for _node, data in chiral.nodes(data=True))


def test_incomplete_directional_markers_warn_on_undirected_graph_input():
    graph = nx.Graph()
    graph.add_node(0, atomic_num=9, aromatic=False, charge=0)
    graph.add_node(1, atomic_num=6, aromatic=False, charge=0)
    graph.add_node(2, atomic_num=6, aromatic=False, charge=0)
    graph.add_node(3, atomic_num=9, aromatic=False, charge=0)
    graph.add_edge(0, 1, bond_type=1, aromatic=False, bond_symbol_raw="/")
    graph.add_edge(1, 2, bond_type=2, aromatic=False)
    graph.add_edge(2, 3, bond_type=1, aromatic=False)

    with pytest.warns(RuntimeWarning, match="Incomplete double-bond directional markers"):
        mol = g2rins.mol_graph_to_rdkit_mol(graph)
    assert mol.GetNumBonds() == 3
