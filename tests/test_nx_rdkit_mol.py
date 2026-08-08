# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import threading

import networkx as nx
import pytest

pytest.importorskip("rdkit", reason="RDKit is an optional dependency")
from rdkit import Chem

from g2rins.nx_rdkit_mol import (
    _run_with_big_stack,
    mol_graph_to_rdkit_mol,
    mol_graph_to_smiles,
    rdkit_mol_to_smiles,
)


def _linear_carbon_graph(n_atoms):
    graph = nx.Graph()
    for i in range(n_atoms):
        graph.add_node(i, atomic_num=6, aromatic=False, charge=0)
    for i in range(n_atoms - 1):
        graph.add_edge(i, i + 1, aromatic=False, bond_type=1)
    return graph


def test_run_with_big_stack_returns_result():
    assert _run_with_big_stack(lambda a, b: a + b, 2, 3) == 5


def test_run_with_big_stack_propagates_exception():
    def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _run_with_big_stack(boom)


def test_run_with_big_stack_falls_back_inline_with_warning(monkeypatch):
    def raiser(size):
        raise ValueError("stack size not supported")

    monkeypatch.setattr(threading, "stack_size", raiser)
    with pytest.warns(RuntimeWarning, match="big-stack"):
        assert _run_with_big_stack(lambda a, b: a + b, 2, 3) == 5


def test_rdkit_mol_to_smiles_matches_direct():
    mol = Chem.MolFromSmiles("CCO")
    assert rdkit_mol_to_smiles(mol) == Chem.MolToSmiles(mol)


def test_mol_graph_to_smiles_small():
    graph = _linear_carbon_graph(3)
    assert mol_graph_to_smiles(graph) == "CCC"
    assert mol_graph_to_smiles(graph) == Chem.MolToSmiles(mol_graph_to_rdkit_mol(graph))


def test_association_edge_renders_as_fragment():
    """bond_type 0 (association edge, e.g. an ion pair) adds no covalent bond:
    the two atoms come out as separate "." fragments."""
    graph = nx.Graph()
    graph.add_node(0, atomic_num=11, aromatic=False, charge=1)
    graph.add_node(1, atomic_num=17, aromatic=False, charge=-1)
    graph.add_edge(0, 1, aromatic=False, bond_type=0)
    assert mol_graph_to_rdkit_mol(graph).GetNumBonds() == 0
    assert sorted(mol_graph_to_smiles(graph).split(".")) == ["[Cl-]", "[Na+]"]


def test_mol_graph_to_smiles_huge_chain():
    # Regression test for the 0xC00000FD stack overflow: a chain this long
    # overflows the default stack in RDKit's SMILES writer without the guard.
    n_atoms = 5000
    graph = _linear_carbon_graph(n_atoms)
    assert mol_graph_to_smiles(graph) == "C" * n_atoms