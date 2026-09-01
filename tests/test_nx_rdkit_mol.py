# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import threading

import networkx as nx
import pytest

import g2rins

pytest.importorskip("rdkit", reason="RDKit is an optional dependency")
from rdkit import Chem

from g2rins.nx_rdkit_mol import (
    _run_with_big_stack,
    mol_graph_to_rdkit_mol,
    mol_graph_to_smiles,
    rdkit_mol_to_smiles,
    rdkit_mol_weight,
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


def test_rdkit_mol_to_smiles_falls_back_when_canonical_ring_labels_are_exhausted(monkeypatch):
    mol = Chem.MolFromSmiles("c1ccccc1")
    direct_mol_to_smiles = Chem.MolToSmiles
    calls = []

    def ring_limited_mol_to_smiles(value, **kwargs):
        calls.append(kwargs)
        if kwargs.get("canonical", True):
            raise ValueError("Too many rings open at once. SMILES cannot be generated.")
        return direct_mol_to_smiles(value, **kwargs)

    monkeypatch.setattr(Chem, "MolToSmiles", ring_limited_mol_to_smiles)
    smiles = rdkit_mol_to_smiles(mol)

    assert calls and calls[0] == {}
    assert calls[1]["canonical"] is False
    root = calls[1].get("rootedAtAtom")
    assert root is None or 0 <= root < mol.GetNumAtoms()
    assert Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=False) == smiles


def test_rdkit_mol_to_smiles_does_not_mask_other_value_errors(monkeypatch):
    mol = Chem.MolFromSmiles("CCO")

    def invalid_mol_to_smiles(_mol, **_kwargs):
        raise ValueError("different serialization failure")

    monkeypatch.setattr(Chem, "MolToSmiles", invalid_mol_to_smiles)
    with pytest.raises(ValueError, match="different serialization failure"):
        rdkit_mol_to_smiles(mol)


def test_benchmark_worker_uses_safe_ring_overflow_fallback(monkeypatch):
    import benchmarks.benchmark_sampling as benchmark_sampling

    direct_mol_to_smiles = benchmark_sampling.Chem.MolToSmiles

    def ring_limited_mol_to_smiles(value, **kwargs):
        if kwargs.get("canonical", True):
            raise ValueError("Too many rings open at once. SMILES cannot be generated.")
        return direct_mol_to_smiles(value, **kwargs)

    monkeypatch.setattr(benchmark_sampling.Chem, "MolToSmiles", ring_limited_mol_to_smiles)

    record = benchmark_sampling._worker(
        "small-linear",
        metadata=True,
        termination="exact",
        max_discards=20,
    )

    assert record["accepted"] is True
    assert record["canonical_smiles_bytes"] > 0


def test_rdkit_mol_to_smiles_falls_back_when_ring_overflow_is_runtime_error(monkeypatch):
    mol = Chem.MolFromSmiles("c1ccccc1")
    direct_mol_to_smiles = Chem.MolToSmiles
    calls = []

    def ring_limited_mol_to_smiles(value, **kwargs):
        calls.append(kwargs)
        if kwargs.get("canonical", True):
            raise RuntimeError("Too many rings open at once. SMILES cannot be generated.")
        return direct_mol_to_smiles(value, **kwargs)

    monkeypatch.setattr(Chem, "MolToSmiles", ring_limited_mol_to_smiles)
    smiles = rdkit_mol_to_smiles(mol)

    assert calls and calls[0] == {}
    assert calls[1]["canonical"] is False
    root = calls[1].get("rootedAtAtom")
    assert root is None or 0 <= root < mol.GetNumAtoms()
    assert Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=False) == smiles


def test_rdkit_mol_to_smiles_handles_benzimidazole_like_cyclic_monomer(monkeypatch):
    # This is the cyclic benzimidazole core extracted from the reported G2RINS
    # input string {[] [>]c1nc2ccc([<])cc2[nH]1; ; []}|...|; the exact stochastic
    # object cannot be sampled directly in isolation, but the monomer core itself
    # exercises the same ring-heavy serialization path that previously failed.
    mol = Chem.MolFromSmiles("c1nc2ccccc2[nH]1")
    direct_mol_to_smiles = Chem.MolToSmiles
    calls = []

    def ring_limited_mol_to_smiles(value, **kwargs):
        calls.append(kwargs)
        if kwargs.get("canonical", True):
            raise ValueError("Too many rings open at once. SMILES cannot be generated.")
        return direct_mol_to_smiles(value, **kwargs)

    monkeypatch.setattr(Chem, "MolToSmiles", ring_limited_mol_to_smiles)
    smiles = rdkit_mol_to_smiles(mol)

    assert calls and calls[0] == {}
    assert calls[1]["canonical"] is False
    root = calls[1].get("rootedAtAtom")
    assert root is None or 0 <= root < mol.GetNumAtoms()
    assert "[nH]" in smiles or "nH" in smiles
    assert Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=False) == smiles


def test_mol_graph_to_smiles_small():
    graph = _linear_carbon_graph(3)
    assert mol_graph_to_smiles(graph) == "CCC"
    assert mol_graph_to_smiles(graph) == Chem.MolToSmiles(mol_graph_to_rdkit_mol(graph))


def test_mol_graph_to_smiles_publishes_every_native_stage():
    stages = []
    graph = _linear_carbon_graph(3)

    assert mol_graph_to_smiles(graph, native_stage_callback=stages.append) == "CCC"
    assert stages == ["build", "sanitize", "property-cache", "smiles"]


def test_rdkit_mol_weight_reuses_molecule_and_publishes_stage():
    stages = []
    mol = Chem.MolFromSmiles("CCO")

    assert rdkit_mol_weight(mol, stages.append) == pytest.approx(46.069)
    assert stages == ["descriptor-molwt"]


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


def test_mol_graph_to_smiles_keeps_tetrahedral_chirality_from_graph_tokens():
    source = "N[C@H](F)Cl"
    graph = g2rins.G2rins.make(source).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    smiles = mol_graph_to_smiles(graph)

    assert "@" in smiles
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(source))
    actual = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert actual == expected


def test_mol_graph_to_smiles_keeps_double_bond_geometry_from_direction_tokens():
    source = "F/C=C/F"
    graph = g2rins.G2rins.make(source).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    smiles = mol_graph_to_smiles(graph)

    assert "/" in smiles or "\\" in smiles
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(source))
    actual = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert actual == expected


def test_mol_graph_to_rdkit_mol_warns_on_incomplete_double_bond_directional_markers():
    graph = nx.MultiDiGraph()
    graph.add_node(0, atomic_num=9, aromatic=False, charge=0)
    graph.add_node(1, atomic_num=6, aromatic=False, charge=0)
    graph.add_node(2, atomic_num=6, aromatic=False, charge=0)
    graph.add_node(3, atomic_num=9, aromatic=False, charge=0)
    graph.add_edge(0, 1, bond_type=1, aromatic=False, bond_symbol_raw="/")
    graph.add_edge(1, 2, bond_type=2, aromatic=False)
    graph.add_edge(2, 3, bond_type=1, aromatic=False)

    with pytest.warns(RuntimeWarning, match="Incomplete double-bond directional markers"):
        mol = mol_graph_to_rdkit_mol(graph)
    assert mol.GetNumBonds() == 3


def test_mol_graph_to_rdkit_mol_warns_on_ambiguous_double_bond_directional_markers():
    graph = nx.MultiDiGraph()
    graph.add_node(0, atomic_num=9, aromatic=False, charge=0)
    graph.add_node(1, atomic_num=6, aromatic=False, charge=0)
    graph.add_node(2, atomic_num=6, aromatic=False, charge=0)
    graph.add_node(3, atomic_num=9, aromatic=False, charge=0)
    graph.add_node(4, atomic_num=17, aromatic=False, charge=0)
    graph.add_edge(0, 1, bond_type=1, aromatic=False, bond_symbol_raw="/")
    graph.add_edge(4, 1, bond_type=1, aromatic=False, bond_symbol_raw="\\")
    graph.add_edge(1, 2, bond_type=2, aromatic=False)
    graph.add_edge(2, 3, bond_type=1, aromatic=False, bond_symbol_raw="/")

    with pytest.warns(
        RuntimeWarning,
        match="Ambiguous double-bond directional markers.*stereoinformation was discarded",
    ):
        mol = mol_graph_to_rdkit_mol(graph)
    assert mol.GetNumBonds() == 4
    double_bonds = [bond for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE]
    assert len(double_bonds) == 1
    assert double_bonds[0].GetStereo() == Chem.BondStereo.STEREONONE


def test_mol_graph_to_rdkit_mol_warns_on_conflicting_double_bond_directional_markers():
    graph = nx.MultiDiGraph()
    graph.add_node(0, atomic_num=9, aromatic=False, charge=0)
    graph.add_node(1, atomic_num=6, aromatic=False, charge=0)
    graph.add_node(2, atomic_num=6, aromatic=False, charge=0)
    graph.add_node(3, atomic_num=9, aromatic=False, charge=0)
    graph.add_edge(0, 1, bond_type=1, aromatic=False, bond_symbol_raw="/")
    graph.add_edge(0, 1, bond_type=1, aromatic=False, bond_symbol_raw="\\")
    graph.add_edge(1, 2, bond_type=2, aromatic=False)
    graph.add_edge(2, 3, bond_type=1, aromatic=False, bond_symbol_raw="/")

    with pytest.warns(
        RuntimeWarning,
        match="Conflicting double-bond directional markers.*stereoinformation was discarded",
    ):
        mol = mol_graph_to_rdkit_mol(graph)
    assert mol.GetNumBonds() == 3
    double_bonds = [bond for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE]
    assert len(double_bonds) == 1
    assert double_bonds[0].GetStereo() == Chem.BondStereo.STEREONONE