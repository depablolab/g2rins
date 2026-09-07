# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import threading
import warnings

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


def _direct_writer_unsupported_mol():
    """Return a molecule whose dative bond deliberately needs RDKit output."""
    return Chem.MolFromSmiles("[NH3]->[Cu+2].C1CCCCC1C2CCCCC2")


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


def test_rdkit_mol_to_smiles_defaults_to_native_noncanonical(monkeypatch):
    mol = Chem.MolFromSmiles("CCO")
    direct_mol_to_smiles = Chem.MolToSmiles
    calls = []

    def recorded_smiles(value, **kwargs):
        calls.append(kwargs)
        return direct_mol_to_smiles(value, **kwargs)

    monkeypatch.setattr(Chem, "MolToSmiles", recorded_smiles)

    assert rdkit_mol_to_smiles(mol) == direct_mol_to_smiles(
        mol,
        canonical=False,
    )
    assert calls == [{"canonical": False}]


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
    smiles = rdkit_mol_to_smiles(mol, smiles_policy="auto")

    assert calls and calls[0] == {}
    assert len(calls) == 1
    assert direct_mol_to_smiles(Chem.MolFromSmiles(smiles)) == direct_mol_to_smiles(mol)


def test_rdkit_mol_to_smiles_tries_alternate_roots_after_noncanonical_overflow(monkeypatch):
    mol = _direct_writer_unsupported_mol()
    direct_mol_to_smiles = Chem.MolToSmiles
    successful_root = mol.GetNumAtoms() // 2
    calls = []

    def traversal_limited_mol_to_smiles(value, **kwargs):
        calls.append(kwargs)
        if kwargs.get("canonical", True) or kwargs.get("rootedAtAtom") != successful_root:
            raise ValueError("Too many rings open at once. SMILES cannot be generated.")
        return direct_mol_to_smiles(value, **kwargs)

    monkeypatch.setattr(Chem, "MolToSmiles", traversal_limited_mol_to_smiles)
    smiles = rdkit_mol_to_smiles(mol, smiles_policy="auto")

    assert calls[0] == {}
    assert all(call.get("canonical") is False for call in calls[1:])
    assert [call.get("rootedAtAtom") for call in calls[1:]][-1] == successful_root
    assert direct_mol_to_smiles(Chem.MolFromSmiles(smiles)) == direct_mol_to_smiles(mol)


def test_rdkit_mol_to_smiles_renumbers_atoms_when_every_original_root_overflows(monkeypatch):
    mol = _direct_writer_unsupported_mol()
    direct_mol_to_smiles = Chem.MolToSmiles
    calls = []

    def atom_order_limited_mol_to_smiles(value, **kwargs):
        calls.append((value, kwargs))
        if kwargs.get("canonical", True) or value is mol:
            raise ValueError("Too many rings open at once. SMILES cannot be generated.")
        return direct_mol_to_smiles(value, **kwargs)

    monkeypatch.setattr(Chem, "MolToSmiles", atom_order_limited_mol_to_smiles)
    smiles = rdkit_mol_to_smiles(mol, smiles_policy="auto")

    assert calls[0] == (mol, {})
    assert any(value is not mol for value, _kwargs in calls)
    assert direct_mol_to_smiles(Chem.MolFromSmiles(smiles)) == direct_mol_to_smiles(mol)


def test_rdkit_mol_to_smiles_uses_seeded_random_traversal_for_large_overflow(monkeypatch):
    import g2rins.nx_rdkit_mol as nx_rdkit_mol
    from rdkit import rdBase

    mol = _direct_writer_unsupported_mol()
    direct_mol_to_smiles = Chem.MolToSmiles
    random_seeds = []
    reset_seeds = []

    def traversal_limited_mol_to_smiles(_value, **_kwargs):
        raise ValueError("Too many rings open at once. SMILES cannot be generated.")

    def seeded_random_smiles(value, count, randomSeed):
        assert value is mol
        assert count == 1
        assert randomSeed == 0
        random_seeds.append(randomSeed)
        if len(random_seeds) == 1:
            raise ValueError("Too many rings open at once. SMILES cannot be generated.")
        return [direct_mol_to_smiles(value, canonical=False)]

    monkeypatch.setattr(nx_rdkit_mol, "_BIG_STACK_ATOM_THRESHOLD", 1)
    monkeypatch.setattr(Chem, "MolToSmiles", traversal_limited_mol_to_smiles)
    monkeypatch.setattr(Chem, "MolToRandomSmilesVect", seeded_random_smiles)
    monkeypatch.setattr(rdBase, "SeedRandomNumberGenerator", reset_seeds.append)

    smiles = rdkit_mol_to_smiles(mol, smiles_policy="auto")

    assert reset_seeds == [1, 2]
    assert random_seeds == [0, 0]
    assert direct_mol_to_smiles(Chem.MolFromSmiles(smiles)) == direct_mol_to_smiles(mol)


def test_rdkit_mol_to_smiles_fast_policy_uses_native_noncanonical_call(monkeypatch):
    mol = Chem.MolFromSmiles("C1CCCCC1")
    direct_mol_to_smiles = Chem.MolToSmiles
    calls = []

    def recorded_smiles(value, **kwargs):
        calls.append(kwargs)
        return direct_mol_to_smiles(value, **kwargs)

    monkeypatch.setattr(Chem, "MolToSmiles", recorded_smiles)
    smiles = rdkit_mol_to_smiles(mol, smiles_policy="fast")

    assert calls == [{"canonical": False}]
    assert Chem.MolFromSmiles(smiles) is not None


def test_rdkit_mol_to_smiles_fast_policy_uses_noncanonical_rdkit_for_unsupported_features(monkeypatch):
    mol = _direct_writer_unsupported_mol()
    direct_mol_to_smiles = Chem.MolToSmiles
    calls = []

    def recorded_smiles(value, **kwargs):
        calls.append(kwargs)
        return direct_mol_to_smiles(value, **kwargs)

    monkeypatch.setattr(Chem, "MolToSmiles", recorded_smiles)
    smiles = rdkit_mol_to_smiles(mol, smiles_policy="fast")

    assert calls == [{"canonical": False}]
    assert direct_mol_to_smiles(Chem.MolFromSmiles(smiles)) == direct_mol_to_smiles(mol)


def test_rdkit_mol_to_smiles_canonical_policy_disables_fallback(monkeypatch):
    mol = Chem.MolFromSmiles("C1CCCCC1")

    def ring_limited_smiles(*_args, **_kwargs):
        raise ValueError("Too many rings open at once. SMILES cannot be generated.")

    monkeypatch.setattr(Chem, "MolToSmiles", ring_limited_smiles)
    with pytest.raises(ValueError, match="rings open at once"):
        rdkit_mol_to_smiles(mol, smiles_policy="canonical")


def test_rdkit_mol_to_smiles_rejects_unknown_policy():
    with pytest.raises(ValueError, match="Unsupported SMILES policy"):
        rdkit_mol_to_smiles(Chem.MolFromSmiles("CC"), smiles_policy="quickish")


def test_rdkit_mol_to_smiles_uses_extended_ring_labels_after_random_overflow(monkeypatch):
    import g2rins.nx_rdkit_mol as nx_rdkit_mol

    mol = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")

    def ring_limited_smiles(_value, *_args, **_kwargs):
        raise ValueError("Too many rings open at once. SMILES cannot be generated.")

    monkeypatch.setattr(nx_rdkit_mol, "_BIG_STACK_ATOM_THRESHOLD", 1)
    monkeypatch.setattr(nx_rdkit_mol, "_RANDOM_SMILES_ATTEMPTS", 1)
    monkeypatch.setattr(Chem, "MolToSmiles", ring_limited_smiles)
    monkeypatch.setattr(Chem, "MolToRandomSmilesVect", ring_limited_smiles)

    smiles = rdkit_mol_to_smiles(mol, smiles_policy="auto")
    reparsed = Chem.MolFromSmiles(smiles)

    assert reparsed is not None
    assert reparsed.GetNumAtoms() == mol.GetNumAtoms()
    assert reparsed.GetNumBonds() == mol.GetNumBonds()


@pytest.mark.parametrize(
    "isomeric_smiles",
    [
        "O[C@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]1O",
        "[C@](F)(Cl)(Br)I",
        "[C@H](F)(Cl)Br",
        "F/C=C/F",
        "F/C=C\\F",
    ],
)
def test_extended_ring_label_writer_preserves_stereochemistry(
    monkeypatch,
    isomeric_smiles,
):
    import g2rins.nx_rdkit_mol as nx_rdkit_mol

    mol = Chem.MolFromSmiles(isomeric_smiles)
    direct_mol_to_smiles = Chem.MolToSmiles

    def ring_limited_smiles(_value, *_args, **_kwargs):
        raise ValueError("Too many rings open at once. SMILES cannot be generated.")

    monkeypatch.setattr(nx_rdkit_mol, "_BIG_STACK_ATOM_THRESHOLD", 1)
    monkeypatch.setattr(nx_rdkit_mol, "_RANDOM_SMILES_ATTEMPTS", 1)
    monkeypatch.setattr(Chem, "MolToSmiles", ring_limited_smiles)
    monkeypatch.setattr(Chem, "MolToRandomSmilesVect", ring_limited_smiles)

    n_atoms = mol.GetNumAtoms()
    atom_orders = [
        list(range(n_atoms)),
        list(reversed(range(n_atoms))),
        list(range(1, n_atoms)) + [0],
    ]
    for atom_order in atom_orders:
        reordered = Chem.RenumberAtoms(mol, atom_order)
        smiles = rdkit_mol_to_smiles(reordered, smiles_policy="auto")
        reparsed = Chem.MolFromSmiles(smiles)

        assert reparsed is not None
        assert direct_mol_to_smiles(
            reparsed,
            canonical=True,
            isomericSmiles=True,
        ) == direct_mol_to_smiles(
            reordered,
            canonical=True,
            isomericSmiles=True,
        )


def test_extended_ring_label_writer_preserves_stereocyclic_polymer(monkeypatch):
    import g2rins.nx_rdkit_mol as nx_rdkit_mol

    repeat = Chem.MolFromSmiles("N[C@H]1CCCO1")
    repeat_size = repeat.GetNumAtoms()
    mol = repeat
    for _ in range(23):
        mol = Chem.CombineMols(mol, repeat)
    editable = Chem.RWMol(mol)
    for repeat_index in range(23):
        linker = editable.AddAtom(Chem.Atom(6))
        editable.AddBond(
            repeat_index * repeat_size,
            linker,
            Chem.BondType.SINGLE,
        )
        editable.AddBond(
            linker,
            (repeat_index + 1) * repeat_size,
            Chem.BondType.SINGLE,
        )
    mol = editable.GetMol()
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    direct_mol_to_smiles = Chem.MolToSmiles

    def ring_limited_smiles(_value, *_args, **_kwargs):
        raise ValueError("Too many rings open at once. SMILES cannot be generated.")

    monkeypatch.setattr(nx_rdkit_mol, "_BIG_STACK_ATOM_THRESHOLD", 1)
    monkeypatch.setattr(nx_rdkit_mol, "_RANDOM_SMILES_ATTEMPTS", 1)
    monkeypatch.setattr(Chem, "MolToSmiles", ring_limited_smiles)
    monkeypatch.setattr(Chem, "MolToRandomSmilesVect", ring_limited_smiles)

    smiles = rdkit_mol_to_smiles(mol, smiles_policy="auto")
    reparsed = Chem.MolFromSmiles(smiles)

    assert reparsed is not None
    assert smiles.count("@") == 24
    assert direct_mol_to_smiles(
        reparsed,
        canonical=True,
        isomericSmiles=True,
    ) == direct_mol_to_smiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


def test_extended_ring_label_writer_preserves_starch_like_maltose(monkeypatch):
    """The forced overflow fallback preserves an alpha-linked glucose motif."""
    import g2rins.nx_rdkit_mol as nx_rdkit_mol
    from rdkit.Chem import rdMolDescriptors

    # Maltose is the alpha-1,4-linked glucose disaccharide motif repeated in
    # amylose and in the linear segments of amylopectin (starch).
    maltose = Chem.MolFromSmiles(
        "OC[C@H]1O[C@@H](O[C@H]2[C@@H](CO)O[C@H](O)[C@H](O)[C@H]2O)"
        "[C@H](O)[C@@H](O)[C@@H]1O"
    )
    direct_mol_to_smiles = Chem.MolToSmiles

    def ring_limited_smiles(_value, *_args, **_kwargs):
        raise ValueError("Too many rings open at once. SMILES cannot be generated.")

    monkeypatch.setattr(nx_rdkit_mol, "_BIG_STACK_ATOM_THRESHOLD", 1)
    monkeypatch.setattr(nx_rdkit_mol, "_RANDOM_SMILES_ATTEMPTS", 1)
    monkeypatch.setattr(Chem, "MolToSmiles", ring_limited_smiles)
    monkeypatch.setattr(Chem, "MolToRandomSmilesVect", ring_limited_smiles)

    smiles = rdkit_mol_to_smiles(maltose, smiles_policy="auto")
    reparsed = Chem.MolFromSmiles(smiles)

    assert reparsed is not None
    assert rdMolDescriptors.CalcMolFormula(reparsed) == "C12H22O11"
    assert sum(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in reparsed.GetAtoms()
    ) == 10
    assert direct_mol_to_smiles(
        reparsed,
        canonical=True,
        isomericSmiles=True,
    ) == direct_mol_to_smiles(
        maltose,
        canonical=True,
        isomericSmiles=True,
    )


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
    smiles = rdkit_mol_to_smiles(mol, smiles_policy="auto")

    assert calls and calls[0] == {}
    assert len(calls) == 1
    assert direct_mol_to_smiles(Chem.MolFromSmiles(smiles)) == direct_mol_to_smiles(mol)


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
    smiles = rdkit_mol_to_smiles(mol, smiles_policy="auto")

    assert calls and calls[0] == {}
    assert len(calls) == 1
    assert "[nH]" in smiles or "nH" in smiles
    assert direct_mol_to_smiles(Chem.MolFromSmiles(smiles)) == direct_mol_to_smiles(mol)


def test_mol_graph_to_smiles_small():
    graph = _linear_carbon_graph(3)
    assert mol_graph_to_smiles(graph) == "CCC"
    assert mol_graph_to_smiles(graph) == Chem.MolToSmiles(mol_graph_to_rdkit_mol(graph))


def test_large_ring_fast_path_avoids_symmetric_ring_sanitization(monkeypatch):
    import g2rins.nx_rdkit_mol as nx_rdkit_mol

    graph = nx.Graph()
    for atom_index in range(4):
        graph.add_node(
            atom_index,
            atomic_num=6,
            aromatic=False,
            charge=0,
        )
    for left, right in ((0, 1), (1, 2), (2, 0), (1, 3), (3, 2)):
        graph.add_edge(left, right, bond_type=1, aromatic=False)

    def unexpected_full_sanitization(*_args, **_kwargs):
        raise AssertionError("large ring fast path must not run SymmSSSR")

    monkeypatch.setattr(nx_rdkit_mol, "_FAST_RING_ATOM_THRESHOLD", 1)
    monkeypatch.setattr(nx_rdkit_mol, "_FAST_RING_EXCESS_THRESHOLD", 1)
    monkeypatch.setattr(Chem, "SanitizeMol", unexpected_full_sanitization)

    mol = mol_graph_to_rdkit_mol(graph)
    smiles = rdkit_mol_to_smiles(mol)
    reparsed = Chem.MolFromSmiles(smiles, sanitize=False)

    assert mol.HasProp(nx_rdkit_mol._PREFER_DIRECT_SMILES_PROPERTY)
    assert reparsed is not None
    assert reparsed.GetNumAtoms() == graph.number_of_nodes()
    assert reparsed.GetNumBonds() == graph.number_of_edges()


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


def test_mol_graph_to_rdkit_mol_handles_directed_graph_incomplete_double_bond_markers():
    graph = nx.DiGraph()
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


def test_mol_graph_to_rdkit_mol_handles_directed_graph_ambiguous_double_bond_markers():
    graph = nx.DiGraph()
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


def test_mol_graph_to_rdkit_mol_can_strip_unresolved_directional_markers_on_incomplete_assignments():
    graph = nx.MultiDiGraph()
    graph.add_node(0, atomic_num=9, aromatic=False, charge=0)
    graph.add_node(1, atomic_num=6, aromatic=False, charge=0)
    graph.add_node(2, atomic_num=6, aromatic=False, charge=0)
    graph.add_node(3, atomic_num=9, aromatic=False, charge=0)
    graph.add_edge(0, 1, bond_type=1, aromatic=False, bond_symbol_raw="/")
    graph.add_edge(1, 2, bond_type=2, aromatic=False)
    graph.add_edge(2, 3, bond_type=1, aromatic=False)

    with pytest.warns(RuntimeWarning, match="Incomplete double-bond directional markers"):
        mol = mol_graph_to_rdkit_mol(
            graph,
            strip_unresolved_directional_markers=True,
        )

    smiles = Chem.MolToSmiles(mol)
    assert "/" not in smiles
    assert "\\" not in smiles
    double_bonds = [bond for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE]
    assert len(double_bonds) == 1
    assert double_bonds[0].GetStereo() == Chem.BondStereo.STEREONONE


def test_mol_graph_to_rdkit_mol_does_not_warn_on_carbonyl_double_bonds_when_only_alkene_is_stereo_defined():
    graph = g2rins.G2rins.make("O=C(/C=C\\C(=O)O)O").get_graph_creator().get_generative_graph(
        include_bond_connectors=False
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mol = mol_graph_to_rdkit_mol(graph)

    directional_warnings = [
        warning
        for warning in caught
        if "double-bond directional markers" in str(warning.message)
    ]
    assert not directional_warnings
    assert "/C=C\\" in Chem.MolToSmiles(mol)


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