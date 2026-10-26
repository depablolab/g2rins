# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import pytest
from rdkit import Chem

from g2rins import nx_rdkit_mol as module
from g2rins.nx_rdkit_mol import (
    _PREFER_DIRECT_SMILES_PROPERTY,
    _direct_smiles_with_extended_ring_labels,
    _is_ring_label_overflow,
)


@pytest.mark.parametrize(
    "exc, expected",
    [
        (ValueError("Too many rings open at once. SMILES cannot be generated."), True),
        (RuntimeError("TOO MANY RINGS OPEN AT ONCE"), True),
        (ValueError("unrelated serialization error"), False),
        (RuntimeError("ring labels exhausted"), False),
        (TypeError("Too many rings open at once"), False),
        (Exception("Too many rings open at once"), False),
        (KeyboardInterrupt("Too many rings open at once"), False),
    ],
)
def test_ring_label_overflow_requires_known_exception_and_message(exc, expected):
    assert _is_ring_label_overflow(exc) is expected


@pytest.mark.parametrize("prefer_direct", [False, True])
@pytest.mark.parametrize(
    "smiles",
    [
        "",
        "CC(O)C",
        "c1ccc(-c2ccccc2)cc1",
        "[13CH3:7][NH2+]CC(=O)[O-].[Na+]",
        "c1cc[nH]c1",
        "[C@](F)(Cl)(Br)I",
        "[C@H](F)(Cl)Br",
        "O[C@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]1O",
        "F/C=C/F",
        "F/C=C\\F",
    ],
)
def test_direct_writer_round_trip_preserves_chemistry_and_stereo(smiles, prefer_direct):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    expected = Chem.MolToSmiles(mol, isomericSmiles=True)
    n_atoms = mol.GetNumAtoms()
    for order in (list(range(n_atoms)), list(reversed(range(n_atoms)))):
        reordered = Chem.RenumberAtoms(mol, order) if n_atoms else Chem.Mol(mol)
        reordered.SetBoolProp(_PREFER_DIRECT_SMILES_PROPERTY, prefer_direct)
        written = _direct_smiles_with_extended_ring_labels(reordered)
        assert written is not None
        reparsed = Chem.MolFromSmiles(written)
        assert reparsed is not None
        assert Chem.MolToSmiles(reparsed, isomericSmiles=True) == expected


def test_direct_writer_emits_all_extended_ring_label_forms():
    mol = Chem.MolFromSmiles(".".join(["C1CC1"] * 100))
    written = _direct_smiles_with_extended_ring_labels(mol)

    assert written.startswith("C1CC1.")
    assert ".C%10CC%10." in written
    assert written.endswith(".C%(100)CC%(100)")
    reparsed = Chem.MolFromSmiles(written)
    assert reparsed is not None
    assert reparsed.GetNumAtoms() == mol.GetNumAtoms()
    assert reparsed.GetNumBonds() == mol.GetNumBonds()
    assert len(Chem.GetMolFrags(reparsed)) == 100


def test_direct_writer_declines_unsupported_dative_bond():
    mol = Chem.MolFromSmiles("[NH3]->[Cu+2]")
    assert mol is not None
    assert _direct_smiles_with_extended_ring_labels(mol) is None


def test_direct_writer_atom_token_cache_is_local_to_each_call(monkeypatch):
    mol = Chem.MolFromSmiles("CCC")
    fragment_writer = Chem.MolFragmentToSmiles
    calls = []

    def recorded_fragment(value, **kwargs):
        calls.append(kwargs["atomsToUse"])
        return fragment_writer(value, **kwargs)

    monkeypatch.setattr(Chem, "MolFragmentToSmiles", recorded_fragment)

    assert _direct_smiles_with_extended_ring_labels(mol) == "CCC"
    assert calls == [[0], [1]]
    assert _direct_smiles_with_extended_ring_labels(mol) == "CCC"
    assert calls == [[0], [1], [0], [1]]


def test_direct_writer_preferred_path_avoids_fragment_writer(monkeypatch):
    mol = Chem.MolFromSmiles("c1cc[nH]c1")
    mol.SetBoolProp(_PREFER_DIRECT_SMILES_PROPERTY, True)

    def unexpected_fragment_writer(*_args, **_kwargs):
        pytest.fail("preferred direct writing must not invoke MolFragmentToSmiles")

    monkeypatch.setattr(Chem, "MolFragmentToSmiles", unexpected_fragment_writer)
    written = _direct_smiles_with_extended_ring_labels(mol)
    assert written is not None
    reparsed = Chem.MolFromSmiles(written)
    assert reparsed is not None
    assert Chem.MolToSmiles(reparsed) == Chem.MolToSmiles(mol)


@pytest.mark.parametrize("n_atoms, expected", [
    (0, []), (1, [0]), (2, [0, 1]), (4, [0, 1, 2, 3]),
    (16, [0, 1, 2, 10, 4, 8, 12, 14, 15]),
])
def test_fallback_root_order_is_stable_and_deduplicated(n_atoms, expected):
    assert module._smiles_fallback_roots(n_atoms) == expected


@pytest.mark.parametrize("large", [False, True])
def test_recovery_policy_preserves_exact_strategy_order(monkeypatch, large):
    mol = Chem.MolFromSmiles("CCCC")
    monkeypatch.setattr(module, "_BIG_STACK_ATOM_THRESHOLD", 3 if large else 100)
    events = []

    def record(name, result=None):
        def call(*args):
            events.append((name, *args[1:]))
            return result
        return call

    monkeypatch.setattr(module, "_direct_smiles_with_extended_ring_labels", record("direct"))
    monkeypatch.setattr(module, "_rooted_smiles", record("root"))
    monkeypatch.setattr(module, "_seeded_randomized_smiles", record("random"))
    monkeypatch.setattr(module, "_breadth_first_smiles", record("bfs"))
    monkeypatch.setattr(Chem, "MolToSmiles", lambda value, **kwargs: events.append(("final", kwargs)) or "last")
    assert module._recover_smiles_ring_overflow(mol) == "last"
    assert events == (
        [("direct",), ("root", []), ("random",), ("bfs",), ("root", [0, 1, 2, 3]), ("final", {"canonical": False})]
        if large else
        [("direct",), ("root", [0]), ("bfs",), ("root", [1, 2, 3]), ("final", {"canonical": False})]
    )


@pytest.mark.parametrize("strategy", ["direct", "initial-root", "random", "bfs", "later-root"])
def test_recovery_returns_first_success_without_later_native_calls(monkeypatch, strategy):
    mol = Chem.MolFromSmiles("CCCC")
    large = strategy == "random"
    monkeypatch.setattr(module, "_BIG_STACK_ATOM_THRESHOLD", 3 if large else 100)
    events = []

    def attempt(name):
        events.append(name)
        return "success" if name == strategy else None

    monkeypatch.setattr(module, "_direct_smiles_with_extended_ring_labels", lambda value: attempt("direct"))
    monkeypatch.setattr(module, "_rooted_smiles", lambda value, roots: attempt("initial-root" if roots == [0] else "later-root"))
    monkeypatch.setattr(module, "_seeded_randomized_smiles", lambda value: attempt("random"))
    monkeypatch.setattr(module, "_breadth_first_smiles", lambda value: attempt("bfs"))
    monkeypatch.setattr(Chem, "MolToSmiles", lambda *args, **kwargs: pytest.fail("unexpected final traversal"))
    assert module._recover_smiles_ring_overflow(mol) == "success"
    assert events[-1] == strategy


def test_rooted_recovery_does_not_swallow_other_errors(monkeypatch):
    error = ValueError("not a ring-label overflow")
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs["rootedAtAtom"])
        raise error

    monkeypatch.setattr(Chem, "MolToSmiles", fail)
    with pytest.raises(ValueError) as caught:
        module._rooted_smiles(Chem.MolFromSmiles("CC"), [0, 1])
    assert caught.value is error
    assert calls == [0]


def test_bfs_recovery_preserves_component_and_reverse_neighbor_order(monkeypatch):
    mol = Chem.MolFromSmiles("C(C)(C)C.CC")
    renumber = Chem.RenumberAtoms
    orders = []

    def reordered(value, order):
        orders.append(order)
        return renumber(value, order)

    def write(value, **kwargs):
        assert kwargs == {"canonical": False, "rootedAtAtom": 0}
        if len(orders) == 1:
            raise ValueError("Too many rings open at once")
        return "success"

    monkeypatch.setattr(Chem, "RenumberAtoms", reordered)
    monkeypatch.setattr(Chem, "MolToSmiles", write)
    assert module._breadth_first_smiles(mol) == "success"
    assert orders == [[0, 3, 2, 1, 4, 5], [0, 1, 2, 3, 4, 5]]


@pytest.mark.parametrize("succeeds", [False, True])
def test_randomized_recovery_reseeds_each_attempt_under_one_lock(monkeypatch, succeeds):
    from rdkit import rdBase

    events = []

    class Lock:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *args):
            events.append("exit")

    def write(mol, count, **kwargs):
        assert count == 1 and kwargs == {"randomSeed": 0}
        seed = events[-1]
        events.append("write")
        if succeeds and seed == 3:
            return ["success"]
        raise RuntimeError("Too many rings open at once")

    monkeypatch.setattr(module, "_RANDOM_SMILES_LOCK", Lock())
    monkeypatch.setattr(module, "_RANDOM_SMILES_ATTEMPTS", 3)
    monkeypatch.setattr(rdBase, "SeedRandomNumberGenerator", events.append)
    monkeypatch.setattr(Chem, "MolToRandomSmilesVect", write)
    assert module._seeded_randomized_smiles(Chem.MolFromSmiles("C")) == ("success" if succeeds else None)
    assert events == ["enter", 1, "write", 2, "write", 3, "write", "exit"]