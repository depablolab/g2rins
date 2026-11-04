# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Stereochemistry and isotopes: graph encoding, sampled molecules and RDKit conversion."""

import re
import warnings

import lark
import numpy as np
import pytest

import g2rins
from g2rins.chem_resource import atom_mass, atomic_masses, chirality_codes, isotope_masses
from g2rins.exception import ChiralityNotApplied, InconsistentChiralityClass, UnknownIsotopeMass

LEFT_CONNECTOR = re.compile(r"\[<[^\]]*\]")
RIGHT_CONNECTOR = re.compile(r"\[>[^\]]*\]")
EMPTY_BRANCH = re.compile(r"\([-=#$:/\\]?\)")


def _chain(units, mw):
    return f"[H]{{[>]{', '.join(units)};;[<]}}|uniform({mw},{mw})|[H]"


def _graph(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return g2rins.G2rins.make(text).get_graph_creator().get_generative_graph()


def _sample(text, seed=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.G2rins.make(text).get_graph_creator().get_ensemble_creator()
        return creator.sample_mol_graph(rng=np.random.default_rng(seed))


def _canonical(mol_graph):
    Chem = pytest.importorskip("rdkit.Chem")
    return Chem.MolToSmiles(Chem.RemoveHs(g2rins.mol_graph_to_rdkit_mol(mol_graph)))


def _double_bond_geometries(mol_graph):
    """(E/Z label, bears a Cl) of every stereo double bond of the sampled molecule."""
    Chem = pytest.importorskip("rdkit.Chem")
    mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    labels = {Chem.BondStereo.STEREOE: "E", Chem.BondStereo.STEREOZ: "Z"}
    geometries = []
    for bond in mol.GetBonds():
        if bond.GetStereo() in labels:
            has_cl = any(nbr.GetAtomicNum() == 17 for atom in (bond.GetBeginAtom(), bond.GetEndAtom()) for nbr in atom.GetNeighbors())
            geometries.append((labels[bond.GetStereo()], has_cl))
    return geometries


def _nested(units, n):
    """Plain SMILES of an n-unit chain between two hydrogens: each unit is written into the [>] of the previous one."""
    text = "[H]"
    for i in reversed(range(n)):
        text = RIGHT_CONNECTOR.sub(lambda _match, text=text: text, LEFT_CONNECTOR.sub("", units[i % len(units)]))
    return "[H]" + text


def _unit_heavy_atoms(unit):
    Chem = pytest.importorskip("rdkit.Chem")
    bare = EMPTY_BRANCH.sub("", RIGHT_CONNECTOR.sub("", LEFT_CONNECTOR.sub("", unit))).strip("-=#$:/\\")
    return Chem.MolFromSmiles(bare).GetNumHeavyAtoms()


def _assert_chain_is_the_written_isomer(units, mw, alternatives=()):
    """The sampled chain is the molecule the units spell out when read as plain SMILES, whatever order the sampler visits the atoms."""
    Chem = pytest.importorskip("rdkit.Chem")
    text = _chain(units, mw)
    mol_graph = _sample(text)
    n = sum(1 for _node, data in mol_graph.nodes(data=True) if data["atomic_num"] > 1) // _unit_heavy_atoms(units[0])
    expected = {Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(_nested(order, n)))) for order in (units, *alternatives)}
    assert _canonical(mol_graph) in expected, text


@pytest.mark.parametrize(
    ("unit", "mw"),
    [
        ("[<]C/C=C/C[>]", 220),
        ("[<]C/C=C\\C[>]", 220),
        ("[<]C[C@H](C)[>]", 170),
        ("[<]C[C@@H](C)[>]", 170),
        ("[<][C@H](C)C[>]", 170),
        ("[<]C[C@H]([>])C", 170),
        ("[<]C/C=C(\\[>])C", 220),
        ("[<][C@H]1CCCC[C@@H]1[>]", 250),
        ("[<]C[C@H](C1=CCCCC1)[>]", 370),
        ("[<]/C=C/C[>]", 220),
        ("[<]C/C=C/[>]", 220),
        ("[<]C/C=C\\[>]", 220),
        ("[<]CC/1CCCCC/C=C1[>]", 250),
        ("[<]CC1CCCCC/C=C/1[>]", 250),
        ("[<]C[Po@SP1](Cl)(Br)[>]", 700),
        ("[<]C[Po@SP3](Cl)(Br)[>]", 700),
    ],
    ids=[
        "trans",
        "cis",
        "chiral_at",
        "chiral_atat",
        "chiral_first_atom",
        "connector_in_branch_chiral",
        "connector_in_branch_alkene",
        "ring_two_centers",
        "chiral_with_ring_substituent",
        "mark_on_left_connector_bond",
        "mark_on_right_connector_bond",
        "cis_across_connector_bond",
        "mark_at_opening_ring_digit",
        "mark_at_closing_ring_digit",
        "square_planar_1",
        "square_planar_3",
    ],
)
def test_sampled_chain_is_the_written_isomer(unit, mw):
    _assert_chain_is_the_written_isomer([unit], mw)


def test_two_alternating_units_keep_their_own_configuration():
    units = ["[<|0 0 0 1|]C[C@H](C)[>|0 0 1 0|]", "[<|0 1 0 0|]C[C@@H](C)[>|1 0 0 0|]"]
    _assert_chain_is_the_written_isomer(units, 170, alternatives=[units[::-1]])


def test_entry_side_does_not_change_the_molecule():
    left = _canonical(_sample("[H]{[>][<]C[C@H](C)[>];;[<]}|uniform(170,170)|[H]"))
    right = _canonical(_sample("[H]{[<][<]C[C@H](C)[>];;[>]}|uniform(170,170)|[H]"))
    assert "@" in left
    assert left == right


@pytest.mark.parametrize("text", ["[H]{[>][<]C[C@H](C)[>];;[<]}|uniform(170,170)|[H]", "CCO", "C{[$][$]CC(CC[$])(CC[$]),[$]CC[$];;[$][H][$]}|flory_schulz(1e-3)|[H]"])
def test_every_node_and_edge_carries_the_stereo_features(text):
    graph = _graph(text)
    for _node, data in graph.nodes(data=True):
        assert isinstance(data["chiral"], int)
        assert isinstance(data["isotope"], int)
    for _u, _v, data in graph.edges(data=True):
        assert data["cis_trans_mark"] in (0, 1, 2)
        assert data["cis_trans_flip"] in (0, 1)
        assert isinstance(data["chiral_neighbor_rank"], list) and len(data["chiral_neighbor_rank"]) == 2
    # The sampled graph carries the same molecular features; the generation-only flip bit stays out of it.
    if "{" in text:
        for _node, data in _sample(text).nodes(data=True):
            assert isinstance(data["chiral"], int) and isinstance(data["isotope"], int)
        for _u, _v, data in _sample(text).edges(data=True):
            assert data["cis_trans_mark"] in (0, 1, 2) and len(data["chiral_neighbor_rank"]) == 2
            assert "cis_trans_flip" not in data


POLYACETYLENE = [
    (r"{[] [<]/C=C\[>]; [H][>]; [<][H] []}|gauss(400,10)|", "Z"),
    (r"{[] [$]\C=C/[$]; [H][$]; [$][H] []}|gauss(400,10)|", "Z"),
    (r"{[] [$]/C=C/[$]; [H][$]; [$][H] []}|gauss(400,10)|", "E"),
    (r"{[] [<]C=C/[>]; [H][>]; [<][H] []}|gauss(400,10)|", "E"),
]


@pytest.mark.parametrize(("text", "geometry"), POLYACETYLENE, ids=["cis_both_marks", "cis_dollar", "trans_dollar", "trans_one_mark"])
def test_every_junction_keeps_the_written_double_bond_geometry(text, geometry):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        creator = g2rins.G2rins.make(text).get_graph_creator().get_ensemble_creator()
    for seed in range(3):
        geometries = _double_bond_geometries(creator.sample_mol_graph(rng=np.random.default_rng(seed)))
        assert len(geometries) >= 3
        assert {label for label, _has_cl in geometries} == {geometry}, text


def test_a_terminator_with_the_opposite_junction_mark_is_re_expressed():
    geometries = _double_bond_geometries(_sample(r"{[] [<]/C=C\[>]; [H][>]; [<]/C=C/Cl []}|gauss(400,10)|"))
    assert [label for label, has_cl in geometries if has_cl] == ["E"]
    assert {label for label, has_cl in geometries if not has_cl} == {"Z"}


@pytest.mark.parametrize(
    ("text", "junctions"),
    [
        (POLYACETYLENE[0][0], [(2, 1)] * 2),
        (POLYACETYLENE[1][0], [(1, 1)] * 4),
        (POLYACETYLENE[2][0], [(1, 0), (1, 1), (2, 0), (2, 1)]),
        (POLYACETYLENE[3][0], [(1, 0), (2, 0)]),
    ],
    ids=["cis_both_marks", "cis_dollar", "trans_dollar", "trans_one_mark"],
)
def test_junction_bond_carries_the_source_mark_and_the_flip_bit(text, junctions):
    graph = _graph(text)
    propagation = [(data["cis_trans_mark"], data["cis_trans_flip"]) for _u, _v, data in graph.edges(data=True) if data["propagation_weight"] > 0]
    assert sorted(propagation) == junctions
    assert all(data["cis_trans_flip"] == 0 for _u, _v, data in graph.edges(data=True) if data["static"])


def test_cis_trans_mark_is_flipped_on_the_reverse_edge():
    graph = _graph("[H]{[>][<]C/C=C\\C[>];;[<]}|uniform(220,220)|[H]")
    marked = [(u, v, data["cis_trans_mark"]) for u, v, data in graph.edges(data=True) if data["cis_trans_mark"]]
    assert {mark for _u, _v, mark in marked} == {1, 2}
    for u, v, mark in marked:
        assert any(data["cis_trans_mark"] == 3 - mark for data in graph.get_edge_data(v, u).values())


def test_chirality_code_table():
    assert chirality_codes[("TH", 1)] == 1 and chirality_codes[("TH", 2)] == 2
    assert chirality_codes[("AL", 1)] == 3 and chirality_codes[("AL", 2)] == 4
    assert chirality_codes[("SP", 1)] == 5 and chirality_codes[("SP", 3)] == 7
    assert chirality_codes[("TB", 1)] == 8 and chirality_codes[("TB", 20)] == 27
    assert chirality_codes[("OH", 1)] == 28 and chirality_codes[("OH", 30)] == 57
    assert len(chirality_codes) == 57


@pytest.mark.parametrize(
    ("unit", "code"),
    [
        ("[<]C[C@H](C)[>]", 1),
        ("[<]C[C@@H](C)[>]", 2),
        ("[<]C[C@TH2H](C)[>]", 2),
        ("[<]C[S@](=O)C[>]", 1),
        ("[<]C=[C@]=C[>]", 3),
        ("[<]C=[C@@]=C[>]", 4),
        ("[<]C=[C@AL2]=C[>]", 4),
        ("[<]C[Po@](Cl)(Br)[>]", 1),
        ("[<]C[Po@SP3](Cl)(Br)[>]", 7),
        ("[<]C[As@](F)(Cl)(Br)[>]", 8),
        ("[<]C[As@TB15](F)(Cl)(Br)[>]", 22),
        ("[<]C[Co@@](F)(Cl)(Br)(I)[>]", 29),
        ("[<]C[Co@OH30](F)(Cl)(Br)(I)[>]", 57),
    ],
)
def test_bare_marks_resolve_to_the_class_of_the_written_degree(unit, code):
    codes = {data["chiral"] for _node, data in _graph(_chain([unit], 500)).nodes(data=True)} - {0}
    assert codes == {code}


@pytest.mark.parametrize("unit", ["[<]C[C@TB1H](C)[>]", "[<]C=[C@TH1]=C[>]", "[<]C[C@OH1](C)(F)[>]", "[<]C[C@](C)(F)(Cl)(Br)(I)[>]"])
def test_inconsistent_chirality_class_is_a_parse_error(unit):
    with pytest.raises(lark.exceptions.VisitError) as excinfo:
        g2rins.G2rins.make(_chain([unit], 500))
    assert isinstance(excinfo.value.orig_exc, InconsistentChiralityClass)


def test_inconsistent_chirality_class_on_a_plain_molecule():
    with pytest.raises(InconsistentChiralityClass):
        g2rins.G2rins.make("C[C@TB1H](C)O").get_graph_creator().get_generative_graph()


@pytest.mark.parametrize(
    ("unit", "klass", "code"),
    [("[<]C=[C@AL1]=C[>]", "AL", 3), ("[<]C[As@TB1](F)(Cl)(Br)[>]", "TB", 8), ("[<]C[Co@OH1](F)(Cl)(Br)(I)[>]", "OH", 28)],
)
def test_other_classes_are_carried_in_the_graphs_but_not_applied(unit, klass, code):
    Chem = pytest.importorskip("rdkit.Chem")
    mol_graph = _sample(_chain([unit], 600))
    assert code in {data["chiral"] for _node, data in mol_graph.nodes(data=True)}
    with pytest.warns(ChiralityNotApplied) as record:
        mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
    assert {str(w.message.token) for w in record if isinstance(w.message, ChiralityNotApplied)} == {klass}
    assert all(atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms())


def test_isotope_is_carried_into_graph_molecule_and_smiles():
    pytest.importorskip("rdkit.Chem")
    text = _chain(["[<]C[13CH2][>]"], 100)
    assert {data["isotope"] for _node, data in _graph(text).nodes(data=True)} == {0, 13}
    mol_graph = _sample(text)
    assert {data["isotope"] for _node, data in mol_graph.nodes(data=True)} == {0, 13}
    mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
    assert sorted({atom.GetIsotope() for atom in mol.GetAtoms()}) == [0, 13]
    assert "[13CH2]" in g2rins.mol_graph_to_smiles(mol_graph)


def test_written_isotope_twelve_differs_from_unspecified():
    assert {data["isotope"] for _node, data in _graph("[12CH3]C").nodes(data=True)} == {0, 12}


def test_atom_mass_uses_the_isotopic_mass():
    assert atom_mass(6, 0) == atomic_masses[6]
    assert atom_mass(6, 13) == isotope_masses[(6, 13)] == pytest.approx(13.00335)
    assert atom_mass(1, 2) == pytest.approx(2.01410)
    with pytest.warns(UnknownIsotopeMass):
        assert atom_mass(6, 11) == 11.0
