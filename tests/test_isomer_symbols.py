# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Isotope, chirality and cis/trans symbols parse and build a graph, and the cis/trans symbols leading to connecting bond
connectors are single bonds that pair and generate."""

from collections import Counter

import pytest
from rdkit import Chem

import g2rins

CIS_ALKENE = r"{[] [<]/C=C\[>]; [H][>]; [<][H] []}|gauss(400,10)|"
TRANS_ALKENE = "{[] [<]/C=C/[>]; [H][>]; [<][H] []}|gauss(400,10)|"


@pytest.mark.filterwarnings("ignore::g2rins.exception.G2RINSWarning")
@pytest.mark.parametrize(
    ("text", "num_atoms"),
    [
        ("[13CH3]C", 2),
        ("[2H]C", 2),
        ("C[C@H](F)Cl", 4),
        ("C[C@@H](F)Cl", 4),
        ("F[C@TH1](Cl)(Br)I", 5),
        ("C[C@AL1]=[C@AL2]=CC", 5),
        ("[C@SP3](F)(Cl)(Br)I", 5),
        ("F[C@TB20](Cl)(Br)(I)C", 6),
        ("F[C@OH30](Cl)(Br)(I)(C)N", 7),
        (r"C/C=C\C", 4),
        ("C/C=C/C", 4),
        (r"C/1CCCCC/C=C1", 8),
    ],
)
def test_every_isomer_symbol_parses_and_builds_a_graph(text, num_atoms):
    molecule = g2rins.G2rins.make(text)
    assert str(molecule) == text
    graph = molecule.get_graph_creator().get_generative_graph()
    assert graph.number_of_nodes() == num_atoms


@pytest.mark.filterwarnings("ignore::g2rins.exception.G2RINSWarning")
@pytest.mark.parametrize("text", [CIS_ALKENE, TRANS_ALKENE])
def test_cis_trans_symbols_at_bond_connectors_pair_as_single_bonds(text):
    graph = g2rins.G2rins.make(text).get_graph_creator().get_generative_graph()
    assert graph.number_of_nodes() == 4
    assert Counter(data["bond_type"] for _u, _v, data in graph.edges(data=True)) == {1: 4, 2: 2}


@pytest.mark.filterwarnings("ignore::g2rins.exception.G2RINSWarning")
def test_cis_alkene_generates_without_losing_the_double_bonds():
    chain = g2rins.G2rins.make(CIS_ALKENE).get_graph_creator().get_ensemble_creator().create_ensemble(1, seed=1)[0]
    bond_types = Counter(data["bond_type"] for _u, _v, data in chain.edges(data=True))
    assert bond_types[2] > 5
    assert bond_types[2] == (chain.number_of_nodes() - 2) // 2


@pytest.mark.filterwarnings("ignore::g2rins.exception.G2RINSWarning")
def test_opposite_directional_symbols_on_both_ring_digits_describe_one_single_bond():
    graph = g2rins.G2rins.make(r"C/1CCCCC\1").get_graph_creator().get_generative_graph()
    assert graph.number_of_nodes() == 6
    assert {data["bond_type"] for _u, _v, data in graph.edges(data=True)} == {1}


@pytest.mark.parametrize(("text", "symbols"), [("C/1CCCCC/1", "/ on the first digit and / on the second"), (r"C\1CCCCC\1", r"\ on the first digit and \ on the second")])
def test_equal_directional_symbols_on_both_ring_digits_raise(text, symbols):
    with pytest.raises(g2rins.exception.DoubleBondSymbolDefinition) as excinfo:
        g2rins.G2rins.make(text).get_graph_creator()
    assert "ring closure 1" in str(excinfo.value)
    assert symbols in str(excinfo.value)


@pytest.mark.parametrize("text", ["C/1CCCCC-1", r"C-1CCCCC\1"])
def test_directional_symbol_and_explicit_single_bond_on_the_ring_digits_raise(text):
    with pytest.raises(g2rins.exception.DoubleBondSymbolDefinition):
        g2rins.G2rins.make(text).get_graph_creator()


DOUBLE_BOND_CONNECTORS = "{[<] [<]=CC=[>];; [>]}|gauss(200,10)|"


def test_bond_symbol_between_two_stochastic_objects_must_agree_with_their_bond_connectors():
    with pytest.raises(g2rins.exception.IncompatibleBondTypeBondConnector) as excinfo:
        g2rins.G2rins.make(f"C={DOUBLE_BOND_CONNECTORS}-{DOUBLE_BOND_CONNECTORS}=C").get_graph_creator().get_generative_graph()
    assert "]-[" in excinfo.value.written_path

    # Without the symbol between the objects, the double bond of their bond connectors is kept.
    graph = g2rins.G2rins.make(f"C={DOUBLE_BOND_CONNECTORS}{DOUBLE_BOND_CONNECTORS}=C").get_graph_creator().get_generative_graph()
    between_objects = [data["bond_type"] for u, v, data in graph.edges(data=True) if {graph.nodes[u]["stochastic_id_tree"][0], graph.nodes[v]["stochastic_id_tree"][0]} == {0, 1}]
    assert between_objects == [2]


@pytest.mark.parametrize(
    "text",
    [
        "C={[<] [<]CC[>];; [>]}|gauss(400,10)|=C",  # next to an atom, on both sides
        "C{[<] [<]CC[>];; [>]}|gauss(400,10)|=C",  # next to an atom, on one side
        "CC{[$] [$]CC[$];; [$]}|gauss(200,10)|={[$] [$]CO[$];; [$]}|gauss(200,10)|CC",  # between two stochastic objects
        "{[] [<]CC(={[$] [$]CC[$];; [$]}|gauss(400,10)|[H])[>]; [H][>]; [<][H] []}|gauss(400,10)|",  # before a nested stochastic object
    ],
)
def test_double_bond_symbol_at_a_stochastic_object_with_unwritten_connector_bonds_raises(text):
    with pytest.raises(g2rins.exception.IncompatibleBondTypeBondConnector):
        g2rins.G2rins.make(text).get_graph_creator().get_generative_graph()


def test_aromatic_bond_symbol_against_an_explicit_single_bond_raises():
    with pytest.raises(g2rins.exception.IncompatibleBondTypeBondConnector):
        g2rins.G2rins.make("{[] [<]-c1ccc(:[>])cc1; C[>]; [<]C []}|gauss(400,10)|").get_graph_creator().get_generative_graph()


def test_incompatible_bond_order_is_reported_on_a_pairing_that_can_form():
    # The first unit's [<]= can bond the second unit's [>], whose bond is unwritten.
    text = "{[] [<]=CC=[>], [<]CC[>]; C=[>], [H][>]; [<]=C, [<][H] []}|gauss(400,10)|"
    with pytest.raises(g2rins.exception.IncompatibleBondTypeBondConnector) as excinfo:
        g2rins.G2rins.make(text).get_graph_creator().get_generative_graph()
    assert excinfo.value.written_path == "C=[<][>]C"
    assert "C=[<][>]C, where no symbol next to an atom is a single bond" in str(excinfo.value)


def test_ring_closed_across_a_branch_keeps_the_molecule_a_seed_generates():
    # The two end groups can only be told apart on the two arms of the ring, whose bonds are added in a fixed order.
    text = "{[] [<]CC1(C(CC[>])CCC1C[>]); F[>]; [<]Cl, [<]Br []}|gauss(280,5)|"
    chain = g2rins.G2rins.make(text).get_graph_creator().get_ensemble_creator().create_ensemble(1, seed=7)[0]
    assert Chem.CanonSmiles(g2rins.mol_graph_to_smiles(chain)) == "FCC1C(CCl)CCC1CCBr"
