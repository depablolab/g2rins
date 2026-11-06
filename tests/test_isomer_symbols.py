# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Isotope, chirality and cis/trans symbols parse and build a graph, and the cis/trans symbols leading to connecting bond
connectors are single bonds that pair and generate."""

from collections import Counter

import pytest

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
