# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import lark
import pytest

import g2rins


def test_chembl_smi_grammar(grammar_parser, chembl_smi_list):
    for smi in chembl_smi_list:
        assert grammar_parser.parse(smi)


def test_g2rins_grammar(grammar_parser, g2rins_list):
    for smi in g2rins_list:
        assert grammar_parser.parse(smi)


def test_invalid_g2rins_grammar(grammar_parser, invalid_g2rins_list):
    for smi in invalid_g2rins_list:
        print(smi)
        with pytest.raises(lark.UnexpectedInput):
            tree = grammar_parser.parse(smi)
            print(tree.pretty())


def test_g2rins_graph(grammar_parser, g2rins_list):
    for smi in g2rins_list:
        print(smi)
        graph_creator = g2rins.G2rins.make(smi).get_graph_creator()
        graph_creator.get_generative_graph(include_bond_connectors=True)
        graph_creator.get_generative_graph(include_bond_connectors=False)
