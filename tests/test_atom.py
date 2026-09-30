# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import pytest

from g2rins.atom import Atom, AtomSymbol, BracketAtom
from g2rins.exception import MissingAtomSymbol, ParsingError, TooManyTokens
from g2rins.parser import get_global_parser
from g2rins.transformer import get_global_transformer


@pytest.mark.parametrize("atom_class", [Atom, BracketAtom])
def test_symbol_less_atom_is_rejected_at_construction(atom_class):
    with pytest.raises(ParsingError, match="Missing atom symbol") as caught:
        atom_class([])
    assert type(caught.value) is MissingAtomSymbol
    assert caught.value.class_name == atom_class.__name__
    assert "Token:" not in str(caught.value)


def test_duplicate_atom_symbols_report_the_conflicting_tokens():
    first, second = AtomSymbol.make("C"), AtomSymbol.make("N")
    with pytest.raises(TooManyTokens) as caught:
        Atom([first, second])
    assert caught.value.existing_token is first
    assert caught.value.new_token is second


@pytest.mark.parametrize("string", ["*", "[*]"])
def test_wildcard_atom_preserves_symbol(string):
    atom = Atom.make(string)
    assert str(atom) == string
    assert str(atom.symbol) == "*"
    assert atom.aromatic is False
    assert atom.charge == 0


@pytest.mark.parametrize("string", ["B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"])
def test_simple_aliphatic_atom(string):
    tree = get_global_parser().parse(string, start="atom")
    a = get_global_transformer().transform(tree)
    assert a.aromatic is False
    assert str(a) == string

    b = Atom.make(string)
    assert b.aromatic is False
    assert str(b) == string
    assert b.charge == 0


@pytest.mark.parametrize(
    "string",
    [
        "b",
        "c",
        "n",
        "o",
        "p",
        "s",
    ],
)
def test_simple_aromatic_atom(string):
    a = Atom.make(string)
    assert a.aromatic is True
    assert str(a) == string
    assert a.charge == 0


@pytest.mark.parametrize("string", ["[se]", "[H]", "[He]", "[Li]", "[Be]", "[B]", "[C]"])
def test_simple_bracket_atom(string):
    a = BracketAtom.make(string)
    assert str(a) == string
    assert isinstance(a, Atom)
    assert a.charge == 0


@pytest.mark.parametrize("string", ["[se@]", "[H@@]", "[He@SP2]", "[Li@TB4]", "[Be@TB14]", "[B@OH3]", "[C@OH45]"])
def test_chiral_bracket_atom(string):
    a = BracketAtom.make(string)
    assert str(a) == string
    assert isinstance(a, Atom)


@pytest.mark.parametrize("string", ["[seH]", "[H]", "[BkH]", "[UH2]"])
def test_h_count_bracket_atom(string):
    a = BracketAtom.make(string)
    assert str(a) == string
    assert isinstance(a, Atom)


@pytest.mark.parametrize(
    ("string", "charge"),
    [
        ("[se-]", -1),
        ("[H+]", +1),
        ("[Bk--]", -2),
        ("[UH++]", 2),
        ("[C-2]", -2),
        ("[Pa+2]", +2),
        ("[Cf]", 0),
    ],
)
def test_charge_bracket_atom(string, charge):
    a = BracketAtom.make(string)
    assert str(a) == string
    assert isinstance(a, Atom)
    assert a.charge == charge


@pytest.mark.parametrize("string", ["[N:2]", "[p:1]", "[I:4]", "[s:14]"])
def test_class_bracket_atom(string):
    a = BracketAtom.make(string)
    assert str(a) == string
    assert isinstance(a, Atom)


def test_everything_atom():
    string = "[13C@OH1H2+1:3]"
    a = BracketAtom.make(string)
    assert str(a) == string
    assert a.isotope.num_nuclei == 13
    assert str(a.chiral) == "@OH1"
    assert a.h_count.num == 2
    assert str(a.h_count) == "H2"
    assert a.charge == 1
    assert a.atom_class.num == 3
    assert str(a.atom_class) == ":3"
