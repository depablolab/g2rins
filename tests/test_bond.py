# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import pytest

import g2rins

test_simple_bond_connectors = [
    ("[$0]", [0], "[$]", "[$]"),
    ("[<]", [0], "[<]", "[<]"),
    ("[>|2|]", [0], "[>|2.0|]", "[>]"),
    ("[$|3|]", [0], "[$|3.0|]", "[$]"),
    ("[<]", [0], "[<]", "[<]"),
    ("[>]", [0], "[>]", "[>]"),
    ("[$5| 2.|]", [5], "[$5|2.0|]", "[$5]"),
    ("[<4| 21. 234. 2134. 64. 657.|]", [4], "[<4|21.0 234.0 2134.0 64.0 657.0|]", "[<4]"),
    ("[<7| 0.0 1.0 0.0|]", [7], "[<7|0.0 1.0 0.0|]", "[<7]"),
    ("[>,$1]", [0, 1], "[>, $1]", "[>, $1]"),
    ("[>, <2|4 |]", [0, 2], "[>, <2|4.0|]", "[>, <2]"),
    ("[$1,$2, <3|0.0 1.0|]", [1, 2, 3], "[$1, $2, <3|0.0 1.0|]", "[$1, $2, <3]"),
]


@pytest.mark.parametrize(("text", "idx", "ref", "big"), test_simple_bond_connectors)
def test_simple_bond_connectors(text, idx, ref, big):
    bond = g2rins.BondConnector.make(text)
    if idx is not None:
        assert bond.idx == idx
    assert str(bond) == ref
    assert bond.generate_string(False) == big
    assert bond.generable


test_terminal_bond_connectors = [
    ("[]", None, "[]", "[]"),
    ("[$|6|]", [0], "[$|6.0|]", "[$]"),
    ("[<1,$2|6. 3. 1 0|]", [1, 2], "[<1, $2|6.0 3.0 1.0 0.0|]", "[<1, $2]"),
]


@pytest.mark.parametrize(("text", "idx", "ref", "big"), test_terminal_bond_connectors)
def test_terminal_bond_connectors(text, idx, ref, big):
    bond = g2rins.bond.TerminalBondConnector.make(text)
    if idx is not None:
        assert bond.idx == idx
    assert str(bond) == ref
    assert bond.generate_string(False) == big
    assert bond.generable


test_terminal_bond_connectors_list = [("[<]|[<1]", ["[<]", "[<1]"]), ("[<1,$2|6. 3. 1 0|]", ["[<1, $2|6.0 3.0 1.0 0.0|]"]), ("[>,$1|6|]|[>2]", ["[>, $1|6.0|]", "[>2]"]), ("[]", ["[]"])]


@pytest.mark.parametrize(("text", "expected_bc_list"), test_terminal_bond_connectors_list)
def test_terminal_bond_connectors_list(text, expected_bc_list):
    terminal_bond_list = g2rins.bond.TerminalBondConnectorList.make(text)
    assert len(terminal_bond_list.terminal_bond_connectors) == len(expected_bc_list)

    for actual, expected in zip(terminal_bond_list.terminal_bond_connectors, expected_bc_list):
        assert str(actual) == expected


compatibility_list = [
    ("[$]", "[$]", True),
    ("[$1]", "[$1]", True),
    ("[$0]", "[$1]", False),
    ("[<]", "[<]", False),
    ("[>]", "[>]", False),
    ("[<]", "[>]", True),
    ("[>]", "[<]", True),
    ("[$|3|]", "[$|1|]", True),
    ("[$, >]", "[$]", True),
    ("[<, <2]", "[>1, $]", False),
]


@pytest.mark.parametrize(("textA", "textB", "compatible"), compatibility_list)
def test_connectors_compatible(textA, textB, compatible):
    bondA = g2rins.BondConnector.make(textA)
    bondB = g2rins.BondConnector.make(textB)

    assert bondA.is_compatible(bondB) == compatible
    assert bondB.is_compatible(bondA) == compatible


bc_smiles_list = [
    ("[C@@H]N(=O)c1ccncc1", []),
    ("[$][C@@H]N(=O)c1ccncc1[$]", ["[$]", "[$]"]),
    ("[$|3|][C@@H]N(=O)c1ccncc1[$|1 2|]", ["[$|3.0|]", "[$|1.0 2.0|]"]),
    ("[$|3|][C@@H]N(=[<]O)c1ccncc1[$|1 2|]", ["[$|3.0|]", "[<]", "[$|1.0 2.0|]"]),
    (
        "[$|3|][C@@H][>]N(=[<]O)c1cc[<|3|]ncc1[$|1 2|]",
        ["[$|3.0|]", "[>]", "[<]", "[<|3.0|]", "[$|1.0 2.0|]"],
    ),
    ("[>,>1]CCN([>,>1|2|])[$|3|]", ["[>, >1]", "[>, >1|2.0|]", "[$|3.0|]"]),
]


@pytest.mark.parametrize(("smi", "expected_bc_list"), bc_smiles_list)
def test_bond_connector_recognition(smi, expected_bc_list):
    big_smi = g2rins.Smiles.make(smi)

    assert len(big_smi.bond_connectors) == len(expected_bc_list)

    for actual, expected in zip(big_smi.bond_connectors, expected_bc_list):
        assert str(actual) == expected
