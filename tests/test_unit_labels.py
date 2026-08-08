# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import warnings
from collections import Counter

import g2rins

# Segmented multiblock coupled by one click bond, written with the blocks
# embedded inline in the unit tokens and with descriptor-paired block units.
INLINE_MULTIBLOCK = (
    "{[] [<]CCC{[>] [<]CC([>])C1=CC=CC=C1; ;  [<]}|log_normal(700.0, 1.05)|[>1], "
    "[<1]{[>] [<]CC=C(C)C[>]; ;  [<]}|log_normal(800.0, 1.05)|CCCCN1N=NC([>])=C1; "
    "C#C[>]; [<][H] []}|gauss(8700.0, 2500.0)|"
)
COMMA_MULTIBLOCK = (
    "{[] [<]CCC[>1], [<1]{[>] [<]CC([>])C1=CC=CC=C1;; [<]}|log_normal(700,1.05)|[>2], "
    "[<2]{[>] [<]CC=C(C)C[>];; [<]}|log_normal(800,1.05)|[>3], [<3]CCCCN1N=NC([>])=C1; "
    "C#C[>]; [<3]CCCCNNN []}|gauss(8700,2500)|"
)
STAR_CORE = "[H]{[$] [$]C(C[<])(C[<])(C[<]), [>]CC[<];; [>][H] []}|gauss(600, 150)|"
ALTERNATING = "{[] [<]CC[>2], [<2]NN[>]; [H][>,>2]; [<2,<][H] []}|gauss(140,0.01)|"
HOMOPOLYMER = "{[] [<]CCO[>]; CCCCO[>]; [<][H] []}|log_normal(298,1.05)|"
NESTED = (
    "{[] [<|9.0|]CC(C)O[>|9.0|], [<|6.0|]CC(CC)O[>|6.0|]; "
    "{[] [<|7.0|]CCO[>|7.0|], [<|4.0|]CC(CC)O[>|4.0|]; CCCCO[>]; [<]}|gauss(680.0, 215.0)|[>]; "
    "[<][H] []}|gauss(1649.0, 521.5)|"
)
NO_INITIATOR = "{[] [<]CCC[>], [<]NNN[>]|2|; ; [<]Cl []}|poisson(400)|"

CCC_FORMULA = ((6, 3),)
PS_FORMULA = ((6, 8),)
PI_FORMULA = ((6, 5),)
TRIAZOLE_FORMULA = ((6, 6), (7, 3))


def _labels(string, include_bond_connectors=False):
    graph = g2rins.G2rins.make(string).get_graph_creator().get_generative_graph(include_bond_connectors=include_bond_connectors)
    return graph, g2rins.derive_unit_labels(graph).unit_id


def _roles_by_formula(graph, unit_id):
    """Map each unit's real-atom formula to its label(s), for id-free asserts."""
    formulas = {}
    for label in set(unit_id.values()):
        atoms = Counter(graph.nodes[n]["atomic_num"] for n in graph.nodes if unit_id[n] == label)
        formula = tuple(sorted((num, count) for num, count in atoms.items() if num > 0))
        formulas.setdefault(formula, set()).add(label)
    return formulas


def _role(formulas, formula):
    roles = {label[0] for label in formulas[formula]}
    assert len(roles) == 1
    return roles.pop()


def test_segmented_multiblock_couplers_are_linkers():
    """The click couplers have no propagation path back to themselves: their
    recurrence comes from the outer distribution replaying whole segments."""
    graph, unit_id = _labels(INLINE_MULTIBLOCK)
    formulas = _roles_by_formula(graph, unit_id)
    assert formulas[CCC_FORMULA] == {"L0"}
    assert formulas[TRIAZOLE_FORMULA] == {"L1"}
    assert formulas[PS_FORMULA] == {"R0"}
    assert formulas[PI_FORMULA] == {"R1"}


def test_multiblock_writings_carry_the_same_roles():
    """Inline-embedded and descriptor-paired block writings label identically."""
    for include_bond_connectors in (False, True):
        inline = _roles_by_formula(*_labels(INLINE_MULTIBLOCK, include_bond_connectors))
        comma = _roles_by_formula(*_labels(COMMA_MULTIBLOCK, include_bond_connectors))
        for formula, expected in ((CCC_FORMULA, "L"), (PS_FORMULA, "R"), (PI_FORMULA, "R"), (TRIAZOLE_FORMULA, "L")):
            assert _role(inline, formula) == _role(comma, formula) == expected


def test_star_core_is_linker():
    """Every arm bond lands on the same extender port, so no walk returns to
    the core: one core per g2rins_object."""
    graph, unit_id = _labels(STAR_CORE)
    formulas = _roles_by_formula(graph, unit_id)
    assert formulas[((6, 4),)] == {"L0"}
    assert formulas[((6, 2),)] == {"R0"}


def test_alternating_copolymer_units_repeat():
    """The two cross bonds close a propagation path through both units."""
    graph, unit_id = _labels(ALTERNATING)
    formulas = _roles_by_formula(graph, unit_id)
    assert _role(formulas, ((6, 2),)) == "R"
    assert _role(formulas, ((7, 2),)) == "R"


def test_homopolymer_repeats():
    _, unit_id = _labels(HOMOPOLYMER)
    assert Counter(label[0] for label in set(unit_id.values())) == {"I": 1, "R": 1, "T": 1}


def test_junction_pseudo_units_are_linkers_and_modes_agree():
    """The nested-object junction (bond-connector atoms only) is a linker, and
    every real atom keeps its role letter in both connector modes."""
    graph_creator = g2rins.G2rins.make(NESTED).get_graph_creator()
    graph_plain = graph_creator.get_generative_graph(include_bond_connectors=False)
    graph_bc = graph_creator.get_generative_graph(include_bond_connectors=True)
    labels_plain = g2rins.derive_unit_labels(graph_plain).unit_id
    labels_bc = g2rins.derive_unit_labels(graph_bc).unit_id

    pseudo_only = {
        label
        for label in set(labels_bc.values())
        if all(graph_bc.nodes[n]["atomic_num"] < 0 for n in graph_bc.nodes if labels_bc[n] == label)
    }
    assert pseudo_only == {"L0"}
    for node, label in labels_plain.items():
        assert labels_bc[node][0] == label[0]


def test_no_initiator_keeps_every_bond():
    """Without initiators nothing is pruned: generation can seed from repeat
    units, and both units keep their propagation path back."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph, unit_id = _labels(NO_INITIATOR)
    roles = Counter(label[0] for label in set(unit_id.values()))
    assert roles == {"R": 2, "T": 1}
    assert not any(data["init_weight"] > 0 for _n, data in graph.nodes(data=True))