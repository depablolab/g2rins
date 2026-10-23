# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for conditional connectivity (group rules).

Phase 0 -- the per-symbol group suffix: parsing and round-tripping of the three
rules (ladder / exclusion / all), the compatibility matrix (ladder rigidity) and
the stochastic-object validation set. Phase 1 -- the generative-graph encoding:
every edge carries the four group-rule attributes, one edge per distinct group
annotation of a compatible bond connector pair, terminal-descriptor edges
annotated on the unit side. Phase 2 -- generation honors the EXCLUSION rule: a
site consumed through an exclusion-typed channel blocks its group siblings, a
site consumed through a plain channel removes their exclusion-typed channels
(whatever mode fired the bond, at any level). Phase 4 -- the LADDER rule: the
first ladder-typed bond commits its group and the remaining members bond the fresh
unit in the same step; a member consumed through any other channel strips the group's
ladder channels from its siblings; a ladder-typed terminator closes the last junction the
same way and the termination estimate prices its cap once, rungs included. Phase 5 -- the
two flavours of the ALL rule: all-typed sites are drawn before any other growth, lowest
all-generation first; a hard site (``allh``) keeps the object from stopping, a soft one
(``alls``) yields to the molecular weight target; hard channels that never settle are refused.
"""

import time
import warnings

import lark
import networkx as nx
import numpy as np
import pytest
from rdkit.Chem import Descriptors

import g2rins
from g2rins import GroupRule
from g2rins import ensemble_creator as _ec
from g2rins.bond import ALL_RULES
from g2rins.exception import (
    AllGroupOvershoot,
    AllRuleOnTargetOnlySite,
    DiscardedSamplingPaths,
    GroupPartnerNotPlain,
    GroupRuleOnNestedObjectBondConnector,
    GroupRuleOnTerminalBondConnector,
    GroupRulesOnBothPathEnds,
    IncompatibleGroupPair,
    IndistinguishableSymbolsInSite,
    LadderMemberWithoutChannel,
    LadderMemberWithoutPartner,
    MixedOuterSymbolsInGroup,
    MixedRulesInGroup,
    RepeatedGroupInSite,
    SingleMemberGroup,
    UnsettledHardAllRule,
)

GROUP_KEYS = ("source_group", "source_rule", "target_group", "target_rule")
SENTINEL = (-1, 0, -1, 0)
WEIGHT_KEYS = ("propagation_weight", "termination_weight", "transition_weight")

ROUND_TRIP_CASES = [
    "[$[$1]1]",
    "[<1[<1]1]",
    "[>, >1[]1]",
    "[>[allh]1]",
    "[>[allh]]",
    "[>[alls]1]",
    "[<1, <[$]2]",
    "[$[<]1, $[>]2]",
    "[<1[<1]1|2.0|]",
    "[<, >2]",
]


@pytest.mark.parametrize("text", ROUND_TRIP_CASES)
def test_round_trip(text):
    bond_connector = g2rins.SimpleBondConnector.make(text)
    assert str(bond_connector) == text


LEGACY_LADDER_FRAGMENTS = [
    # Formerly in smi.json big_smiles_features_unsupported_by_g2rins: ladder
    # nesting parses since phase 0, so they round-trip as unit-text fragments.
    "C([<1[<1]1])F(C[<1[<1]1])(N[>1[>1]2])N[>1[>1]2]",
    "CC([$1[$1]1])COc1ccccc1(CC(N)[$1[$1]1])(CC[$1[$1]2])CC(N)[$1[$1]2]",
]


@pytest.mark.parametrize("text", LEGACY_LADDER_FRAGMENTS)
def test_legacy_ladder_fragments_parse(text):
    assert str(g2rins.Smiles.make(text)) == text


def test_implicit_group_zero_is_omitted():
    bond_connector = g2rins.SimpleBondConnector.make("[>[allh]0]")
    assert str(bond_connector) == "[>[allh]]"


def test_terminal_bond_connector_parses_group_suffix():
    # The grammar accepts the suffix on any bond connector; a stochastic object
    # refuses it on its terminal bond connectors (see VALIDATION_ERROR_CASES).
    terminal = g2rins.TerminalBondConnector.make("[<[$]1]")
    assert str(terminal) == "[<[$]1]"
    (symbol,) = terminal.symbol
    assert symbol.group_rule == GroupRule.LADDER


def test_suffix_semantics():
    (symbol,) = g2rins.SimpleBondConnector.make("[>2[allh]3]").symbol
    assert symbol.idx == 2
    assert symbol.group_rule == GroupRule.ALL_HARD
    assert symbol.group_suffix.group_id == 3

    (symbol,) = g2rins.SimpleBondConnector.make("[>2[alls]3]").symbol
    assert symbol.group_rule == GroupRule.ALL_SOFT
    assert symbol.group_suffix.group_id == 3

    (symbol,) = g2rins.SimpleBondConnector.make("[<1[<4]5]").symbol
    assert symbol.group_rule == GroupRule.LADDER
    assert symbol.group_suffix.inner_symbol.idx == 4
    assert symbol.group_suffix.group_id == 5

    plain, exclusion = g2rins.SimpleBondConnector.make("[>, >1[]1]").symbol
    assert plain.group_rule == GroupRule.NONE
    assert plain.group_suffix is None
    assert exclusion.group_rule == GroupRule.EXCLUSION
    assert exclusion.group_id == 1


def test_rule_enum_values_stable():
    # Graph-feature encoding: these ints are documented and must never be renumbered.
    assert GroupRule.NONE == 0
    assert GroupRule.LADDER == 1
    assert GroupRule.EXCLUSION == 2
    assert GroupRule.ALL_HARD == 3
    assert GroupRule.ALL_SOFT == 4


def _single_symbol(text):
    return g2rins.SimpleBondConnector.make(text).symbol[0]


def test_ladder_rigidity():
    ladder = _single_symbol("[<[$]1]")
    plain = _single_symbol("[>]")
    assert not ladder.is_compatible(plain)
    assert not plain.is_compatible(ladder)
    assert ladder.is_compatible(_single_symbol("[>[$]2]"))
    assert not ladder.is_compatible(_single_symbol("[>[<]2]"))
    assert _single_symbol("[<[<1]1]").is_compatible(_single_symbol("[>[>1]2]"))


def test_nonladder_rules_pair_with_plain():
    assert _single_symbol("[>1[]1]").is_compatible(_single_symbol("[<1]"))
    assert _single_symbol("[>[allh]1]").is_compatible(_single_symbol("[<]"))


def test_group_edge_attrs_one_entry_per_distinct_symbol_pair():
    dual = g2rins.SimpleBondConnector.make("[<1, <[<]2]")
    partner = g2rins.SimpleBondConnector.make("[>1, >[>]1]")
    assert [tuple(entry[key] for key in GROUP_KEYS) for entry in dual.group_edge_attrs(partner)] == [SENTINEL, (2, 1, 1, 1)]
    assert [tuple(entry[key] for key in GROUP_KEYS) for entry in partner.group_edge_attrs(dual)] == [SENTINEL, (1, 1, 2, 1)]
    # Same-annotation duplicates collapse; incompatible pairs yield nothing.
    assert len(g2rins.SimpleBondConnector.make("[>, >]").group_edge_attrs(g2rins.SimpleBondConnector.make("[<]"))) == 1
    assert dual.group_edge_attrs(g2rins.SimpleBondConnector.make("[>3]")) == []


VALIDATION_ERROR_CASES = [
    pytest.param(
        "{[] [<]C([>1[allh]1])C([>2[]1])C[>]; ; [H][<] []}",
        MixedRulesInGroup,
        id="mixed-rules-in-group",
    ),
    pytest.param(
        "{[] [<]C([<[$]1])C([>[$]1])C[>]; ; [H][<] []}",
        MixedOuterSymbolsInGroup,
        id="mixed-outer-symbols-in-group",
    ),
    pytest.param(
        "{[] [<]C(C[>1[]1, >2[allh]1])C[>]; ; [H][<] []}",
        RepeatedGroupInSite,
        id="repeated-group-in-site",
    ),
    pytest.param(
        "{[] [<]C([<[$]1])C([<[$]1])C[>], [<]C([>[$]2])C([>[$]2])C([>[$]2])C[>]; ; [H][<] []}",
        IncompatibleGroupPair,
        id="group-pair-sizes-differ",
    ),
    pytest.param(
        "{[] [<]C([<[<1]1])C([<[<1]1])C[>], [<]C([>[>1]2])C([>[>2]2])C[>]; ; [H][<] []}",
        IncompatibleGroupPair,
        id="group-pair-inner-classes-differ",
    ),
    pytest.param(
        # One compatible member pair makes the groups partners; the other members
        # could never complete the rung.
        "{[] [<]C([<[<1]1])C([<[<2]1])C[>], [<]C([>[>1]2])C([>[>3]2])C[>]; ; [H][<] []}",
        IncompatibleGroupPair,
        id="group-pair-partial-inner-overlap",
    ),
    pytest.param(
        "{[] [<]C([>1[]1])C([>1[]1])C[>], [<]C([<1[]2])C([<1[]2])C[>]; ; [H][<] []}",
        GroupPartnerNotPlain,
        id="exclusion-partner-not-plain",
    ),
    pytest.param(
        # A '$' exclusion channel is compatible with its own symbol on the next
        # unit instance, which is a group-typed partner too.
        "{[] [$[]1]CC[$]; C[$]; [H][$] []}",
        GroupPartnerNotPlain,
        id="exclusion-self-pair-not-plain",
    ),
    pytest.param(
        "{[] [<]C([<[allh]1])C([<[allh]1])C[>], [<]C([>[allh]2])C([>[allh]2])C[>]; ; [H][<] []}",
        GroupPartnerNotPlain,
        id="all-partner-not-plain",
    ),
    pytest.param(
        # An initiator channel does bond to a repeat unit's channel.
        "{[] [$]CC[$[]1]; C[$[]1]; [H][$] []}",
        GroupPartnerNotPlain,
        id="exclusion-initiator-vs-repeat-not-plain",
    ),
    pytest.param(
        # No repeat unit takes index 1, so this initiator really bonds to the terminators.
        "{[] [<]CC[>]; C([>1[]1])([>1[]1]); [<1[]1]CC[<1[]1], [<][H] []}|poisson(300)|",
        GroupPartnerNotPlain,
        id="partnerless-initiator-vs-typed-terminator",
    ),
    pytest.param(
        "{[] [<]C([>1[]1])C([>1[]1])C[>]; ; [<1[]1][H], [<][H] []}|poisson(200)|",
        GroupPartnerNotPlain,
        id="exclusion-unit-vs-typed-terminator",
    ),
    pytest.param(
        "{[<[]1] [<]CC([>])[>]; ; [H][<] []}",
        GroupRuleOnTerminalBondConnector,
        id="group-rule-on-terminal-bond-connector",
    ),
    pytest.param(
        # The bond connector after the nested object relays its exits to this level.
        "{[] [<]CC[>], [<]{[>] [<]CC(C)O[>]; ; [<]F [<]}|poisson(100)|[>[]1]; [<][H]; [<][H] []}|poisson(400)|",
        GroupRuleOnNestedObjectBondConnector,
        id="group-rule-on-nested-object-bond-connector",
    ),
    pytest.param(
        # Formerly the multilevel fixture: group 2 sits on the two bond connectors
        # that attach the nested object, interior to every bond connector path.
        "{[] [<[allh]2]{[>] [<[allh]1]CC(C[<[allh]1])O[>]; ; [<]F [<]}|poisson(100)|[>[allh]2]; [<][H]; [<][H] []}|poisson(400)|",
        GroupRuleOnNestedObjectBondConnector,
        id="group-rule-on-nested-object-bond-connector-all",
    ),
]


@pytest.mark.parametrize("text, expected_error", VALIDATION_ERROR_CASES)
def test_validation_errors(text, expected_error):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(expected_error):
            try:
                g2rins.StochasticObject.make(text)
            except lark.exceptions.VisitError as exc:
                raise exc.__context__  # trunk-ignore(ruff/B904)


END_GROUP_ONLY_PAIR_CASES = [
    # Initiators never bond to initiators and terminators never to terminators, and an
    # initiator that a repeat unit takes never reaches the terminators either, so
    # group-typed symbols that conjugate only across such pairs have no partner to check.
    pytest.param("{[] [$]CC[$]; C([$[allh]1])([$[allh]1])([$[allh]1]); [$][H] []}|poisson(300)|", id="dollar-star-all-initiator"),
    pytest.param("{[] [$]CC[$]; C([$[]1])([$[]1]); [$][H] []}|poisson(300)|", id="dollar-exclusion-initiator"),
    pytest.param("{[] [$]CC[$]; C[$]; [$[]1][H], [$[]1]F []}|poisson(300)|", id="dollar-exclusion-terminators"),
    pytest.param("{[] [<]CC[>]; C([>[>]1])([>[>]1]), C([>[>]1])([>[>]1])([>[>]1]); [H][<] []}|poisson(300)|", id="ladder-groups-on-two-initiators"),
    pytest.param("{[] [<]CC[>]; C([>[]1])([>[]1]); [<[]1]CC[<[]1] []}|poisson(300)|", id="connected-exclusion-initiator-vs-typed-terminators"),
    pytest.param("{[] [<]CC[>]; C([>,>[>]1])([>,>[>]1]); [<[<]1]C([<[<]1])[<[<]1] []}|poisson(300)|", id="connected-ladder-initiator-vs-larger-ladder-terminator-group"),
]


@pytest.mark.parametrize("text", END_GROUP_ONLY_PAIR_CASES)
def test_end_group_only_pairs_are_not_partners(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g2rins.StochasticObject.make(text)


def test_dollar_star_all_initiator_encoding():
    text = "{[] [$]CC[$]; C([$[allh]1])([$[allh]1])([$[allh]1]); [$][H] []}|poisson(300)|"
    _assert_no_diagnostics(text)
    annotated = [edge for edge in _bond_connector_edges(text) if edge[3] != SENTINEL]
    # Three initiator sites x the two sites of the repeat unit, all through the all-group.
    assert len(annotated) == 6
    assert all(edge == ("[$[allh]1]", "[$]", "transition_weight", (1, 3, -1, 0)) for edge in annotated)


def test_connected_initiator_never_reaches_typed_terminators():
    text = "{[] [<]CC[>]; C([>[]1])([>[]1]); [<[]1]CC[<[]1] []}|poisson(300)|"
    _assert_no_diagnostics(text)
    annotated = [edge for edge in _bond_connector_edges(text) if edge[3] != SENTINEL]
    # Two initiator transitions into the unit and two unit terminations; no initiator-terminator edge.
    expected = [("[>[]1]", "[<]", "transition_weight", (1, 2, -1, 0))] * 2 + [("[>]", "[<[]1]", "termination_weight", (-1, 0, 1, 2))] * 2
    assert sorted(annotated) == sorted(expected)


def test_exclusion_beside_ladder_idx_reuse_is_legal():
    # Ladder rigidity means the exclusion channel and the ladder channel never
    # form an edge, so sharing outer index 1 is not an exclusion-partner error.
    text = "{[] [<]C([>1[]1])C([>1[]1])C[>], [<]C([<1[<]1])C([<1[<]1])C[>], [<]C([>1[>]2])C([>1[>]2])C[>]; C[>]; [H][<] []}|poisson(100)|"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g2rins.StochasticObject.make(text)
    assert not caught


def test_disjoint_ladder_channels_are_not_partners():
    # Groups 1/2 pair through inner channel 1 and groups 3/4 through inner
    # channel 2; conjugate outer symbols alone (1 vs 4, 3 vs 2) make no partner.
    text = "{[] [<]C([<[<1]1])C([<[<1]1])C[>], [<]C([>[>1]2])C([>[>1]2])C[>], [<]C([<[<2]3])C([<[<2]3])C[>], [<]C([>[>2]4])C([>[>2]4])C[>]; ; [H][<] []}|poisson(400)|"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g2rins.StochasticObject.make(text)  # no IncompatibleGroupPair
    # The string declares no initiator, so only the initiation warnings may fire.
    assert not [caught_warning for caught_warning in caught if issubclass(caught_warning.category, (SingleMemberGroup, IndistinguishableSymbolsInSite))]
    edges = _bond_connector_edges(text)
    ladder_edges = {(source, target) for source, target, _mode, values in edges if values != SENTINEL}
    assert ladder_edges == {("[<[<1]1]", "[>[>1]2]"), ("[>[>1]2]", "[<[<1]1]"), ("[<[<2]3]", "[>[>2]4]"), ("[>[>2]4]", "[<[<2]3]")}


def test_single_member_group_warns():
    with pytest.warns(SingleMemberGroup):
        g2rins.StochasticObject.make("{[] [<]C(C[>9, <[$]1])C[>]; ; [H][<] []}")


def test_ladder_only_chain_ends_raise_no_diagnostics():
    # Ladder-only sites may finish unreacted at conversion (implicit valence):
    # entry-side groups are consumed at engagement, initiator groups initiate,
    # and incomplete chain-end groups are intended behavior -- no diagnostics.
    text = "{[] [<[<]2]OC(O[<[<]2])CC(O[>[>]1])O[>[>]1]; C(O[>[>]1])O[>[>]1]; [<][H] []}|poisson(1000)|"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stochastic_object = g2rins.StochasticObject.make(text)
    assert stochastic_object is not None
    assert not caught


INDISTINGUISHABLE_CASES = [
    pytest.param("{[<] [<]CC([>,>[allh]1])C([>,>[allh]1])[>]; ; [H][<] [<]}|poisson(200)|", id="plain-beside-all"),
    pytest.param("{[] [<]CC([>,>[]1])C([>,>[]1])[>]; C[>]; [H][<] []}|poisson(200)|", id="plain-beside-exclusion"),
]


@pytest.mark.parametrize("text", INDISTINGUISHABLE_CASES)
def test_indistinguishable_symbols_warn(text):
    # A plain symbol with the same outer symbol and index as an all- or
    # exclusion-typed one leaves partners no way to pick the channel.
    with pytest.warns(IndistinguishableSymbolsInSite):
        g2rins.StochasticObject.make(text)


# --- Phase 1: generative-graph encoding ---------------------------------------

EXCLUSION_TEXT = "{[] [>,>1[]1]N([>,>1[]1])CCN([>,>1[]2])[>,>1[]2], [<1]C(=O)CCCCCO[<1], [<]CCO[>]; O[>1]; [H][<], [H][<1] []}|poisson(500)|"
EMBEDDED_EXIT_TEXT = "CC{[<] [<]CC([>,>1[]1])[>]; ; [H][<] [<1]}|poisson(200)|CC"
EXIT_INTO_ENCLOSING_TEXT = "{[] [<]{[>] [<]CC([>,>1[]1])C([>,>1[]1])[>]; ; [<]F [<1]}|poisson(100)|[>]; [>][H]; [<][H] []}|poisson(400)|"
NESTED_BLOCK_TEXT = "{[] [<]CC(C)O[>]; {[] [>,>1[]]N([>,>1[]])CCN([>,>1[]1])([>,>1[]1]), [<1]C(=O)CCCC(=O)[<1]; O[>1], [H][<]; [>1]O [<]}|gauss(4000,500)|[>]; [<][H] []}|gauss(5400,1000)|"
# The typed site [<1[]1] is only ever entered (no outgoing edge): the group rule must still see it when the chain starts there.
ENTRY_ONLY_SITE_TEXT = "{[] [<1[]1]C([>2[]1])[>], [<]CC[>]; O[>1]; [<][H], [<2]F []}|poisson(100)|"
# A nested object splits the unit in two static fragments: the sites around it are one exclusion group all the same,
# through one object, two objects joined directly, and an inner object that ends the middle unit.
SPLIT_UNIT_TEXT = "{[] [<]C(O[>1[]1]){[>] [<]CC[>];; [<]}|uniform(56,56)|C(O[>1[]1])[>]; C[>]; [<][H], [<1]Br []}|uniform(700,700)|"
JOINED_OBJECTS_SPLIT_UNIT_TEXT = "{[] [<]C(O[>1[]1]){[>] [<]CC[>];; [<]}|uniform(56,56)|{[>] [<]OO[>];; [<]}|uniform(64,64)|C(O[>1[]1])[>]; C[>]; [<][H], [<1]Br []}|uniform(800,800)|"
TWO_LEVEL_SPLIT_UNIT_TEXT = "{[] [<]C(O[>1[]1]){[>] [<]C{[>] [<]CC[>];; [<]}|uniform(56,56)|[>];; [<]}|uniform(170,170)|C(O[>1[]1])[>]; C[>]; [<][H], [<1]Br []}|uniform(900,900)|"
# The object ends the unit: its exit opens the next unit, whose site is a group of its own.
OBJECT_ENDS_UNIT_TEXT = "{[] [<]C(O[>1[]1]){[>] [<]CC[>];; [<]}|uniform(56,56)|[>]; C[>]; [<][H], [<1]Br []}|uniform(700,700)|"
# The chain enters the unit through a group member before the object; the group's other member sits in the text after it.
# Typed entry: that member is blocked (no Br). Plain entry through the same site: it loses the typed channel and keeps the plain one (F).
TYPED_ENTRY_TAIL_TEXT = "{[] [>1[]1]C{[>] [<]CC[>];; [<]}|uniform(56,56)|C(O[>2[]1])[>], [<]CC[>]; C[<1]; [<][H], [<2]Br []}|uniform(300,300)|"
PLAIN_ENTRY_TAIL_TEXT = "{[] [>1[]1,>3]C{[>] [<]CC[>];; [<]}|uniform(56,56)|C(O[>2[]1,>4])[>], [<]CC[>]; C[<3]; [<][H], [<1]Cl, [<2]Br, [<4]F []}|uniform(300,300)|"
TYPED_ENTRY_TAIL_WITH_PLAIN_CAP_TEXT = "{[] [>1[]1,>3]C{[>] [<]CC[>];; [<]}|uniform(56,56)|C(O[>2[]1,>4])[>], [<]CC[>]; C[<1]; [<][H], [<3]Cl, [<2]Br, [<4]F []}|uniform(300,300)|"
# Two typed sibling cap sites per unit realize one cap; the one-site string is the reference for the mass they add.
TYPED_SIBLING_CAPS_TEXT = "{[] [<]CC([>1[]1])([>1[]1])[>]; O[>]; [<][H], [<1]Br []}|poisson(600)|"
SINGLE_CAP_SITE_TEXT = "{[] [<]CC([>1])[>]; O[>]; [<][H], [<1]Br []}|poisson(600)|"
LADDER_TEXT = "{[] [<[<]2]OC(O[<[<]2])CC(O[>[>]1])O[>[>]1]; C(O[>[>]1])O[>[>]1]; [<][H] []}|poisson(1000)|"
DUAL_CHANNEL_TEXT = "{[] [<1,<[<]2]OC(O[<1,<[<]2])CC(O[>1,>[>]1])O[>1,>[>]1]; C(O[>1,>[>]1])O[>1,>[>]1]; [<1][H] []}|poisson(1000)|"
DUAL_CHANNEL_SWAPPED_TEXT = "{[] [<[<]2,<1]OC(O[<[<]2,<1])CC(O[>[>]1,>1])O[>[>]1,>1]; C(O[>[>]1,>1])O[>[>]1,>1]; [<1][H] []}|poisson(1000)|"
ALL_TEXT = "{[] [<]CCO[>]; C(O[>[allh]1])(CO[>[allh]1])(CO[>[allh]1]); [H][<] []}|poisson(300)|"


def _graph_creator(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return g2rins.G2rins.make(text).get_graph_creator()


def _generative_graph(graph_creator, include_bond_connectors=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return graph_creator.get_generative_graph(include_bond_connectors=include_bond_connectors)


def _group_values(data):
    return tuple(data[key] for key in GROUP_KEYS)


def _mode(data):
    return next(key for key in WEIGHT_KEYS if data[key] > 0)


def _bond_connector_edges(text):
    """(source text, target text, weight key, group values) of every edge between two bond connector nodes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph, extra_graph_info = _graph_creator(text).get_generative_graph(include_bond_connectors=True, return_extra_graph_info=True)
    edges = []
    for u, v, data in graph.edges(data=True):
        source, target = graph.nodes[u]["atomic_num"], graph.nodes[v]["atomic_num"]
        # Bond connectors are the non-atom nodes; a nested object adjacent to a bond
        # connector also yields static edges between two of them, which carry no rule.
        if source < 0 and target < 0 and not data["static"]:
            edges.append((extra_graph_info[source], extra_graph_info[target], _mode(data), _group_values(data)))
    return edges


def _non_static_parallel_pairs(graph):
    pairs = []
    for u in graph:
        for v in graph[u]:
            edges = [data for data in graph[u][v].values() if not data["static"]]  # key order = insertion order
            if len(edges) > 1:
                pairs.append(edges)
    return pairs


def test_every_edge_carries_group_schema(g2rins_list):
    # The GNN guarantee: the same four int keys on every edge of every graph,
    # sentinels for strings that declare no group rule.
    for text in g2rins_list:
        graph_creator = _graph_creator(text)
        for include_bond_connectors in (True, False):
            for _u, _v, data in _generative_graph(graph_creator, include_bond_connectors).edges(data=True):
                values = _group_values(data)
                assert all(type(value) is int for value in values)
                assert values == SENTINEL


def test_exclusion_edge_annotations():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g2rins.G2rins.make(EXCLUSION_TEXT)
    assert not caught
    edges = _bond_connector_edges(EXCLUSION_TEXT)
    assert len(edges) == 37  # one compatible symbol pair per descriptor pair: no parallel edges
    site = "[>, >1[]1]"
    # Leaving through the group channel (growth AND termination) is group-typed on the source side ...
    assert {(mode, values) for source, target, mode, values in edges if source == site and target == "[<1]"} == {("propagation_weight", (1, 2, -1, 0)), ("termination_weight", (1, 2, -1, 0))}
    assert {values for source, target, mode, values in edges if source == "[>, >1[]2]" and target == "[<1]"} == {(2, 2, -1, 0)}
    # ... entering it is group-typed on the target side ...
    assert {values for source, target, mode, values in edges if source == "[<1]" and target == site} == {(-1, 0, 1, 2)}
    # ... and the plain channel, the plain cap and the initiator stay plain.
    assert {values for source, target, mode, values in edges if source == site and target == "[<]"} == {SENTINEL}
    assert {values for source, target, mode, values in edges if source == "[>1]"} == {SENTINEL}


def test_ladder_edge_annotations():
    edges = _bond_connector_edges(LADDER_TEXT)
    assert len(edges) == 12
    # Ladder edges exist only between inner-conjugate groups, never between two
    # group-1 sites; the plain cap is rigid-incompatible, so no termination edge.
    assert {values for *_, values in edges} == {(2, 1, 1, 1), (1, 1, 2, 1)}
    assert all(mode != "termination_weight" for _source, _target, mode, _values in edges)


@pytest.mark.parametrize(
    "text, expected_order",
    [
        pytest.param(DUAL_CHANNEL_TEXT, (SENTINEL, "ladder"), id="plain-first"),
        pytest.param(DUAL_CHANNEL_SWAPPED_TEXT, ("ladder", SENTINEL), id="ladder-first"),
    ],
)
def test_dual_channel_sites_yield_parallel_edges(text, expected_order):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g2rins.G2rins.make(text)
    assert not caught
    edges = _bond_connector_edges(text)
    assert len(edges) == 26  # 12 descriptor pairs x (plain + ladder) + 2 plain termination edges
    generative_graph = _generative_graph(_graph_creator(text))
    parallel_pairs = _non_static_parallel_pairs(generative_graph)
    assert len(parallel_pairs) == 12
    for pair in parallel_pairs:
        values = [_group_values(data) for data in pair]
        ladder = (2, 1, 1, 1) if values[0] == (2, 1, 1, 1) or values[-1] == (2, 1, 1, 1) else (1, 1, 2, 1)
        assert values == [ladder if entry == "ladder" else entry for entry in expected_order]
        mode = _mode(pair[0])
        # Parallel edges of one descriptor pair share its weight.
        assert pair[0][mode] == pair[1][mode] > 0


def test_all_edge_annotations():
    generative_graph = _generative_graph(_graph_creator(ALL_TEXT))
    annotated = [data for _u, _v, data in generative_graph.edges(data=True) if _group_values(data) != SENTINEL]
    assert len(annotated) == 3
    assert all(_group_values(data) == (1, 3, -1, 0) and _mode(data) == "transition_weight" for data in annotated)


def test_ladder_self_pair_beside_plain_channel_yields_parallel_self_loops():
    text = "{[] [$[$]1, $]CC[$[$]1, $]; C[$]; [H][$] []}|poisson(100)|"
    assert {values for source, target, _mode, values in _bond_connector_edges(text) if source == target == "[$[$]1, $]"} == {(1, 1, 1, 1), SENTINEL}
    pairs = [[_group_values(data) for data in pair] for pair in _non_static_parallel_pairs(_generative_graph(_graph_creator(text)))]
    assert pairs and all(sorted(pair) == sorted([(1, 1, 1, 1), SENTINEL]) for pair in pairs)


def test_terminal_descriptor_edges_carry_unit_side_annotation():
    # A repeat-unit site leaving the stochastic object through a group channel
    # is a group-typed consumption; the terminal descriptor's own side is plain.
    exit_edges = _bond_connector_edges("{[<] [<]CC([>,>1[]1])[>]; ; [H][<] [<1]}|poisson(200)|")
    assert {values for source, target, _mode, values in exit_edges if target == "[<1]"} == {(1, 2, -1, 0)}
    entry_edges = _bond_connector_edges("{[>1] [<,<1[]1]CC([<,<1[]1])[>]; ; [H][<] [>]}|poisson(200)|")
    assert {values for source, target, _mode, values in entry_edges if source == "[>1]"} == {(-1, 0, 1, 2)}
    # Embedded, the exit edge reaches the bond-connector-free graph and generation accepts it.
    generative_graph = _generative_graph(_graph_creator(EMBEDDED_EXIT_TEXT))
    annotated = [data for _u, _v, data in generative_graph.edges(data=True) if _group_values(data) != SENTINEL]
    assert [(_group_values(data), _mode(data)) for data in annotated] == [((1, 2, -1, 0), "transition_weight")]
    g2rins.EnsembleCreator(generative_graph)


# Each level-2 nitrogen holds two dual-channel sites: the typed channel is the level-0 unit's only entry and fires as
# promoted growth after two hand-offs, the plain one grows at levels 2 and 1. The level masses leave level 1 little
# growth, so enough typed channels survive to the level-0 hand-off for every chain to complete.
MULTILEVEL_EXCLUSION_TEXT = "{[] [<,<2]C(=O)CC[>2]; {[] [<1]CC[>1]; {[] [<1]CC(CCN([>1,>[]])([>1,>[]]))CCN([>1,>[]])([>1,>[]]), [<1]CCO[>1]; O[>1]; [<]|[<1]}|poisson(4000)|[>]|[>1]; [<1][H] [<]}|poisson(4250)|[>]; [<2][H] []}|poisson(4600)|"


def test_group_rule_survives_bond_connector_path_across_levels():
    # The exclusion channel of the level-2 unit exits through a terminal bond
    # connector, a level-1 bond connector, another terminal bond connector and a
    # level-0 bond connector before reaching the level-0 unit. Every relay is
    # plain, so the bond carries the rule of the unit it leaves.
    _assert_no_diagnostics(MULTILEVEL_EXCLUSION_TEXT)
    generative_graph = _generative_graph(_graph_creator(MULTILEVEL_EXCLUSION_TEXT))
    annotated = [(u, v, _mode(data), _group_values(data)) for u, v, data in generative_graph.edges(data=True) if _group_values(data) != SENTINEL]
    assert len(annotated) == 4  # one per nitrogen site of the level-2 unit
    for u, v, mode, values in annotated:
        assert (mode, values) == ("transition_weight", (0, 2, -1, 0))
        # From a split node of the level-2 nitrogen to the level-0 carbonyl carbon.
        assert generative_graph.nodes[u]["atomic_num"] == 0
        assert 7 in {generative_graph.nodes[w]["atomic_num"] for w in generative_graph.neighbors(u)}
        assert generative_graph.nodes[v]["atomic_num"] == 6


def test_group_rule_exit_into_enclosing_object_reaches_generation():
    # A ruled exit followed by plain relays used to contract to sentinels; the
    # annotation now survives to the graph the sampler consumes.
    generative_graph = _generative_graph(_graph_creator(EXIT_INTO_ENCLOSING_TEXT))
    assert {_group_values(data) for _u, _v, data in generative_graph.edges(data=True) if _group_values(data) != SENTINEL} == {(1, 2, -1, 0)}
    g2rins.EnsembleCreator(generative_graph)


def test_group_rules_on_both_path_ends_are_refused():
    # The level-1 exclusion channel exits into a level-0 exclusion channel: one
    # bond would carry a rule in two unit instances.
    text = "{[] [<,<1[]1]CC[>]; {[] [<]CC([>,>1[]1])[>]; ; [<]F [<1]}|poisson(100)|[>1]; [<][H] []}|poisson(400)|"
    graph_creator = _graph_creator(text)
    _generative_graph(graph_creator, include_bond_connectors=True)
    with pytest.raises(GroupRulesOnBothPathEnds):
        _generative_graph(graph_creator)


def test_group_rules_at_one_level_survive_nesting():
    # A bond crossing a level carries at most one annotation here, so it contracts.
    generative_graph = _generative_graph(_graph_creator(NESTED_BLOCK_TEXT))
    assert {(-1, 0, 0, 2), (0, 2, -1, 0), (-1, 0, 1, 2), (1, 2, -1, 0)} <= {_group_values(data) for _u, _v, data in generative_graph.edges(data=True)}


def test_exclusion_only_graphs_build_an_ensemble_creator():
    text = "{[] [<]C([>1[]1])C([>1[]1])C[>]; ; [H][<], [H][<1] []}|poisson(200)|"
    graph_creator = _graph_creator(text)
    # The bond-connector-free graph is a complete, exportable product, and the
    # creator builds whichever way it is constructed (EXCLUSION is no longer gated).
    generative_graph = _generative_graph(graph_creator)
    assert (1, 2, -1, 0) in {_group_values(data) for _u, _v, data in generative_graph.edges(data=True)}
    exported = g2rins.generative_graph_json_data(generative_graph)
    assert all(set(GROUP_KEYS) <= set(edge) for edge in exported["graph"]["edges"])
    g2rins.EnsembleCreator(generative_graph)
    graph_creator.get_ensemble_creator()


def test_consumers_read_absent_group_keys_as_sentinels():
    generative_graph = _generative_graph(_graph_creator("{[] [<]CC([>])c1ccccc1; [>][H]; [<][H] []}|gauss(1000, 45)|"))
    for _u, _v, data in generative_graph.edges(data=True):
        for key in GROUP_KEYS:
            data.pop(key)
    g2rins.EnsembleCreator(generative_graph)


# --- Real-life strings: regression fixtures for every phase (phase 1 pins the encoding) ---

# Published ladder synthesis (Polym. Chem. 2026, 17(24), 2539-2547): regiospecific
# inner classes, ladder-typed initiator and terminator with implicit group 0.
LADDER_REFERENCE_TEXT = "{[] [>[>1]2]C(C(OC([>[>2]2])=O)=C1)=CC2=C1C([<[<2]1])=C([<[<1]1])C(O2)=O; O=C([>[>2]])OC1=C([>[>1]])C=C2C(CC(O2)=O)=C1; O=C(C([<[<1]])=C1[<[<2]])OC2=C1C=C3C(CC(O3)=O)=C2 []}|gauss(5000,1000)|"
# Three-site initiator whose sites must all initiate before propagation (implicit group 0).
ALL_STAR_TEXT = "{[] [<]C(C)C(=O)O[>], [<]CC(=O)O[>]; [>[allh]]OCC(O[>[allh]])CO[>[allh]]; [<][H] []}|schulz_zimm(1800, 1200)|"


def _assert_no_diagnostics(text):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g2rins.G2rins.make(text)
    assert not caught


def test_ladder_reference_string_encoding():
    _assert_no_diagnostics(LADDER_REFERENCE_TEXT)
    edges = _bond_connector_edges(LADDER_REFERENCE_TEXT)
    # Every site has exactly one partner (inner classes 1 and 2 pair regiospecifically).
    assert len(edges) == 8
    assert set(edges) == {
        ("[<[<1]1]", "[>[>1]2]", "propagation_weight", (1, 1, 2, 1)),
        ("[<[<2]1]", "[>[>2]2]", "propagation_weight", (1, 1, 2, 1)),
        ("[>[>1]2]", "[<[<1]1]", "propagation_weight", (2, 1, 1, 1)),
        ("[>[>2]2]", "[<[<2]1]", "propagation_weight", (2, 1, 1, 1)),
        ("[>[>1]2]", "[<[<1]]", "termination_weight", (2, 1, 0, 1)),
        ("[>[>2]2]", "[<[<2]]", "termination_weight", (2, 1, 0, 1)),
        ("[>[>1]]", "[<[<1]1]", "transition_weight", (0, 1, 1, 1)),
        ("[>[>2]]", "[<[<2]1]", "transition_weight", (0, 1, 1, 1)),
    }


def test_all_star_string_encoding():
    _assert_no_diagnostics(ALL_STAR_TEXT)
    edges = _bond_connector_edges(ALL_STAR_TEXT)
    assert len(edges) == 16
    annotated = [edge for edge in edges if edge[3] != SENTINEL]
    # Three initiator sites x two repeat units, all transitions through the all-group.
    assert len(annotated) == 6
    assert all(edge == ("[>[allh]]", "[<]", "transition_weight", (0, 3, -1, 0)) for edge in annotated)
    # ALL is no longer gated: the creator builds; sampling is pinned by the phase-3 tests.
    g2rins.EnsembleCreator(_generative_graph(_graph_creator(ALL_STAR_TEXT)))


# -- Phase 2: generation under the EXCLUSION rule --------------------------------------


def _timeline(monkeypatch):
    """Record every realized bond (fresh unit's tag, consumed source site, consumed target site); a site is
    (instance tag, exclusion groups held, channel group, channel rule, ladder groups held); a rung is logged like a bond. Unit tags are logged too: tag 1 opens a
    chain, a tag drawn again inside a chain is a checkpoint rollback that discarded the units from that tag on.
    The fragments the termination-mass estimate builds on a throwaway tracker draw tags as well; they are skipped."""
    log, observing = [], []
    next_tag, realize, observe = _ec._StochasticObjectTracker.next_unit_instance, _ec._PartialAtomGraph.realize_bond, _ec._PartialAtomGraph._observational_fragment
    internal = _ec._PartialAtomGraph._realize_internal_bond

    def tag(self):
        value = next_tag(self)
        if not observing:
            log.append(("unit", value))
        return value

    def fragment(self, *args):
        observing.append(True)
        try:
            return observe(self, *args)
        finally:
            observing.pop()

    def site(half_bond, group, rule):
        return None if half_bond is None else (half_bond.instance, frozenset(half_bond.exclusion_groups()), group, rule, frozenset(half_bond.ladder_groups()))

    def bond(self, other, source_half_bond, fired_edge, other_idx, bond_attr, rng):
        source = site(source_half_bond, fired_edge.get("source_group", -1), fired_edge.get("source_rule", 0))
        target = site(other._consumed_half_bond, fired_edge.get("target_group", -1), fired_edge.get("target_rule", 0))
        log.append(("bond", self.stochastic_tracker._unit_instances, source, target))
        return realize(self, other, source_half_bond, fired_edge, other_idx, bond_attr, rng)

    def rung(self, member, partner, edge):
        source = site(member, edge.get("source_group", -1), edge.get("source_rule", 0))
        target = site(partner, edge.get("target_group", -1), edge.get("target_rule", 0))
        log.append(("rung", self.stochastic_tracker._unit_instances, source, target))
        return internal(self, member, partner, edge)

    monkeypatch.setattr(_ec._StochasticObjectTracker, "next_unit_instance", tag)
    monkeypatch.setattr(_ec._PartialAtomGraph, "realize_bond", bond)
    monkeypatch.setattr(_ec._PartialAtomGraph, "_realize_internal_bond", rung)
    monkeypatch.setattr(_ec._PartialAtomGraph, "_observational_fragment", fragment)
    return log


def _surviving_bonds_per_chain(log):
    chains, high = [], 0
    for event in log:
        if event[0] == "unit":
            if event[1] == 1:
                chains.append([])
            elif event[1] <= high:
                chains[-1][:] = [bond for bond in chains[-1] if bond[1] < event[1]]
            high = event[1]
        else:
            chains[-1].append(event)
    return chains


def _rule_violations(bonds):
    """Per instance and group: after a member fired through the typed channel no member fires again; after a member
    fired through a plain channel no member fires through the typed one. Returns (violations, typed consumptions)."""
    engaged, bystander, violations, typed = set(), set(), 0, 0
    for _kind, _fresh_tag, source, target in bonds:
        for site in (source, target):
            if site is None or not site[1]:
                continue
            tag, groups, group, rule, _ladder = site
            is_typed = rule == GroupRule.EXCLUSION and group in groups
            violations += sum(1 for g in groups if (tag, g) in engaged or (is_typed and (tag, g) in bystander))
            (engaged if is_typed else bystander).update((tag, g) for g in groups)
            typed += is_typed
    return violations, typed


def _sample(text, n_chains, seed=7, output_format="mol_graph"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _graph_creator(text).get_ensemble_creator().create_ensemble(n_chains, output_format=output_format, seed=seed)


@pytest.mark.parametrize(
    "text, n_chains",
    [(EXCLUSION_TEXT, 12), (EMBEDDED_EXIT_TEXT, 12), (EXIT_INTO_ENCLOSING_TEXT, 12), (NESTED_BLOCK_TEXT, 8), (ENTRY_ONLY_SITE_TEXT, 12), (MULTILEVEL_EXCLUSION_TEXT, 6), (SPLIT_UNIT_TEXT, 6)],
)
def test_sampling_obeys_the_channel_rule_at_every_level(monkeypatch, text, n_chains):
    # Same-level growth and caps, exits through terminal bond connectors into the
    # enclosing object, a nested block, and a typed channel that fires as promoted
    # growth two hand-offs after its unit was placed: the rule holds on the sampler's
    # own bond events, and the typed channel does fire in each string.
    log = _timeline(monkeypatch)
    chains = _sample(text, n_chains)
    assert chains is not None and len(chains) == n_chains
    assert all(data["atomic_num"] > 0 for chain in chains for _node, data in chain.nodes(data=True))
    results = [_rule_violations(bonds) for bonds in _surviving_bonds_per_chain(log)]
    assert sum(violations for violations, _typed in results) == 0
    assert sum(typed for _violations, typed in results) > 0


def _site_nodes(generative_graph, origin):
    """The atom plus the connector placeholders split off it (static edges); they hold the atom's site edges."""
    nodes, queue = {origin}, [origin]
    while queue:
        current = queue.pop()
        for u, v, data in list(generative_graph.out_edges(current, data=True)) + list(generative_graph.in_edges(current, data=True)):
            other = v if u == current else u
            if data.get("static") and generative_graph.nodes[other]["atomic_num"] == 0 and other not in nodes:
                nodes.add(other)
                queue.append(other)
    return nodes


def _bond_channel(generative_graph, site_origin, neighbour_origin):
    """'typed' or 'plain' for a realized non-static bond, read from the generative edge that fired it; 'static' otherwise."""
    site_nodes, neighbour_nodes = _site_nodes(generative_graph, site_origin), _site_nodes(generative_graph, neighbour_origin)
    for node in site_nodes:
        for _u, v, data in generative_graph.out_edges(node, data=True):
            if v in neighbour_nodes and not data.get("static"):
                return "typed" if data.get("source_rule", 0) == GroupRule.EXCLUSION else "plain"
        for u, _v, data in generative_graph.in_edges(node, data=True):
            if u in neighbour_nodes and not data.get("static"):
                return "typed" if data.get("target_rule", 0) == GroupRule.EXCLUSION else "plain"
    return "static"


def _grouped_site_bond_counts(generative_graph, chains):
    """(typed, plain) realized bond counts of every atom holding exclusion channels, over all chains."""
    grouped = {
        str(node)
        for node, data in generative_graph.nodes(data=True)
        if data["atomic_num"] > 0
        and any(edge.get("source_rule", 0) == GroupRule.EXCLUSION for site in _site_nodes(generative_graph, node) for _u, _v, edge in generative_graph.out_edges(site, data=True))
    }
    counts = []
    for chain in chains:
        for node, data in chain.nodes(data=True):
            if data["origin_idx"] in grouped:
                kinds = [_bond_channel(generative_graph, data["origin_idx"], chain.nodes[neighbour]["origin_idx"]) for neighbour in chain.neighbors(node)]
                counts.append((kinds.count("typed"), kinds.count("plain")))
    return counts


def test_exclusion_sites_follow_the_channel_rule_in_the_output():
    # Output-level form of the rule on the canonical string (its grouped atoms' static
    # neighbours differ from every growth target, so bonds classify unambiguously):
    # one typed bond and nothing else on the atom, or two plain bonds; both occur;
    # never one typed beside a plain one, never two typed.
    graph_creator = _graph_creator(EXCLUSION_TEXT)
    generative_graph = _generative_graph(graph_creator)
    with warnings.catch_warnings():  # one parse: the chains' origin ids must be this graph's node ids
        warnings.simplefilter("ignore")
        chains = graph_creator.get_ensemble_creator().create_ensemble(24, output_format="mol_graph", seed=7)
    counts = _grouped_site_bond_counts(generative_graph, chains)
    assert counts
    assert all(typed <= 1 and not (typed == 1 and plain) for typed, plain in counts)
    assert {(1, 0), (0, 2)} <= set(counts)


def test_exclusion_generation_is_seed_reproducible():
    assert list(_sample(EXCLUSION_TEXT, 6, seed=11, output_format="smiles")) == list(_sample(EXCLUSION_TEXT, 6, seed=11, output_format="smiles"))


@pytest.mark.parametrize("text", [SPLIT_UNIT_TEXT, JOINED_OBJECTS_SPLIT_UNIT_TEXT, TWO_LEVEL_SPLIT_UNIT_TEXT, OBJECT_ENDS_UNIT_TEXT])
def test_sites_around_a_nested_object_form_one_group(text):
    # The text after the object is built by the object's forced exit, as a fragment of its own:
    # it takes the tag of the unit that entered the object, so the typed cap on one of the two
    # sites blocks the other and every enclosing unit ends with exactly one. A unit the object
    # ends is followed by a fresh unit, whose site is a group of its own.
    graph_creator = _graph_creator(text)
    generative_graph = _generative_graph(graph_creator)
    first_oxygen = str(min(str(node) for node, data in generative_graph.nodes(data=True) if data["atomic_num"] == 8 and data["stochastic_id_tree"][0] == 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chains = graph_creator.get_ensemble_creator().create_ensemble(12, output_format="mol_graph", seed=1)
    units = [sum(1 for _node, data in chain.nodes(data=True) if data["origin_idx"] == first_oxygen) for chain in chains]
    caps = [sum(1 for _node, data in chain.nodes(data=True) if data["atomic_num"] == 35) for chain in chains]
    assert max(units) >= 2
    assert caps == units


@pytest.mark.parametrize(
    "text, caps_per_chain",
    [(TYPED_ENTRY_TAIL_TEXT, {35: 0}), (PLAIN_ENTRY_TAIL_TEXT, {35: 0, 9: 1, 17: 0}), (TYPED_ENTRY_TAIL_WITH_PLAIN_CAP_TEXT, {35: 0, 9: 0, 17: 0})],
)
def test_a_group_consumed_before_the_object_rules_the_text_after_it(text, caps_per_chain):
    # The entry consumed the group before the object's exit placed the text after it: the kill is
    # recorded under the unit's tag and replayed on that text's sites, so a typed entry blocks the
    # member there and a plain entry leaves it its plain channel only.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chains = _graph_creator(text).get_ensemble_creator().create_ensemble(6, output_format="mol_graph", seed=2)
    for atomic_num, count in caps_per_chain.items():
        assert [sum(1 for _node, data in chain.nodes(data=True) if data["atomic_num"] == atomic_num) for chain in chains] == [count] * 6


def _instantiate(generative_graph, static_graph, tracker, source_node, rng):
    """A fresh unit containing ``source_node``, registered as its own instance."""
    tree = list(generative_graph.nodes[source_node]["stochastic_id_tree"])
    sto_atom_id, _parents = tracker.register_parent_atom_instances(tree[0], tree[1], [level for level in tree[1:] if level >= 0])
    return _ec._PartialAtomGraph(generative_graph, static_graph, source_node, tracker, sto_atom_id, rng), sto_atom_id


@pytest.mark.parametrize("mode", ["propagation_weight", "termination_weight"])
@pytest.mark.parametrize("rule, outcome", [(GroupRule.EXCLUSION, "blocked"), (GroupRule.NONE, "plain channels only")])
def test_source_site_rule_reads_the_channel_not_the_mode(mode, rule, outcome):
    # The same channel gives the same outcome whether a growth or a termination edge fired it.
    generative_graph = _generative_graph(_graph_creator(EXCLUSION_TEXT))
    static_graph = _ec.EnsembleCreator._create_static_graph(generative_graph)
    tracker, rng = _ec._StochasticObjectTracker(generative_graph), np.random.default_rng(0)
    nitrogen = next(node for node, data in generative_graph.nodes(data=True) if data["atomic_num"] == 7)
    molecule, sto_atom_id = _instantiate(generative_graph, static_graph, tracker, nitrogen, rng)
    pool = molecule._open_half_bond_map[sto_atom_id]
    consumed = next(site for site in pool if site.exclusion_groups())
    sibling = next(site for site in pool if site is not consumed and site.exclusion_groups() == consumed.exclusion_groups())
    edges, targets, _molar = consumed.get_mode_bonds(mode)
    index = next(i for i, edge in enumerate(edges) if edge.get("source_rule", 0) == rule)
    pool.remove(consumed)  # the sampler pops the fired site before realizing the bond
    fresh, fresh_id = _instantiate(generative_graph, static_graph, tracker, targets[index], rng)
    fresh.pop_target_open_half_bond(fresh_id, targets[index])
    molecule._consume_sites(consumed, edges[index], fresh)
    if outcome == "blocked":
        assert sibling not in pool and not sibling.has_any_bonds()
    else:
        assert sibling in pool and sibling.has_any_bonds() and not sibling.exclusion_groups()
    assert all(site.exclusion_groups() for site in pool if site is not sibling)  # the other group is untouched


@pytest.mark.parametrize("rule, outcome", [(GroupRule.EXCLUSION, "blocked"), (GroupRule.NONE, "plain channels only")])
def test_entered_site_sibling_follows_the_channel_rule(rule, outcome):
    # A site of another unit fires INTO a grouped site of a fresh unit: the fresh
    # unit's sibling follows the same rule before the unit joins the pool.
    generative_graph = _generative_graph(_graph_creator(EXCLUSION_TEXT))
    static_graph = _ec.EnsembleCreator._create_static_graph(generative_graph)
    tracker, rng = _ec._StochasticObjectTracker(generative_graph), np.random.default_rng(0)
    nitrogen = next(node for node, data in generative_graph.nodes(data=True) if data["atomic_num"] == 7)
    grouped_sites = _site_nodes(generative_graph, nitrogen)
    u, v, fired = next((u, v, data) for u, v, data in generative_graph.edges(data=True) if v in grouped_sites and data.get("propagation_weight", 0) > 0 and data.get("target_rule", 0) == rule)
    source_unit, source_id = _instantiate(generative_graph, static_graph, tracker, u, rng)
    consumed = next(site for site in source_unit._open_half_bond_map[source_id] if site.node_idx == u)
    source_unit._open_half_bond_map[source_id].remove(consumed)
    fresh, fresh_id = _instantiate(generative_graph, static_graph, tracker, v, rng)
    fresh.pop_target_open_half_bond(fresh_id, v)
    source_unit._consume_sites(consumed, fired, fresh)
    remaining = sorted(sorted(site.exclusion_groups()) for site in fresh._open_half_bond_map[fresh_id])
    if outcome == "blocked":
        assert remaining == [[2], [2]]  # the entered group's sibling is gone, the other group is intact
    else:
        assert remaining == [[], [2], [2]]  # the sibling stays, without its exclusion-typed channel
        assert all(site.has_any_bonds() for site in fresh._open_half_bond_map[fresh_id])


def test_entry_only_site_keeps_its_group_membership():
    # The entered site has no outgoing edge, so it never joins the pool; popping it as the
    # target still hands the rule its channel: the group's sibling is blocked, the plain site stays.
    generative_graph = _generative_graph(_graph_creator(ENTRY_ONLY_SITE_TEXT))
    static_graph = _ec.EnsembleCreator._create_static_graph(generative_graph)
    tracker, rng = _ec._StochasticObjectTracker(generative_graph), np.random.default_rng(0)
    u, v, fired = next((u, v, data) for u, v, data in generative_graph.edges(data=True) if data.get("target_rule", 0) == GroupRule.EXCLUSION)
    source_unit, source_id = _instantiate(generative_graph, static_graph, tracker, u, rng)
    consumed = next(site for site in source_unit._open_half_bond_map[source_id] if site.node_idx == u)
    source_unit._open_half_bond_map[source_id].remove(consumed)
    fresh, fresh_id = _instantiate(generative_graph, static_graph, tracker, v, rng)
    assert all(site.node_idx != v for site in fresh._open_half_bond_map[fresh_id])
    fresh.pop_target_open_half_bond(fresh_id, v)
    assert fresh._consumed_half_bond is not None and fresh._consumed_half_bond.exclusion_groups() == {1}
    source_unit._consume_sites(consumed, fired, fresh)
    assert [sorted(site.exclusion_groups()) for site in fresh._open_half_bond_map[fresh_id]] == [[]]


def test_entry_through_the_typed_channel_counts_as_a_typed_consumption(monkeypatch):
    # Every chain of this string starts by entering the typed entry-only site: the timeline
    # records that entry as the unit's one grouped bond, and the sibling's typed channel (the
    # only route to the second terminator) never fires afterwards.
    log = _timeline(monkeypatch)
    assert len(_sample(ENTRY_ONLY_SITE_TEXT, 6)) == 6
    chains = _surviving_bonds_per_chain(log)
    assert len(chains) == 6
    for bonds in chains:
        grouped = [(side, site) for _kind, _tag, source, target in bonds for side, site in (("source", source), ("target", target)) if site is not None and site[1]]
        assert len(grouped) == 1 and grouped[0][0] == "target" and grouped[0][1][1:4] == (frozenset({1}), 1, GroupRule.EXCLUSION)
    assert sum(_rule_violations(bonds)[0] for bonds in chains) == 0


# -- Termination-mass estimate under the EXCLUSION rule ----------------------------------


@pytest.mark.parametrize(
    "members, expected",
    [
        # two typed-only siblings: whichever fires first blocks the other, one cap is realized
        ([(1.0, frozenset({1}), [(1, GroupRule.EXCLUSION, 1.0, 80.0)])] * 2, 80.0),
        # the first to fire is drawn by site weight
        ([(3.0, frozenset({1}), [(1, GroupRule.EXCLUSION, 1.0, 80.0)]), (1.0, frozenset({1}), [(1, GroupRule.EXCLUSION, 1.0, 10.0)])], 0.75 * 80.0 + 0.25 * 10.0),
        # dual channels: a typed first draw ends the group, a plain one leaves the sibling its plain cap
        ([(1.0, frozenset({1}), [(-1, GroupRule.NONE, 0.5, 79.0), (1, GroupRule.EXCLUSION, 0.5, 126.0)])] * 2, 0.5 * 126.0 + 0.5 * (79.0 + 79.0)),
        # two groups of one instance never reach each other
        ([(1.0, frozenset({1}), [(1, GroupRule.EXCLUSION, 1.0, 80.0)]), (1.0, frozenset({2}), [(2, GroupRule.EXCLUSION, 1.0, 80.0)])], 160.0),
        # a member of both groups: its typed cap blocks group 1 and kills group 2's typed channel on the
        # third site, which then takes its plain cap; by hand (100 + 130 + 130) / 3 over the three first movers
        (
            [
                (1.0, frozenset({1, 2}), [(1, GroupRule.EXCLUSION, 1.0, 80.0)]),
                (1.0, frozenset({1}), [(1, GroupRule.EXCLUSION, 1.0, 80.0)]),
                (1.0, frozenset({2}), [(2, GroupRule.EXCLUSION, 0.5, 80.0), (-1, GroupRule.NONE, 0.5, 20.0)]),
            ],
            120.0,
        ),
        # Ladder siblings: whichever fires takes the other as a rung, the channel's mass holding both ends already
        ([(1.0, frozenset({1}), [(1, GroupRule.LADDER, 1.0, 44.0)])] * 2, 44.0),
        # A rung-welded member also holding exclusion group 2 takes that group with it: the third site's typed channel
        # dies and it caps plain; by hand (100 + 100 + 130) / 3 over the three first movers
        (
            [
                (1.0, frozenset({1}), [(1, GroupRule.LADDER, 1.0, 80.0)]),
                (1.0, frozenset({1, 2}), [(1, GroupRule.LADDER, 1.0, 80.0)]),
                (1.0, frozenset({2}), [(2, GroupRule.EXCLUSION, 0.5, 80.0), (-1, GroupRule.NONE, 0.5, 20.0)]),
            ],
            110.0,
        ),
    ],
)
def test_cluster_cap_estimate_follows_the_firing_sequence(members, expected):
    assert _ec._PartialAtomGraph._expected_cluster_cap_mw(members) == pytest.approx(expected)


def test_independent_cluster_members_are_priced_apart():
    # Ten pairs of typed-plus-plain siblings in one unit instance never reach each other: the estimate
    # is ten times the pair's, and the recursion prices the pairs apart instead of enumerating their
    # 4**10 joint states (over a minute before the split).
    pair = [(1.0, frozenset({1}), [(1, GroupRule.EXCLUSION, 0.5, 80.0), (-1, GroupRule.NONE, 0.5, 19.0)])] * 2
    members = [(1.0, frozenset({group}), [(group, GroupRule.EXCLUSION, 0.5, 80.0), (-1, GroupRule.NONE, 0.5, 19.0)]) for group in range(1, 11) for _ in range(2)]
    started = time.perf_counter()
    assert _ec._PartialAtomGraph._expected_cluster_cap_mw(members) == pytest.approx(10 * _ec._PartialAtomGraph._expected_cluster_cap_mw(pair))
    assert time.perf_counter() - started < 5.0


def _fresh_unit(text):
    """One freshly instantiated repeat unit of ``text``, with its instance id, static graph and rng."""
    generative_graph = _generative_graph(_graph_creator(text))
    static_graph = _ec.EnsembleCreator._create_static_graph(generative_graph)
    tracker, rng = _ec._StochasticObjectTracker(generative_graph), np.random.default_rng(0)
    carbon = next(node for node, data in generative_graph.nodes(data=True) if data["atomic_num"] == 6)
    unit, sto_atom_id = _instantiate(generative_graph, static_graph, tracker, carbon, rng)
    return unit, sto_atom_id, static_graph, rng


def _unit_termination_estimate(text):
    """Termination-mass estimate of one freshly instantiated repeat unit of ``text``."""
    unit, sto_atom_id, static_graph, rng = _fresh_unit(text)
    return unit.get_average_termination_mw(sto_atom_id, static_graph, rng)


def _realized_cap_delta(text):
    """Mass one freshly instantiated repeat unit of ``text`` gains when it terminates."""
    unit, sto_atom_id, _static_graph, rng = _fresh_unit(text)
    before = Descriptors.MolWt(g2rins.mol_graph_to_rdkit_mol(unit.atom_graph))
    unit.terminate_graph(sto_atom_id, rng)
    return Descriptors.MolWt(g2rins.mol_graph_to_rdkit_mol(unit.atom_graph)) - before


# One repeat unit carrying ten exclusion groups, each a pair of sites with a typed and a plain cap channel.
TEN_GROUP_UNIT_TEXT = (
    "{[] [<]C([>1[]1,>2])([>1[]1,>2])C([>1[]2,>2])([>1[]2,>2])C([>1[]3,>2])([>1[]3,>2])C([>1[]4,>2])([>1[]4,>2])C([>1[]5,>2])([>1[]5,>2])"
    "C([>1[]6,>2])([>1[]6,>2])C([>1[]7,>2])([>1[]7,>2])C([>1[]8,>2])([>1[]8,>2])C([>1[]9,>2])([>1[]9,>2])C([>1[]10,>2])([>1[]10,>2])[>]; O[>]; [<][H], [<1]Br, [<2]F []}|poisson(3000)|"
)
ONE_GROUP_UNIT_TEXT = "{[] [<]C([>1[]1,>2])([>1[]1,>2])[>]; O[>]; [<][H], [<1]Br, [<2]F []}|poisson(3000)|"


def test_ten_independent_groups_price_as_ten_pairs():
    ten, one = _unit_termination_estimate(TEN_GROUP_UNIT_TEXT), _unit_termination_estimate(ONE_GROUP_UNIT_TEXT)
    assert one > 0 and ten == pytest.approx(10 * one)


def test_typed_sibling_caps_are_priced_as_one_cap():
    # Whichever typed sibling fires first blocks the other, so the unit with two typed cap sites
    # budgets the same termination mass as the unit with one.
    typed, single = _unit_termination_estimate(TYPED_SIBLING_CAPS_TEXT), _unit_termination_estimate(SINGLE_CAP_SITE_TEXT)
    assert typed > 0 and typed == pytest.approx(single)


def test_typed_sibling_caps_keep_the_number_average_on_target():
    # A margin that budgeted both sibling caps parked growth early; both strings land on the target.
    for text in (TYPED_SIBLING_CAPS_TEXT, SINGLE_CAP_SITE_TEXT):
        chains = _sample(text, 40, seed=1)
        number_average = np.mean([Descriptors.MolWt(g2rins.mol_graph_to_rdkit_mol(chain)) for chain in chains])
        assert abs(number_average - 600) / 600 < 0.1, f"Mn {number_average:.0f} vs target 600 for {text}"


# -- Phase 3: generation under the ALL rule ---------------------------------------------

# Two all-typed initiator sites beside a plain one: the plain site waits, at the opening move too.
MIXED_ALL_TEXT = "{[] [<]CCO[>]; C(O[>[allh]1])(CO[>[allh]1])(CO[>]); [H][<] []}|poisson(300)|"
# Target below the initiator plus one unit per all-typed site: every chain must overshoot.
SMALL_TARGET_ALL_TEXT = "{[] [<]CCO[>]; C(O[>[allh]1])(CO[>[allh]1])(CO[>[allh]1]); [H][<] []}|poisson(120)|"
# A nested initiator whose all-group overshoots its own target, beside a plain one; the second repeat unit's exit
# only matches a terminator, so the enclosing object sometimes ends below target and that attempt is discarded.
NESTED_OVERSHOOT_DISCARD_TEXT = "{[] [<]CC[>]|0.95|, [<]CC[>1]|0.05|; {[] [<]CCO[>]; C(O[>[allh]])CO[>[allh]]; [<][H] [<]}|uniform(80,80)|[>], C[>]; [<][H], [<1][H] []}|uniform(300,300)|"


def _all_rule_violations(bonds):
    """Bonds fired from a plain source channel before the chain's last bond fired from an all-typed one: the
    all rule places every all-typed site before any other growth, so none may occur."""
    rules = [source[3] for _kind, _fresh_tag, source, _target in bonds if source is not None]
    last_all = max((i for i, rule in enumerate(rules) if rule in ALL_RULES), default=-1)
    return sum(1 for rule in rules[:last_all] if rule not in ALL_RULES), last_all >= 0


@pytest.mark.parametrize("text, n_chains", [(ALL_TEXT, 12), (ALL_STAR_TEXT, 4), (MIXED_ALL_TEXT, 12), (SMALL_TARGET_ALL_TEXT, 12)])
def test_all_typed_sites_fire_before_any_other_growth(monkeypatch, text, n_chains):
    log = _timeline(monkeypatch)
    chains = _sample(text, n_chains)
    assert chains is not None and len(chains) == n_chains
    results = [_all_rule_violations(bonds) for bonds in _surviving_bonds_per_chain(log)]
    assert len(results) == n_chains
    assert sum(violations for violations, _fired in results) == 0
    assert all(fired for _violations, fired in results)


def _all_typed_oxygen_origins(generative_graph):
    return {
        str(node)
        for node, data in generative_graph.nodes(data=True)
        if data["atomic_num"] == 8 and any(edge.get("source_rule", 0) in ALL_RULES for site in _site_nodes(generative_graph, node) for _u, _v, edge in generative_graph.out_edges(site, data=True))
    }


def _chains_with_every_all_typed_site_used(generative_graph, chains):
    origins = _all_typed_oxygen_origins(generative_graph)
    assert len(origins) == 3
    return sum(1 for chain in chains if sum(1 for node, data in chain.nodes(data=True) if data["origin_idx"] in origins and chain.degree(node) == 2) == 3)


@pytest.mark.parametrize("text, n_chains", [(ALL_TEXT, 8), (ALL_STAR_TEXT, 3)])
def test_every_all_typed_site_is_used_in_the_output(text, n_chains):
    graph_creator = _graph_creator(text)
    generative_graph = _generative_graph(graph_creator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chains = graph_creator.get_ensemble_creator().create_ensemble(n_chains, output_format="mol_graph", seed=7)
    assert _chains_with_every_all_typed_site_used(generative_graph, chains) == n_chains


@pytest.mark.parametrize("termination_flag", [None, 1])
def test_all_rule_overshoots_a_small_target_and_warns_once(termination_flag):
    graph_creator = _graph_creator(SMALL_TARGET_ALL_TEXT)
    generative_graph = _generative_graph(graph_creator)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chains = graph_creator.get_ensemble_creator().create_ensemble(6, output_format="mol_graph", termination_flag=termination_flag, seed=3)
    assert _chains_with_every_all_typed_site_used(generative_graph, chains) == 6
    reported = [record.message for record in caught if issubclass(record.category, AllGroupOvershoot)]
    assert len(reported) == 1
    assert (reported[0].chain_count, reported[0].chain_total) == (6, 6)


def test_all_rule_warning_is_aggregated_on_the_parallel_path():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parallel = _graph_creator(SMALL_TARGET_ALL_TEXT).get_ensemble_creator().create_ensemble(4, output_format="smiles", parallel=True, n_workers=2, seed=3)
    serial = _sample(SMALL_TARGET_ALL_TEXT, 4, seed=3, output_format="smiles")
    assert list(parallel) == list(serial)
    reported = [record.message for record in caught if issubclass(record.category, AllGroupOvershoot)]
    assert len(reported) == 1 and (reported[0].chain_count, reported[0].chain_total) == (4, 4)


def test_all_rule_overshoot_counts_returned_chains_only(monkeypatch):
    attempts = []
    attempt_chain = _ec._attempt_chain

    def recording(*args):
        result = attempt_chain(*args)
        attempts.append((result[0] is not None, any(issubclass(caught.category, AllGroupOvershoot) for caught in result[3])))
        return result

    monkeypatch.setattr(_ec, "_attempt_chain", recording)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        serial = _graph_creator(NESTED_OVERSHOOT_DISCARD_TEXT).get_ensemble_creator().create_ensemble(8, output_format="smiles", seed=1)
    # At least one discarded attempt overshot: that is what the count must leave out.
    assert sum(1 for accepted, overshoot in attempts if not accepted and overshoot) >= 1
    expected = sum(1 for accepted, overshoot in attempts if accepted and overshoot)
    reported = [record.message for record in caught if issubclass(record.category, AllGroupOvershoot)]
    assert [(message.chain_count, message.chain_total) for message in reported] == [(expected, 8)]
    # The workers run the unpatched attempt helper; same seed, same chains, same count.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parallel = _graph_creator(NESTED_OVERSHOOT_DISCARD_TEXT).get_ensemble_creator().create_ensemble(8, output_format="smiles", parallel=True, n_workers=2, seed=1)
    assert list(parallel) == list(serial)
    reported = [record.message for record in caught if issubclass(record.category, AllGroupOvershoot)]
    assert [(message.chain_count, message.chain_total) for message in reported] == [(expected, 8)]


def test_all_generation_is_seed_reproducible():
    assert list(_sample(ALL_TEXT, 4, seed=5, output_format="smiles")) == list(_sample(ALL_TEXT, 4, seed=5, output_format="smiles"))


# -- Phase 4: generation under the LADDER rule ------------------------------------------

# Regiospecific inner classes: every rung pairs class 1 with class 1 and class 2 with class 2.
REGIO_LADDER_TEXT = "{[] [<[<1]2]OC(O[<[<2]2])CC(O[>[>1]1])O[>[>2]1]; C(O[>[>1]1])O[>[>2]1]; [<][H] []}|poisson(1000)|"
# Three members per class on separate atoms: after the first bond two members face two free partners, so the
# completion draws (three sites on one atom would give parallel bonds a simple graph cannot hold).
THREE_RUNG_TEXT = "{[] [<[<]2]OC(O[<[<]2])(O[<[<]2])CC(O[>[>]1])(O[>[>]1])O[>[>]1]; C(O[>[>]1])(O[>[>]1])O[>[>]1]; [<][H] []}|poisson(1000)|"
COMPLETION_DRAW = "ladder completion partner selection"
# Ladder sites whose partners all sit on the initiator only receive bonds: no outgoing edge, never in the pool.
INCOMING_ONLY_LADDER_TEXT = "{[] [<[<]1]OC(O[<[<]1])CC[>], [<]CC[>]; C(O[>[>]2])O[>[>]2]; [<][H] []}|uniform(400,400)|"
# The second exit site may only pair with the first entry site; a first exit that fires and takes it leaves no partner.
RESTRICTED_PARTNER_TEXT = "{[] [<[<]2]OC(O[<[<]2])CC(O[>[>]1|1 1 0 0|])O[>[>]1|1 0 0 0|]; C(O[>[>]1])O[>[>]1]; [<][H] []}|uniform(500,500)|"
# A two-site ladder-typed cap: the first member to terminate draws it, the other welds the rung to its second site.
LADDER_CAP_TEXT = "{[] [<[<]2]OC(O[<[<]2])CC(O[>[>]1])O[>[>]1]; C(O[>[>]1])O[>[>]1]; [<[<]]OCO[<[<]] []}|poisson(1000)|"
# A weight list that leaves the `>2` member no channel while its sibling fires: no junction fired from that side could close.
ZERO_WEIGHT_MEMBER_TEXT = (
    "{[] [<[<1]2|0 0 1 1|]OC(O[<[<2]2|0 0 1 1|])CC(O[>[>1]1])O[>[>2]1|0 0 0 0|]; C(O[>[>1]1])O[>[>2]1], C(O[<[<1]2|0 0 1 1|])O[<[<2]2|0 0 1 1|]; [<1,<2,>1,>2][H] []}|uniform(500,500)|"
)
# Both members without a channel: the group never fires toward the repeat unit, so nothing is broken.
INERT_GROUP_TEXT = (
    "{[] [<[<1]2|0 0 1 1|]OC(O[<[<2]2|0 0 1 1|])CC(O[>[>1]1|0 0 0 0|])O[>[>2]1|0 0 0 0|]; C(O[>[>1]1])O[>[>2]1], C(O[<[<1]2|0 0 1 1|])O[<[<2]2|0 0 1 1|]; [<1,<2,>1,>2][H] []}|uniform(500,500)|"
)
# An initiator member nothing targets holds no edge at all once its weight list is zero: only its declaration tells it is a member.
ZERO_WEIGHT_INITIATOR_MEMBER_TEXT = "{[] [<[<]1]OC(O[<[<]1])CC[>], [<]CC[>]; C(O[>[>]2])O[>[>]2|0 0|]; [<][H] []}|uniform(300,300)|"
# A scalar zero weight keeps the member's edges (a weight list zeroes them) but takes the site out of every draw: no channel either.
SCALAR_ZERO_MEMBER_TEXT = "{[] [<[<1]1]OC(O[<[<2]1])CC[>], [<]CC[>]; C(O[>[>1]2])O[>[>2]2|0|]; [<][H] []}|uniform(300,300)|"


def _unit_anchors(ensemble):
    """One template node per unit, as the chains carry it, to count unit instances in a chain."""
    return {str(next(iter(unit["subgraph"].nodes))) for unit in ensemble.units.values()}


def _ensemble(text, n_chains, seed=5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _graph_creator(text).get_ensemble_creator().create_ensemble(n_chains, output_format="mol_graph", ensemble_info=True, seed=seed)


@pytest.mark.parametrize(
    "text, n_chains, rungs_per_junction, mass_window",
    [(LADDER_TEXT, 6, 2, (800, 1200)), (REGIO_LADDER_TEXT, 6, 2, (800, 1200)), (THREE_RUNG_TEXT, 4, 3, (800, 1200))],
)
def test_every_ladder_junction_closes_with_its_rungs(text, n_chains, rungs_per_junction, mass_window):
    ensemble = _ensemble(text, n_chains)
    anchors = _unit_anchors(ensemble)
    assert len(ensemble.chains) == n_chains
    for chain in ensemble.chains:
        undirected = nx.Graph(chain)
        units = sum(1 for _node, data in chain.nodes(data=True) if data["origin_idx"] in anchors)
        # Each junction is a cycle per rung beyond the first; a single-bonded junction closes none.
        cycle_rank = undirected.number_of_edges() - undirected.number_of_nodes() + nx.number_connected_components(undirected)
        assert units > 1 and cycle_rank == (units - 1) * (rungs_per_junction - 1)
        assert mass_window[0] <= Descriptors.MolWt(g2rins.mol_graph_to_rdkit_mol(chain)) <= mass_window[1]


def _junction_records(ensemble, left, right):
    return {tuple(record["labels"]): record["count"] for record in ensemble.bonds if record["labels"][0].startswith(left) and record["labels"][1].startswith(right)}


@pytest.mark.parametrize("text, pairings", [(LADDER_TEXT, 4), (REGIO_LADDER_TEXT, 2)])
def test_rungs_are_bond_records_of_their_own(text, pairings):
    # Two-member classes: the first bond picks a pair and the rung takes the other; a degenerate class
    # reaches every pairing over the chains, regiospecific classes only the declared ones.
    ensemble = _ensemble(text, 12)
    between_units = _junction_records(ensemble, "R0", "R0")
    assert len(between_units) == pairings
    assert sum(between_units.values()) == 2 * (ensemble.units["R0"]["count"] - 12)
    assert sum(_junction_records(ensemble, "I0", "R0").values()) == 2 * 12


def test_completion_draws_only_when_a_member_has_a_choice(monkeypatch):
    contexts = []
    choose = _ec._StochasticObjectTracker.choose

    def recording(self, rng, n_candidates, weights, context):
        contexts.append(context)
        return choose(self, rng, n_candidates, weights, context)

    monkeypatch.setattr(_ec._StochasticObjectTracker, "choose", recording)
    assert _sample(LADDER_TEXT, 4) is not None and COMPLETION_DRAW not in contexts
    contexts.clear()
    assert _sample(THREE_RUNG_TEXT, 4) is not None and COMPLETION_DRAW in contexts


def _ladder_strip_violations(bonds):
    """Ladder-channel fires of a group after a member of it was consumed through another channel, and the rungs."""
    stripped, violations, rungs = set(), 0, 0
    for kind, _fresh_tag, source, target in bonds:
        for side in (source, target):
            if side is None:
                continue
            instance, _exclusion, group, rule, ladder = side
            if rule == GroupRule.LADDER:
                violations += (instance, group) in stripped
                stripped.update((instance, held) for held in ladder if held != group)
            else:
                stripped.update((instance, held) for held in ladder)
        rungs += kind == "rung"
    return violations, rungs


def test_plain_consumption_strips_the_ladder_group(monkeypatch):
    log = _timeline(monkeypatch)
    chains = _sample(DUAL_CHANNEL_TEXT, 8)
    assert chains is not None and len(chains) == 8
    per_chain = _surviving_bonds_per_chain(log)
    results = [_ladder_strip_violations(bonds) for bonds in per_chain]
    assert sum(violations for violations, _rungs in results) == 0
    assert sum(rungs for _violations, rungs in results) > 0  # junctions where the ladder channel won
    assert any(source is not None and source[3] == GroupRule.NONE for bonds in per_chain for _kind, _tag, source, _target in bonds)  # plain growth
    assert any(target is None for bonds in per_chain for _kind, _tag, _source, target in bonds)  # caps


def test_ladder_generation_is_seed_reproducible():
    assert list(_sample(LADDER_TEXT, 4, seed=5, output_format="smiles")) == list(_sample(LADDER_TEXT, 4, seed=5, output_format="smiles"))


def _cycle_rank(chain):
    undirected = nx.Graph(chain)
    return undirected.number_of_edges() - undirected.number_of_nodes() + nx.number_connected_components(undirected)


def test_incoming_only_ladder_sites_close_the_junction():
    # The initiator's two-rung junction closes one ring per chain; the rest of the chain is linear.
    ensemble = _ensemble(INCOMING_ONLY_LADDER_TEXT, 6)
    assert [_cycle_rank(chain) for chain in ensemble.chains] == [1] * 6
    assert sum(_junction_records(ensemble, "I0", "L0").values()) == 2 * 6


def test_member_without_a_free_partner_discards_the_attempt(monkeypatch):
    causes = []
    attempt_chain = _ec._attempt_chain

    def recording(*args):
        result = attempt_chain(*args)
        causes.append(result[2])
        return result

    monkeypatch.setattr(_ec, "_attempt_chain", recording)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ensemble = _graph_creator(RESTRICTED_PARTNER_TEXT).get_ensemble_creator().create_ensemble(6, output_format="mol_graph", ensemble_info=True, seed=7)
    rejected = [cause for cause in causes if isinstance(cause, LadderMemberWithoutPartner)]
    assert rejected and all(cause.group == 1 for cause in rejected)
    reported = [record.message for record in caught if issubclass(record.category, DiscardedSamplingPaths)]
    assert len(reported) == 1 and dict(reported[0].reasons)["LadderMemberWithoutPartner"] == len(rejected)
    # Every returned chain closes all of its junctions.
    anchors = _unit_anchors(ensemble)
    for chain in ensemble.chains:
        units = sum(1 for _node, data in chain.nodes(data=True) if data["origin_idx"] in anchors)
        assert _cycle_rank(chain) == units - 1


@pytest.mark.parametrize("text, n_chains, seed", [(LADDER_CAP_TEXT, 6, 5), (LADDER_REFERENCE_TEXT, 3, 1)])
def test_ladder_typed_cap_closes_the_last_junction(text, n_chains, seed):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ensemble = _graph_creator(text).get_ensemble_creator().create_ensemble(n_chains, output_format="mol_graph", ensemble_info=True, seed=seed)
    assert not any(issubclass(record.category, DiscardedSamplingPaths) for record in caught)
    assert len(ensemble.chains) == n_chains
    # One cap per chain, bonded twice: the chain start, every growth step and the cap each close with two rungs.
    assert ensemble.units["T0"]["count"] == n_chains
    assert sum(_junction_records(ensemble, "I0", "R0").values()) == 2 * n_chains
    assert sum(_junction_records(ensemble, "R0", "R0").values()) == 2 * (ensemble.units["R0"]["count"] - n_chains)
    assert sum(_junction_records(ensemble, "R0", "T0").values()) == 2 * n_chains
    # Every chain's cycle rank is one per junction on top of the rings its units bring.
    intrinsic = {str(next(iter(unit["subgraph"].nodes))): _cycle_rank(unit["subgraph"]) for unit in ensemble.units.values()}
    for chain in ensemble.chains:
        units = [data["origin_idx"] for _node, data in chain.nodes(data=True) if data["origin_idx"] in intrinsic]
        assert _cycle_rank(chain) == len(units) - 1 + sum(intrinsic[anchor] for anchor in units)
        assert Descriptors.MolWt(g2rins.mol_graph_to_rdkit_mol(chain)) > 0


@pytest.mark.parametrize("text", [LADDER_CAP_TEXT, LADDER_REFERENCE_TEXT])
def test_ladder_cap_is_priced_once_with_its_rungs(text):
    # The cap is the only terminator, so the expectation is the one mass termination realizes: the cap
    # attached at both sites of the pair, every end shedding its hydrogen.
    assert _unit_termination_estimate(text) == pytest.approx(_realized_cap_delta(text))


# -- Phase 5: the two flavours of the ALL rule ----------------------------------------

# The same star with soft sites: the target wins as soon as it is reached, the other sites close plain.
SMALL_TARGET_SOFT_TEXT = "{[] [<]CCO[>]; C(O[>[alls]1])(CO[>[alls]1])(CO[>[alls]1]); [H][<] []}|poisson(120)|"
# Two soft initiator sites beside a plain one: the plain site still waits for the tier.
MIXED_SOFT_TEXT = "{[] [<]CCO[>]; C(O[>[alls]1])(CO[>[alls]1])(CO[>]); [H][<] []}|poisson(300)|"
# A unit whose two soft sites point back at itself: under the hard rule it could never settle.
SOFT_DENDRIMER_TEXT = "{[] [<]CCN([>[alls]])([>[alls]]); C[>]; [<][H] []}|poisson(200)|"
HARD_DENDRIMER_TEXT = "{[] [<]CCN([>[allh]])([>[allh]]); C[>]; [<][H] []}|poisson(200)|"
# Two units whose hard sites feed each other: a two-unit cycle.
HARD_CYCLE_TEXT = "{[] [<]CC([>1[allh]])[>], [<1]CC([>[allh]])O[>1]; C[>]; [<][H], [<1][H] []}|poisson(300)|"
# A hard pendant into a plain unit, and a hard pendant into a unit whose own hard pendant leads into a plain one: both settle.
HARD_PENDANT_TEXT = "{[] [<]CC([>1[allh]1])[>], [<1]CCO[>]; C[>]; [<][H] []}|poisson(300)|"
HARD_PENDANT_CHAIN_TEXT = "{[] [<]CC([>1[allh]1])[>], [<1]CC([>2[allh]2])O[>], [<2]C(=O)C[>1]; C[>]; [<][H] []}|poisson(300)|"
# A plain-grown backbone whose every unit carries one soft pendant; the grafts are drawn at the outer level.
SOFT_BACKBONE_TEXT = "{[] [<1]CCO[>1]; {[] [<]CC([>1[alls]1])[>]; C[>]; [<][H] [<1]}|uniform(300,300)|[>1]; [<1][H] []}|uniform(600,600)|"
# An all-typed site that is only ever a bond target (the unit is entered through it alone): the rule never acts.
TARGET_ONLY_ALL_TEXT = "{[] [<[allh]]CC[>]; C[>]; [<][H] []}|poisson(300)|"
TARGET_ONLY_SOFT_CAP_TEXT = "{[] [<]CC[>]; C[>]; [<[alls]][H] []}|poisson(300)|"
# The same unit with the hard site on the growth side: it opens on every placement and feeds the unit itself.
HARD_GROWTH_TEXT = "{[] [<]CC[>[allh]]; C[>]; [<][H] []}|poisson(300)|"
# A probe of the entry-site rule, not a chemistry example: the hard channel enters the fresh unit through the site that carries it, which spends it.
SPENT_ENTRY_SITE_TEXT = "{[] [<]CC[<1,>1[allh]]; C[>]; [>][H], [>1][H] []}|uniform(40,40)|"


def test_bare_all_is_not_a_keyword():
    with pytest.raises(lark.exceptions.UnexpectedInput):
        g2rins.SimpleBondConnector.make("[>[all]]")


@pytest.mark.parametrize("text", ["{[] [<]CC([>1[alls]1])[>]; C[>]; [<][H], [<1][H] []}|poisson(100)|", "{[] [<]CC([>1[allh]1])[>]; C[>]; [<][H], [<1][H] []}|poisson(100)|"])
def test_single_member_group_is_silent_for_the_all_rules(text):
    # One all-typed site per unit is the grafting use of the rule; the exclusion twin still warns.
    _assert_no_diagnostics(text)
    with pytest.warns(SingleMemberGroup):
        g2rins.StochasticObject.make(text.replace("[alls]", "[]").replace("[allh]", "[]"))


def test_soft_all_edge_annotations():
    generative_graph = _generative_graph(_graph_creator(SMALL_TARGET_SOFT_TEXT))
    annotated = [data for _u, _v, data in generative_graph.edges(data=True) if _group_values(data) != SENTINEL]
    assert len(annotated) == 3
    assert all(_group_values(data) == (1, 4, -1, 0) and _mode(data) == "transition_weight" for data in annotated)


@pytest.mark.parametrize("text, cycle_units", [(HARD_DENDRIMER_TEXT, 1), (HARD_CYCLE_TEXT, 2), (HARD_GROWTH_TEXT, 1)])
def test_unsettled_hard_all_rule_is_refused_at_graph_build(text, cycle_units):
    graph_creator = _graph_creator(text)
    with pytest.raises(UnsettledHardAllRule) as info:
        _generative_graph(graph_creator)
    error = info.value
    # The cycle starts and ends on the same unit and passes through the offender.
    assert error.cycle[0] == error.cycle[-1] and error.unit_text in error.cycle
    assert len(set(error.cycle)) == cycle_units
    assert error.unit_text and "allh" in error.unit_text and "`alls`" in str(error)
    with pytest.raises(UnsettledHardAllRule):
        graph_creator.get_ensemble_creator()
    # The graph with bond connectors stays available, as for a bond ruled at both ends.
    assert _generative_graph(graph_creator, include_bond_connectors=True).number_of_nodes() > 0


@pytest.mark.parametrize("text", [ALL_TEXT, HARD_PENDANT_TEXT, HARD_PENDANT_CHAIN_TEXT, SOFT_DENDRIMER_TEXT, TARGET_ONLY_ALL_TEXT, SPENT_ENTRY_SITE_TEXT])
def test_settling_hard_channels_and_soft_rules_build(text):
    assert _generative_graph(_graph_creator(text)).number_of_nodes() > 0


@pytest.mark.parametrize(
    "text, bond_connector, unit, label",
    [(TARGET_ONLY_ALL_TEXT, "[<[allh]]", "[<[allh]]CC[>]", "hard all"), (TARGET_ONLY_SOFT_CAP_TEXT, "[<[alls]]", "[<[alls]][H]", "soft all")],
)
def test_target_only_all_site_warns_and_samples_as_plain(text, bond_connector, unit, label):
    graph_creator = _graph_creator(text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph_creator.get_generative_graph()
    reported = [record.message for record in caught if issubclass(record.category, AllRuleOnTargetOnlySite)]
    assert [(message.bond_connector, str(message.token), message.rule_label) for message in reported] == [(bond_connector, unit, label)]
    assert "never acts" in str(reported[0])
    # The site never opens, so its channel never fires: the string samples exactly as its plain twin.
    plain = text.replace("[allh]", "").replace("[alls]", "")
    assert list(_sample(text, 6, seed=1, output_format="smiles")) == list(_sample(plain, 6, seed=1, output_format="smiles"))


def test_hard_pendants_all_fire():
    # The object cannot stop while a hard pendant is open: every backbone unit carries its graft.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ensemble = _graph_creator(HARD_PENDANT_TEXT).get_ensemble_creator().create_ensemble(6, output_format="mol_graph", ensemble_info=True, seed=1)
    assert not any(issubclass(record.category, DiscardedSamplingPaths) for record in caught)
    backbone = next(key for key, unit in ensemble.units.items() if "allh" in unit["g2rins"])
    graft = next(key for key, unit in ensemble.units.items() if unit["g2rins"].replace(" ", "") == "[<1]CCO[>]")
    # Bonds into the graft unit's entry site (its exit site continues the backbone and does not count).
    grafts = sum(record["count"] for record in ensemble.bonds if f"{graft}.1" in record["labels"] and any(label.startswith(backbone) for label in record["labels"]))
    assert grafts == ensemble.units[backbone]["count"] > 0


@pytest.mark.parametrize("text, n_chains", [(MIXED_SOFT_TEXT, 12), (SMALL_TARGET_SOFT_TEXT, 12)])
def test_soft_all_typed_sites_fire_before_any_other_growth(monkeypatch, text, n_chains):
    log = _timeline(monkeypatch)
    chains = _sample(text, n_chains)
    assert chains is not None and len(chains) == n_chains
    results = [_all_rule_violations(bonds) for bonds in _surviving_bonds_per_chain(log)]
    assert len(results) == n_chains
    assert sum(violations for violations, _fired in results) == 0
    assert all(fired for _violations, fired in results)


def test_soft_all_yields_to_the_target():
    # The hard twin of this string overshoots every chain and warns; the soft one stops at the first arm.
    graph_creator = _graph_creator(SMALL_TARGET_SOFT_TEXT)
    generative_graph = _generative_graph(graph_creator)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chains = graph_creator.get_ensemble_creator().create_ensemble(12, output_format="mol_graph", seed=3)
    assert not any(issubclass(record.category, AllGroupOvershoot) for record in caught)
    origins = _all_typed_oxygen_origins(generative_graph)
    used = [sum(1 for node, data in chain.nodes(data=True) if data["origin_idx"] in origins and chain.degree(node) == 2) for chain in chains]
    assert used == [1] * 12
    # Initiator plus one repeat unit; the two unfired sites closed with hydrogen.
    assert all(abs(Descriptors.MolWt(g2rins.mol_graph_to_rdkit_mol(chain)) - 136.15) < 0.5 for chain in chains)


def _nitrogen_generations(chain):
    """Units per all-generation of a CCN dendrimer chain rooted at the initiator carbon."""
    heavy = chain.subgraph([node for node, data in chain.nodes(data=True) if data["atomic_num"] > 1])
    (root,) = [node for node, data in heavy.nodes(data=True) if data["atomic_num"] == 6 and heavy.degree(node) == 1]
    distance = nx.single_source_shortest_path_length(heavy, root)
    levels = {}
    for node, data in heavy.nodes(data=True):
        if data["atomic_num"] == 7:
            levels[(distance[node] - 3) // 3] = levels.get((distance[node] - 3) // 3, 0) + 1
    return [levels.get(generation, 0) for generation in range(max(levels) + 1)]


def test_soft_dendrimer_fills_generation_by_generation():
    # The tier draws the lowest all-generation first: every generation but the last is complete.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chains = _graph_creator(SOFT_DENDRIMER_TEXT).get_ensemble_creator().create_ensemble(20, output_format="mol_graph", seed=1)
    assert not any(issubclass(record.category, (AllGroupOvershoot, DiscardedSamplingPaths)) for record in caught)
    profiles = [_nitrogen_generations(chain) for chain in chains]
    assert all(profile[:-1] == [2**generation for generation in range(len(profile) - 1)] for profile in profiles)
    assert any(len(profile) >= 3 for profile in profiles)


def test_soft_pendants_graft_across_the_backbone():
    # Plain-grown backbone units share all-generation 0, so the pendants are one weighted draw: the grafts
    # are not placed from the backbone start on, and the budget runs out before the tier, so no graft grows.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph_creator = _graph_creator(SOFT_BACKBONE_TEXT)
        generative_graph = _generative_graph(graph_creator)
        ensemble = graph_creator.get_ensemble_creator().create_ensemble(10, output_format="mol_graph", ensemble_info=True, seed=1)
    assert not any(issubclass(record.category, (AllGroupOvershoot, DiscardedSamplingPaths)) for record in caught)
    backbone = next(key for key, unit in ensemble.units.items() if "alls" in unit["g2rins"])
    graft = next(key for key, unit in ensemble.units.items() if unit["g2rins"].replace(" ", "") == "[<1]CCO[>1]")
    start = next(key for key, unit in ensemble.units.items() if unit["g2rins"].replace(" ", "") == "C[>]")
    graft_bonds = [record for record in ensemble.bonds if {label.split(".")[0] for label in record["labels"]} == {backbone, graft}]
    assert len(graft_bonds) == 1 and not any({label.split(".")[0] for label in record["labels"]} == {graft} for record in ensemble.bonds)
    site = next(node for node, label in zip(graft_bonds[0]["nodes"], graft_bonds[0]["labels"]) if label.startswith(backbone))
    if generative_graph.nodes[site]["atomic_num"] == 0:
        # The pendant site is a connector placeholder split off its carbon; the chains hold the carbon.
        (site,) = {
            u if v == site else v
            for u, v, data in list(generative_graph.in_edges(site, data=True)) + list(generative_graph.out_edges(site, data=True))
            if data.get("static") and generative_graph.nodes[u if v == site else v]["atomic_num"] > 0
        }
    pendant_carbon = str(site)
    graft_origins = {str(node) for node in ensemble.units[graft]["subgraph"].nodes}
    root_origins = {str(node) for node in ensemble.units[start]["subgraph"].nodes}
    patterns = []
    for chain in ensemble.chains:
        heavy = chain.subgraph([node for node, data in chain.nodes(data=True) if data["atomic_num"] > 1])
        root = next(node for node, data in heavy.nodes(data=True) if data["origin_idx"] in root_origins)
        distance = nx.single_source_shortest_path_length(heavy, root)
        carbons = sorted((node for node, data in heavy.nodes(data=True) if data["origin_idx"] == pendant_carbon), key=distance.get)
        patterns.append([any(heavy.nodes[other]["origin_idx"] in graft_origins for other in heavy.neighbors(node)) for node in carbons])
    assert all(len(pattern) >= 8 and 0 < sum(pattern) < len(pattern) for pattern in patterns), patterns
    assert any(pattern.index(False) < len(pattern) - 1 - pattern[::-1].index(True) for pattern in patterns), patterns


def _unit_nodes(generative_graph, text):
    """The nodes of the unit parsed from ``text``, through the graph's provenance record."""
    unit_id = next(unit_id for unit_id, unit_text in generative_graph.graph["unit_g2rins"].items() if unit_text == text)
    return generative_graph.graph["unit_node_ids"][unit_id]


def test_hard_all_is_validated_for_the_sampled_source():
    # A chain started in the unit opens its `<` site, which the initiator-started chains never open.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = _graph_creator(TARGET_ONLY_ALL_TEXT).get_ensemble_creator()
    source = _unit_nodes(ensemble_creator.generative_graph, "[<[allh]]CC[>]")[0]
    with pytest.raises(UnsettledHardAllRule) as info:
        ensemble_creator.sample_mol_graph(source=source, use_repeat_units_as_source=True, rng=np.random.default_rng(1))
    assert (info.value.start_text, info.value.unit_text, info.value.cycle) == ("[<[allh]]CC[>]", "[<[allh]]CC[>]", ["[<[allh]]CC[>]"] * 2)
    assert "A chain started in the unit [<[allh]]CC[>]" in str(info.value)
    # A chain started in a unit whose hard channels settle samples as before.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = _graph_creator(HARD_PENDANT_TEXT).get_ensemble_creator()
        source = _unit_nodes(ensemble_creator.generative_graph, "[<1]CCO[>]")[0]
        chain = ensemble_creator.sample_mol_graph(source=source, use_repeat_units_as_source=True, rng=np.random.default_rng(1))
    assert chain.number_of_nodes() > 0


@pytest.mark.parametrize(
    "text, bond_connector, unit_start, partner_start",
    [
        (ZERO_WEIGHT_MEMBER_TEXT, "[>[>2]1|0.0 0.0 0.0 0.0|]", "[<[<1]2|", "[<[<1]2|"),
        (ZERO_WEIGHT_INITIATOR_MEMBER_TEXT, "[>[>]2|0.0 0.0|]", "C(O[>[>]2])", "[<[<]1]OC("),
        (SCALAR_ZERO_MEMBER_TEXT, "[>[>2]2|0.0|]", "C(O[>[>1]2])", "[<[<1]1]OC("),
    ],
)
def test_ladder_member_without_channel_is_refused_at_graph_build(text, bond_connector, unit_start, partner_start):
    graph_creator = _graph_creator(text)
    with pytest.raises(LadderMemberWithoutChannel) as info:
        _generative_graph(graph_creator)
    error = info.value
    assert error.bond_connector == bond_connector
    assert error.unit_text.startswith(unit_start) and error.partner_text.startswith(partner_start)
    assert "same group pair" in str(error)
    with pytest.raises(LadderMemberWithoutChannel):
        graph_creator.get_ensemble_creator()
    # The graph with bond connectors stays available, as for a bond ruled at both ends.
    assert _generative_graph(graph_creator, include_bond_connectors=True).number_of_nodes() > 0
    # A group none of whose members fires toward the unit is inert, not broken, and a restricted partner list is still a channel.
    for text in (INERT_GROUP_TEXT, RESTRICTED_PARTNER_TEXT):
        assert _generative_graph(_graph_creator(text)).number_of_nodes() > 0
