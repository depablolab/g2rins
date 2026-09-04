# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for conditional connectivity (group rules).

Phase 0 -- the per-symbol group suffix: parsing and round-tripping of the three
rules (ladder / exclusion / all), the compatibility matrix (ladder rigidity) and
the stochastic-object validation set. Phase 1 -- the generative-graph encoding:
every edge carries the four group-rule attributes, one edge per distinct pair of
compatible symbols, terminal-descriptor edges annotated on the unit side, and
the temporary generation gate in EnsembleCreator.
"""

import warnings

import lark
import pytest

import g2rins
from g2rins import GroupRule
from g2rins.exception import (
    GroupPartnerNotPlain,
    GroupRuleOnNestedObjectBondConnector,
    GroupRuleOnTerminalBondConnector,
    GroupRulesOnBothPathEnds,
    IncompatibleGroupPair,
    IndistinguishableSymbolsInSite,
    MixedOuterSymbolsInGroup,
    MixedRulesInGroup,
    RepeatedGroupInSite,
    SingleMemberGroup,
)

GROUP_KEYS = ("source_group", "source_rule", "target_group", "target_rule")
SENTINEL = (-1, 0, -1, 0)
WEIGHT_KEYS = ("propagation_weight", "termination_weight", "transition_weight")

ROUND_TRIP_CASES = [
    "[$[$1]1]",
    "[<1[<1]1]",
    "[>, >1[]1]",
    "[>[all]1]",
    "[>[all]]",
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
    bond_connector = g2rins.SimpleBondConnector.make("[>[all]0]")
    assert str(bond_connector) == "[>[all]]"


def test_terminal_bond_connector_parses_group_suffix():
    # The grammar accepts the suffix on any bond connector; a stochastic object
    # refuses it on its terminal bond connectors (see VALIDATION_ERROR_CASES).
    terminal = g2rins.TerminalBondConnector.make("[<[$]1]")
    assert str(terminal) == "[<[$]1]"
    (symbol,) = terminal.symbol
    assert symbol.group_rule == GroupRule.LADDER


def test_suffix_semantics():
    (symbol,) = g2rins.SimpleBondConnector.make("[>2[all]3]").symbol
    assert symbol.idx == 2
    assert symbol.group_rule == GroupRule.ALL
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
    assert GroupRule.ALL == 3


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
    assert _single_symbol("[>[all]1]").is_compatible(_single_symbol("[<]"))


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
        "{[] [<]C([>1[all]1])C([>2[]1])C[>]; ; [H][<] []}",
        MixedRulesInGroup,
        id="mixed-rules-in-group",
    ),
    pytest.param(
        "{[] [<]C([<[$]1])C([>[$]1])C[>]; ; [H][<] []}",
        MixedOuterSymbolsInGroup,
        id="mixed-outer-symbols-in-group",
    ),
    pytest.param(
        "{[] [<]C(C[>1[]1, >2[all]1])C[>]; ; [H][<] []}",
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
        "{[] [<]C([<[all]1])C([<[all]1])C[>], [<]C([>[all]2])C([>[all]2])C[>]; ; [H][<] []}",
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
        "{[] [<[all]2]{[>] [<[all]1]CC(C[<[all]1])O[>]; ; [<]F [<]}|poisson(100)|[>[all]2]; [<][H]; [<][H] []}|poisson(400)|",
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
    # Initiators never bond to initiators and terminators never to terminators, so
    # group-typed symbols that conjugate only across such pairs have no partner to check.
    pytest.param("{[] [$]CC[$]; C([$[all]1])([$[all]1])([$[all]1]); [$][H] []}|poisson(300)|", id="dollar-star-all-initiator"),
    pytest.param("{[] [$]CC[$]; C([$[]1])([$[]1]); [$][H] []}|poisson(300)|", id="dollar-exclusion-initiator"),
    pytest.param("{[] [$]CC[$]; C[$]; [$[]1][H], [$[]1]F []}|poisson(300)|", id="dollar-exclusion-terminators"),
    pytest.param("{[] [<]CC[>]; C([>[>]1])([>[>]1]), C([>[>]1])([>[>]1])([>[>]1]); [H][<] []}|poisson(300)|", id="ladder-groups-on-two-initiators"),
]


@pytest.mark.parametrize("text", END_GROUP_ONLY_PAIR_CASES)
def test_end_group_only_pairs_are_not_partners(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g2rins.StochasticObject.make(text)


def test_dollar_star_all_initiator_encoding():
    text = "{[] [$]CC[$]; C([$[all]1])([$[all]1])([$[all]1]); [$][H] []}|poisson(300)|"
    _assert_no_diagnostics(text)
    annotated = [edge for edge in _bond_connector_edges(text) if edge[3] != SENTINEL]
    # Three initiator sites x the two sites of the repeat unit, all through the all-group.
    assert len(annotated) == 6
    assert all(edge == ("[$[all]1]", "[$]", "transition_weight", (1, 3, -1, 0)) for edge in annotated)


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
    pytest.param("{[<] [<]CC([>,>[all]1])C([>,>[all]1])[>]; ; [H][<] [<]}|poisson(200)|", id="plain-beside-all"),
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
LADDER_TEXT = "{[] [<[<]2]OC(O[<[<]2])CC(O[>[>]1])O[>[>]1]; C(O[>[>]1])O[>[>]1]; [<][H] []}|poisson(1000)|"
DUAL_CHANNEL_TEXT = "{[] [<1,<[<]2]OC(O[<1,<[<]2])CC(O[>1,>[>]1])O[>1,>[>]1]; C(O[>1,>[>]1])O[>1,>[>]1]; [<1][H] []}|poisson(1000)|"
DUAL_CHANNEL_SWAPPED_TEXT = "{[] [<[<]2,<1]OC(O[<[<]2,<1])CC(O[>[>]1,>1])O[>[>]1,>1]; C(O[>[>]1,>1])O[>[>]1,>1]; [<1][H] []}|poisson(1000)|"
ALL_TEXT = "{[] [<]CCO[>]; C(O[>[all]1])(CO[>[all]1])(CO[>[all]1]); [H][<] []}|poisson(300)|"


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
    # Embedded, the exit edge reaches the bond-connector-free graph and gates generation.
    generative_graph = _generative_graph(_graph_creator("CC{[<] [<]CC([>,>1[]1])[>]; ; [H][<] [<1]}|poisson(200)|CC"))
    annotated = [data for _u, _v, data in generative_graph.edges(data=True) if _group_values(data) != SENTINEL]
    assert [(_group_values(data), _mode(data)) for data in annotated] == [((1, 2, -1, 0), "transition_weight")]
    with pytest.raises(NotImplementedError, match="EXCLUSION"):
        g2rins.EnsembleCreator(generative_graph)


MULTILEVEL_EXCLUSION_TEXT = "{[] [<,<2]C(=O)CC[>2]; {[] [<1]CC[>1]; {[] [<1]CCN([>1,>[]])([>1,>[]]), [<1]CCO[>1]; O[>1]; [<]|[<1]}|poisson(1000)|[>]|[>1]; [<]}|poisson(3000)|[>]; [<1][H] []}|poisson(5000)|"


def test_group_rule_survives_bond_connector_path_across_levels():
    # The exclusion channel of the level-2 unit exits through a terminal bond
    # connector, a level-1 bond connector, another terminal bond connector and a
    # level-0 bond connector before reaching the level-0 unit. Every relay is
    # plain, so the bond carries the rule of the unit it leaves.
    _assert_no_diagnostics(MULTILEVEL_EXCLUSION_TEXT)
    generative_graph = _generative_graph(_graph_creator(MULTILEVEL_EXCLUSION_TEXT))
    annotated = [(u, v, _mode(data), _group_values(data)) for u, v, data in generative_graph.edges(data=True) if _group_values(data) != SENTINEL]
    assert len(annotated) == 2  # one per site of the level-2 unit
    for u, v, mode, values in annotated:
        assert (mode, values) == ("transition_weight", (0, 2, -1, 0))
        # From a split node of the level-2 nitrogen to the level-0 carbonyl carbon.
        assert generative_graph.nodes[u]["atomic_num"] == 0
        assert 7 in {generative_graph.nodes[w]["atomic_num"] for w in generative_graph.neighbors(u)}
        assert generative_graph.nodes[v]["atomic_num"] == 6
    with pytest.raises(NotImplementedError, match="EXCLUSION"):
        g2rins.EnsembleCreator(generative_graph)


def test_group_rule_exit_into_enclosing_object_reaches_gate():
    # A ruled exit followed by plain relays used to contract to sentinels and
    # slip past the generation gate.
    text = "{[] [<]{[>] [<]CC([>,>1[]1])C([>,>1[]1])[>]; ; [<]F [<1]}|poisson(100)|[>]; [>][H]; [<][H] []}|poisson(400)|"
    generative_graph = _generative_graph(_graph_creator(text))
    assert {_group_values(data) for _u, _v, data in generative_graph.edges(data=True) if _group_values(data) != SENTINEL} == {(1, 2, -1, 0)}
    with pytest.raises(NotImplementedError, match="EXCLUSION"):
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
    text = "{[] [<]CC(C)O[>]; {[] [>,>1[]]N([>,>1[]])CCN([>,>1[]1])([>,>1[]1]), [<1]C(=O)CCCC(=O)[<1]; O[>1], [H][<]; [>1]O [<]}|gauss(4000,500)|[>]; [<][H] []}|gauss(5400,1000)|"
    generative_graph = _generative_graph(_graph_creator(text))
    assert {(-1, 0, 0, 2), (0, 2, -1, 0), (-1, 0, 1, 2), (1, 2, -1, 0)} <= {_group_values(data) for _u, _v, data in generative_graph.edges(data=True)}


def test_generation_gate():
    text = "{[] [<]C([>1[]1])C([>1[]1])C[>]; ; [H][<], [H][<1] []}|poisson(200)|"
    graph_creator = _graph_creator(text)
    # The bond-connector-free graph is a complete, exportable product ...
    generative_graph = _generative_graph(graph_creator)
    assert (1, 2, -1, 0) in {_group_values(data) for _u, _v, data in generative_graph.edges(data=True)}
    exported = g2rins.generative_graph_json_data(generative_graph)
    assert all(set(GROUP_KEYS) <= set(edge) for edge in exported["graph"]["edges"])
    # ... but sampling cannot honor group rules yet, whichever way the creator is built.
    with pytest.raises(NotImplementedError, match="EXCLUSION"):
        g2rins.EnsembleCreator(generative_graph)
    with pytest.raises(NotImplementedError):
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
ALL_STAR_TEXT = "{[] [<]C(C)C(=O)O[>], [<]CC(=O)O[>]; [>[all]]OCC(O[>[all]])CO[>[all]]; [<][H] []}|schulz_zimm(1800, 1200)|"


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
    with pytest.raises(NotImplementedError, match="LADDER"):
        g2rins.EnsembleCreator(_generative_graph(_graph_creator(LADDER_REFERENCE_TEXT)))


def test_all_star_string_encoding():
    _assert_no_diagnostics(ALL_STAR_TEXT)
    edges = _bond_connector_edges(ALL_STAR_TEXT)
    assert len(edges) == 16
    annotated = [edge for edge in edges if edge[3] != SENTINEL]
    # Three initiator sites x two repeat units, all transitions through the all-group.
    assert len(annotated) == 6
    assert all(edge == ("[>[all]]", "[<]", "transition_weight", (0, 3, -1, 0)) for edge in annotated)
    with pytest.raises(NotImplementedError, match="ALL"):
        g2rins.EnsembleCreator(_generative_graph(_graph_creator(ALL_STAR_TEXT)))
