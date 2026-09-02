# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Phase 0 tests for conditional connectivity (group rules).

Covers the per-symbol group suffix: parsing and round-tripping of the three
rules (ladder / exclusion / all), the compatibility matrix (ladder rigidity),
the stochastic-object validation set, and the temporary generation gate that
holds until the generation phases land.
"""

import warnings

import lark
import pytest

import g2rins
from g2rins import GroupRule
from g2rins.exception import (
    ExclusionPartnerNotPlain,
    IncompatibleGroupPair,
    MixedOuterSymbolsInGroup,
    MixedRulesInGroup,
    RepeatedGroupInSite,
    SingleMemberGroup,
)

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


def test_terminal_bond_connector_carries_group_suffix():
    # The grammar accepts the suffix on terminal bond connectors; validation
    # treats each terminal as a one-site scope (always a single-member group).
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
        "{[] [<]C([>1[]1])C([>1[]1])C[>], [<]C([<1[]2])C([<1[]2])C[>]; ; [H][<] []}",
        ExclusionPartnerNotPlain,
        id="exclusion-partner-not-plain",
    ),
    pytest.param(
        # A '$' exclusion channel is compatible with its own symbol on the next
        # unit instance, which is a group-typed partner too.
        "{[] [$[]1]CC[$]; C[$]; [H][$] []}",
        ExclusionPartnerNotPlain,
        id="exclusion-self-pair-not-plain",
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


def test_exclusion_beside_ladder_idx_reuse_is_legal():
    # Ladder rigidity means the exclusion channel and the ladder channel never
    # form an edge, so sharing outer index 1 is not an exclusion-partner error.
    text = "{[] [<]C([>1[]1])C([>1[]1])C[>], [<]C([<1[<]1])C([<1[<]1])C[>], [<]C([>1[>]2])C([>1[>]2])C[>]; C[>]; [H][<] []}|poisson(100)|"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g2rins.StochasticObject.make(text)
    assert not caught


def test_single_member_group_warns():
    with pytest.warns(SingleMemberGroup):
        g2rins.StochasticObject.make("{[] [<]C(C[>9, <[$]1])C[>]; ; [H][<] []}")


def test_ladder_only_chain_ends_raise_no_diagnostics():
    # Ladder-only sites may finish unreacted at conversion (implicit valence):
    # entry-side groups are consumed at engagement, initiator groups initiate,
    # and incomplete chain-end groups are intended behavior — no diagnostics.
    text = "{[] [<[<]2]OC(O[<[<]2])CC(O[>[>]1])O[>[>]1]; C(O[>[>]1])O[>[>]1]; [<][H] []}|poisson(1000)|"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stochastic_object = g2rins.StochasticObject.make(text)
    assert stochastic_object is not None
    assert not caught


def test_generation_gate():
    text = "{[] [<]C([>1[]1])C([>1[]1])C[>]; ; [H][<], [H][<1] []}|poisson(200)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stochastic_object = g2rins.StochasticObject.make(text)
        graph_creator = stochastic_object.get_graph_creator()
        with pytest.raises(NotImplementedError):
            graph_creator.get_ensemble_creator()
        # The bond-connector-free graph is the generation-bound product; it is
        # gated too, so the direct EnsembleCreator(graph) path cannot bypass.
        with pytest.raises(NotImplementedError):
            graph_creator.get_generative_graph()
        # The descriptor-level graph stays available (validation inspects it).
        graph_creator.get_generative_graph(include_bond_connectors=True)