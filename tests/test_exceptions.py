# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import copy
import pickle
import warnings

import lark
import pytest

import g2rins
from g2rins.atom import AtomSymbol


@pytest.mark.parametrize(
    "diagnostic",
    [
        g2rins.exception.ParsingError("missing symbol"),
        g2rins.exception.MissingAtomSymbol("Atom"),
        g2rins.exception.TooManyTokens("Atom", "C", "N"),
        g2rins.exception.UnsupportedBondDescriptor("[<]", "O=[<]C[>]", 2),
    ],
)
@pytest.mark.parametrize("copy_method", ["pickle", "deepcopy"])
def test_parser_and_descriptor_diagnostics_preserve_context(diagnostic, copy_method):
    restored = pickle.loads(pickle.dumps(diagnostic)) if copy_method == "pickle" else copy.deepcopy(diagnostic)
    assert type(restored) is type(diagnostic)
    assert restored.args == diagnostic.args
    assert vars(restored) == vars(diagnostic)
    assert str(restored) == str(diagnostic)


@pytest.mark.parametrize(
    "diagnostic",
    (
        g2rins.exception.DeadSamplingPath("nested target"),
        g2rins.exception.EmptyTruncatedDistributionSupport(
            "nested distribution",
            1.0,
            2.0,
        ),
        g2rins.exception.DiscardedSamplingPaths(
            3,
            (("DeadSamplingPath", 2), ("EmptyTruncatedDistributionSupport", 1)),
        ),
        g2rins.exception.NoValidGenerationSource(False),
        g2rins.exception.NoValidGenerationSource(True),
    ),
)
def test_sampling_path_diagnostics_pickle_roundtrip(diagnostic):
    """Sampling diagnostics must cross future process-worker boundaries intact."""
    restored = pickle.loads(pickle.dumps(diagnostic))

    assert type(restored) is type(diagnostic)
    assert restored.args == diagnostic.args
    assert str(restored) == str(diagnostic)


@pytest.mark.parametrize("invalid_name", ("Alfred", "Hitch"))
def test_unknown_distribution(invalid_name):
    with pytest.raises(lark.exceptions.UnexpectedCharacters):
        g2rins.G2rins.make("C{[$] [$]CC[$];; [$]}" + f"|{invalid_name}(4, 3)|C")

    with pytest.raises(g2rins.exception.UnknownDistribution):
        g2rins.StochasticDistribution.make(f"{invalid_name}(4, 3)")


def test_unsupported_big_smiles_features(big_smiles_features_unsupported):
    for smi in big_smiles_features_unsupported:
        with pytest.raises(g2rins.exception.UnsupportedBigSMILES):
            try:
                g2rins.G2rins.make(smi)
            except lark.exceptions.VisitError as exc:
                raise exc.__context__  # trunk-ignore(ruff/B904)


@pytest.mark.skip(reason="TODO: rewrite these so they pass the G2RINS grammar and reach the UnsupportedBigSMILES refusal instead of failing to parse")
def test_big_smiles_features_to_rewrite(big_smiles_features_to_rewrite):
    for smi in big_smiles_features_to_rewrite:
        with pytest.raises(g2rins.exception.UnsupportedBigSMILES):
            try:
                g2rins.G2rins.make(smi)
            except lark.exceptions.VisitError as exc:
                raise exc.__context__  # trunk-ignore(ruff/B904)


@pytest.mark.parametrize("invalid_smiles", [])
def test_double_bond_symbol_smiles(invalid_smiles):
    with pytest.raises(g2rins.exception.DoubleBondSymbolDefinition):
        g2rins.Smiles.make(invalid_smiles)


invalid_bond_connector_sequence = [
    "{[] [>]CC[>][<];; []}|gauss(400.,50.)|",
    "{[] [<]N(CC[>][<])N[>], [>]CC[<]; [$][H], [<]Br; []}|gauss(100.,20.)|",
    "{[<] [<]NN[>], [$]CC[$][$]; [$][H], [<]Br; [>]}|gauss(400.,20.)|",
    "{[<] [<]NN[>], [>]CC[<]; [$][$][H], [<]Br; [>]}|gauss(100.,14.)|",
]


@pytest.mark.parametrize("stochastic_smi", invalid_bond_connector_sequence)
def test_bond_connector_sequence(stochastic_smi):
    with pytest.raises(g2rins.exception.ConcatenatedBondConnectors):
        try:
            g2rins.StochasticObject.make(stochastic_smi)
        except lark.exceptions.VisitError as exc:
            raise exc.__context__  # trunk-ignore(ruff/B904)


valid_concatenated_bond_connectors = [
    "{[] [>]CC[>][<];; []}",
    "{[] [<]N(CC[>][<])N[>], [>]CC[<]; [$][H], [<]Br; []}",
    "{[<] [<]NN[>], [$]CC[$][$]; [$][H], [<]Br; [>]}",
]


@pytest.mark.parametrize("stochastic_smi", valid_concatenated_bond_connectors)
def test_valid_concatenated_bond_connectors(stochastic_smi):
    g2rins.StochasticObject.make(stochastic_smi)


invalid_monomer_stochastic = [
    "{[] [$]CC;; []}",
    "{[] CC ;;[]}",
    "{[] [$]CC[$], CC; [$]Br; []}",
    "{[$] [$]CC[$], [$]CC; [$]Br ;[$]}",
    "{[<] [>]CC, [<]C([$])C[>]; [$]Br; [>]}",
]


@pytest.mark.parametrize("stochastic_smi", invalid_monomer_stochastic)
def test_invalid_monomer_stochastic(stochastic_smi):
    with pytest.raises(g2rins.exception.MonomerHasTwoOrMoreBondConnectors):
        try:
            g2rins.StochasticObject.make(stochastic_smi)
        except lark.exceptions.VisitError as exc:
            raise exc.__context__  # trunk-ignore(ruff/B904)

    with pytest.raises(g2rins.exception.IncorrectNumberOfBondConnectors):
        try:
            g2rins.StochasticObject.make(stochastic_smi)
        except lark.exceptions.VisitError as exc:
            raise exc.__context__  # trunk-ignore(ruff/B904)


invalid_end_stochastic = [
    "{[] [$]CC[$]; C; []}",
    "{[] [$]CC[$]; [$]Br, N; []}",
]


@pytest.mark.parametrize("stochastic_smi", invalid_end_stochastic)
def test_invalid_end_stochastic(stochastic_smi):
    with pytest.raises(g2rins.exception.EndGroupHasBondConnectors):
        try:
            g2rins.StochasticObject.make(stochastic_smi)
        except lark.exceptions.VisitError as exc:
            raise exc.__context__  # trunk-ignore(ruff/B904)

    with pytest.raises(g2rins.exception.IncorrectNumberOfBondConnectors):
        try:
            g2rins.StochasticObject.make(stochastic_smi)
        except lark.exceptions.VisitError as exc:
            raise exc.__context__  # trunk-ignore(ruff/B904)


@pytest.mark.parametrize(
    "smi",
    [
        "{[] [$]CCC[$];; []}|poisson(10)|",
        "{[] [$][C-][$];; []}|flory_schulz(0.8)|",
    ],
)
def test_warn_empty_terminal_bond_connector_without_end_groups(smi):
    with pytest.warns(g2rins.exception.NoExplicitInitiation):
        g2rins.G2rins.make(smi)
    with pytest.warns(g2rins.exception.NoExplicitTermination):
        g2rins.G2rins.make(smi)


@pytest.mark.parametrize("smi", [])
def test_warn_no_initiation_for_stochastic_object(smi):
    with pytest.warns(g2rins.exception.NoInitiationForStochasticObject):
        obj = g2rins.G2rins.make(smi)
        obj.get_graph_creator()


TERMINATION_DECLARATION_CASES = [
    pytest.param(
        # Inner [<]Cl and outer [<]Br both reach the NN exit: the nearer
        # declaration wins and the outer edge is dropped.
        "{[] [<]CC[>], [<]{[>] [<]NN([>1])[>];; [<]Cl [<]|[<1]}|poisson(100)|[>]|[>1]; [>]I; [<]Br, [<1]F []}|poisson(400)|",
        g2rins.exception.ShadowedTerminationDeclaration,
        id="shadowed",
    ),
    pytest.param(
        # The NN side port declares no terminator of its own, so the enclosing
        # object's [<1]F caps it and controls when.
        "{[] [<]CC[>], [<]{[>] [<]NN([>1])[>];; [<]|[<1]}|poisson(100)|[>]|[>1]; [>]Cl; [<]Br, [<1]F []}|poisson(400)|",
        g2rins.exception.InheritedTermination,
        id="inherited",
    ),
    pytest.param(
        # [<1]Cl is declared beside the unit, but its bond connector belongs to
        # the enclosing object, which fires the cap and carries its mass.
        "{[] [<1]CC(C(=O)O)[>1]; {[] [<0]C(C)(C(=O)OCCOC(=O)C(C)(C)[>1])C[>0]; COC(=O)C(C)[>0]; [<0]Br, [<1]Cl [<1]}|gauss(4000.0, 500.0)|[>1]; [<1]Br []}|gauss(8000.0, 1000.0)|",
        g2rins.exception.ForeignControlledTermination,
        id="foreign-controlled",
    ),
    pytest.param(
        # A nested object used as the initiator: its block ends inherit the
        # enclosing object's Cl through the terminal bond connector path.
        "{[] [<]CCO[>]; {[] [<]CC(C)O[>]; O([>])[>]; [<]}|poisson(440)|[>]; [<]Cl []}|poisson(2200)|",
        g2rins.exception.InheritedTermination,
        id="inherited-initiator-section",
    ),
    pytest.param(
        # Same string with no terminator anywhere: the block ends hand their
        # growth to the enclosing object and nothing can cap them.
        "{[] [<]CCO[>]; {[] [<]CC(C)O[>]; O([>])[>]; [<]}|poisson(440)|[>]; []}|poisson(2200)|",
        g2rins.exception.MissingTermination,
        id="missing",
    ),
    pytest.param(
        # The inner [>2] sites have no growth partner at the enclosing level, so
        # the inner object owns them and fires the enclosing object's [<2]Br.
        "{[] [<1]{[>1] [<1]CCCO[>2], [<2]CCO[>2]; ; [<2]}|poisson(200)|[>2]; {[] [<][Si](C)([>1])O[>]; O[>]; [<][H] [<1]}|poisson(1000)|[>1]; [<2]Br []}|poisson(2000)|",
        g2rins.exception.InheritedTermination,
        id="inherited-fired-by-the-owner",
    ),
]


@pytest.mark.parametrize(("smi", "category"), TERMINATION_DECLARATION_CASES)
def test_warn_termination_declaration(smi, category):
    with pytest.warns(category):
        g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph(include_bond_connectors=False)


@pytest.mark.parametrize(
    ("smi", "declared_level", "firing_level", "phrase"),
    [
        # The block ends also exit to the enclosing object, which therefore fires the cap it declares.
        ("{[] [<]CCO[>]; {[] [<]CC(C)O[>]; O([>])[>]; [<]}|poisson(440)|[>]; [<]Cl []}|poisson(2200)|", 0, 0, "That object decides when the site is capped"),
        # The inner object owns the sites, so it fires the cap the enclosing object declares and carries its mass.
        (TERMINATION_DECLARATION_CASES[-1].values[0], 0, 1, "Stochastic object 1, which owns the site's bond descriptor, fires the cap"),
    ],
)
def test_inherited_termination_names_the_object_that_fires_the_cap(smi, declared_level, firing_level, phrase):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        generative_graph = g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    inherited = [w.message for w in caught if isinstance(w.message, g2rins.exception.InheritedTermination)]
    assert inherited, "no InheritedTermination warning"
    for warning in inherited:
        assert (warning.declared_level, warning.firing_level) == (declared_level, firing_level)
        assert phrase in str(warning)
    # The warning's firing object is the one stamped on the surviving cap edges.
    terminators = {node for node, data in generative_graph.nodes(data=True) if data["atomic_num"] in (17, 35)}
    assert {data["stochastic_id"] for _u, v, data in generative_graph.edges(data=True) if v in terminators and data["termination_weight"] > 0} == {firing_level}


def test_zero_amount_declaration_neither_caps_nor_shadows():
    """A terminator declared with molar amount 0 is no declaration for the
    resolution: the site inherits the live outer cap instead, nothing is
    reported as shadowed, and the zero-amount cap keeps no edge."""
    smi = "{[] [<]CC[>], [<]{[>] [<]NN[>];; [<]Cl|0| [<]}|poisson(100)|[>]; [>]I; [<]Br []}|poisson(400)|"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        generative_graph = g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    assert not any(isinstance(w.message, g2rins.exception.ShadowedTerminationDeclaration) for w in caught)
    inherited = [w.message for w in caught if isinstance(w.message, g2rins.exception.InheritedTermination)]
    assert inherited and all(warning.terminators == ("[<]Br",) for warning in inherited)
    chlorine = {node for node, data in generative_graph.nodes(data=True) if data["atomic_num"] == 17}
    assert chlorine and not any(v in chlorine and data["termination_weight"] > 0 for _u, v, data in generative_graph.edges(data=True))


@pytest.mark.parametrize(
    "smi",
    [
        "{[] [<]CC[>]; C[>]; [<][H] []}|poisson(50)|",
        "C{[>][<]CC(C)[>];;[<]}|poisson(900)|[H]",
        "{[] [<|.8|]CCO[>|.8|], [<|.2|]CC(C)O[>|.2|]; [>][H] ; [<]Br []}|log_normal(1400, 1.15)|",
        # Nested objects whose exits reach a written endcap ([H] in the unit
        # text) or that are only ENTERED from the site: realized with the
        # unit, never left open, so no warning.
        "{[] [<]CC({[<] [<]NN[>];; [>]}|poisson(80)|[H])C({[<] [<]C(C)O[>];; [>]}|poisson(80)|[H])C[>]; [<][H]; [>][H] []}|poisson(400)|",
        "{[] [<]CC({[<] [<]NN({[<] [<]C(C)O[>];; [>]}|poisson(80)|[H])[>];; [>]}|poisson(200)|[H])CC[>]; [<][H]; []}|poisson(800)|",
        # Two nested objects joined directly: the join is a forced exit, delivered on every
        # instance of the first object, so its sites are never left open.
        "{[] [<]CC({[<] [<]NN[>];; [>]}|poisson(80)|{[<] [<]CO[>];; [>]}|poisson(80)|Br)C[>]; C[>]; [<][H] []}|poisson(2000)|",
        # An initiator written as a nested object followed by atoms: the tail's
        # bond connector is a plain initiator port, consumed by its own object's
        # growth, so neither a cap nor a warning belongs to it.
        "{[] [<]CCO[>]; {[] [<]CC(C)O[>]; CO[>]; [<]}|poisson(700)|CC(C)([>])C[>]; [<]Cl []}|poisson(800)|",
    ],
)
def test_terminator_declared_beside_its_unit_is_silent(smi):
    """Capping a site in the step that grows it is the canonical case and must
    stay quiet; only configurations that hand control to another stochastic
    object are worth a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    reported = [str(entry.message) for entry in caught if isinstance(entry.message, g2rins.exception.TerminationDeclarationWarning)]
    assert not reported, reported


def test_a_transition_at_the_site_own_level_is_not_a_missing_cap():
    """The initiator of the middle object, a nested object followed by atoms,
    has no terminator of its own and needs none: its tail's bond connector
    starts that object's growth, like any initiator port. Only a site whose
    growth continues at an enclosing object's discretion can be left open, so
    the site's own level never makes a cap missing."""
    smi = "{[] [<]CC([>])C(=O)OC(C)(C)C; {[] [<]CC([>])(C)C(=O)OCCO; {[] [<]CCO[>]; CO[>]; [<]}|poisson(3000)|CCOC(=O)C(C)(C)[>]; [<]}|poisson(6000)|[>]; [<]Br []}|poisson(10000)|"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    missing = [str(entry.message) for entry in caught if isinstance(entry.message, g2rins.exception.MissingTermination)]
    assert not missing, missing


@pytest.mark.parametrize("smi", ["{[$] [>]CC[<];; [>]}|flory_schulz(0.9)|"])
def test_warn_stochastic_missing_path(smi):
    with pytest.warns(g2rins.exception.StochasticMissingPath):
        obj = g2rins.G2rins.make(smi)
        obj.get_graph_creator()


@pytest.mark.parametrize("smi", ["{[] [<]CC(C[NH3+])[>].[Cl-].[Cl-]; C[>]; [<][H] []}|poisson(500.0)|"])
def test_warn_counterion_fallback_and_imbalance(smi):
    # The surplus second Cl- finds no remaining opposite-charge site (falls
    # back to the first atom) and leaves the unit with a net charge.
    with pytest.warns(g2rins.exception.UnmatchedCounterion):
        g2rins.G2rins.make(smi).get_graph_creator()
    with pytest.warns(g2rins.exception.CounterionChargeImbalance):
        g2rins.G2rins.make(smi).get_graph_creator()


@pytest.mark.parametrize("smi", [])
def test_warn_incompatible_bond_type_bond_connectors(smi):
    with pytest.warns(g2rins.exception.IncompatibleBondTypeBondConnector):
        obj = g2rins.G2rins.make(smi)
        obj.get_graph_creator()


undefined_distribution = [
    "{[] [$]CC[$]; [$]C; []}",
]


@pytest.mark.parametrize("smi", undefined_distribution)
def test_undefined_distribution(smi):
    with pytest.raises(g2rins.exception.UndefinedDistribution):
        obj = g2rins.G2rins.make(smi)
        obj.get_graph_creator()


# TODO: implement tests for IncorrectNumberOfBondProbabilities and EmptyBondConnectorInTerminalBondConnectorList. Add nested examples.


def test_too_many_tokens_round_trips_with_atom_symbol_tokens():
    """Real symbol tokens have no equality, so compare by type and rendering."""
    first, second = AtomSymbol.make("C"), AtomSymbol.make("N")
    diagnostic = g2rins.exception.TooManyTokens("Atom", first, second)
    for restored in (pickle.loads(pickle.dumps(diagnostic)), copy.deepcopy(diagnostic)):
        assert type(restored) is g2rins.exception.TooManyTokens
        assert str(restored) == str(diagnostic)
        assert [str(token) for token in restored.args] == ["Atom", "C", "N"]
        assert str(restored.token) == "N"
        assert str(restored.existing_token) == "C"
