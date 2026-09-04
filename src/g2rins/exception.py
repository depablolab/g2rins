# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only


class G2RINSError(Exception):
    """
    Generic Exception raised by G2RINS.
    """

    pass


class G2RINSWarning(Warning):
    """
    Generic Warning raised by G2RINS.
    """

    def __init__(self, token):
        self.token = token


class ParsingError(G2RINSError):
    """
    Parsing the Grammar went in an unanticipated manner.
    Please report bug with input string.
    """

    def __init__(self, token):
        super().__init__()
        self.token = token

    def __str__(self):
        return f"Unanticipated error while parsing. Please report and provide the input string. Token: {self.token}"


class ParsingWarning(G2RINSWarning):
    """
    Parsing the your string doesn't invalidate the grammar, but there is something to consider to fix.
    """

    pass


class TooManyTokens(ParsingError):
    def __init__(self, class_name, existing_token, new_token):
        super().__init__()
        self.class_name = class_name
        self.existing_token = existing_token
        self.new_token = new_token

    def __str__(self):
        string = f"Parsing Error {self.class_name} only expected one token, but got more. "
        string += f"The existing token is {self.existing_token} which conflicts with the new "
        string += f"token {self.new_token}. Most likely in implementation error, please report."
        return string


class UnknownDistribution(G2RINSError):
    def __init__(self, distribution_text: str):
        super().__init__()
        self.distribution_text = distribution_text

    def __str__(self):
        string = f"G2RINS a distribution with the following text {self.distribution_text} is unknown."
        string += " Typo or not implemented distribution."
        return string


class UnsupportedBigSMILES(G2RINSError):
    def __init__(self, token_type: str, other):
        self.token_type = token_type
        self.other = other

    def __str__(self):
        string = f"The provided token {self.token_type} is supported in BigSMILES but not in G2RINS. "
        string += f"This token was requested by the following parsed text {self.other}."
        return string


class GenerationError(G2RINSError):
    pass


class DoubleBondSymbolDefinition(GenerationError):
    def __init__(self, partial_graph, symbol, bond_attributes):
        self.partial_graph = partial_graph
        self.symbol = symbol
        self.bond_attributes = bond_attributes

    def __str__(self):
        return f"{self.partial_graph}, {self.symbol}, {self.bond_attributes}"


class UndefinedDistribution(GenerationError):
    def __init__(self, stochastic_obj):
        self.stochastic_obj = stochastic_obj

    def __str__(self):
        return f"The stochastic distribution of the stochastic object {self.stochastic_obj} is not defined. The creation of the generative graph requires that the distribution is defined for each stochastic object."


class ConcatenatedBondConnectors(ParsingError):
    def __init__(self, obj, stochastic_obj):
        self.obj = obj
        self.stochastic_obj = stochastic_obj

    def __str__(self):
        return f"The object {self.obj} in stochastic object {self.stochastic_obj} has concatenated bond connectors which is forbidden by G-G2rins grammar."


class IncorrectNumberOfBondConnectors(ParsingError):
    def __init__(self, obj, expected_number_of_bond_connectors):
        self.obj = obj
        self.expected_number_of_bond_connectors = expected_number_of_bond_connectors

    def __str__(self):
        return f"Incorrect Number of BondConnectors we expected {self.expected_number_of_bond_connectors} but the object {str(self.obj)} of type {type(self.obj)} has {len(self.obj.bond_connectors)}."


class SmilesHasNonZeroBondConnectors(IncorrectNumberOfBondConnectors):
    def __init__(self, obj):
        super().__init__(obj, 0)

    def __str__(self):
        return f"Outside of Stochastic Objects we expect {self.expected_number_of_bond_connectors} bond connectors, but the object {self.obj} of type {type(self.obj)} has the following bond connectors {[str(bc) for bc in self.obj.bond_connectors]}."


class UnmatchedCounterion(ParsingWarning):
    def __init__(self, counterion):
        super().__init__(counterion)

    def __str__(self) -> str:
        return f"The counterion {str(self.token)} found no opposite-charge site in its repeat unit; it is attached to the first atom of the unit instead."


class CounterionChargeImbalance(ParsingWarning):
    def __init__(self, residue, total_charge):
        super().__init__(residue)
        self.total_charge = total_charge

    def __str__(self) -> str:
        return f"The repeat unit {str(self.token)} together with its counterions has a non-zero net charge of {self.total_charge:+d}."


class MonomerHasTwoOrMoreBondConnectors(IncorrectNumberOfBondConnectors):
    def __init__(self, monomer_obj, stochastic_obj):
        super().__init__(monomer_obj, 2)

        self.stochastic_obj = stochastic_obj

    def __str__(self) -> str:
        return f"Monomer repeat units must have at least {self.expected_number_of_bond_connectors} bond connectors. But this object {str(self.obj)} has {len(self.obj.bond_connectors)} bond connectors inside this stochastic object {str(self.stochastic_obj)}."


class EndGroupHasBondConnectors(IncorrectNumberOfBondConnectors):
    def __init__(self, end_obj, stochastic_obj):
        super().__init__(end_obj, 1)

        self.stochastic_obj = stochastic_obj

    def __str__(self) -> str:
        return f"End groups must have at least {self.expected_number_of_bond_connectors} bond connector. But this object {str(self.obj)} has {len(self.obj.bond_connectors)} bond connectors inside this stochastic object {str(self.stochastic_obj)}."


class NoExplicitInitiation(ParsingWarning):
    def __init__(self, stochastic_object):
        super().__init__(stochastic_object)

    def __str__(self) -> str:
        string = f"No explicit initiator defined. The stochastic object {str(self.token)} has an empty left terminal bond connector '[]' and an empty list of initiators."
        # string += "Repeat units will be used as initiators."
        return string


class NoExplicitTermination(ParsingWarning):
    def __init__(self, stochastic_object):
        super().__init__(stochastic_object)

    def __str__(self) -> str:
        string = f"No explicit terminator defined. The stochastic object {str(self.token)} has an empty right terminal bond connector '[]' and an empty list of terminators. "
        string += "In this case, please consider adding hydrogen groups in the list of terminators, with bond connector symbols that match the G2RINS string: '{[] ... ; [$/</>][H] []}'"
        return string


class EmptyBondConnectorInTerminalBondConnectorList(ParsingError):
    def __init__(self, terminal_bond_connectors, stochastic_object):
        self.terminal_bond_connectors = terminal_bond_connectors
        self.stochastic_object = stochastic_object

    def __str__(self) -> str:
        return f"The terminal bond connector list {self.terminal_bond_connectors} of the stochastic object:\n{self.stochastic_object}\nhas empty bonds.\nEmpty bonds in lists of terminal bond connectors with length > 1 are not allowed."


class IncorrectNumberOfBondProbabilities(ParsingError):
    def __init__(self, token, bond_connector, expected_length):
        super().__init__(token)
        self.bond_connector = bond_connector
        self.expected_length = expected_length
        if self.bond_connector.bond_probabilities is not None:
            raise RuntimeError(f"Implementation error, please report on GitHub https://github.com/depablolab/g2rins/issues . {str(self.bond_connector)} {str(self.token)} ")

    def __str__(self):
        return f"The bond connector '{str(self.bond_connector)}' from the stochastic object '{str(self.token)}' specifies {len(self.bond_connector.bond_probabilities)} bond probabilities, but the stochastic object has {self.expected_length} bond connectors. Adjust the bond probabilities to match the bond connectors."


class NoInitiationForStochasticObject(ParsingWarning):
    def __init__(self, stochastic_obj, partial_graph):
        super().__init__(stochastic_obj)
        self.partial_graph = partial_graph

    def __str__(self):
        return f"The stochastic object {str(self.token)} cannot generate entry points to start initiations. Check if the left terminal bond connector is meant to be empty or if you have correct end groups that can act as initiators."


class NoTerminationForStochasticObject(ParsingWarning):
    def __init__(self, stochastic_obj, partial_graph):
        super().__init__(stochastic_obj)
        self.partial_graph = partial_graph

    def __str__(self):
        return f"The stochastic object {str(self.token)} lacks explicit termination."  # TODO: implement automatic hydrogenation of repeat unit with no explicit termination


# No NoLeftTransitions/NoRightTransitions warnings exist: a declared terminal
# symbol always creates the corresponding half-bond (unconditional appends in
# StochasticObject._generate_partial_graph), so those states are unreachable;
# a declared entry/exit that generation cannot use is reported by
# StochasticMissingPath instead.


class StochasticMissingPath(ParsingWarning):
    def __init__(self, stochastic_obj, source_bc_pos):
        super().__init__(stochastic_obj)
        self.source_bc_pos = source_bc_pos

    def __str__(self):
        return f"The stochastic object {str(self.token)} defines that it can be entered via the bond connector in position {str(self.source_bc_pos)} as defined by the left terminal connector. However, when entered there there is no path to reach any of the exit bond connectors as defined by the right terminal bond connector."


class IncompatibleBondTypeBondConnector(ParsingWarning):
    def __init__(self, bond_type_lhs, bond_type_rhs):
        super().__init__(None)
        self.bond_type_lhs = bond_type_lhs
        self.bond_type_rhs = bond_type_rhs

    def __str__(self):
        return f"There is a connection between bond connectors with different bond types, the left is {str(self.bond_type_lhs)} and the right is {str(self.bond_type_rhs)}. There will be no generation path, since the bond type is undefined. This maybe an incorrect input G2RINS string, check the bond types for compatibility around the bond connectors: connecting bond connectors have of same type i.e. '[<]=CC[>]' is invalid, since it connects a double bond = with a single bond implicit `-`, correct would be `[<]=CC=[>]`."


class UnvalidatedGenerationSource(G2RINSWarning):
    def __init__(self, source, known_source_ids, graph):
        self.source = source
        self.known_source_ids = known_source_ids
        self.graph = graph

    def __str__(self):
        return f"Attempt to create an atom graph from a generative graph with source node_idx {self.source} but this is not one of the known starting points ({self.known_source_ids}) of the generative graph."


class InvalidGenerationSource(G2RINSError):
    def __init__(self, source, nodes, graph):
        self.source = source
        self.nodes = nodes
        self.graph = graph
        super().__init__(source)

    def __str__(self):
        return (
            f"Attempt to create an atom graph from source node_idx {self.source!r}, "
            f"which is not one of the {len(self.nodes)} node idx of the generative graph."
        )


class NoValidGenerationSource(G2RINSError):
    """Automatic source selection has no candidates in the requested mode."""

    def __init__(self, use_repeat_units_as_source=False):
        self.use_repeat_units_as_source = bool(use_repeat_units_as_source)
        super().__init__(self.use_repeat_units_as_source)

    def __str__(self):
        mode = (
            "repeat-unit"
            if self.use_repeat_units_as_source
            else "default"
        )
        return (
            f"No valid automatic generation source is available in {mode} source mode. "
            "Supply an explicit valid source or revise the G2RINS initiation paths."
        )


class PossibleNonRepresentativePolymerChain(G2RINSWarning):
    _MESSAGE = (
        "The created atom graph might not be representative. This can happen when the process of creation "
        "leads to unintended behaviors. For instance, a prepolymer represented by an inner stochastic "
        "object reached itself the target MW of a larger polymer represented by a higher-order stochastic "
        "object. This can lead to the interruption of the chain creation in complex structures. Please, "
        "check the G2RINS string and evaluate if the chain should be discarded."
    )

    def __init__(self, token=None):
        self.token = token
        super().__init__(token)
        Warning.__init__(self, self._MESSAGE)

    def __str__(self):
        if self.token is None:
            return self._MESSAGE
        return f"{self._MESSAGE} Token: {self.token}"


class UndershootSnapshotMissed(G2RINSWarning):
    _MESSAGE = (
        "A molecular weight crossing was reached before an undershoot snapshot was taken, so this "
        "chain falls back to the overshoot state, which can bias it above its target molecular "
        "weight. This should be rare: a single growth step added more than the adaptive lookahead "
        "(a fixed multiple of the largest mass gain previously seen for the stochastic object, "
        "e.g. a nested stochastic object grew unusually large in one step). If ensembles look "
        "biased, please report the string."
    )

    def __init__(self, token=None):
        self.token = token
        super().__init__(token)
        Warning.__init__(self, self._MESSAGE)

    def __str__(self):
        if self.token is None:
            return self._MESSAGE
        return f"{self._MESSAGE} Token: {self.token}"


class ForcedOvershootNoBoundary(G2RINSWarning):
    """A final explicit undershoot remained impossible because no valid
    owner-level boundary existed below the target.

    This is distinct from :class:`UndershootSnapshotMissed`: the adaptive
    lookahead did not lose a usable state. The object either crossed on its
    first owner-level step or its immediately preceding complete boundary was
    already at/over target, so the only chemically complete final result is the
    overshoot topology.
    """

    _MESSAGE = (
        "An undershoot was requested, but this stochastic object crossed its target before "
        "a chemically complete owner-level undershoot boundary existed. The chain therefore "
        "uses the unavoidable overshoot state."
    )

    def __init__(self, token=None):
        self.token = token
        super().__init__(token)
        Warning.__init__(self, self._MESSAGE)

    def __str__(self):
        if self.token is None:
            return self._MESSAGE
        return f"{self._MESSAGE} Token: {self.token}"


class DeadSamplingPath(G2RINSError):
    """The current stochastic chain reached a local sampling dead end.

    Unlike a fatal model error, another chain can take a different earlier
    stochastic path and avoid this state.  ``create_ensemble`` may therefore
    reject this chain explicitly and retry.  Direct sampling receives this
    wrapper; the precise underlying sampling error is retained in
    ``__cause__`` (except when re-raised from a ``parallel=True`` run: the
    cause chain does not survive pickling across the process boundary).

    ``*details`` keeps subclasses reconstructible from ``Exception.args`` for
    process workers and pickle round-trips.
    """

    def __init__(self, context, *details):
        self.context = context
        super().__init__(context, *details)

    def __str__(self):
        return (
            f"The current chain reached the dead sampling path '{self.context}'. "
            "Another chain may avoid this path, but rejecting it conditions the "
            "returned ensemble on successful generation."
        )


class EmptyTruncatedDistributionSupport(DeadSamplingPath):
    """A truncated molecular-weight draw was requested over an interval that
    carries no probability mass (e.g. a nested stochastic object whose
    distribution lies entirely above the remaining parent budget).

    Chain-local: whether a chain reaches such a draw depends on the parent
    target it happened to draw, so create_ensemble treats this as a counted
    discard rather than aborting the run.
    """

    def __init__(self, context, lower, upper):
        self.lower = lower
        self.upper = upper
        super().__init__(context, lower, upper)

    def __str__(self):
        return (
            f"The truncated sampling interval [{self.lower:g}, {self.upper:g}] of '{self.context}' "
            "contains no probability mass; the requested bounds exclude the distribution's support "
            "(e.g. a nested target distribution lying above the remaining parent budget)."
        )


class AllZeroSamplingWeights(G2RINSError):
    """Every candidate of a sampling decision carries zero weight.

    The generation weights are products of the declared bond weights and molar
    amounts.  At a route shared by every allowed source this is a fatal model
    error (for example, a |0| molar amount on every alternative).  If the
    decision was reached only through one stochastic route, the sampler may
    wrap it in ``DeadSamplingPath`` so an ensemble can reject that chain and
    try a viable route.
    """

    def __init__(self, context):
        self.context = context
        super().__init__(context)

    def __str__(self):
        return (
            f"Every candidate of the '{self.context}' decision has zero sampling weight "
            "(bond weight x molar amount). Check the G2RINS string for zero molar "
            "amounts (e.g. '|0|') or zero bond weights that leave no valid alternative."
        )


class DiscardedSamplingPaths(G2RINSWarning):
    """Summary of chain-local paths rejected while building an ensemble."""

    def __init__(self, discarded_count, reasons):
        self.discarded_count = int(discarded_count)
        self.reasons = tuple(
            sorted((str(reason), int(count)) for reason, count in reasons)
        )
        Warning.__init__(self, self.discarded_count, self.reasons)

    def __str__(self):
        reason_text = ", ".join(
            f"{reason}: {count}" for reason, count in self.reasons
        )
        return (
            f"Discarded {self.discarded_count} chain-local sampling path(s) "
            f"while generating the ensemble ({reason_text}). The returned "
            "ensemble is conditioned on avoiding these unsuccessful paths."
        )


class TooManyStochasticObjects(G2RINSError):
    """The string declares more stochastic objects than the graph schema holds.

    Per-node vectors (molecular_weight_distribution, unit_molar_amounts,
    stochastic_id_tree) have one slot per stochastic object id up to a fixed
    depth, so ids beyond it cannot be represented.
    """

    def __init__(self, count, limit):
        self.count = count
        self.limit = limit
        super().__init__(count)

    def __str__(self):
        return (
            f"The G2RINS string declares {self.count} stochastic objects, but the "
            f"generating-graph schema supports at most {self.limit} (fixed-size per-node "
            "vectors, see _STOCHASTIC_TREE_DEPTH). Split the string or increase the limit."
        )


class IncompatibleGenerativeGraphSchema(G2RINSError):
    """A generative graph misses attributes the sampler requires.

    Sampling silently generates truncated, end-group-less molecules when edges
    lack the per-edge 'stochastic_id', so an explicit check rejects graphs
    built against an older schema.
    """

    def __init__(self, missing_attribute):
        self.missing_attribute = missing_attribute
        super().__init__(missing_attribute)

    def __str__(self):
        return (
            f"The provided generative_graph has edges without the '{self.missing_attribute}' attribute "
            "that generation requires. It was probably serialized or built against an older "
            "graph schema (which used per-edge 'hierarchy'); regenerate it with "
            "get_graph_creator().get_generative_graph(include_bond_connectors=False)."
        )


class TooManyDiscardedChains(G2RINSWarning):

    def __init__(self, max_number_of_discarded_chains, token=None):
        self.token = token
        self.max_number_of_discarded_chains = max_number_of_discarded_chains
        super().__init__(token)
        Warning.__init__(self)

    def __str__(self):
        if self.token is None:
            return f"The number of discarded non-representative chains exceeded max_number_of_discarded_chains = {self.max_number_of_discarded_chains}. This might be due to problems in the G2RINS string or an implementation issue. Please revise the string. If correct, consider increasing max_number_of_discarded_chains. If the issue persists, please report the problem."

        return f"The number of discarded non-representative chains exceeded max_number_of_discarded_chains = {self.max_number_of_discarded_chains}. This might be due to problems in the G2RINS string or an implementation issue. Please revise the string. If correct, consider increasing max_number_of_discarded_chains. If the issue persists, please report the problem. Token: {self.token}"


class IncompleteStochasticGeneration(G2RINSError):
    def __init__(self, partial_atom_graph):
        self._partial_atom_graph = partial_atom_graph
        self.atom_graph = partial_atom_graph.atom_graph

    @property
    def num_open_bonds(self):
        num_bonds = 0
        for sto_atom_id in self._partial_atom_graph._open_half_bond_map:
            for _bond in self._partial_atom_graph._open_half_bond_map[sto_atom_id]:
                num_bonds += 1
        return num_bonds

    def __str__(self):
        num_bonds = self.num_open_bonds
        if num_bonds == 0:
            return f"Incomplete Stochastic Generation: since there are {num_bonds} open bonds this may be intended. You can catch this exception and use the `atom_graph` property as a result."
        return f"Incomplete Stochastic Generation: {num_bonds} are still unaccounted for this is likely an imprecise G2RINS string or a bug."


class MixedRulesInGroup(ParsingError):
    def __init__(self, group_id, owner, stochastic_obj):
        self.group_id = group_id
        self.owner = owner
        self.stochastic_obj = stochastic_obj

    def __str__(self):
        return f"All members of group {self.group_id} in the unit {str(self.owner)} of the stochastic object {str(self.stochastic_obj)} must declare the same group rule, but the group mixes rules."


class MixedOuterSymbolsInGroup(ParsingError):
    def __init__(self, group_id, owner, stochastic_obj):
        self.group_id = group_id
        self.owner = owner
        self.stochastic_obj = stochastic_obj

    def __str__(self):
        return f"All members of ladder group {self.group_id} in the unit {str(self.owner)} of the stochastic object {str(self.stochastic_obj)} must carry the same outer bond connector symbol and index. A self-connecting group uses '$'; mixed '<'/'>' outer symbols within one group are not allowed."


class RepeatedGroupInSite(ParsingError):
    def __init__(self, group_id, bond_connector, stochastic_obj):
        self.group_id = group_id
        self.bond_connector = bond_connector
        self.stochastic_obj = stochastic_obj

    def __str__(self):
        return f"The bond connector {str(self.bond_connector)} in the stochastic object {str(self.stochastic_obj)} lists group {self.group_id} on more than one of its symbols; a site may join a group at most once."


class IncompatibleGroupPair(ParsingError):
    def __init__(self, group_a, owner_a, group_b, owner_b, stochastic_obj, reason):
        self.group_a = group_a
        self.owner_a = owner_a
        self.group_b = group_b
        self.owner_b = owner_b
        self.stochastic_obj = stochastic_obj
        self.reason = reason

    def __str__(self):
        return f"Ladder group {self.group_a} of unit {str(self.owner_a)} and ladder group {self.group_b} of unit {str(self.owner_b)} in the stochastic object {str(self.stochastic_obj)} can engage through a compatible symbol pair but cannot complete: {self.reason}."


class GroupPartnerNotPlain(ParsingError):
    def __init__(self, symbol, partner_symbol, stochastic_obj):
        self.symbol = symbol
        self.partner_symbol = partner_symbol
        self.stochastic_obj = stochastic_obj

    def __str__(self):
        return f"The {self.symbol.group_rule.name.lower()}-typed symbol {str(self.symbol)} in the stochastic object {str(self.stochastic_obj)} is compatible with {str(self.partner_symbol)}, which also carries a group suffix. Exclusion and all channels must point at plain bond connector symbols."


class GroupRuleOnTerminalBondConnector(ParsingError):
    def __init__(self, bond_connector, stochastic_obj):
        self.bond_connector = bond_connector
        self.stochastic_obj = stochastic_obj

    def __str__(self):
        return f"The terminal bond connector {str(self.bond_connector)} of the stochastic object {str(self.stochastic_obj)} carries a group suffix. A terminal bond connector only relays bonds to the enclosing level and must be plain; group rules belong to the bond connectors of units."


class GroupRuleOnNestedObjectBondConnector(ParsingError):
    def __init__(self, bond_connector, owner, stochastic_obj):
        self.bond_connector = bond_connector
        self.owner = owner
        self.stochastic_obj = stochastic_obj

    def __str__(self):
        return f"The bond connector {str(self.bond_connector)} in the unit {str(self.owner)} of the stochastic object {str(self.stochastic_obj)} attaches a nested stochastic object and carries a group suffix. Such a bond connector only relays bonds between levels and must be plain; a bond carries the group rules of the units at its two ends."


class SingleMemberGroup(ParsingWarning):
    def __init__(self, group_id, rule_name, owner):
        super().__init__(owner)
        self.group_id = group_id
        self.rule_name = rule_name

    def __str__(self):
        return f"Group {self.group_id} ({self.rule_name}) in the unit {str(self.token)} has a single member; a one-member group has no effect."


class IndistinguishableSymbolsInSite(ParsingWarning):
    def __init__(self, symbol, bond_connector, owner):
        super().__init__(owner)
        self.symbol = symbol
        self.bond_connector = bond_connector

    def __str__(self):
        return f"The bond connector {str(self.bond_connector)} in the unit {str(self.token)} lists the {self.symbol.group_rule.name.lower()}-typed symbol {str(self.symbol)} beside a plain symbol with the same outer symbol and index; partners cannot tell the two apart, so bonds are drawn through either with equal odds."


class GroupRulesOnBothPathEnds(G2RINSError):
    def __init__(self, source_bond_connector, source_values, target_bond_connector, target_values):
        self.source_bond_connector = source_bond_connector
        self.source_values = source_values
        self.target_bond_connector = target_bond_connector
        self.target_values = target_values

    def __str__(self):
        from .bond import GroupRule

        return f"The bond from the bond connector {self.source_bond_connector} to the bond connector {self.target_bond_connector} crosses a nesting level and both ends declare a group rule (group {self.source_values[0]} {GroupRule(self.source_values[1]).name} and group {self.target_values[0]} {GroupRule(self.target_values[1]).name}). Exclusion and all channels must point at plain bond connector symbols, so a bond connector path may carry a group rule at one end only."
