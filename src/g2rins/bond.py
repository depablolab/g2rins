# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import uuid
from enum import IntEnum

import networkx as nx

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .core import G2rinsBase, GenerationBase
from .generative_graph import _GROUP_EDGE_ATTR, _HalfBond, _PartialGraph


class GroupRule(IntEnum):
    """Group rules of conditional connectivity; values are stable graph-feature encodings."""

    NONE = 0
    LADDER = 1
    EXCLUSION = 2
    ALL = 3


class BondSymbol(G2rinsBase):
    def __init__(self, children: list):
        super().__init__(children)
        self._symbol = str(self._children[0])

    @property
    def generable(self):
        return True

    def generate_string(self, extension):
        return self._symbol


class RingBond(G2rinsBase):
    def __init__(self, children: list):
        super().__init__(children)

        self._bond_symbol: None | BondSymbol = None
        self._has_dollar: bool = False

        num_text = ""
        for child in self._children:
            if isinstance(child, BondSymbol):
                self._bond_symbol = child
            elif str(child) == "%":
                self._has_dollar = True
            elif str(child).isdigit():
                num_text += str(child)
        self._num: int = int(num_text)

    def generate_string(self, extension):
        string = ""
        if self._bond_symbol is not None:
            string += self._bond_symbol.generate_string(extension)
        if self._has_dollar:
            string += "%"
        string += str(self._num)
        return string

    @property
    def generable(self):
        return True

    @property
    def idx(self) -> int:
        return self._num


class BondConnectorSymbol(G2rinsBase):
    def __init__(self, children: list):
        super().__init__(children)

    def generate_string(self, extension):
        return str(self._children[0])

    def generable(self):
        return True


class BondConnectorSymbolIdx(BondConnectorSymbol):
    def __init__(self, children):
        super().__init__(children)
        self._idx = 0
        self._group_suffix = None
        if len(self._children) > 1:
            self._idx = int(self._children[1])

    @property
    def idx(self):
        return self._idx

    @property
    def symbol_char(self):
        return str(self._children[0])

    @property
    def group_suffix(self):
        return self._group_suffix

    @property
    def group_rule(self):
        if self._group_suffix is None:
            return GroupRule.NONE
        return self._group_suffix.rule

    @property
    def group_id(self):
        if self._group_suffix is None:
            return None
        return self._group_suffix.group_id

    @property
    def group_edge_values(self):
        """(group id, rule) as stored on generative-graph edges: -1 and 0 for a plain symbol."""
        if self._group_suffix is None:
            return (-1, 0)
        return (self._group_suffix.group_id, int(self._group_suffix.rule))

    def attach_group_suffix(self, group_suffix):
        self._group_suffix = group_suffix

    def generate_string(self, extension):
        string = super().generate_string(extension)
        if self.idx != 0:
            string += str(self.idx)
        if self._group_suffix is not None:
            string += self._group_suffix.generate_string(extension)
        return string

    def generable(self):
        return True

    def outer_conjugate(self, other):
        if self.idx != other.idx:
            return False

        self_str = self.symbol_char
        other_str = other.symbol_char

        if self_str == "$" and other_str == "$":
            return True

        return self_str in ("<", ">") and other_str in ("<", ">") and self_str != other_str

    def is_compatible(self, other):
        if other is None:
            return False
        if not isinstance(other, BondConnectorSymbolIdx):
            raise RuntimeError(f"Only BondConnectorSymbolIdx can be compared for compatibility. But 'other' is of type {type(other)}.")

        self_ladder = self.group_rule == GroupRule.LADDER
        # Ladder channels are rigid: they pair only with ladder channels.
        if self_ladder != (other.group_rule == GroupRule.LADDER):
            return False

        if not self.outer_conjugate(other):
            return False

        if self_ladder:
            return self._group_suffix.inner_symbol.is_compatible(other._group_suffix.inner_symbol)

        return True


class RuleKeyword(G2rinsBase):
    # Future group-rule keywords register here alongside the grammar rule alternatives.
    _KEYWORD_RULES = {"all": GroupRule.ALL}

    def __init__(self, children):
        super().__init__(children)
        self._keyword = str(self._children[0])

    @property
    def rule(self):
        return self._KEYWORD_RULES[self._keyword]

    def generate_string(self, extension):
        return self._keyword

    def generable(self):
        return True


class GroupSuffix(G2rinsBase):
    def __init__(self, children):
        super().__init__(children)
        self._inner_symbol = None
        self._rule_keyword = None
        self._group_id = 0
        for child in self._children:
            if isinstance(child, BondConnectorSymbolIdx):
                self._inner_symbol = child
            elif isinstance(child, RuleKeyword):
                self._rule_keyword = child
            elif isinstance(child, int):
                self._group_id = child

    @property
    def rule(self):
        if self._inner_symbol is not None:
            return GroupRule.LADDER
        if self._rule_keyword is not None:
            return self._rule_keyword.rule
        return GroupRule.EXCLUSION

    @property
    def group_id(self):
        return self._group_id

    @property
    def inner_symbol(self):
        return self._inner_symbol

    def generate_string(self, extension):
        string = "["
        if self._inner_symbol is not None:
            string += self._inner_symbol.generate_string(extension)
        elif self._rule_keyword is not None:
            string += self._rule_keyword.generate_string(extension)
        string += "]"
        if self._group_id != 0:
            string += str(self._group_id)
        return string

    def generable(self):
        return True


class BondConnectorGeneration(G2rinsBase):
    def __init__(self, children):
        super().__init__(children)
        self._bond_probabilities = None
        self._weight = 1.0
        if len(self._children) > 0:
            # Strip out the "|"
            parse = self._children[1:-1]
            self._weight = float(parse[0])
            if len(parse) > 1:
                self._bond_probabilities = [float(number) for number in parse]
                self._weight = sum(self.bond_probabilities)

    @property
    def bond_probabilities(self):
        return self._bond_probabilities

    @property
    def weight(self):
        return self._weight

    def generate_string(self, extension):
        if extension and (self.weight != 1.0 or self.bond_probabilities is not None):
            string = "|"
            if self.bond_probabilities:
                for trans in self.bond_probabilities:
                    string += str(trans) + " "
            else:
                string += str(self.weight)
            string = string.strip() + "|"
            return string
        return ""

    def generable(self):
        return True


class InnerBondConnector(G2rinsBase):
    def __init__(self, children):
        super().__init__(children)

        self._generation = BondConnectorGeneration([])
        self._symbol_list = []
        for child in self._children:
            if isinstance(child, BondConnectorSymbolIdx):
                self._symbol_list.append(child)
            if isinstance(child, BondConnectorGeneration):
                self._generation = child

    def generate_string(self, extension):
        string = ", ".join(symbol.generate_string(extension) for symbol in self._symbol_list)
        string += self._generation.generate_string(extension)
        return string

    def generable(self):
        return True

    @property
    def idx(self):
        idx_list = []
        for symbol in self._symbol_list:
            idx_list.append(symbol.idx)
        return idx_list

    @property
    def weight(self):
        return self._generation.weight

    @property
    def bond_probabilities(self):
        return self._generation.bond_probabilities


class BondConnector(G2rinsBase, GenerationBase):
    @classmethod
    def make(cls, text: str) -> Self:
        if "$" in text or "<" in text or ">" in text:
            return SimpleBondConnector.make(text)
        return TerminalBondConnector.make(text)

    @property
    def symbol(self):
        return None

    def is_compatible(self, other):
        if other is None:
            return False
        if not isinstance(other, BondConnector):
            raise RuntimeError(f"Only BondConnectors can be compared for compatibility. But 'other' is of type {type(other)}.")

        if self.symbol is None or other.symbol is None:
            return False
        return any(symbol.is_compatible(other_symbol) for symbol in self.symbol for other_symbol in other.symbol)

    def group_edge_attrs(self, other):
        """Group-rule edge attributes of every distinct compatible symbol pair (self = source), in parse order; empty when incompatible."""
        attrs = []
        for symbol in self.symbol or []:
            for other_symbol in other.symbol or []:
                if symbol.is_compatible(other_symbol):
                    entry = dict(zip(_GROUP_EDGE_ATTR, symbol.group_edge_values + other_symbol.group_edge_values))
                    if entry not in attrs:
                        attrs.append(entry)
        return attrs

    @property
    def bond_connectors(self):
        return [self]

    def _generate_partial_graph(self):
        g = nx.MultiDiGraph()
        node_idx = str(uuid.uuid4())
        g.add_node(node_idx, smi_text=str(self), obj=self)
        partial_graph = _PartialGraph(g)
        partial_graph.left_half_bonds.append(_HalfBond(self, node_idx, {}))
        partial_graph.right_half_bonds.append(_HalfBond(self, node_idx, {}))

        return partial_graph


class SimpleBondConnector(BondConnector):
    def __init__(self, children):
        super().__init__(children)
        for child in self._children:
            if isinstance(child, InnerBondConnector):
                self._inner_bond_connector = child

    @classmethod
    def make(cls, text: str) -> Self:
        # We use G2rinsBase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return G2rinsBase.make.__func__(cls, text)

    def generate_string(self, extension):
        return "[" + self._inner_bond_connector.generate_string(extension) + "]"

    def generable(self):
        return self._inner_bond_connector.generable

    @property
    def idx(self):
        return self._inner_bond_connector.idx

    @property
    def weight(self):
        return self._inner_bond_connector.weight

    @property
    def bond_probabilities(self):
        return self._inner_bond_connector.bond_probabilities

    @property
    def symbol(self):
        return self._inner_bond_connector._symbol_list


class TerminalBondConnectorList(G2rinsBase):
    def __init__(self, children):
        super().__init__(children)

        self.terminal_bond_connectors = [child for child in self._children if isinstance(child, TerminalBondConnector)]

    def generate_string(self, extension):
        string = "|".join(terminal_bond_connector.generate_string(extension) for terminal_bond_connector in self.terminal_bond_connectors)
        return string

    def generable(self):
        return True


class BondConnectorList(G2rinsBase, GenerationBase):
    def __init__(self, children):
        super().__init__(children)

        self._bond_connector_list = [child for child in self._children if isinstance(child, BondConnector)]

    def generate_string(self, extension):
        string = "|".join(bond_connector.generate_string(extension) for bond_connector in self._bond_connector_list)
        return string

    def generable(self):
        return True

    def _generate_partial_graph(self) -> _PartialGraph:

        partial_graph = self._bond_connector_list[0]._generate_partial_graph()
        matching_index = 0
        for node in partial_graph.g.nodes:
            partial_graph.g.nodes[node]["matching_index"] = matching_index

        for bond_connector in self._bond_connector_list[1:]:
            other_graph = bond_connector._generate_partial_graph()
            partial_graph.g = nx.compose(partial_graph.g, other_graph.g)
            partial_graph.left_half_bonds.extend(other_graph.left_half_bonds)
            partial_graph.right_half_bonds.extend(other_graph.right_half_bonds)
            matching_index += 1
            for node in other_graph.g.nodes:
                partial_graph.g.nodes[node]["matching_index"] = matching_index

        return partial_graph

    @property
    def bond_connectors(self):
        return self._bond_connector_list


class TerminalBondConnector(BondConnector):
    def __init__(self, children):
        super().__init__(children)

        self._inner_bond_connector = None

        for child in self._children:
            if isinstance(child, InnerBondConnector):
                self._inner_bond_connector = child

    @classmethod
    def make(cls, text: str) -> Self:
        # We use G2rinsBase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return G2rinsBase.make.__func__(cls, text)

    def generate_string(self, extension):
        try:
            return "[" + self._inner_bond_connector.generate_string(extension) + "]"
        except AttributeError:
            return "[]"

    def generable(self):
        return True

    @property
    def idx(self):
        try:
            return self._inner_bond_connector.idx
        except AttributeError:
            return None

    @property
    def weight(self):
        try:
            return self._inner_bond_connector.weight
        except AttributeError:
            return 1.0

    @property
    def bond_probabilities(self):
        try:
            return self._inner_bond_connector.bond_probabilities
        except AttributeError:
            return None

    @property
    def symbol(self):
        try:
            return self._inner_bond_connector._symbol_list
        except AttributeError:
            return None
