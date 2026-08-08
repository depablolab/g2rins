# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import uuid

import networkx as nx

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .core import G2rinsBase, GenerationBase
from .generative_graph import _HalfBond, _PartialGraph


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
        if len(self._children) > 1:
            self._idx = int(self._children[1])

    @property
    def idx(self):
        return self._idx

    def generate_string(self, extension):
        string = super().generate_string(extension)
        if self.idx != 0:
            string += str(self.idx)
        return string

    def generable(self):
        return True

    def is_compatible(self, other):
        if other is None:
            return False
        if not isinstance(other, BondConnectorSymbolIdx):
            raise RuntimeError(f"Only BondConnectorSymbolIdx can be compared for compatibility. But 'other' is of type {type(other)}.")

        if self.idx != other.idx:
            return False

        self_str = str(self._children[0])
        other_str = str(other._children[0])

        if self_str == "$" and other_str == "$":
            return True

        if self_str in ("<", ">") and other_str in ("<", ">") and self_str != other_str:
            return True

        return False


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
