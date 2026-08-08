# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import warnings
from itertools import product

from .g2rins_molecule import _AbstractIterativeGenerativeClass
from .bond import BondSymbol, RingBond
from .core import G2rinsBase, GenerationBase
from .exception import (
    CounterionChargeImbalance,
    DoubleBondSymbolDefinition,
    SmilesHasNonZeroBondConnectors,
    UnmatchedCounterion,
)
from .generative_graph import _BOND_TYPE_NAME, _PartialGraph


class Branch(G2rinsBase, GenerationBase):
    def __init__(self, children: list):
        super().__init__(children)

        self._bond_symbol: None | BondSymbol = None
        self._elements: list = []

        for child in self._children:
            if self._bond_symbol is None and isinstance(child, BondSymbol):
                self._bond_symbol = child
            elif isinstance(child, G2rinsBase):
                self._elements.append(child)

    def _set_stochastic_parent(self, parent):
        for child in self._children:
            try:
                child._set_stochastic_parent(parent)
            except AttributeError:
                pass

    @property
    def bond_symbol(self):
        return self._bond_symbol

    @property
    def generable(self):
        gen = True
        for element in self._elements:
            gen = gen and element.generable
        if self.bond_symbol is not None:
            gen = gen and self.bond_symbol.generable
        return gen

    def generate_string(self, extension: bool) -> str:
        string = "("
        if self.bond_symbol is not None:
            string += self.bond_symbol.generate_string(extension)
        for element in self._elements:
            string += element.generate_string(extension)
        return string + ")"

    def _generate_partial_graph(self) -> _PartialGraph:
        partial_graph = self._elements[0]._generate_partial_graph()
        if self._bond_symbol is not None:
            for lhb in partial_graph.left_half_bonds:
                if _BOND_TYPE_NAME in lhb.bond_attributes:
                    raise DoubleBondSymbolDefinition(partial_graph, self._bond_symbol, lhb.bond_attributes)
                lhb.bond_attributes[_BOND_TYPE_NAME] = self._bond_symbol

        for element in self._elements[1:]:
            element_partial_graph = element._generate_partial_graph()
            bonds_to_add = product(partial_graph.right_half_bonds, element_partial_graph.left_half_bonds)
            # Transfer right_half_bonds to new partial graph
            partial_graph.right_half_bonds = element_partial_graph.right_half_bonds
            element_partial_graph.right_half_bonds = []
            element_partial_graph.left_half_bonds = []

            partial_graph.merge(element_partial_graph, bonds_to_add)

        # A branch cannot connect to anything on the right
        partial_graph.right_half_bonds = []
        return partial_graph

    @property
    def bond_connectors(self):
        bond_connectors = []
        for element in self._elements:
            bond_connectors += element.bond_connectors
        return bond_connectors


class BranchedAtom(G2rinsBase, GenerationBase):
    def __init__(self, children):
        super().__init__(children)

        self._atom_stand_in: G2rinsBase | None = None
        self._branches: list[Branch] = []
        self._ring_bonds: list[RingBond] = []

        for child in self._children:
            if self._atom_stand_in is None and isinstance(child, G2rinsBase):
                self._atom_stand_in = child
            if isinstance(child, RingBond):
                self._ring_bonds.append(child)
            if isinstance(child, Branch):
                self._branches.append(child)

    def _set_stochastic_parent(self, parent):
        for child in self._children:
            try:
                child._set_stochastic_parent(parent)
            except AttributeError:
                pass

    @property
    def generable(self):
        gen = self._atom_stand_in.generable
        for ring_bond in self._ring_bonds:
            gen = gen and ring_bond.generable
        for branch in self._branches:
            gen = gen and branch.generable

        return gen

    def generate_string(self, extension: bool) -> str:
        string = self._atom_stand_in.generate_string(extension)
        for ring_bond in self._ring_bonds:
            string += ring_bond.generate_string(extension)
        for branch in self._branches:
            string += branch.generate_string(extension)
        return string

    def _generate_partial_graph(self) -> _PartialGraph:
        partial_graph = self._atom_stand_in._generate_partial_graph()
        # Adding ring bonds
        for ring_idx, half_bond in product(self._ring_bonds, partial_graph.right_half_bonds):
            partial_graph.add_ring_bond(ring_idx, half_bond)

        # Adding branches
        for branch in self._branches:
            branch_partial_graph = branch._generate_partial_graph()
            bonds_to_add = product(partial_graph.right_half_bonds, branch_partial_graph.left_half_bonds)
            # Branches have empty right hand half bonds, so only resetting left ones.
            branch_partial_graph.left_half_bonds = []

            partial_graph.merge(branch_partial_graph, bonds_to_add)

        # Not resetting right bonds, because this can bond to more on the right (not a branch)
        return partial_graph

    @property
    def bond_connectors(self):
        bond_connectors = []

        if self._atom_stand_in:
            bond_connectors += self._atom_stand_in.bond_connectors
        for branch in self._branches:
            bond_connectors += branch.bond_connectors

        return bond_connectors


class AtomAssembly(G2rinsBase, GenerationBase):
    def __init__(self, children: list):
        super().__init__(children)
        self._symbol: None | BondSymbol = None
        self._branched_atom: None | BranchedAtom = None

        for child in self._children:
            if isinstance(child, BondSymbol):
                self._symbol = child
            elif isinstance(child, BranchedAtom):
                self._branched_atom = child

    def _set_stochastic_parent(self, parent):
        self._branched_atom._set_stochastic_parent(parent)

    @property
    def bond_symbol(self) -> None | BondSymbol:
        return self._symbol

    @property
    def generable(self) -> bool:
        gen = self._branched_atom.generable
        if self.bond_symbol is not None:
            gen = gen and self.bond_symbol.generable
        return gen

    def generate_string(self, extension: bool) -> str:
        string = ""
        if self.bond_symbol is not None:
            string += self.bond_symbol.generate_string(extension)
        string += self._branched_atom.generate_string(extension)
        return string

    def _generate_partial_graph(self) -> _PartialGraph:
        partial_graph = self._branched_atom._generate_partial_graph()
        if self.bond_symbol:
            for half_bond in partial_graph.left_half_bonds:
                if _BOND_TYPE_NAME in half_bond.bond_attributes:
                    raise DoubleBondSymbolDefinition(partial_graph, self.bond_symbol, half_bond.bond_attributes)
                half_bond.bond_attributes[_BOND_TYPE_NAME] = self.bond_symbol

        return partial_graph

    @property
    def bond_connectors(self) -> list:
        return self._branched_atom.bond_connectors


class CounterionBranchedAtom(BranchedAtom):
    pass


class CounterionAssembly(AtomAssembly):
    pass


class Counterion(G2rinsBase, GenerationBase):
    def __init__(self, children: list):
        super().__init__(children)
        self._elements: list = [child for child in self._children if isinstance(child, G2rinsBase)]
        for element in self._elements:
            if len(element.bond_connectors) != 0:
                raise SmilesHasNonZeroBondConnectors(element)

    @property
    def generable(self) -> bool:
        gen = True
        for element in self._elements:
            gen = gen and element.generable
        return gen

    def generate_string(self, extension: bool) -> str:
        string = "."
        for element in self._elements:
            string += element.generate_string(extension)
        return string

    def _generate_partial_graph(self) -> _PartialGraph:
        partial_graph = self._elements[0]._generate_partial_graph()
        for element in self._elements[1:]:
            element_partial_graph = element._generate_partial_graph()
            bonds_to_add = product(partial_graph.right_half_bonds, element_partial_graph.left_half_bonds)
            partial_graph.right_half_bonds = element_partial_graph.right_half_bonds
            element_partial_graph.right_half_bonds = []
            element_partial_graph.left_half_bonds = []
            partial_graph.merge(element_partial_graph, bonds_to_add)
        # Counterion fragments never bond covalently to anything.
        partial_graph.left_half_bonds = []
        partial_graph.right_half_bonds = []
        return partial_graph


def _attach_counterions_to_partial_graph(partial_graph, counterions, residue):
    """Merge counterion fragments into a repeat-unit graph, adding bond_type "." association edges by greedy charge matching."""

    def _node_charge(node_id):
        return getattr(partial_graph.g.nodes[node_id]["obj"], "charge", None)

    unit_nodes = list(partial_graph.g.nodes())
    total_charge = sum(_node_charge(node_id) or 0 for node_id in unit_nodes)
    site_supply = {node_id: abs(_node_charge(node_id)) for node_id in unit_nodes if _node_charge(node_id)}
    first_atom = next(node_id for node_id in unit_nodes if _node_charge(node_id) is not None)

    for counterion in counterions:
        ion_graph = counterion._generate_partial_graph()
        ion_nodes = list(ion_graph.g.nodes())
        ion_charge = {node_id: getattr(ion_graph.g.nodes[node_id]["obj"], "charge", None) for node_id in ion_nodes}
        demand = sum(charge or 0 for charge in ion_charge.values())
        total_charge += demand
        anchor = next((node_id for node_id in ion_nodes if ion_charge[node_id]), ion_nodes[0])
        partial_graph.merge(ion_graph, [])

        remaining = abs(demand)
        num_edges = 0
        for site in site_supply:
            if remaining == 0:
                break
            if site_supply[site] > 0 and _node_charge(site) * demand < 0:
                transfer = min(remaining, site_supply[site])
                partial_graph.g.add_edge(site, anchor, **{_BOND_TYPE_NAME: "."})
                site_supply[site] -= transfer
                remaining -= transfer
                num_edges += 1
        if num_edges == 0:
            partial_graph.g.add_edge(first_atom, anchor, **{_BOND_TYPE_NAME: "."})
            if demand != 0:
                warnings.warn(UnmatchedCounterion(counterion), stacklevel=1)

    if total_charge != 0:
        warnings.warn(CounterionChargeImbalance(residue, total_charge), stacklevel=1)


class MolarAmount(G2rinsBase, GenerationBase):
    def __init__(self, children):
        super().__init__(children)
        self._amount: float = 1.0
        len_children = len(self._children)
        for child in self._children[1:len_children-1]:
            if isinstance(child,float):
                self._amount = float(child)

    def amount(self):
        return self._amount

    def generate_string(self, extension: bool) -> str:
        return f"|{self._amount}|"

    def generable(self) -> bool:
        return True

    def _generate_partial_graph(self) -> _PartialGraph:
        return _PartialGraph()

class Dot(G2rinsBase, GenerationBase):
    @property
    def generable(self):
        return True

    def generate_string(self, extension: bool) -> str:
        return "."

    def _generate_partial_graph(self) -> _PartialGraph:
        return _PartialGraph()


class Smiles(_AbstractIterativeGenerativeClass):
    def __init__(self, children):
        super().__init__(children)

    @property
    def molar_amount(self) -> float:
        for child in self._children:
            if isinstance(child, MolarAmount):
                return child.amount()
        return 1.0

    def _generate_partial_graph(self) -> _PartialGraph:
        partial_graph = super()._generate_partial_graph()
        # Outermost Smiles wins (overwrites nested branch stamps), but nodes of
        # nested stochastic objects keep the text of their own inner residue.
        text = str(self)
        for _node_idx, node_data in partial_graph.g.nodes(data=True):
            if "stochastic_obj" not in node_data:
                node_data["token_text"] = text
        return partial_graph
