# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

from itertools import product

import networkx as nx

from .core import G2rinsBase, GenerationBase
from .exception import ParsingError, SmilesHasNonZeroBondConnectors
from .generative_graph import _PartialGraph


class _AbstractIterativeClass(G2rinsBase):
    @property
    def generable(self):
        gen = True
        for child in self._children:
            gen = gen and child.generable
        return gen

    def _set_stochastic_parent(self, parent):
        for child in self._children:
            try:
                child._set_stochastic_parent(parent)
            except AttributeError:
                pass

    def generate_string(self, extension: bool) -> str:
        string = ""
        for child in self._children:
            string += child.generate_string(extension)
        return string

    @property
    def bond_connectors(self) -> list:
        bond_connectors = []
        for child in self._children:
            bond_connectors += child.bond_connectors
        return bond_connectors

    @property
    def molar_amount(self) -> float:
        return 1.0

            
class _AbstractIterativeGenerativeClass(_AbstractIterativeClass, GenerationBase):
    def _generate_partial_graph(self) -> _PartialGraph:
        partial_graph = _PartialGraph()
        if len(self._children) > 0:
            partial_graph = self._children[0]._generate_partial_graph()

            for child in self._children[1:]:
                child_partial_graph = child._generate_partial_graph()
                bonds_to_add = product(partial_graph.right_half_bonds, child_partial_graph.left_half_bonds)
                # Transfer the child right bond to the partial graph, and reset partial graph
                partial_graph.right_half_bonds = child_partial_graph.right_half_bonds
                child_partial_graph.right_half_bonds = []
                child_partial_graph.left_half_bonds = []

                partial_graph.merge(child_partial_graph, bonds_to_add)
        return partial_graph


class G2rinsMolecule(_AbstractIterativeGenerativeClass):
    def __init__(self, children: list):
        super().__init__(children)

        self._dot_generation: None | DotGeneration = None
        for child in self._children:
            if isinstance(child, DotGeneration):
                if self._dot_generation is not None:
                    raise ParsingError(self)
                self._dot_generation = child

        self._post_parse_validation()

    def _post_parse_validation(self):
        for child in self._children:
            if len(child.bond_connectors) != 0:
                raise SmilesHasNonZeroBondConnectors(child)

    @property
    def mol_molecular_weight(self) -> float | None:
        if self._dot_generation:
            return self._dot_generation.molecular_weight

    def _generate_partial_graph(self) -> _PartialGraph:
        partial_graph = super()._generate_partial_graph()
        if self.mol_molecular_weight is not None:
            nx.set_node_attributes(partial_graph.g, values=self.mol_molecular_weight, name="mol_molecular_weight")

        # Remove previous assignments of init weights
        for node_idx in list(partial_graph.g.nodes()):
            try:
                del partial_graph.g.nodes[node_idx]["init_weight"]
            except KeyError:
                pass

        # set weights, if no weight is provided we use 1 as a positive fill
        init_weight = 1
        if self.mol_molecular_weight is not None:
            if len(partial_graph.left_half_bonds) > 0:
                # Divide by length of possible entry points
                init_weight = self.mol_molecular_weight / len(partial_graph.left_half_bonds)

        # Open left half bonds are entry points
        for half_bond in partial_graph.left_half_bonds:
            try:
                partial_graph.g.nodes[half_bond.node_id]["init_weight"] = half_bond.bond_attributes["init_weight"]
            except KeyError:
                partial_graph.g.nodes[half_bond.node_id]["init_weight"] = init_weight

        return partial_graph


class G2rins(_AbstractIterativeGenerativeClass):

    @property
    def mol_molecular_weight_map(self) -> dict[G2rinsMolecule, float | None]:
        return {mol: mol.mol_molecular_weight for mol in self._children}

    @property
    def total_molecular_weight(self) -> None | float:
        total_mol_weight: float = 0.0
        for molw in self.mol_molecular_weight_map.values():
            if molw is not None:
                total_mol_weight += molw
        if total_mol_weight > 0:
            return total_mol_weight
        return None

    def _generate_partial_graph(self) -> _PartialGraph:
        partial_graph = super()._generate_partial_graph()
        if self.total_molecular_weight is not None:
            for node_idx in list(partial_graph.g.nodes()):
                partial_graph.g.nodes[node_idx]["total_molecular_weight"] = self.total_molecular_weight
        return partial_graph

    @property
    def num_mol_species(self):
        return len(self._children)


class DotGeneration(_AbstractIterativeGenerativeClass):
    def __init__(self, children):
        from .smiles import Dot

        super().__init__(children)

        self._dot: None | Dot = None
        self._dot_system_size: None | DotSystemSize = None
        for child in self._children:
            if isinstance(child, Dot):
                self._dot = child
            if isinstance(child, DotSystemSize):
                self._dot_system_size = child

    @property
    def molecular_weight(self):
        if self._dot_system_size is not None:
            return self._dot_system_size.molecular_weight
        return 0.0


class DotSystemSize(G2rinsBase, GenerationBase):
    def __init__(self, children: list):
        super().__init__(children)

        self._molecular_weight: float = -1.0
        for child in self._children:
            if isinstance(child, float):
                if self._molecular_weight >= 0:
                    raise ValueError("Internal Error, please report on github.com")
                self._molecular_weight = child

    @property
    def molecular_weight(self) -> float:
        return self._molecular_weight

    def generate_string(self, extension: bool) -> str:
        string = ""
        if extension:
            string += f"|{self.molecular_weight}|"
        return string

    def _generate_partial_graph(self) -> _PartialGraph:
        return _PartialGraph()
