# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import copy
import warnings

import networkx as nx
import numpy as np

from .g2rins_molecule import G2rinsMolecule
from .bond import (
    BondConnector,
    TerminalBondConnectorList,
)
from .core import G2rinsBase, GenerationBase
from .distribution import StochasticGeneration
from .exception import (
    ConcatenatedBondConnectors,
    EmptyBondConnectorInTerminalBondConnectorList,
    EndGroupHasBondConnectors,
    IncorrectNumberOfBondProbabilities,
    MonomerHasTwoOrMoreBondConnectors,
    NoExplicitInitiation,
    NoExplicitTermination,
    NoInitiationForStochasticObject,
    StochasticMissingPath,
    UndefinedDistribution,
)
from .generative_graph import (
    _PROPAGATION_NAME,
    _TERMINATION_NAME,
    _TRANSITION_NAME,
    _HalfBond,
    _PartialGraph,
    is_static_edge,
)
from .smiles import Counterion, Smiles, _attach_counterions_to_partial_graph


class StochasticObject(G2rinsBase, GenerationBase):
    def __init__(self, children: list):
        super().__init__(children)
        self._repeat_residues: list = []
        self._initiation_residues: list = []
        self._termination_residues: list = []
        self._counterion_map: dict = {}
        self._left_terminal_bc_list: None | TerminalBondConnectorList = None
        self._right_terminal_bc_list: None | TerminalBondConnectorList = None

        self._generation: None | StochasticGeneration = None
        self._stochastic_parent: None | StochasticObject = None

        # Parse info
        initiation_separator_found = False
        termination_separator_found = False
        last_residue = None
        for child in self._children:
            if isinstance(child, TerminalBondConnectorList):
                if self._left_terminal_bc_list is None:
                    self._left_terminal_bc_list = child
                else:
                    if self._right_terminal_bc_list is not None:
                        raise ValueError(f"{self}, {self._children}, {self._right_terminal_bc_list}")
                    self._right_terminal_bc_list = child

            if str(child) == ";":
                if initiation_separator_found:
                    termination_separator_found = True
                else:
                    initiation_separator_found = True

            if isinstance(child, G2rinsMolecule) or isinstance(child, Smiles):
                if not initiation_separator_found:
                    self._repeat_residues.append(child)
                elif not termination_separator_found:
                    self._initiation_residues.append(child)
                else:
                    self._termination_residues.append(child)
                last_residue = child
            if isinstance(child, Counterion):
                self._counterion_map.setdefault(last_residue, []).append(child)
            if isinstance(child, StochasticGeneration):
                self._generation = child

        self._post_parse_validation()

        # Set the parent of child stochastic objects
        for child in self._children:
            try:
                child._set_stochastic_parent(self)
            except AttributeError:
                pass

    @property
    def stochastic_parent(self):
        return self._stochastic_parent

    def _set_stochastic_parent(self, parent):
        if self._stochastic_parent is not None:
            raise RuntimeError("A stochastic parent can only be set once. If this most likely a bug, please report it on github.")
        self._stochastic_parent = parent

    def _get_class(self, child, target_class):
        try:
            for grandchild in child._children:
                if isinstance(grandchild, target_class):
                    return grandchild  # found it
                result = self._get_class(grandchild, target_class)
                if result is not None:
                    return result
            return None
        except AttributeError:
            return None

    def _post_parse_validation(self):
        try:
            for element in [self] + self._repeat_residues + self._initiation_residues + self._termination_residues:
                graph_creator = element.get_graph_creator()
                generative_graph = graph_creator.get_generative_graph(include_bond_connectors=True)
                for node, data in generative_graph.nodes(data=True):
                    if node in graph_creator._bc_idx_set:
                        if data["stochastic_id_tree"][0] == data["stochastic_id_tree"][1]:
                            for _u, v in generative_graph.out_edges(node):
                                stochastic_id = generative_graph.nodes(data=True)[v]["stochastic_id_tree"][0]
                                if (v in graph_creator._bc_idx_set) and (data["stochastic_id_tree"][0] == stochastic_id):
                                    raise ConcatenatedBondConnectors(element, self)
        except UndefinedDistribution:
            pass

        for smi in self._repeat_residues:
            if len(smi.bond_connectors) < 2:
                raise MonomerHasTwoOrMoreBondConnectors(smi, self)
        for smi in self._initiation_residues + self._termination_residues:
            if len(smi.bond_connectors) < 1:
                raise EndGroupHasBondConnectors(smi, self)

        # Empty left bond connectors need end-groups to start initiation
        if self._left_terminal_bc_list.terminal_bond_connectors[0].symbol is None:
            if len(self._initiation_residues) < 1:
                warnings.warn(NoExplicitInitiation(self), stacklevel=1)

        if self._right_terminal_bc_list.terminal_bond_connectors[0].symbol is None:
            if len(self._termination_residues) < 1:
                warnings.warn(NoExplicitTermination(self), stacklevel=1)

        repeat_units_bond_connectors = []
        for element in self._repeat_residues:
            repeat_units_bond_connectors += element.bond_connectors

        for bc in self._left_terminal_bc_list.terminal_bond_connectors + self._right_terminal_bc_list.terminal_bond_connectors + repeat_units_bond_connectors:
            if bc.bond_probabilities is not None and len(bc.bond_probabilities) != len(repeat_units_bond_connectors):
                raise IncorrectNumberOfBondProbabilities(self, bc, len(repeat_units_bond_connectors))

        if len(self._left_terminal_bc_list.terminal_bond_connectors) > 1:
            if None in [bond_connector.symbol for bond_connector in self._left_terminal_bc_list.terminal_bond_connectors]:
                raise EmptyBondConnectorInTerminalBondConnectorList(self._left_terminal_bc_list, self)

        if len(self._right_terminal_bc_list.terminal_bond_connectors) > 1:
            if None in [bond_connector.symbol for bond_connector in self._right_terminal_bc_list.terminal_bond_connectors]:
                raise EmptyBondConnectorInTerminalBondConnectorList(self._right_terminal_bc_list, self)

    def _residue_string(self, residue, extension: bool) -> str:
        string = residue.generate_string(extension)
        for counterion in self._counterion_map.get(residue, []):
            string += counterion.generate_string(extension)
        return string

    def generate_string(self, extension: bool):
        string = "{" + self._left_terminal_bc_list.generate_string(extension) + " "
        if len(self._repeat_residues) > 0:
            string += self._residue_string(self._repeat_residues[0], extension)
            for residue in self._repeat_residues[1:]:
                string += ", " + self._residue_string(residue, extension)

        string += "; "

        if len(self._initiation_residues) > 0:
            string += self._residue_string(self._initiation_residues[0], extension)
            for residue in self._initiation_residues[1:]:
                string += ", " + self._residue_string(residue, extension)
        string += "; "

        if len(self._termination_residues) > 0:
            string += self._residue_string(self._termination_residues[0], extension)
            for residue in self._termination_residues[1:]:
                string += ", " + self._residue_string(residue, extension)

        string += " " + self._right_terminal_bc_list.generate_string(extension) + "}"

        if self._generation:
            string += self._generation.generate_string(extension)

        return string

    @property
    def bond_connectors(self):
        return []

    @property
    def stochastic_generation(self):
        return self._generation

    def _generate_partial_graph(self) -> _PartialGraph:

        if self.stochastic_generation is None:
            raise UndefinedDistribution(self)

        import matplotlib.pyplot as plt

        def quick_viz(G, title="Graph"):
            plt.figure(figsize=(12, 8))

            # Get layout
            pos = nx.spring_layout(G, k=1, iterations=50)

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")

            # Draw edges (for MultiDiGraph, this handles multiple edges automatically)
            nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, arrowstyle="->", connectionstyle="arc3,rad=0.1")  # Curves edges to show multiple

            # Draw labels using smi_text attribute
            labels = nx.get_node_attributes(G, "smi_text")
            nx.draw_networkx_labels(G, pos, labels, font_size=8)

            plt.axis("off")
            plt.tight_layout()
            plt.title(title)
            plt.show()

        def build_idx(residues, graph):
            """
            Build a list that maps uuid of all bond_connectors to their position in the string.
            The position is important for the bond probabilities.

            Example:
            -------
            build_idx(..)[3] gives you the uuid graph index of the 3rd bond connector in the stochastic element.

            """
            bond_connectors = []
            for res in residues:
                bond_connectors += res.bond_connectors
            bc_idx = [None] * len(bond_connectors)
            for bc in graph.nodes(data=True):
                bc_obj = bc[1]["obj"]
                if isinstance(bc_obj, BondConnector):
                    if bc_obj in bond_connectors:
                        bc_idx[bond_connectors.index(bc[1]["obj"])] = bc[0]
            return bc_idx

        def _connect(
            graph,
            first_idx,
            second_idx,
            full_idx,
            attr_name=_PROPAGATION_NAME,
            ignore_bond_probabilities=False,
            prints=False,
        ):
            non_connected_bc = []
            if prints:
                print("first index is: ", first_idx)
                print("target index is: ", second_idx)
            for bc_idx_a in first_idx:
                obj_a = graph.nodes[bc_idx_a]["obj"]
                if obj_a.bond_probabilities is not None and not ignore_bond_probabilities:
                    # Note that this spans the end groups
                    probabilities = obj_a.bond_probabilities
                else:
                    # This does not span the end groups
                    probabilities = [graph.nodes[bc_idx_b]["obj"].weight for bc_idx_b in second_idx]
                probabilities = np.asarray(probabilities)
                # Set weights to zero if bond are incompatible, note different lengths from above.
                if prints:
                    print("before checking compatibility, probabilities are ", probabilities)
                for i in range(len(probabilities)):
                    bc_idx_b = full_idx[i]
                    obj_b = graph.nodes[bc_idx_b]["obj"]
                    if prints:
                        print("target node is: ", graph.nodes[bc_idx_b])
                    if not obj_a.is_compatible(obj_b):
                        probabilities[i] = 0
                if prints:
                    print("after checking compatibility, probabilities are ", probabilities)
                # Normalizing probabilities
                if probabilities.sum() > 0:
                    probabilities /= probabilities.sum()
                # Assign bond attribute (stochastic, transition or termination)
                for i, prob in enumerate(probabilities):
                    if prob > 0:
                        bc_idx_b = full_idx[i]
                        graph.add_edge(bc_idx_a, bc_idx_b, **dict([(attr_name, prob)]))
                if prints:
                    print("At the end, probabilities are: ", probabilities)
                if sum(probabilities) == 0:
                    non_connected_bc.append(bc_idx_a)
            return graph, non_connected_bc

        def connect_monomers_to_monomers(graph, mono_idx_pos):
            return _connect(
                graph,
                mono_idx_pos,
                mono_idx_pos,
                mono_idx_pos,
                ignore_bond_probabilities=False,
                attr_name=_PROPAGATION_NAME,
            )

        def connect_monomers_to_terminators(graph, mono_idx_pos, end_idx_pos):
            result = _connect(
                graph,
                mono_idx_pos,
                terminator_idx_pos,
                terminator_idx_pos,
                ignore_bond_probabilities=True,
                attr_name=_TERMINATION_NAME,
            )
            return result

        def connect_initiators_to_monomers(graph, mono_idx_pos, end_idx_pos):
            return _connect(
                graph,
                end_idx_pos,
                mono_idx_pos,
                mono_idx_pos,
                ignore_bond_probabilities=False,
                attr_name=_TRANSITION_NAME,
                prints=False,
            )

        def connect_initiators_to_terminators(graph, initiator_idx_pos, terminator_idx_pos):
            return _connect(
                graph,
                initiator_idx_pos,
                terminator_idx_pos,
                terminator_idx_pos,
                ignore_bond_probabilities=True,
                attr_name=_TERMINATION_NAME,
                prints=False,
            )

        # Build graph without any connections between bond connectors.

        repeat_subgraphs = [monomer.get_graph_creator()._partial_graph for monomer in self._repeat_residues]
        initiator_subgraphs = [initiator.get_graph_creator()._partial_graph for initiator in self._initiation_residues]
        terminator_subgraphs = [terminator.get_graph_creator()._partial_graph for terminator in self._termination_residues]

        # Merge counterion fragments into their residue subgraphs before molar
        # amounts are stamped. Full residue text (incl. counterions) wins over
        # segment-level stamps; nodes of nested stochastic objects keep their
        # own inner residue text.
        for residues, subgraphs in ((self._repeat_residues, repeat_subgraphs), (self._initiation_residues, initiator_subgraphs), (self._termination_residues, terminator_subgraphs)):
            for residue, subgraph in zip(residues, subgraphs):
                if residue in self._counterion_map:
                    _attach_counterions_to_partial_graph(subgraph, self._counterion_map[residue], residue)
                residue_text = self._residue_string(residue, extension=True)
                for _node_idx, node_data in subgraph.g.nodes(data=True):
                    if "stochastic_obj" not in node_data:
                        node_data["token_text"] = residue_text

        # Add molar amount information
        for i, subgraph in enumerate(repeat_subgraphs):
            for node, data in subgraph.g.nodes(data=True):
                if "molar_amount" in data:
                    data["molar_amount"][id(self)] = self._repeat_residues[i].molar_amount
                else:
                    data["molar_amount"] = {id(self): self._repeat_residues[i].molar_amount}
        for i, subgraph in enumerate(initiator_subgraphs):
            for node, data in subgraph.g.nodes(data=True):
                if "molar_amount" in data:
                    data["molar_amount"][id(self)] = self._initiation_residues[i].molar_amount
                else:
                    data["molar_amount"] = {id(self): self._initiation_residues[i].molar_amount}
        for i, subgraph in enumerate(terminator_subgraphs):
            for node, data in subgraph.g.nodes(data=True):
                if "molar_amount" in data:
                    data["molar_amount"][id(self)] = self._termination_residues[i].molar_amount
                else:
                    data["molar_amount"] = {id(self): self._termination_residues[i].molar_amount}
        partial_graph = _PartialGraph(None)
        copy_of_initiator_subgraphs = copy.deepcopy(initiator_subgraphs)

        for gen_graph in repeat_subgraphs + initiator_subgraphs + terminator_subgraphs:
            gen_graph.left_half_bonds = []
            gen_graph.right_half_bonds = []
            partial_graph.merge(gen_graph, [])

        graph = partial_graph.g

        # List of monomer repeat unit bond connector IDX
        mono_idx_pos = build_idx(self._repeat_residues, graph)
        # Same for initiator units
        initiator_idx_pos = build_idx(self._initiation_residues, graph)

        # Same for terminator units
        terminator_idx_pos = build_idx(self._termination_residues, graph)

        bond_idx_to_initiator_idx = {}

        for node_idx in initiator_idx_pos:
            for initiator_subgraph in initiator_subgraphs:
                if node_idx in initiator_subgraph.g.nodes():
                    bond_idx_to_initiator_idx[node_idx] = initiator_subgraphs.index(initiator_subgraph)

        graph, _non_connected_bc = connect_monomers_to_monomers(graph, mono_idx_pos)

        # if self._right_terminal_bond_d.symbol is None:
        graph, _non_connected_bc = connect_monomers_to_terminators(graph, mono_idx_pos, terminator_idx_pos)

        initiator_non_connected_bc = []
        if len(self._left_terminal_bc_list.terminal_bond_connectors) == 1 and self._left_terminal_bc_list.terminal_bond_connectors[0].symbol is None:
            graph, initiator_non_connected_bc = connect_initiators_to_monomers(graph, mono_idx_pos, initiator_idx_pos)
            if len(initiator_non_connected_bc) > 0:
                graph, _non_connected_bc = connect_initiators_to_terminators(graph, initiator_non_connected_bc, terminator_idx_pos)

        # Add initiating bonds
        if self._left_terminal_bc_list.terminal_bond_connectors[0].symbol is None:
            weights = [graph.nodes[bc_idx]["obj"].weight for bc_idx in initiator_idx_pos]
            weights = np.asarray(weights)
            if len(weights) != len(initiator_idx_pos):
                raise RuntimeError(f"Implementation error, please report on GitHub https://github.com/depablolab/g2rins/issues . {weights} {initiator_idx_pos}")

            if weights.sum() == 0:
                weights += 1

            probabilities = weights / weights.sum()
            for i, prob in enumerate(probabilities):
                if prob > 0:
                    initiator_idx = bond_idx_to_initiator_idx[initiator_idx_pos[i]]
                    if "{" in self._initiation_residues[initiator_idx].generate_string(True):
                        for half_bond in copy_of_initiator_subgraphs[initiator_idx].left_half_bonds:
                            node = half_bond.node
                            node_idx = half_bond.node_id
                            prob_init = prob * half_bond.bond_attributes["init_weight"]
                            partial_graph.left_half_bonds.append(_HalfBond(node, node_idx, dict([(_TRANSITION_NAME, prob_init), ("init_weight", prob_init)])))
                    else:
                        node_idx = initiator_idx_pos[i]
                        node = graph.nodes[node_idx]["obj"]
                        partial_graph.left_half_bonds.append(_HalfBond(node, node_idx, dict([(_TRANSITION_NAME, prob), ("init_weight", prob)])))
        else:
            # TODO: loop over left_terminal_bc_list to connect things with all left bond connectors
            left_partial_graph = self._left_terminal_bc_list.terminal_bond_connectors[0]._generate_partial_graph()
            left_partial_graph.left_half_bonds = []
            left_partial_graph.right_half_bonds = []
            left_idx = list(left_partial_graph.g.nodes)[0]
            partial_graph.merge(left_partial_graph, [])
            partial_graph.left_half_bonds.append(_HalfBond(self._left_terminal_bc_list.terminal_bond_connectors[0], left_idx, {}))
            graph = partial_graph.g

            # With non-empty left bond connectors we connect first to one of the monomers inside.
            left_bc = self._left_terminal_bc_list.terminal_bond_connectors[0]
            if left_bc.bond_probabilities is not None:
                weights = left_bc.bond_probabilities[: len(mono_idx_pos)]
            else:
                weights = [graph.nodes[bc_idx]["obj"].weight for bc_idx in mono_idx_pos]
            weights = np.asarray(weights)

            for i, bc_idx in enumerate(mono_idx_pos):
                if not left_bc.is_compatible(graph.nodes[bc_idx]["obj"]):
                    weights[i] = 0

            probabilities = []
            if weights.sum() > 0:
                probabilities = weights / weights.sum()
            for i, prob in enumerate(probabilities):
                if prob > 0:
                    node_idx = mono_idx_pos[i]
                    graph.add_edge(left_idx, node_idx, **dict([(_TRANSITION_NAME, prob)]))

        # Add out-going bonds

        if self._right_terminal_bc_list.terminal_bond_connectors[0].symbol is not None:
            if len(self._right_terminal_bc_list.terminal_bond_connectors) > 1:
                matching_index = 0
            else:
                matching_index = -1

            for right_terminal_bond_connector in self._right_terminal_bc_list.terminal_bond_connectors:
                right_partial_graph = right_terminal_bond_connector._generate_partial_graph()
                right_partial_graph.left_half_bonds = []
                right_partial_graph.right_half_bonds = []
                right_idx = list(right_partial_graph.g.nodes)[0]
                partial_graph.merge(right_partial_graph, [])
                partial_graph.right_half_bonds.append(_HalfBond(right_terminal_bond_connector, right_idx, {}))

                graph = partial_graph.g

                graph.nodes[right_idx]["matching_index"] = matching_index
                matching_index += 1
                weights = []
                for i, bc_idx in enumerate(mono_idx_pos):
                    node = graph.nodes[bc_idx]["obj"]
                    weight = node.weight
                    if right_terminal_bond_connector.bond_probabilities is not None:
                        weight = right_terminal_bond_connector.bond_probabilities[i]
                    if not right_terminal_bond_connector.is_compatible(node):
                        weight = 0

                    weights.append(weight)

                for bc_idx in initiator_non_connected_bc:
                    node = graph.nodes[bc_idx]["obj"]
                    weight = node.weight
                    if not right_terminal_bond_connector.is_compatible(node):
                        weight = 0
                    weights.append(weight)

                weights = np.asarray(weights)
                probabilities = []
                if weights.sum() > 0:
                    probabilities = weights / weights.sum()
                full_bc_idx = mono_idx_pos + initiator_non_connected_bc

                for i, prob in enumerate(probabilities):
                    if prob > 0:
                        bc_idx = full_bc_idx[i]
                        graph.add_edge(bc_idx, right_idx, **dict([(_TRANSITION_NAME, prob)]))

        # Add mol weight distribution to all nodes
        for node_idx in partial_graph.g:
            if "stochastic_obj" not in graph.nodes[node_idx]:  # Nested objects have that already
                graph.nodes[node_idx]["stochastic_obj"] = self

        self._post_validate_partial_graph(partial_graph, mono_idx_pos + initiator_idx_pos + terminator_idx_pos)

        return partial_graph

    def _post_validate_partial_graph(self, partial_graph, bc_idx):

        if len(partial_graph.left_half_bonds) == 0:
            # Only reachable with an empty '[]' left terminal: a declared left
            # symbol always creates a left half-bond (unconditional append in
            # _generate_partial_graph), and a declared entry that generation
            # cannot use is reported by StochasticMissingPath below.
            warnings.warn(NoInitiationForStochasticObject(self, partial_graph), stacklevel=1)

        # The right side needs no warning at all: a declared right terminal
        # symbol always creates a right half-bond (unconditional append in
        # _generate_partial_graph), so an empty right_half_bonds list only ever
        # means a deliberately closed '[]' object (whose missing-terminator
        # case NoExplicitTermination covers). A declared exit that generation
        # cannot reach is caught by the StochasticMissingPath check below.



        # To successfully generate molecules, there needs to be path from each entry point (left half bonds)
        # to at least one end, right half bonds
        # Reachability must follow generation direction: STATIC covalent bonds are
        # walkable both ways in the molecule but exist only in SMILES string order
        # at this stage (reverse duplicates are added later in the generating
        # graph), so they are mirrored here; non-static (stochastic/transition/
        # termination) edges are only ever traversed forward by the sampler, so
        # mirroring them too (a fully undirected view) hid direction-dependent
        # dead ends such as asymmetric zero bond-probability vectors.
        if len(partial_graph.right_half_bonds) > 0:
            target_idx = set([rhb.node_id for rhb in partial_graph.right_half_bonds])
            source_idx = set([lhb.node_id for lhb in partial_graph.left_half_bonds])

            reachability_graph = partial_graph.g.copy()
            for u, v, d in partial_graph.g.edges(data=True):
                if is_static_edge(d):
                    reachability_graph.add_edge(v, u)
            for source in source_idx:
                tree = nx.dfs_tree(reachability_graph, source=source)
                reachable_nodes = set(tree.nodes)

                if target_idx.isdisjoint(reachable_nodes):
                    warnings.warn(
                        StochasticMissingPath(self, partial_graph.g.nodes[source]["obj"]),
                        stacklevel=1,
                    )


"""Deprecated with the grammar based G2RINS, use StochasticObject instead."""
Stochastic = StochasticObject
