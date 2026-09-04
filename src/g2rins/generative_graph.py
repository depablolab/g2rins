# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import json
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

import networkx as nx
from scipy.stats import rankdata

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .chem_resource import (
    atom_color_mapping,
    atom_name_mapping,
    atom_name_num,
    smi_bond_mapping,
)
from .exception import (
    GroupRulesOnBothPathEnds,
    IncompatibleBondTypeBondConnector,
    TooManyStochasticObjects,
)
from .util import _determine_darkness_from_hex

_PROPAGATION_NAME = "propagation_weight"
_TERMINATION_NAME = "termination_weight"
_TRANSITION_NAME = "transition_weight"
_STATIC_NAME = "static"
_EDGE_STOCHASTIC_ID_NAME = "stochastic_id"
_AROMATIC_NAME = "aromatic"
_BOND_TYPE_NAME = "bond_type"
_NON_STATIC_ATTR = (_PROPAGATION_NAME, _TERMINATION_NAME, _TRANSITION_NAME)
_SOURCE_GROUP_NAME = "source_group"
_SOURCE_RULE_NAME = "source_rule"
_TARGET_GROUP_NAME = "target_group"
_TARGET_RULE_NAME = "target_rule"
# Group-rule attributes carried by EVERY edge (fixed schema); -1 = no group, 0 = GroupRule.NONE.
_GROUP_EDGE_ATTR = (_SOURCE_GROUP_NAME, _SOURCE_RULE_NAME, _TARGET_GROUP_NAME, _TARGET_RULE_NAME)
_GROUP_EDGE_SENTINELS = (-1, 0, -1, 0)
_STOCHASTIC_TREE_DEPTH = 10


def is_static_edge(edge_data):
    if _STATIC_NAME in edge_data:
        return edge_data[_STATIC_NAME]
    weight = 0
    for attr in _NON_STATIC_ATTR:
        if attr in edge_data:
            weight += edge_data[attr]
    return not weight > 0


_DERIVED_NODE_FIELDS = ("unit_id", "bond_id")


@dataclass(frozen=True)
class UnitLabels:
    """Derived unit/bond annotations of a generative graph (see :func:`derive_unit_labels`)."""

    unit_id: dict
    bond_id: dict


def derive_unit_labels(generative_graph, node_sort_key=None):
    """
    Derive a ``unit_id`` for every node and a ``bond_id`` for every connection
    atom of a generative graph (see :meth:`GraphCreator.get_generative_graph`).

    The labels are annotations computed on demand, never stored on the graph.
    The derivation reads only properties every generative graph carries — the static
    edge flag, the non-static weights, ``init_weight`` and the node order — so
    it works on parsed and generated graphs alike.

    A *unit* is a maximal group of nodes connected through static bonds; units
    are joined to one another only by non-static (propagation / transition /
    termination) bonds. Each unit is classified by its role in the generation
    and labelled accordingly:

    - Initiators   ``I{n}``: positive init_weight (entry points of the
      generation); they only have outgoing non-static edges.
    - Repeat units ``R{n}``: init_weight == -1, with a propagation path back
      to the unit: a walk that leaves the unit through one connection atom
      and returns through another, using propagation bonds as the only
      connections between units.
    - Linkers      ``L{n}``: init_weight == -1, with an outgoing non-static
      edge but no propagation path back to the unit; they join blocks without
      being stochastically repeated.
    - Terminators  ``T{n}``: init_weight == -1, reached by incoming non-static
      edges but with no outgoing non-static edge.

    The path test considers only the bonds a generation traversal can form
    (the rule that hides unused bonds in
    :meth:`GraphCreator.get_dot_string`); a graph with no initiator keeps
    every bond.

    Initiators and terminators are numbered by their first atom in node order.
    Repeat units and linkers are numbered by *generation*: a breadth-first
    traversal of the unit-level graph along non-static edges starting from the
    initiators, so units adjacent to an initiator get the lowest numbers, the
    next shell the following ones, and so on (ties broken by node order);
    units not reachable from any initiator fall to the end.

    A *connection atom* is a node with at least one incident non-static edge.
    Its ``bond_id`` labels the atom's open-bond position within its unit;
    connection atoms are numbered 1, 2, ... per unit in node order, matching
    the ``[*:n]`` star map numbers of the unit's P-SMILES.

    ``node_sort_key`` orders the nodes for the numbering. The default is node
    insertion order — parse order on parsed graphs, emission order on
    generated ones — which is reproducible while node ids (UUIDs) are not.

    Returns a :class:`UnitLabels` with ``unit_id`` (node -> e.g. ``"R0"``) and
    ``bond_id`` (connection node -> int).
    """
    if node_sort_key is None:
        _node_order = {node: index for index, node in enumerate(generative_graph.nodes())}

        def node_sort_key(graph_node):
            return _node_order[graph_node]

    static_unit_graph = nx.Graph()
    static_unit_graph.add_nodes_from(generative_graph.nodes())
    for u, v, d in generative_graph.edges(data=True):
        if is_static_edge(d):
            static_unit_graph.add_edge(u, v)

    # `units[i]` is the sorted node list of unit i; `node_to_unit` maps every
    # node to its unit index. The unit order is deterministic (by node order).
    units = sorted(
        (sorted(component, key=node_sort_key) for component in nx.connected_components(static_unit_graph)),
        key=lambda component: node_sort_key(component[0]),
    )
    node_to_unit = {n: i for i, unit_nodes in enumerate(units) for n in unit_nodes}

    # Propagation bonds the generation can form, as unordered connection-atom
    # pairs (mirrored directions collapse; without initiators nothing prunes).
    used_edges = GraphCreator._used_nonstatic_edges(generative_graph, node_to_unit)
    propagation_bonds = {
        frozenset((u, v))
        for u, v, k, d in generative_graph.edges(keys=True, data=True)
        if d.get(_PROPAGATION_NAME, 0) > 0 and (used_edges is None or (u, v, k) in used_edges)
    }

    # Walk states per connection atom: "bonded" = arrived through a bond,
    # "crossed" = arrived through the unit interior. A propagation path back
    # to a unit exists iff one of its bonded->crossed crossings lies on a
    # cycle of bond steps alternating with distinct-atom unit crossings.
    walk_graph = nx.DiGraph()
    unit_ports = defaultdict(set)
    for bond in propagation_bonds:
        pair = tuple(bond)
        walk_graph.add_edge((pair[0], "crossed"), (pair[-1], "bonded"))
        walk_graph.add_edge((pair[-1], "crossed"), (pair[0], "bonded"))
        for port in bond:
            unit_ports[node_to_unit[port]].add(port)
    for ports in unit_ports.values():
        for entry in ports:
            for exit_port in ports:
                if entry != exit_port:
                    walk_graph.add_edge((entry, "bonded"), (exit_port, "crossed"))

    state_component = {}
    for component_index, component in enumerate(nx.strongly_connected_components(walk_graph)):
        for state in component:
            state_component[state] = component_index

    repeated_units = {
        unit_index
        for unit_index, ports in unit_ports.items()
        for entry in ports
        for exit_port in ports
        if entry != exit_port and state_component[(entry, "bonded")] == state_component[(exit_port, "crossed")]
    }

    # Classify every unit by its role in the generation.
    unit_role = []
    for unit_index, unit_nodes in enumerate(units):
        is_initiator = any(generative_graph.nodes[n]["init_weight"] > 0 for n in unit_nodes)
        has_outgoing_non_static = any(
            not is_static_edge(d) for n in unit_nodes for _u, _v, d in generative_graph.out_edges(n, data=True)
        )
        if is_initiator:
            unit_role.append("I")
        elif not has_outgoing_non_static:
            unit_role.append("T")
        elif unit_index in repeated_units:
            unit_role.append("R")
        else:
            unit_role.append("L")

    # Directed unit-level graph over non-static edges: an edge u_unit -> v_unit
    # for every non-static bond between two different units. The BFS distance
    # from the initiator units along this graph is the unit's "generation".
    unit_digraph = nx.DiGraph()
    unit_digraph.add_nodes_from(range(len(units)))
    for u, v, d in generative_graph.edges(data=True):
        if not is_static_edge(d):
            source_unit, target_unit = node_to_unit[u], node_to_unit[v]
            if source_unit != target_unit:
                unit_digraph.add_edge(source_unit, target_unit)

    # Multi-source breadth-first search: every initiator unit starts at
    # generation 0 and each shell of successors gets the next number.
    initiator_units = [i for i, role in enumerate(unit_role) if role == "I"]
    generation = {}
    if initiator_units:
        for depth, layer in enumerate(nx.bfs_layers(unit_digraph, initiator_units)):
            for unit_index in layer:
                generation[unit_index] = depth

    def _generation_then_atom(repeat_idx):
        return generation.get(repeat_idx, float("inf")), node_sort_key(units[repeat_idx][0])

    ordered_by_role = {
        "I": sorted((i for i, r in enumerate(unit_role) if r == "I"), key=lambda i: node_sort_key(units[i][0])),
        "R": sorted((i for i, r in enumerate(unit_role) if r == "R"), key=_generation_then_atom),
        "L": sorted((i for i, r in enumerate(unit_role) if r == "L"), key=_generation_then_atom),
        "T": sorted((i for i, r in enumerate(unit_role) if r == "T"), key=lambda i: node_sort_key(units[i][0])),
    }

    unit_id_by_index = {}
    for prefix, unit_indexes in ordered_by_role.items():
        for counter, unit_index in enumerate(unit_indexes):
            unit_id_by_index[unit_index] = f"{prefix}{counter}"

    unit_id = {}
    for unit_index, unit_nodes in enumerate(units):
        for n in unit_nodes:
            unit_id[n] = unit_id_by_index[unit_index]

    connection_nodes = {n for u, v, d in generative_graph.edges(data=True) if not is_static_edge(d) for n in (u, v)}
    bond_id = {}
    for unit_nodes in units:
        counter = 1
        for n in unit_nodes:
            if n in connection_nodes:
                bond_id[n] = counter
                counter += 1

    return UnitLabels(unit_id=unit_id, bond_id=bond_id)


def generative_graph_json_data(generative_graph):
    """
    JSON-serializable dict for a generative graph: a self-describing ``format`` block
    plus the node-link ``graph`` (:func:`networkx.node_link_data`) with the
    derived annotations of :func:`derive_unit_labels` injected into each node
    dict (``unit_id`` on every node, ``bond_id`` on connection atoms).

    Load the graph back with
    ``networkx.node_link_graph(data["graph"], edges="edges")``; strip the
    ``derived_node_fields`` before ML training.
    """
    labels = derive_unit_labels(generative_graph)
    data = nx.node_link_data(generative_graph, edges="edges")
    for node_dict in data["nodes"]:
        node = node_dict["id"]
        node_dict["unit_id"] = labels.unit_id[node]
        if node in labels.bond_id:
            node_dict["bond_id"] = labels.bond_id[node]
    return {
        "format": {
            "version": 1,
            "derived_node_fields": list(_DERIVED_NODE_FIELDS),
            "note": (
                "unit_id and bond_id are derived annotations injected at export, not stored graph"
                " attributes; a bond id labels the atom's open-bond position within its unit and"
                " matches the [*:n] star map of the unit's P-SMILES. Strip derived_node_fields"
                " before ML training."
            ),
        },
        "graph": data,
    }


class _HalfBond:
    def __init__(self, node, node_id: str, bond_attributes: dict):
        self.node = node
        self.node_id = node_id
        self.bond_attributes = bond_attributes

    def __str__(self):
        return f"HalfBond({str(self.node)}, {self.node_id}, {self.bond_attributes})"


class _PartialGraph:

    def __init__(self, g: None | nx.MultiDiGraph = None):
        if g is None:
            g = nx.MultiDiGraph()
        self.g = g

        self.left_half_bonds: list[_HalfBond] = []
        self.right_half_bonds: list[_HalfBond] = []
        self.ring_bond_map: dict[int, _HalfBond] = {}

    def merge(self, other: Self, half_bond_tuples: list[tuple[_HalfBond, _HalfBond]]) -> Self:
        """
        Strictly only merges the graphs, handling of left/right bond halves has to be performed before hand.
        It does handle the merging of ring bonds though.

        """

        if len(other.left_half_bonds) != 0:
            raise ValueError(other.left_half_bonds)
        if len(other.right_half_bonds) != 0:
            raise ValueError(other.right_half_bonds)

        half_bond_tuples = list(half_bond_tuples)

        new_ring_bond_map = self.ring_bond_map
        for ring_bond_idx in other.ring_bond_map:
            if ring_bond_idx in new_ring_bond_map:
                half_bond_tuples.append((new_ring_bond_map[ring_bond_idx], other.ring_bond_map[ring_bond_idx]))
                del new_ring_bond_map[ring_bond_idx]
            else:
                new_ring_bond_map[ring_bond_idx] = other.ring_bond_map[ring_bond_idx]

        self.ring_bond_map = new_ring_bond_map

        self.g = nx.union(self.g, other.g)

        for self_bond, other_bond in half_bond_tuples:
            if ("matching_index" in self.g.nodes[self_bond.node_id]) and ("matching_index" in other.g.nodes[other_bond.node_id]):
                if self.g.nodes[self_bond.node_id]["matching_index"] == other.g.nodes[other_bond.node_id]["matching_index"]:
                    self.add_half_bond_edge(self_bond, other_bond)
            else:
                self.add_half_bond_edge(self_bond, other_bond)

    def add_half_bond_edge(self, self_half_bond_edge: _HalfBond, other_half_bond_edge: _HalfBond) -> None:
        overlapping_keys = self_half_bond_edge.bond_attributes.keys() & other_half_bond_edge.bond_attributes.keys()
        if len(overlapping_keys) > 0:
            raise ValueError(overlapping_keys)

        new_bond_attributes = self_half_bond_edge.bond_attributes | other_half_bond_edge.bond_attributes
        self.g.add_edge(self_half_bond_edge.node_id, other_half_bond_edge.node_id, **new_bond_attributes)

    def add_ring_bond(self, ring_bond, half_bond: _HalfBond) -> bool:

        if ring_bond.idx in self.ring_bond_map:
            self.add_half_bond_edge(half_bond, self.ring_bond_map[ring_bond.idx])
            del self.ring_bond_map[ring_bond.idx]
            return False

        self.ring_bond_map[ring_bond.idx] = half_bond
        return True

    def __str__(self):
        return f"PartialGraph({self.g}, {self.left_half_bonds}, {self.right_half_bonds}, {self.ring_bond_map})"

    def __getitem__(self, idx):
        return self.g.nodes[idx]


def _docstring_format(*args, **kwargs):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return dec


class GraphCreator:
    def __init__(self, final_partial_graph: _PartialGraph, text: str = ""):
        self._partial_graph = final_partial_graph
        self._graph_without_bond_connectors = None
        self._g = self._partial_graph.g
        self.text: str = text

        self._g = GraphCreator._mark_aromatic_bonds(self.g)
        self._duplicate_static_edges()
        self._assign_stochastic_ids()
        self._build_molar_amount_dict()
        self._bc_idx_set = self._create_bc_idx_set(self.g)
        self._split_atoms_with_multiple_bond_connectors()

    @staticmethod
    def _create_bc_idx_set(graph):
        from .bond import BondConnector

        bc_idx_set = set()
        for node_idx, data in graph.nodes(data=True):
            if isinstance(data["obj"], BondConnector):
                bc_idx_set.add(node_idx)
        return bc_idx_set

    @staticmethod
    def _create_bracket_atom(string):
        from .atom import BracketAtom

        return BracketAtom.make(string)

    def _build_molar_amount_dict(self):

        for node_idx, data in self._g.nodes(data=True):
            if "molar_amount" in data:

                if "molar_amount" in data:
                    for sto_obj, molar_amount in data["molar_amount"].items():
                        stochastic_id = self._stochastic_id_map[sto_obj]
                        try:
                            data["molar_amount_dict"][stochastic_id] = molar_amount
                        except KeyError:
                            data["molar_amount_dict"] = {stochastic_id: molar_amount}

    def _assign_stochastic_ids(self):
        self._stochastic_id_map = {-1: -1}
        for _node, data in self._g.nodes(data=True):
            if "stochastic_obj" in data:
                if id(data["stochastic_obj"]) not in self._stochastic_id_map:
                    self._stochastic_id_map[id(data["stochastic_obj"])] = max(self._stochastic_id_map.values()) + 1
                stochastic_id = self._stochastic_id_map[id(data["stochastic_obj"])]
                data["stochastic_id"] = stochastic_id
        # Ids index fixed-size per-node vectors (molecular_weight_distribution,
        # unit_molar_amounts, stochastic_id_tree) of length _STOCHASTIC_TREE_DEPTH;
        # an id beyond that raised a bare IndexError deep in graph construction.
        n_stochastic_objects = len(self._stochastic_id_map) - 1  # minus the -1 sentinel
        if n_stochastic_objects > _STOCHASTIC_TREE_DEPTH:
            raise TooManyStochasticObjects(n_stochastic_objects, _STOCHASTIC_TREE_DEPTH)

    def _split_atoms_with_multiple_bond_connectors(self):
        graph = self.g
        bc_idx_set = self._bc_idx_set
        list_of_multiple_bonds = {}
        for node in graph.nodes():

            if node not in bc_idx_set:  # Regular atoms
                # Order-preserving dedup: iterating a SET of (uuid string) node ids
                # is PYTHONHASHSEED-dependent, and the loop below inserts new nodes
                # in this order — which must be parse-stable for the generated
                # graph structure to be reproducible across processes.
                attached_bc = []
                for _u, v in graph.out_edges(node):
                    if v in bc_idx_set and v not in attached_bc:
                        attached_bc.append(v)
                if len(attached_bc) > 1:
                    list_of_multiple_bonds[node] = list(attached_bc)
        for node in list_of_multiple_bonds:
            for bc in list_of_multiple_bonds[node]:
                node_to_bc_edges_data = list(graph.get_edge_data(node, bc).values())
                bc_to_node_edges_data = list(graph.get_edge_data(bc, node).values())
                self._g.remove_edges_from([(node, bc, k) for k in list(graph[node][bc].keys())])
                self._g.remove_edges_from([(bc, node, k) for k in list(graph[bc][node].keys())])

                new_node_idx = str(uuid.uuid4())

                self._g.add_node(new_node_idx)
                self._g.nodes[new_node_idx]["smi_text"] = "[*]"
                obj = self._create_bracket_atom("[*]")
                self._g.nodes[new_node_idx]["obj"] = obj

                try:
                    self._g.nodes[new_node_idx]["stochastic_obj"] = graph.nodes[node]["stochastic_obj"]
                except KeyError:
                    pass
                try:
                    self._g.nodes[new_node_idx]["stochastic_id"] = self._stochastic_id_map[id(graph.nodes[node]["stochastic_obj"])]
                except KeyError:
                    pass

                for attrs in node_to_bc_edges_data:
                    self._g.add_edge(new_node_idx, bc, **deepcopy(attrs))
                for attrs in bc_to_node_edges_data:
                    self._g.add_edge(bc, new_node_idx, **deepcopy(attrs))
                self._g.add_edge(node, new_node_idx)
                self._g.add_edge(new_node_idx, node)

    @staticmethod
    def _mark_aromatic_bonds(graph):
        from .atom import Atom

        # Post-process, marking aromatic bonds
        for edge in graph.edges(data=True):
            node_a = graph.nodes()[edge[0]]["obj"]
            node_b = graph.nodes()[edge[1]]["obj"]
            if isinstance(node_a, Atom) and isinstance(node_b, Atom):
                # Association edges (bond_type ".") between aromatic ions are not aromatic bonds.
                if node_a.aromatic and node_b.aromatic and edge[2].get(_BOND_TYPE_NAME) != ".":
                    edge[2][_AROMATIC_NAME] = True

        return graph

    def _duplicate_static_edges(self):
        for u, v, _k, d in list(self._g.edges(keys=True, data=True)):
            if is_static_edge(d):
                alternate_direction_data = self._g.get_edge_data(v, u)
                edge_found = False
                if alternate_direction_data is not None:
                    for key in alternate_direction_data:
                        if d == alternate_direction_data[key]:
                            edge_found = True
                if not edge_found:
                    self._g.add_edge(v, u, **d)

    # @staticmethod
    # def _remove_unnecessary_static_edges(graph):
    #     init_nodes = {}
    #     for node, data in graph.nodes(data=True):
    #         if "init_weight" in data and data["init_weight"] is not None:
    #             init_nodes[node] = set(nx.bfs_tree(graph, source=node).nodes())

    #     if len(init_nodes) > 0:
    #         for node in list(graph.nodes()):
    #             for u, v, k, d in graph.edges(node, keys=True, data=True):
    #                 if is_static_edge(d):
    #                     tmp_graph = graph.copy()
    #                     tmp_graph.remove_edge(u,v,k)

    #                     accept_graph = True
    #                     for start_node in init_nodes:
    #                         new_set = set(nx.bfs_tree(tmp_graph, source=start_node).nodes())
    #                         if new_set != init_nodes[start_node]:
    #                             accept_graph = False
    #                             break
    #                     if accept_graph:
    #                         print(u,v,k,d)
    #                         graph = tmp_graph
    #     return graph

    def __str__(self):
        return f"GraphCreator({self.g})"

    @property
    def g(self):
        return self._g.copy()

    def get_graph_without_bond_connectors(self):
        def conditional_traversal(graph, source, stop_condition):
            """
            Perform a graph traversal starting from the source node.
            Stop traversal if a node fulfills the stop_condition and return all nodes where traversal stopped.

            Args:
                graph (nx.DiGraph): The directed graph.
                source: The starting node for the traversal.
                stop_condition (function): A function that takes a node and returns True if traversal should stop.

            Returns:
                set: A set of nodes where the traversal stopped.

            """
            stopped_nodes = set()
            visited = set()

            def dfs(node):
                if node in visited:
                    return
                visited.add(node)

                if stop_condition(node):
                    stopped_nodes.add(node)
                    return  # Stop traversal from this node

                # Continue traversal to neighbors
                for neighbor in graph.neighbors(node):
                    dfs(neighbor)

            dfs(source)
            return stopped_nodes

        graph = self.g.copy()

        bc_idx_set = self._bc_idx_set
        outer_self = self

        class BondConnectorPath:
            def __init__(self, edge_path: list[tuple], graph):
                self.graph = graph
                self.edge_path = edge_path
                self._stochastic_id_map = outer_self._stochastic_id_map

                node_path: list = []
                data_path: list = []

                last_node = None
                have_last = False

                for edge in self.edge_path:
                    data = graph.get_edge_data(*edge)
                    data_path.append(data)

                    for node in edge[:2]:
                        if not have_last:
                            node_path.append(node)
                            last_node = node
                            have_last = True
                        else:
                            if last_node != node:
                                node_path.append(node)
                                last_node = node

                self.node_path = node_path
                self.data_path = data_path
                self._weight, self._combined_attr = self.create_combined_attr()

            def create_combined_attr(self) -> dict | None:
                data = {}
                weight = 1.0
                weight_type_list = []
                non_static_attribute_list = list(_NON_STATIC_ATTR)
                for d in self.data_path:
                    if _STATIC_NAME in d:
                        if _STATIC_NAME in data:
                            data[_STATIC_NAME] &= d[_STATIC_NAME]
                        else:
                            data[_STATIC_NAME] = d[_STATIC_NAME]
                    if _AROMATIC_NAME in d:
                        if _AROMATIC_NAME in data:
                            data[_AROMATIC_NAME] |= d[_AROMATIC_NAME]
                        else:
                            data[_AROMATIC_NAME] = d[_AROMATIC_NAME]
                    if _BOND_TYPE_NAME in d:
                        if _BOND_TYPE_NAME in data:
                            if str(data[_BOND_TYPE_NAME]) != str(d[_BOND_TYPE_NAME]):
                                warnings.warn(
                                    IncompatibleBondTypeBondConnector(str(data[_BOND_TYPE_NAME]), str(d[_BOND_TYPE_NAME])),
                                    stacklevel=2,
                                )
                                return 0.0, None
                        else:
                            data[_BOND_TYPE_NAME] = d[_BOND_TYPE_NAME]

                    non_static_weights = [d[attr] if attr in d else 0 for attr in non_static_attribute_list]
                    if max(non_static_weights) > 0:
                        for attr, w in zip(non_static_attribute_list, non_static_weights):
                            if w > 0:
                                current_type = attr
                                weight *= w
                    else:
                        current_type = _STATIC_NAME

                    weight_type_list.append(current_type)

                last_rank = 0
                max_rank = 0

                for edge in self.edge_path:
                    u, v = edge[:2]

                    if "stochastic_obj" in graph.nodes[u]:
                        if "stochastic_obj" in graph.nodes[v]:
                            if self.graph.nodes[u]["stochastic_obj"].stochastic_parent is not None:
                                source_parent = self._stochastic_id_map[id(graph.nodes[u]["stochastic_obj"].stochastic_parent)]
                                if graph.nodes[v]["stochastic_id"] == source_parent:
                                    last_rank += 1
                            if self.graph.nodes[v]["stochastic_obj"].stochastic_parent is not None:
                                target_parent = self._stochastic_id_map[id(graph.nodes[v]["stochastic_obj"].stochastic_parent)]
                                if graph.nodes[u]["stochastic_id"] == target_parent:
                                    last_rank -= 1
                        #    if termination: rank +=1 ?
                        else:
                            last_rank += 1

                    max_rank = max(max_rank, last_rank)

                data["rank"] = [max_rank, last_rank]

                # sto_id = the "managing SO": the SO at max_rank levels up from the source atom's SO.
                # This is the first unterminated ancestor in a termination cascade, i.e. the SO
                # whose bond connectors the path traverses at its outermost point and which
                # therefore decides to create the next repeat unit.
                # Firing sto_id=X means: "SO X is the first unterminated ancestor — use its transitions."
                #
                # Special case — "forced nested" transition (parent -> descendant with no parent BC
                # on the path): when the source atom descends into a child SO via a SMILES branch
                # without any of its own SO's bond connectors mediating the bond, the parent has no
                # stochastic decision to make and the transition is controlled by the highest-order
                # child SO encountered on the descent (the first BC met after the source).
                source_node = self.node_path[0]
                if "stochastic_obj" in graph.nodes[source_node]:
                    source_so_id = self._stochastic_id_map[id(graph.nodes[source_node]["stochastic_obj"])]

                    forced_nested_sto_id = None
                    if max_rank == 0 and last_rank < 0:
                        first_descendant_so_id = None
                        parent_bc_present = False
                        for bc_node in self.node_path[1:len(self.node_path)-1]:
                            if "stochastic_obj" not in graph.nodes[bc_node]:
                                continue
                            bc_so_id = self._stochastic_id_map[id(graph.nodes[bc_node]["stochastic_obj"])]
                            if bc_so_id == source_so_id:
                                parent_bc_present = True
                                break
                            if first_descendant_so_id is None:
                                first_descendant_so_id = bc_so_id
                        if not parent_bc_present and first_descendant_so_id is not None:
                            forced_nested_sto_id = first_descendant_so_id

                    if forced_nested_sto_id is not None:
                        data["sto_id"] = forced_nested_sto_id
                    else:
                        managing_so = graph.nodes[source_node]["stochastic_obj"]
                        for _ in range(max_rank):
                            if managing_so.stochastic_parent is not None:
                                managing_so = managing_so.stochastic_parent
                            else:
                                break
                        data["sto_id"] = self._stochastic_id_map[id(managing_so)]
                else:
                    data["sto_id"] = None

                last_weight_type = _STATIC_NAME
                for weight_type in weight_type_list:
                    if last_weight_type != _STATIC_NAME and weight_type != _STATIC_NAME:
                        return 0.0, None
                    last_weight_type = weight_type

                weight_type = _STATIC_NAME
                for attr in [_PROPAGATION_NAME, _TRANSITION_NAME, _TERMINATION_NAME]:
                    if attr in weight_type_list:
                        weight_type = attr

                if max_rank == 0 and last_rank < 0:
                    zeroth_node = self.node_path[0]
                    first_node = self.node_path[1]
                    if "stochastic_obj" in graph.nodes[zeroth_node] and "stochastic_obj" in graph.nodes[first_node]:
                        if graph.nodes[zeroth_node]["stochastic_id"] == graph.nodes[first_node]["stochastic_id"]:
                            if self.init_weight is None:
                                weight_type = _PROPAGATION_NAME

                if weight > 0:
                    data[weight_type] = weight

                return weight, data

            def __len__(self):
                return len(self.node_path)

            @property
            def group_values(self):
                """Group attributes of the contracted bond: the source side of the first bond connector edge on the path and the target side of the last.

                The bond connectors in between (terminal descriptors and the bond connectors
                that attach a nested stochastic object) are plain by validation, so a path
                across nesting levels carries the rule of the unit it leaves or of the unit
                it reaches; both at once is refused.
                """
                edges = [d for d in self.data_path if any(d.get(attr, 0) > 0 for attr in _NON_STATIC_ATTR)]
                if not edges:
                    return _GROUP_EDGE_SENTINELS
                source = tuple(edges[0].get(key, sentinel) for key, sentinel in zip(_GROUP_EDGE_ATTR[:2], _GROUP_EDGE_SENTINELS[:2]))
                target = tuple(edges[-1].get(key, sentinel) for key, sentinel in zip(_GROUP_EDGE_ATTR[2:], _GROUP_EDGE_SENTINELS[2:]))
                if len(edges) > 1 and source[1] != 0 and target[1] != 0:
                    raise GroupRulesOnBothPathEnds(str(self.graph.nodes[self.node_path[1]]["obj"]), source, str(self.graph.nodes[self.node_path[-2]]["obj"]), target)
                return source + target

            @property
            def only_bond_connectors(self):
                return len(self.node_path) > 2 and set(self.node_path[1:len(self.node_path)-1]).issubset(bc_idx_set)

            def contains_bc(self, bc_idx):
                return bc_idx in self.node_path

            @property
            def weight(self):
                return self._weight

            @property
            def combined_attr(self):
                return self._combined_attr

            def non_zero_weight(self):
                return self.weight > 0

            def valid(self, bc_idx):
                return (
                    self.only_bond_connectors
                    and self.contains_bc(bc_idx)
                    and (self.combined_attr is not None)
                    and (self.weight > 0)
                    # and self.num_stochastic_transitions < 3
                )

            def get_stochastic_transition_path(self):
                stochastic_id_list = []
                for node in self.node_path:
                    try:
                        stochastic_id = self.graph.nodes[node]["stochastic_id"]
                    except KeyError:
                        stochastic_id = -1

                    if len(stochastic_id_list) == 0 or stochastic_id_list[len(stochastic_id_list)-1] != stochastic_id:
                        stochastic_id_list += [stochastic_id]

                return stochastic_id_list

            @property
            def num_stochastic_transitions(self):
                return len(self.get_stochastic_transition_path())

            def __str__(self):
                string = str(graph.nodes[self.node_path[0]]["obj"]) + " "
                for edge, data in zip(self.edge_path, self.data_path):
                    string += str(data) + "\n"
                    string += str(graph.nodes[edge[1]]["obj"]) + " "
                return string

            @property
            def init_weight(self):
                init_weight = None
                # for node_idx in self.node_path:
                if len(self.node_path) > 1:
                    node_idx = self.node_path[1]
                    node_init_weight = self.graph.nodes[node_idx].get("init_weight", None)
                    if node_init_weight is not None and node_init_weight > 0:
                        init_weight = node_init_weight
                return init_weight

        class GraphDecider:
            def __init__(self, bc_idx_set, in_idx):
                self.bc_idx_set = bc_idx_set
                self.in_idx = in_idx

            def __call__(self, x):
                return (x not in self.bc_idx_set) and (x != self.in_idx)

        edges_to_add = []

        # Iterating SETS of (uuid string) node ids is PYTHONHASHSEED-dependent;
        # the loops below build weighted edges in iteration order, which must be
        # parse-stable so the generated graph is reproducible across processes.
        _graph_node_order = {node: index for index, node in enumerate(graph.nodes())}

        # Add edges jumping over the pairs of bond connectors with correct weights.
        # The following routine loops over all bond connectors and finds the in_edges that are atoms. Then it finds successor atoms
        # for those. These successors are then used to find all paths between the initial atom and the successor atoms, and pick only those paths
        # that are valid considering the criteria built in the BondConnectorPath class.
        for bc_idx in sorted(bc_idx_set, key=_graph_node_order.__getitem__):
            for in_edge in graph.in_edges(bc_idx, data=True):
                in_idx = in_edge[0]
                # Only do it for sources of the bond connectors that are not bond connectors themselves. I.e. at the start of a chain.
                if in_idx not in bc_idx_set:
                    traversal_condition = GraphDecider(bc_idx_set, in_idx)
                    non_bond_connector_successor = conditional_traversal(graph, in_idx, traversal_condition)
                    for target in sorted(non_bond_connector_successor, key=_graph_node_order.__getitem__):
                        all_paths = list(nx.all_simple_edge_paths(graph, in_idx, target))
                        for path in all_paths:
                            bond_connector_path = BondConnectorPath(path, graph)
                            if bond_connector_path.valid(bc_idx):
                                data = bond_connector_path.combined_attr
                                data.update(zip(_GROUP_EDGE_ATTR, bond_connector_path.group_values))
                                edges_to_add.append((in_idx, target, data))
                                if bond_connector_path.init_weight is not None:
                                    graph.nodes[in_idx]["init_weight"] = bond_connector_path.init_weight
                                graph.nodes[in_idx]["weight"] = graph.nodes[bc_idx]["obj"].weight

        # The previous approach does not handle self-loops on bond connectors, since they are cycles.
        # However, these are important and easily addressed manually.
        for bc_idx in sorted(bc_idx_set, key=_graph_node_order.__getitem__):
            in_edges = set(graph.in_edges(bc_idx, keys=True))
            out_edges = set(graph.out_edges(bc_idx, keys=True))
            # A loop to itself is the intersection
            loop_edges = in_edges.intersection(out_edges)
            for loop_edge in sorted(loop_edges, key=lambda edge: (_graph_node_order[edge[0]], _graph_node_order[edge[1]], edge[2])):
                for in_u, in_v, in_k, in_data in graph.in_edges(bc_idx, keys=True, data=True):
                    if is_static_edge(in_data):
                        for out_u, out_v, out_k, out_data in graph.out_edges(bc_idx, keys=True, data=True):
                            if is_static_edge(out_data):
                                path = [(in_u, in_v, in_k), loop_edge, (out_u, out_v, out_k)]
                                bond_connector_path = BondConnectorPath(path, graph)
                                if bond_connector_path.valid(bc_idx):
                                    data = bond_connector_path.combined_attr
                                    data.update(zip(_GROUP_EDGE_ATTR, bond_connector_path.group_values))
                                    edges_to_add.append((in_u, out_v, data))

        for edge in edges_to_add:
            graph.add_edge(edge[0], edge[1], **edge[2])
        # Remove all bond connectors from the graph.
        graph.remove_nodes_from(bc_idx_set)

        # Normalize transition weights
        for node in graph.nodes():
            total_weight = 0
            for _u, _v, _k, d in graph.out_edges(node, keys=True, data=True):
                if _TRANSITION_NAME in d:
                    total_weight += d[_TRANSITION_NAME]

            if total_weight > 0:
                for u, v, k, d in graph.out_edges(node, keys=True, data=True):
                    if _TRANSITION_NAME in d:
                        graph.edges[u, v, k][_TRANSITION_NAME] /= total_weight

        graph = GraphCreator._mark_aromatic_bonds(graph)

        return graph

    @_docstring_format(
        propagation_name=_PROPAGATION_NAME,
        termination_name=_TERMINATION_NAME,
        transition_name=_TRANSITION_NAME,
        static_name=_STATIC_NAME,
        stochastic_id_name=_EDGE_STOCHASTIC_ID_NAME,
        aromatic_name=_AROMATIC_NAME,
        bond_type_name=_BOND_TYPE_NAME,
        source_group_name=_SOURCE_GROUP_NAME,
        source_rule_name=_SOURCE_RULE_NAME,
        target_group_name=_TARGET_GROUP_NAME,
        target_rule_name=_TARGET_RULE_NAME,
        smi_bond_mapping=smi_bond_mapping,
    )
    def get_generative_graph(self, include_bond_connectors=False, return_extra_graph_info=False):
        r"""
        The generative graph has well defined properties that do not rely on the specifics of this library.

        Nodes (Atoms) have the following properties:

        - **atomic_num**: int Atomic number, can be converted to Chemical Symbol Name or one-hot encoding.
        - **{aromatic_name}**: bool Indicating the aromaticity of the atom.
        - **charge**: float Nominal charge (not partial charge in Force-Fields) in elementary unit *e*.
        - **molecular_weight_distribution**: array[float] representing the molecular weight distribution of each stochastic object in the graph. The index of each vector is the stochastic id.
        - **mol_molecular_weight** float Molecular Weight of the total molecular weight in the system from this molecular species. If this is unspecified by the string, negative values are used.
        - **total_molecular_weight** float Molecular Weight of the entire material system, this is equal to the sum **mol_molecular_weight** of the comprising molecules. If only one molecule species is present, they are identical. If this is unspecified by the string, negative values are used.
        - **unit_molar_amounts** vector[float] Molar amount of the unit (repeat unit, initiator, or terminator) to which the node belongs, at each stochastic object. Default is 1.0.
        - **init_weight** float Molecular Weight fractions for entry points into the graph generation. If no molecular weights are specified, 1.0 is used. Negative values indicate nodes that are not starting positions for the generation.
        - **gen_weight** float Weight to select this atom for the next generation step.
        - **gen_hierarchy**: int Hierarchy for the selection of the atom for the next generation step. Default is 0. Atoms with a higher hierarchy number will be chosen first.
        - **stochastic_id_tree**: vector[int] Identification of the stochastic objects to which the node belongs. The vector is ordered from nearest to farthest ancestor (e.g., first element is node stochastic id, second element is parent, third is grandparent, etc.).  All elements are -1 if not a stochastic object, element is -2 if ancestor is absent.

        Edges (Bonds) have the following properties:

        - **{static_name}**: bool indicating static edges, that are always present.
        - **{propagation_name}**: float Propagation probability. If bond connectors are connecting between monomer repeat units inside stochastic objects, this indicates the probability \in [0, 1].
        - **{termination_name}**: float Termination Probabilities. If bond connectors terminate with end-groups after the molecular weight is reached, this is the probability \in [0, 1].
        - **{transition_name}**: float Transition Probabilities. If transitioning between stochastic objects, this is the probability to take.
        - **{stochastic_id_name}**: integer Stochastic-object id that manages the bond. Transition bonds carry the managing SO's id (-1 for cross-family/global transitions fired after all SOs terminate); termination bonds carry the target terminator's SO id; propagation and static bonds carry the source node's SO id. -2 marks edges of the include_bond_connectors=True graph, where no assignment is performed.
        - **{bond_type_name}**: int Integer category that maps to different bond_types as follows{smi_bond_mapping}. Category 0 is an association edge (e.g. an ion pair with a trailing counterion): the atoms travel together with the unit but share no covalent bond.
        - **{aromatic_name}**: bool Indicates aromatic bonds.
        - **{source_group_name}**, **{target_group_name}**: int Group id declared by the bond connector symbol of the unit the bond leaves from, respectively arrives at (conditional connectivity). -1 when that symbol declares no group.
        - **{source_rule_name}**, **{target_rule_name}**: int Group rule of that symbol: 0 NONE, 1 LADDER, 2 EXCLUSION, 3 ALL (the ``GroupRule`` encoding, stable across versions). A bond across nesting levels passes through terminal bond connectors and the bond connectors that attach the nested stochastic object; those are plain, so the bond carries the rule of the unit it leaves or of the unit it reaches, and a bond ruled at both ends is refused (``GroupRulesOnBothPathEnds``). Static bonds and the terminal-bond-connector side of a stochastic object's entry or exit carry the sentinels -1 and 0. One bond per distinct group annotation of a compatible bond connector pair: a pair whose symbols match in two differently grouped ways yields two parallel bonds sharing that pair's weight.

        The graph carries the G2RINS string it was generated from as the
        graph-level attribute **g2rins_string**, and a mapping from unit_id to
        the G2RINS text of the token each unit originates from as the
        graph-level attribute **unit_g2rins**. Graph-level attributes are
        parser provenance: only a parsed graph carries them, and no consumer
        may require them.

        Unit and bond labels (``unit_id``, ``bond_id``) are NOT stored on
        nodes; they are derived annotations — compute them on demand with
        :func:`derive_unit_labels`. The node and edge properties listed above
        are the full graph contract: a generative model emitting
        **atomic_num**, **{aromatic_name}**, **charge**, **num_explicit_h**,
        **init_weight**, the **{static_name}** edge flag and the three
        non-static weights produces a graph that every consumer, including the
        label derivation, can handle; consumers read absent group-rule
        attributes as their sentinels (no group, rule NONE).
        """

        from .distribution import StochasticDistribution

        extra_graph_info = {0: "None"}
        extra_graph_info_reverse = {"None": 0}

        if include_bond_connectors:
            graph = self.g.copy()
        else:
            graph = self.get_graph_without_bond_connectors().copy()

        generative_graph = nx.MultiDiGraph()
        generative_graph.graph["g2rins_string"] = self.text

        graph_with_bond_connectors = self.g.copy()
        MW_distribution_array = [StochasticDistribution.get_empty_serial_vector()] * 10

        for node, data in graph_with_bond_connectors.nodes(data=True):
            try:
                stochastic_id = data["stochastic_id"]
                MW_distribution_vector = data["stochastic_obj"].stochastic_generation.get_serial_vector()
                MW_distribution_array[stochastic_id] = MW_distribution_vector
            except KeyError:
                pass

        for node, data in graph.nodes(data=True):
            obj = data["obj"]
            try:
                aromatic = obj.aromatic
            except AttributeError:
                aromatic = False
            atomic_symbol = str(obj.symbol)
            if aromatic:
                atomic_symbol = atomic_symbol.upper()
            try:
                atomic_num = int(atom_name_num[atomic_symbol])
            except KeyError:
                string = str(obj)
                if string in extra_graph_info_reverse:
                    atomic_num = extra_graph_info_reverse[string]
                else:
                    idx = min(extra_graph_info.keys()) - 1
                    atomic_num = idx
                    extra_graph_info[idx] = string
                    extra_graph_info_reverse[string] = idx

            try:
                gen_weight = obj.weight
            except AttributeError:
                try:
                    gen_weight = data["weight"]
                except KeyError:
                    gen_weight = -1

            try:
                charge = obj.charge
            except AttributeError:
                charge = float("nan")

            # Explicit hydrogen count of AROMATIC bracket atoms that write one
            # (e.g. [nH]). Only aromatic atoms need this: it is the one case where
            # valence inference fails (an aromatic ring bond is over-counted, and
            # RDKit cannot tell a pyrrole-type N-H from a pyridine-type N), so the
            # written count is required for the ring to kekulize and for the tracked
            # MW to match. -1 means "infer H from valence" and is used for every
            # non-aromatic atom: the written H count is a monomer-level property
            # that does NOT survive polymerization when the sampler realizes a
            # different coordination (branch points, chain termini, unfired bond
            # connectors), whereas valence inference correctly adapts. Forcing the
            # written count on non-aromatic atoms produced radicals when
            # under-coordinated and over-valence crashes when over-coordinated.
            num_explicit_h = -1
            if aromatic:
                try:
                    h_count = obj.h_count
                    if h_count is not None:
                        num_explicit_h = h_count.num
                except AttributeError:
                    num_explicit_h = -1

            if "stochastic_obj" not in data or data["stochastic_obj"].stochastic_generation is None:
                stochastic_id = -1
            else:
                stochastic_id = data["stochastic_id"]

            mol_molecular_weight = -1.0
            if "mol_molecular_weight" in data and data["mol_molecular_weight"] is not None:
                mol_molecular_weight = data["mol_molecular_weight"]

            total_molecular_weight = -1.0
            if "total_molecular_weight" in data and data["total_molecular_weight"] is not None:
                total_molecular_weight = data["total_molecular_weight"]

            init_weight = -1.0
            if "init_weight" in data and data["init_weight"] is not None:
                init_weight = data["init_weight"]

            molar_amount = [1.0] * _STOCHASTIC_TREE_DEPTH
            if "molar_amount_dict" in data:
                for key, value in data["molar_amount_dict"].items():
                    molar_amount[int(key)] = value

            stochastic_id_tree = [-1] * _STOCHASTIC_TREE_DEPTH

            if stochastic_id != -1:
                stochastic_id_tree = [-2] * _STOCHASTIC_TREE_DEPTH
                stochastic_id_tree[0] = stochastic_id
                depth_idx = 1
                if "stochastic_obj" in data:
                    current_stochastic_obj = data["stochastic_obj"]
                    while current_stochastic_obj.stochastic_parent is not None:
                        stochastic_id_tree[depth_idx] = self._stochastic_id_map[id(current_stochastic_obj.stochastic_parent)]
                        current_stochastic_obj = current_stochastic_obj.stochastic_parent
                        depth_idx += 1

            generative_graph.add_node(
                node,
                **{
                    "atomic_num": atomic_num,
                    _AROMATIC_NAME: aromatic,
                    "charge": charge,
                    "num_explicit_h": int(num_explicit_h),
                    "molecular_weight_distribution": MW_distribution_array,
                    "mol_molecular_weight": mol_molecular_weight,
                    "total_molecular_weight": total_molecular_weight,
                    "unit_molar_amounts": molar_amount,
                    "init_weight": float(init_weight),
                    "gen_weight": float(gen_weight),
                    "gen_hierarchy": 0,
                    "stochastic_id_tree": stochastic_id_tree,
                },
            )

        for u, v, _k, d in graph.edges(keys=True, data=True):

            d.setdefault(_STATIC_NAME, is_static_edge(d))
            d.setdefault(_PROPAGATION_NAME, 0)
            d.setdefault(_TERMINATION_NAME, 0)
            d.setdefault(_TRANSITION_NAME, 0)
            d.setdefault(_EDGE_STOCHASTIC_ID_NAME, -2)  # -2 = unassigned; real ids start at 0, -1 is the global level
            for key, sentinel in zip(_GROUP_EDGE_ATTR, _GROUP_EDGE_SENTINELS):
                d.setdefault(key, sentinel)

            if _BOND_TYPE_NAME in d:
                d[_BOND_TYPE_NAME] = smi_bond_mapping.get(str(d[_BOND_TYPE_NAME]), 1)
            else:
                d[_BOND_TYPE_NAME] = 1
            d.setdefault(_AROMATIC_NAME, False)

            generative_graph.add_edge(u, v, **d)

        if not include_bond_connectors:
            for node in generative_graph.nodes():
                source_so_id = generative_graph.nodes[node]["stochastic_id_tree"][0]

                for _u, _v, _k, d in generative_graph.out_edges(node, keys=True, data=True):
                    if d.get(_TRANSITION_NAME, 0) > 0:
                        # Transition edges: sto_id = managing SO if source and target share ancestry
                        # (one is ancestor of the other, or they share a common ancestor);
                        # -1 (external control) if either node has no SO or they are in different families.
                        source_ids = {x for x in generative_graph.nodes[_u]["stochastic_id_tree"] if x >= 0}
                        target_ids = {x for x in generative_graph.nodes[_v]["stochastic_id_tree"] if x >= 0}

                        if source_ids & target_ids:
                            d[_EDGE_STOCHASTIC_ID_NAME] = d["sto_id"]
                        else:
                            d[_EDGE_STOCHASTIC_ID_NAME] = -1
                    elif d.get(_TERMINATION_NAME, 0) > 0:
                        # Termination edges: stochastic_id = target node's SO.
                        # A node in an inner SO may have terminators at multiple SO levels;
                        # the target terminator's SO determines which cascade level it belongs to.
                        target_so_id = generative_graph.nodes[_v]["stochastic_id_tree"][0]
                        d[_EDGE_STOCHASTIC_ID_NAME] = target_so_id
                    else:
                        # Stochastic and static edges: stochastic_id = source node's SO.
                        d[_EDGE_STOCHASTIC_ID_NAME] = source_so_id
                for _u, _v, _k, d in generative_graph.out_edges(node, keys=True, data=True):
                    d.pop("rank", None)
                    d.pop("sto_id", None)

        # Set the generation hierarchy of nodes when more than one node can transition from the same SO.
        # First, we set the lowest level of the stochastic tree.
        sto_id_to_node_indexes = {}

        for node, data in generative_graph.nodes(data=True):
            stochastic_id = data["stochastic_id_tree"][0]
            try:
                sto_id_to_node_indexes[stochastic_id] += [node]
            except KeyError:
                sto_id_to_node_indexes[stochastic_id] = [node]
        sto_id_to_transition_edge = {}
        for stochastic_id, node_indexes in sto_id_to_node_indexes.items():
            transition_node_to_target = {}
            for node in node_indexes:
                for u, v, d in generative_graph.out_edges(node, data=True):
                    if d[_TRANSITION_NAME] > 0 and (generative_graph.nodes[v]["stochastic_id_tree"][0] != stochastic_id):
                        try:
                            sto_id_to_transition_edge[stochastic_id] += [(u, v)]
                        except KeyError:
                            sto_id_to_transition_edge[stochastic_id] = [(u, v)]
                        try:
                            transition_node_to_target[node] += [v]
                        except KeyError:
                            transition_node_to_target[node] = [v]

            targets_stochastic_level = []
            transition_node_indexes = list(transition_node_to_target.keys())

            for node, target in transition_node_to_target.items():
                node_stochastic_id_tree = generative_graph.nodes[node]["stochastic_id_tree"]
                # Note: this way of establishing hierarchy only works when all possible transitions from each node are directed to the same stochastic object:
                target_stochastic_id = generative_graph.nodes[target[0]]["stochastic_id_tree"][0]
                # TODO: consider siblings, cousins etc. Consider nodes with multiple targets that might be part of different SO
                try:
                    index = node_stochastic_id_tree.index(target_stochastic_id)
                except ValueError:
                    index = 0
                targets_stochastic_level.append(index)

            gen_hierarchy = len(set(targets_stochastic_level)) - 1

            try:
                min_level = min(targets_stochastic_level)
                while True:
                    indexes = [i for i, level in enumerate(targets_stochastic_level) if level == min_level]
                    for i in indexes:
                        generative_graph.nodes[transition_node_indexes[i]]["gen_hierarchy"] = gen_hierarchy
                    gen_hierarchy = gen_hierarchy - 1
                    try:
                        min_level = min([level for level in targets_stochastic_level if level > min_level])
                    except ValueError:
                        break
            except ValueError:
                pass
        # Normalization of bond weights

        for node in generative_graph.nodes():

            total_propagation_weight = 0.0
            if include_bond_connectors:
                total_transition_weight = 0.0
            else:
                # Keyed by the edge's stochastic_id, which is a raw SO id: it can be
                # -1 (cross-family transitions) and is not bounded by the nesting
                # depth, so a fixed-size list would alias -1 into the last slot and
                # crash for graphs with more SOs than _STOCHASTIC_TREE_DEPTH.
                total_transition_weight = defaultdict(float)
            total_termination_weight = 0.0

            for u, v, d in generative_graph.out_edges(node, data=True):
                if d[_PROPAGATION_NAME] > 0:
                    total_propagation_weight += d[_PROPAGATION_NAME]
                if d[_TRANSITION_NAME] > 0:
                    if include_bond_connectors:
                        total_transition_weight += d[_TRANSITION_NAME]
                    else:
                        hierarchy_level = d[_EDGE_STOCHASTIC_ID_NAME]
                        total_transition_weight[hierarchy_level] += d[_TRANSITION_NAME]
                if d[_TERMINATION_NAME] > 0:
                    total_termination_weight += d[_TERMINATION_NAME]

            for u, v, d in generative_graph.out_edges(node, data=True):
                if d[_PROPAGATION_NAME] > 0:
                    d[_PROPAGATION_NAME] /= total_propagation_weight
                if d[_TRANSITION_NAME] > 0:
                    if include_bond_connectors:
                        d[_TRANSITION_NAME] /= total_transition_weight
                    else:
                        hierarchy_level = d[_EDGE_STOCHASTIC_ID_NAME]
                        d[_TRANSITION_NAME] /= total_transition_weight[hierarchy_level]
                if d[_TERMINATION_NAME] > 0:
                    d[_TERMINATION_NAME] /= total_termination_weight

        # unit_id -> G2RINS text of the token each unit originates from.
        # token_text is parser provenance: it is read from the generative graph
        # and never copied onto generative_graph nodes (unit/bond labels are likewise
        # not stored -- consumers derive them with derive_unit_labels).
        unit_id_map = derive_unit_labels(generative_graph).unit_id
        unit_g2rins = {}
        for n, unit_id in unit_id_map.items():
            if unit_id not in unit_g2rins and "token_text" in graph.nodes[n]:
                unit_g2rins[unit_id] = graph.nodes[n]["token_text"]
        generative_graph.graph["unit_g2rins"] = unit_g2rins

        if return_extra_graph_info:
            return generative_graph, extra_graph_info
        return generative_graph

    def write_generative_graph_json(self, filename):
        """
        Write the generative graph (see :meth:`get_generative_graph`) to `filename` as JSON: a
        self-describing ``format`` block plus the node-link ``graph`` with the
        derived unit/bond annotations injected (see :func:`generative_graph_json_data`).

        Load the graph back with
        ``networkx.node_link_graph(json.load(fp)["graph"], edges="edges")``.
        """
        with open(filename, "w") as file_handle:
            json.dump(generative_graph_json_data(self.get_generative_graph()), file_handle, indent=2)

    _DEFAULT_EDGE_COLOR = {
        _STATIC_NAME: "#000000",
        _PROPAGATION_NAME: "#ff0000",
        _TRANSITION_NAME: "#00ff00",
        _TERMINATION_NAME: "#0000ff",
    }
    _DEFAULT_BOND_TO_ARROW = {
        0: "normal",
        1: "normal",
        2: "diamond",
        3: "dot",
        4: "box",
    }

    @staticmethod
    def _used_nonstatic_edges(graph, node_to_unit):
        """
        Identify the non-static bonds that survive a generation traversal of
        `graph` starting from the initiator units (see :meth:`get_dot_string`).

        A non-static bond ``X -> Y`` is *used* iff ``X``'s unit is entered
        through at least one node other than ``X``; initiator units have no such
        restriction. Reaching a unit through a new node can unlock bonds that
        were previously blocked, so the search is iterated to a fixed point.

        `node_to_unit` maps every node to its unit id (see
        :func:`derive_unit_labels`).

        Returns a set of ``(source, target, key)`` edge identifiers, or ``None``
        if the graph has no initiator unit (the caller then prunes nothing).
        """
        from collections import defaultdict, deque

        unit_to_nodes = defaultdict(list)
        initiator_units = set()
        for node, node_data in graph.nodes(data=True):
            unit_id = node_to_unit[node]
            unit_to_nodes[unit_id].append(node)
            if node_data.get("init_weight", -1) > 0:
                initiator_units.add(unit_id)

        if not initiator_units:
            return None

        entry_nodes = defaultdict(set)  # unit_id -> nodes it was entered through

        def _can_exit(check_unit, check_node):
            # Initiators always generate outward; any other unit may exit from a
            # node only if it was entered through some other node.
            if check_unit in initiator_units:
                return True
            return any(entry != check_node for entry in entry_nodes[check_unit])

        used_edges = set()
        queue = deque(initiator_units)
        queued = set(initiator_units)
        while queue:
            unit_id = queue.popleft()
            queued.discard(unit_id)
            for source in unit_to_nodes[unit_id]:
                if not _can_exit(unit_id, source):
                    continue
                for _source, target, key, edge_data in graph.out_edges(source, keys=True, data=True):
                    if is_static_edge(edge_data):
                        continue
                    used_edges.add((source, target, key))
                    target_unit = node_to_unit[target]
                    if target not in entry_nodes[target_unit]:
                        entry_nodes[target_unit].add(target)
                        if target_unit not in queued:
                            queue.append(target_unit)
                            queued.add(target_unit)
        return used_edges

    def get_dot_string(self, include_bond_connectors=False, edge_colors=None, bond_to_arrow=None, node_prefix="", max_digits_count: int = -1, color_by="atom", hide_unused_nonstatic_bonds=True):
        """
        Render the generative graph as a Graphviz DOT string.

        Parameters
        ----------
        color_by: str
            How to color the node fill. ``"atom"`` (default) colors each node
            by its chemical element. ``"unit"`` colors each node by its unit
            (see :meth:`get_generative_graph`): every role gets a colour family --
            initiators blue, repeat units purple, linkers green, terminators
            orange -- and units within a role are shaded apart, so the atoms of
            a unit share a fill color while distinct units stay distinguishable.
        hide_unused_nonstatic_bonds: bool
            When True (default) and ``include_bond_connectors`` is False, omit
            the non-static bonds that a generation traversal from the initiators
            never uses (a bond ``X -> Y`` is unused when ``X``'s unit is only
            ever entered through ``X``). No effect when bond connectors are
            included or when the graph has no initiator.
        """

        if edge_colors is None:
            edge_colors = self._DEFAULT_EDGE_COLOR
        if bond_to_arrow is None:
            bond_to_arrow = self._DEFAULT_BOND_TO_ARROW
        if color_by not in ("atom", "unit"):
            raise ValueError(f"color_by must be 'atom' or 'unit', got {color_by!r}")

        graph, extra_graph_info = self.get_generative_graph(include_bond_connectors=include_bond_connectors, return_extra_graph_info=True)

        node_to_unit = None
        if color_by == "unit" or (hide_unused_nonstatic_bonds and not include_bond_connectors):
            node_to_unit = derive_unit_labels(graph).unit_id

        # Non-static bonds that a generation traversal from the initiators never
        # uses are hidden (default case only); None means "draw every bond".
        used_nonstatic_edges = None
        if hide_unused_nonstatic_bonds and not include_bond_connectors:
            used_nonstatic_edges = self._used_nonstatic_edges(graph, node_to_unit)

        # Optional per-unit coloring: each unit role gets its own colour family
        # -- initiators blue, repeat units purple, linkers green, terminators
        # orange -- and the units within a role are shaded apart so every
        # unit_id still maps to a distinct fill.
        unit_id_color_map = {}
        if color_by == "unit":
            import colorsys

            role_base_hue = {"I": 0.60, "R": 0.79, "L": 0.33, "T": 0.07}

            units_by_role = {}
            for unit_id in set(node_to_unit.values()):
                units_by_role.setdefault(unit_id[0], []).append(unit_id)

            for role, role_unit_ids in units_by_role.items():
                base_hue = role_base_hue.get(role, 0.0)
                role_unit_ids = sorted(role_unit_ids, key=lambda uid: int(uid[1:]))
                count = len(role_unit_ids)
                for index, unit_id in enumerate(role_unit_ids):
                    # Shade within the family; a lone unit lands in the middle.
                    shade = (index + 1) / (count + 1)
                    hue = (base_hue + (shade - 0.5) * 0.04) % 1.0
                    saturation = 0.45 + 0.45 * shade
                    value = 0.95 - 0.30 * shade
                    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
                    unit_id_color_map[unit_id] = "#{:02x}{:02x}{:02x}".format(int(red * 255), int(green * 255), int(blue * 255))

        dot_str = "digraph{\n"
        dot_str += f'label="{self.text}";\n'
        for node in graph.nodes(data=True):
            if node[1]["atomic_num"] >= 0:
                label = f"{atom_name_mapping[node[1]['atomic_num']]}"
                color = "#" + atom_color_mapping[node[1]["atomic_num"]]
            else:
                label = extra_graph_info[node[1]["atomic_num"]]
                color = "#FFFFFF"

            if color_by == "unit":
                color = unit_id_color_map[node_to_unit[node[0]]]

            extra_attr = f'style=filled, fillcolor="{color}", '
            if _determine_darkness_from_hex(color):
                extra_attr += "fontcolor=white,"

            dot_str += f'"{node_prefix}{node[0]}" [{extra_attr} label="{label}"];\n'

        for u, v, _k, d in graph.edges(keys=True, data=True):
            if used_nonstatic_edges is not None and not is_static_edge(d) and (u, v, _k) not in used_nonstatic_edges:
                continue
            bond_type = d[_BOND_TYPE_NAME]
            color = "black"
            value = 1.0
            for key in edge_colors:
                if d[key] > 0:
                    color = edge_colors[key]
                    value = d[key]
            style = "solid"
            if d[_AROMATIC_NAME]:

                style = "dashed"
            if max_digits_count < 0:
                dot_str += f'"{node_prefix}{u}" -> "{node_prefix}{v}" [arrowhead="{bond_to_arrow[bond_type]}", label="{float(value)}", color="{color}", style="{style}"];\n'
            else:
                if "." not in str(value):
                    formatted_value = float(value)
                else:
                    float_value = float(value)
                    str_value = str(float_value)
                    integer_part, decimal_part = str_value.split(".")
                    if len(decimal_part) < max_digits_count:
                        formatted_value = float_value
                    else:
                        format_str = "{:." + str(max_digits_count) + "f}"
                        formatted_value = format_str.format(float(value))
                dot_str += f'"{node_prefix}{u}" -> "{node_prefix}{v}" [arrowhead="{bond_to_arrow[bond_type]}", label="{formatted_value}", color="{color}", style="{style}"];\n'
        dot_str += "}\n"
        return dot_str

    def get_ensemble_creator(self):
        from .ensemble_creator import EnsembleCreator

        return EnsembleCreator(self.get_generative_graph(include_bond_connectors=False))

