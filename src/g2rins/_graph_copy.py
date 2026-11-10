# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Conservative, order-preserving copies of private unit prototypes."""

from copy import deepcopy

import networkx as nx


# Compare type identities without invoking extension metaclass hash/equality.
_PLAIN_SCALAR_IDS = frozenset(map(id, (str, int, float, bool, bytes, type(None))))
_GRAPH_FIELDS = frozenset(("graph", "_node", "_adj", "__networkx_cache__"))


def _copy_unit_graph(graph, *, deepcopy=deepcopy):
    """Copy ordinary integer-node prototypes; deeply copy everything else.

    Validate on every call: prototypes remain mutable, and even cached NetworkX
    views require the full deepcopy semantics. No recursive attribute checks or
    persistent eligibility cache are used. The injected fallback keeps legacy
    coordinator deepcopy hooks effective for unsupported graphs.
    """
    if type(graph) is not nx.Graph:
        return deepcopy(graph)
    state = graph.__dict__
    if type(state) is not dict:
        return deepcopy(graph)
    if any(type(key) is not str or key not in _GRAPH_FIELDS for key in state):
        return deepcopy(graph)
    if not {"graph", "_node", "_adj"}.issubset(state):
        return deepcopy(graph)
    if any(type(value) is not dict for value in state.values()):
        return deepcopy(graph)
    if state["graph"] or state.get("__networkx_cache__"):
        return deepcopy(graph)

    nodes, adjacency = state["_node"], state["_adj"]
    if any(type(node) is not int for node in nodes):
        return deepcopy(graph)
    if nodes.keys() != adjacency.keys():
        return deepcopy(graph)
    # Structural containers must be ordinary, distinct dictionaries. Attribute
    # aliases are supported below, but aliases into the graph structure are not.
    structural_ids = {id(value) for value in state.values()}
    if len(structural_ids) != len(state):
        return deepcopy(graph)
    for node, neighbors in adjacency.items():
        if type(node) is not int or type(neighbors) is not dict:
            return deepcopy(graph)
        if id(neighbors) in structural_ids:
            return deepcopy(graph)
        structural_ids.add(id(neighbors))
    attribute_copies = {}

    def copy_attributes(attributes):
        if type(attributes) is not dict or id(attributes) in structural_ids:
            return None
        identity = id(attributes)
        if identity in attribute_copies:
            return attribute_copies[identity]
        if any(
            id(type(key)) not in _PLAIN_SCALAR_IDS or id(type(value)) not in _PLAIN_SCALAR_IDS
            for key, value in attributes.items()
        ):
            return None
        copied = attributes.copy()
        attribute_copies[identity] = copied
        return copied

    copied_nodes = {}
    for node, attributes in nodes.items():
        copied = copy_attributes(attributes)
        if copied is None:
            return deepcopy(graph)
        copied_nodes[node] = copied

    copied_adjacency = {}
    for node, neighbors in adjacency.items():
        copied_neighbors = {}
        for neighbor, attributes in neighbors.items():
            if (
                type(neighbor) is not int
                or neighbor not in adjacency
                or adjacency[neighbor].get(node) is not attributes
            ):
                return deepcopy(graph)
            copied = copy_attributes(attributes)
            if copied is None:
                return deepcopy(graph)
            copied_neighbors[neighbor] = copied
        copied_adjacency[node] = copied_neighbors

    # Edge insertion would change some per-node neighbor orders. Install the
    # independently ordered dictionaries directly, retaining mirrored/self-loop
    # attribute identity through the call-local attribute memo.
    result = nx.Graph()
    result.__dict__.clear()
    result.__dict__.update(
        (key, copied_nodes if key == "_node" else copied_adjacency if key == "_adj" else {})
        for key in state
    )
    return result