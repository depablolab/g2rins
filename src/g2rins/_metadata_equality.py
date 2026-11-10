# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Pure structural comparisons for graph-valued ensemble metadata."""

from collections import deque
from collections.abc import Collection

import networkx as nx
import numpy as np


def _edge_ids(graph):
    """Edge identities of ``graph``, orientation-free for undirected graphs."""
    edges = graph.edges(keys=True) if graph.is_multigraph() else graph.edges
    if graph.is_directed():
        return set(edges)
    return {(frozenset(edge[:2]), *edge[2:]) for edge in edges}


def _graphs_equal(left, right, active):
    if left.is_directed() != right.is_directed() or left.is_multigraph() != right.is_multigraph():
        return False
    if set(left.nodes) != set(right.nodes) or not _metadata_equal(left.graph, right.graph, active):
        return False
    if not all(_metadata_equal(left.nodes[node], right.nodes[node], active) for node in left.nodes):
        return False
    if _edge_ids(left) != _edge_ids(right):
        return False
    edges = left.edges(keys=True) if left.is_multigraph() else left.edges
    return all(_metadata_equal(left.edges[edge], right.edges[edge], active) for edge in edges)


def _array_equal(left, right, active):
    if left.shape != right.shape:
        return False
    if left.dtype.names is not None or right.dtype.names is not None:
        if left.dtype != right.dtype:
            return False
        if left.dtype.hasobject:
            return all(_array_equal(left[name], right[name], active) for name in left.dtype.names)
        return bool(np.array_equal(left, right))
    if left.dtype.kind == "O" or right.dtype.kind == "O":
        return _metadata_equal(left.tolist(), right.tolist(), active)
    try:
        return bool(np.array_equal(left, right))
    except Exception:
        return False


def _masked_equal(left, right, active):
    try:
        left, right = np.ma.asarray(left), np.ma.asarray(right)
        left_mask, right_mask = np.ma.getmaskarray(left), np.ma.getmaskarray(right)
    except Exception:
        return False
    if left.shape != right.shape:
        return False
    if left.dtype.names is not None or right.dtype.names is not None:
        return left.dtype == right.dtype and all(
            _masked_equal(left[name], right[name], active) for name in left.dtype.names
        )
    if not np.array_equal(left_mask, right_mask):
        return False
    kept = ~left_mask
    return _array_equal(np.ma.getdata(left)[kept], np.ma.getdata(right)[kept], active)


def _comparison_result(left, right):
    try:
        result = left == right
    except Exception:
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _is_collection(value):
    return isinstance(value, Collection) and not isinstance(
        value, (str, bytes, bytearray, np.ndarray)
    )


_RECURSIVE_METADATA_TYPES = (dict, list, tuple, deque, np.ndarray, nx.Graph)


def _metadata_equal(left, right, active=None):
    """Compare containers and graph-valued ensemble metadata structurally."""
    if left is right:
        return True
    if active is None:
        active = set()
    pair = None
    if isinstance(left, _RECURSIVE_METADATA_TYPES) and isinstance(right, _RECURSIVE_METADATA_TYPES):
        pair = (id(left), id(right))
        if pair in active:
            return False
        active.add(pair)
    try:
        if isinstance(left, nx.Graph) or isinstance(right, nx.Graph):
            return isinstance(left, nx.Graph) and isinstance(right, nx.Graph) and _graphs_equal(left, right, active)
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                return False
            for key in left:
                if not _metadata_equal(left[key], right[key], active):
                    return False
            return True
        if isinstance(left, (list, tuple, deque)) and isinstance(right, (list, tuple, deque)):
            if len(left) != len(right):
                return False
            for first, second in zip(left, right, strict=True):
                if not _metadata_equal(first, second, active):
                    return False
            return True
        if _is_collection(left) != _is_collection(right):
            return False
        if isinstance(left, np.ma.MaskedArray) or isinstance(right, np.ma.MaskedArray):
            return _masked_equal(left, right, active)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray) or (
            isinstance(left, np.generic) and isinstance(right, np.generic)
        ):
            try:
                left_array, right_array = np.asarray(left), np.asarray(right)
            except Exception:
                return False
            return _array_equal(left_array, right_array, active)
        return _comparison_result(left, right)
    finally:
        if pair is not None:
            active.discard(pair)
