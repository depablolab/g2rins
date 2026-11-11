# SPDX-License-Identifier: GPL-3.0-only
"""Conservative, ordered model identity for convergence checkpoints only."""

import hashlib
import json
import re

from .generative_graph import _json_safe


MODEL_FINGERPRINT_SCHEMA = "g2rins-model-fingerprint/v1"


def model_fingerprint(graph):
    """Hash the effective graph, not UUIDs, pickle bytes or runtime caches.

    Node insertion order and both adjacency orders are sampling semantics, not
    incidental serialization order. Number nodes by insertion order; never sort
    their random IDs. Attribute mappings are sorted by canonical JSON instead.
    All node/edge attributes are included conservatively (including stereo and
    distribution vectors). Unknown non-JSON attributes fail closed. Graph-level
    provenance is restricted to the fields consumed by the library.
    """
    node_index = {node: index for index, node in enumerate(graph)}
    metadata = {
        name: graph.graph[name]
        for name in (
            "g2rins_string", "unit_g2rins", "molecular_weight_distribution_schema",
        )
        if name in graph.graph
    }
    if "unit_node_ids" in graph.graph:
        # Membership, unlike adjacency, is unordered. Resolve UUID references
        # explicitly rather than replacing arbitrary attribute strings.
        try:
            metadata["unit_node_ids"] = {
                unit: sorted(node_index[node] for node in nodes)
                for unit, nodes in graph.graph["unit_node_ids"].items()
            }
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError(
                "Cannot compute checkpoint model fingerprint: invalid unit membership."
            ) from error

    payload = {
        "schema": MODEL_FINGERPRINT_SCHEMA,
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
        "metadata": metadata,
        "nodes": [data for _, data in graph.nodes(data=True)],
        "successors": [
            [
                [node_index[target], list(edges.items())]
                for target, edges in graph.adj[node].items()
            ]
            for node in graph
        ],
        # Incoming neighbor order is not recoverable from a grouped edge list.
        "predecessors": [
            [[node_index[source], list(edges)] for source, edges in graph.pred[node].items()]
            for node in graph
        ],
    }
    try:
        encoded = json.dumps(
            _json_safe(payload), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Cannot compute checkpoint model fingerprint: unsupported graph attributes."
        ) from error
    # Existing checkpoint counters and retained records refer to origin IDs.
    # Equal chemistry alone cannot safely rebase those records to fresh UUIDs.
    # Keep that operational namespace separate from the UUID-free model hash.
    origin_bytes = json.dumps(
        [str(node) for node in graph], ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": MODEL_FINGERPRINT_SCHEMA,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "origin_sha256": hashlib.sha256(origin_bytes).hexdigest(),
    }


def validate_model_fingerprint(saved, current):
    """Reject unverifiable legacy, unknown-version and different-model states."""
    if (
        not isinstance(saved, dict)
        or set(saved) != {"schema", "sha256", "origin_sha256"}
        or saved.get("schema") != MODEL_FINGERPRINT_SCHEMA
        or any(
            not isinstance(saved.get(name), str)
            or re.fullmatch(r"[0-9a-f]{64}", saved[name]) is None
            for name in ("sha256", "origin_sha256")
        )
    ):
        raise ValueError(
            "checkpoint model fingerprint is missing, malformed or unsupported; "
            "start a new ensemble. Legacy checkpoints cannot be verified."
        )
    if saved["sha256"] != current["sha256"]:
        raise ValueError(
            "checkpoint model fingerprint does not match this model's chemistry "
            "or graph settings; start a new ensemble."
        )
    if saved["origin_sha256"] != current["origin_sha256"]:
        raise ValueError(
            "checkpoint model fingerprint origin namespace does not match; "
            "restore the original EnsembleCreator (for example from pickle) "
            "or start a new ensemble. Freshly parsed models cannot rebase "
            "checkpoint origin-indexed metadata."
        )