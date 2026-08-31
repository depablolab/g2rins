# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Create stable, node-id-independent behavioral records for seeded samples."""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

import g2rins
import g2rins.ensemble_creator as ensemble_module
from g2rins.ensemble_creator import EnsembleData

from .cases import CASES, REFERENCE_CASES

REFERENCE_PATH = Path(__file__).with_name("seeded_references.json")
_NODE_ATTRIBUTES = ("atomic_num", "aromatic", "charge", "num_explicit_h", "credited_h")
_EDGE_ATTRIBUTES = ("bond_type", "aromatic")


def _graph_signature(graph: nx.Graph) -> dict:
    """Return a topology/chemistry signature independent of raw node ids."""
    normalized = nx.Graph()
    for node, data in graph.nodes(data=True):
        normalized.add_node(node, signature=json.dumps([data.get(name) for name in _NODE_ATTRIBUTES], separators=(",", ":")))
    for left, right, data in graph.edges(data=True):
        normalized.add_edge(left, right, signature=json.dumps([data.get(name) for name in _EDGE_ATTRIBUTES], separators=(",", ":")))
    return {
        "atoms": graph.number_of_nodes(),
        "bonds": graph.number_of_edges(),
        "elements": {str(element): count for element, count in sorted(Counter(data["atomic_num"] for _, data in graph.nodes(data=True)).items())},
        "weisfeiler_lehman": nx.weisfeiler_lehman_graph_hash(normalized, node_attr="signature", edge_attr="signature", iterations=5),
    }


def _round_trace(trace: list[dict]) -> list[dict]:
    stable = []
    for event in trace:
        if event["kind"] != "crossing":
            continue
        stable.append(
            {
                key: (
                    round(float(value), 9)
                    if isinstance(value, (float, np.floating))
                    else value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for key, value in event.items()
                if key not in {"id"}
            }
        )
    return stable


def build_reference(case_name: str) -> dict:
    """Sample one chain and capture public outputs plus rounding decisions."""
    case = CASES[case_name]
    trace = []
    previous_trace = ensemble_module._DECISION_TRACE
    ensemble_module._DECISION_TRACE = trace
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            creator = g2rins.G2rins.make(case.g2rins).get_graph_creator().get_ensemble_creator()
            result = creator.create_ensemble(1, output_format="mol_graph", ensemble_info=True, seed=case.seed)
    finally:
        ensemble_module._DECISION_TRACE = previous_trace

    if not isinstance(result, EnsembleData) or len(result.chains) != 1:
        raise RuntimeError(f"reference case {case_name!r} did not produce exactly one chain")

    graph = result.chains[0]
    return {
        "seed": case.seed,
        "canonical_smiles": g2rins.mol_graph_to_smiles(graph),
        "graph": _graph_signature(graph),
        "molecular_weight": round(float(result.molecular_weights[0]), 9),
        "units": {
            unit_id: {
                "count": info["count"],
                "psmiles": info["psmiles"],
                "g2rins": info["g2rins"],
            }
            for unit_id, info in result.units.items()
        },
        "contacts": [
            {"labels": record["labels"], "count": record["count"]}
            for record in result.bonds
        ],
        "sequences": [
            [g2rins.mol_graph_to_smiles(unit, kekulize=False) for unit in sequence]
            for sequence in result.sequences[0]
        ],
        "warnings": [warning.category.__name__ for warning in caught],
        "rounding_trace": _round_trace(trace),
    }


def build_references() -> dict:
    return {name: build_reference(name) for name in REFERENCE_CASES}


def write_references(path: Path = REFERENCE_PATH) -> None:
    path.write_text(json.dumps(build_references(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_references()
