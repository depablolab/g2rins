# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Deterministic, subprocess-isolated sampling benchmark.

Run from the repository root, for example::

    python -m benchmarks.benchmark_sampling --case small-linear --runs 3

The command emits JSON. Each run uses a fresh process so ``peak_rss_mib`` is
comparable across cases and metadata modes. Wall-clock values are descriptive;
regression gates should compare relative medians on the same machine.
"""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

import g2rins
import g2rins.ensemble_creator as ensemble_module
from g2rins.ensemble_creator import _GrowthTransaction, _PartialAtomGraph, _attempt_chain
from g2rins.nx_rdkit_mol import rdkit_mol_to_smiles

from .cases import CASES

_TERMINATION_FLAGS = {"exact": None, "overshoot": 0, "undershoot": 1}


def _elapsed(callable_):
    start = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - start


def _peak_rss_mib() -> float:
    # Linux reports KiB; macOS reports bytes. G2RINS CI and reference profiling
    # are Linux, but retaining this branch makes local output unsurprising.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return rss / divisor


def _worker(case_name: str, metadata: bool, termination: str, max_discards: int) -> dict:
    case = CASES[case_name]
    parsed, parse_seconds = _elapsed(lambda: g2rins.G2rins.make(case.g2rins))
    graph_creator, graph_creator_seconds = _elapsed(parsed.get_graph_creator)
    creator, ensemble_creator_seconds = _elapsed(graph_creator.get_ensemble_creator)

    legacy_checkpoint_copies = 0
    transaction_checkpoints = 0
    original_deepcopy = ensemble_module.copy.deepcopy
    original_transaction_init = _GrowthTransaction.__init__

    def counted_deepcopy(value, memo=None):
        nonlocal legacy_checkpoint_copies
        if isinstance(value, _PartialAtomGraph):
            legacy_checkpoint_copies += 1
        if memo is None:
            return original_deepcopy(value)
        return original_deepcopy(value, memo)

    def counted_transaction_init(transaction, *args, **kwargs):
        nonlocal transaction_checkpoints
        transaction_checkpoints += 1
        original_transaction_init(transaction, *args, **kwargs)

    trace = []
    previous_trace = ensemble_module._DECISION_TRACE
    ensemble_module._DECISION_TRACE = trace
    ensemble_module.copy.deepcopy = counted_deepcopy
    _GrowthTransaction.__init__ = counted_transaction_init
    discard_reasons = Counter()
    deferred_warnings = []
    sample = None
    rng = np.random.default_rng(np.random.SeedSequence(case.seed).spawn(1)[0])
    start = time.perf_counter()
    try:
        for discard_attempts in range(max_discards):
            sample, reasons, _cause, attempt_warnings = _attempt_chain(
                creator,
                metadata,
                _TERMINATION_FLAGS[termination],
                rng,
            )
            deferred_warnings.extend(attempt_warnings)
            if sample is not None:
                break
            discard_reasons.update(reasons)
        sampling_seconds = time.perf_counter() - start
    finally:
        _GrowthTransaction.__init__ = original_transaction_init
        ensemble_module.copy.deepcopy = original_deepcopy
        ensemble_module._DECISION_TRACE = previous_trace

    if sample is None:
        return {
            "case": case_name,
            "metadata": "full" if metadata else "none",
            "termination": termination,
            "parse_seconds": parse_seconds,
            "graph_creator_seconds": graph_creator_seconds,
            "ensemble_creator_seconds": ensemble_creator_seconds,
            "sampling_seconds": sampling_seconds,
            "accepted": False,
            "discard_attempts": max_discards,
            "discard_reasons": dict(discard_reasons),
            "checkpoint_copies": legacy_checkpoint_copies + transaction_checkpoints,
            "transaction_checkpoints": transaction_checkpoints,
            "whole_molecule_checkpoint_copies": legacy_checkpoint_copies,
            "rollbacks": sum(event.get("kind") == "crossing" and not event.get("adopt_overshoot", True) for event in trace),
            "peak_rss_mib": _peak_rss_mib(),
        }

    graph = sample[0] if metadata else sample
    molecule, rdkit_construction_seconds = _elapsed(lambda: g2rins.mol_graph_to_rdkit_mol(graph))
    _, sanitization_seconds = _elapsed(lambda: Chem.SanitizeMol(molecule))
    smiles, serialization_seconds = _elapsed(lambda: rdkit_mol_to_smiles(molecule))
    payload, ipc_serialization_seconds = _elapsed(lambda: pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))

    unit_occurrences = None
    compact_sequences = None
    if metadata:
        unit_occurrences = sum(sample[1].values())
        compact_sequences = [len(sequence) for sequence in sample[3]]

    return {
        "case": case_name,
        "description": case.description,
        "seed": case.seed,
        "metadata": "full" if metadata else "none",
        "termination": termination,
        "parse_seconds": parse_seconds,
        "graph_creator_seconds": graph_creator_seconds,
        "ensemble_creator_seconds": ensemble_creator_seconds,
        "sampling_seconds": sampling_seconds,
        "rdkit_construction_seconds": rdkit_construction_seconds,
        "sanitization_seconds": sanitization_seconds,
        "serialization_seconds": serialization_seconds,
        "ipc_serialization_seconds": ipc_serialization_seconds,
        "accepted": True,
        "atoms": graph.number_of_nodes(),
        "bonds": graph.number_of_edges(),
        "molecular_weight": Descriptors.MolWt(molecule),
        "unit_occurrences": unit_occurrences,
        "sequence_lengths": compact_sequences,
        "checkpoint_copies": legacy_checkpoint_copies + transaction_checkpoints,
        "transaction_checkpoints": transaction_checkpoints,
        "whole_molecule_checkpoint_copies": legacy_checkpoint_copies,
        "rollbacks": sum(event.get("kind") == "crossing" and not event.get("adopt_overshoot", True) for event in trace),
        "discard_attempts": discard_attempts,
        "discard_reasons": dict(discard_reasons),
        "warnings": [caught.category.__name__ for caught in deferred_warnings],
        "ipc_bytes": len(payload),
        "canonical_smiles_bytes": len(smiles.encode("utf-8")),
        "peak_rss_mib": _peak_rss_mib(),
    }


def _median_records(records: list[dict]) -> dict:
    keys = set.intersection(*(set(record) for record in records))
    median = {}
    for key in sorted(keys):
        values = [record[key] for record in records]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            median[key] = statistics.median(values)
        elif all(value == values[0] for value in values):
            median[key] = values[0]
    return median


def _run_isolated(case_name: str, metadata: str, termination: str, max_discards: int) -> dict:
    command = [
        sys.executable,
        "-m",
        "benchmarks.benchmark_sampling",
        "--worker",
        "--case",
        case_name,
        "--metadata",
        metadata,
        "--termination",
        termination,
        "--max-discards",
        str(max_discards),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=sorted(CASES), help="Case to run; repeat for multiple cases.")
    parser.add_argument("--runs", type=int, default=3, help="Fresh-process repetitions per configuration.")
    parser.add_argument("--metadata", choices=("none", "full", "both"), default="both")
    parser.add_argument("--termination", choices=tuple(_TERMINATION_FLAGS), default="exact")
    parser.add_argument("--max-discards", type=int, default=100)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Emit medians without embedding every raw run.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.max_discards < 1:
        parser.error("--max-discards must be positive")

    case_names = args.case or ["small-linear"]
    if args.worker:
        if len(case_names) != 1 or args.metadata == "both":
            parser.error("worker mode requires one case and one metadata level")
        record = _worker(case_names[0], args.metadata == "full", args.termination, args.max_discards)
        print(json.dumps(record, sort_keys=True))
        return 0

    metadata_levels = ("none", "full") if args.metadata == "both" else (args.metadata,)
    output = {"runs": args.runs, "results": []}
    for case_name in case_names:
        for metadata in metadata_levels:
            records = [
                _run_isolated(case_name, metadata, args.termination, args.max_discards)
                for _ in range(args.runs)
            ]
            result = {
                "case": case_name,
                "metadata": metadata,
                "median": _median_records(records),
            }
            if not args.summary_only:
                result["runs"] = records
            output["results"].append(result)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
