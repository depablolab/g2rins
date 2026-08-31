# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Attribute deterministic sampler CPU time without treating it as wall time."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats

import numpy as np

import g2rins
from g2rins.ensemble_creator import _attempt_chain

from .benchmark_sampling import _TERMINATION_FLAGS
from .cases import CASES


def _profile_record(case_name, metadata, termination, max_discards, limit):
    case = CASES[case_name]
    creator = (
        g2rins.G2rins.make(case.g2rins)
        .get_graph_creator()
        .get_ensemble_creator()
    )
    rng = np.random.default_rng(np.random.SeedSequence(case.seed).spawn(1)[0])
    sample = None
    profiler = cProfile.Profile()
    profiler.enable()
    for _attempt in range(max_discards):
        sample, _reasons, _cause, _warnings = _attempt_chain(
            creator,
            metadata == "full",
            _TERMINATION_FLAGS[termination],
            rng,
        )
        if sample is not None:
            break
    profiler.disable()
    if sample is None:
        raise RuntimeError(
            f"{case_name} did not produce an accepted chain in {max_discards} attempts"
        )
    graph = sample[0] if metadata == "full" else sample

    statistics = pstats.Stats(profiler)
    rows = []
    categories = {
        "g2rins": 0.0,
        "networkx": 0.0,
        "rdkit_python": 0.0,
        "stdlib_and_other": 0.0,
    }
    for (filename, line, function), (_cc, calls, self_time, cumulative, _callers) in statistics.stats.items():
        normalized = filename.replace(os.sep, "/")
        if "/g2rins/" in normalized and "/src/g2rins/" in normalized:
            category = "g2rins"
        elif "/networkx/" in normalized:
            category = "networkx"
        elif "/rdkit/" in normalized:
            category = "rdkit_python"
        else:
            category = "stdlib_and_other"
        categories[category] += self_time
        rows.append(
            {
                "file": normalized,
                "line": line,
                "function": function,
                "calls": calls,
                "self_seconds": self_time,
                "cumulative_seconds": cumulative,
            }
        )

    rows.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
    total_self = sum(categories.values())
    return {
        "case": case_name,
        "metadata": metadata,
        "termination": termination,
        "atoms": graph.number_of_nodes(),
        "profile_total_self_seconds": total_self,
        "self_time_by_category": categories,
        "self_time_fraction_by_category": {
            category: seconds / total_self if total_self else 0.0
            for category, seconds in categories.items()
        },
        "top_cumulative": rows[:limit],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), default="large-linear")
    parser.add_argument("--metadata", choices=("none", "full"), default="none")
    parser.add_argument(
        "--termination",
        choices=("exact", "overshoot", "undershoot"),
        default="exact",
    )
    parser.add_argument("--max-discards", type=int, default=100)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args(argv)
    if args.max_discards < 1:
        parser.error("--max-discards must be positive")
    if args.limit < 1:
        parser.error("--limit must be positive")
    print(
        json.dumps(
            _profile_record(
                args.case,
                args.metadata,
                args.termination,
                args.max_discards,
                args.limit,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
