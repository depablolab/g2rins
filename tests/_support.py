# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Shared sampling recipes, builders, and independent statistics reference.

Keep this module independent of test modules so collection order cannot affect
the reusable helpers. The retained-record oracle must not use production
statistics accumulators or motif builders.
"""

from collections import Counter, defaultdict

import numpy as np
import pytest

import g2rins
import g2rins.util
from benchmarks.cases import CASES


FAST_SMI = "{[] [<]CC([>])c1ccccc1; CO[>]; [<][H] []}|gauss(1000, 45)|"

SOURCES = [
    pytest.param(FAST_SMI, id="fast-linear"),
    pytest.param(CASES["branched"].g2rins, id="branched"),
]
CONVERGENCE = {
    "batch_size": 2, "max_samples": 4, "window": 10,
    "output_format": "smiles", "seed": 7,
}


def _reset_rngs(seed):
    """Reset the two RNG streams the sampler consumes: numpy's legacy global
    (truncated/nested draws) and the library global Generator (top-level
    draws; plain get_global_rng(seed) does not reset an existing generator)."""
    np.random.seed(seed)
    g2rins.util._GLOBAL_RNG = np.random.default_rng(seed)


def _creator(source=FAST_SMI):
    return g2rins.G2rins.make(source).get_graph_creator().get_ensemble_creator()


def _ordinary_fields(result):
    # Compare everything else, including sequences, masses, path statistics,
    # convergence settings/trace, and the existing optional analytics fields.
    return {
        name: value for name, value in vars(result).items()
        if name != "building_block_statistics"
    }


def _bond_record_oracle(records):
    """Reconstruct root groups and directed endpoints from retained wire rows.

    This deliberately does not call the statistics accumulator, occurrence-view
    builder, or path-motif helpers. Occurrence IDs group neighbors within one
    chain only; stable native unit labels are used unchanged in aggregate keys.
    Comparing against raw population counts requires records retaining every
    occurrence: use non-H caps, since topology omits ordinary terminal H.
    """
    units, unit_presence = Counter(), Counter()
    stars, star_presence = Counter(), Counter()
    endpoints, endpoint_presence = Counter(), Counter()
    for record in records:
        labels = dict(record["occurrences"])
        local_units = Counter(labels.values())
        local_stars, local_endpoints = Counter(), Counter()
        for root, root_label in labels.items():
            groups = defaultdict(list)
            for left, left_site, right, right_site, bond, aromatic in record["edges"]:
                if bond == "0":
                    continue
                # Independent endpoint tests also preserve both self-link arms.
                if left == root:
                    groups[right].append((left_site, right_site, bond, aromatic))
                if right == root:
                    groups[left].append((right_site, left_site, bond, aromatic))
            signature = tuple(sorted(
                (labels[neighbor], neighbor == root, tuple(sorted(arms)))
                for neighbor, arms in groups.items()
            ))
            local_stars[root_label, signature] += 1
            for neighbor, arms in groups.items():
                for source_site, target_site, bond, aromatic in arms:
                    local_endpoints[
                        root_label, source_site, labels[neighbor], target_site,
                        bond, aromatic,
                    ] += 1
        units.update(local_units)
        unit_presence.update(local_units.keys())
        stars.update(local_stars)
        star_presence.update(local_stars.keys())
        endpoints.update(local_endpoints)
        endpoint_presence.update(local_endpoints.keys())

    occupied = Counter()
    for (source, site, *_rest), count in endpoints.items():
        occupied[source, site] += count
    return {
        "schema": "building-block-statistics/v1",
        "population": "all-accepted-samples",
        "accepted_count": len(records),
        "neighborhood_definition": "root-incident-covalent-multistar/v1",
        "probability_definition": "conditional-on-occupied-source-site-endpoints",
        "building_blocks": [
            {"unit_id": unit, "occurrence_count": count,
             "chain_presence_count": unit_presence[unit]}
            for unit, count in sorted(units.items())
        ],
        "neighborhoods": [
            {
                "motif_id": f"bbn:{ordinal}", "center_unit": center,
                "neighbors": [
                    {
                        "unit_id": unit, "is_center": is_center,
                        "connections": [
                            {"source_site": source, "target_site": target,
                             "bond_type": bond, "aromatic": aromatic}
                            for source, target, bond, aromatic in arms
                        ],
                    }
                    for unit, is_center, arms in groups
                ],
                "occurrence_count": count,
                "chain_presence_count": star_presence[center, groups],
            }
            for ordinal, ((center, groups), count) in enumerate(sorted(stars.items()), 1)
        ],
        "site_connections": [
            {
                "source_unit": source, "source_site": source_site,
                "target_unit": target, "target_site": target_site,
                "bond_type": bond, "aromatic": aromatic,
                "endpoint_count": count,
                "chain_presence_count": endpoint_presence[key],
                "occupied_endpoint_count": occupied[source, source_site],
                "probability": count / occupied[source, source_site],
            }
            for key, count in sorted(endpoints.items())
            for source, source_site, target, target_site, bond, aromatic in [key]
        ],
    }