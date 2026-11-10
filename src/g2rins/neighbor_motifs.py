# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Streaming centered-neighborhood counts, before representative selection.

Only aggregate rows survive an update. Each row's local topology includes the
center's incident edges and unique neighbor occurrences, not edges between
neighbors. Cut edges retain their occupied port and bond category. Unoccupied
ports are distinct from these boundary connections.
"""

import json


def _sites_token(sites):
    if not sites:
        return "null"
    if len(sites) == 1:
        return str(sites[0])
    return "{" + ",".join(map(str, sites)) + "}"


def centered_motif_rows(record):
    """Yield one detached, occurrence-ID-independent row per central unit."""
    units = dict(record["occurrences"])
    roots = {component["root"] for component in record["components"]}
    edges = record["edges"]
    adjacency = {oid: [] for oid in units}
    for eid, (left, ls, right, rs, _bond, _aromatic) in enumerate(edges):
        adjacency[left].append((int(ls), right, int(rs), eid))
        adjacency[right].append((int(rs), left, int(ls), eid))
    for center, unit in units.items():
        incident = sorted(adjacency[center])
        neighbors = sorted(
            (site, units[other], other_site,
             tuple(sorted(s for s, _n, _ns, eid in adjacency[other] if eid != edge_id)))
            for site, other, other_site, edge_id in incident
        )

        def neighbor_token(neighbor, reverse=False):
            _site, neighbor_unit, neighbor_site, outer_sites = neighbor
            outer = _sites_token(outer_sites)
            if reverse:
                return f"{outer}.{neighbor_unit}.{neighbor_site}"
            return f"{neighbor_site}.{neighbor_unit}.{outer}"

        if not neighbors:
            motif = f"null.{unit}.null"
        elif len(neighbors) == 1:
            neighbor = neighbors[0]
            if center in roots:
                motif = f"null.{unit}.{neighbor[0]}-{neighbor_token(neighbor)}"
            else:
                motif = f"{neighbor_token(neighbor, True)}-{neighbor[0]}.{unit}.null"
        elif len(neighbors) == 2:
            left, right = neighbors
            motif = (f"{neighbor_token(left, True)}-"
                     f"{left[0]}.{unit}.{right[0]}-{neighbor_token(right)}")
        else:
            motif = unit + "{" + ";".join(
                f"{neighbor[0]}:{neighbor_token(neighbor)}" for neighbor in neighbors
            ) + "}"
        center_position = (0 if not neighbors or (len(neighbors) == 1 and center in roots)
                           else 1 if len(neighbors) <= 2 else None)

        # Canonical port order assigns local IDs; repeated edges to one neighbor
        # and self-links reference existing local occurrences rather than copies.
        local_ids = {center: 0}
        for _site, other, _other_site, _eid in incident:
            if other not in local_ids:
                local_ids[other] = len(local_ids)
        local_edges = []
        incident_ids = {edge[-1] for edge in incident}
        for eid in incident_ids:
            left, ls, right, rs, bond, aromatic = edges[eid]
            a, b = local_ids[left], local_ids[right]
            if (a, int(ls)) > (b, int(rs)):
                a, ls, b, rs = b, rs, a, ls
            local_edges.append([a, int(ls), b, int(rs), bond, aromatic])
        boundary = sorted(
            [local_id, site, edges[eid][4], edges[eid][5]]
            for oid, local_id in local_ids.items()
            for site, _other, _other_site, eid in adjacency[oid]
            if eid not in incident_ids
        )
        yield {
            "center_unit": unit,
            "center_position": center_position,
            "motif": motif,
            "topology": {
                "units": [units[oid] for oid in local_ids],
                "edges": sorted(local_edges),
                "boundary": boundary,
            },
        }


class CenteredMotifStatistics:
    """Exact occurrence counts and at-least-once incidence per accepted chain."""

    def __init__(self, *, hydrogen_termination=None):
        self.accepted_count = 0
        self.rows = {}
        self.hydrogen_termination = hydrogen_termination

    def update(self, record, weight=1):
        seen = set()
        for row in centered_motif_rows(record):
            key = json.dumps(row, sort_keys=True, separators=(",", ":"))
            if key not in self.rows:
                self.rows[key] = {**row, "count": 0, "chain_count": 0,
                                  "retained_chain_count": 0}
            self.rows[key]["count"] += weight
            seen.add(key)
        for key in seen:
            self.rows[key]["chain_count"] += weight
            self.rows[key]["retained_chain_count"] += 1
        self.accepted_count += weight

    def finalize(self):
        return {
            "schema": "centered-motif-population/v1",
            **({"hydrogen_termination": self.hydrogen_termination}
               if self.hydrogen_termination is not None else {}),
            "accepted_count": self.accepted_count,
            "rows": [json.loads(json.dumps(self.rows[key])) for key in sorted(self.rows)],
        }