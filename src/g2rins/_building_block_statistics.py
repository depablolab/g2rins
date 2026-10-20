# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Exact streaming building-block statistics, independent of retained samples.

The caller opts in separately with ``building_block_statistics=True`` and calls
``update`` once per accepted sample, including empty samples. Only the shared
occurrence view is consumed; no molecular graph or chemical toolkit is needed.

A neighborhood is a root-incident covalent multistar, NOT an induced radius-one
graph: edges between neighbors are absent. Parallel arms to one occurrence form
one group, distinct occurrences remain distinct groups even with identical unit
labels, and self-links form a center group containing both oriented arms. Unit
labels are identities, not chemical equivalence classes. Association bonds (0)
are validated but excluded. The caller must supply every physical edge once,
including self-links, in ``occurrence_view[1]``.

Keys contain only labels and sorted port/bond tuples, never occurrence IDs.
Only aggregate counters survive an update. Finalization creates detached,
JSON-safe snapshots; motif IDs are deterministic sorted, snapshot-local ordinals
(not durable identifiers across updates that introduce new motifs).
"""

from collections import Counter


def _positive_site(value):
    """Accept only canonical, positive ASCII decimal strings, without coercion."""
    return (
        type(value) is str
        and bool(value)
        and value[0] in "123456789"
        and all(character in "0123456789" for character in value)
    )


class _BuildingBlockStatistics:
    """Accumulate all-accepted occurrence, multistar, and directed-site counts."""

    def __init__(self):
        self.chain_count = 0
        self._units = Counter()
        self._unit_presence = Counter()
        self._neighborhoods = Counter()
        self._neighborhood_presence = Counter()
        self._endpoints = Counter()
        self._endpoint_presence = Counter()

    def update(self, metadata, occurrence_view):
        """Validate and stage one sample before mutating any aggregate counter.

        ``metadata.occurrences`` contains objects with nonempty string ``unit_id``
        labels. ``occurrence_view`` is ``(adjacency, edges, atom_owner)``; only
        edges are read, since adjacency may deliberately omit self-links. Each
        edge is ``(left_index, right_index, (bond_type, aromatic, left_site,
        right_site))`` with strict integer indices (not bool), bond strings 0–4,
        boolean aromatic flags and canonical positive decimal site strings.
        Malformed samples raise ``ValueError`` without changing prior counts.
        """
        try:
            labels = tuple(occurrence.unit_id for occurrence in metadata.occurrences)
        except (AttributeError, TypeError) as exc:
            raise ValueError("metadata must provide occurrences with unit_id labels") from exc
        if any(type(label) is not str or not label for label in labels):
            raise ValueError("unit_id must be a nonempty string")
        if not isinstance(occurrence_view, tuple) or len(occurrence_view) != 3:
            raise ValueError("occurrence_view must be (adjacency, edges, atom_owner)")
        try:
            edges = iter(occurrence_view[1])
        except TypeError as exc:
            raise ValueError("occurrence edges must be iterable") from exc

        # Occurrence IDs are temporary grouping addresses, not motif identity.
        incident = [{} for _ in labels]
        endpoints = Counter()
        for edge in edges:
            if not isinstance(edge, (tuple, list)) or len(edge) != 3:
                raise ValueError("edge must contain two occurrence indices and a token")
            left, right, token = edge
            if any(type(index) is not int or not 0 <= index < len(labels)
                   for index in (left, right)):
                raise ValueError("edge endpoint index must identify an occurrence")
            if not isinstance(token, (tuple, list)) or len(token) != 4:
                raise ValueError("edge token must contain bond, aromatic, and two sites")
            bond, aromatic, left_site, right_site = token
            if type(bond) is not str or bond not in ("0", "1", "2", "3", "4"):
                raise ValueError("bond_type must be a string from 0 through 4")
            if type(aromatic) is not bool:
                raise ValueError("aromatic must be a boolean")
            if not _positive_site(left_site) or not _positive_site(right_site):
                raise ValueError("sites must be canonical positive decimal strings")
            if bond == "0":
                continue
            for source, target, source_site, target_site in (
                (left, right, left_site, right_site),
                (right, left, right_site, left_site),
            ):
                arm = (source_site, target_site, bond, aromatic)
                incident[source].setdefault(target, []).append(arm)
                endpoints[(labels[source], source_site, labels[target],
                           target_site, bond, aromatic)] += 1

        units = Counter(labels)
        neighborhoods = Counter()
        for center, groups in enumerate(incident):
            neighbors = tuple(sorted(
                (labels[neighbor], neighbor == center, tuple(sorted(arms)))
                for neighbor, arms in groups.items()
            ))
            neighborhoods[(labels[center], neighbors)] += 1

        self._units.update(units)
        self._unit_presence.update(units.keys())
        self._neighborhoods.update(neighborhoods)
        self._neighborhood_presence.update(neighborhoods.keys())
        self._endpoints.update(endpoints)
        self._endpoint_presence.update(endpoints.keys())
        self.chain_count += 1

    def merge(self, other):
        """Add a worker's raw counters, not its normalized probabilities.

        Workers contribute one successful chain at a time. Only results accepted
        by the ordered parent loop are merged, so retries, failed jobs, and
        unconsumed results cannot inflate presence counts. Final motif IDs and
        site denominators are assigned once from the combined population.
        """
        if not isinstance(other, _BuildingBlockStatistics):
            raise TypeError("expected building-block statistics counters")
        self._units.update(other._units)
        self._unit_presence.update(other._unit_presence)
        self._neighborhoods.update(other._neighborhoods)
        self._neighborhood_presence.update(other._neighborhood_presence)
        self._endpoints.update(other._endpoints)
        self._endpoint_presence.update(other._endpoint_presence)
        self.chain_count += other.chain_count

    def finalize(self):
        """Return a fresh deterministic JSON-safe aggregate, without mutation.

        Site probabilities condition on occupied source-site endpoints, not
        chains or unit occurrences. Every covalent physical edge contributes
        both orientations, including two endpoints for homotypic/self edges.
        """
        neighborhoods = []
        for ordinal, (key, count) in enumerate(sorted(self._neighborhoods.items()), 1):
            center, groups = key
            neighborhoods.append({
                "motif_id": f"bbn:{ordinal}",
                "center_unit": center,
                "neighbors": [
                    {
                        "unit_id": unit,
                        "is_center": is_center,
                        "connections": [
                            {"source_site": source, "target_site": target,
                             "bond_type": bond, "aromatic": aromatic}
                            for source, target, bond, aromatic in arms
                        ],
                    }
                    for unit, is_center, arms in groups
                ],
                "occurrence_count": count,
                "chain_presence_count": self._neighborhood_presence[key],
            })

        occupied = Counter()
        for key, count in self._endpoints.items():
            occupied[key[:2]] += count
        site_connections = []
        for key, count in sorted(self._endpoints.items()):
            source, source_site, target, target_site, bond, aromatic = key
            denominator = occupied[key[:2]]
            site_connections.append({
                "source_unit": source, "source_site": source_site,
                "target_unit": target, "target_site": target_site,
                "bond_type": bond, "aromatic": aromatic,
                "endpoint_count": count,
                "chain_presence_count": self._endpoint_presence[key],
                "occupied_endpoint_count": denominator,
                "probability": count / denominator,
            })

        return {
            "schema": "building-block-statistics/v1",
            "population": "all-accepted-samples",
            "accepted_count": self.chain_count,
            "neighborhood_definition": "root-incident-covalent-multistar/v1",
            "probability_definition": "conditional-on-occupied-source-site-endpoints",
            "building_blocks": [
                {"unit_id": unit, "occurrence_count": count,
                 "chain_presence_count": self._unit_presence[unit]}
                for unit, count in sorted(self._units.items())
            ],
            "neighborhoods": neighborhoods,
            "site_connections": site_connections,
        }