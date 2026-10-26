# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Accepted-record reduction, separate from sampling and retry orchestration."""

from collections import Counter


def _assemble_ensemble_data(
    creator,
    records,
    materialize_units,
    *,
    record_type,
    result_type,
    bond_records_from_labeled,
    bond_records,
):
    """Assemble metadata while keeping result classes and hooks in their module.

    The creator supplies legacy origin lookup and cached unit materialization.
    Those attributes are accessed lazily, just as in the original coordinator.
    """
    molecules = [record["molecule"] for record in records]
    molecular_weights = [record["molecular_weight"] for record in records]
    bond_counts = Counter()
    labeled_bond_counts = Counter()
    unit_counts = Counter()
    mol_weight_lists = {}
    ensemble_distributions = {}
    sequences = []

    for record in records:
        if isinstance(record, record_type):
            unit_counts.update(record.unit_counts or {})
            labeled_bond_counts.update(record.contact_counts or {})
        else:
            for unit, count in (record["molecule_units"] or {}).items():
                unit_id = creator._unit_id_by_origin[
                    next(iter(unit.nodes(data=True)))[1]["origin_idx"]
                ]
                unit_counts[unit_id] += count
            bond_counts.update(record["bonds"])
        for stochastic_id, weights in record["mol_weights"].items():
            mol_weight_lists.setdefault(stochastic_id, []).extend(weights)
        for stochastic_id, distribution in record["distributions"].items():
            ensemble_distributions.setdefault(stochastic_id, distribution)
        sequences.append(record["sequences"])

    canonical_units = (
        creator._materialize_unit_counts(unit_counts)
        if materialize_units
        else {
            unit_id: {"count": count}
            for unit_id, count in unit_counts.items()
        }
    )
    return result_type(
        chains=molecules,
        units=canonical_units,
        bonds=(
            bond_records_from_labeled(
                labeled_bond_counts,
                getattr(creator, "_origin_node_by_text", {}),
            )
            if labeled_bond_counts
            else bond_records(
                bond_counts,
                creator._origin_endpoint,
                getattr(creator, "_origin_node_by_text", {}),
            )
        ),
        sequences=sequences,
        sequence_unit_ids=(
            [record.sequence_unit_ids for record in records]
            if all(isinstance(record, record_type) and record.sequence_unit_ids is not None
                   for record in records)
            else None
        ),
        mol_weights=mol_weight_lists,
        distributions=ensemble_distributions,
        molecular_weights=molecular_weights,
    )