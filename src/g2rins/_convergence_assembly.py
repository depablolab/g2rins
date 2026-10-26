# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Convergence payload assembly, separate from sampling and checkpoint restore.

Result classes and legacy helper hooks are supplied by the coordinator. Creator
attributes remain lazy, so disabled metadata does not require unit prototypes or
model identity. No objects from this module enter worker or checkpoint transport.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _ConvergencePayload:
    """Retained outputs and sidecars, without copying their mutable contents."""

    retained: list
    chains: list
    sequences: list
    masses: list
    retained_bond_sequence_payloads: list
    retained_bond_sequence_indices: list
    retained_bond_sequence_weights: Any
    retained_bond_sequence_parents: Any
    bond_sequence_metadata: Any
    retained_mol_weights: dict
    chain_motif_index: Any
    motif_population_statistics: Any
    returned_representative_counts: Any


def _assemble_convergence_payload(
    creator,
    *,
    retain_motif_membership,
    metadata,
    reservoir_size,
    representative_distance,
    aggregate_motif_occurrences,
    retained_motif_by_index,
    accepted_count,
    aggregate_motif_presence,
    representative_counts,
    retained_records,
    retain_chains,
    retain_sequences,
    retain_samples,
    retain_bond_sequences,
    retained_topology_by_index,
    bond_sequence_format,
    motif_dictionary_from_counts,
    motif_population_statistics_from_counts,
    bond_sequence_export_metadata,
):
    """Assemble retained payloads in their original evaluation/mutation order."""
    chain_motif_index = None
    motif_population_statistics = None
    if retain_motif_membership and metadata:
        retention = "all"
        if reservoir_size is not None:
            retention = "reservoir"
        elif representative_distance is not None:
            retention = "cover"
        dictionary, lookup = motif_dictionary_from_counts(aggregate_motif_occurrences, creator._canonical_unit_info)
        rows = []
        for chain_index in sorted(retained_motif_by_index):
            entry = retained_motif_by_index[chain_index]
            motif_counts = [
                [lookup[canonical], int(count)]
                for canonical, count in sorted(entry["_counts"].items(), key=lambda item: lookup[item[0]])
                if count > 0
            ]
            rows.append({
                "chain_index": chain_index,
                "molar_mass": float(entry["molar_mass"]),
                "motif_counts": motif_counts,
                "motif_exposure": int(entry["motif_exposure"]),
            })
        chain_motif_index = {
            "schema": "chain-motif-membership/v1",
            "population": "retained-samples",
            "accepted_count": int(accepted_count),
            "retained_count": len(rows),
            "retention": retention,
            "retained_payload_status": "available",
            "mass_unit": "g/mol",
            "motif_dictionary_id": "unit-graph-simple-paths/v2",
            "rows": rows,
        }
        motif_population_statistics = motif_population_statistics_from_counts(
            accepted_count,
            aggregate_motif_occurrences,
            aggregate_motif_presence,
            creator._canonical_unit_info,
        )
    returned_representative_counts = None
    if (
        representative_distance is not None
        and len(representative_counts) == len(retained_records)
    ):
        retained_with_counts = sorted(
            zip(retained_records, representative_counts),
            key=lambda item: item[0][0],
        )
        retained_records = [entry for entry, _count in retained_with_counts]
        returned_representative_counts = [
            count for _entry, count in retained_with_counts
        ]
    else:
        retained_records.sort(key=lambda item: item[0])
    retained = [record for _index, record in retained_records]
    chains = [record.molecule for record in retained] if retain_chains else []
    sequences = (
        [record.sequences for record in retained]
        if retain_sequences
        else []
    )
    masses = (
        [record.molecular_weight for record in retained]
        if retain_samples
        else []
    )
    retained_bond_sequence_payloads = []
    retained_bond_sequence_indices = []
    retained_bond_sequence_weights = None
    retained_bond_sequence_parents = None
    bond_sequence_metadata = None
    if retain_bond_sequences and metadata:
        retention = "all"
        weight_semantics = "retained-observation"
        if reservoir_size is not None:
            retention = "reservoir"
            weight_semantics = "not-applicable"
        elif representative_distance is not None:
            retention = "cover"
            weight_semantics = "accepted-coverage"
        retained_bond_sequence_indices = [chain_index for chain_index, _ in retained_records]
        retained_bond_sequence_payloads = [
            retained_topology_by_index.get(chain_index)
            for chain_index in retained_bond_sequence_indices
        ]
        if any(payload is None for payload in retained_bond_sequence_payloads):
            raise RuntimeError(
                "A selected chain is missing its retained bond-sequence topology"
            )
        if retention == "all":
            retained_bond_sequence_weights = [1] * len(retained_bond_sequence_payloads)
        elif retention == "cover":
            if (
                returned_representative_counts is None
                or len(returned_representative_counts)
                != len(retained_bond_sequence_payloads)
            ):
                raise RuntimeError(
                    "Retained bond-sequence topology is not aligned with coverage weights"
                )
            retained_bond_sequence_weights = list(returned_representative_counts)
        elif retained_bond_sequence_payloads:
            retained_bond_sequence_weights = None
        bond_sequence_metadata = bond_sequence_export_metadata(
            retained_bond_sequence_payloads,
            retention=retention,
            weight_semantics=weight_semantics,
            retained_payload_status="available",
            accepted_count=accepted_count,
            unit_info=creator._canonical_unit_info,
            model_identity=creator._generative_graph.graph.get("g2rins_string", ""),
        )
        if bond_sequence_format == "paths":
            from .bond_sequences import bond_sequence_path_metadata, render_bond_sequence_paths

            retained_bond_sequence_payloads = [
                render_bond_sequence_paths(payload)
                for payload in retained_bond_sequence_payloads
            ]
            bond_sequence_metadata = bond_sequence_path_metadata(bond_sequence_metadata)
        elif bond_sequence_format == "branches":
            from .bond_sequences import bond_sequence_branch_metadata, render_bond_sequence_branches

            branch_records = [
                render_bond_sequence_branches(payload)
                for payload in retained_bond_sequence_payloads
            ]
            retained_bond_sequence_payloads = [segments for segments, _ in branch_records]
            retained_bond_sequence_parents = [parents for _, parents in branch_records]
            bond_sequence_metadata = bond_sequence_branch_metadata(bond_sequence_metadata)
    retained_mol_weights = {}
    if metadata:
        for record in retained:
            for stochastic_id, weights in record.mol_weights.items():
                retained_mol_weights.setdefault(stochastic_id, []).extend(weights)
    return _ConvergencePayload(
        retained=retained,
        chains=chains,
        sequences=sequences,
        masses=masses,
        retained_bond_sequence_payloads=retained_bond_sequence_payloads,
        retained_bond_sequence_indices=retained_bond_sequence_indices,
        retained_bond_sequence_weights=retained_bond_sequence_weights,
        retained_bond_sequence_parents=retained_bond_sequence_parents,
        bond_sequence_metadata=bond_sequence_metadata,
        retained_mol_weights=retained_mol_weights,
        chain_motif_index=chain_motif_index,
        motif_population_statistics=motif_population_statistics,
        returned_representative_counts=returned_representative_counts,
    )


def _assemble_convergence_checkpoint(
    creator,
    *,
    next_chain_index,
    accepted_count,
    batch_index,
    seed,
    mass_sum,
    mass_square_sum,
    contact_counts,
    contact_total,
    aggregate_unit_counts,
    aggregate_bond_counts,
    aggregate_distributions,
    tracker,
    retained_records,
    checkpoint_policy,
    retained_seen,
    reservoir_rng,
    checkpoint_settings,
    representative_features,
    representative_counts,
    aggregate_unit_path_counts,
    aggregate_unit_path_diagnostics,
    aggregate_unit_junction_counts,
    unit_junctions,
    checkpoint_type,
    result_type,
    bond_records_from_labeled,
    deepcopy,
):
    """Build the existing checkpoint, preserving constructor and copy order."""
    return checkpoint_type(
        next_chain_index=next_chain_index,
        accepted_count=accepted_count,
        batch_index=batch_index,
        seed=seed,
        mass_sum=mass_sum,
        mass_square_sum=mass_square_sum,
        contact_counts=dict(contact_counts),
        contact_total=contact_total,
        aggregate=result_type(
            [],
            {
                unit_id: {"count": count}
                for unit_id, count in aggregate_unit_counts.items()
            },
            bond_records_from_labeled(
                aggregate_bond_counts,
                getattr(creator, "_origin_node_by_text", {}),
            ),
            [],
            {},
            dict(aggregate_distributions),
        ),
        convergence_history=deepcopy(tracker.history),
        retained_records=(
            tuple(deepcopy(retained_records))
            if checkpoint_policy == "full"
            else ()
        ),
        retained_seen=retained_seen,
        reservoir_rng_state=(
            deepcopy(reservoir_rng.bit_generator.state)
            if reservoir_rng is not None
            else None
        ),
        settings=dict(checkpoint_settings),
        policy=checkpoint_policy,
        representative_features=tuple(
            deepcopy(representative_features)
        ),
        representative_counts=tuple(representative_counts),
        unit_path_counts={
            k: dict(counts)
            for k, counts in aggregate_unit_path_counts.items()
        },
        unit_path_diagnostics=dict(aggregate_unit_path_diagnostics),
        unit_junction_counts=dict(aggregate_unit_junction_counts) if unit_junctions else None,
    )