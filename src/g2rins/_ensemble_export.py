# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Fixed-size JSON projection, without file I/O or sampler dependencies."""

import networkx as nx

from .generative_graph import (
    _AROMATIC_NAME,
    _BOND_TYPE_NAME,
    _CONNECTOR_PLACEHOLDER_NAME,
)


def _fixed_ensemble_json_data(
    generative_graph,
    ensemble_data,
    molecule_format,
    json_max_chains,
    origin_node_by_text,
    sequence_unit_to_smiles,
    json_safe,
    graph_json_data,
    *,
    retain_motif_membership=False,
    building_block_statistics=False,
):
    """Build and normalize the entire payload before the caller opens a file.

    Serializers are explicit so the coordinator's existing private hooks remain
    effective. Only chains are sliced; sequences and statistics remain complete.
    """
    if molecule_format == "smiles":

        def _chain_json(molecule):
            return molecule

        def _sequence_unit_smiles(unit):
            return unit

    else:

        def _chain_json(molecule):
            data = nx.node_link_data(molecule, edges="edges")
            node_attributes = (
                "atomic_num",
                _CONNECTOR_PLACEHOLDER_NAME,
                _AROMATIC_NAME,
                "charge",
                "num_explicit_h",
                "origin_idx",
            )
            for node_data in data["nodes"]:
                original = dict(node_data)
                node_data.clear()
                node_data["id"] = original["id"]
                for name in node_attributes:
                    if name in original:
                        value = original[name]
                        if name == "origin_idx":
                            value = origin_node_by_text.get(str(value), value)
                        node_data[name] = value
            for edge_data in data["edges"]:
                original = dict(edge_data)
                edge_data.clear()
                edge_data["source"] = original["source"]
                edge_data["target"] = original["target"]
                for name in (_BOND_TYPE_NAME, _AROMATIC_NAME):
                    if name in original:
                        edge_data[name] = original[name]
            return data

        def _sequence_unit_smiles(unit):
            return sequence_unit_to_smiles(unit)

    list_of_molecules = ensemble_data.chains
    saved_chains = (
        list_of_molecules
        if json_max_chains is None
        else list_of_molecules[:json_max_chains]
    )
    json_data = {
        "string": json_safe(
            generative_graph.graph.get("g2rins_string", ""),
            "$['string']",
        )
    }
    json_data.update(graph_json_data(generative_graph))
    json_data["format"]["chain_format"] = molecule_format
    json_data["ensemble"] = json_safe(
        {
            "units": {
                unit_id: {
                    **info,
                    "subgraph": nx.node_link_data(info["subgraph"], edges="edges"),
                }
                for unit_id, info in ensemble_data.units.items()
            },
            "chains": [_chain_json(molecule) for molecule in saved_chains],
            "bonds": ensemble_data.bonds,
            "mol_weights": ensemble_data.mol_weights,
            "distributions": ensemble_data.distributions,
            "sequences": [
                [
                    [_sequence_unit_smiles(unit) for unit in sequence]
                    for sequence in chain_sequences
                ]
                for chain_sequences in ensemble_data.sequences
            ],
            "sequence_unit_ids": ensemble_data.sequence_unit_ids,
        },
        "$['ensemble']",
    )
    if retain_motif_membership:
        json_data["ensemble"]["chain_motif_index"] = ensemble_data.chain_motif_index
        json_data["ensemble"]["motif_population_statistics"] = (
            ensemble_data.motif_population_statistics
        )
    if building_block_statistics:
        json_data["ensemble"]["building_block_statistics"] = (
            ensemble_data.building_block_statistics
        )
    return json_safe(json_data)
