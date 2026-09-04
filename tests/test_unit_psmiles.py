# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import warnings

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

import g2rins
from g2rins.ensemble_creator import EnsembleCreator
from g2rins.exception import InvalidUnitPSmiles, NoValidGenerationSource

PEI = "{[] [<]CCN([>])[>]; [<][H]; O[>], [<][H] []}|poisson(200)|"
HYPERBRANCHED_CH = "{[] [<][CH]([>])[>]; [<][H]; [>][H] []}|poisson(100)|"


def _make_creator(text):
    return g2rins.G2rins.make(text).get_graph_creator().get_ensemble_creator()


def _create_one(text, output_format="mol_graph", max_number_of_discarded_chains=100):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _make_creator(text).create_ensemble(
            1,
            output_format=output_format,
            ensemble_info=True,
            max_number_of_discarded_chains=max_number_of_discarded_chains,
            seed=0,
        )


def _canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return Chem.MolToSmiles(mol)


def _parse_with_explicit_hydrogens(smiles):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    return Chem.MolFromSmiles(smiles, params)


def _assert_public_unit_contract(creator, result):
    graph = creator._generative_graph
    labels = g2rins.derive_unit_labels(graph)
    for unit_id, info in result.units.items():
        template_nodes = [node for node in graph if labels.unit_id[node] == unit_id]
        expected_maps = sorted(labels.bond_id[node] for node in template_nodes if node in labels.bond_id)
        expected_real_atoms = sum(graph.nodes[node]["atomic_num"] > 0 for node in template_nodes)

        # Keep explicit [H] unit atoms: the default parser folds some of them
        # into neighboring atoms, which would make a serialized round-trip look
        # as though the renderer had lost a real template node.
        mol = _parse_with_explicit_hydrogens(info["psmiles"])
        assert mol is not None
        dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
        assert sorted(atom.GetAtomMapNum() for atom in dummy_atoms) == expected_maps
        assert all(atom.GetDegree() == 1 for atom in dummy_atoms)
        assert sum(atom.GetAtomicNum() > 0 for atom in mol.GetAtoms()) == expected_real_atoms


@pytest.mark.parametrize(
    ("text", "reference", "expected_neighbor_atomic_nums", "expected_real_atoms"),
    [
        (PEI, "[*:1]CCN([*:2])[*:3]", {1: 6, 2: 7, 3: 7}, 3),
        (HYPERBRANCHED_CH, "[CH]([*:1])([*:2])[*:3]", {1: 6, 2: 6, 3: 6}, 1),
    ],
)
def test_hyperbranched_unit_psmiles_uses_placeholders_as_stars(text, reference, expected_neighbor_atomic_nums, expected_real_atoms):
    creator = _make_creator(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = creator.create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)

    psmiles = result.units["R0"]["psmiles"]
    assert _canonical(psmiles) == _canonical(reference)
    mol = _parse_with_explicit_hydrogens(psmiles)
    dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    assert sorted(atom.GetAtomMapNum() for atom in dummy_atoms) == [1, 2, 3]
    assert all(atom.GetDegree() == 1 for atom in dummy_atoms)
    assert sum(atom.GetAtomicNum() > 0 for atom in mol.GetAtoms()) == expected_real_atoms
    assert {atom.GetAtomMapNum(): atom.GetNeighbors()[0].GetAtomicNum() for atom in dummy_atoms} == expected_neighbor_atomic_nums
    _assert_public_unit_contract(creator, result)


def test_unit_star_renderer_does_not_mutate_sampled_snapshot():
    creator = _make_creator(PEI)
    labels = g2rins.derive_unit_labels(creator._generative_graph)
    origin_bond_id = {str(node): bond_id for node, bond_id in labels.bond_id.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        units = creator.sample_mol_graph(rng=np.random.default_rng(0), molecule_info=True)[1]
    repeat_unit = next(unit for unit in units if labels.unit_id[next(iter(unit.nodes(data=True)))[1]["origin_idx"]] == "R0")
    nodes_before = [(node, dict(data)) for node, data in repeat_unit.nodes(data=True)]
    edges_before = [(u, v, dict(data)) for u, v, data in repeat_unit.edges(data=True)]

    star_graph = creator._unit_graph_with_stars(repeat_unit, origin_bond_id)

    assert [(node, dict(data)) for node, data in repeat_unit.nodes(data=True)] == nodes_before
    assert [(u, v, dict(data)) for u, v, data in repeat_unit.edges(data=True)] == edges_before
    assert any("connection" in data for _, data in star_graph.nodes(data=True))
    assert not any("connection" in data for _, data in repeat_unit.nodes(data=True))


def test_placeholder_bond_ids_and_bond_records_remain_unchanged():
    creator = _make_creator(PEI)
    labels = g2rins.derive_unit_labels(creator._generative_graph)
    graph_data = g2rins.generative_graph_json_data(creator._generative_graph)
    placeholder_bond_ids = sorted(node["bond_id"] for node in graph_data["graph"]["nodes"] if node["unit_id"] == "R0" and node["atomic_num"] == 0)
    assert placeholder_bond_ids == [2, 3]
    assert sorted(labels.bond_id[node] for node, data in creator._generative_graph.nodes(data=True) if labels.unit_id[node] == "R0" and data["atomic_num"] == 0) == [2, 3]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = creator.create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    r0_endpoints = {endpoint for record in result.bonds for endpoint in record["between"] if endpoint.startswith("R0.")}
    assert {"R0.2", "R0.3"} <= r0_endpoints


def test_sequence_phantom_contraction_preserves_junction_edge_attributes():
    unit = nx.Graph()
    unit.add_node("real", atomic_num=7)
    unit.add_node("phantom", atomic_num=0, origin_idx="split-site")
    unit.add_node("C0", atomic_num=0, connection=0, origin_idx="far-side-placeholder")
    unit.add_edge("real", "phantom", bond_type=1, aromatic=False, source="static")
    unit.add_edge("phantom", "C0", bond_type=2, aromatic=False, source="junction")

    EnsembleCreator._contract_sequence_phantoms([[unit]])

    assert "phantom" not in unit
    assert unit.has_edge("real", "C0")
    assert unit["real"]["C0"] == {"bond_type": 2, "aromatic": False, "source": "junction"}
    assert unit.nodes["C0"]["connection"] == 0

    edges_after_first_pass = list(unit.edges(data=True))
    EnsembleCreator._contract_sequence_phantoms([[unit]])
    assert list(unit.edges(data=True)) == edges_after_first_pass


def test_generated_sequence_graphs_have_only_mapped_connection_dummies():
    result = _create_one(PEI, output_format="mol_graph")
    dummy_count = 0
    for chain_sequences in result.sequences:
        for sequence in chain_sequences:
            for unit in sequence:
                for node, data in unit.nodes(data=True):
                    if data.get("atomic_num") != 0:
                        continue
                    dummy_count += 1
                    assert "connection" in data
                    assert unit.degree[node] == 1
                    neighbor = next(iter(unit.neighbors(node)))
                    assert unit.nodes[neighbor]["atomic_num"] > 0
    assert dummy_count > 0


@pytest.mark.parametrize("output_format", ["mol", "smiles"])
def test_converted_sequence_units_have_only_mapped_degree_one_dummies(output_format):
    result = _create_one(PEI, output_format=output_format)
    for chain_sequences in result.sequences:
        for sequence in chain_sequences:
            for unit in sequence:
                mol = unit if output_format == "mol" else Chem.MolFromSmiles(unit)
                assert mol is not None
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        assert atom.GetAtomMapNum() > 0
                        assert atom.GetDegree() == 1


def test_unit_psmiles_validator_accepts_the_reference_structure():
    mol = Chem.MolFromSmiles("[*:1]CCN([*:2])[*:3]")
    EnsembleCreator._validate_unit_psmiles_mol("R0", mol, [1, 2, 3], 3)


@pytest.mark.parametrize(
    ("smiles", "expected_maps", "expected_real_atoms"),
    [
        ("[*:1]CCN([*:2])[*:3].*", [1, 2, 3], 3),
        ("[*:1]CCN([*:2])[*:2]", [1, 2, 3], 3),
        ("C[*:1]C", [1], 2),
        ("[*:1]CCN([*:2])[*:3]", [1, 2, 3], 4),
    ],
)
def test_unit_psmiles_validator_rejects_invalid_public_units(smiles, expected_maps, expected_real_atoms):
    mol = Chem.MolFromSmiles(smiles)
    with pytest.raises(InvalidUnitPSmiles) as caught:
        EnsembleCreator._validate_unit_psmiles_mol("R0", mol, expected_maps, expected_real_atoms)
    assert caught.value.unit_id == "R0"
    assert caught.value.expected_maps == tuple(sorted(expected_maps))


def test_json_export_stops_before_writing_invalid_unit_psmiles(tmp_path, monkeypatch):
    creator = _make_creator(PEI)
    original = creator._unit_graph_with_stars

    def malformed_renderer(unit_graph, origin_bond_id):
        star_graph = original(unit_graph, origin_bond_id)
        star_graph.add_node(("invalid-unmapped-star", len(star_graph)), atomic_num=0, aromatic=False, charge=0)
        return star_graph

    monkeypatch.setattr(creator, "_unit_graph_with_stars", malformed_renderer)
    json_path = tmp_path / "invalid-ensemble.json"
    with warnings.catch_warnings(), pytest.raises(InvalidUnitPSmiles):
        warnings.simplefilter("ignore")
        creator.create_ensemble(1, output_format="smiles", json_file=str(json_path), seed=0)
    assert not json_path.exists()


def test_corpus_unit_psmiles_follow_template_contract(g2rins_list):
    for text in g2rins_list:
        creator = _make_creator(text)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = creator.create_ensemble(
                    1,
                    output_format="mol_graph",
                    ensemble_info=True,
                    max_number_of_discarded_chains=2,
                    seed=0,
                )
        except NoValidGenerationSource:
            # Some corpus entries intentionally have no default automatic source.
            continue
        assert result is not None
        _assert_public_unit_contract(creator, result)
