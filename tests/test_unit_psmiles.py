# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import copy
import json
import pickle
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

import g2rins
from g2rins.ensemble_creator import EnsembleCreator
from g2rins.exception import InvalidUnitPSmiles, NoValidGenerationSource
from g2rins.nx_rdkit_mol import mol_graph_to_rdkit_mol

PEI = "{[] [<]CCN([>])[>]; [<][H]; O[>], [<][H] []}|poisson(200)|"
HYPERBRANCHED_CH = "{[] [<][CH]([>])[>]; [<][H]; [>][H] []}|poisson(100)|"
PARTNERLESS_INITIATOR = "{[] [<]CCO[>]; N([>])([>])[>1]; [<][H] []}|poisson(300)|"
PARTNERLESS_REPEAT = "{[] [<]CCN([>])[>1]; O[>]; [<][H] []}|poisson(300)|"
PARTNERLESS_SINGLE_SITE = "{[] [<]CC([>1])O[>]; CO[>]; [<][H] []}|poisson(300)|"
PARTNERLESS_CASES = [
    pytest.param(PARTNERLESS_INITIATOR, "I0", "N([*:1])[*:2]", id="B1-split-initiator"),
    pytest.param(PARTNERLESS_REPEAT, "R0", "C(C[*:1])N[*:2]", id="B5-split-repeat"),
    pytest.param(PARTNERLESS_SINGLE_SITE, "R0", "C(C[*:1])O[*:2]", id="B4-single-site-control"),
]

# Keep every input, including ordinary units that can expose an over-strict
# validator. Names and expected outcomes are keyed by input text, so reordering
# smi.json cannot move the expected failure onto a different input.
CORPUS_CASES = json.loads(Path(__file__).with_name("unit_psmiles_corpus.json").read_text(encoding="utf-8"))
CORPUS_TEXTS = json.loads(Path(__file__).with_name("smi.json").read_text(encoding="utf-8"))["g2rins"]
CORPUS_ERRORS = {None: None, "NoValidGenerationSource": NoValidGenerationSource}


def test_corpus_metadata_covers_inputs():
    assert set(CORPUS_CASES) == set(CORPUS_TEXTS), "Give every corpus input a descriptive test ID and expected outcome."
    assert all(isinstance(case, dict) and isinstance(case.get("id"), str) for case in CORPUS_CASES.values()), "Each corpus case needs a string test ID."
    assert len({case["id"] for case in CORPUS_CASES.values()}) == len(CORPUS_CASES), "Corpus test IDs must be unique."


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


@pytest.mark.parametrize("text", [PEI, PARTNERLESS_REPEAT], ids=["active-sites", "inactive-site"])
def test_unit_star_renderer_does_not_mutate_sampled_snapshot(text):
    creator = _make_creator(text)
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
    unit.add_node("phantom", atomic_num=0, is_connector_placeholder=True, origin_idx="split-site")
    unit.add_node("C0", atomic_num=0, is_connector_placeholder=False, connection=0, origin_idx="far-side-placeholder")
    unit.add_edge("real", "phantom", bond_type=1, aromatic=False, source="static")
    unit.add_edge("phantom", "C0", bond_type=2, aromatic=False, source="junction")

    EnsembleCreator._contract_sequence_phantoms([[unit]], {"split-site": 1})

    assert "phantom" not in unit
    assert unit.has_edge("real", "C0")
    assert unit["real"]["C0"] == {"bond_type": 2, "aromatic": False, "source": "junction"}
    assert unit.nodes["C0"]["connection"] == 0
    assert unit.nodes["C0"]["is_connector_placeholder"] is False

    edges_after_first_pass = list(unit.edges(data=True))
    EnsembleCreator._contract_sequence_phantoms([[unit]], {"split-site": 1})
    assert list(unit.edges(data=True)) == edges_after_first_pass


def test_generated_sequence_graphs_mark_every_connection_site():
    result = _create_one(PEI, output_format="mol_graph")
    mapped_count = 0
    unmapped_count = 0
    for chain_sequences in result.sequences:
        for sequence in chain_sequences:
            for unit in sequence:
                for node, data in unit.nodes(data=True):
                    if data.get("atomic_num") != 0:
                        continue
                    # Both kinds are degree-one dummies hanging off a real atom.
                    assert unit.degree[node] == 1
                    neighbor = next(iter(unit.neighbors(node)))
                    assert unit.nodes[neighbor]["atomic_num"] > 0
                    if "connection" in data:
                        mapped_count += 1
                        assert data["is_connector_placeholder"] is False
                        assert set(data) == {"atomic_num", "is_connector_placeholder", "aromatic", "charge", "num_explicit_h", "origin_idx", "connection"}
                    else:
                        # A split site with no recorded connection keeps its
                        # placeholder: dropping it would render an interior
                        # fragment as a complete small molecule.
                        unmapped_count += 1
                        assert data["is_connector_placeholder"] is True
    assert mapped_count > 0
    assert unmapped_count > 0


@pytest.mark.parametrize("output_format", ["mol", "smiles"])
def test_converted_sequence_units_have_degree_one_dummies(output_format):
    result = _create_one(PEI, output_format=output_format)
    mapped_count = 0
    for chain_sequences in result.sequences:
        for sequence in chain_sequences:
            for unit in sequence:
                mol = unit if output_format == "mol" else Chem.MolFromSmiles(unit)
                assert mol is not None
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        assert atom.GetDegree() == 1
                        mapped_count += atom.GetAtomMapNum() > 0
    assert mapped_count > 0


def test_unit_psmiles_validator_accepts_the_reference_structure():
    mol = Chem.MolFromSmiles("[*:1]CCN([*:2])[*:3]")
    EnsembleCreator._validate_unit_psmiles_mol("R0", mol, [1, 2, 3], 3)


@pytest.mark.parametrize("copy_method", ["pickle", "deepcopy"])
def test_invalid_unit_psmiles_preserves_normalized_diagnostics(copy_method):
    error = InvalidUnitPSmiles("R0", iter([1, 2]), iter([0, 1, 2]), iter([(3, 0, 2)]), np.int64(4), np.int64(3))
    expected_args = ("R0", (1, 2), (0, 1, 2), ((3, 0, 2),), 4, 3)
    assert error.args == expected_args
    assert type(error.args[-2]) is int and type(error.args[-1]) is int
    restored = pickle.loads(pickle.dumps(error)) if copy_method == "pickle" else copy.deepcopy(error)
    assert restored.args == expected_args
    assert vars(restored) == vars(error)
    assert str(restored) == str(error)
    for field in ("unit_id", "expected_maps", "actual_maps", "invalid_dummy_degrees", "expected_real_atom_count", "actual_real_atom_count"):
        assert f"{field}={getattr(error, field)!r}" in str(error)


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


@pytest.mark.parametrize(("text", "unit_id", "reference"), PARTNERLESS_CASES)
def test_partnerless_connector_unit_psmiles(text, unit_id, reference):
    creator = _make_creator(text)
    result = creator.create_ensemble(1, output_format="mol_graph", ensemble_info=True, seed=0)
    assert _canonical(result.units[unit_id]["psmiles"]) == _canonical(reference)
    _assert_public_unit_contract(creator, result)
    assert all(data["atomic_num"] > 0 for chain in result.chains for _, data in chain.nodes(data=True))
    dummy_count = 0
    for sequence in result.sequences[0]:
        for unit in sequence:
            for node, data in unit.nodes(data=True):
                if data["atomic_num"] != 0:
                    continue
                dummy_count += 1
                assert unit.degree[node] == 1
                assert unit.nodes[next(iter(unit.neighbors(node)))]["atomic_num"] > 0
    assert dummy_count > 0


@pytest.mark.parametrize(
    ("text", "unit_id", "expected_fragment"),
    [
        pytest.param(PARTNERLESS_INITIATOR, "I0", "*N*", id="partnerless-initiator"),
        pytest.param(PARTNERLESS_REPEAT, "R0", "*NCC", id="partnerless-repeat"),
    ],
)
@pytest.mark.parametrize("output_format", ["mol_graph", "mol", "smiles"])
def test_partnerless_sequence_fragments_omit_inactive_sites(text, unit_id, expected_fragment, output_format):
    creator = _make_creator(text)
    labels = g2rins.derive_unit_labels(creator.generative_graph)
    inactive_origins = {str(node) for node, data in creator.generative_graph.nodes(data=True) if data.get("is_connector_placeholder") and node not in labels.bond_id}
    active_origins = {str(node) for node in labels.bond_id}
    assert len(inactive_origins) == 1
    result = creator.create_ensemble(1, output_format=output_format, ensemble_info=True, seed=0)
    _assert_public_unit_contract(creator, result)
    unit_mol = _parse_with_explicit_hydrogens(result.units[unit_id]["psmiles"])
    assert [atom.GetTotalNumHs() for atom in unit_mol.GetAtoms() if atom.GetAtomicNum() == 7] == [1]
    nitrogen_fragments = 0
    for sequence in result.sequences[0]:
        for unit in sequence:
            if output_format == "mol_graph":
                assert not inactive_origins.intersection(data["origin_idx"] for _, data in unit.nodes(data=True))
                for _, data in unit.nodes(data=True):
                    if data.get("is_connector_placeholder"):
                        assert data["origin_idx"] in active_origins
                mol = mol_graph_to_rdkit_mol(unit, kekulize=False)
            else:
                mol = Chem.Mol(unit) if output_format == "mol" else _parse_with_explicit_hydrogens(unit)
            assert mol is not None
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    assert atom.GetDegree() == 1
                    atom.SetAtomMapNum(0)
            nitrogens = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7]
            if nitrogens:
                nitrogen_fragments += 1
                assert [atom.GetTotalNumHs() for atom in nitrogens] == [1]
                assert Chem.MolToSmiles(mol) == _canonical(expected_fragment)
    assert nitrogen_fragments > 0


@pytest.mark.parametrize(("text", "unit_id", "reference"), PARTNERLESS_CASES)
def test_partnerless_connector_json_only_export(text, unit_id, reference, tmp_path):
    creator = _make_creator(text)
    path = tmp_path / "ensemble.json"
    chains = creator.create_ensemble(1, output_format="smiles", json_file=str(path), seed=0)
    exported = json.loads(path.read_text())
    assert exported["ensemble"]["chains"] == chains
    assert _canonical(exported["ensemble"]["units"][unit_id]["psmiles"]) == _canonical(reference)
    placeholders = [node for node in exported["graph"]["nodes"] if node["atomic_num"] == 0]
    if text == PARTNERLESS_SINGLE_SITE:
        assert not placeholders
    else:
        assert placeholders and all(node["is_connector_placeholder"] for node in placeholders)
        assert len([node for node in placeholders if "bond_id" not in node]) == 1
    assert creator.create_ensemble(1, output_format="smiles", seed=0) == chains


def test_partnerless_initiator_parallel_matches_serial():
    creator = _make_creator(PARTNERLESS_INITIATOR)
    serial = creator.create_ensemble(2, output_format="smiles", ensemble_info=True, seed=0)
    parallel = creator.create_ensemble(2, output_format="smiles", ensemble_info=True, seed=0, parallel=True, n_workers=2)
    assert parallel == serial
    _assert_public_unit_contract(creator, parallel)


def test_unit_renderer_drops_only_identified_inactive_placeholders():
    unit = nx.Graph()
    unit.add_node("real", atomic_num=7, origin_idx="real", is_connector_placeholder=False)
    unit.add_node("inactive", atomic_num=0, origin_idx="inactive", is_connector_placeholder=True)
    unit.add_node("wildcard", atomic_num=0, origin_idx="wildcard", is_connector_placeholder=False)
    unit.add_edges_from([("real", "inactive"), ("real", "wildcard")])
    rendered = EnsembleCreator._unit_graph_with_stars(unit, {})
    assert set(rendered) == {"real", "wildcard"}
    assert rendered.has_edge("real", "wildcard")
    assert set(unit) == {"real", "inactive", "wildcard"}


def test_sequence_finalizer_drops_only_identified_inactive_placeholders():
    unit = nx.Graph()
    unit.add_node("real", atomic_num=7, origin_idx="real", is_connector_placeholder=False)
    unit.add_node("inactive", atomic_num=0, origin_idx="inactive", is_connector_placeholder=True)
    unit.add_node("active", atomic_num=0, origin_idx="active", is_connector_placeholder=True)
    unit.add_node("wildcard", atomic_num=0, origin_idx="wildcard", is_connector_placeholder=False)
    unit.add_node("stub", atomic_num=0, origin_idx="far-side", is_connector_placeholder=False, connection=0)
    unit.add_edges_from(("real", node) for node in ("inactive", "active", "wildcard", "stub"))

    EnsembleCreator._contract_sequence_phantoms([[unit]], {"active": 1})

    assert set(unit) == {"real", "active", "wildcard", "stub"}
    assert set(unit.neighbors("real")) == {"active", "wildcard", "stub"}
    assert "connection" not in unit.nodes["active"]
    assert unit.nodes["stub"]["connection"] == 0


@pytest.mark.parametrize(
    "text",
    [pytest.param(text, id=case["id"] if isinstance(case, dict) and isinstance(case.get("id"), str) else f"corpus-{index}") for index, (text, case) in enumerate(CORPUS_CASES.items())],
)
def test_corpus_unit_psmiles_follow_template_contract(text):
    case = CORPUS_CASES[text]
    assert isinstance(case, dict), "Corpus case metadata must be an object."
    error_name = case.get("expected_error")
    assert error_name is None or isinstance(error_name, str), "Expected error must be a known name."
    assert error_name in CORPUS_ERRORS, f"Unknown expected error: {error_name!r}"
    expected_error = CORPUS_ERRORS[error_name]
    creator = _make_creator(text)
    if expected_error is not None:
        # This input intentionally has no default source. Any unexpected
        # NoValidGenerationSource from another case must fail the test.
        with pytest.raises(expected_error):
            creator.create_ensemble(1, ensemble_info=True, seed=0)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = creator.create_ensemble(1, output_format="mol_graph", ensemble_info=True, max_number_of_discarded_chains=2, seed=0)
    assert result is not None
    _assert_public_unit_contract(creator, result)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("{[] [<][NH2+]C[>]; [>][H]; [<][H] []}|uniform(100,100)|", id="charged-neighbor"),
        # poisson(600): a chain long enough that a junction between two ring
        # atoms is realized, which is what puts an aromatic bond on a stub.
        pytest.param("{[] [<]c1ccc([>])cc1[>]; [<][H]; [>][H] []}|poisson(600)|", id="aromatic-neighbor"),
    ],
)
@pytest.mark.parametrize("output_format", ["mol_graph", "mol", "smiles"])
def test_sequence_stubs_are_neutral_and_non_aromatic(text, output_format):
    creator = _make_creator(text)
    result = creator.create_ensemble(1, output_format=output_format, ensemble_info=True, seed=0)
    stub_count = 0
    for sequence in result.sequences[0]:
        for unit in sequence:
            if output_format == "mol_graph":
                for node, data in unit.nodes(data=True):
                    if "connection" in data:
                        stub_count += 1
                        assert data["atomic_num"] == 0
                        assert data["charge"] == 0
                        assert data["aromatic"] is False
                        assert data["num_explicit_h"] == -1
                        source = creator.generative_graph.nodes[data["origin_idx"]]
                        assert source["charge"] > 0 or source["aromatic"]
                        for neighbor in unit.neighbors(node):
                            assert unit[node][neighbor]["aromatic"] is False
            else:
                mol = unit if output_format == "mol" else _parse_with_explicit_hydrogens(unit)
                assert mol is not None
                # A stub bond left aromatic survives fragment-mode conversion and
                # only fails here, on the kekulizing sanitization a caller runs.
                Chem.SanitizeMol(Chem.Mol(mol))
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() > 0:
                        stub_count += 1
                        assert atom.GetFormalCharge() == 0
                        assert not atom.GetIsAromatic()
    assert stub_count > 0


@pytest.mark.parametrize(
    ("text", "expected", "collapsed"),
    [
        pytest.param("{[] [$][C-][$]; [$][H];[$][H] []}|gauss(300.,20.)|", "*[CH-]*", "[CH3-]", id="divalent-carbanion"),
        pytest.param("{[] [<]CC[>], [<]C(F)(F)[>]; [<][H]; [>][H] []}|poisson(200)|", "*C(*)(F)F", "FCF", id="difluoro-comonomer"),
    ],
)
def test_interior_sequence_fragments_keep_their_connection_sites(text, expected, collapsed):
    """An interior unit must not render as a complete small molecule.

    Dropping a split site the sampler did not record turns a divalent carbanion
    into methanide, inflating its hydrogen count, so composition computed from
    sequences is wrong while chains stay correct.
    """
    creator = _make_creator(text)
    result = creator.create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    fragments = {unit for group in result.sequences[0] for unit in group}
    assert expected in fragments
    assert collapsed not in fragments
