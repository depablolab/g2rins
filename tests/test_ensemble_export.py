# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Regression tests for ensemble JSON export and template provenance."""

import warnings

import numpy as np
import pytest

import g2rins
from tests._support import _reset_rngs


def test_create_ensemble_json_file(tmp_path):
    """The json_file export must be strict JSON (no NaN/numpy leakage) with the
    documented layout: {string, format, graph, ensemble}, derived unit/bond
    labels injected into the graph nodes, canonical unit entries with numbered
    P-SMILES stars and static unit subgraphs, undirected bond records pairing
    endpoint labels with generative-graph node ids, chains capped by
    json_max_chains while statistics keep every sampled chain."""
    import json

    smi = "{[] [<]CC([>])c1ccccc1; CO[>]; [<][H] []}|gauss(1000, 45)|"
    path = tmp_path / "ensemble.json"
    _reset_rngs(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
        chains = ensemble_creator.create_ensemble(3, output_format="smiles", json_file=str(path), json_max_chains=2)

    # json_file alone keeps the plain return shape.
    assert isinstance(chains, list) and len(chains) == 3

    def _reject_constant(name):
        raise AssertionError(f"non-strict JSON constant {name} in export")

    data = json.loads(path.read_text(), parse_constant=_reject_constant)

    assert list(data) == ["string", "format", "graph", "ensemble"]
    # The string is the regenerated canonical text, not the verbatim input.
    assert data["string"].startswith("{[] [<]CC([>])c1ccccc1;")
    assert data["format"]["version"] == 2
    assert data["format"]["derived_node_fields"] == ["unit_id", "bond_id"]
    assert data["format"]["chain_format"] == "smiles"

    labels = g2rins.derive_unit_labels(ensemble_creator._generative_graph)
    for node_dict in data["graph"]["nodes"]:
        assert node_dict["unit_id"] == labels.unit_id[node_dict["id"]]
        assert node_dict.get("bond_id") == labels.bond_id.get(node_dict["id"])

    ensemble = data["ensemble"]
    assert list(ensemble) == [
        "units", "chains", "bonds", "mol_weights", "distributions", "sequences",
        "sequence_unit_ids",
    ]

    assert len(ensemble["chains"]) == 2, "json_max_chains caps the stored chains"
    assert ensemble["chains"] == chains[:2]
    # Statistics always cover ALL sampled chains (one gauss instance per chain).
    assert all(len(mw_list) == 3 for mw_list in ensemble["mol_weights"].values())
    assert len(ensemble["sequences"]) == 3
    # Sequence labels, like sequences/statistics, are not chain-sliced.
    assert len(ensemble["sequence_unit_ids"]) == 3
    for sequences, unit_ids in zip(
        ensemble["sequences"], ensemble["sequence_unit_ids"], strict=True,
    ):
        assert [len(segment) for segment in unit_ids] == [
            len(segment) for segment in sequences
        ]
        assert all(unit_id in ensemble["units"] for segment in unit_ids for unit_id in segment)

    assert set(ensemble["units"]) == {"I0", "R0", "T0"}
    for unit_id, info in ensemble["units"].items():
        assert list(info) == ["psmiles", "g2rins", "subgraph", "count"]
        assert "[*:1]" in info["psmiles"]
        assert info["g2rins"] and info["count"] > 0
        # The subgraph is the unit's static piece of the generative graph:
        # the same node ids, static edges only, unit_id (only) on its nodes.
        subgraph_ids = {node_dict["id"] for node_dict in info["subgraph"]["nodes"]}
        assert subgraph_ids == {node for node, uid in labels.unit_id.items() if uid == unit_id}
        assert all(node_dict["unit_id"] == unit_id and "bond_id" not in node_dict for node_dict in info["subgraph"]["nodes"])
        # Exactly the template's static edges among the unit's nodes, with
        # their keys and data, and the template's node data plus unit_id.
        template = ensemble_creator._generative_graph
        expected_edges = {(u, v, key): data for u, v, key, data in template.edges(keys=True, data=True) if u in subgraph_ids and v in subgraph_ids and data["static"]}
        stored_edges = {(edge["source"], edge["target"], edge["key"]): edge for edge in info["subgraph"]["edges"]}
        assert set(stored_edges) == set(expected_edges)
        assert expected_edges or len(subgraph_ids) == 1
        for edge_key, expected in expected_edges.items():
            assert {name: value for name, value in stored_edges[edge_key].items() if name not in ("source", "target", "key")} == expected
        for node_dict in info["subgraph"]["nodes"]:
            assert {name: value for name, value in node_dict.items() if name != "id"} == {**template.nodes[node_dict["id"]], "unit_id": unit_id}
    assert "[*:2]" in ensemble["units"]["R0"]["psmiles"], "repeat unit carries two numbered stars"

    for record in ensemble["bonds"]:
        assert list(record) == ["labels", "nodes", "count"] and record["count"] > 0
        for endpoint, node_id in zip(record["labels"], record["nodes"], strict=True):
            unit_id, bond_id = endpoint.rsplit(".", 1)
            assert unit_id in ensemble["units"] and int(bond_id) >= 1
            # nodes are aligned with labels: both name the same connection atom.
            assert labels.unit_id[node_id] == unit_id
            assert labels.bond_id[node_id] == int(bond_id)
    linkages = {tuple(record["labels"]) for record in ensemble["bonds"]}
    assert ("I0.1", "R0.1") in linkages and ("R0.2", "T0.1") in linkages


def test_create_ensemble_json_file_mol_graph_chains(tmp_path):
    """With output_format="mol_graph" the file's chains follow the caller's
    format as node-link graph dicts (picking the format is picking the file
    size); sequences are written as SMILES regardless."""
    import json

    import networkx as nx

    smi = "{[] [<]CC([>])c1ccccc1; CO[>]; [<][H] []}|gauss(1000, 45)|"
    path = tmp_path / "ensemble_graphs.json"
    _reset_rngs(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
        chains = ensemble_creator.create_ensemble(2, output_format="mol_graph", json_file=str(path))

    data = json.loads(path.read_text())
    assert data["format"]["chain_format"] == "mol_graph"
    stored_chains = data["ensemble"]["chains"]
    assert len(stored_chains) == 2
    node_keys = ["id", "atomic_num", "is_connector_placeholder", "aromatic", "charge", "num_explicit_h", "origin_idx"]
    edge_keys = ["source", "target", "bond_type", "aromatic"]
    for chain_data, chain_graph in zip(stored_chains, chains, strict=True):
        # The file carries the atom and bond attributes plus the template
        # provenance, in a fixed key order, and no sampler bookkeeping.
        assert all(list(node) == node_keys for node in chain_data["nodes"])
        assert all(list(edge) == edge_keys for edge in chain_data["edges"])
        restored = nx.node_link_graph(chain_data, edges="edges")
        assert set(restored.nodes) == set(chain_graph.nodes)
        assert {frozenset(edge) for edge in restored.edges} == {frozenset(edge) for edge in chain_graph.edges}
        for node, attributes in restored.nodes(data=True):
            assert attributes == {key: chain_graph.nodes[node][key] for key in node_keys[1:]}
        for u, v, attributes in restored.edges(data=True):
            assert attributes == {key: chain_graph.edges[u, v][key] for key in edge_keys[2:]}
    for chain_sequences in data["ensemble"]["sequences"]:
        for sequence in chain_sequences:
            assert sequence and all(isinstance(unit, str) for unit in sequence)


def test_generative_graph_and_ensemble_share_node_ids():
    """Node ids are UUIDs minted per GraphCreator, so a generative graph and an
    ensemble only correspond when both come from ONE held creator (or when the
    ensemble creator is built directly from the graph in hand)."""
    g2rins_object = g2rins.G2rins.make("{[] [<]CC([>])c1ccccc1; [>][H]; [<][H] []}|gauss(1000, 45)|")

    graph_creator = g2rins_object.get_graph_creator()
    generative_graph = graph_creator.get_generative_graph()
    ensemble_creator = graph_creator.get_ensemble_creator()

    molecule = ensemble_creator.sample_mol_graph(rng=np.random.default_rng(0))
    origins = {str(data["origin_idx"]) for _node, data in molecule.nodes(data=True) if "origin_idx" in data}

    assert origins, "sampled atoms must carry origin_idx back to the generative graph"
    assert origins <= {str(node) for node in generative_graph.nodes}
    assert origins <= {str(node) for node in ensemble_creator.generative_graph.nodes}

    # By-construction path: an EnsembleCreator built from the graph in hand shares its ids.
    direct = g2rins.EnsembleCreator(generative_graph)
    direct_molecule = direct.sample_mol_graph(rng=np.random.default_rng(0))
    direct_origins = {str(data["origin_idx"]) for _node, data in direct_molecule.nodes(data=True) if "origin_idx" in data}
    assert direct_origins <= {str(node) for node in generative_graph.nodes}


def test_ensemble_json_normalizes_numpy_values_in_the_ensemble_section(tmp_path):
    """The ensemble section of the JSON file (unit subgraphs, node-link chains,
    weights) is normalized like the graph section: NumPy values become JSON
    values, and the graph in memory keeps its NumPy values."""
    import json

    smi = "{[] [<]CC([>])c1ccccc1; CO[>]; [<][H] []}|gauss(1000, 45)|"
    generative_graph = g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph()
    node = next(node for node, data in generative_graph.nodes(data=True) if data["atomic_num"] == 6)
    generative_graph.nodes[node]["metadata"] = {"score": np.float32(0.25), "counts": np.array([1, 2], dtype=np.int16), "flag": np.bool_(True)}
    path = tmp_path / "ensemble.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g2rins.EnsembleCreator(generative_graph).create_ensemble(2, output_format="mol_graph", json_file=str(path), seed=0)

    data = json.loads(path.read_text())
    stored = [node_data for unit in data["ensemble"]["units"].values() for node_data in unit["subgraph"]["nodes"] if "metadata" in node_data]
    assert len(stored) == 1
    assert stored[0]["metadata"] == {"score": 0.25, "counts": [1, 2], "flag": True}
    assert [type(value) for value in stored[0]["metadata"].values()] == [float, list, bool]
    assert isinstance(generative_graph.nodes[node]["metadata"]["score"], np.float32)
    assert isinstance(generative_graph.nodes[node]["metadata"]["counts"], np.ndarray)


def test_ensemble_json_refuses_non_finite_values_in_the_ensemble_section(tmp_path, monkeypatch):
    """A non-finite value that reaches the file only through the ensemble
    section (here a unit subgraph) is refused before the file is opened. The
    graph section is replaced by a clean export so it cannot reject first."""
    import copy

    import g2rins.ensemble_creator as ensemble_creator_module

    smi = "{[] [<]CC([>])c1ccccc1; CO[>]; [<][H] []}|gauss(1000, 45)|"
    generative_graph = g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph()
    clean_export = g2rins.generative_graph_json_data(generative_graph)
    generative_graph.nodes[next(iter(generative_graph))]["metadata"] = {"score": float("nan")}
    monkeypatch.setattr(ensemble_creator_module, "generative_graph_json_data", lambda graph: copy.deepcopy(clean_export))
    path = tmp_path / "ensemble.json"
    with pytest.raises(ValueError, match=r"Non-finite JSON value at \$\['ensemble'\]"):
        g2rins.EnsembleCreator(generative_graph).create_ensemble(1, output_format="smiles", json_file=str(path), seed=0)
    assert not path.exists()


def test_ensemble_json_normalizes_the_source_string_before_writing(tmp_path):
    """The top-level source string is normalized like every other part of the
    file: NumPy-backed provenance is written as a plain string, and provenance
    that cannot be serialized is refused before the file is opened, so an
    existing output file is left untouched."""
    import json

    smi = "{[] [<]CC([>])c1ccccc1; CO[>]; [<][H] []}|gauss(1000, 45)|"
    generative_graph = g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph()
    text = generative_graph.graph["g2rins_string"]
    generative_graph.graph["g2rins_string"] = np.array(text)
    path = tmp_path / "ensemble.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g2rins.EnsembleCreator(generative_graph).create_ensemble(1, output_format="smiles", json_file=str(path), seed=0)
    stored = json.loads(path.read_text())["string"]
    assert stored == text and type(stored) is str

    previous = path.read_bytes()
    generative_graph.graph["g2rins_string"] = 1 + 2j
    with pytest.raises(TypeError, match=r"Unsupported JSON value at \$\['string'\]"):
        g2rins.EnsembleCreator(generative_graph).create_ensemble(1, output_format="smiles", json_file=str(path), seed=0)
    assert path.read_bytes() == previous


def test_exported_chain_provenance_uses_template_node_keys(tmp_path):
    """Chain atoms in the file name their template node by the template's own
    key, whatever its type, so a consumer can look the node up in the graph
    section, the unit subgraphs and the bond records, which all use that key.
    The in-memory chain graphs keep the sampler's string form."""
    import json

    import networkx as nx

    text = "{[] [<]CCO[>]; CO[>]; [<][H] []}|poisson(200)|"
    generative_graph = g2rins.G2rins.make(text).get_graph_creator().get_generative_graph()
    labels = g2rins.derive_unit_labels(generative_graph)
    initiator_site = next(node for node, unit_id in labels.unit_id.items() if unit_id == "I0" and node in labels.bond_id)
    generative_graph = nx.relabel_nodes(generative_graph, {initiator_site: 42})
    generative_graph.graph.pop("unit_node_ids", None)
    generative_graph.graph.pop("unit_g2rins", None)
    path = tmp_path / "ensemble.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chains = g2rins.EnsembleCreator(generative_graph).create_ensemble(2, output_format="mol_graph", ensemble_info=True, json_file=str(path), seed=0).chains
    data = json.loads(path.read_text())
    origins = [node["origin_idx"] for chain in data["ensemble"]["chains"] for node in chain["nodes"]]
    assert all(origin in generative_graph for origin in origins)
    assert 42 in origins
    assert {node["id"] for node in data["graph"]["nodes"]} >= set(origins)
    assert all(isinstance(data["origin_idx"], str) for chain in chains for _node, data in chain.nodes(data=True))


def test_ensemble_from_reloaded_graph_carries_no_stale_derived_fields():
    """A template loaded back from an export carries the injected unit_id and
    bond_id as node attributes. The creator drops them, so unit subgraphs
    carry the current unit_id only and bond records follow the current
    derivation, even when the reloaded node order renumbers the labels; a
    reload in the exported order reproduces the original ensemble."""
    import json

    import networkx as nx

    text = "{[] [<]CCN([>])[>]; [<][H]; O[>], [<][H] []}|poisson(200)|"
    generative_graph = g2rins.G2rins.make(text).get_graph_creator().get_generative_graph()
    payload = json.loads(json.dumps(g2rins.generative_graph_json_data(generative_graph)))
    restored = nx.node_link_graph(payload["graph"], edges="edges")
    payload["graph"]["nodes"].reverse()
    reordered = nx.node_link_graph(payload["graph"], edges="edges")
    assert all("unit_id" in data for _node, data in restored.nodes(data=True))
    reordered_labels = g2rins.derive_unit_labels(reordered)
    # The reversed order renumbers connection sites, so stale bond ids conflict.
    assert any(data["bond_id"] != reordered_labels.bond_id.get(node) for node, data in reordered.nodes(data=True) if "bond_id" in data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = g2rins.EnsembleCreator(generative_graph).create_ensemble(2, output_format="smiles", ensemble_info=True, seed=0)
        actual = g2rins.EnsembleCreator(restored).create_ensemble(2, output_format="smiles", ensemble_info=True, seed=0)
        actual_reordered = g2rins.EnsembleCreator(reordered).create_ensemble(2, output_format="smiles", ensemble_info=True, seed=0)
    for result, graph in ((actual, restored), (actual_reordered, reordered)):
        labels = g2rins.derive_unit_labels(graph)
        for unit_id, info in result.units.items():
            for node, data in info["subgraph"].nodes(data=True):
                assert "bond_id" not in data
                assert data["unit_id"] == unit_id == labels.unit_id[node]
        for record in result.bonds:
            for label, node in zip(record["labels"], record["nodes"], strict=True):
                assert label == f"{labels.unit_id[node]}.{labels.bond_id[node]}"
        # The caller's graph keeps its attributes; only the creator's copy drops them.
        assert all("unit_id" in data for _node, data in graph.nodes(data=True))
    assert actual == expected


def test_unit_subgraphs_do_not_copy_graph_level_metadata():
    """Unit subgraphs copy node and edge data only: graph-level metadata of
    the template is neither copied nor required to be copyable."""
    import threading

    text = "{[] [<]CCO[>]; CO[>]; [<][H] []}|poisson(200)|"
    generative_graph = g2rins.G2rins.make(text).get_graph_creator().get_generative_graph()
    generative_graph.graph["lock"] = threading.Lock()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = g2rins.EnsembleCreator(generative_graph).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    assert result.units
    assert all(info["subgraph"].graph == {} for info in result.units.values())


def test_ensemble_export_is_independent_of_the_hash_seed(tmp_path):
    """Unit subgraph node order, chain attribute order and the JSON bytes do
    not depend on PYTHONHASHSEED: the same serialized template and sampling
    seed give identical files in fresh interpreters with different hash
    seeds. The nested template has units small enough that a subgraph view
    of the template would have iterated their membership set, which is the
    regression this guards."""
    import json
    import os
    import subprocess
    import sys

    text = "{[] [<]CC([>])C(=O)O{[>] [<]CCO[>]; ; [<]C []}|poisson(300)|; COC(=O)C(C)[>]; [<]Br []}|poisson(800)|"
    generative_graph = g2rins.G2rins.make(text).get_graph_creator().get_generative_graph()
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(g2rins.generative_graph_json_data(generative_graph)))
    script = tmp_path / "export.py"
    script.write_text(
        "import json, sys, warnings\n"
        "import networkx as nx\n"
        "import g2rins\n"
        "warnings.simplefilter('ignore')\n"
        "graph = nx.node_link_graph(json.load(open(sys.argv[1]))['graph'], edges='edges')\n"
        "g2rins.EnsembleCreator(graph).create_ensemble(2, output_format='mol_graph', json_file=sys.argv[2], seed=0)\n"
    )
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(g2rins.__file__)))
    python_path = os.pathsep.join(entry for entry in (package_root, os.environ.get("PYTHONPATH")) if entry)
    outputs = []
    for hash_seed in ("0", "1", "2"):
        out = tmp_path / f"ensemble_{hash_seed}.json"
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": python_path}
        completed = subprocess.run([sys.executable, str(script), str(template_path), str(out)], env=env, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        outputs.append(out.read_bytes())
    assert len(set(outputs)) == 1
    units = json.loads(outputs[0])["ensemble"]["units"]
    assert {"I0", "R0", "R1"} <= set(units) and all(unit["subgraph"]["nodes"] for unit in units.values())


def test_bond_record_nodes_are_template_node_keys():
    """``nodes`` holds the template's own node keys, whatever their type, and
    each one corresponds to the label at the same position."""
    import networkx as nx

    text = "{[] [<]CCO[>]; CO[>]; [<][H] []}|poisson(200)|"
    generative_graph = g2rins.G2rins.make(text).get_graph_creator().get_generative_graph()
    labels = g2rins.derive_unit_labels(generative_graph)
    initiator_site = next(node for node, unit_id in labels.unit_id.items() if unit_id == "I0" and node in labels.bond_id)
    generative_graph = nx.relabel_nodes(generative_graph, {initiator_site: 42})
    generative_graph.graph.pop("unit_node_ids", None)
    generative_graph.graph.pop("unit_g2rins", None)
    labels = g2rins.derive_unit_labels(generative_graph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = g2rins.EnsembleCreator(generative_graph).create_ensemble(1, output_format="smiles", ensemble_info=True, seed=0)
    assert result.bonds
    for record in result.bonds:
        for label, node in zip(record["labels"], record["nodes"], strict=True):
            assert node in generative_graph
            assert label == f"{labels.unit_id[node]}.{labels.bond_id[node]}"
    assert any(42 in record["nodes"] for record in result.bonds)
    assert 42 in result.units["I0"]["subgraph"]