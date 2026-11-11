# SPDX-License-Identifier: GPL-3.0-only
"""Checkpoint model identity must be validated before any resume side effects."""

import copy
import json
import os
import pickle
import subprocess
import sys
import warnings
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

import g2rins
from g2rins._model_fingerprint import MODEL_FINGERPRINT_SCHEMA, model_fingerprint
from g2rins.generative_graph import generative_graph_json_data


def _creator(repeat="CC", distribution="mwd(1500,1000)"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return g2rins.G2rins.make(
            "{[] [<]" + repeat + "[>]; [H][>]; [<][H] []}|" + distribution + "|"
        ).get_graph_creator().get_ensemble_creator()


def _options(policy="full"):
    return dict(batch_size=1, window=10, seed=42, output_format="smiles",
                retain_chains=False, retain_sequences=False, checkpoint_policy=policy)


def _checkpoint(creator, policy="full"):
    saved = []
    creator.create_ensemble_until_converged(
        **_options(policy), max_samples=1, checkpoint_callback=saved.append,
    )
    return pickle.loads(pickle.dumps(saved[-1]))


@pytest.mark.parametrize("policy", ["full", "statistics"])
def test_different_chemistry_with_same_mwd_is_rejected(policy):
    checkpoint = _checkpoint(_creator("CC"), policy)
    with pytest.raises(ValueError, match="model fingerprint"):
        _creator("CO").create_ensemble_until_converged(
            **_options(policy), max_samples=2, checkpoint=checkpoint,
        )


def test_checkpoint_without_model_fingerprint_is_rejected():
    creator = _creator()
    checkpoint = _checkpoint(creator)
    settings = dict(checkpoint.settings)
    settings.pop("model_fingerprint", None)
    with pytest.raises(ValueError, match="model fingerprint"):
        creator.create_ensemble_until_converged(
            **_options(), max_samples=2,
            checkpoint=replace(checkpoint, settings=settings),
        )


@pytest.mark.parametrize("policy", ["full", "statistics"])
def test_fingerprint_is_versioned_and_same_across_checkpoint_policies(policy):
    creator = _creator()
    fingerprint = _checkpoint(creator, policy).settings["model_fingerprint"]
    assert fingerprint == model_fingerprint(creator._generative_graph)
    assert fingerprint["schema"] == MODEL_FINGERPRINT_SCHEMA
    assert len(fingerprint["sha256"]) == 64


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("saved", [
    None, {}, "old-digest", ("v1", "digest"),
    {"schema": "g2rins-model-fingerprint/v999", "sha256": "0" * 64},
    {"schema": MODEL_FINGERPRINT_SCHEMA, "sha256": "z" * 64},
    {"schema": MODEL_FINGERPRINT_SCHEMA, "sha256": "A" * 64},
    {"schema": MODEL_FINGERPRINT_SCHEMA, "sha256": 123},
    {"schema": MODEL_FINGERPRINT_SCHEMA, "sha256": "0" * 63},
    {"schema": MODEL_FINGERPRINT_SCHEMA, "sha256": "0" * 64, "extra": True},
    {"schema": MODEL_FINGERPRINT_SCHEMA, "sha256": "0" * 64},
])
def test_bad_fingerprint_rejected_before_rng_sampling_workers_or_callbacks(saved, parallel, monkeypatch):
    creator = _creator()
    checkpoint = _checkpoint(creator)
    if isinstance(saved, dict) and "schema" in saved:
        saved = {"origin_sha256": checkpoint.settings["model_fingerprint"]["origin_sha256"], **saved}
    checkpoint.settings["model_fingerprint"] = saved

    def unexpected(*args, **kwargs):
        pytest.fail("Fingerprint rejection must precede resume side effects")

    monkeypatch.setattr(np.random, "default_rng", unexpected)
    monkeypatch.setattr(np.random, "SeedSequence", unexpected)
    monkeypatch.setattr(creator, "_iter_chain_records", unexpected)
    monkeypatch.setattr(creator, "sample_mol_graph", unexpected)
    # This is the parallel executor entry point, regardless of resume settings.
    monkeypatch.setattr("g2rins.ensemble_creator._ParallelChainScheduler.__enter__", unexpected)
    with pytest.raises(ValueError, match="model fingerprint"):
        creator.create_ensemble_until_converged(
            **dict(_options(), batch_size=2, parallel=parallel,
                   n_workers=2 if parallel else None), max_samples=2, checkpoint=checkpoint,
            checkpoint_callback=unexpected, sample_callback=unexpected,
        )


@pytest.mark.parametrize("distribution", ["mwd(1500,1000)", "gauss(1000,45)"])
@pytest.mark.parametrize("policy", ["full", "statistics"])
@pytest.mark.parametrize("parallel", [False, True])
def test_restored_identical_creator_resumes_exactly(distribution, policy, parallel):
    first = _creator(distribution=distribution)
    fresh = pickle.loads(pickle.dumps(first))
    options = dict(_options(policy), parallel=parallel,
                   n_workers=2 if parallel else None, retain_chains=policy == "full")
    checkpoints = []
    first.create_ensemble_until_converged(
        **options, max_samples=1, checkpoint_callback=checkpoints.append,
    )
    resumed = fresh.create_ensemble_until_converged(
        **options, max_samples=3,
        checkpoint=pickle.loads(pickle.dumps(checkpoints[-1])),
    )
    whole = fresh.create_ensemble_until_converged(**options, max_samples=3)
    assert resumed.chains == whole.chains
    assert resumed.molecular_weights == whole.molecular_weights
    assert resumed.convergence_trace == whole.convergence_trace
    assert resumed.number_average_molecular_weight == whole.number_average_molecular_weight
    assert resumed.weight_average_molecular_weight == whole.weight_average_molecular_weight
    assert resumed.bonds == whole.bonds


@pytest.mark.parametrize("policy", ["full", "statistics"])
def test_fresh_uuid_namespace_is_rejected_without_metadata_rebase(policy):
    original = _creator()
    fresh = _creator()
    before = model_fingerprint(original._generative_graph)
    after = model_fingerprint(fresh._generative_graph)
    assert before["sha256"] == after["sha256"]
    assert before["origin_sha256"] != after["origin_sha256"]
    checkpoint = _checkpoint(original, policy)
    with pytest.raises(ValueError, match="origin namespace"):
        fresh.create_ensemble_until_converged(
            **_options(policy), max_samples=2, checkpoint=checkpoint,
        )


@pytest.mark.parametrize("transport", ["pickle", "json", "numpy", "mapping_order", "membership_order"])
def test_equivalent_representations_have_same_fingerprint(transport):
    graph = _creator()._generative_graph
    expected = model_fingerprint(graph)
    restored = copy.deepcopy(graph)
    if transport == "pickle":
        restored = pickle.loads(pickle.dumps(restored))
    elif transport == "json":
        payload = json.loads(json.dumps(generative_graph_json_data(restored)))
        restored = g2rins.EnsembleCreator(
            nx.node_link_graph(payload["graph"], edges="edges")
        )._generative_graph
    elif transport == "numpy":
        for _, data in restored.nodes(data=True):
            data["atomic_num"] = np.int64(data["atomic_num"])
            data["molecular_weight_distribution"] = np.array(data["molecular_weight_distribution"])
    elif transport == "mapping_order":
        for _, data in restored.nodes(data=True):
            items = list(data.items())[::-1]
            data.clear()
            data.update(items)
    else:
        for nodes in restored.graph["unit_node_ids"].values():
            nodes.reverse()
    assert model_fingerprint(restored) == expected


def test_fingerprint_is_stable_across_process_hash_seeds():
    expected = model_fingerprint(_creator()._generative_graph)["sha256"]
    completed = subprocess.run(
        [sys.executable, "-c", "import json; "
         "from tests.test_checkpoint_model_fingerprint import _creator; "
         "from g2rins._model_fingerprint import model_fingerprint; "
         "print(json.dumps(model_fingerprint(_creator()._generative_graph)['sha256']))"],
        env={**os.environ, "PYTHONHASHSEED": "127"},
        check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout) == expected


@pytest.mark.parametrize("attribute", [
    "atomic_num", "charge", "num_explicit_h", "aromatic", "gen_weight",
    "gen_hierarchy", "init_weight", "mol_molecular_weight", "total_molecular_weight",
    "unit_molar_amounts", "stochastic_id_tree", "molecular_weight_distribution",
    "atom_chiral_token",
])
def test_node_semantics_are_bound_even_with_unchanged_source(attribute):
    graph = _creator()._generative_graph
    changed = copy.deepcopy(graph)
    data = changed.nodes[next(iter(changed))]
    if attribute == "atom_chiral_token":
        data[attribute] = "@@"
    elif attribute == "molecular_weight_distribution":
        data[attribute][0][14] += 1
    elif isinstance(data[attribute], list):
        data[attribute][0] += 1
    elif isinstance(data[attribute], bool):
        data[attribute] = not data[attribute]
    else:
        data[attribute] += 1
    assert changed.graph["g2rins_string"] == graph.graph["g2rins_string"]
    assert model_fingerprint(changed) != model_fingerprint(graph)


@pytest.mark.parametrize("attribute", [
    "bond_type", "aromatic", "static", "propagation_weight", "termination_weight",
    "transition_weight", "stochastic_id", "bond_symbol_raw", "double_bond_stereo_defined",
])
def test_edge_semantics_are_bound(attribute):
    graph = _creator()._generative_graph
    changed = copy.deepcopy(graph)
    data = next(iter(changed.edges(data=True)))[2]
    old = data.get(attribute)
    data[attribute] = not old if isinstance(old, bool) else (old + 1 if old is not None else "/")
    assert model_fingerprint(changed) != model_fingerprint(graph)


@pytest.mark.parametrize("order", ["nodes", "outgoing", "incoming", "parallel_edges"])
def test_operational_order_is_part_of_model_identity(order):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(["a", "b", "c"])
    changed = nx.MultiDiGraph()
    changed.add_nodes_from(["b", "a", "c"] if order == "nodes" else graph)
    edges = [("a", "b", 0), ("a", "c", 0)]
    if order == "incoming":
        edges = [("a", "c", 0), ("b", "c", 0)]
    elif order == "parallel_edges":
        edges = [("a", "b", 0), ("a", "b", 1)]
    graph.add_edges_from(edges)
    changed.add_edges_from(edges if order == "nodes" else reversed(edges))
    assert model_fingerprint(changed) != model_fingerprint(graph)


@pytest.mark.parametrize("value", [object(), float("nan"), float("inf"), complex(1, 2)])
def test_unrepresentable_attributes_fail_closed(value):
    graph = _creator()._generative_graph
    graph.nodes[next(iter(graph))]["external_attribute"] = value
    with pytest.raises(ValueError, match="model fingerprint"):
        model_fingerprint(graph)


def test_plain_sampling_does_not_compute_fingerprint(monkeypatch):
    creator = _creator()

    def unexpected(*args, **kwargs):
        pytest.fail("Non-checkpoint sampling must not hash the model")

    monkeypatch.setattr("g2rins.ensemble_creator.model_fingerprint", unexpected)
    creator.create_ensemble(1, seed=7)
    creator.create_ensemble_until_converged(**_options(), max_samples=1)


def test_fingerprint_is_recomputed_after_a_graph_change():
    creator = _creator()
    checkpoint = _checkpoint(creator)
    creator._generative_graph.nodes[next(iter(creator._generative_graph))]["charge"] = 1
    with pytest.raises(ValueError, match="model fingerprint"):
        creator.create_ensemble_until_converged(
            **_options(), max_samples=2, checkpoint=checkpoint,
        )


@pytest.mark.parametrize("value", [None, "invalid", "0" * 64])
def test_invalid_or_changed_origin_digest_is_rejected(value):
    creator = _creator()
    checkpoint = _checkpoint(creator)
    checkpoint.settings["model_fingerprint"]["origin_sha256"] = value
    with pytest.raises(ValueError, match="model fingerprint"):
        creator.create_ensemble_until_converged(
            **_options(), max_samples=2, checkpoint=checkpoint,
        )


@pytest.mark.parametrize("text", [
    "{[] [<]CC[>]; [H][>]; [<]O []}|mwd(1500,1000)|",
    "{[] [<]CC[>], [<]CO[>]; [H][>]; [<][H] []}|mwd(1500,1000)|",
    "{[] [<]CC[>]; [H][>]; [<][H] []}|mwd(1600,1000)|",
    "{[] [<]CC[>]; [H][>]; [<][H] []}|gauss(1000,45)|",
])
def test_changed_endgroups_composition_or_law_cannot_resume(text):
    checkpoint = _checkpoint(_creator())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        changed = g2rins.G2rins.make(text).get_graph_creator().get_ensemble_creator()
    with pytest.raises(ValueError, match="model fingerprint"):
        changed.create_ensemble_until_converged(
            **_options(), max_samples=2, checkpoint=checkpoint,
        )


def test_nested_local_distribution_is_bound():
    text = "{[] [<]CC({[<] [<]NN[>];; [>]}|poisson(80)|[H])CC[>];; [<][H], [>][H] []}|mwd(1500,1000)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original = g2rins.G2rins.make(text).get_graph_creator().get_ensemble_creator()
        changed = g2rins.G2rins.make(text.replace("poisson(80)", "poisson(90)")).get_graph_creator().get_ensemble_creator()
    checkpoint = _checkpoint(original)
    with pytest.raises(ValueError, match="model fingerprint"):
        changed.create_ensemble_until_converged(
            **_options(), max_samples=2, checkpoint=checkpoint,
        )


def test_legacy_distribution_chemistry_change_is_also_rejected():
    checkpoint = _checkpoint(_creator("CC", "gauss(1000,45)"))
    with pytest.raises(ValueError, match="model fingerprint"):
        _creator("CO", "gauss(1000,45)").create_ensemble_until_converged(
            **_options(), max_samples=2, checkpoint=checkpoint,
        )