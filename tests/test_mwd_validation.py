"""Independent numerical, chemistry and migration checks for the NB rollback."""

import ast
import copy
import inspect
import math
import pickle
import textwrap
import warnings

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

import g2rins
import g2rins.generative_graph as graph_module
from g2rins._metadata_equality import _metadata_equal
from g2rins.exception import EmptyTruncatedDistributionSupport
from g2rins.generative_graph import GraphCreator, generative_graph_json_data


LINEAR = "{[] [<]CC[>];; [<][H], [>][H] []}|mwd(1500,1000)|"
CHEMISTRIES = [
    LINEAR,
    "{[] [<]CC[>], [<]CO[>];; [<][H], [>][H] []}|mwd(740,500)|",
    "{[] [$]CC([$])C[$];; [$][H] []}|mwd(740,500)|",
    "C{[>][<]CC[>];;[<]}|mwd(740,500)|[H]",
    "{[] [<]CC({[<] [<]NN[>];; [>]}|poisson(80)|[H])CC[>];; [<][H], [>][H] []}|mwd(1500,1000)|",
]


def graph_factory(text=LINEAR):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return g2rins.G2rins.make(text).get_graph_creator()


def creator(text=LINEAR):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return graph_factory(text).get_ensemble_creator()


@pytest.mark.parametrize("lower,upper", [(400000, 401000), (0, 1e-200)])
def test_positive_gamma_interval_underflow_is_not_empty_support(lower, upper):
    law = g2rins.Mwd.make("mwd(1500,1000)")
    # Gamma has strictly positive density on every positive finite interval.
    # Numerical failure must not become a retryable chain-local support miss.
    with pytest.raises(RuntimeError, match="numerical|representable|underflow"):
        law.draw_mw(np.random.default_rng(42), lower=lower, upper=upper)


def test_gamma_numerical_failure_aborts_without_retry(monkeypatch):
    instance = creator()
    law = g2rins.Mwd.make("mwd(1500,1000)")
    calls = []

    def impossible_numeric_draw(**kwargs):
        calls.append(None)
        return law.draw_mw(kwargs["rng"], lower=400000, upper=401000)

    monkeypatch.setattr(instance, "sample_mol_graph", impossible_numeric_draw)
    with pytest.raises(RuntimeError, match="underflow"):
        instance.create_ensemble(1, seed=7, max_number_of_discarded_chains=5)
    assert len(calls) == 1


class FixedUniform:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.value


@pytest.mark.parametrize("lower,upper", [(0, 100), (100, 2000), (20000, 21000), (300000, 301000)])
@pytest.mark.parametrize("u", [0.001, 0.1, 0.5, 0.9, 0.999])
def test_bounded_quantile_against_independent_erlang_survival(lower, upper, u):
    # For shape=2, SF(x)=exp(-x/scale)*(1+x/scale), without SciPy CDF/PPF.
    def survival(mass):
        z = mass / 500
        return math.exp(-z) * (1 + z)

    law = g2rins.Mwd.make("mwd(1500,1000)")
    rng = FixedUniform(u)
    value = law.draw_mw(rng, lower=lower, upper=upper)
    assert lower <= value <= upper
    conditional_sf = (survival(value) - survival(upper)) / (survival(lower) - survival(upper))
    # The base sampler uses CDF inversion below the median, SF above it.
    expected = u if survival(lower) <= 0.5 else 1 - u
    assert conditional_sf == pytest.approx(expected, abs=1e-10)
    assert rng.calls == 1


@pytest.mark.parametrize("Mn", [1e-150, 1000, 1e150])
def test_representable_extreme_moments_roundtrip(Mn):
    law = g2rins.Mwd.from_moments(Mn=Mn, dispersity=1.1)
    restored = g2rins.StochasticDistribution.from_serial_vector(law.get_serial_vector())
    assert restored.serialize() == law.serialize()
    values = [law.draw_mw(np.random.default_rng(seed)) for seed in range(10)]
    assert all(math.isfinite(value) and value > 0 for value in values)


def test_adjacent_float_moments_do_not_become_a_point_mass():
    Mn = 1000.0
    Mw = math.nextafter(Mn, math.inf)
    law = g2rins.Mwd.from_moments(Mn=Mn, Mw=Mw)
    assert law.moments["variance"] == Mn * (Mw - Mn) > 0
    assert law.support_mw() == (0, math.inf)


@pytest.mark.parametrize("text", CHEMISTRIES)
def test_actual_molecules_and_reported_masses_are_consistent(text):
    result = creator(text).create_ensemble(5, output_format="mol_graph", ensemble_info=True, seed=19)
    masses = []
    for graph in result.chains:
        assert nx.is_connected(graph.to_undirected())
        assert not any(key.startswith("nb_") for key in graph.graph)
        mol = g2rins.mol_graph_to_rdkit_mol(graph)
        Chem.SanitizeMol(mol)
        reparsed = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        assert reparsed is not None and len(Chem.GetMolFrags(reparsed)) == 1
        assert Descriptors.MolWt(reparsed) == pytest.approx(Descriptors.MolWt(mol))
        masses.append(Descriptors.MolWt(mol))
    assert result.molecular_weights == pytest.approx(masses)


@pytest.mark.parametrize("text", CHEMISTRIES)
def test_connector_pruning_matches_all_simple_paths_oracle(text):
    # Reuse the unchanged contraction/attribute rules, but replace ONLY the new
    # pruned path iterator with independent NetworkX exhaustive enumeration.
    tree = ast.parse(textwrap.dedent(inspect.getsource(GraphCreator.get_graph_without_bond_connectors)))
    replacements = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "connector_paths":
                node.iter = ast.parse("nx.all_simple_edge_paths(graph, in_idx, target)", mode="eval").body
                replacements += 1
    assert replacements == 1
    namespace = vars(graph_module).copy()
    exec(compile(ast.fix_missing_locations(tree), "<exhaustive-connector-oracle>", "exec"), namespace)
    factory = graph_factory(text)
    current = factory.get_graph_without_bond_connectors()
    exhaustive = namespace["get_graph_without_bond_connectors"](factory)
    assert list(current.nodes) == list(exhaustive.nodes)
    assert list(current.edges(keys=True)) == list(exhaustive.edges(keys=True))
    assert _metadata_equal(current, exhaustive)


@pytest.mark.parametrize("damage", ["boolean", "nonfinite", "partial", "multiple", "unknown_schema", "no_mwd",
                                    "old_schema", "old_context", "old_width", "wrong_count", "mixed_width"])
def test_malformed_mwd_rejected_on_every_public_restore_boundary(damage):
    instance = creator()
    state = copy.deepcopy(instance.__getstate__())
    graph = state["_generative_graph"]
    vectors = next(iter(graph.nodes(data=True)))[1]["molecular_weight_distribution"]
    if damage == "boolean":
        vectors[0][14] = True
    elif damage == "nonfinite":
        vectors[0][14] = math.inf
    elif damage == "partial":
        vectors[0][15] = -1
    elif damage == "multiple":
        vectors[0][4:6] = [1000, 100]
    elif damage == "unknown_schema":
        graph.graph["molecular_weight_distribution_schema"] = "g2rins-distribution-vector/v999"
    elif damage == "old_schema":
        graph.graph["molecular_weight_distribution_schema"] = "g2rins-distribution-vector/v3"
    elif damage == "old_context":
        graph.graph["nb_recipe_context"] = {}
    elif damage == "old_width":
        for vector in vectors:
            del vector[14:]
    elif damage == "wrong_count":
        vectors.pop()
    elif damage == "mixed_width":
        del vectors[-1][12:]
    else:
        vectors[0][14:16] = [-1, -1]
    with pytest.raises(ValueError):
        g2rins.EnsembleCreator(graph)
    with pytest.raises(ValueError):
        generative_graph_json_data(graph)
    # Real unpickle invokes __setstate__, not just the public graph constructor.
    instance.__dict__.update(state)
    with pytest.raises(ValueError):
        pickle.loads(pickle.dumps(instance))


@pytest.mark.parametrize("damage", ["nb_identity", "nb_origins", "changed_moments", "changed_semantics"])
def test_checkpoint_rejection_precedes_rng_and_callbacks(damage, monkeypatch):
    instance = creator()
    checkpoints = []
    options = dict(batch_size=1, window=10, seed=42, retain_chains=False, retain_sequences=False)
    instance.create_ensemble_until_converged(max_samples=1, checkpoint_callback=checkpoints.append, **options)
    checkpoint = copy.deepcopy(checkpoints[-1])
    if damage == "nb_identity":
        checkpoint.settings["nb_model_identity"] = "retired"
    elif damage == "nb_origins":
        checkpoint.nb_origin_order = ("retired",)
    elif damage == "changed_semantics":
        checkpoint.settings["mwd_semantics"] = ("unknown-law/v1", ())
    else:
        instance = creator(LINEAR.replace("1500,1000", "1600,1000"))

    def unexpected(*args, **kwargs):
        pytest.fail("Rejected checkpoints must not construct RNGs, sample or call back")

    monkeypatch.setattr(np.random, "default_rng", unexpected)
    monkeypatch.setattr(instance, "_iter_chain_records", unexpected)
    with pytest.raises(ValueError, match="Retired NB|settings"):
        instance.create_ensemble_until_converged(max_samples=2, checkpoint=checkpoint,
                                               checkpoint_callback=unexpected, **options)


@pytest.mark.parametrize("policy", ["full", "statistics"])
def test_mwd_parallel_checkpoint_resume_matches_uninterrupted(policy):
    instance = creator()
    options = dict(batch_size=2, window=10, seed=42, parallel=True, n_workers=2,
                   checkpoint_policy=policy, retain_chains=False, retain_sequences=False)
    checkpoints = []
    instance.create_ensemble_until_converged(max_samples=2, checkpoint_callback=checkpoints.append, **options)
    checkpoint = pickle.loads(pickle.dumps(checkpoints[-1]))
    resumed = instance.create_ensemble_until_converged(max_samples=4, checkpoint=checkpoint, **options)
    whole = instance.create_ensemble_until_converged(max_samples=4, **options)
    assert _metadata_equal(vars(resumed), vars(whole))