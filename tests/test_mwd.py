"""Direct number-target-mass law, sampling integration and NB migration guards."""

import copy
import json
import pickle
import warnings
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from scipy import stats

import g2rins
from g2rins._metadata_equality import _metadata_equal
from g2rins._mwd_transport import MWD_SCHEMA, validate_mwd_transport
from g2rins.exception import EmptyTruncatedDistributionSupport
from g2rins.generative_graph import generative_graph_json_data
from g2rins.util import RememberAdd


LINEAR = "{[] [<]CC[>];; [<][H], [>][H] []}|mwd(740,500)|"


def factory(text=LINEAR):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return g2rins.G2rins.make(text).get_graph_creator()


def interval(lo, hi):
    value = RememberAdd(lo)
    value += hi - lo
    return value


@pytest.mark.parametrize("Mn", [1.0, 1000.0, 1e6])
@pytest.mark.parametrize("D", [1.0001, 1.1, 1.5, 2.0, 5.0])
def test_analytic_number_moments(Mn, D):
    law = g2rins.Mwd.from_moments(Mn=Mn, dispersity=D)
    mean, variance = law._distribution.stats(moments="mv")
    assert float(mean) == pytest.approx(Mn, rel=2e-14)
    assert float(variance) == pytest.approx(Mn * Mn * (D - 1), rel=2e-12)
    assert float(mean + variance / mean) == pytest.approx(Mn * D)
    assert law.moments["variance"] == pytest.approx(variance)
    with pytest.raises(TypeError):
        law.moments["Mn"] = 0


def test_unbounded_draws_are_number_weighted_not_mass_weighted():
    law = g2rins.Mwd.make("mwd(1500,1000)")
    rng = np.random.default_rng(42)
    values = np.array([law.draw_mw(rng) for _ in range(30000)])
    assert values.mean() == pytest.approx(1000, rel=0.02)
    assert np.mean(values**2) / values.mean() == pytest.approx(1500, rel=0.03)


@pytest.mark.parametrize("Mw,Mn", [(0, 0), (1, 2), (-1, 1), (np.inf, 1), (2, np.nan)])
def test_invalid_moments(Mw, Mn):
    with pytest.raises(ValueError):
        g2rins.Mwd.from_moments(Mw=Mw, Mn=Mn)


def test_redundant_moments_and_roundtrip():
    law = g2rins.Mwd.from_moments(Mn=1000, Mw=1500, dispersity=1.5)
    assert g2rins.Mwd.make(str(law).strip("|")).serialize() == law.serialize()
    with pytest.raises(ValueError, match="inconsistent"):
        g2rins.Mwd.from_moments(Mn=1000, Mw=1500, dispersity=2)
    with pytest.raises(ValueError, match="Supply"):
        g2rins.Mwd.from_moments(Mn=1000)
    with pytest.raises(ValueError):
        g2rins.Mwd.from_moments(Mn=1000, dispersity=0.9)


def test_point_mass_bounds_probabilities_and_rng():
    law = g2rins.Mwd.make("mwd(1000,1000)")
    rng = np.random.default_rng(7)
    before = copy.deepcopy(rng.bit_generator.state)
    assert law.generable and bool(law)
    assert law.support_mw() == (1000, 1000)
    assert law.draw_mw(rng, lower=1000, upper=1000) == 1000
    assert law.draw_mw(rng) == 1000
    assert rng.bit_generator.state == before
    assert law.prob_mw(1000) == 1
    assert law.prob_mw(999) == 0
    assert law.prob_mw(interval(999, 1000)) == 1
    assert law.prob_mw(interval(1000, 1001)) == 0
    for lo, hi in [(1001, None), (None, 999), (np.nan, 2000), (1000, np.nan)]:
        with pytest.raises(EmptyTruncatedDistributionSupport):
            law.draw_mw(rng, lower=lo, upper=hi)


@pytest.mark.parametrize("lo,hi", [(100, 200), (20000, 21000), (0, 1), (500, None), (None, 500)])
def test_bounded_gamma_draws(lo, hi):
    law = g2rins.Mwd.make("mwd(1500,1000)")
    rng = np.random.default_rng(7)
    for _ in range(20):
        draw = law.draw_mw(rng, lower=lo, upper=hi)
        assert draw >= (0 if lo is None else lo)
        assert draw <= (np.inf if hi is None else hi)


@pytest.mark.parametrize("lo,hi", [(200, 100), (-10, -1), (100, 100), (np.nan, 200)])
def test_empty_gamma_intervals(lo, hi):
    with pytest.raises(EmptyTruncatedDistributionSupport):
        g2rins.Mwd.make("mwd(1500,1000)").draw_mw(np.random.default_rng(0), lower=lo, upper=hi)


def test_pdf_and_interval_probability():
    law = g2rins.Mwd.make("mwd(1500,1000)")
    expected = stats.gamma(a=2, scale=500)
    assert law.prob_mw(750) == pytest.approx(expected.pdf(750))
    assert law.prob_mw(interval(100, 500)) == pytest.approx(expected.cdf(500) - expected.cdf(100))
    assert law.prob_mw(interval(20000, 21000)) == pytest.approx(expected.sf(20000) - expected.sf(21000), abs=0)


def test_new_codec_preserves_retired_slot_and_legacy_widths():
    law = g2rins.Mwd.make("mwd(740,500)")
    vector = law.get_serial_vector()
    assert len(vector) == 16 and vector[12:14] == [-1, -1]
    assert g2rins.StochasticDistribution.from_serial_vector(vector).serialize() == law.serialize()
    with pytest.raises(ValueError, match="Retired"):
        g2rins.StochasticDistribution.from_serial_vector([-1] * 12 + [740, 500])
    vector[12:14] = [740, 500]
    with pytest.raises(ValueError, match="Retired"):
        g2rins.StochasticDistribution.from_serial_vector(vector)
    assert len(g2rins.StochasticDistribution.make("gauss(100,10)").get_serial_vector()) == 12
    assert g2rins.StochasticDistribution.from_serial_vector([-1] * 10) is None
    assert g2rins.StochasticDistribution.from_serial_vector([-1] * 12) is None


def test_mixed_model_transport_without_recipe_and_json_pickle(tmp_path):
    text = "{[] [<]CC({[<] [<]NN[>];; [>]}|poisson(80)|[H])CC[>];; [<][H], [>][H] []}|mwd(1500,1000)|"
    graph_factory = factory(text)
    graph = graph_factory.get_generative_graph()
    assert graph.graph["molecular_weight_distribution_schema"] == MWD_SCHEMA
    assert "nb_recipe_context" not in graph.graph
    assert all(len(v) == 16 for _, d in graph.nodes(data=True) for v in d["molecular_weight_distribution"])
    payload = json.loads(json.dumps(generative_graph_json_data(graph)))
    restored = nx.node_link_graph(payload["graph"], edges="edges")
    validate_mwd_transport(restored)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = g2rins.EnsembleCreator(restored)
    actual = creator.create_ensemble(2, output_format="smiles", seed=7)
    assert actual == pickle.loads(pickle.dumps(creator)).create_ensemble(2, output_format="smiles", seed=7)
    path = tmp_path / "model.json"
    graph_factory.write_generative_graph_json(path)
    assert json.loads(path.read_text())["graph"]["graph"]["molecular_weight_distribution_schema"] == MWD_SCHEMA


@pytest.mark.parametrize("text", [
    LINEAR,
    "{[] [<]CC[>], [<]CO[>];; [<][H], [>][H] []}|mwd(740,500)|",
    "{[] [$]CC([$])C[$];; [$][H] []}|mwd(740,500)|",
    "C{[>][<]CC[>];;[<]}|mwd(740,500)|[H]",
])
def test_standard_mass_sampler_accepts_chemistry_without_nb_certificates(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = factory(text).get_ensemble_creator()
        result = creator.create_ensemble(3, output_format="mol_graph", seed=7)
    assert len(result) == 3
    assert all("nb_count_ledger" not in graph.graph for graph in result)
    assert not hasattr(creator, "_nb_count_distribution")


def test_serial_parallel_seed_parity():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = factory().get_ensemble_creator()
    serial = creator.create_ensemble(4, output_format="smiles", seed=42)
    parallel = creator.create_ensemble(4, output_format="smiles", seed=42, parallel=True, n_workers=2)
    assert serial == parallel


@pytest.mark.parametrize("policy", ["full", "statistics"])
def test_checkpoint_resume_and_old_nb_rejection(policy):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = factory().get_ensemble_creator()
    options = dict(batch_size=2, window=10, output_format="smiles", seed=42,
                   checkpoint_policy=policy, retain_chains=False, retain_sequences=False)
    checkpoints = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator.create_ensemble_until_converged(max_samples=2, checkpoint_callback=checkpoints.append, **options)
        checkpoint = pickle.loads(pickle.dumps(checkpoints[-1]))
        assert checkpoint.settings["mwd_semantics"][0] == "gamma-number-target/v1"
        resumed = creator.create_ensemble_until_converged(max_samples=4, checkpoint=checkpoint, **options)
        whole = creator.create_ensemble_until_converged(max_samples=4, **options)
    assert _metadata_equal(vars(resumed), vars(whole))
    stale = replace(checkpoint, settings={**checkpoint.settings, "nb_model_identity": "old-nb"})
    with pytest.raises(ValueError, match="Retired NB"):
        creator.create_ensemble_until_converged(max_samples=4, checkpoint=stale, **options)
    changed = dict(checkpoint.settings)
    del changed["mwd_semantics"]
    with pytest.raises(ValueError, match="settings"):
        creator.create_ensemble_until_converged(max_samples=4, checkpoint=replace(checkpoint, settings=changed), **options)


@pytest.mark.parametrize("change", ["old_schema", "old_context", "old_width", "missing_schema", "reserved", "replica"])
def test_old_and_malformed_graphs_fail_closed(change):
    graph = copy.deepcopy(factory().get_generative_graph())
    if change == "old_schema":
        graph.graph["molecular_weight_distribution_schema"] = "g2rins-distribution-vector/v3"
    elif change == "old_context":
        graph.graph["nb_recipe_context"] = {}
    elif change == "missing_schema":
        del graph.graph["molecular_weight_distribution_schema"]
    else:
        node = next(iter(graph))
        vectors = copy.deepcopy(graph.nodes[node]["molecular_weight_distribution"])
        if change == "old_width":
            vectors = [v[:14] for v in vectors]
        elif change == "reserved":
            vectors[0][12:14] = [740, 500]
        else:
            vectors[0][-2] += 1
        graph.nodes[node]["molecular_weight_distribution"] = vectors
    with pytest.raises(ValueError):
        g2rins.EnsembleCreator(graph)
    with pytest.raises(ValueError):
        generative_graph_json_data(graph)


def test_retired_public_names_are_not_aliases():
    assert not hasattr(g2rins, "ShiftedNegativeBinomial")
    with pytest.raises(Exception):
        g2rins.StochasticDistribution.make("shifted_negative_binomial(740,500)")
    with pytest.raises(TypeError, match="nb_realization"):
        factory().get_ensemble_creator(nb_realization="physical_compound")


def test_transport_validates_slots_after_the_first_mwd():
    graph = copy.deepcopy(factory().get_generative_graph())
    # All replicas intentionally alias the same array in a freshly built graph.
    vectors = next(iter(graph.nodes(data=True)))[1]["molecular_weight_distribution"]
    vectors[5][12:14] = [740, 500]
    with pytest.raises(ValueError, match="Retired"):
        generative_graph_json_data(graph)
    with pytest.raises(ValueError, match="Retired"):
        g2rins.EnsembleCreator(graph)