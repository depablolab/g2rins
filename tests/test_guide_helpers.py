# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Small, headless contracts for the optional repository-local guide helpers."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def guide(monkeypatch):
    # These are deliberately optional; the base library/test extra has no GUI
    # dependencies. Install .[test,notebook] to run the example contracts too.
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("IPython.display")
    matplotlib.use("Agg", force=True)
    from examples import guide_helpers

    with monkeypatch.context() as patch:
        patch.setattr(guide_helpers.plt, "show", lambda: None)
        yield guide_helpers
    guide_helpers.plt.close("all")


def _sampling_result(guide, monkeypatch, result):
    creator = SimpleNamespace(create_ensemble=lambda *args, **kwargs: result)
    graph_creator = SimpleNamespace(get_ensemble_creator=lambda: creator)
    model = SimpleNamespace(get_graph_creator=lambda: graph_creator)
    monkeypatch.setattr(guide.g2rins.G2rins, "make", lambda _source: model)


def test_setup_resets_rng_without_invalidating_imported_aliases(guide):
    imported_rng = guide.rng
    try:
        guide.reset_example_rng()
        first = imported_rng.random(4)
        guide.reset_example_rng()
        assert guide.rng is imported_rng
        np.testing.assert_array_equal(imported_rng.random(4), first)
        np.testing.assert_array_equal(first, np.random.default_rng(42).random(4))
        guide.reset_example_rng(7)
        np.testing.assert_array_equal(imported_rng.random(4), np.random.default_rng(7).random(4))
    finally:
        # Do not leak the example's generator state into other tests.
        if hasattr(guide, "reset_example_rng"):
            guide.reset_example_rng()


def test_notebook_setup_explicitly_resets_cached_helper_rng():
    notebook = json.loads((Path(__file__).parents[1] / "G2RINS_guide.ipynb").read_text())
    setup = next(
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "from examples.guide_helpers import" in "".join(cell["source"])
    )
    tree = ast.parse("".join(setup["source"]))
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "reset_example_rng"
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("options", [
    {"n_chains": 0}, {"n_chains": -1}, {"chunk_size": 0}, {"chunk_size": -1},
])
def test_chunked_sampling_rejects_nonpositive_counts_before_sampling(guide, monkeypatch, options):
    def unexpected(_source):
        pytest.fail("invalid chunk counts must fail before parsing/sampling")

    monkeypatch.setattr(guide.g2rins.G2rins, "make", unexpected)
    with pytest.raises(ValueError, match="positive"):
        guide.create_ensemble_and_plot_mw("C", **options)


@pytest.mark.parametrize("weights", [{}, {0: []}])
def test_accepted_chains_without_mass_data_do_not_create_empty_plot(guide, monkeypatch, capsys, weights):
    result = SimpleNamespace(chains=["C"], mol_weights=weights, distributions={})
    _sampling_result(guide, monkeypatch, result)
    monkeypatch.setattr(guide.plt, "subplots", lambda *args, **kwargs: pytest.fail("no mass data to plot"))
    assert guide.create_ensemble_and_plot_mw("C", n_chains=1) == ["C"]
    assert "No molecular-weight data" in capsys.readouterr().out


@pytest.mark.parametrize("labels, message", [
    ([], "No backbone"),
    ([["I0"]], "No backbone"),
    ([["I0"], ["T0"]], "No backbone"),
    ([["I0"], ["R0", "R0"], ["T0"]], "No grown side chains"),
])
def test_graft_report_handles_empty_or_non_grafted_samples(guide, monkeypatch, capsys, labels, message):
    result = SimpleNamespace(sequence_unit_ids=[labels] if labels else [])
    _sampling_result(guide, monkeypatch, result)
    monkeypatch.setattr(guide.plt, "subplots", lambda *args, **kwargs: pytest.fail("no graft data to plot"))
    with np.errstate(all="raise"):
        guide.report_graft_statistics("example", n_chains=1)
    assert message in capsys.readouterr().out


def test_graft_report_uses_sidecars_without_reconstructing_fragment_identity(guide, monkeypatch, capsys):
    # No graph/unit dictionary is supplied: labels alone define this heuristic.
    result = SimpleNamespace(sequence_unit_ids=[
        [["I0"], ["R0", "R0"], ["R1", "R1"], ["T0"]],
        [["I0"], ["R0", "R0", "R0"], ["R1"], ["R1", "R1"], ["T0"]],
    ])
    _sampling_result(guide, monkeypatch, result)
    guide.report_graft_statistics("example", n_chains=2)
    output = capsys.readouterr().out
    assert "Backbone DP: mean 2.5" in output
    assert "Side chain DP: mean 1.67, max 2" in output


@pytest.mark.parametrize("chains, message", [
    ([], "No chains"), (["CC"], "No nitrogen atoms"),
])
def test_amine_report_handles_no_chains_or_no_nitrogen(guide, monkeypatch, capsys, chains, message):
    _sampling_result(guide, monkeypatch, chains)
    guide.report_amine_branching("example", n_chains=1)
    assert message in capsys.readouterr().out


def test_amine_report_preserves_branching_denominator(guide, monkeypatch, capsys):
    _sampling_result(guide, monkeypatch, ["CN", "CNC", "CN(C)C"])
    guide.report_amine_branching("example", n_chains=3)
    output = capsys.readouterr().out
    assert "3 nitrogen atoms in 3 chains (1.0 per chain)" in output
    assert "Degree of branching (terminal + dendritic)/(all N): 0.67" in output


def test_static_smiles_mass_stream_still_plots_without_target_distribution(guide):
    assert guide.create_ensemble_and_plot_mw("C", n_chains=1) == ["C"]