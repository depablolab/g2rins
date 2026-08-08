# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the opt-in parallel create_ensemble.

Worker-side semantics are unit-tested in-process against the module-level
worker function (mocks cannot cross a process boundary); the real
ProcessPoolExecutor is spawned only where the process boundary IS the point:
seed reproducibility (doubles as the template-picklability canary), warning
and discard transport, fatal propagation, and the unguarded-script spawn
safety of _no_main_reimport.
"""

import pickle
import subprocess
import sys
import warnings

import numpy as np
import pytest

import g2rins
from g2rins.ensemble_creator import EnsembleCreator, _sample_chain_batch
from g2rins.exception import (
    AllZeroSamplingWeights,
    DiscardedSamplingPaths,
    EmptyTruncatedDistributionSupport,
    TooManyDiscardedChains,
)

FAST_SMI = "{[] [<]CC([>])c1ccccc1; CO[>]; [<][H] []}|gauss(1000, 45)|"
# Same monofunctional-inner-graft template as test_generation_regressions:
# every sample is a truncated chain, so every attempt is a counted discard.
TRUNCATING_SMI = "{[] [<|9.0|]CC(C)O[>|9.0|], [<|6.0|]CC(CC)O[>|6.0|]; {[] [<|7.0|]CCO[>|7.0|], [<|4.0|]CC(CC)O[>|4.0|]; CCCCO[>]; [<] []}|gauss(680.0, 215.0)|[>]; [<][H] []}|gauss(1649.0, 521.5)|"
# Every alternative declared |0|: provably fatal AllZeroSamplingWeights.
FATAL_SMI = "C{[>][<]CC[>]|0|;;[<]}|poisson(900)|[H]"
# One productive and one dead source alternative: per-chain source selection
# yields a genuine mix of successful and retryable-dead paths.
CONDITIONAL_SOURCE_SMI = "{[] [<1]CC[>1], [<2]NN[>2]; C[>1], O[>2]; [<1][H]|0|, [<2][H] []}|uniform(80,80)|"
# One productive zero-target global arm and one dead sibling. Every chain must
# process both declared arms, so the unusable cap rejects every ordering.
DEAD_ARM_SMI = "C(O{[>1][<1]CC[>2];;[<2][H]|0| []}|uniform(80,80)|)(N{[>3][<3]NN[>4];;[<4][H] []}|uniform(0,0)|)"


def test_n_workers_requires_parallel():
    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)
    with pytest.raises(ValueError, match="parallel"):
        ensemble_creator.create_ensemble(2, n_workers=4)


@pytest.mark.parametrize("n_workers", (0, -2))
def test_n_workers_must_be_positive(n_workers):
    """A typo like n_workers=0 must fail fast, not silently degrade to serial."""
    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)
    with pytest.raises(ValueError, match="positive"):
        ensemble_creator.create_ensemble(2, parallel=True, n_workers=n_workers)


def test_single_worker_hatch_spawns_no_pool(monkeypatch):
    """parallel=True with n_workers=1 is the documented escape hatch: it must
    take the serial path and never construct a process pool."""

    def boom(*_args, **_kwargs):
        raise AssertionError("ProcessPoolExecutor must not be constructed for n_workers=1")

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", boom)

    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)
    molecule = "MOL"
    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", lambda **_kwargs: molecule)

    assert ensemble_creator.create_ensemble(2, parallel=True, n_workers=1) == [molecule, molecule]


def test_sample_chain_batch_budget_and_failure_records(monkeypatch):
    """The worker function counts a per-chain consecutive-discard budget,
    preserves chain indices and order, returns failure entries (record=None,
    detached first cause) instead of raising, and its batch survives the
    pickle round-trip the pool depends on."""
    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)
    # Chain 0 fails twice then succeeds; chain 1 always fails.
    outcomes = iter(
        (
            EmptyTruncatedDistributionSupport("nested", 1.0, 2.0),
            EmptyTruncatedDistributionSupport("nested", 3.0, 4.0),
            "MOL",
        )
    )

    calls = {"total": 0}

    def sample(**_kwargs):
        calls["total"] += 1
        try:
            outcome = next(outcomes)
        except StopIteration:
            raise EmptyTruncatedDistributionSupport("nested", 5.0, 6.0) from None
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", sample)

    chain_jobs = list(enumerate(np.random.SeedSequence(0).spawn(2)))
    batch = _sample_chain_batch(ensemble_creator, chain_jobs, "mol_graph", False, 3, None)

    assert [entry["chain_index"] for entry in batch] == [0, 1]

    accepted, failed = batch
    assert accepted["record"]["molecule"] == "MOL"
    assert accepted["discards"] == 2
    assert accepted["reasons"] == (("EmptyTruncatedDistributionSupport", 2),)

    assert failed["record"] is None
    assert failed["discards"] == 3
    assert failed["reasons"] == (("EmptyTruncatedDistributionSupport", 3),)
    assert isinstance(failed["first_cause"], EmptyTruncatedDistributionSupport)
    assert failed["first_cause"].__traceback__ is None
    assert calls["total"] == 3 + 3

    restored = pickle.loads(pickle.dumps(batch))
    assert isinstance(restored[1]["first_cause"], EmptyTruncatedDistributionSupport)


def test_parallel_seed_equivalence():
    """The same seed reproduces the same ensemble across serial and parallel
    (per-chain streams keyed by chain index). Also the picklability canary
    for the whole EnsembleCreator template."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        serial = ensemble_creator.create_ensemble(6, output_format="smiles", seed=7)
        pooled = ensemble_creator.create_ensemble(6, output_format="smiles", seed=7, parallel=True, n_workers=2)
    assert serial == pooled


def test_worker_discards_surface_in_parent():
    """Discards inside worker processes must reach the caller: the aggregated
    reason tally, the budget warning, and the serial total-failure verdict
    (None) all surface in the parent."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(TRUNCATING_SMI).get_graph_creator().get_ensemble_creator()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ensemble_creator.create_ensemble(
            2, output_format="smiles", parallel=True, n_workers=2, max_number_of_discarded_chains=2, seed=0
        )
    assert result is None
    assert any(isinstance(w.message, TooManyDiscardedChains) for w in caught)
    summaries = [w.message for w in caught if isinstance(w.message, DiscardedSamplingPaths)]
    assert len(summaries) == 1
    assert summaries[0].discarded_count == 4
    assert summaries[0].reasons == (("PossibleNonRepresentativePolymerChain", 4),)


def test_fatal_error_propagates_from_workers():
    """A fatal model error raised inside a worker re-raises from
    create_ensemble instead of being retried, matching the serial contract."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(FATAL_SMI).get_graph_creator().get_ensemble_creator()
        with pytest.raises(AllZeroSamplingWeights):
            ensemble_creator.create_ensemble(4, parallel=True, n_workers=2, seed=0)


def test_parallel_partial_ensemble_preserved():
    """Chains that exhaust their per-chain budget must not erase sibling
    successes: the parallel verdict preserves every succeeding chain and
    reports the discards. Roughly half the source-selection streams reject, so
    a budget of 1 over 10 chains yields a mix."""
    from g2rins.exception import DeadSamplingPath

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(CONDITIONAL_SOURCE_SMI).get_graph_creator().get_ensemble_creator()
    for seed in range(16):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = ensemble_creator.create_ensemble(
                    10, output_format="smiles", parallel=True, n_workers=2, max_number_of_discarded_chains=1, seed=seed
                )
            except DeadSamplingPath:
                continue  # every chain drew the dead source: try another seed
        if result is not None and len(result) < 10:
            break
    else:
        pytest.fail("no seed in range(16) produced a partial ensemble")

    assert 0 < len(result) < 10
    assert any(isinstance(w.message, TooManyDiscardedChains) for w in caught)
    summaries = [w.message for w in caught if isinstance(w.message, DiscardedSamplingPaths)]
    assert len(summaries) == 1
    assert summaries[0].discarded_count == 10 - len(result)


def test_parallel_total_failure_reraises_transported_cause():
    """When every chain exhausts its budget on a retryable dead end, the
    parent re-raises the DeadSamplingPath cause that crossed the boundary."""
    from g2rins.exception import DeadSamplingPath

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(DEAD_ARM_SMI).get_graph_creator().get_ensemble_creator()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(DeadSamplingPath) as raised_info:
            ensemble_creator.create_ensemble(
                1, output_format="smiles", parallel=True, n_workers=2, max_number_of_discarded_chains=1, seed=0
            )

    raised = raised_info.value
    assert raised.__traceback__ is not None  # re-raised in the parent
    summaries = [w.message for w in caught if isinstance(w.message, DiscardedSamplingPaths)]
    assert len(summaries) == 1 and summaries[0].discarded_count == 1


def test_unguarded_script_runs_once_and_completes(tmp_path):
    """The pool skips re-importing the calling script (_no_main_reimport), so
    a deliberately guard-less top-level script must run its body exactly once
    and finish — instead of the classic spawn re-execution/BrokenProcessPool.
    Also the alarm for stdlib drift in multiprocessing.spawn internals."""
    script = tmp_path / "unguarded.py"
    script.write_text(
        "import warnings\n"
        "import g2rins\n"
        "print('SCRIPT-BODY')\n"
        "with warnings.catch_warnings():\n"
        "    warnings.simplefilter('ignore')\n"
        "    ensemble_creator = g2rins.G2rins.make('{[] [<]CC[>]; C[>]; [<][H] []}|poisson(400.0)|').get_graph_creator().get_ensemble_creator()\n"
        "    chains = ensemble_creator.create_ensemble(2, output_format='smiles', parallel=True, n_workers=2, seed=1)\n"
        "print(f'UNGUARDED-OK {len(chains)}')\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-u", str(script)], capture_output=True, text=True, timeout=300
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("SCRIPT-BODY") == 1
    assert "UNGUARDED-OK 2" in completed.stdout
