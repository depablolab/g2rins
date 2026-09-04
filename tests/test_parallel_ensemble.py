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
import json
import multiprocessing.spawn
import subprocess
import sys
import threading
import warnings
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pytest

import g2rins
import g2rins.ensemble_creator as ensemble_module
from g2rins.ensemble_creator import EnsembleCreator, _sample_chain_batch
from g2rins.exception import (
    AllZeroSamplingWeights,
    DiscardedSamplingPaths,
    EmptyTruncatedDistributionSupport,
    TooManyDiscardedChains,
    WorkerProcessFailure,
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


def test_max_worker_restarts_must_be_non_negative():
    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)
    with pytest.raises(ValueError, match="max_worker_restarts"):
        ensemble_creator.create_ensemble(
            2,
            parallel=True,
            n_workers=2,
            max_worker_restarts=-1,
        )


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


def test_create_ensemble_rescues_remaining_jobs_after_worker_crash(monkeypatch):
    submission_count = 0

    class MidStreamBrokenExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def submit(self, function, *args):
            nonlocal submission_count
            submission_count += 1
            future = Future()
            if submission_count == 3:
                future.set_exception(BrokenProcessPool("simulated worker death"))
            else:
                future.set_result(function(*args))
            return future

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(
        ensemble_module.concurrent.futures,
        "ProcessPoolExecutor",
        MidStreamBrokenExecutor,
    )
    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)
    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", lambda **_kwargs: "MOL")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chain_results = list(
            ensemble_creator._iter_chain_records(
                n_samples=5,
                molecule_format="mol_graph",
                collect_info=False,
                max_discards=2,
                termination_flag=None,
                parallel=True,
                n_workers=2,
                seed=6,
                max_worker_restarts=0,
                fallback_on_worker_crash=True,
            )
        )
        result = ensemble_creator.create_ensemble(
            5,
            output_format="mol_graph",
            parallel=True,
            n_workers=2,
            seed=6,
            max_worker_restarts=0,
        )

    assert [entry["chain_index"] for entry in chain_results] == [0, 1, 2, 3, 4]
    assert len(result) == 5
    assert result == ["MOL"] * 5
    assert any(
        "continuing remaining chain jobs in serial mode" in str(w.message)
        for w in caught
    )


def test_create_ensemble_can_disable_worker_crash_fallback(monkeypatch):
    class BrokenExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def submit(self, _function, *_args):
            future = Future()
            future.set_exception(BrokenProcessPool("simulated worker death"))
            return future

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(
        ensemble_module.concurrent.futures,
        "ProcessPoolExecutor",
        BrokenExecutor,
    )
    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)
    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", lambda **_kwargs: "MOL")

    with pytest.raises(WorkerProcessFailure):
        ensemble_creator.create_ensemble(
            2,
            parallel=True,
            n_workers=2,
            max_worker_restarts=0,
            fallback_on_worker_crash=False,
        )


def test_convergence_rescues_remaining_jobs_after_worker_crash(monkeypatch):
    submission_count = 0

    class MidStreamBrokenExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def submit(self, function, *args):
            nonlocal submission_count
            submission_count += 1
            future = Future()
            if submission_count == 4:
                future.set_exception(BrokenProcessPool("simulated worker death"))
            else:
                future.set_result(function(*args))
            return future

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(
        ensemble_module.concurrent.futures,
        "ProcessPoolExecutor",
        MidStreamBrokenExecutor,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        creator = (
            g2rins.G2rins.make(FAST_SMI)
            .get_graph_creator()
            .get_ensemble_creator()
        )
        result = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=4,
            window=10,
            output_format="smiles",
            seed=57,
            parallel=True,
            n_workers=2,
            max_worker_restarts=0,
            fallback_on_worker_crash=True,
        )

    assert len(result.chains) == 4
    assert any(
        "continuing remaining chain jobs in serial mode" in str(w.message)
        for w in caught
    )


def test_create_ensemble_aggregates_directional_stereo_warnings(monkeypatch):
    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)

    chain_result = {
        "chain_index": 0,
        "record": {"molecule": "MOL"},
        "discards": 0,
        "reasons": (),
        "first_cause": None,
        "warnings": [
            (
                "Incomplete double-bond directional markers near atoms 0-1; E/Z stereochemistry is left unspecified.",
                RuntimeWarning,
                __file__,
                1,
            ),
            (
                "Incomplete double-bond directional markers near atoms 2-3; E/Z stereochemistry is left unspecified.",
                RuntimeWarning,
                __file__,
                1,
            ),
            (
                "Ambiguous double-bond directional markers near atoms 4-5; directional stereoinformation was discarded and E/Z stereochemistry is left unspecified.",
                RuntimeWarning,
                __file__,
                1,
            ),
            ("transported non-directional warning", UserWarning, __file__, 1),
        ],
    }

    monkeypatch.setattr(
        ensemble_creator,
        "_iter_chain_records",
        lambda **_kwargs: iter([chain_result]),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        molecules = ensemble_creator.create_ensemble(1, output_format="smiles")

    assert molecules == ["MOL"]

    runtime_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
    ]
    assert len(runtime_warnings) == 1
    summary = str(runtime_warnings[0].message)
    assert summary.startswith(
        "Encountered unresolved double-bond directional markers during ensemble conversion"
    )
    assert "incomplete=2" in summary
    assert "ambiguous=1" in summary

    user_warnings = [
        warning for warning in caught if issubclass(warning.category, UserWarning)
    ]
    assert len(user_warnings) == 1
    assert str(user_warnings[0].message) == "transported non-directional warning"


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


def test_parallel_compact_metadata_record_equivalence():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = (
            g2rins.G2rins.make(FAST_SMI)
            .get_graph_creator()
            .get_ensemble_creator()
        )
        serial = creator.create_ensemble(
            3,
            output_format="smiles",
            ensemble_info=True,
            seed=31,
        )
        pooled = creator.create_ensemble(
            3,
            output_format="smiles",
            ensemble_info=True,
            seed=31,
            parallel=True,
            n_workers=2,
        )

    assert serial.chains == pooled.chains
    assert serial.sequences == pooled.sequences
    assert serial.bonds == pooled.bonds
    assert serial.mol_weights == pooled.mol_weights
    assert serial.distributions == pooled.distributions
    assert serial.molecular_weights == pooled.molecular_weights
    assert {
        unit_id: {
            "psmiles": info["psmiles"],
            "g2rins": info["g2rins"],
            "count": info["count"],
        }
        for unit_id, info in serial.units.items()
    } == {
        unit_id: {
            "psmiles": info["psmiles"],
            "g2rins": info["g2rins"],
            "count": info["count"],
        }
        for unit_id, info in pooled.units.items()
    }


def test_parallel_deferred_convergence_record_equivalence():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = (
            g2rins.G2rins.make(FAST_SMI)
            .get_graph_creator()
            .get_ensemble_creator()
        )
        settings = {
            "batch_size": 2,
            "max_samples": 4,
            "window": 10,
            "output_format": "smiles",
            "seed": 37,
            "reservoir_size": 2,
        }
        serial = creator.create_ensemble_until_converged(**settings)
        pooled = creator.create_ensemble_until_converged(
            **settings,
            parallel=True,
            n_workers=2,
        )

    assert serial.chains == pooled.chains
    assert serial.sequences == pooled.sequences
    assert serial.bonds == pooled.bonds
    assert serial.mol_weights == pooled.mol_weights
    assert serial.distributions == pooled.distributions
    assert serial.molecular_weights == pooled.molecular_weights
    assert serial.convergence_trace == pooled.convergence_trace
    assert serial.number_average_molecular_weight == pytest.approx(
        pooled.number_average_molecular_weight
    )
    assert serial.weight_average_molecular_weight == pytest.approx(
        pooled.weight_average_molecular_weight
    )


def test_parallel_convergence_reuses_one_executor_across_batches(monkeypatch):
    executor_count = 0

    class ImmediateExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            nonlocal executor_count
            executor_count += 1
            initializer(*initargs)

        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            return future

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(
        ensemble_module.concurrent.futures,
        "ProcessPoolExecutor",
        ImmediateExecutor,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = (
            g2rins.G2rins.make(FAST_SMI)
            .get_graph_creator()
            .get_ensemble_creator()
        )
        result = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=4,
            window=10,
            seed=41,
            parallel=True,
            n_workers=2,
        )

    assert executor_count == 1
    assert result.n_batches == 2
    assert len(result.chains) == 4


def test_parallel_convergence_restarts_broken_pool_between_batches(monkeypatch):
    executor_count = 0
    submitted_indices = []

    class BetweenBatchFailureExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            nonlocal executor_count
            executor_count += 1
            self.generation = executor_count
            initializer(*initargs)

        def submit(self, function, *args):
            chain_index = args[0][0]
            submitted_indices.append((self.generation, chain_index))
            if self.generation == 1 and chain_index >= 2:
                raise BrokenProcessPool("simulated death between batches")
            future = Future()
            future.set_result(function(*args))
            return future

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(
        ensemble_module.concurrent.futures,
        "ProcessPoolExecutor",
        BetweenBatchFailureExecutor,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = (
            g2rins.G2rins.make(FAST_SMI)
            .get_graph_creator()
            .get_ensemble_creator()
        )
        result = creator.create_ensemble_until_converged(
            batch_size=2,
            max_samples=4,
            window=10,
            seed=43,
            parallel=True,
            n_workers=2,
            max_worker_restarts=1,
        )

    assert executor_count == 2
    assert submitted_indices.count((1, 0)) == 1
    assert submitted_indices.count((1, 1)) == 1
    assert [index for generation, index in submitted_indices if generation == 2] == [
        2,
        3,
    ]
    assert len(result.chains) == 4


def test_persistent_scheduler_restores_spawn_configuration_after_error(monkeypatch):
    original_preparation = multiprocessing.spawn.get_preparation_data
    for name in ensemble_module._NATIVE_THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, f"original-{name}")

    class FatalExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def submit(self, _function, *_args):
            future = Future()
            future.set_exception(RuntimeError("simulated fatal worker error"))
            return future

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(
        ensemble_module.concurrent.futures,
        "ProcessPoolExecutor",
        FatalExecutor,
    )
    creator = EnsembleCreator.__new__(EnsembleCreator)

    with pytest.raises(RuntimeError, match="simulated fatal worker error"):
        creator.create_ensemble_until_converged(
            batch_size=1,
            max_samples=1,
            window=2,
            seed=47,
            parallel=True,
            n_workers=2,
        )

    assert multiprocessing.spawn.get_preparation_data is original_preparation
    assert {
        name: ensemble_module.os.environ[name]
        for name in ensemble_module._NATIVE_THREAD_ENVIRONMENT
    } == {
        name: f"original-{name}"
        for name in ensemble_module._NATIVE_THREAD_ENVIRONMENT
    }


def test_parallel_callback_does_not_reenter_convergence_scheduler(monkeypatch):
    executor_count = 0

    class ImmediateExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            nonlocal executor_count
            executor_count += 1
            initializer(*initargs)

        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            return future

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(
        ensemble_module.concurrent.futures,
        "ProcessPoolExecutor",
        ImmediateExecutor,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        creator = (
            g2rins.G2rins.make(FAST_SMI)
            .get_graph_creator()
            .get_ensemble_creator()
        )
        nested = []

        def sample_callback(_index, _record):
            nested.extend(
                creator.create_ensemble(
                    1,
                    parallel=True,
                    n_workers=2,
                    seed=53,
                )
            )

        result = creator.create_ensemble_until_converged(
            batch_size=1,
            max_samples=1,
            window=2,
            seed=53,
            parallel=True,
            n_workers=2,
            sample_callback=sample_callback,
        )

    assert executor_count == 2
    assert len(result.chains) == 1
    assert len(nested) == 1


def test_native_diagnostics_record_chain_seed_versions_and_stages(tmp_path):
    diagnostics = tmp_path / "native-state.jsonl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(FAST_SMI).get_graph_creator().get_ensemble_creator()
        chains = ensemble_creator.create_ensemble(
            2,
            output_format="smiles",
            ensemble_info=True,
            seed=7,
            parallel=True,
            n_workers=2,
            native_diagnostics_path=diagnostics,
        )

    states = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    assert len(chains.chains) == 2
    assert {state["chain_index"] for state in states} == {0, 1}
    assert all(state["atom_count"] > 0 for state in states)
    assert all(state["bond_count"] > 0 for state in states)
    assert all(state["g2rins_version"] for state in states)
    assert all(state["rdkit_version"] for state in states)
    assert all(state["worker_pid"] > 0 for state in states)
    for chain_index in (0, 1):
        chain_states = [
            state
            for state in states
            if state["chain_index"] == chain_index
            and state["native_stage"].startswith("molecule-")
        ]
        assert [state["native_stage"] for state in chain_states] == [
            "molecule-build",
            "molecule-sanitize",
            "molecule-property-cache",
            "molecule-smiles",
        ]
        assert chain_states[0]["seed"]["entropy"] == 7
        assert chain_states[0]["seed"]["spawn_key"] == [chain_index]


def test_chain_record_iterator_preserves_global_order_across_modes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = (
            g2rins.G2rins.make(FAST_SMI)
            .get_graph_creator()
            .get_ensemble_creator()
        )
        kwargs = {
            "n_samples": 4,
            "molecule_format": "smiles",
            "collect_info": False,
            "max_discards": 10,
            "termination_flag": None,
            "seed": 7,
            "start_index": 12,
        }
        serial = list(
            ensemble_creator._iter_chain_records(
                **kwargs,
                parallel=False,
                n_workers=None,
            )
        )
        pooled = list(
            ensemble_creator._iter_chain_records(
                **kwargs,
                parallel=True,
                n_workers=2,
            )
        )

    assert [result["chain_index"] for result in serial] == [12, 13, 14, 15]
    assert [result["chain_index"] for result in pooled] == [12, 13, 14, 15]
    assert [result["record"] for result in serial] == [
        result["record"] for result in pooled
    ]


def test_parallel_scheduler_bounds_compact_inflight_jobs(monkeypatch):
    submitted_functions = []
    maximum_inflight = 0
    current_inflight = 0

    class ImmediateExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            self.outstanding = 0
            initializer(*initargs)

        def submit(self, function, *args):
            nonlocal current_inflight, maximum_inflight
            submitted_functions.append((function, args[0]))
            current_inflight += 1
            maximum_inflight = max(maximum_inflight, current_inflight)
            future = Future()
            future.set_result(function(*args))
            return future

        def shutdown(self, **_kwargs):
            self.outstanding = 0

    real_wait = ensemble_module.concurrent.futures.wait

    def tracking_wait(futures, **kwargs):
        nonlocal current_inflight
        done, pending = real_wait(futures, **kwargs)
        current_inflight -= len(done)
        return done, pending

    monkeypatch.setattr(
        ensemble_module.concurrent.futures,
        "ProcessPoolExecutor",
        ImmediateExecutor,
    )
    monkeypatch.setattr(ensemble_module.concurrent.futures, "wait", tracking_wait)
    creator = EnsembleCreator.__new__(EnsembleCreator)
    monkeypatch.setattr(creator, "sample_mol_graph", lambda **_kwargs: "MOL")

    results = list(
        creator._iter_chain_records(
            n_samples=9,
            molecule_format="mol_graph",
            collect_info=False,
            max_discards=2,
            termination_flag=None,
            parallel=True,
            n_workers=2,
            seed=5,
        )
    )

    assert maximum_inflight <= 4
    assert [result["chain_index"] for result in results] == list(range(9))
    assert all(function is ensemble_module._sample_chain_job for function, _job in submitted_functions)
    assert all(isinstance(job, tuple) and len(job) == 2 for _function, job in submitted_functions)


def test_parallel_scheduler_recovers_broken_pool_in_order(monkeypatch):
    executor_count = 0
    first_submission = True

    class RecoveringExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            nonlocal executor_count
            executor_count += 1
            initializer(*initargs)

        def submit(self, function, *args):
            nonlocal first_submission
            if first_submission:
                first_submission = False
                raise BrokenProcessPool("simulated worker death during submit")
            future = Future()
            future.set_result(function(*args))
            return future

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(
        ensemble_module.concurrent.futures,
        "ProcessPoolExecutor",
        RecoveringExecutor,
    )
    creator = EnsembleCreator.__new__(EnsembleCreator)
    monkeypatch.setattr(creator, "sample_mol_graph", lambda **_kwargs: "MOL")

    results = list(
        creator._iter_chain_records(
            n_samples=5,
            molecule_format="mol_graph",
            collect_info=False,
            max_discards=2,
            termination_flag=None,
            parallel=True,
            n_workers=2,
            seed=5,
            max_worker_restarts=1,
        )
    )

    assert executor_count == 2
    assert [result["chain_index"] for result in results] == list(range(5))
    assert [result["record"]["molecule"] for result in results] == ["MOL"] * 5


def test_parallel_scheduler_reports_last_native_state_after_recovery_exhaustion(
    monkeypatch, tmp_path
):
    diagnostics = tmp_path / "native-state.jsonl"
    diagnostics.write_text(
        json.dumps(
            {
                "chain_index": 3,
                "native_stage": "molecule-sanitize",
                "atom_count": 9000,
                "bond_count": 8999,
            }
        )
        + "\n"
        + ("incomplete-native-record" * 600)
    )

    class BrokenExecutor:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def submit(self, _function, *_args):
            future = Future()
            future.set_exception(BrokenProcessPool("simulated worker death"))
            return future

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(
        ensemble_module.concurrent.futures,
        "ProcessPoolExecutor",
        BrokenExecutor,
    )
    creator = EnsembleCreator.__new__(EnsembleCreator)

    with pytest.raises(WorkerProcessFailure) as raised:
        list(
            creator._iter_chain_records(
                n_samples=1,
                molecule_format="mol_graph",
                collect_info=False,
                max_discards=2,
                termination_flag=None,
                parallel=True,
                n_workers=2,
                seed=5,
                native_diagnostics_path=diagnostics,
                max_worker_restarts=0,
            )
        )

    assert raised.value.native_state["chain_index"] == 3
    assert raised.value.native_state["native_stage"] == "molecule-sanitize"
    assert "atoms=9000" in str(raised.value)


def test_spawn_configuration_is_serialized_and_restored(monkeypatch):
    """Concurrent pool setup must not interleave process-global save/restore."""
    original_preparation = multiprocessing.spawn.get_preparation_data
    for name in ensemble_module._NATIVE_THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, f"original-{name}")

    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()

    def hold_first_configuration():
        with ensemble_module._single_native_thread_environment(), ensemble_module._no_main_reimport():
            first_entered.set()
            assert release_first.wait(timeout=5)

    def enter_second_configuration():
        second_started.set()
        with ensemble_module._single_native_thread_environment(), ensemble_module._no_main_reimport():
            second_entered.set()

    first = threading.Thread(target=hold_first_configuration)
    second = threading.Thread(target=enter_second_configuration)
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    assert second_started.wait(timeout=5)
    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()
    assert multiprocessing.spawn.get_preparation_data is original_preparation
    assert {
        name: ensemble_module.os.environ[name]
        for name in ensemble_module._NATIVE_THREAD_ENVIRONMENT
    } == {
        name: f"original-{name}"
        for name in ensemble_module._NATIVE_THREAD_ENVIRONMENT
    }


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
