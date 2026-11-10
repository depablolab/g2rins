# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Exact worker-side counts, real process transport, and scheduler recovery."""

import json
import os
import pickle
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace

import pytest

import g2rins.ensemble_creator as module
from g2rins._building_block_statistics import _BuildingBlockStatistics
from tests._support import (
    CONVERGENCE, FAST_SMI, SOURCES, _bond_record_oracle, _creator, _ordinary_fields,
)


_ORIGINAL_JOB = module._sample_chain_job


def _checked_statistics_job(*args, **kwargs):
    """Picklable instrumentation: prove sampling/counting runs outside parent."""
    creator = module._WORKER_ENSEMBLE_CREATOR
    assert os.getpid() != creator._statistics_parent_pid
    result = _ORIGINAL_JOB(*args, **kwargs)
    assert isinstance(result["building_block_statistics"], _BuildingBlockStatistics)
    assert result["building_block_statistics"].chain_count == 1
    return result


def _one_chain(labels, edges):
    counters = _BuildingBlockStatistics()
    metadata = SimpleNamespace(occurrences=[SimpleNamespace(unit_id=label) for label in labels])
    counters.update(metadata, ({}, edges, {}))
    return counters


def test_merge_raw_counts_not_worker_probabilities_or_local_motif_ids():
    first = _one_chain(["R1", "T0"], [(0, 1, ("1", False, "1", "1"))])
    second = _one_chain(["R1", "R1", "I0"], [
        (0, 2, ("1", False, "1", "2")),
        (1, 2, ("1", False, "1", "3")),
    ])
    empty = _one_chain([], [])
    snapshots = [item.finalize() for item in (first, second, empty)]
    result = _BuildingBlockStatistics()
    for item in (first, second, empty):
        result.merge(pickle.loads(pickle.dumps(item)))
    reverse = _BuildingBlockStatistics()
    for item in (empty, second, first):
        reverse.merge(item)
    assert result.finalize() == reverse.finalize()
    assert [item.finalize() for item in (first, second, empty)] == snapshots
    statistics = result.finalize()
    assert statistics["accepted_count"] == 3
    r1 = next(row for row in statistics["building_blocks"] if row["unit_id"] == "R1")
    assert (r1["occurrence_count"], r1["chain_presence_count"]) == (3, 2)
    rows = [row for row in statistics["site_connections"] if row["source_unit"] == "R1"]
    assert {row["probability"] for row in rows} == {1 / 3}
    assert all(row["occupied_endpoint_count"] == 3 for row in rows)
    with pytest.raises(TypeError):
        result.merge(statistics)


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("api, output_format, metadata", [
    ("fixed", "smiles", True), ("fixed", "mol_graph", True),
    ("convergence", "smiles", True), ("convergence", "mol_graph", False),
])
def test_real_parallel_matches_serial_counts_and_parallel_disabled_fields(
    source, api, output_format, metadata, monkeypatch, tmp_path,
):
    creator = _creator(source)
    creator._statistics_parent_pid = os.getpid()
    if api == "fixed":
        options = {"ensemble_info": True, "output_format": output_format, "seed": 7}
        baseline = creator.create_ensemble(4, parallel=True, n_workers=2, **options)
        serial = creator.create_ensemble(4, building_block_statistics=True, **options)
        path = tmp_path / "parallel.json"
        monkeypatch.setattr(module, "_sample_chain_job", _checked_statistics_job)
        enabled = creator.create_ensemble(
            4, parallel=True, n_workers=2, building_block_statistics=True,
            json_file=str(path), json_max_chains=1, **options,
        )
        payload = json.loads(path.read_text())["ensemble"]
        assert payload["building_block_statistics"] == enabled.building_block_statistics
        assert len(payload["chains"]) == 1
    else:
        options = {
            **CONVERGENCE, "output_format": output_format, "metadata": metadata,
            "retain_chains": metadata, "retain_sequences": metadata,
        }
        baseline = creator.create_ensemble_until_converged(parallel=True, n_workers=2, **options)
        serial = creator.create_ensemble_until_converged(building_block_statistics=True, **options)
        monkeypatch.setattr(module, "_sample_chain_job", _checked_statistics_job)
        enabled = creator.create_ensemble_until_converged(
            parallel=True, n_workers=2, building_block_statistics=True, **options,
        )
        assert enabled.convergence_trace[-1]["n_samples"] == 4
    assert module._metadata_equal(_ordinary_fields(enabled), _ordinary_fields(baseline))
    assert enabled.building_block_statistics == serial.building_block_statistics
    assert enabled.building_block_statistics["accepted_count"] == 4
    assert enabled.bond_sequences is None and enabled.chain_motif_index is None


@pytest.mark.parametrize("selection", [
    {"reservoir_size": 0}, {"reservoir_size": 1}, {"representative_distance": 100.0},
])
def test_parallel_selection_counts_all_accepted_not_just_retained(selection):
    # The independent topology oracle needs all raw occurrences. Ordinary H
    # caps are intentionally omitted from topology, but remain in raw counts.
    creator = _creator(FAST_SMI.replace("[<][H]", "[<]C"))
    reference = creator.create_ensemble_until_converged(retain_bond_sequences=True, **CONVERGENCE)
    result = creator.create_ensemble_until_converged(
        parallel=True, n_workers=2, building_block_statistics=True,
        **CONVERGENCE, **selection,
    )
    assert result.building_block_statistics == _bond_record_oracle(reference.bond_sequences)
    assert len(result.chains) <= 1


@pytest.mark.parametrize("api", ["fixed", "convergence"])
def test_single_worker_uses_existing_serial_escape_hatch(api, monkeypatch):
    creator = _creator()
    def no_pool(*args, **kwargs):
        pytest.fail("n_workers=1 must not start a worker pool")
    monkeypatch.setattr(module.concurrent.futures, "ProcessPoolExecutor", no_pool)
    if api == "fixed":
        result = creator.create_ensemble(
            2, ensemble_info=True, seed=7, parallel=True, n_workers=1,
            building_block_statistics=True,
        )
    else:
        result = creator.create_ensemble_until_converged(
            **CONVERGENCE, parallel=True, n_workers=1, building_block_statistics=True,
        )
    assert result.building_block_statistics["accepted_count"] == len(result.chains)


@pytest.mark.parametrize("api", ["fixed", "convergence"])
def test_restarted_jobs_do_not_double_count_worker_contributions(api, monkeypatch):
    creator = _creator()
    options = {"seed": 7, "output_format": "smiles"}
    if api == "fixed":
        reference = creator.create_ensemble(4, ensemble_info=True, building_block_statistics=True, **options)
    else:
        reference = creator.create_ensemble_until_converged(**CONVERGENCE, building_block_statistics=True)
    submissions = 0
    pools = 0
    class RestartOnceExecutor:
        def __init__(self, *, initializer, initargs, **kwargs):
            nonlocal pools
            pools += 1
            initializer(*initargs)
        def submit(self, function, *args):
            nonlocal submissions
            submissions += 1
            future = Future()
            if submissions == 3:
                future.set_exception(BrokenProcessPool("one simulated worker death"))
            else:
                future.set_result(function(*args))
            return future
        def shutdown(self, **kwargs):
            pass
    monkeypatch.setattr(module.concurrent.futures, "ProcessPoolExecutor", RestartOnceExecutor)
    if api == "fixed":
        result = creator.create_ensemble(
            4, ensemble_info=True, parallel=True, n_workers=2,
            building_block_statistics=True, **options,
        )
    else:
        result = creator.create_ensemble_until_converged(
            **CONVERGENCE, parallel=True, n_workers=2, building_block_statistics=True,
        )
    assert pools == 2
    assert result.building_block_statistics == reference.building_block_statistics


def _checked_disabled_job(*args, **kwargs):
    assert len(args) == 11 and not kwargs
    def no_statistics(*args, **kwargs):
        raise AssertionError("disabled worker must not construct statistics")
    module._BuildingBlockStatistics = no_statistics
    result = _ORIGINAL_JOB(*args, **kwargs)
    assert "building_block_statistics" not in result
    return result


@pytest.mark.parametrize("api", ["fixed", "convergence"])
def test_disabled_parallel_does_not_add_worker_statistics_or_job_arguments(api, monkeypatch):
    creator = _creator()
    def no_statistics(*args, **kwargs):
        pytest.fail("disabled parent must not construct statistics")
    monkeypatch.setattr(module, "_BuildingBlockStatistics", no_statistics)
    monkeypatch.setattr(module, "_sample_chain_job", _checked_disabled_job)
    if api == "fixed":
        result = creator.create_ensemble(2, ensemble_info=True, parallel=True, n_workers=2, seed=7)
    else:
        result = creator.create_ensemble_until_converged(
            **CONVERGENCE, parallel=True, n_workers=2, metadata=False,
            retain_chains=False, retain_sequences=False,
        )
    assert result.building_block_statistics is None


def test_worker_retry_counts_only_success_and_failed_job_has_no_contribution(monkeypatch):
    import numpy as np
    creator = _creator()
    reference = creator.create_ensemble(
        1, ensemble_info=True, seed=7, building_block_statistics=True,
    ).building_block_statistics
    original = module._attempt_chain
    attempts = 0
    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return None, {"forced-test-retry": 1}, None, []
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "_attempt_chain", fail_once)
    options = {"molecule_format": "smiles", "collect_info": module._MetadataLevel.COMPACT_SEQUENCES,
               "max_discards": 2, "termination_flag": None, "building_block_statistics": True}
    jobs = [(0, np.random.SeedSequence(7).spawn(1)[0])]
    success = module._sample_chain_batch(creator, jobs, **options)[0]
    assert success["discards"] == 1
    assert success["building_block_statistics"].finalize() == reference
    monkeypatch.setattr(module, "_attempt_chain", lambda *args, **kwargs: (None, {}, None, []))
    failure = module._sample_chain_batch(creator, jobs, **options)[0]
    assert failure["record"] is None and failure["discards"] == 2
    assert "building_block_statistics" not in failure


def test_default_workers_and_serial_final_batch_share_one_population(monkeypatch):
    creator = _creator()
    monkeypatch.setattr(module.os, "cpu_count", lambda: 4)
    options = {**CONVERGENCE, "max_samples": 3, "metadata": False,
               "retain_chains": False, "retain_sequences": False}
    serial = creator.create_ensemble_until_converged(building_block_statistics=True, **options)
    parallel = creator.create_ensemble_until_converged(
        parallel=True, building_block_statistics=True, **options,
    )
    assert parallel.building_block_statistics == serial.building_block_statistics
    assert parallel.convergence_trace[-1]["n_samples"] == 3


def test_lean_parallel_counts_do_not_run_in_parent_or_materialize_output(monkeypatch):
    creator = _creator()
    options = {**CONVERGENCE, "metadata": False,
               "retain_chains": False, "retain_sequences": False}
    serial = creator.create_ensemble_until_converged(building_block_statistics=True, **options)
    def unexpected(*args, **kwargs):
        pytest.fail("lean parallel counts must remain in workers without retained/path work")
    for name in ("_build_occurrence_adjacency", "_unit_path_counts",
                 "_materialize_deferred_chain", "_compact_bond_sequence_record",
                 "_bond_sequence_record_encoded_size"):
        monkeypatch.setattr(module, name, unexpected)
    parallel = creator.create_ensemble_until_converged(
        parallel=True, n_workers=2, building_block_statistics=True, **options,
    )
    assert parallel.building_block_statistics == serial.building_block_statistics
    assert parallel.chains == parallel.sequences == parallel.molecular_weights == []


def test_parallel_early_convergence_and_callback_preserve_population():
    creator = _creator()
    options = {**CONVERGENCE, "max_samples": 20, "window": 2,
               "mass_tolerance": 1e6, "contact_tolerance": 1e6}
    serial = creator.create_ensemble_until_converged(building_block_statistics=True, **options)
    callbacks = []
    parallel = creator.create_ensemble_until_converged(
        parallel=True, n_workers=2, building_block_statistics=True,
        sample_callback=lambda index, record: callbacks.append((index, record)),
        **options,
    )
    assert parallel.converged and serial.converged
    count = parallel.building_block_statistics["accepted_count"]
    assert count < 20 and len(callbacks) == count
    assert [index for index, _record in callbacks] == list(range(count))
    assert parallel.building_block_statistics == serial.building_block_statistics