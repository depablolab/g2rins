# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Lightweight benchmark-contract tests; never run polymer benchmarks."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import benchmark_bond_sequences as benchmark


def _result():
    return SimpleNamespace(
        chains=["CCO"], sequences=[[["CC", "O"]]],
        units={"R0": {"count": 2, "psmiles": "*CC*", "g2rins": "CC", "subgraph": object()}},
        bonds=[{"labels": ["R0.0", "R0.1"], "nodes": ["parse-a", "parse-b"], "count": 2}],
        mol_weights={"parse-stochastic": [46.069]},
        distributions={"parse-stochastic": "poisson(50)"},
        molecular_weights=[46.069], converged=False, n_batches=1,
        convergence_trace=[{"n_samples": 1, "mn": 46.069, "mw": 46.069,
                            "contact_frequencies": {"R0.0|R0.1": 1.0}}],
        number_average_molecular_weight=46.069,
        weight_average_molecular_weight=46.069, dispersity=1.0,
        representative_counts=None, unit_path_statistics={"counts": [2]},
        unit_junction_statistics=None,
        bond_sequence_metadata=None, bond_sequences=None,
        bond_sequence_indices=None, bond_sequence_weights=None,
        building_block_statistics=None,
    )


def _mock_worker(monkeypatch, result, method=None, events=None):
    events = events if events is not None else []
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        events.append("ensemble")
        return result

    creator = SimpleNamespace(create_ensemble_until_converged=method or create)
    graph_creator = SimpleNamespace(get_ensemble_creator=lambda: creator)
    molecule = SimpleNamespace(get_graph_creator=lambda: graph_creator)
    library = SimpleNamespace(G2rins=SimpleNamespace(make=lambda text: molecule))

    def size(record):
        events.append("size")
        return 123

    ensemble = SimpleNamespace(_bond_sequence_record_encoded_size=size)
    package = benchmark._ROOT / "src/g2rins"
    monkeypatch.setattr(benchmark, "_load_source", lambda: (library, ensemble, package))
    monkeypatch.setattr(benchmark, "_source_identity", lambda *args: {"source_package": str(package)})
    monkeypatch.setattr(benchmark, "_verify_imports", lambda *args: {})

    def rss():
        events.append("rss")
        return 12.5

    monkeypatch.setattr(benchmark, "_peak_rss_mib", rss)
    monkeypatch.setattr(benchmark, "_chemistry", lambda value, library, **kw: {"chemical": value})
    return calls, creator


@pytest.mark.parametrize("workload", benchmark._WORKLOADS)
@pytest.mark.parametrize("mode", ["off", "baseline"])
@pytest.mark.parametrize("feature", benchmark._FEATURES)
def test_disabled_and_baseline_omit_every_feature_kwarg(monkeypatch, workload, mode, feature):
    calls, _ = _mock_worker(monkeypatch, _result())
    record = benchmark._worker("small-linear", mode, 1, 7, workload, feature)
    assert len(calls) == 1
    assert set(calls[0]) == {
        "batch_size", "max_samples", "window", "seed", "output_format",
        "retain_chains", "retain_sequences", "metadata",
    }
    assert calls[0]["retain_sequences"] is (workload != "stats")
    assert calls[0]["retain_chains"] is (workload != "stats")
    assert calls[0]["metadata"] is not (feature == "building-block-statistics" and workload == "stats")
    assert calls[0]["output_format"] == ("smiles" if workload in {"retained-smiles", "production-json"} else "mol_graph")
    assert record["mode"] == mode
    assert record["workload"] == workload
    assert record["feature"] == feature


def test_baseline_needs_only_legacy_convergence_api(monkeypatch):
    captured = {}

    def legacy(batch_size, max_samples, window, seed, output_format,
               retain_chains, retain_sequences, metadata):
        captured.update(locals())
        return _result()

    _mock_worker(monkeypatch, _result(), method=legacy)
    record = benchmark._worker("small-linear", "baseline", 1, 7, "production-json")
    assert captured["retain_sequences"] is True
    assert record["topology_records"] == 0
    assert record["output_bytes"] > 0


@pytest.mark.parametrize("workload", benchmark._WORKLOADS)
def test_retained_request_needs_only_enable_flag(workload):
    assert benchmark._request_kwargs("retained", workload, 2, 7) == {
        **benchmark._request_kwargs("off", workload, 2, 7),
        "retain_bond_sequences": True,
    }


@pytest.mark.parametrize("workload", benchmark._WORKLOADS)
def test_building_block_request_enables_only_statistics(workload):
    feature = "building-block-statistics"
    assert benchmark._request_kwargs("retained", workload, 2, 7, feature) == {
        **benchmark._request_kwargs("off", workload, 2, 7, feature),
        "building_block_statistics": True,
    }


@pytest.mark.parametrize("mode", ["off", "retained", "baseline"])
def test_building_block_stats_worker_needs_no_metadata_or_retention(monkeypatch, mode):
    result = _result()
    result.chains = result.sequences = result.molecular_weights = []
    result.units = result.mol_weights = result.distributions = {}
    result.bonds = []
    result.unit_path_statistics = result.unit_junction_statistics = None
    if mode == "retained":
        result.building_block_statistics = {"accepted_count": 1}
    calls, _ = _mock_worker(monkeypatch, result)
    record = benchmark._worker("small-linear", mode, 1, 7, "stats", "building-block-statistics")
    assert calls[0]["metadata"] is False
    assert calls[0]["retain_chains"] is False
    assert calls[0]["retain_sequences"] is False
    assert "retain_bond_sequences" not in calls[0]
    assert calls[0].get("building_block_statistics", False) is (mode == "retained")
    assert record["topology_records"] == record["topology_encoded_bytes"] == 0
    assert record["accepted_samples"] == 1


def test_retained_kwargs_and_rss_precede_sizing_and_normalization(monkeypatch):
    events = []
    result = _result()
    result.bond_sequences = [{"occurrences": [1, 2], "edges": [1]}]
    result.bond_sequence_indices = [0]
    calls, _ = _mock_worker(monkeypatch, result, events=events)
    original = benchmark._parity_record

    def parity(*args):
        events.append("parity")
        return original(*args)

    monkeypatch.setattr(benchmark, "_parity_record", parity)
    original_write = benchmark._write_production_output

    def write(result):
        events.append("write")
        return original_write(result)

    monkeypatch.setattr(benchmark, "_write_production_output", write)
    record = benchmark._worker("small-linear", "retained", 1, 7, "production-json")
    assert events == ["ensemble", "write", "rss", "size", "parity"]
    assert calls[0] == {
        **benchmark._request_kwargs("off", "production-json", 1, 7),
        "retain_bond_sequences": True,
    }
    assert record["peak_rss_mib"] == 12.5
    assert record["topology_encoded_bytes"] == 123
    assert record["output_schema"] == "bond-sequence-benchmark-production/v1"
    assert record["output_write_seconds"] > 0
    assert record["end_to_end_seconds"] >= sum(record[key] for key in (
        "import_seconds", "setup_seconds", "ensemble_seconds", "output_write_seconds"))


def test_production_export_includes_smiles_sequences_masses_and_topology(monkeypatch):
    result = _result()
    result.bond_sequences = [{"occurrences": [1], "edges": []}]
    result.bond_sequence_indices = [0]
    result.bond_sequence_weights = [1]
    fsync = []
    monkeypatch.setattr(benchmark.os, "fsync", lambda fd: fsync.append(fd))
    payload = json.loads(benchmark._write_production_output(result))
    assert payload["chains"] == ["CCO"]
    assert payload["sequences"] == [[["CC", "O"]]]
    assert payload["molecular_weights"] == [46.069]
    assert payload["bond_sequences"] == result.bond_sequences
    assert payload["bond_sequence_indices"] == [0]
    assert "subgraph" not in payload["units"]["R0"]
    assert len(fsync) == 1
    result.bond_sequences = None
    assert "bond_sequences" not in benchmark._production_payload(result)
    assert "building_block_statistics" not in benchmark._production_payload(result)
    result.building_block_statistics = {"schema": "building-block-statistics/v1", "accepted_count": 1}
    assert json.loads(benchmark._write_production_output(result))["building_block_statistics"] == result.building_block_statistics


@pytest.mark.parametrize("field,value", [
    ("chains", ["CCN"]), ("sequences", [[["CO", "O"]]]),
    ("molecular_weights", [47.0]), ("representative_counts", [2]),
    ("bond_sequence_indices", [9]),
    ("bond_sequence_weights", [2]),
    ("convergence_trace", [{"n_samples": 2, "mn": 1.0, "mw": 2.0}]),
    ("bonds", [{"labels": ["R0.0", "R0.1"], "count": 3}]),
    ("mol_weights", {"parse-stochastic": [47.0]}),
    ("distributions", {"parse-stochastic": "poisson(51)"}),
    ("number_average_molecular_weight", 47.0),
    ("weight_average_molecular_weight", 47.0),
])
def test_digest_detects_output_changes(monkeypatch, field, value):
    monkeypatch.setattr(benchmark, "_chemistry", lambda value, library, **kw: value)
    result = _result()
    before = benchmark._digest(benchmark._parity_record(result, None, "production-json"))
    setattr(result, field, value)
    assert benchmark._digest(benchmark._parity_record(result, None, "production-json")) != before


def test_digest_ignores_fresh_parse_ids_and_topology_payload(monkeypatch):
    monkeypatch.setattr(benchmark, "_chemistry", lambda value, library, **kw: value)
    before = _result()
    after = copy.deepcopy(before)
    after.bonds[0]["nodes"] = ["fresh-parse-a", "fresh-parse-b"]
    after.mol_weights = {"new-uuid": [46.069]}
    after.distributions = {"new-uuid": "poisson(50)"}
    after.bond_sequences = [{"topology": "intentionally excluded"}]
    after.bond_sequence_metadata = {"fingerprint": "fresh-parse"}
    after.bond_sequence_indices = [0]
    after.bond_sequence_weights = [1]
    after.building_block_statistics = {"on_only": "intentionally excluded"}
    assert benchmark._digest(benchmark._parity_record(before, None)) == benchmark._digest(benchmark._parity_record(after, None))


def test_stats_compares_only_shared_aggregates_not_retention_side_effects(monkeypatch):
    monkeypatch.setattr(benchmark, "_chemistry", lambda value, library, **kw: value)
    off = _result()
    off.chains = off.sequences = off.molecular_weights = []
    off.mol_weights = {}
    retained = copy.deepcopy(off)
    retained.molecular_weights = [46.069]
    retained.mol_weights = {"new-uuid": [46.069]}
    retained.distributions = {"new-uuid": "poisson(50)"}
    retained.mol_weights[-1] = [15.0, 1.0]
    retained.bond_sequence_indices = [0]
    assert benchmark._digest(benchmark._parity_record(off, None, "stats")) == benchmark._digest(benchmark._parity_record(retained, None, "stats"))


def test_metadata_free_stats_parity_compares_counts_masses_and_trace(monkeypatch):
    monkeypatch.setattr(benchmark, "_chemistry", lambda *a, **k: pytest.fail("no chemistry needed"))
    off = _result()
    off.chains = off.sequences = off.molecular_weights = []
    off.units = off.mol_weights = off.distributions = {}
    off.bonds = []
    off.unit_path_statistics = off.unit_junction_statistics = None
    on = copy.deepcopy(off)
    on.building_block_statistics = {"accepted_count": 1, "building_blocks": [{"unit_id": "R0"}]}
    parity = benchmark._parity_record(off, None, "stats")
    assert benchmark._parity_record(on, None, "stats") == parity
    assert "building_block_statistics" not in parity
    assert parity["unit_path_statistics"] is None
    for field, value in (
        ("n_batches", 2), ("number_average_molecular_weight", 47.0),
        ("weight_average_molecular_weight", 47.0),
        ("convergence_trace", [{"n_samples": 2}]),
    ):
        changed = copy.deepcopy(on)
        setattr(changed, field, value)
        assert benchmark._parity_record(changed, None, "stats") != parity


def test_chemistry_normalizes_smiles_and_graph_node_ids():
    import networkx as nx
    import g2rins

    assert benchmark._chemistry("CCO", g2rins) == benchmark._chemistry("OCC", g2rins)
    assert benchmark._chemistry("c1ccccc1", g2rins) == benchmark._chemistry("C1=CC=CC=C1", g2rins)
    assert benchmark._chemistry("F[C@](Cl)(Br)I", g2rins) != benchmark._chemistry("F[C@@](Cl)(Br)I", g2rins)
    graph = nx.Graph()
    graph.add_node("parse-a", atomic_num=6, aromatic=False, charge=0, num_explicit_h=-1)
    graph.add_node("parse-b", atomic_num=8, aromatic=False, charge=0, num_explicit_h=-1)
    graph.add_edge("parse-a", "parse-b", bond_type=1, aromatic=False)
    changed_ids = nx.relabel_nodes(graph, {"parse-a": "new-a", "parse-b": "new-b"})
    assert benchmark._chemistry(graph, g2rins) == benchmark._chemistry(changed_ids, g2rins)


@pytest.mark.parametrize("feature", benchmark._FEATURES)
def test_subprocess_uses_absolute_current_harness_and_explicit_source(monkeypatch, tmp_path, feature):
    package = tmp_path / "historical/src/g2rins"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        identity = {
            "source_package": str(package),
            "imported_modules": {"g2rins": str(package / "__init__.py"),
                                 "g2rins.ensemble_creator": str(package / "ensemble_creator.py")},
        }
        return SimpleNamespace(stdout=json.dumps({"source_identity": identity}))

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    monkeypatch.setenv(benchmark._SOURCE_ENV, "/wrong/inherited/source")
    benchmark._run_isolated("small-linear", "baseline", 1, 7, "production-json", package.parent.parent, feature)
    command, kwargs = calls[0]
    assert command[:2] == [sys.executable, str(Path(benchmark.__file__).resolve())]
    assert "-m" not in command
    assert kwargs["env"][benchmark._SOURCE_ENV] == str(package)
    assert kwargs["cwd"] == benchmark._ROOT
    assert command[-2:] == ["--workload", "production-json"]
    assert command[command.index("--feature") + 1] == feature
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"


def test_source_injection_happens_before_import(monkeypatch, tmp_path):
    package = tmp_path / "src/g2rins"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    monkeypatch.setenv(benchmark._SOURCE_ENV, str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    imported = []

    def import_module(name):
        assert sys.path[0] == str(package.parent)
        imported.append(name)
        return SimpleNamespace(__file__=str(package / "__init__.py"))

    monkeypatch.setattr(benchmark.importlib, "import_module", import_module)
    verified = []
    monkeypatch.setattr(benchmark, "_verify_imports", lambda path: verified.append(path))
    assert benchmark._load_source()[2] == package
    assert imported == ["g2rins", "g2rins.ensemble_creator"]
    assert verified == [package]


def test_current_worker_overrides_inherited_baseline_routing(monkeypatch):
    package = benchmark._package_path(benchmark._ROOT)
    monkeypatch.setenv(benchmark._SOURCE_ENV, "/wrong/inherited/source")

    def run(command, **kwargs):
        assert kwargs["env"][benchmark._SOURCE_ENV] == str(package)
        return SimpleNamespace(stdout=json.dumps({"source_identity": {
            "source_package": str(package),
            "imported_modules": {"g2rins": str(package / "__init__.py"),
                                 "g2rins.ensemble_creator": str(package / "ensemble_creator.py")},
        }}))

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    benchmark._run_isolated("small-linear", "off", 1, 7)


def test_parent_rejects_worker_import_mismatch(monkeypatch):
    package = benchmark._package_path(benchmark._ROOT)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({"source_identity": {
                            "source_package": str(package),
                            "imported_modules": {"g2rins": "/wrong/g2rins/__init__.py",
                                                 "g2rins.ensemble_creator": "/wrong/g2rins/ensemble_creator.py"},
                        }})))
    with pytest.raises(RuntimeError, match="imported module paths"):
        benchmark._run_isolated("small-linear", "off", 1, 7)


def test_subprocess_error_reports_unsupported_source_stderr(monkeypatch):
    def run(*args, **kwargs):
        raise benchmark.subprocess.CalledProcessError(1, args[0], stderr="Unsupported convergence API")

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="baseline benchmark worker failed.*Unsupported convergence API"):
        benchmark._run_isolated("small-linear", "baseline", 1, 7, source=benchmark._ROOT)


def test_verify_imports_rejects_already_loaded_wrong_tree(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "g2rins", SimpleNamespace(__file__="/wrong/g2rins/__init__.py"))
    with pytest.raises(RuntimeError, match="Source routing mismatch"):
        benchmark._verify_imports(tmp_path)


def test_missing_and_identical_baselines_fail_without_running(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(benchmark, "_run_isolated", lambda *a, **k: pytest.fail("must not run workers"))
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--baseline-source", str(tmp_path / "absent")])
    assert exc.value.code == 2
    assert "Invalid source" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--baseline-source", str(benchmark._ROOT)])
    assert exc.value.code == 2
    assert "Identical baseline/current source-tree hashes" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["missing", "signature"])
def test_unsupported_baseline_api_has_clear_error(monkeypatch, kind):
    _, creator = _mock_worker(monkeypatch, _result())
    if kind == "missing":
        del creator.create_ensemble_until_converged
    else:
        creator.create_ensemble_until_converged = lambda: None
    with pytest.raises(RuntimeError, match="Unsupported.*(?:source|convergence API)"):
        benchmark._worker("small-linear", "baseline", 1, 7)


def _run_record(mode, samples, digest="same"):
    elapsed = {"baseline": 1.0, "off": 1.02, "retained": 1.08}[mode]
    return {
        "parity_digest": digest, "accepted_samples": samples,
        "ensemble_seconds": elapsed, "end_to_end_seconds": elapsed + 0.5,
        "setup_seconds": 0.25, "import_seconds": 0.25,
        "output_write_seconds": 0.0, "peak_rss_mib": 10.0,
        "topology_encoded_bytes": 123 if mode == "retained" else 0,
        "topology_occurrences": 2 if mode == "retained" else 0,
        "source_identity": {"source_tree_sha256": "old" if mode == "baseline" else "current"},
    }


def test_repeated_workloads_and_absent_baseline_are_explicit(monkeypatch, capsys):
    calls = []

    def run(case, mode, samples, seed, workload, source, feature="bond-sequences"):
        calls.append((case, mode, seed, workload, source))
        assert feature == "bond-sequences"
        return _run_record(mode, samples)

    monkeypatch.setattr(benchmark, "_run_isolated", run)
    monkeypatch.setattr(benchmark, "_source_identity", lambda: {})
    assert benchmark.main(["--case", "small-linear", "--runs", "2", "--warmups", "0",
                           "--workload", "stats", "--workload", "production-json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["historical_disabled_gate_status"] == "unverified"
    assert output["configuration"]["feature"] == "bond-sequences"
    assert output["configuration"]["metadata"] is True
    assert "retained_mode_budgets" not in output["configuration"]
    assert len(output["results"]) == 2
    for result in output["results"]:
        assert result["summary"]["disabled_gate"]["status"] == "unverified"
        assert result["summary"]["disabled_gate"]["max_overhead_percent"] == 1
        assert result["summary"]["retained_gate"]["max_overhead_percent"] == 5
        assert result["summary"]["retained_gate"]["status"] == "failed"
        assert result["runs"][0]["order"] == ["off", "retained"]
        assert result["runs"][1]["order"] == ["retained", "off"]
    assert {call[1] for call in calls} == {"off", "retained"}
    assert all(call[-1] is None for call in calls)


def test_historical_paired_medians_parity_and_thresholds(monkeypatch, tmp_path, capsys):
    package = tmp_path / "src/g2rins"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    calls = []

    def run(case, mode, samples, seed, workload, source, feature="bond-sequences"):
        calls.append((mode, source))
        return _run_record(mode, samples)

    monkeypatch.setattr(benchmark, "_run_isolated", run)
    monkeypatch.setattr(benchmark, "_source_identity", lambda: {})
    assert benchmark.main(["--baseline-source", str(tmp_path), "--case", "small-linear",
                           "--runs", "2", "--warmups", "0",
                           "--max-disabled-overhead-percent", "3",
                           "--max-retained-overhead-percent", "10"]) == 0
    output = json.loads(capsys.readouterr().out)
    summary = output["results"][0]["summary"]
    assert summary["baseline_median_ensemble_seconds"] == 1.0
    assert summary["disabled_median_paired_ensemble_overhead_percent"] == pytest.approx(2)
    assert summary["disabled_gate"]["status"] == "passed"
    assert summary["retained_gate"]["status"] == "passed"
    assert summary["historical_all_pairs_same_sampling_result"] is True
    assert output["historical_disabled_gate_status"] == "passed"
    assert [mode for mode, source in calls] == ["off", "retained", "baseline", "baseline", "retained", "off"]
    assert all(source == package for mode, source in calls if mode == "baseline")


def test_parity_failure_is_not_reported_as_success(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "_run_isolated", lambda case, mode, samples, *args:
                        _run_record(mode, samples, digest=mode))
    monkeypatch.setattr(benchmark, "_source_identity", lambda: {})
    assert benchmark.main(["--case", "small-linear", "--runs", "1", "--warmups", "0"]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["results"][0]["summary"]["all_pairs_same_sampling_result"] is False
    assert output["results"][0]["summary"]["retained_gate"]["status"] == "failed"


def test_historical_parity_failure_has_failed_gate(monkeypatch, tmp_path, capsys):
    package = tmp_path / "src/g2rins"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    monkeypatch.setattr(benchmark, "_run_isolated", lambda case, mode, samples, *args:
                        _run_record(mode, samples, digest="different" if mode == "baseline" else "same"))
    monkeypatch.setattr(benchmark, "_source_identity", lambda: {})
    assert benchmark.main(["--baseline-source", str(tmp_path), "--case", "small-linear",
                           "--runs", "1", "--warmups", "0",
                           "--max-disabled-overhead-percent", "100"]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["historical_disabled_gate_status"] == "failed"
    assert output["results"][0]["summary"]["historical_all_pairs_same_sampling_result"] is False


def test_gate_does_not_hide_ensemble_regression_behind_startup():
    assert benchmark._gate(0.5, 1.0, True, ensemble_overhead=10)["status"] == "failed"


def test_legacy_worker_cli_defaults_to_stats(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(benchmark, "_worker", lambda *args: calls.append(args) or {"ok": True})
    assert benchmark.main(["--worker", "--case", "small-linear", "--mode", "off"]) == 0
    assert calls[0] == ("small-linear", "off", 5, 20261017, "stats", "bond-sequences")
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_building_block_cli_routes_warmups_pairs_and_configuration(monkeypatch, capsys):
    calls = []

    def run(case, mode, samples, seed, workload="stats", source=None, feature="bond-sequences"):
        calls.append((mode, workload, feature))
        return _run_record(mode, samples)

    monkeypatch.setattr(benchmark, "_run_isolated", run)
    monkeypatch.setattr(benchmark, "_source_identity", lambda: {})
    assert benchmark.main(["--feature", "building-block-statistics", "--case", "small-linear",
                           "--runs", "1", "--warmups", "1", "--workload", "stats",
                           "--workload", "production-json"]) == 0
    output = json.loads(capsys.readouterr().out)
    config = output["configuration"]
    assert config["feature"] == "building-block-statistics"
    assert config["metadata"] is False
    assert config["retain_chains"] is config["retain_sequences"] is False
    assert config["workload_settings"]["stats"]["metadata"] is False
    assert config["workload_settings"]["production-json"]["metadata"] is True
    assert len(calls) == 8
    assert all(feature == "building-block-statistics" for _, _, feature in calls)
    assert all(row["feature"] == "building-block-statistics" for row in output["results"])


@pytest.mark.parametrize("mode", ["off", "retained"])
def test_building_block_worker_cli_routes_selector(monkeypatch, capsys, mode):
    calls = []
    monkeypatch.setattr(benchmark, "_worker", lambda *args: calls.append(args) or {"ok": True})
    assert benchmark.main(["--worker", "--case", "small-linear", "--mode", mode,
                           "--feature", "building-block-statistics"]) == 0
    assert calls == [("small-linear", mode, 5, 20261017, "stats", "building-block-statistics")]
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_fresh_worker_routes_to_explicit_legacy_package(tmp_path):
    """Exercise real subprocess isolation, not just mocked import routing."""
    package = tmp_path / "src/g2rins"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('''
from types import SimpleNamespace

class G2rins:
    @staticmethod
    def make(text):
        return G2rins()

    def get_graph_creator(self):
        return self

    def get_ensemble_creator(self):
        return self

    def create_ensemble_until_converged(self, batch_size, max_samples, window,
                                       seed, output_format, retain_chains,
                                       retain_sequences, metadata):
        assert not retain_chains and not retain_sequences and metadata
        return SimpleNamespace(
            chains=[], sequences=[], molecular_weights=[], units={}, bonds=[],
            mol_weights={}, distributions={}, converged=False, n_batches=1,
            convergence_trace=[{"n_samples": max_samples}],
            number_average_molecular_weight=16.0,
            weight_average_molecular_weight=16.0, dispersity=1.0,
        )
''')
    (package / "ensemble_creator.py").write_text("# No topology API in this source.\n")
    (package / "generative_graph.py").write_text("# Legacy source identity fixture.\n")
    record = benchmark._run_isolated("small-linear", "baseline", 1, 7, "stats", tmp_path)
    assert record["accepted_samples"] == 1
    assert record["topology_records"] == 0
    assert record["source_identity"]["g2rins_module"] == str(package / "__init__.py")
    assert all(Path(path).is_relative_to(package) for path in record["source_identity"]["imported_modules"].values())


@pytest.mark.parametrize("mode", ["off", "retained", "baseline"])
def test_fresh_building_block_worker_routes_selector_without_retention(tmp_path, mode):
    """CLI/subprocess smoke with a tiny source stub; no polymer sampling."""
    package = tmp_path / "src/g2rins"
    package.mkdir(parents=True)
    expected_features = {"building_block_statistics": True} if mode == "retained" else {}
    (package / "__init__.py").write_text('''
from types import SimpleNamespace

class G2rins:
    @staticmethod
    def make(text):
        return G2rins()

    def get_graph_creator(self):
        return self

    def get_ensemble_creator(self):
        return self

    def create_ensemble_until_converged(self, batch_size, max_samples, window,
                                       seed, output_format, retain_chains,
                                       retain_sequences, metadata, **features):
        assert not retain_chains and not retain_sequences and not metadata
        assert features == EXPECTED_FEATURES
        return SimpleNamespace(
            chains=[], sequences=[], molecular_weights=[], units={}, bonds=[],
            mol_weights={}, distributions={}, converged=False, n_batches=1,
            convergence_trace=[{"n_samples": max_samples}],
            number_average_molecular_weight=16.0,
            weight_average_molecular_weight=16.0, dispersity=1.0,
            building_block_statistics={"accepted_count": max_samples} if features else None,
        )
'''.replace("EXPECTED_FEATURES", repr(expected_features)))
    (package / "ensemble_creator.py").write_text("# No topology API needed.\n")
    (package / "generative_graph.py").write_text("# Source identity fixture.\n")
    record = benchmark._run_isolated("small-linear", mode, 1, 7, "stats", tmp_path,
                                     "building-block-statistics")
    assert record["feature"] == "building-block-statistics"
    assert record["accepted_samples"] == 1
    assert record["topology_records"] == record["topology_encoded_bytes"] == 0
    assert record["source_identity"]["g2rins_module"] == str(package / "__init__.py")