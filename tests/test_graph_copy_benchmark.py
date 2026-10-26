# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Cheap harness-contract tests; no real sampling, timings, or subprocesses."""

import copy
import json
from types import SimpleNamespace
import warnings

import pytest

from benchmarks import benchmark_graph_copy as benchmark


@pytest.fixture(autouse=True)
def no_real_workloads(monkeypatch):
    from benchmarks import benchmark_sampling

    def forbidden(*args, **kwargs):
        pytest.fail("Graph-copy unit tests must not launch benchmarks or sampling")

    monkeypatch.setattr(benchmark, "_isolated", forbidden)
    monkeypatch.setattr(benchmark.subprocess, "run", forbidden)
    monkeypatch.setattr(benchmark_sampling, "_worker", forbidden)


@pytest.fixture
def main_environment(monkeypatch):
    environment = {"python": "fake-python", "versions": {}}
    monkeypatch.setattr(benchmark, "_source_hash", lambda: "stable-source")
    monkeypatch.setattr(benchmark, "_environment", lambda: environment)
    return environment


@pytest.mark.parametrize("workflow", ["sample", "fixed"])
def test_main_alternates_pairs_excludes_warmups_and_preserves_records(
    monkeypatch, tmp_path, capsys, main_environment, workflow,
):
    calls = []
    raw_records = {}
    # Deliberately nonmonotonic; warmup must not influence any median.
    times = [(1000.0, 900.0), (10.0, 5.0), (20.0, 18.0), (40.0, 10.0)]

    def fake_isolated(case, selected_workflow, variant):
        pair = len(calls) // 2
        calls.append((case, selected_workflow, variant))
        record = {
            "seconds": times[pair][variant == "optimized"],
            "peak_rss_mib": 12.5 + pair,
            "fingerprint": "stable-parity",
            "parity": {"accepted": 1, "nested": ["unchanged"]},
        }
        raw_records[pair, variant] = copy.deepcopy(record)
        return record

    monkeypatch.setattr(benchmark, "_isolated", fake_isolated)
    output = tmp_path / "report.json"
    assert benchmark.main([
        "--case", "small-linear", "--workflow", workflow,
        "--runs", "3", "--warmups", "1", "--output", str(output),
    ]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert json.loads(output.read_text()) == report
    assert report["source_tree_sha256"] == "stable-source"
    assert report["environment"] == main_environment
    assert report["workflow"] == workflow
    assert (report["runs"], report["warmups"], report["hash_seed"]) == (3, 1, 0)
    assert len(report["results"]) == 1
    result = report["results"][0]
    assert result["case"] == "small-linear"
    assert result["seed"] == benchmark.CASES["small-linear"].seed
    expected_orders = [
        ["legacy", "optimized"], ["optimized", "legacy"],
        ["legacy", "optimized"], ["optimized", "legacy"],
    ]
    assert calls == [
        ("small-linear", workflow, variant)
        for order in expected_orders for variant in order
    ]
    assert len(result["pairs"]) == 4
    for index, pair in enumerate(result["pairs"]):
        assert pair["pair"] == index + 1
        assert pair["order"] == expected_orders[index]
        assert pair["warmup"] is (index == 0)
        assert pair["parity_match"] is True
        assert pair["legacy_s"] == times[index][0]
        assert pair["optimized_s"] == times[index][1]
        assert pair["paired_saving_fraction"] == pytest.approx(
            1 - times[index][1] / times[index][0]
        )
        assert pair["records"] == {
            variant: raw_records[index, variant] for variant in ("legacy", "optimized")
        }
    assert result["median"] == pytest.approx({
        "legacy_s": 20.0, "optimized_s": 10.0, "paired_saving_fraction": 0.5,
    })
    assert "warmup" in captured.err
    assert "measured" in captured.err


@pytest.mark.parametrize("mismatch_call", [1, 2])
def test_main_rejects_variant_mismatch_and_cross_pair_drift(
    monkeypatch, main_environment, mismatch_call,
):
    calls = []

    def fake_isolated(*args):
        index = len(calls)
        calls.append(args)
        return {
            "seconds": 1.0, "peak_rss_mib": 1.0,
            "fingerprint": "changed" if index >= mismatch_call else "original",
            "parity": {},
        }

    monkeypatch.setattr(benchmark, "_isolated", fake_isolated)
    with pytest.raises(RuntimeError, match=r"Parity mismatch: small-linear, pair"):
        benchmark.main(["--runs", "1", "--warmups", "1"])
    assert len(calls) == (2 if mismatch_call == 1 else 4)


def test_main_rejects_source_mutation_before_writing_report(
    monkeypatch, tmp_path, capsys, main_environment,
):
    hashes = iter(["before", "after"])
    monkeypatch.setattr(benchmark, "_source_hash", lambda: next(hashes))
    monkeypatch.setattr(benchmark, "_isolated", lambda *args: {
        "seconds": 1.0, "peak_rss_mib": 1.0, "fingerprint": "same", "parity": {},
    })
    output = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="Source tree changed during benchmark"):
        benchmark.main(["--runs", "1", "--warmups", "0", "--output", str(output)])
    assert not output.exists()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("argv", [
    ["--runs", "0"], ["--runs", "-1"], ["--runs", "not-an-integer"],
    ["--warmups", "-1"], ["--case", "unknown"], ["--workflow", "unknown"],
    ["--variant", "unknown"], ["--variant", "legacy"], ["--worker"],
    ["--worker", "--variant", "legacy", "--case", "small-linear", "--case", "branched"],
])
def test_cli_rejects_bad_arguments_without_work(monkeypatch, capsys, argv):
    def forbidden():
        pytest.fail("Invalid arguments must fail before source hashing")

    monkeypatch.setattr(benchmark, "_source_hash", forbidden)
    with pytest.raises(SystemExit) as exc:
        benchmark.main(argv)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_source_hash_includes_untracked_helpers_and_relative_names(monkeypatch, tmp_path):
    # A synthetic root without .git proves hashing is independent of tracking.
    package = tmp_path / "src" / "g2rins"
    package.mkdir(parents=True)
    harness = tmp_path / "benchmarks"
    harness.mkdir()
    for name in ("benchmark_graph_copy.py", "benchmark_sampling.py", "cases.py", "_common.py"):
        (harness / name).write_text("# synthetic harness\n")
    (package / "__init__.py").write_text("# synthetic package\n")
    grammar = package / "grammar.lark"
    grammar.write_text('start: "C"\n')
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    original = benchmark._source_hash()
    assert benchmark._source_hash() == original
    helper = package / "nested" / "new_helper.py"
    helper.parent.mkdir()
    helper.write_text("VALUE = 1\n")
    added = benchmark._source_hash()
    assert added != original
    helper.write_text("VALUE = 2\n")
    changed = benchmark._source_hash()
    assert changed != added
    helper.rename(helper.with_name("renamed_helper.py"))
    renamed = benchmark._source_hash()
    assert renamed != changed
    (package / "notes.txt").write_text("ignored non-source\n")
    assert benchmark._source_hash() == renamed
    grammar.write_text('start: "O"\n')
    assert benchmark._source_hash() != renamed


@pytest.mark.parametrize("variant", ["legacy", "optimized"])
@pytest.mark.parametrize("fails", [False, True])
def test_worker_copy_override_is_scoped_even_on_failure(monkeypatch, variant, fails):
    import g2rins.ensemble_creator as module
    from benchmarks import _common, benchmark_sampling

    original = module._copy_unit_graph
    calls = []
    sample_record = {field: index + 1 for index, field in enumerate(benchmark.PARITY_FIELDS)}
    sample_record["sampling_seconds"] = 1.0
    sample_record["not_a_parity_field"] = "excluded"

    def fake_worker(*args):
        calls.append(args)
        if variant == "optimized":
            assert module._copy_unit_graph is original
        else:
            assert module._copy_unit_graph is not original
            graph, cloned = object(), object()
            copied = []

            def deepcopy(value):
                copied.append(value)
                return cloned

            assert module._copy_unit_graph(graph, deepcopy=deepcopy) is cloned
            assert copied == [graph]
        if fails:
            raise ValueError("synthetic sampling failure")
        warnings.warn("synthetic worker warning", UserWarning)
        return sample_record

    monkeypatch.setattr(benchmark_sampling, "_worker", fake_worker)
    monkeypatch.setattr(_common, "_peak_rss_mib", lambda: 12.5)
    if fails:
        with pytest.raises(ValueError, match="synthetic sampling failure"):
            benchmark._worker("small-linear", "sample", variant)
    else:
        record = benchmark._worker("small-linear", "sample", variant)
        expected = {field: sample_record[field] for field in benchmark.PARITY_FIELDS}
        expected["emitted_warnings"] = ["UserWarning"]
        assert record == {
            "seconds": 1.0, "peak_rss_mib": 12.5,
            "fingerprint": benchmark._fingerprint(expected), "parity": expected,
        }
    assert calls == [("small-linear", True, "exact", 100)]
    assert module._copy_unit_graph is original


def test_fixed_payload_uses_protected_writer_and_stable_metadata(monkeypatch):
    import g2rins
    from rdkit import Chem

    calls = []

    def protected_writer(molecule, *, smiles_policy):
        calls.append((molecule.GetNumAtoms(), smiles_policy))
        return Chem.MolToSmiles(molecule, canonical=True)

    monkeypatch.setattr(g2rins, "rdkit_mol_to_smiles", protected_writer)
    result = SimpleNamespace(
        chains=["OCC"], sequences=[[["OC", "[H]O"]]],
        sequence_unit_ids=[[["R0", "R1"]]],
        units={"R0": {"count": 2, "subgraph": object()}, "R1": {"count": 1}},
        bonds=[
            {"labels": ["R1", "R0"], "count": 2, "nodes": ["random-a", "random-b"]},
            {"labels": ["R0", "R0"], "count": 1, "nodes": ["random-c", "random-d"]},
        ],
        molecular_weights=[46.069], mol_weights={"stable-key": [46.069]},
        distributions={"stable-key": "poisson(50)"},
    )
    payload = benchmark._fixed_payload(result)
    assert calls == [(3, "canonical"), (2, "canonical"), (2, "canonical")]
    assert payload == {
        "chains": ["CCO"], "sequences": [[["CO", "[H]O"]]],
        "sequence_unit_ids": result.sequence_unit_ids,
        "unit_counts": {"R0": 2, "R1": 1},
        "bonds": [
            {"labels": ["R0", "R0"], "count": 1},
            {"labels": ["R0", "R1"], "count": 2},
        ],
        "molecular_weights": [46.069], "mol_weights": {"stable-key": [46.069]},
        "distributions": {"stable-key": "poisson(50)"},
    }
    changed = copy.deepcopy(result)
    changed.chains = ["CCO"]
    changed.sequences = [[["CO", "[H]O"]]]
    changed.units = dict(reversed(list(changed.units.items())))
    changed.bonds.reverse()
    for bond in changed.bonds:
        bond["labels"].reverse()
        bond["nodes"] = ["fresh-parse-a", "fresh-parse-b"]
    assert benchmark._fingerprint(benchmark._fixed_payload(changed)) == benchmark._fingerprint(payload)
    assert result.bonds[0]["labels"] == ["R1", "R0"]
    for field, value in (
        ("sequence_unit_ids", [[["R1", "R0"]]]),
        ("molecular_weights", [47.0]), ("mol_weights", {"stable-key": [47.0]}),
        ("distributions", {"stable-key": "poisson(51)"}),
    ):
        altered = copy.deepcopy(result)
        setattr(altered, field, value)
        assert benchmark._fingerprint(benchmark._fixed_payload(altered)) != benchmark._fingerprint(payload)