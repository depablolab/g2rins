# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the deterministic benchmark reporting interface."""

import json

import pytest

from benchmarks import benchmark_sampling, compare_versions
from benchmarks.profile_sampling import _profile_record


def test_summary_only_omits_raw_runs(monkeypatch, capsys):
    record = {
        "case": "small-linear",
        "metadata": "none",
        "sampling_seconds": 1.0,
        "accepted": True,
    }
    monkeypatch.setattr(
        benchmark_sampling,
        "_run_isolated",
        lambda *_args, **_kwargs: dict(record),
    )

    assert (
        benchmark_sampling.main(
            [
                "--case",
                "small-linear",
                "--runs",
                "2",
                "--metadata",
                "none",
                "--summary-only",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert output["runs"] == 2
    assert output["results"] == [
        {
            "case": "small-linear",
            "metadata": "none",
            "median": record,
        }
    ]


def test_sampling_profile_reports_cpu_categories():
    profile = _profile_record(
        "small-linear",
        "none",
        "exact",
        max_discards=10,
        limit=3,
    )

    assert profile["atoms"] == 65
    assert len(profile["top_cumulative"]) == 3
    assert profile["self_time_by_category"]["networkx"] > 0
    assert sum(profile["self_time_fraction_by_category"].values()) == pytest.approx(1.0)


def test_version_comparison_reports_speed_memory_and_output_parity():
    def record(sampling_seconds, startup_seconds, peak_rss_mib, molecular_weight):
        return {
            "timing": {
                    "import_seconds": 0.5,
                    "parse_seconds": 0.25,
                    "graph_creator_seconds": 0.125,
                    "ensemble_creator_seconds": 0.125,
                "sampling_seconds": sampling_seconds,
                "startup_to_sample_seconds": startup_seconds,
            },
            "peak_rss_after_sample_mib": peak_rss_mib,
            "output": {
                "graph": {"atoms": 10, "bonds": 9},
                "molecular_weight": molecular_weight,
            },
        }

    summary = compare_versions._summarize(
        {
            "original": [record(4.0, 5.0, 200.0, 100.0)] * 3,
            "optimized": [record(1.0, 2.0, 150.0, 100.0 + 1e-10)] * 3,
        }
    )

    assert summary["validation"] == {
        "original_repeatable": True,
        "optimized_repeatable": True,
        "cross_version_equal": True,
    }
    assert summary["improvement"]["sampling_speedup"] == 4.0
    assert summary["improvement"]["sampling_reduction_percent"] == 75.0
    assert summary["improvement"]["peak_rss_reduction_mib"] == 50.0
    assert summary["improvement"]["peak_rss_reduction_percent"] == 25.0
