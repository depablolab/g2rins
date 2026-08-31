# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the deterministic benchmark reporting interface."""

import json

from benchmarks import benchmark_sampling
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
    assert sum(profile["self_time_fraction_by_category"].values()) == 1.0
