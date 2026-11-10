"""Exact centered motif presence must be independent of retained coverage."""

from g2rins.neighbor_motifs import CenteredMotifStatistics


def chain(length):
    return {
        "occurrences": [[i, "R0"] for i in range(length)],
        "edges": [[i, "2", i + 1, "1", "1", False] for i in range(length - 1)],
        "components": [{"root": 0, "occurrence_ids": list(range(length))}],
    }


def test_presence_is_once_per_chain_and_snapshots_are_detached():
    statistics = CenteredMotifStatistics()
    statistics.update(chain(6))
    statistics.update(chain(7))
    snapshot = statistics.finalize()
    repeated = next(row for row in snapshot["rows"]
                    if row["motif"] == "1.R0.2-1.R0.2-1.R0.2")
    assert repeated["count"] == 5
    assert repeated["chain_count"] == 2
    repeated["topology"]["units"].clear()
    assert statistics.finalize()["rows"] != snapshot["rows"]


def test_parallel_connections_keep_one_neighbor_and_self_links_keep_one_center():
    statistics = CenteredMotifStatistics()
    record = {
        "occurrences": [[0, "R0"], [1, "R1"]],
        "edges": [[0, "1", 1, "1", "1", False], [0, "2", 1, "2", "1", False]],
        "components": [{"root": 0, "occurrence_ids": [0, 1]}],
    }
    statistics.update(record)
    for row in statistics.finalize()["rows"]:
        assert len(row["topology"]["units"]) == 2
        assert len(row["topology"]["edges"]) == 2
        assert row["topology"]["boundary"] == []
    statistics = CenteredMotifStatistics()
    record["occurrences"] = [[0, "R0"]]
    record["edges"] = [[0, "1", 0, "2", "1", False]]
    record["components"][0]["occurrence_ids"] = [0]
    statistics.update(record)
    row = statistics.finalize()["rows"][0]
    assert row["topology"]["units"] == ["R0"]
    assert row["count"] == row["chain_count"] == 1