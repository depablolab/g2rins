"""Centered motifs must follow occurrence edges, never segment-list adjacency."""

import pytest

from g2rins.neighbor_motifs import CenteredMotifStatistics, next_neighbor_motif_membership


def export(records, weights=None):
    sites = {}
    for record in records:
        labels = dict(record["occurrences"])
        for unit in labels.values():
            sites.setdefault(unit, set())
        for left, ls, right, rs, _bond, _aromatic in record["edges"]:
            sites[labels[left]].add(int(ls))
            sites[labels[right]].add(int(rs))
    units = {unit: {"psmiles": "C" + "".join(f"([*:{site}])" for site in sorted(ports))}
             for unit, ports in sites.items()}
    population = CenteredMotifStatistics()
    for record, weight in zip(records, weights or [1] * len(records)):
        for _ in range(weight):
            population.update(record)
    return next_neighbor_motif_membership(
        records=records,
        weights=weights or [1] * len(records),
        retained_chain_count=len(records),
        accepted_count=sum(weights or [1] * len(records)),
        weight_semantics="accepted-coverage" if weights else "retained-observation",
        units=units,
        population_statistics=population.finalize(),
    )


def topology(units, edges):
    return {
        "schema": "bond-sequence-topology/v1",
        "occurrences": list(enumerate(units)),
        "edges": [[a, str(sa), b, str(sb), "1", False] for a, sa, b, sb in edges],
        "components": [{"root": 0, "occurrence_ids": list(range(len(units)))}],
    }


def test_linear_centered_motifs_include_sites_and_endpoints():
    record = topology(
        ["I0", "R0", "R0", "R0", "R1", "T1"],
        [(0, 1, 1, 1), (1, 2, 2, 1), (2, 2, 3, 1), (3, 2, 4, 1), (4, 3, 5, 1)],
    )
    result = export([record])
    motifs = {(row["center_unit"], row["motif"]): row["count"]
              for row in result["next_neighbor_motifs"]}
    assert motifs["I0", "null.I0.1-1.R0.2"] == 1
    assert motifs["R0", "null.I0.1-1.R0.2-1.R0.2"] == 1
    assert motifs["R0", "1.R0.2-1.R0.2-1.R0.2"] == 1
    assert motifs["R0", "1.R0.2-1.R0.2-1.R1.3"] == 1
    assert motifs["T1", "1.R1.3-1.T1.null"] == 1
    assert sum(motifs.values()) == 6
    assert result["motif_membership"] == result["next_neighbor_motifs"]


def test_branch_star_counts_occurrences_once_and_never_connects_arms():
    record = topology(
        ["I0", "R1", "R2", "R3", "T1"],
        [(0, 1, 1, 1), (1, 2, 2, 1), (1, 3, 3, 1), (3, 2, 4, 1)],
    )
    result = export([record], [7])
    rows = result["next_neighbor_motifs"]
    star = next(row for row in rows if row["center_unit"] == "R1")
    assert star["motif"] == "R1{1:1.I0.null;2:1.R2.null;3:1.R3.2}"
    assert star["count"] == 7
    assert "neighbors" not in star
    assert star["center_position"] is None
    assert star["psmiles"] and star["bigsmiles"]
    assert sum(row["count"] for row in rows) == 35
    assert not any("R2" in row["motif"] and "R3" in row["motif"]
                   for row in rows if row["center_unit"] != "R1")
    assert result["next_neighbor_motif_metadata"] == {
        "schema": "next-neighbor-motif-membership/v3",
    }
    assert star["chain_count"] == 7
    assert star["retained_chain_count"] == 1
    assert "{2,3}" in next(row["motif"] for row in rows if row["center_unit"] == "I0")


def test_motif_counts_ignore_occurrence_numbers_and_edge_order():
    record = topology(["I0", "R0", "T1"], [(0, 1, 1, 1), (1, 2, 2, 1)])
    reordered = {
        "schema": record["schema"],
        "occurrences": [[12, "T1"], [9, "R0"], [42, "I0"]],
        "edges": [[12, "1", 9, "2", "1", False], [9, "1", 42, "1", "1", False]],
        "components": [{"root": 42, "occurrence_ids": [9, 42, 12]}],
    }
    result = export([record, reordered])
    assert len(result["next_neighbor_motifs"]) == 3
    assert all(row["count"] == 2 for row in result["next_neighbor_motifs"])


def test_isolated_occurrences_are_motifs_too():
    result = export([topology(["I0"], [])])
    assert result["next_neighbor_motifs"][0]["motif"] == "null.I0.null"
    assert result["next_neighbor_motif_metadata"] == {
        "schema": "next-neighbor-motif-membership/v3",
    }


def test_missing_topology_is_not_silently_replaced_by_sequences():
    with pytest.raises(RuntimeError, match="topology"):
        next_neighbor_motif_membership(
            records=None, weights=[1], retained_chain_count=1,
            accepted_count=1, weight_semantics="retained-observation",
        )


def test_sites_sort_numerically_and_keep_neighbor_connections():
    record = topology(["R0", "R1", "T1"], [(0, 10, 1, 1), (0, 2, 2, 1)])
    row = next(row for row in export([record])["next_neighbor_motifs"]
               if row["center_unit"] == "R0")
    assert row["motif"] == "null.T1.1-2.R0.10-1.R1.null"
    assert "neighbors" not in row


def test_cycles_and_disconnected_components_do_not_require_path_expansion():
    record = topology(["R0"] * 3, [(0, 2, 1, 1), (1, 2, 2, 1), (2, 2, 0, 1)])
    record["occurrences"].append([3, "I0"])
    record["components"].append({"root": 3, "occurrence_ids": [3]})
    result = export([record])
    assert sum(row["count"] for row in result["next_neighbor_motifs"]) == 4
    assert next(row for row in result["next_neighbor_motifs"]
                if row["center_unit"] == "R0")["count"] == 3


def test_chain_incidence_counts_each_chain_once_not_each_occurrence():
    record = topology(["R0"] * 6, [(i, 2, i + 1, 1) for i in range(5)])
    rows = export([record, record])["next_neighbor_motifs"]
    repeated = next(row for row in rows if row["motif"] == "1.R0.2-1.R0.2-1.R0.2")
    assert repeated["count"] == 4
    assert repeated["chain_count"] == 2
    assert repeated["retained_chain_count"] == 2
    assert repeated["chain_fraction"] == 1.0


def test_representative_weights_cannot_be_used_as_exact_chain_incidence():
    record = topology(["I0"], [])
    with pytest.raises(RuntimeError, match="all-accepted centered motif statistics"):
        next_neighbor_motif_membership(
            records=[record], weights=[7], retained_chain_count=1,
            accepted_count=7, weight_semantics="accepted-coverage", units={"I0": {"psmiles": "C"}},
        )
