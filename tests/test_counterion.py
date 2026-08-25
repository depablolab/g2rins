# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Dedicated tests for trailing counterions.

A counterion trails its repeat unit or end group as a `.`-fragment
(`[<]CC(C[NH3+])[>].[Cl-]`) and is wired to opposite-charge sites of the
unit by greedy charge matching in written order. The wire is an
*association edge* (bond_type 0): it travels with the unit's subgraph but
is no covalent bond, so the ion renders as a separate `.` fragment in the
generated SMILES and adds no valence.
"""

import warnings

import numpy as np
import pytest

import g2rins
from g2rins.exception import (
    CounterionChargeImbalance,
    SmilesHasNonZeroBondConnectors,
    UnmatchedCounterion,
)
from g2rins.generative_graph import _BOND_TYPE_NAME


def _association_pairs(generative_graph):
    """Unique undirected association edges (static edges carry both directions)."""
    return {frozenset((u, v)) for u, v, d in generative_graph.edges(data=True) if d[_BOND_TYPE_NAME] == 0}


def _association_degrees(generative_graph, atomic_num):
    """Sorted association-edge counts of every atom of the given element."""
    pairs = _association_pairs(generative_graph)
    return sorted(
        sum(1 for pair in pairs if node in pair)
        for node, data in generative_graph.nodes(data=True)
        if data["atomic_num"] == atomic_num
    )


MONOVALENT = "{[] [<]CC(C[NH3+])[>].[Cl-]; C[>]; [<][H] []}|poisson(500.0)|"
CA_BRIDGE = "{[] [<]CC([>])(C(=O)[O-])C(=O)[O-].[Ca+2]; C[>]; [<][H] []}|poisson(500.0)|"
TWO_NA = "{[] [<]CC([>])(C(=O)[O-])C(=O)[O-].[Na+].[Na+]; C[>]; [<][H] []}|poisson(500.0)|"
NEUTRAL_GUEST = "{[] [<]CCO[>].O; C[>]; [<][H] []}|poisson(500.0)|"
TRANSPORTATION = "{[] [<]CC([>])(C[N+]1(C)CCCCC1)C(C[N+]1(C)CCCCC1)C[N+]1(C)CCCCC1.[P-3]; C[>]; [<][H] []}|poisson(800.0)|"
IMBALANCE = "{[] [<]CC(C[NH3+])[>].[Cl-].[Cl-]; C[>]; [<][H] []}|poisson(500.0)|"
END_GROUP_ION = "{[] [<]CC[>]; C[NH3+][>].[Cl-]; [<][H] []}|poisson(400.0)|"
ACETATE = "{[] [<]CC(C[NH3+])[>].CC(=O)[O-]; C[>]; [<][H] []}|poisson(500.0)|"
MIXTURE = MONOVALENT + ".|5000.0|" + "{[] [<]CCO[>]; C[>]; [<][H] []}|poisson(500.0)|.|4000.0|"


@pytest.mark.parametrize(
    "smi",
    (MONOVALENT, CA_BRIDGE, TWO_NA, NEUTRAL_GUEST, TRANSPORTATION, IMBALANCE, END_GROUP_ION, ACETATE, MIXTURE),
)
def test_counterion_roundtrip(smi):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert str(g2rins.G2rins.make(smi)) == smi


@pytest.mark.parametrize(
    ("smi", "n_pairs", "ion_atomic_num", "ion_degrees"),
    [
        # One Cl- per repeat unit, one edge to the NH3+ site.
        pytest.param(MONOVALENT, 1, 17, [1], id="monovalent"),
        # ONE Ca node bridging both carboxylates: two edges on the same ion.
        pytest.param(CA_BRIDGE, 2, 20, [2], id="divalent-bridge"),
        # TWO Na nodes, one edge each: the graph distinguishes bridged from unbridged.
        pytest.param(TWO_NA, 2, 11, [1, 1], id="two-monovalent"),
        # One P-3 split greedily over the three N+ sites in written order.
        pytest.param(TRANSPORTATION, 3, 15, [3], id="charge-transportation"),
        # The ion trails an end group (initiator), not a repeat unit.
        pytest.param(END_GROUP_ION, 1, 17, [1], id="end-group-ion"),
    ],
)
def test_association_edge_topology(smi, n_pairs, ion_atomic_num, ion_degrees):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generative_graph = g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph()
    assert len(_association_pairs(generative_graph)) == n_pairs
    assert _association_degrees(generative_graph, ion_atomic_num) == ion_degrees


def test_neutral_guest_attaches_silently():
    """A neutral fragment (e.g. hydration water) takes the first-atom fallback
    without any warning: a legitimate guest molecule, not a matching failure."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        generative_graph = g2rins.G2rins.make(NEUTRAL_GUEST).get_graph_creator().get_generative_graph()
    assert len(_association_pairs(generative_graph)) == 1
    assert not any(issubclass(w.category, (UnmatchedCounterion, CounterionChargeImbalance)) for w in caught)


def test_multi_atom_ion_anchors_on_charged_atom():
    """The ion-side endpoint is the ion's first CHARGED atom (acetate's O-),
    not its first written atom."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generative_graph = g2rins.G2rins.make(ACETATE).get_graph_creator().get_generative_graph()
    pairs = _association_pairs(generative_graph)
    assert len(pairs) == 1
    (pair,) = pairs
    charges = sorted(generative_graph.nodes[node]["charge"] for node in pair)
    assert charges == [-1, 1]


def test_imbalance_warns_and_still_generates():
    """A surplus ion falls back to the first atom (UnmatchedCounterion) and the
    unit's non-zero net charge is reported (CounterionChargeImbalance); neither
    is fatal."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = g2rins.G2rins.make(IMBALANCE)
        generative_graph = obj.get_graph_creator().get_generative_graph()
    assert any(issubclass(w.category, UnmatchedCounterion) for w in caught)
    assert any(issubclass(w.category, CounterionChargeImbalance) for w in caught)
    assert len(_association_pairs(generative_graph)) == 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mol_graph = obj.get_graph_creator().get_ensemble_creator().sample_mol_graph(rng=np.random.default_rng(0))
    assert mol_graph.number_of_nodes() > 0


def test_balanced_units_do_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for smi in (MONOVALENT, CA_BRIDGE, TWO_NA, TRANSPORTATION, END_GROUP_ION, ACETATE):
            g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph()
    assert not any(issubclass(w.category, (UnmatchedCounterion, CounterionChargeImbalance)) for w in caught)


def test_counterion_with_bond_connector_raises():
    import lark

    smi = "{[] [<]CC(C[NH3+])[>].C([<])O; C[>]; [<][H] []}|poisson(500.0)|"
    with pytest.raises(SmilesHasNonZeroBondConnectors):
        try:
            g2rins.G2rins.make(smi)
        except lark.exceptions.VisitError as exc:
            raise exc.__context__  # trunk-ignore(ruff/B904)


def test_generated_smiles_carries_one_ion_per_unit():
    """Every realized repeat unit brings its counterion along: the generated
    SMILES has exactly one [Cl-] fragment per [NH3+] site, none of them bonded."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(MONOVALENT).get_graph_creator().get_ensemble_creator()
        smiles = g2rins.mol_graph_to_smiles(ensemble_creator.sample_mol_graph(rng=np.random.default_rng(0)))
    fragments = smiles.split(".")
    n_sites = smiles.count("[NH3+]")
    assert n_sites > 0
    assert fragments.count("[Cl-]") == n_sites


def test_end_group_ion_appears_once_per_chain():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(END_GROUP_ION).get_graph_creator().get_ensemble_creator()
        smiles = g2rins.mol_graph_to_smiles(ensemble_creator.sample_mol_graph(rng=np.random.default_rng(0)))
    assert smiles.split(".").count("[Cl-]") == 1


def test_ionic_unit_through_derived_label_pipeline():
    """Counterions integrate with the derived unit/bond labels: the ion shares
    its unit's unit_id, carries no bond_id (no non-static incidence), shows up
    as a `.` fragment inside the unit's P-SMILES and its g2rins text, and the
    association edge never surfaces as a bond record."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(MONOVALENT).get_graph_creator().get_ensemble_creator()
        data = ensemble_creator.create_ensemble(3, output_format="smiles", ensemble_info=True)

    repeat_unit = data.units["R0"]
    assert repeat_unit["g2rins"] == "[<]CC(C[NH3+])[>].[Cl-]"
    psmiles_fragments = repeat_unit["psmiles"].split(".")
    assert len(psmiles_fragments) == 2 and "[Cl-]" in psmiles_fragments
    assert "[*:1]" in repeat_unit["psmiles"] and "[*:2]" in repeat_unit["psmiles"]

    labels = g2rins.derive_unit_labels(ensemble_creator._generative_graph)
    chloride_nodes = [n for n, d in ensemble_creator._generative_graph.nodes(data=True) if d["atomic_num"] == 17]
    assert chloride_nodes
    for node in chloride_nodes:
        assert labels.unit_id[node] == "R0"
        assert node not in labels.bond_id

    endpoints = {endpoint for record in data.bonds for endpoint in record["labels"]}
    assert endpoints == {"I0.1", "R0.1", "R0.2", "T0.1"}
    # The counterion rides inside its unit's static subgraph, association
    # edge included.
    assert any(data["atomic_num"] == 17 for _node, data in repeat_unit["subgraph"].nodes(data=True))
