# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Regression tests for the stochastic-generation fixes (commit 7295cec).

Every string here reproduced a concrete sampler bug before that commit. Each
case asserts the failure mode stays gone: generation completes, no phantom
(atomic_num 0) nodes leak into the returned graph, the molecule sanitizes in
RDKit (over-bonded atoms raise AtomValenceException there), and the total
molecular weight stays inside a generous band around the target — catching
both runaway/over-counted growth and collapsed/truncated chains.
"""

import warnings

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

import g2rins
import g2rins.util
from g2rins.exception import (
    DiscardedSamplingPaths,
    PossibleNonRepresentativePolymerChain,
)

SEEDS = (0, 1, 2)


def _reset_rngs(seed):
    """Reset the two RNG streams the sampler consumes: numpy's legacy global
    (truncated/nested draws) and the library global Generator (top-level
    draws; plain get_global_rng(seed) does not reset an existing generator)."""
    np.random.seed(seed)
    g2rins.util._GLOBAL_RNG = np.random.default_rng(seed)


# (id, G2RINS, target MW of the outermost stochastic object,
#  accepted MW band as (lower, upper) ratios of the target)
REGRESSION_CASES = [
    pytest.param(
        # Cross-bucket termination sweep + re-added consumed bonds merged a
        # second terminator onto capped junction atoms: every molecule failed
        # RDKit sanitization with AtomValenceException (trivalent O).
        "{[] [<]NN{[>] [<]CCO[>]; ; [<]F [<]}|poisson(100)|[>], [<1]{[>] [<]CCO[>]; ; [<]F [<]}|poisson(100)|[>]; C(O[>1])C(O[>1])CO[>1]; [<][H] []}|poisson(2000)|",
        2000.0,
        (0.4, 2.0),
        id="multiarm-nested-with-terminators",
    ),
    pytest.param(
        # Two different nested stochastic objects per monomer share one parent
        # instance; the termination cascade re-terminated it and crashed with
        # an uncaught RuntimeError in 39/50 seeds.
        "{[] [<]CC({[<] [<]NN[>];; [>]}|poisson(80)|[H])C({[<] [<]C(C)O[>];; [>]}|poisson(80)|[H])C[>]; [<][H]; [>][H] []}|poisson(400)|",
        400.0,
        (0.4, 2.0),
        id="two-nested-sos-per-monomer",
    ),
    pytest.param(
        # Same multi-arm topology without inner terminators; guards the
        # transition bucket rewrite (dropped arms / one-arm growth).
        "{[] [<]NN{[>] [<]CCO[>];; [<]}|poisson(100)|[>], [<1]{[>] [<]CCO[>];; [<]}|poisson(100)|[>]; C(O[>1])C(O[>1])CO[>1]; [<][H] []}|poisson(2000)|",
        2000.0,
        (0.4, 2.0),
        id="multiarm-nested-no-terminators",
    ),
    pytest.param(
        # Grandparent MW accounting: single-level parent_map entries starved
        # ancestor tallies, producing molecules at ~3x the outer target.
        "{[] [<]CC({[<] [<]NN({[<] [<]C(C)O[>];; [>]}|poisson(80)|[H])[>];; [>]}|poisson(200)|[H])CC[>]; [<][H]; []}|poisson(800)|",
        800.0,
        (0.4, 2.0),
        id="three-level-nested-mw",
    ),
    pytest.param(
        # Plain linear chain: guards that the fixes leave simple generation
        # untouched.
        "C{[>][<]CC(C)[>];;[<]}|poisson(900)|[H]",
        900.0,
        (0.4, 2.0),
        id="linear-poisson",
    ),
    pytest.param(
        # [$]-connector repeat unit with genuine self-loop stochastic edges.
        "N#CC(C)(C){[$][$]CC(C(=O)OC)[$];;[$]}|poisson(1000)|[H]",
        1000.0,
        (0.4, 2.0),
        id="dollar-connector",
    ),
    pytest.param(
        # Real workload: PEG/PPO random copolymer. The log-normal tail is
        # fat, so the band is wider than for the poisson cases.
        "{[] [<|.8|]CCO[>|.8|], [<|.2|]CC(C)O[>|.2|]; [>][H] ; [<]Br []}|log_normal(1400, 1.15)|",
        1400.0,
        (0.3, 3.0),
        id="peg-ppo-lognormal",
    ),
    pytest.param(
        # Real workload: grafted gauss-in-gauss bottlebrush.
        "{[] [<1]CC(C(=O)O)[>1]; {[] [<0]C(C)(C(=O)OCCOC(=O)C(C)(C)[>1])C[>0]; [H][>0]; [<0]Br [<1]}|gauss(4000.0, 500.0)|[>1]; [<1]Br []}|gauss(8000.0, 1000.0)|",
        8000.0,
        (0.3, 2.5),
        id="grafted-gauss-in-gauss",
    ),
]


@pytest.mark.parametrize(("smi", "target_mw", "band"), REGRESSION_CASES)
def test_generation_regression(smi, target_mw, band):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    lower, upper = band
    for seed in SEEDS:
        _reset_rngs(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mol_graph = ensemble_creator.sample_mol_graph()

        phantom_nodes = [n for n, data in mol_graph.nodes(data=True) if data.get("atomic_num") == 0]
        assert not phantom_nodes, f"seed {seed}: phantom placeholder atoms left in the sampled graph"

        mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
        Chem.SanitizeMol(mol)

        mol_weight = Descriptors.MolWt(mol)
        assert lower * target_mw <= mol_weight <= upper * target_mw, f"seed {seed}: MW {mol_weight:.0f} outside [{lower:g}, {upper:g}] x target {target_mw:g}"


# Monofunctional inner stochastic object (right terminal "[]"): the generating
# graph splits in two, chains seed inside the graft and dead-end below the
# outer target, so every sample is a truncated chain.
TRUNCATING_SMI = (
    "{[] [<|9.0|]CC(C)O[>|9.0|], [<|6.0|]CC(CC)O[>|6.0|]; {[] [<|7.0|]CCO[>|7.0|], [<|4.0|]CC(CC)O[>|4.0|]; CCCCO[>]; [<] []}|gauss(680.0, 215.0)|[>]; [<][H] []}|gauss(1649.0, 521.5)|"
)


def test_truncated_chain_contract():
    """Truncated chains must be finalized like completed ones: warned about,
    phantom-free, returned in the documented shape, and discarded explicitly
    by create_ensemble (which used to rely on an accidental unpack error and
    crashed on 6-atom truncated chains)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(TRUNCATING_SMI).get_graph_creator().get_ensemble_creator()

    _reset_rngs(0)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = ensemble_creator.sample_mol_graph(molecule_info=True)
    assert any(issubclass(w.category, PossibleNonRepresentativePolymerChain) for w in caught_warnings)
    assert isinstance(result, tuple) and len(result) == 6
    mol_graph = result[0]
    assert all(data.get("atomic_num") != 0 for _, data in mol_graph.nodes(data=True))
    Chem.SanitizeMol(g2rins.mol_graph_to_rdkit_mol(mol_graph))

    _reset_rngs(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble = ensemble_creator.create_ensemble(n_samples=1, output_format="mol", max_number_of_discarded_chains=3)
    assert ensemble is None


def test_unit_id_deterministic():
    """unit_id numbering must not depend on the (random UUID) node ids: parsing
    the same string repeatedly must label the same atoms with the same units
    (numbering used to swap R0/R1 between parses)."""
    smi = "{[]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H]; [<][H] []}|gauss(1500, 50)|"
    signatures = []
    for _parse in range(4):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            generative_graph = g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph(include_bond_connectors=False)
        unit_labels = g2rins.derive_unit_labels(generative_graph).unit_id
        per_unit = {}
        for node, data in generative_graph.nodes(data=True):
            per_unit.setdefault(unit_labels[node], []).append(data["atomic_num"])
        signatures.append(tuple(sorted((unit_id, tuple(sorted(nums))) for unit_id, nums in per_unit.items())))
    assert all(signature == signatures[0] for signature in signatures)


def test_undershoot_snapshot_contract():
    """termination_flag=1 chains are guaranteed undershoot exactly when no
    UndershootSnapshotMissed warning fires; for a plain linear chain the
    adaptive lookahead must never miss."""
    from g2rins.exception import UndershootSnapshotMissed

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make("C{[>][<]CC(C)[>];;[<]}|poisson(900)|[H]").get_graph_creator().get_ensemble_creator()
    for seed in SEEDS:
        _reset_rngs(seed)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            ensemble_creator.sample_mol_graph(termination_flag=1)
        assert not any(issubclass(w.category, UndershootSnapshotMissed) for w in caught_warnings)


# ---------------------------------------------------------------------------
# Worked examples: architecture + molecular-weight-distribution checks.
#
# These parse a range of real polymer architectures (random copolymer, a
# three-level nested graft, a hyperbranched amine, a three-arm star) and assert
# that the sampled ensemble reproduces the distribution named in the string.
# They cover the nested-MW-accounting fix (the graft would otherwise sample far
# above target) and the SchulzZimm dispersity fix (a shape-parameter off-by-one
# that made every Schulz-Zimm polymer too narrow).
# ---------------------------------------------------------------------------

# (id, distribution string, target Mn, target dispersity, Mn rel-tol, D abs-tol)
DISTRIBUTION_MOMENT_CASES = [
    pytest.param("log_normal(1400.0, 1.15)", 1400.0, 1.15, 0.04, 0.04, id="log_normal"),
    pytest.param("schulz_zimm(1800.0, 1200.0)", 1200.0, 1.50, 0.05, 0.04, id="schulz_zimm-D1.5"),
    pytest.param("schulz_zimm(2400.0, 1200.0)", 1200.0, 2.00, 0.05, 0.05, id="schulz_zimm-D2.0"),
    pytest.param("poisson(2000.0)", 2000.0, 1.00, 0.02, 0.02, id="poisson"),
]


@pytest.mark.parametrize(("dist_string", "target_mn", "target_d", "mn_rtol", "d_atol"), DISTRIBUTION_MOMENT_CASES)
def test_distribution_moments(dist_string, target_mn, target_d, mn_rtol, d_atol):
    """The raw draws of each distribution must reproduce the number-average and
    dispersity named in the string. This guards the distribution math directly
    (independent of any topology) — notably the Schulz-Zimm shape parameter,
    which if wrong keeps Mn correct but collapses the dispersity toward
    (z+2)/(z+1) instead of the requested (z+1)/z."""
    distribution = g2rins.StochasticDistribution.make(dist_string)
    samples = np.array([distribution.draw_mw(np.random.default_rng(i)) for i in range(20000)])
    mn = samples.mean()
    mw = (samples * samples).sum() / samples.sum()
    dispersity = mw / mn
    assert abs(mn - target_mn) / target_mn < mn_rtol, f"Mn {mn:.0f} vs target {target_mn:.0f}"
    assert abs(dispersity - target_d) < d_atol, f"D {dispersity:.3f} vs target {target_d:.2f}"


# (id, G2RINS, target Mn, Mn rel-tol, n_samples, min dispersity or None)
EXAMPLE_ENSEMBLE_CASES = [
    pytest.param(
        # Methanol-initiated PEG/PPO random copolymer.
        "{[] [<|0.8|]CCO[>|0.8|], [<|0.2|]CC(C)O[>|0.2|]; CO[>]; [<][H] []}|log_normal(1400.0, 1.15)|",
        1400.0,
        0.12,
        250,
        None,
        id="peg-ppo-copolymer",
    ),
    pytest.param(
        # Three-level nested graft (tBMA backbone / HEMA-PEG side chains),
        # Br-terminated. Before the nested-MW-accounting fix this sampled ~3x
        # the outer target; the whole molecule must land near 10000, not 30000.
        "{[] [<]CC([>])C(=O)OC(C)(C)C; {[] [<]CC([>])(C)C(=O)OCCO; {[] [<]CCO[>]; CO[>];  [<]}|poisson(3000.0)|CCOC(=O)C(C)(C)[>];  [<]}|poisson(6000.0)|[>]; [<]Br []}|poisson(10000.0)|",
        10000.0,
        0.10,
        25,
        None,
        id="nested-graft",
    ),
    pytest.param(
        # Hyperbranched poly(ethyleneimine): the AB2 monomer [<]CCN([>])[>]
        # branches at every nitrogen.
        "{[] [<]CCN([>])[>]; [<][H]; O[>], [<][H] []}|poisson(2000.0)|",
        2000.0,
        0.10,
        10,
        None,
        id="hyperbranched",
    ),
    pytest.param(
        # Glycerol-cored three-arm star PLGA (lactic/glycolic random copolymer).
        # The whole star shares one Schulz-Zimm budget, so its dispersity tracks
        # the distribution (~1.5); the min-D guard catches a Schulz-Zimm
        # regression (which would pull it toward 1.33).
        "{[] [<]C(C)C(=O)O[>], [<]CC(=O)O[>]; [>]OCC(O[>])CO[>]; [<][H] []}|schulz_zimm(1800.0, 1200.0)|",
        1200.0,
        0.12,
        400,
        1.40,
        id="star-plga",
    ),
]


@pytest.mark.parametrize(("smi", "target_mn", "mn_rtol", "n_samples", "min_dispersity"), EXAMPLE_ENSEMBLE_CASES)
def test_example_ensemble(smi, target_mn, mn_rtol, n_samples, min_dispersity):
    """Each architecture generates a full ensemble whose number-average lands
    near the string's target, and (where requested) whose dispersity clears a
    lower bound. Sampling is seeded for reproducibility."""
    _reset_rngs(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
        molecules = ensemble_creator.create_ensemble(n_samples=n_samples, output_format="mol")
    assert molecules is not None and len(molecules) == n_samples
    weights = np.array([Descriptors.MolWt(m) for m in molecules])
    mn = weights.mean()
    assert abs(mn - target_mn) / target_mn < mn_rtol, f"Mn {mn:.0f} vs target {target_mn:.0f}"
    if min_dispersity is not None:
        dispersity = (weights * weights).sum() / weights.sum() / mn
        assert dispersity > min_dispersity, f"D {dispersity:.3f} below {min_dispersity}"


def test_cascade_terminated_parent_gets_declared_cap():
    """A parent stochastic object that terminates through the cascade (the
    normal path for graft backbones, whose MW crossing is detected while a
    nested side chain is active) must still fire its declared terminator:
    every grafting-through bottlebrush chain carries exactly one backbone Br
    cap. Before the cascade fix this was 0 Br on every chain (the cascade was
    bookkeeping-only and the chain ended in an implicit H)."""
    smi = "{[] [<]CC([>])C(=O)O{[>] [<]CCO[>]; ; [<]C []}|poisson(400)|; COC(=O)C(C)[>]; [<]Br []}|poisson(9000)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    for seed in SEEDS:
        _reset_rngs(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mol_graph = ensemble_creator.sample_mol_graph()
        mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
        Chem.SanitizeMol(mol)
        bromine_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "Br")
        assert bromine_count == 1, f"seed {seed}: expected 1 backbone Br cap, found {bromine_count}"


@pytest.mark.parametrize(
    ("smi", "label"),
    [
        ("{[] [<]CC([>])c1cc[nH]c1; [<][H]; [>][H] []}|poisson(1500)|", "pyrrole"),
        ("{[] [<]CC([>])Cc1c[nH]c2ccccc12; [<][H]; [>][H] []}|poisson(2000)|", "indole"),
        ("{[] [<]CC([>])c1c[nH]cn1; [<][H]; [>][H] []}|poisson(1500)|", "imidazole"),
    ],
)
def test_nh_heteroaromatic_kekulizes(smi, label):
    """Pyrrole-type aromatic nitrogen ([nH]) must keep its explicit hydrogen
    through graph construction so the ring kekulizes; without it the mol graph
    reads a pyridine-type N and the sampled molecule raises KekulizeException."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    for seed in SEEDS:
        _reset_rngs(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mol_graph = ensemble_creator.sample_mol_graph()
        mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)  # kekulizes internally
        Chem.SanitizeMol(mol)
        # every repeat unit contributes one aromatic N that carries its H
        nh_nitrogens = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0)
        assert nh_nitrogens > 0, f"{label} seed {seed}: aromatic N-H hydrogen was lost"


@pytest.mark.parametrize(
    ("smi", "label"),
    [
        ("{[] [<][CH]([>])[>]; [<][H]; [>][H] []}|poisson(400)|", "CH-hyperbranch"),
        ("{[] [<][NH]([>])[>]; [<][H]; [>][H] []}|poisson(600)|", "NH-AB2"),
        ("{[] [<]C[NH2][>]; [<][H]; [>][H] []}|poisson(400)|", "NH2-backbone"),
    ],
)
def test_nonaromatic_bracket_h_infers_hydrogens(smi, label):
    """A NON-aromatic bracket atom that writes an H token (e.g. [CH], [NH],
    [NH2]) must keep valence-based implicit-H completion, because the written
    count is only valid for one coordination and the sampler realizes a
    different one at branch points, termini and unfired bond connectors.
    Forcing the written count made under-coordinated atoms silent radicals and
    over-coordinated atoms crash; only aromatic N-H is exempt (handled by
    test_nh_heteroaromatic_kekulizes)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    for seed in SEEDS:
        _reset_rngs(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mol_graph = ensemble_creator.sample_mol_graph()
        mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)  # must not raise AtomValenceException
        Chem.SanitizeMol(mol)
        assert Descriptors.NumRadicalElectrons(mol) == 0, f"{label} seed {seed}: forced H count left radicals"


def test_bare_bracket_atom_keeps_inferred_hydrogens():
    """A bracket atom that writes NO hydrogen token (e.g. the carbanion [C-])
    must keep RDKit's valence-based implicit-H completion, not be force-locked to
    zero H. This guards the scope of the num_explicit_h fix: only atoms that
    actually write an H token get SetNoImplicit; over-broadening it silently
    changed the formula/mass of such polymers."""
    smi = "{[] [$][C-][$]; [$][H]; [$][H] []}|poisson(400)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    _reset_rngs(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mol_graph = ensemble_creator.sample_mol_graph()
    mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
    Chem.SanitizeMol(mol)
    # the internal carbanion carbons have 2 backbone bonds; with charge -1 (valence
    # 3) each carries one inferred H, so the chain is not a bare-carbon skeleton.
    carbanions_with_h = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C" and atom.GetFormalCharge() == -1 and atom.GetTotalNumHs() > 0)
    assert carbanions_with_h > 0, "bare bracket [C-] was force-locked to zero H"


def test_nh_polymer_tracked_mw_matches_rdkit():
    """The sampler's termination MW accounting must agree with the RDKit molecule
    for [nH] units: add_molw now reads num_explicit_h, so a pyrrole chain's
    tracked weight equals its RDKit MolWt instead of undercounting the N-H."""
    smi = "{[] [<]CC([>])c1cc[nH]c1; [<][H]; [>][H] []}|poisson(1500)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    _reset_rngs(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mol_graph, _units, _bonds, _seq, tracked, _dist = ensemble_creator.sample_mol_graph(molecule_info=True)
    rdkit_mw = Descriptors.MolWt(g2rins.mol_graph_to_rdkit_mol(mol_graph))
    tracked_mw = sum(sum(v) for v in tracked.values())
    assert abs(rdkit_mw - tracked_mw) < 1.0, f"tracked {tracked_mw:.1f} vs RDKit {rdkit_mw:.1f}"


def test_create_ensemble_aromatic_core_unit():
    """create_ensemble must not crash on a polymer whose repeat/core unit
    carries an aromatic heterocycle. A unit is a static-connected fragment, so
    a ring atom bearing an inter-unit bond has a dangling valence and cannot be
    kekulized in isolation; the fragment conversion must handle that (and must
    not run at all unless ensemble information is requested)."""
    # 4-arm star: pentaerythritol-tetratriazole core + four polystyrene arms.
    star = (
        "C(COC(=O)CCc1nnn(c1){[>] [<]C(c1ccccc1)C[>]; ; [<]C(=O)OCC []}|gauss(1400,240)|)"
        "(COC(=O)CCc1nnn(c1){[>] [<]C(c1ccccc1)C[>]; ; [<]C(=O)OCC []}|gauss(1400,240)|)"
        "(COC(=O)CCc1nnn(c1){[>] [<]C(c1ccccc1)C[>]; ; [<]C(=O)OCC []}|gauss(1400,240)|)"
        "(COC(=O)CCc1nnn(c1){[>] [<]C(c1ccccc1)C[>]; ; [<]C(=O)OCC []}|gauss(1400,240)|)"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(star).get_graph_creator().get_ensemble_creator()
        for output_format in ("mol_graph", "mol", "smiles"):
            for with_info in (False, True):
                _reset_rngs(0)
                result = ensemble_creator.create_ensemble(n_samples=2, output_format=output_format, ensemble_info=with_info)
                molecules = result.chains if with_info else result
                assert molecules is not None and len(molecules) == 2
                if with_info:
                    # units are returned and the aromatic triazole survives the
                    # kekulize-free fragment conversion.
                    assert len(result.units) > 0


def test_star_initiator_has_three_arms():
    """The glycerol core of the star must expose three arm-attachment atoms in
    the generative graph — each of the three [>] oxygens propagates into the
    stochastic object (to either comonomer, since the arm is a random
    copolymer)."""
    smi = "{[] [<]C(C)C(=O)O[>], [<]CC(=O)O[>]; [>]OCC(O[>])CO[>]; [<][H] []}|schulz_zimm(1800.0, 1200.0)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generative_graph = g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    unit_labels = g2rins.derive_unit_labels(generative_graph).unit_id
    initiator_nodes = {n for n in generative_graph.nodes() if unit_labels[n].startswith("I")}
    arm_source_atoms = {u for u, _v, data in generative_graph.out_edges(initiator_nodes, data=True) if data.get("transition_weight", 0) > 0}
    assert len(arm_source_atoms) == 3, f"expected 3 arms, found {len(arm_source_atoms)}"


# ---------------------------------------------------------------------------
# Regression tests for the 2026-07 dev-clean code-review fixes.
# ---------------------------------------------------------------------------


def test_zero_molar_amount_raises_clear_error():
    """An all-zero sampling decision (every alternative declared '|0|') is
    fixed by the string, so it must raise AllZeroSamplingWeights. It used to
    surface as a NaN-probability ValueError from rng.choice, which
    create_ensemble masked as max_number_of_discarded_chains 'discarded
    chains' before returning None."""
    from g2rins.exception import AllZeroSamplingWeights

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make("C{[>][<]CC[>]|0|;;[<]}|poisson(900)|[H]").get_graph_creator().get_ensemble_creator()
    for call in (lambda: ensemble_creator.sample_mol_graph(), lambda: ensemble_creator.create_ensemble(n_samples=2, max_number_of_discarded_chains=4)):
        _reset_rngs(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(AllZeroSamplingWeights):
                call()


def test_conditional_zero_molar_path_is_a_counted_discard():
    """An optional nested path with no valid unit must reject only that chain.

    Returning the requested ensemble conditions it on avoiding the dead path,
    so the rejection must also be surfaced in a summary warning.
    """
    smi = "{[] [<]CC[>], " "[<|0.05|]CC({[<] [<]NN[>]|0|;; [>]}|poisson(80)|[H])" "[>|0.05|]; [<][H]; [>][H] []}|poisson(600)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    _reset_rngs(0)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        molecules = ensemble_creator.create_ensemble(
            n_samples=1,
            output_format="mol",
            max_number_of_discarded_chains=10,
        )

    assert molecules is not None and len(molecules) == 1
    summaries = [warning.message for warning in caught_warnings if isinstance(warning.message, DiscardedSamplingPaths)]
    assert summaries and summaries[0].discarded_count > 0
    assert any("DeadSamplingPath" in reason for reason, _count in summaries[0].reasons)


def test_undershoot_rollback_preserves_conditional_provenance(monkeypatch):
    """Adopting an undershoot checkpoint must not make provenance regress.

    The checkpoint is captured before the crossing growth step marks the path
    conditional.  Once that step is rejected, a later chain-local dead end
    must still be wrapped as DeadSamplingPath rather than aborting the whole
    ensemble as a fatal AllZeroSamplingWeights error.
    """
    from g2rins.ensemble_creator import _PartialAtomGraph
    from g2rins.exception import AllZeroSamplingWeights, DeadSamplingPath

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make("C{[>][<]CC[>];;[<]}|uniform(40,40)|[H]").get_graph_creator().get_ensemble_creator()

    source = next(node for node, data in ensemble_creator.generative_graph.nodes(data=True) if data["stochastic_id_tree"][0] == 0 and data["gen_weight"] > 0)
    original_terminate = _PartialAtomGraph.terminate_graph

    def inject_post_rollback_dead_end(self, sto_atom_id, rng):
        gen_id = self.stochastic_tracker._stochastic_atom_id_to_gen_id[sto_atom_id]
        if gen_id == 0:
            self.stochastic_tracker.normalized_probabilities(
                [0.0],
                "post-rollback termination",
            )
        return original_terminate(self, sto_atom_id, rng)

    monkeypatch.setattr(
        _PartialAtomGraph,
        "terminate_graph",
        inject_post_rollback_dead_end,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(DeadSamplingPath) as caught:
            ensemble_creator.sample_mol_graph(
                source=source,
                use_repeat_units_as_source=True,
                rng=np.random.default_rng(0),
                termination_flag=1,
            )

    assert isinstance(caught.value.__cause__, AllZeroSamplingWeights)


def test_unavoidable_all_zero_route_fails_fast_through_tracker(monkeypatch):
    """A real unavoidable all-zero route must fail on its first attempt.

    This deliberately exercises tracker normalization.  The older stub test
    injected AllZeroSamplingWeights above the tracker and therefore could not
    detect its conversion into a retryable DeadSamplingPath.
    """
    from g2rins.exception import AllZeroSamplingWeights

    # Both genuine 50/50 initiators necessarily reach the sole repeat unit,
    # whose declared molar amount is zero.  The initial source branch therefore
    # cannot make the later structural error avoidable.
    smi = "{[] [<]CC[>]|0|; C[>], N[>]; [<][H] []}|poisson(100)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    original_sample = ensemble_creator.sample_mol_graph
    calls = 0

    def counted_sample(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", counted_sample)
    _reset_rngs(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(
            AllZeroSamplingWeights,
            match="transition target selection",
        ):
            ensemble_creator.create_ensemble(
                n_samples=1,
                max_number_of_discarded_chains=3,
            )

    assert calls == 1


def test_unavoidable_nested_all_zero_after_branch_fails_fast(monkeypatch):
    """Branches do not make a dead end retryable when every route is dead.

    Each positive outer repeat-unit choice below necessarily instantiates a
    nested stochastic object whose only entry has zero molar amount.  Retrying
    the outer choice can therefore never produce a molecule.
    """
    from g2rins.exception import AllZeroSamplingWeights

    smi = "{[] [<]CC({[<] [<]NN[>]|0|;; [>]}|poisson(80)|[H])[>], " "[<]OO({[<] [<]SS[>]|0|;; [>]}|poisson(80)|[H])[>]; " "[<][H]; [>][H] []}|poisson(600)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    original_sample = ensemble_creator.sample_mol_graph
    calls = 0

    def counted_sample(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", counted_sample)
    _reset_rngs(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(
            AllZeroSamplingWeights,
            match="nested special-target selection",
        ):
            ensemble_creator.create_ensemble(
                n_samples=1,
                max_number_of_discarded_chains=3,
            )

    assert calls == 1


def test_unavoidable_zero_terminator_after_growth_fails_fast(monkeypatch):
    """Ordinary growth cannot make a structurally zero terminator retryable."""
    from g2rins.exception import AllZeroSamplingWeights

    smi = "C{[>][<]CC[>];;[<][H]|0| []}|uniform(80,80)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    original_sample = ensemble_creator.sample_mol_graph
    calls = 0

    def counted_sample(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", counted_sample)
    _reset_rngs(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(
            AllZeroSamplingWeights,
            match="termination MW estimate",
        ):
            ensemble_creator.create_ensemble(
                n_samples=1,
                max_number_of_discarded_chains=3,
            )

    assert calls == 1


@pytest.mark.parametrize("distribution", ("poisson(1)", "uniform(0,0)"))
def test_zero_target_does_not_bypass_unavoidable_zero_terminator(
    distribution,
):
    """A real zero target still requires the architecture's declared caps."""
    from g2rins.exception import AllZeroSamplingWeights

    smi = "C{[>1][<1]CC[>2];;[<2][H]|0| []}|" f"{distribution}|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    assert ensemble_creator._automatic_zero_support_is_unavoidable[False]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AllZeroSamplingWeights, match="termination MW estimate"):
            ensemble_creator.sample_mol_graph(rng=np.random.default_rng(0))


def test_exact_zero_target_terminates_at_first_molecular_boundary(monkeypatch):
    """A genuine zero target is distinct from the negative no-target sentinel.

    This architecture has no structural dead end, so skipping its target check
    grows ``CC`` units forever. Forbid any sampling-loop growth so a sentinel
    regression fails immediately instead of hanging the test process.
    """
    from g2rins.ensemble_creator import _PartialAtomGraph

    smi = "{[] [<]CC[>]; [<][H]; [>][H] []}|uniform(0,0)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    def unexpected_growth(*_args, **_kwargs):
        raise AssertionError("an exact-zero target reached repeat-unit growth")

    monkeypatch.setattr(_PartialAtomGraph, "propagate_graph", unexpected_growth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        molecule = ensemble_creator.sample_mol_graph(rng=np.random.default_rng(0))

    assert sorted(data.get("atomic_num") for _, data in molecule.nodes(data=True)) == [1, 1, 6, 6]
    assert molecule.number_of_edges() == 3


def test_unavoidable_zero_terminator_from_repeat_source_fails_fast():
    """A repeat unit used as the source exposes its caps before any growth."""
    from g2rins.exception import AllZeroSamplingWeights

    smi = "{[] [<]CC[>];;[<][H]|0| []}|uniform(80,80)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    assert ensemble_creator._automatic_zero_support_is_unavoidable[True]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(
            AllZeroSamplingWeights,
            match="termination MW estimate",
        ):
            ensemble_creator.sample_mol_graph(
                use_repeat_units_as_source=True,
                rng=np.random.default_rng(0),
            )


def test_unavoidable_empty_nested_mw_support_fails_fast(monkeypatch):
    """A nested MW support above every parent budget must not burn retries.

    The truncated draw upper bound never exceeds the parent distribution's
    support ceiling, so uniform(500,600) under uniform(100,200) is empty on
    every chain and the first attempt must fail fatally with the precise
    empty-support error preserved as the cause.
    """
    from g2rins.exception import (
        AllZeroSamplingWeights,
        EmptyTruncatedDistributionSupport,
    )

    smi = "{[] [<]CC({[<] [<]NN[>];; [>]}|uniform(500,600)|[H])CC[>]; " "[<][H]; [>][H] []}|uniform(100,200)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    assert ensemble_creator._statically_empty_nested_mw_sto_gen_ids == frozenset({1})
    assert ensemble_creator._automatic_zero_support_is_unavoidable[False]

    original_sample = ensemble_creator.sample_mol_graph
    calls = 0

    def counted_sample(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", counted_sample)
    _reset_rngs(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AllZeroSamplingWeights) as caught:
            ensemble_creator.create_ensemble(
                n_samples=1,
                max_number_of_discarded_chains=5,
            )

    assert calls == 1
    assert isinstance(caught.value.__cause__, EmptyTruncatedDistributionSupport)


def test_overlapping_nested_mw_support_stays_retryable():
    """A budget-dependent empty support remains a chain-local rejection."""
    from g2rins.exception import EmptyTruncatedDistributionSupport

    smi = "{[] [<]CC({[<] [<]NN[>];; [>]}|uniform(150,600)|[H])CC[>]; " "[<][H]; [>][H] []}|uniform(100,200)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    assert not ensemble_creator._statically_empty_nested_mw_sto_gen_ids
    assert not ensemble_creator._automatic_zero_support_is_unavoidable[False]

    # Whether a chain dies depends on the drawn parent budget (about half of
    # all seeds), so demonstrate both outcomes across seeds instead of one.
    successes = 0
    rejections = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(64):
            if successes and rejections:
                break
            try:
                molecule = ensemble_creator.sample_mol_graph(rng=np.random.default_rng(seed))
            except EmptyTruncatedDistributionSupport:
                rejections += 1
            else:
                assert molecule.number_of_nodes() > 0
                successes += 1
    assert successes and rejections


def test_zero_molar_initiation_raises_domain_error():
    """Zero molar mass on every initiation route must not divide by zero.

    Building the atom graph used to raise a raw ZeroDivisionError from
    _create_init_weights; the routes are simply unreachable, and automatic
    sampling reports the missing source as a domain error.
    """
    from g2rins.exception import NoValidGenerationSource

    smi = "{[][<1]CC([>1])c1ccccc1, [<2]CC([>2])C(=O)OC; " "CC(C)[>1]|0|, CC(C)[>2]|0|; " "[<1][Br], [<2][Br][]}|schulz_zimm(700, 600)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    assert ensemble_creator._starting_node_idx == []
    with pytest.raises(NoValidGenerationSource):
        ensemble_creator.sample_mol_graph(rng=np.random.default_rng(0))


def test_zero_terminator_source_with_productive_alternative_is_retryable(
    monkeypatch,
):
    """A dead cap route remains local when another initiator can complete."""
    from g2rins.exception import AllZeroSamplingWeights, DeadSamplingPath

    smi = "{[] [<1]CC[>1], [<2]NN[>2]; C[>1], O[>2]; " "[<1][H]|0|, [<2][H] []}|uniform(80,80)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    graph = ensemble_creator.generative_graph
    unit_labels = g2rins.derive_unit_labels(graph).unit_id
    sources_by_unit = {unit_labels[source]: source for source in ensemble_creator._starting_node_idx}
    assert not ensemble_creator._automatic_zero_support_is_unavoidable[False]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monkeypatch.setattr(
            ensemble_creator,
            "_get_random_start_node",
            lambda _rng, _repeat=False: sources_by_unit["I0"],
        )
        with pytest.raises(DeadSamplingPath) as caught:
            ensemble_creator.sample_mol_graph(rng=np.random.default_rng(0))

        monkeypatch.setattr(
            ensemble_creator,
            "_get_random_start_node",
            lambda _rng, _repeat=False: sources_by_unit["I1"],
        )
        molecule = ensemble_creator.sample_mol_graph(rng=np.random.default_rng(0))

    assert isinstance(caught.value.__cause__, AllZeroSamplingWeights)
    assert molecule.number_of_nodes() > 0


def test_zero_target_global_arm_does_not_skip_dead_sibling():
    """Finishing a zero-target arm cannot hide another declared dead arm."""
    smi = "C(O{[>1][<1]CC[>2];;[<2][H]|0| []}|uniform(80,80)|)" "(N{[>3][<3]NN[>4];;[<4][H] []}|uniform(0,0)|)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    assert not ensemble_creator._automatic_zero_support_is_unavoidable[False]

    from g2rins.exception import AllZeroSamplingWeights, DeadSamplingPath

    # The arm order is stochastic, but both declared arms must be processed.
    # The unusable cap therefore rejects every order instead of returning a
    # molecule with that arm silently omitted.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(64):
            with pytest.raises(DeadSamplingPath) as caught:
                ensemble_creator.sample_mol_graph(rng=np.random.default_rng(seed))
            assert isinstance(caught.value.__cause__, AllZeroSamplingWeights)


def test_zero_probability_productive_source_does_not_mask_fatal_routes(monkeypatch):
    """An unreachable source cannot make the reachable dead routes retryable."""
    from g2rins.exception import AllZeroSamplingWeights

    smi = "{[] [<1]CC[>1]|0|, [<2]NN[>2]; " "C[>1], N[>1], O[>2]|0|; " "[<1][H], [<2][H] []}|poisson(100)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    assert np.count_nonzero(ensemble_creator._starting_node_weight > 0) == 2
    assert np.count_nonzero(ensemble_creator._starting_node_weight == 0) == 1

    original_sample = ensemble_creator.sample_mol_graph
    calls = 0

    def counted_sample(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", counted_sample)
    _reset_rngs(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AllZeroSamplingWeights):
            ensemble_creator.create_ensemble(
                n_samples=1,
                max_number_of_discarded_chains=3,
            )

    assert calls == 1


def test_dead_construction_proof_respects_consumed_and_dropped_half_bonds():
    """Only retained, unconsumed special half-bonds recurse into children."""
    import networkx as nx

    from g2rins.ensemble_creator import EnsembleCreator
    from g2rins.generative_graph import (
        _EDGE_STOCHASTIC_ID_NAME,
        _TRANSITION_NAME,
    )

    graph = nx.MultiDiGraph()
    graph.add_node(
        "outer_entry",
        stochastic_id_tree=[0],
        unit_molar_amounts=[1.0, 1.0, 1.0],
        gen_weight=1.0,
    )
    graph.add_node(
        "outer_other",
        stochastic_id_tree=[0],
        unit_molar_amounts=[1.0, 1.0, 1.0],
        gen_weight=1.0,
    )
    graph.add_node(
        "child_entry",
        stochastic_id_tree=[1, 0],
        unit_molar_amounts=[1.0, 1.0, 1.0],
        gen_weight=1.0,
    )
    graph.add_node(
        "dead_leaf",
        stochastic_id_tree=[2, 1, 0],
        unit_molar_amounts=[1.0, 1.0, 0.0],
        gen_weight=1.0,
    )
    graph.add_edge(
        "outer_entry",
        "child_entry",
        static=False,
        **{
            _TRANSITION_NAME: 1.0,
            _EDGE_STOCHASTIC_ID_NAME: 1,
        },
    )
    graph.add_edge(
        "child_entry",
        "dead_leaf",
        static=False,
        **{
            _TRANSITION_NAME: 1.0,
            _EDGE_STOCHASTIC_ID_NAME: 2,
        },
    )

    probe = EnsembleCreator.__new__(EnsembleCreator)
    probe._generative_graph = graph
    probe._static_components = (
        frozenset(("outer_entry", "outer_other")),
        frozenset(("child_entry",)),
        frozenset(("dead_leaf",)),
    )
    probe._node_to_static_component = {node: component_id for component_id, component in enumerate(probe._static_components) for node in component}
    probe._statically_empty_nested_mw_sto_gen_ids = frozenset()

    dead_states, immediate_components = probe._find_provably_dead_construction_states()
    assert 1 in immediate_components
    assert (0, "outer_entry") not in dead_states
    assert (0, "outer_other") in dead_states

    # Positive nested support is normalized, but gen_weight=0 drops the
    # half-bond before nested_transition can follow its dead child.
    graph.nodes["outer_entry"]["gen_weight"] = 0.0
    dead_states, immediate_components = probe._find_provably_dead_construction_states()
    assert 0 not in immediate_components
    assert (0, "outer_other") not in dead_states

    # Empty support fails during construction, before either drop or pop.
    graph.nodes["child_entry"]["unit_molar_amounts"][1] = 0.0
    dead_states, immediate_components = probe._find_provably_dead_construction_states()
    assert 0 in immediate_components
    assert (0, "outer_entry") in dead_states
    assert (0, "outer_other") in dead_states


def test_source_dead_proof_keeps_cross_owner_and_malformed_routes_unknown():
    """Dynamic bucket transfers and malformed hierarchy never prove fatal."""
    import networkx as nx

    from g2rins.ensemble_creator import EnsembleCreator
    from g2rins.generative_graph import (
        _EDGE_STOCHASTIC_ID_NAME,
        _TRANSITION_NAME,
    )

    graph = nx.MultiDiGraph()
    graph.add_node(
        "source",
        stochastic_id_tree=[0],
        unit_molar_amounts=[1.0, 1.0],
        gen_weight=1.0,
        gen_hierarchy=0,
    )
    graph.add_node(
        "target",
        stochastic_id_tree=[1, 0],
        unit_molar_amounts=[1.0, 1.0],
        gen_weight=1.0,
        gen_hierarchy=0,
    )
    graph.add_edge(
        "source",
        "target",
        static=False,
        **{
            _TRANSITION_NAME: 1.0,
            _EDGE_STOCHASTIC_ID_NAME: 0,
        },
    )

    probe = EnsembleCreator.__new__(EnsembleCreator)
    probe._generative_graph = graph
    probe._static_proof_supported = True
    probe._static_components = (
        frozenset(("source",)),
        frozenset(("target",)),
    )
    probe._node_to_static_component = {"source": 0, "target": 1}
    probe._provably_dead_construction_states = frozenset(((0, "source"),))
    probe._provably_immediate_zero_components = frozenset()
    probe._source_provably_dead_cache = {}

    assert not probe._source_is_provably_dead("source")

    # The same-owner route would retain/follow the source expansion, so the
    # identical state proof is usable there.
    graph.nodes["target"]["stochastic_id_tree"] = [0]
    probe._source_provably_dead_cache.clear()
    assert probe._source_is_provably_dead("source")

    graph.nodes["source"]["gen_hierarchy"] = np.nan
    probe._source_provably_dead_cache.clear()
    assert not probe._source_is_provably_dead("source")


def test_asymmetric_static_graph_disables_fatal_template_proof():
    """Undirected component reasoning is used only for symmetric templates."""
    from g2rins.ensemble_creator import EnsembleCreator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generative_graph = g2rins.G2rins.make("C{[>][<]CC[>]|0|;;[<]}|poisson(100)|[H]").get_graph_creator().get_generative_graph(include_bond_connectors=False)

    reverse_removed = False
    for u, v, _key, data in list(generative_graph.edges(keys=True, data=True)):
        if u == v or not data.get("static") or not generative_graph.has_edge(v, u):
            continue
        for reverse_key, reverse_data in list(generative_graph.get_edge_data(v, u).items()):
            if reverse_data.get("static"):
                generative_graph.remove_edge(v, u, reverse_key)
                reverse_removed = True
        if reverse_removed:
            break

    assert reverse_removed, "fixture must contain a bidirectional static edge"
    ensemble_creator = EnsembleCreator(generative_graph)
    assert not ensemble_creator._static_proof_supported
    assert not ensemble_creator._automatic_zero_support_is_unavoidable[False]


def test_source_branch_with_productive_alternative_remains_retryable(monkeypatch):
    """A dead initiator is chain-local when another initiator is productive."""
    from g2rins.exception import AllZeroSamplingWeights, DeadSamplingPath

    smi = "{[] [<1]CC[>1]|0|, [<2]NN[>2]; " "C[>1], O[>2]; [<1][H], [<2][H] []}|poisson(100)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    graph = ensemble_creator.generative_graph
    unit_labels = g2rins.derive_unit_labels(graph).unit_id
    sources_by_unit = {unit_labels[source]: source for source in ensemble_creator._starting_node_idx}

    # Force each automatic-source outcome so the provenance assertion does not
    # depend on NumPy's seed-to-choice mapping or template node order.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monkeypatch.setattr(
            ensemble_creator,
            "_get_random_start_node",
            lambda _rng, _repeat=False: sources_by_unit["I0"],
        )
        with pytest.raises(DeadSamplingPath) as caught:
            ensemble_creator.sample_mol_graph(rng=np.random.default_rng(0))

        monkeypatch.setattr(
            ensemble_creator,
            "_get_random_start_node",
            lambda _rng, _repeat=False: sources_by_unit["I1"],
        )
        molecule = ensemble_creator.sample_mol_graph(rng=np.random.default_rng(0))

    assert isinstance(caught.value.__cause__, AllZeroSamplingWeights)
    assert molecule.number_of_nodes() > 0


def test_explicit_dead_source_is_fatal_when_automatic_mode_can_retry():
    """An explicit source fixes the route, so its structural error is fatal."""
    from g2rins.exception import AllZeroSamplingWeights

    smi = "{[] [<1]CC[>1]|0|, [<2]NN[>2]; " "C[>1], O[>2]; [<1][H], [<2][H] []}|poisson(100)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()

    graph = ensemble_creator.generative_graph
    unit_labels = g2rins.derive_unit_labels(graph).unit_id
    dead_source = next(source for source in ensemble_creator._starting_node_idx if unit_labels[source] == "I0")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AllZeroSamplingWeights):
            ensemble_creator.sample_mol_graph(
                source=dead_source,
                rng=np.random.default_rng(0),
            )


def test_termination_mw_estimate_preserves_live_provenance_and_rng(monkeypatch):
    """Termination lookahead is observational and must leave live state alone."""
    from g2rins.ensemble_creator import _PartialAtomGraph

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make("{[] [<]CC[>]; [<][H]; [>][H], [>]N []}|uniform(200, 200)|").get_graph_creator().get_ensemble_creator()

    original_estimate = _PartialAtomGraph.get_average_termination_mw
    observations = []

    def observe_estimate(self, sto_atom_id, static_graph, rng):
        before = (
            self.stochastic_tracker.path_is_conditional,
            repr(rng.bit_generator.state),
        )
        result = original_estimate(self, sto_atom_id, static_graph, rng)
        after = (
            self.stochastic_tracker.path_is_conditional,
            repr(rng.bit_generator.state),
        )
        observations.append((before, after))
        return result

    monkeypatch.setattr(
        _PartialAtomGraph,
        "get_average_termination_mw",
        observe_estimate,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator.sample_mol_graph(
            rng=np.random.default_rng(0),
            termination_flag=0,
        )

    assert observations
    assert all(before_rng == after_rng for (_before_flag, before_rng), (_after_flag, after_rng) in observations), "termination-MW estimation advanced the live RNG"
    assert all(before_flag == after_flag for (before_flag, _before_rng), (after_flag, _after_rng) in observations), "termination-MW estimation changed live path provenance"


@pytest.mark.parametrize(
    ("smi", "use_repeat_units_as_source"),
    (
        pytest.param(
            ("{[>|1 1 0 0|] [<|0 0 0 0|]CC[<|0 0 0 0|], " "[>|1 1 0 0|]OO[>|1 1 0 0|];; [<]}|uniform(300, 400)|"),
            False,
            id="default-source-mode",
        ),
        pytest.param(
            "C{[>][<]CC[>];;[<]}|uniform(40,40)|[H]",
            True,
            id="repeat-unit-source-mode",
        ),
    ),
)
def test_empty_automatic_source_raises_domain_error(
    smi,
    use_repeat_units_as_source,
):
    """Both automatic-source modes must reject an empty candidate set cleanly."""
    from g2rins.exception import NoValidGenerationSource

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
        with pytest.raises(NoValidGenerationSource) as caught:
            ensemble_creator.sample_mol_graph(
                use_repeat_units_as_source=use_repeat_units_as_source,
                rng=np.random.default_rng(0),
            )
    assert caught.value.use_repeat_units_as_source is use_repeat_units_as_source


def test_create_ensemble_propagates_unexpected_value_error(monkeypatch):
    """Programming/input ValueErrors are not retryable sampling outcomes."""
    from g2rins.ensemble_creator import EnsembleCreator

    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)
    expected_error = ValueError("sentinel implementation failure")
    calls = 0

    def fail(**_kwargs):
        nonlocal calls
        calls += 1
        raise expected_error

    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", fail)

    with pytest.raises(ValueError) as caught:
        ensemble_creator.create_ensemble(n_samples=1, max_number_of_discarded_chains=3)

    assert caught.value is expected_error
    assert calls == 1


def test_create_ensemble_does_not_retry_fatal_sampling_weights(monkeypatch):
    """A statically invalid all-zero configuration remains fail-fast."""
    from g2rins.ensemble_creator import EnsembleCreator
    from g2rins.exception import AllZeroSamplingWeights

    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)
    expected_error = AllZeroSamplingWeights("root target selection")
    calls = 0

    def fail(**_kwargs):
        nonlocal calls
        calls += 1
        raise expected_error

    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", fail)

    with pytest.raises(AllZeroSamplingWeights) as caught:
        ensemble_creator.create_ensemble(n_samples=1, max_number_of_discarded_chains=3)

    assert caught.value is expected_error
    assert calls == 1


def test_create_ensemble_retries_chain_local_error_with_diagnostic(monkeypatch):
    """A chain-local retry preserves successes and is reported once."""
    from g2rins.ensemble_creator import EnsembleCreator
    from g2rins.exception import EmptyTruncatedDistributionSupport

    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)
    dead_path = EmptyTruncatedDistributionSupport("nested", 1.0, 2.0)
    first_molecule = object()
    second_molecule = object()
    # Plain-mode create_ensemble samples without molecule_info: the fake
    # mirrors the bare-molecule return shape.
    outcomes = iter((first_molecule, dead_path, second_molecule))
    calls = 0

    def sample(**_kwargs):
        nonlocal calls
        calls += 1
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", sample)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        molecules = ensemble_creator.create_ensemble(n_samples=2)

    assert molecules == [first_molecule, second_molecule]
    assert calls == 3
    summaries = [warning.message for warning in caught_warnings if isinstance(warning.message, DiscardedSamplingPaths)]
    assert len(summaries) == 1
    assert summaries[0].discarded_count == 1
    assert summaries[0].reasons == (("EmptyTruncatedDistributionSupport", 1),)


def test_create_ensemble_discard_cap_reraises_original_cause(monkeypatch):
    """With no successful chain, exhausting retries preserves the first cause."""
    from g2rins.ensemble_creator import EnsembleCreator
    from g2rins.exception import EmptyTruncatedDistributionSupport

    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)
    first_error = EmptyTruncatedDistributionSupport("nested", 1.0, 2.0)
    second_error = EmptyTruncatedDistributionSupport("nested", 3.0, 4.0)
    errors = iter((first_error, second_error))
    calls = 0

    def fail(**_kwargs):
        nonlocal calls
        calls += 1
        raise next(errors)

    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", fail)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(EmptyTruncatedDistributionSupport) as caught:
            ensemble_creator.create_ensemble(
                n_samples=1,
                max_number_of_discarded_chains=2,
            )

    assert caught.value is first_error
    assert calls == 2


def test_create_ensemble_discard_cap_preserves_completed_chains(monkeypatch):
    """A later failed chain must not erase earlier accepted molecules."""
    from g2rins.ensemble_creator import EnsembleCreator
    from g2rins.exception import EmptyTruncatedDistributionSupport

    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)
    molecule = object()
    outcomes = iter(
        (
            molecule,
            EmptyTruncatedDistributionSupport("nested", 1.0, 2.0),
            EmptyTruncatedDistributionSupport("nested", 3.0, 4.0),
        )
    )
    calls = 0

    def sample(**_kwargs):
        nonlocal calls
        calls += 1
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", sample)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        molecules = ensemble_creator.create_ensemble(
            n_samples=2,
            max_number_of_discarded_chains=2,
        )

    assert molecules == [molecule]
    assert calls == 3


def test_create_ensemble_restores_process_warning_filters(monkeypatch):
    """Ensemble-local warning capture must not mutate process-wide filters.

    The capture only runs inside the sampling loop, so the check must drive
    at least one chain through it (n_samples=0 would never enter the loop).
    """
    from g2rins.ensemble_creator import EnsembleCreator

    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)
    molecule = object()

    def sample(**_kwargs):
        warnings.warn("chain-local detail", UserWarning, stacklevel=1)
        return molecule

    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", sample)
    filters_before = list(warnings.filters)

    assert ensemble_creator.create_ensemble(n_samples=1) == [molecule]
    assert warnings.filters == filters_before


def test_create_ensemble_total_failure_emits_discard_summary(monkeypatch):
    """The reason breakdown must reach the caller when no chain succeeded."""
    from g2rins.ensemble_creator import EnsembleCreator
    from g2rins.exception import EmptyTruncatedDistributionSupport

    ensemble_creator = EnsembleCreator.__new__(EnsembleCreator)

    def fail(**_kwargs):
        raise EmptyTruncatedDistributionSupport("nested", 1.0, 2.0)

    monkeypatch.setattr(ensemble_creator, "sample_mol_graph", fail)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(EmptyTruncatedDistributionSupport):
            ensemble_creator.create_ensemble(
                n_samples=1,
                max_number_of_discarded_chains=3,
            )

    summaries = [warning.message for warning in caught_warnings if isinstance(warning.message, DiscardedSamplingPaths)]
    assert len(summaries) == 1
    assert summaries[0].discarded_count == 3
    assert summaries[0].reasons == (("EmptyTruncatedDistributionSupport", 3),)


def test_special_target_molar_ratio_honored():
    """Forced-nested (special-target) arm entry units must be drawn by their
    declared molar amounts. The draw used to read the parent-SO slot of
    unit_molar_amounts — identical for every candidate, so it cancelled and a
    declared 90/10 ratio sampled 50/50."""
    smi = "{[] [<]CC({[<] [<]NN[>]|9.0|, [<]C(C)O[>]|1.0|;; [>]}|poisson(80)|[H])CC[>]; [<][H]; [>][H] []}|poisson(400)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    n_count = o_count = 0
    for seed in range(6):
        _reset_rngs(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mol_graph = ensemble_creator.sample_mol_graph()
        for _node, data in mol_graph.nodes(data=True):
            if data["atomic_num"] == 7:
                n_count += 1
            elif data["atomic_num"] == 8:
                o_count += 1
    fraction = n_count / (n_count + o_count)
    assert fraction > 0.7, f"NN-arm fraction {fraction:.2f} ignores the declared 9:1 molar ratio"


def test_arm_dead_end_warns_only_at_chain_level():
    """A nested arm that structurally dead-ends below its own drawn target
    retires silently; only a chain-level (parentless) instance below target
    emits the discardable PossibleNonRepresentativePolymerChain. On this
    string the backbone truncates (exactly one warning per sample); the dead
    arm used to add a second warning, which made create_ensemble discard
    whole on-target chains of arm-dead-ending architectures."""
    smi = "{[] [<]CC([<])C(=O)O{[>] [<]CC([<])C(=O)O{[>][<]CCO[>];;[<]C[]}|poisson(300)|;;[<]C[]}|poisson(1200)|; COC(=O)C(C)[>]; [<]Br[]}|poisson(9000)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    for seed in SEEDS:
        _reset_rngs(seed)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            ensemble_creator.sample_mol_graph()
        count = sum(1 for w in caught_warnings if issubclass(w.category, PossibleNonRepresentativePolymerChain))
        assert count == 1, f"seed {seed}: {count} non-representative warnings (arm retirement must not warn)"


def test_directional_dead_end_warns_at_parse():
    """Direction-dependent dead ends (asymmetric zero bond-probability
    vectors that make every exit unreachable along generation direction) must
    warn StochasticMissingPath at parse time. The reachability check briefly
    ran on a fully undirected view, which hid them until create_ensemble
    failed at runtime with no diagnostic; valid strings whose route only
    walks a static bond backwards must stay warning-free."""
    from g2rins.exception import StochasticMissingPath

    dead_end = "{[>|1 1 0 0|] [<|0 0 0 0|]CC[<|0 0 0 0|], [>|1 1 0 0|]OO[>|1 1 0 0|];; [<]}|uniform(300, 400)|"
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        g2rins.G2rins.make(dead_end).get_graph_creator()
    assert any(issubclass(w.category, StochasticMissingPath) for w in caught_warnings)

    valid = "{[] [<|0.8|]CCO[>|0.8|], [<|0.2|]CC(C)O[>|0.2|]; CO[>]; [<][H] []}|log_normal(1400.0, 1.15)|"
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        g2rins.G2rins.make(valid).get_graph_creator()
    assert not any(issubclass(w.category, StochasticMissingPath) for w in caught_warnings)


def test_too_many_stochastic_objects_raises():
    """Stochastic-object ids index fixed-size per-node vectors, so an 11th
    sequential stochastic object cannot be represented: it must raise
    TooManyStochasticObjects at graph construction instead of the bare
    IndexError it used to hit; ten still build."""
    from g2rins.exception import TooManyStochasticObjects

    block = "{[<] [<]CC[>];; [>]}|uniform(100, 200)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g2rins.G2rins.make("CCO" + block * 10 + "CCN").get_graph_creator().get_generative_graph(include_bond_connectors=False)
        with pytest.raises(TooManyStochasticObjects):
            g2rins.G2rins.make("CCO" + block * 11 + "CCN").get_graph_creator().get_generative_graph(include_bond_connectors=False)


def test_first_check_crossing_does_not_warn_undershoot():
    """A target MW at or below one repeat unit crosses on the instance's
    first check, where no undershoot snapshot can exist and overshoot is
    unavoidable (main behaved identically, silently). The
    UndershootSnapshotMissed warning is reserved for genuine adaptive-
    lookahead misses; it used to fire here on ~4/10 seeds blaming a growth
    step that never happened."""
    from g2rins.exception import UndershootSnapshotMissed

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make("C{[>][<]CC[>];;[<]}|gauss(30.0, 5.0)|[H]").get_graph_creator().get_ensemble_creator()
    for seed in range(10):
        _reset_rngs(seed)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            ensemble_creator.sample_mol_graph()
        assert not any(issubclass(w.category, UndershootSnapshotMissed) for w in caught_warnings), f"seed {seed}"


def test_connection_dummies_carry_no_hydrogen():
    """Connection placeholder atoms (atomic_num 0) copy every attribute of
    their real neighbor; the neighbor's written aromatic H count must not
    render as a phantom locked hydrogen ([*H:n]) on the stub in the per-unit
    sequence output."""
    smi = "{[] [<]c1ccc[bH-]1[>]; [<][H]; [>][H] []}|poisson(400)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
        _reset_rngs(0)
        result = ensemble_creator.create_ensemble(n_samples=1, output_format="smiles", ensemble_info=True)
    unit_smiles = [s for chain in result.sequences for unit in chain for s in unit]
    assert unit_smiles and not any("*H" in s for s in unit_smiles), f"phantom H on connection dummy: {unit_smiles}"


def test_legacy_edge_schema_rejected():
    """Sampling filters every non-static decision by the per-edge
    'stochastic_id'; a graph built against the older schema (per-edge
    'hierarchy') used to generate silently truncated, end-group-less
    molecules. EnsembleCreator must reject it loudly instead."""
    from g2rins.ensemble_creator import EnsembleCreator
    from g2rins.exception import IncompatibleGenerativeGraphSchema

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generative_graph = g2rins.G2rins.make("C{[>][<]CC(C)[>];;[<]}|poisson(900)|[H]").get_graph_creator().get_generative_graph(include_bond_connectors=False)
    for _u, _v, data in generative_graph.edges(data=True):
        if "stochastic_id" in data:
            data["hierarchy"] = data.pop("stochastic_id")
    with pytest.raises(IncompatibleGenerativeGraphSchema):
        EnsembleCreator(generative_graph)


@pytest.mark.parametrize(
    ("atomic_num", "charge", "bonds", "expected_h"),
    [
        (15, +1, 0, 4),  # P+ fills to PH4+ — the old single-default-valence math gave 6
        (15, +1, 3, 1),  # R3PH+ phosphonium
        (15, 0, 0, 3),  # PH3, not PH5 (chem_resource stores P as 5)
        (15, 0, 3, 0),  # R3P phosphine, no phantom H
        (16, 0, 3, 1),  # S climbs to its tetravalent tier
        (16, 0, 2, 0),  # thioether
        (7, +1, 0, 4),  # NH4+
        (8, -1, 1, 0),  # alkoxide
        (5, -1, 1, 3),  # borohydride-like B-
        (6, +1, 1, 2),  # carbocation
        (26, 0, 0, 0),  # metals get no implicit H
    ],
)
def test_hydrogen_inference_matches_chemistry(atomic_num, charge, bonds, expected_h):
    """MW tracking must count the hydrogens RDKit will realize. These are
    literature-known counts (PH4+, phosphine, sulfonium, ammonium, ...);
    the old arithmetic over chem_resource.default_valence overcounted
    multivalent elements (P+ tracked 6 H instead of 4), overstating tracked
    MW and terminating phosphorus-containing chains early."""
    from g2rins.ensemble_creator import _infer_hydrogen_count

    assert _infer_hydrogen_count(atomic_num, charge, bonds) == expected_h


@pytest.mark.parametrize(
    ("atomic_num", "expected_h"),
    [
        (6, 1),  # benzene carbon keeps its ring hydrogen
        (7, 0),  # pyridine nitrogen
        (8, 0),  # furan oxygen
        (16, 0),  # thiophene sulfur — NOT the tetravalent tier of non-aromatic S(3)
        (34, 0),  # selenophene selenium
    ],
)
def test_aromatic_hydrogen_inference(atomic_num, expected_h):
    """Aromatic atoms never climb past their default valence: at ring valence 3
    (two ring bonds plus the aromatic increment) thiophene S binds no hydrogen,
    while delegating to RDKit without the aromatic flag let multivalent
    heteroatoms climb tiers and credited every thiophene ring a phantom H."""
    from g2rins.ensemble_creator import _infer_hydrogen_count

    assert _infer_hydrogen_count(atomic_num, 0, 3, aromatic=True) == expected_h


def test_polythiophene_tracked_mw_matches_rdkit():
    """End-to-end guard for aromatic multivalent heteroatoms: a polythiophene's
    tracked weight must equal its RDKit MolWt (aromatic S used to be credited
    one phantom hydrogen per ring)."""
    smi = "{[] [<]c1ccc([>])s1; [<][H]; [>][H] []}|poisson(420)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    _reset_rngs(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mol_graph, _units, _bonds, _seq, tracked, _dist = ensemble_creator.sample_mol_graph(molecule_info=True)
    mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
    Chem.SanitizeMol(mol)
    rdkit_mw = Descriptors.MolWt(mol)
    tracked_mw = sum(sum(v) for v in tracked.values())
    assert abs(rdkit_mw - tracked_mw) < 1.0, f"tracked {tracked_mw:.1f} vs RDKit {rdkit_mw:.1f}"


def test_phosphonium_tracked_mw_matches_rdkit():
    """End-to-end guard for multivalent-element MW tracking: a phosphonium
    polymer's tracked weight must equal its RDKit MolWt (P+ used to be
    credited two phantom hydrogens per cation)."""
    smi = "{[] [<]CC([>])C[P+](C)(C)C, [<]CC[>]; [>][H]; [<][H] []}|uniform(600, 900)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    _reset_rngs(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mol_graph, _units, _bonds, _seq, tracked, _dist = ensemble_creator.sample_mol_graph(molecule_info=True)
    mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
    Chem.SanitizeMol(mol)
    rdkit_mw = Descriptors.MolWt(mol)
    tracked_mw = sum(sum(v) for v in tracked.values())
    assert abs(rdkit_mw - tracked_mw) < 1.0, f"tracked {tracked_mw:.1f} vs RDKit {rdkit_mw:.1f}"


def test_nested_side_chain_so_gets_one_instance_per_junction():
    """A nested stochastic object with its own MW distribution used as a
    repeat unit (graft side chains entered through backbone ports) grows one
    instance with one independent draw per junction. The transition sweep
    used to file the converted sibling ports under the LANDING instance
    instead of the fired-level instance: every graft pooled into that single
    poisson(200) draw, side chains never propagated past the junction unit,
    the unfired ports were destroyed with the instance's terminate-time wipe,
    and the outer target became unreachable — every chain was discarded as
    non-representative."""
    smi = "{[] [<1]{[>1] [<1]CCCO[>2], [<2]CCO[>2]; ; [<2]}|poisson(200)|[>2]; " "{[] [<][Si](C)([>1])O[>]; O[>]; [<][H] [<1]}|poisson(1000)|[>1]; " "[<2][H] []}|poisson(2000)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    for seed in SEEDS:
        _reset_rngs(seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mol_graph, _units, _bonds, _seq, tracked, dist = ensemble_creator.sample_mol_graph(molecule_info=True)
        bad = [w.category.__name__ for w in caught if issubclass(w.category, (PossibleNonRepresentativePolymerChain, DiscardedSamplingPaths))]
        assert not bad, f"seed {seed}: graft chain flagged non-representative: {bad}"
        mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
        Chem.SanitizeMol(mol)
        side_id = next(i for i, d in dist.items() if d == "|poisson(200.0)|")
        side_masses = tracked[side_id]
        assert len(side_masses) >= 2, f"seed {seed}: grafts pooled into {len(side_masses)} side-chain instance(s): {side_masses}"
        # The crossing rounding is all-or-nothing: a kept side chain always
        # carries the junction plus repeat units. A bare ~59 Da junction means
        # the ports were captured into a nested instance again.
        assert min(side_masses) > 90.0, f"seed {seed}: bare junction graft survived: {side_masses}"


def test_multifunctional_ports_compete_with_chain_continuation():
    """A multifunctional initiator's [>1] ports enter a nested arm SO while
    the chain also continues through units embedding another nested SO. Port
    initiation and chain continuation must COMPETE in the owner's weighted
    draw (the multifunctional initiation principle generalized to nested
    levels): every port grows its own arm with its own MW draw. The deferred
    continuation used to fire directly from the finished child's bucket,
    bypassing the owner's pool entirely, so exactly one arm ever grew and the
    remaining ports were silently wiped at the outer termination."""
    smi = "{[] [<]NNNN{[>] [<]CCO[>];; [<]}|poisson(100)|[>], " "[<1]{[>] [<]CCO[>];; [<]}|poisson(100)|[>]; " "C(O[>1])C(O[>1])CO[>1]; [<][H] []}|poisson(2000)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    for seed in SEEDS:
        _reset_rngs(seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mol_graph, _units, _bonds, _seq, tracked, dist = ensemble_creator.sample_mol_graph(molecule_info=True)
        bad = [w.category.__name__ for w in caught if issubclass(w.category, (PossibleNonRepresentativePolymerChain, DiscardedSamplingPaths))]
        assert not bad, f"seed {seed}: chain flagged non-representative: {bad}"
        mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
        Chem.SanitizeMol(mol)
        # Construction order is parse-stable: gen 1 is the SO embedded in the
        # chain unit, gen 2 the arm SO entered through the [>1] ports.
        assert dist[2] == "|poisson(100.0)|"
        arm_masses = tracked[2]
        assert len(arm_masses) == 3, f"seed {seed}: expected one arm per initiator port, got {arm_masses}"
        assert min(arm_masses) > 40.0, f"seed {seed}: empty arm: {arm_masses}"
        assert len(tracked[1]) >= 2, f"seed {seed}: chain continuation lost every draw: {tracked[1]}"


def test_transition_bond_selection_is_level_aware():
    """The dead-end verdict of transition bond selection must be
    deterministic: _pop_random_bond filters candidates by the requested
    stochastic level BEFORE hierarchy selection and the weighted draw, so
    None means no bond in the bucket serves that level, and a served level is
    always found. It used to draw one bond from the max-hierarchy tier of ALL
    bonds and only then check the level — a single mismatched pick reported a
    false dead end (while a compatible bond sat in the bucket) that the
    sampler's truncation path treats as terminal for the whole molecule."""
    from g2rins.ensemble_creator import _PartialAtomGraph, _StochasticObjectTracker
    from g2rins.generative_graph import _EDGE_STOCHASTIC_ID_NAME, _TRANSITION_NAME

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make("C{[>][<]CC(C)[>];;[<]}|poisson(900)|[H]").get_graph_creator().get_ensemble_creator()
    generative_graph = ensemble_creator.generative_graph
    rng = np.random.default_rng(0)
    tracker = _StochasticObjectTracker(generative_graph, rng)
    source = ensemble_creator._starting_node_idx[0]
    tree = generative_graph.nodes[source]["stochastic_id_tree"]
    sto_atom_id, _parents = tracker.register_parent_atom_instances(tree[0], tree[1], tree[1:])
    partial = _PartialAtomGraph(generative_graph, ensemble_creator._static_graph, source, tracker, sto_atom_id, rng)
    bucket = partial._open_half_bond_map[sto_atom_id]
    assert bucket

    # A level nothing serves: deterministic None, bucket untouched, no draw.
    before = list(bucket)
    bond, non_used = partial._pop_random_bond(bucket, sto_atom_id, 999, rng)
    assert bond is None and non_used == [] and bucket == before

    # A level that IS served must always be found.
    served_levels = {attr.get(_EDGE_STOCHASTIC_ID_NAME) for half_bond in bucket for attr in half_bond._mode_attr_map.get(_TRANSITION_NAME, [])}
    assert served_levels, "test string must expose at least one transition level"
    level = next(iter(served_levels))
    bond, _non_used = partial._pop_random_bond(bucket, sto_atom_id, level, rng)
    assert bond is not None
    assert any(attr.get(_EDGE_STOCHASTIC_ID_NAME) == level for attr in bond._mode_attr_map[_TRANSITION_NAME])


def test_charged_atom_tracked_mw_matches_rdkit():
    """Charged bracket atoms bind more/fewer hydrogens than the neutral
    default valence implies ([NH3+] realizes three, not one). The tracker's
    charge-aware inference must agree with the RDKit molecule; the charge-
    blind formula undercounted ~2 Da per cation, so charged chains overshot
    their target MW before should_terminate fired. Also guards the ancestor
    credit, which used an unclamped hydrogen term that subtracted phantom
    mass from parents of over-coordinated atoms."""
    smi = "{[] [<]CC([>])C[NH3+], [<]CC[>]; [>][H]; [<][H] []}|uniform(600, 900)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    _reset_rngs(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mol_graph, _units, _bonds, _seq, tracked, _dist = ensemble_creator.sample_mol_graph(molecule_info=True)
    mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
    Chem.SanitizeMol(mol)
    rdkit_mw = Descriptors.MolWt(mol)
    tracked_mw = sum(sum(v) for v in tracked.values())
    assert abs(rdkit_mw - tracked_mw) < 1.0, f"tracked {tracked_mw:.1f} vs RDKit {rdkit_mw:.1f}"


def test_counterion_tracked_mw_matches_rdkit():
    """Counterions ride along on association edges (bond_type 0): they add no
    valence (no phantom hydrogens on the ion or its site) but their mass
    belongs to the chain, so the tracked MW must still match RDKit."""
    smi = "{[] [<]CC([>])C[NH3+].[Cl-], [<]CC[>]; [>][H]; [<][H] []}|uniform(600, 900)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    _reset_rngs(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mol_graph, _units, _bonds, _seq, tracked, _dist = ensemble_creator.sample_mol_graph(molecule_info=True)
    mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
    Chem.SanitizeMol(mol)
    rdkit_mw = Descriptors.MolWt(mol)
    tracked_mw = sum(sum(v) for v in tracked.values())
    assert abs(rdkit_mw - tracked_mw) < 1.0, f"tracked {tracked_mw:.1f} vs RDKit {rdkit_mw:.1f}"


def test_hyperbranched_molecule_branches():
    """A sampled hyperbranched chain must contain many tri-substituted
    nitrogens (branch points), confirming the AB2 monomer branches."""
    smi = "{[] [<]CCN([>])[>]; [<][H]; O[>], [<][H] []}|poisson(2000.0)|"
    _reset_rngs(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mol_graph = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator().sample_mol_graph()
    mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
    Chem.SanitizeMol(mol)
    branch_points = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "N" and sum(1 for nb in atom.GetNeighbors() if nb.GetSymbol() == "C") >= 3)
    assert branch_points > 3, f"expected a branched network, found {branch_points} branch points"


# Stochastic object nested as a repeat unit inside another nested repeat unit
# (three levels). The outer level's continuation junction physically ends up
# in a terminated grandchild's bucket, which the single-bucket hand-off never
# reached: ~92% of chains starved below the outer target and were discarded
# as non-representative (and implicitly created intermediate instances lacked
# parent_map chains, mis-routing the continuation level).
TRIPLE_NESTED_SMI = "{[] [<]{[>] [<]CC[>], [<]{[>] [<]NN[>];; [<]}|poisson(100)|[>];; [<]}|poisson(300)|[>], [<]OO[>]; Cl[>]; [<][H] []}|poisson(1000)|"


def test_triple_nested_repeat_unit_does_not_starve():
    """Triple-nested SO-as-repeat-unit chains complete without discards, land
    on the outer target, and carry the outer declared Cl cap exactly once —
    even when the capped junction lives in a terminated grandchild's bucket
    rather than the finished child's own bucket."""
    n_samples = 12
    _reset_rngs(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(TRIPLE_NESTED_SMI).get_graph_creator().get_ensemble_creator()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        molecules = ensemble_creator.create_ensemble(
            n_samples=n_samples,
            output_format="mol",
            max_number_of_discarded_chains=40,
        )
    assert molecules is not None and len(molecules) == n_samples
    discards = [w for w in caught_warnings if issubclass(w.category, DiscardedSamplingPaths)]
    assert not discards, f"starvation discards returned: {[str(w.message) for w in discards]}"
    weights = np.array([Descriptors.MolWt(m) for m in molecules])
    mn = weights.mean()
    assert 850 <= mn <= 1150, f"Mn {mn:.0f} off the 1000 outer target"
    for i, mol in enumerate(molecules):
        chlorine_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "Cl")
        assert chlorine_count == 1, f"chain {i}: expected exactly 1 outer Cl cap, found {chlorine_count}"


def test_so_repeat_unit_keeps_outer_end_groups():
    """SO-as-repeat-unit strings keep the OUTER declared end group: the Cl cap
    rides a junction bond parked in the inner instance's bucket, which a
    single-bucket termination scan silently dropped on ~90% of chains. Pinned
    with visible-atom caps because [H] caps hide the loss behind implicit
    hydrogens."""
    smi = "{[] [<]CC[>], [<]{[>] [<]NN([>1])N[>];; [<1]Br [<]}|poisson(400)|[>]; O[>]; [<]Cl []}|gauss(2000,300)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    for seed in SEEDS:
        _reset_rngs(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mol_graph = ensemble_creator.sample_mol_graph()
        mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
        Chem.SanitizeMol(mol)
        chlorine_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "Cl")
        bromine_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "Br")
        assert chlorine_count == 1, f"seed {seed}: expected exactly 1 outer Cl cap, found {chlorine_count}"
        assert bromine_count >= 1, f"seed {seed}: expected global-level Br caps, found none"


def test_multifunctional_initiator_grown_arms_get_caps():
    """Both grown arms of a difunctional initiator end in the declared [H]
    cap, counted as explicit H NODES in the mol graph (SMILES and MolWt are
    blind to a lost cap: it is one implicit hydrogen), even when an arm's
    terminal bond sits parked in a terminated nested instance's bucket at
    root termination, and even when it reaches the root through a
    transition-conversion hand-off — that path used to rebuild the converted
    copy without its termination modes, shedding the cap permanently (seed 14
    was the last such loss). Chains whose initiator port never grew are
    skipped: initiator ports carry no termination edges, which is the one
    remaining gap."""
    smi = "{[] [<]PP[>], [<]{[>] [<]{[>] [<]CC[>], [<]{[>] [<]NN[>];; [<]}|poisson(100)|[>];; [<]}|poisson(300)|[>], [<]OO[>]; ;[<]}|poisson(1000)|[>]; O([>])[>]; [<][H] []}|poisson(4000)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    unit_labels = g2rins.derive_unit_labels(ensemble_creator._generative_graph).unit_id
    central = [n for n, d in ensemble_creator._generative_graph.nodes(data=True) if d.get("atomic_num") == 8 and unit_labels[n] == "I0"]
    assert len(central) == 1, f"expected one difunctional initiator oxygen, found {len(central)}"
    central_origin = str(central[0])
    for seed in range(30):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mol_graph = ensemble_creator.sample_mol_graph(rng=np.random.default_rng(seed))
        central_nodes = [n for n, d in mol_graph.nodes(data=True) if str(d.get("origin_idx")) == central_origin]
        assert len(central_nodes) == 1
        if mol_graph.degree(central_nodes[0]) < 2:
            continue
        h_caps = sum(1 for _node, data in mol_graph.nodes(data=True) if data.get("atomic_num") == 1)
        assert h_caps == 2, f"seed {seed}: both arms grew but only {h_caps} [H] cap(s) landed"


def test_side_port_caps_via_terminal_bond_connector_lists():
    """Terminal bond connector lists ([<]|[<1] inside, [>]|[>1] outside)
    expose an inner unit's side port to the outer level, where the
    outer-declared [<1]F caps it as a post-polymerization modification:
    every NN unit must carry exactly one F, on top of the ordinary Cl/Br
    chain ends, and the F mass rides on top of the level targets. The caps
    travel on bonds whose only mode is outer-level termination — the
    termination sweep used to destroy those silently when the inner block
    finalized (0 caps ever delivered at 399c8d2)."""
    smi = "{[] [<]CC[>], [<]{[>] [<]NN([>1])[>];; [<]|[<1]}|poisson(100)|[>]|[>1]; [>]Cl; [<]Br, [<1]F []}|poisson(400)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
    for seed in SEEDS:
        _reset_rngs(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mol_graph = ensemble_creator.sample_mol_graph()
        mol = g2rins.mol_graph_to_rdkit_mol(mol_graph)
        Chem.SanitizeMol(mol)
        counts = {symbol: sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == symbol) for symbol in ("N", "F", "Cl", "Br")}
        assert counts["N"] > 0 and counts["N"] % 2 == 0, f"seed {seed}: expected NN units, found {counts['N']} nitrogens"
        assert counts["F"] == counts["N"] // 2, f"seed {seed}: {counts['N'] // 2} NN side ports but {counts['F']} F cap(s)"
        assert counts["Cl"] == 1, f"seed {seed}: expected exactly 1 outer Cl cap, found {counts['Cl']}"
        assert counts["Br"] == 1, f"seed {seed}: expected exactly 1 outer Br cap, found {counts['Br']}"


def test_negative_target_draw_still_terminates():
    """A high-dispersity gauss target can draw negative; storing that draw like
    the negative no-target sentinel disables crossing checks and grows forever.
    The draw must be re-conditioned on a positive target and remain bounded."""
    smi = "{[] [<]CC[>]; C[>]; [<][H] []}|gauss(50, 500)|"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
        for seed in range(20):
            mol_graph = ensemble_creator.sample_mol_graph(rng=np.random.default_rng(seed))
            assert mol_graph.number_of_nodes() < 10_000, f"seed {seed}: runaway chain"


def test_create_ensemble_json_file(tmp_path):
    """The json_file export must be strict JSON (no NaN/numpy leakage) with the
    documented layout: {string, format, graph, ensemble}, derived unit/bond
    labels injected into the graph nodes, canonical unit entries with numbered
    P-SMILES stars, undirected bond records, chains capped by json_max_chains
    while statistics keep every sampled chain."""
    import json

    smi = "{[] [<]CC([>])c1ccccc1; CO[>]; [<][H] []}|gauss(1000, 45)|"
    path = tmp_path / "ensemble.json"
    _reset_rngs(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_creator = g2rins.G2rins.make(smi).get_graph_creator().get_ensemble_creator()
        chains = ensemble_creator.create_ensemble(3, output_format="smiles", json_file=str(path), json_max_chains=2)

    # json_file alone keeps the plain return shape.
    assert isinstance(chains, list) and len(chains) == 3

    def _reject_constant(name):
        raise AssertionError(f"non-strict JSON constant {name} in export")

    data = json.loads(path.read_text(), parse_constant=_reject_constant)

    assert list(data) == ["string", "format", "graph", "ensemble"]
    # The string is the regenerated canonical text, not the verbatim input.
    assert data["string"].startswith("{[] [<]CC([>])c1ccccc1;")
    assert data["format"]["version"] == 1
    assert data["format"]["derived_node_fields"] == ["unit_id", "bond_id"]

    labels = g2rins.derive_unit_labels(ensemble_creator._generative_graph)
    for node_dict in data["graph"]["nodes"]:
        assert node_dict["unit_id"] == labels.unit_id[node_dict["id"]]
        assert node_dict.get("bond_id") == labels.bond_id.get(node_dict["id"])

    ensemble = data["ensemble"]
    assert list(ensemble) == ["units", "chains", "bonds", "mol_weights", "distributions", "sequences"]

    assert len(ensemble["chains"]) == 2, "json_max_chains caps the stored chains"
    assert ensemble["chains"] == chains[:2]
    # Statistics always cover ALL sampled chains (one gauss instance per chain).
    assert all(len(mw_list) == 3 for mw_list in ensemble["mol_weights"].values())
    assert len(ensemble["sequences"]) == 3

    assert set(ensemble["units"]) == {"I0", "R0", "T0"}
    for unit_id, info in ensemble["units"].items():
        assert list(info) == ["psmiles", "g2rins", "frequency"]
        assert "[*:1]" in info["psmiles"]
        assert info["g2rins"] and info["frequency"] > 0
    assert "[*:2]" in ensemble["units"]["R0"]["psmiles"], "repeat unit carries two numbered stars"

    for record in ensemble["bonds"]:
        assert list(record) == ["between", "count"] and record["count"] > 0
        for endpoint in record["between"]:
            unit_id, bond_id = endpoint.rsplit(".", 1)
            assert unit_id in ensemble["units"] and int(bond_id) >= 1
    linkages = {tuple(record["between"]) for record in ensemble["bonds"]}
    assert ("I0.1", "R0.1") in linkages and ("R0.2", "T0.1") in linkages


def test_generative_graph_and_ensemble_share_node_ids():
    """Node ids are UUIDs minted per GraphCreator, so a generative graph and an
    ensemble only correspond when both come from ONE held creator (or when the
    ensemble creator is built directly from the graph in hand)."""
    g2rins_object = g2rins.G2rins.make("{[] [<]CC([>])c1ccccc1; [>][H]; [<][H] []}|gauss(1000, 45)|")

    graph_creator = g2rins_object.get_graph_creator()
    generative_graph = graph_creator.get_generative_graph()
    ensemble_creator = graph_creator.get_ensemble_creator()

    molecule = ensemble_creator.sample_mol_graph(rng=np.random.default_rng(0))
    origins = {str(data["origin_idx"]) for _node, data in molecule.nodes(data=True) if "origin_idx" in data}

    assert origins, "sampled atoms must carry origin_idx back to the generative graph"
    assert origins <= {str(node) for node in generative_graph.nodes}
    assert origins <= {str(node) for node in ensemble_creator.generative_graph.nodes}

    # By-construction path: an EnsembleCreator built from the graph in hand shares its ids.
    direct = g2rins.EnsembleCreator(generative_graph)
    direct_molecule = direct.sample_mol_graph(rng=np.random.default_rng(0))
    direct_origins = {str(data["origin_idx"]) for _node, data in direct_molecule.nodes(data=True) if "origin_idx" in data}
    assert direct_origins <= {str(node) for node in generative_graph.nodes}
