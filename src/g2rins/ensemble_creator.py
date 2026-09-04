# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import concurrent.futures
import copy
import functools
import json
import multiprocessing.spawn
import os
import pickle
import warnings
from collections import Counter, OrderedDict, deque
from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx
import numpy as np
from rdkit import Chem, rdBase

from .nx_rdkit_mol import mol_graph_to_rdkit_mol, mol_graph_to_smiles, rdkit_mol_to_smiles
from .chem_resource import (
    atom_color_mapping,
    atom_name_mapping,
    atomic_masses,
    smi_bond_mapping,
)
from .distribution import StochasticDistribution
from .exception import (
    AllZeroSamplingWeights,
    DeadSamplingPath,
    DiscardedSamplingPaths,
    EmptyTruncatedDistributionSupport,
    ForcedOvershootNoBoundary,
    IncompatibleGenerativeGraphSchema,
    IncompleteStochasticGeneration,
    InvalidGenerationSource,
    InvalidUnitPSmiles,
    NoValidGenerationSource,
    PossibleNonRepresentativePolymerChain,
    TooManyDiscardedChains,
    UndershootSnapshotMissed,
    UnvalidatedGenerationSource,
)
from .generative_graph import (
    _AROMATIC_NAME,
    _BOND_TYPE_NAME,
    _EDGE_STOCHASTIC_ID_NAME,
    _NON_STATIC_ATTR,
    _PROPAGATION_NAME,
    _TERMINATION_NAME,
    _TRANSITION_NAME,
    derive_unit_labels,
    generative_graph_json_data,
)
from .util import _determine_darkness_from_hex, get_global_rng

# Lazy-snapshot tuning. The sample loop only deepcopies + terminates the partial
# graph (for the undershoot reference used in stochastic MW rounding) when the
# NEXT growth step could cross the active SO's target MW. The lookahead is
# adaptive — an instance's first activation always snapshots, afterwards this
# margin times the largest observed per-activation mass gain is used. A missed
# crossing falls back to overshoot and warns UndershootSnapshotMissed.
# (A previous fixed floor of 0.25x the target MW kept a whole-molecule deepcopy
# running on most iterations of the last quarter of every chain.)
_LOOKAHEAD_MARGIN = 3.0

# Debug hook: when set to a list, sample_mol_graph appends one dict per
# crossing / retire / finalize decision. Diagnostic only, no runtime cost
# when None.
_DECISION_TRACE = None


def _normalized_probabilities(weights, context: str):
    """Normalize a draw or report its all-zero total as a domain error.

    The attempt tracker decides whether the failed decision is fatal or belongs
    to a conditionally reached path.
    """
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if not total > 0:
        raise AllZeroSamplingWeights(context)
    return weights / total


@functools.lru_cache(maxsize=None)
def _rdkit_implicit_hydrogens(atomic_num: int, charge: int, occupied_valence: int, aromatic: bool) -> int:
    """Hydrogens RDKit assigns an atom of this element/charge/aromaticity
    whose existing bond orders sum to occupied_valence.

    RDKit's implicit-H rule has no simple closed form (multivalent elements
    climb allowed-valence tiers, charges partly behave isoelectronically —
    N+ fills like C, Cl+ like S — but S- climbs its own tiers, metals get
    none, and AROMATIC atoms never climb past their default valence: thiophene
    sulfur at ring valence 3 gets zero H where non-aromatic S(3) gets one);
    every arithmetic approximation tried here broke on some element, so ask
    RDKit itself and cache per (element, charge, valence, aromatic) — the key
    space is tiny and the sample loop then pays a dict hit.

    The occupied valence is expressed as an explicit-H stand-in on a lone
    probe atom (RDKit defines valence as bond-order sum plus specified H
    count), because a scaffold with real bonds cannot carry the aromatic
    flag: a lone aromatic atom outside a ring fails SanitizeMol's
    kekulization, while UpdatePropertyCache(strict=False) runs exactly the
    valence/implicit-H perception the sanitized molecule ends up with
    (verified atom-by-atom against sanitized aromatic ring molecules).
    """
    atom = Chem.Atom(atomic_num)
    atom.SetFormalCharge(charge)
    atom.SetIsAromatic(aromatic)
    atom.SetNumExplicitHs(occupied_valence)  # valence stand-in, not real hydrogens
    probe = Chem.RWMol()
    probe_atom = probe.GetAtomWithIdx(probe.AddAtom(atom))
    with rdBase.BlockLogs():
        probe_atom.UpdatePropertyCache(strict=False)  # strict=False: over-valent input yields 0, not a throw
    # NOT GetTotalNumHs(): that would count the stand-in explicit Hs.
    return int(probe_atom.GetNumImplicitHs())


def _infer_hydrogen_count(atomic_num: int, charge, total_bond: int, num_explicit_h=-1, aromatic=False) -> int:
    """Number of hydrogens completing an atom's valence for MW tracking.

    A bracket atom that wrote its H count (num_explicit_h >= 0; aromatic-only
    by graph construction) fixes the count exactly. Everything else must match
    the molecule RDKit builds, so the count is delegated to RDKit (cached).
    A single default valence per element was wrong for multivalent elements:
    chem_resource stores phosphorus as 5, so P+ tracked six hydrogens where
    the RDKit molecule realizes four (PH4+), and neutral phosphines gained two
    phantom hydrogens. Aromaticity must be part of the question: an aromatic
    sulfur at ring valence 3 binds no hydrogen (thiophene), while delegating
    without the flag let it climb to the tetravalent tier and credited every
    thiophene ring a phantom hydrogen.
    """
    if num_explicit_h is not None and num_explicit_h >= 0:
        return int(num_explicit_h)
    # Missing/NaN charge (unparsed) counts as neutral.
    charge = int(charge) if charge is not None and np.isfinite(charge) else 0
    return _rdkit_implicit_hydrogens(int(atomic_num), charge, int(total_bond), bool(aromatic))

def _detach_tracebacks(error):
    """Drop traceback frames from ``error`` and its cause/context chain.

    A retained discard cause would otherwise pin the failed attempt's whole
    sampling state (partial graphs and deepcopied checkpoints) in memory for
    the remaining retries.
    """
    stack = [error]
    seen = set()
    while stack:
        exc = stack.pop()
        if exc is None or id(exc) in seen:
            continue
        seen.add(id(exc))
        exc.__traceback__ = None
        stack.append(exc.__cause__)
        stack.append(exc.__context__)
    return error


class _HalfAtomBond:
    def __init__(self, atom_idx: int, node_idx: str, graph, stochastic_tracker, rng):
        self.atom_idx: int = atom_idx
        self.node_idx: str = node_idx
        self.weight: float = graph.nodes[node_idx]["gen_weight"]
        self.molar_amounts: float = graph.nodes[node_idx]["unit_molar_amounts"]
        self.gen_hierarchy: int = graph.nodes[node_idx]["gen_hierarchy"]
        self.stochastic_id: int = graph.nodes[node_idx]["stochastic_id_tree"][0]
        self.parent: int = graph.nodes[node_idx]["stochastic_id_tree"][1]
        self._graph = graph

        self._mode_attr_map = {}
        self._mode_target_map = {}
        self._mode_target_molar_amounts_map = {}

        self._special_target = None
        special_target_list = []
        special_target_weight = []
        special_target_molar_amounts = []

        for u, v, d in graph.out_edges(node_idx, data=True):
            if not d["static"]:
                for k in _NON_STATIC_ATTR:
                    if d[k] > 0:
                        try:
                            self._mode_attr_map[k] += [d]
                        except KeyError:
                            self._mode_attr_map[k] = [d]
                        try:
                            self._mode_target_map[k] += [v]
                        except KeyError:
                            self._mode_target_map[k] = [v]
                        try:
                            self._mode_target_molar_amounts_map[k] += [graph.nodes[v]["unit_molar_amounts"]]
                        except KeyError:
                            self._mode_target_molar_amounts_map[k] = [graph.nodes[v]["unit_molar_amounts"]]

                if d[_TRANSITION_NAME] > 0:
                    target_stochastic_id = graph.nodes[v]["stochastic_id_tree"][0]
                    target_parents_stochastic_id = graph.nodes[v]["stochastic_id_tree"][1:]
                    if (self.stochastic_id in target_parents_stochastic_id and
                            d.get(_EDGE_STOCHASTIC_ID_NAME) == target_stochastic_id and target_stochastic_id !=-1):
                        special_target_list += [(v, d)]
                        special_target_weight += [d[_TRANSITION_NAME]]
                        special_target_molar_amounts += [graph.nodes[v]["unit_molar_amounts"]]

        if len(special_target_weight) > 0:
            weights = np.asarray(special_target_weight, dtype=float)
            # Weight each entry unit by its declared molar amount at its OWN
            # (managing child-SO) level — the same slot the normal transition
            # draw uses. The parent-SO slot is shared by every candidate here,
            # so it would cancel in normalization and silently ignore the
            # declared entry ratios.
            molar = np.asarray(
                [m[graph.nodes[v]["stochastic_id_tree"][0]] for (v, _d), m in zip(special_target_list, special_target_molar_amounts)],
                dtype=float,
            )
            weights *= molar
            chosen = stochastic_tracker.choose(
                rng,
                len(special_target_list),
                weights,
                "nested special-target selection",
            )
            self._special_target = special_target_list[chosen]

    def has_any_bonds(self):
        has_bonds = False
        for key in self._mode_attr_map:
            if len(self._mode_attr_map[key]) > 0:
                has_bonds = True
        return has_bonds

    def has_mode_bonds(self, mode):
        if mode not in self._mode_attr_map:
            return False
        return len(self._mode_attr_map[mode]) > 0

    @property
    def propagation_suitable(self):
        return self.has_mode_bonds(_PROPAGATION_NAME)

    def get_mode_bonds(self, mode):
        try:
            return self._mode_attr_map[mode], self._mode_target_map[mode], self._mode_target_molar_amounts_map[mode]
        except KeyError:
            return [], [], []

    def __str__(self):
        return f"HalfAtomBond({self.atom_idx}, {self.node_idx}, {self.weight}, {self._mode_attr_map}, {self._mode_target_map})"


class _StochasticObjectTracker:
    def __init__(
        self,
        generative_graph,
        rng=None,
        path_is_conditional=False,
        zero_support_is_unavoidable=False,
    ):
        self._rng = rng
        self._path_is_conditional = bool(path_is_conditional)
        # Immutable per attempt.  A conservative template analysis sets this
        # only when every allowed source is proven to reach zero support.  It
        # overrides sticky branch provenance so a branch shared by exclusively
        # dead routes cannot turn a fatal model error into futile retries.
        self._zero_support_is_unavoidable = bool(zero_support_is_unavoidable)
        # Stochastic **sto_gen_id** is the id of the stochastic object as found in the generative graph.
        # Stochastic **sto_atom_id** is the id of an instance of that particular stochastic gen id.
        # In most cases they are the same as we have exactly one instance for each stochastic object.
        # However, with nested stochastic objects that is not the case.
        # Consider a linear polymer, where each back-bone monomer has a stochastic side arm like {[] [<]CC({[<] [<]NN[>] [>]}[H])CC[>] []}
        # From the outer stochastic object we only have one instance. And every "C" has the same `sto_gen_id` and `sto_atom_id` of 0.
        # But each monomer spawns a new instance of the inner stochastic object. So every "N" has the sto_gen_id of 1, but every monomer has a different stochastic atom id and counting
        self._stochastic_gen_id_to_atom_id = {}
        self._stochastic_atom_id_to_gen_id = OrderedDict()
        self._sto_gen_id_distribution = {}
        self._sto_atom_id_actual_molw = OrderedDict()
        self._sto_atom_id_expected_molw = OrderedDict()
        self._terminated_sto_atom_ids = set()
        self.parent_map = {}
        self._parent_molw = {}

        for _node_idx, data in generative_graph.nodes(data=True):
            for index, _stochastic_vector in enumerate(data["molecular_weight_distribution"]):
                stochastic_vector = _stochastic_vector.copy()
                distribution = StochasticDistribution.from_serial_vector(stochastic_vector)
                self._register_sto_gen_id(index, distribution)
            break

    @property
    def sto_atom_id_expected_molw(self):
        return self._sto_atom_id_expected_molw

    @property
    def sto_atom_id_actual_molw(self):
        return self._sto_atom_id_actual_molw

    @property
    def path_is_conditional(self):
        return self._path_is_conditional

    @property
    def zero_support_is_unavoidable(self):
        return self._zero_support_is_unavoidable

    def mark_path_conditional(self):
        """Make later sampling dead ends eligible for chain-local rejection."""
        self._path_is_conditional = True

    def normalized_probabilities(self, weights, context, *, record_branch=True):
        """Normalize one draw and retain whether this attempt branched.

        A zero-weight decision on a provably all-dead template is always a
        fatal model error.  Otherwise, once a multi-way draw or committed
        growth makes the current state path-dependent, the same condition
        rejects only this chain.  Observational callers may disable branch
        recording while preserving the current path's classification.
        """
        try:
            probabilities = _normalized_probabilities(weights, context)
        except AllZeroSamplingWeights as error:
            if self._zero_support_is_unavoidable:
                raise
            if self._path_is_conditional:
                raise DeadSamplingPath(context) from error
            raise

        if (
            record_branch
            and not self._path_is_conditional
            and np.count_nonzero(probabilities > 0.0) > 1
        ):
            self._path_is_conditional = True
        return probabilities

    def choose(self, rng, n_candidates, weights, context):
        """Draw one index among ``n_candidates`` weighted by ``weights``."""
        probabilities = self.normalized_probabilities(weights, context)
        return rng.choice(n_candidates, p=probabilities)

    def has_sto_gen_id_unterminated_sto_ids(self, sto_gen_id: int):
        if sto_gen_id not in self._stochastic_gen_id_to_atom_id:
            return False
        found = False
        for sto_atom_id in self._stochastic_gen_id_to_atom_id[sto_gen_id]:
            if not self.is_terminated(sto_atom_id):
                found = True
                break
        return found

    def _register_sto_gen_id(self, sto_gen_id, distribution):
        self._sto_gen_id_distribution[sto_gen_id] = distribution

    def register_new_atom_instance(self, sto_gen_id, old_atom_id, parent_expected_molw=None, is_nested_parent=False):
        if sto_gen_id > 0:
            if not self._is_sto_gen_id_known(sto_gen_id):
                raise RuntimeError("You cannot register the an already known atomic instance as new. Please report on github.")

        try:
            new_sto_atom_id = max(self._stochastic_atom_id_to_gen_id) + 1
        except ValueError:
            new_sto_atom_id = 0

        self._stochastic_atom_id_to_gen_id[new_sto_atom_id] = sto_gen_id
        try:
            self._stochastic_gen_id_to_atom_id[sto_gen_id].add(new_sto_atom_id)
        except KeyError:
            self._stochastic_gen_id_to_atom_id[sto_gen_id] = {new_sto_atom_id}

        if sto_gen_id >= 0:
            if parent_expected_molw is not None:
                try:
                    new_molw = self._sto_gen_id_distribution[sto_gen_id].draw_mw(self._rng, lower=1.0, upper=parent_expected_molw)
                except EmptyTruncatedDistributionSupport as error:
                    if self._zero_support_is_unavoidable:
                        # Mirror normalized_probabilities: on a provably
                        # all-dead template the empty support is a model
                        # error, not per-chain budget luck.
                        raise AllZeroSamplingWeights(
                            "nested molecular-weight draw (empty truncated support)"
                        ) from error
                    raise
            else:
                new_molw = self._sto_gen_id_distribution[sto_gen_id].draw_mw(self._rng)
                if new_molw < 0:
                    # A negative target is meaningless and, stored as-is, would
                    # collide with the -1 "no real target" sentinel in the
                    # crossing detector so the chain would grow without a
                    # termination check; redraw conditioned on a positive
                    # target. An exact 0 stays: it is a valid lower-bound
                    # target (e.g. uniform(0,0), poisson) and terminates at the
                    # earliest molecular boundary the architecture can form.
                    new_molw = self._sto_gen_id_distribution[sto_gen_id].draw_mw(self._rng, lower=1.0)
            self._sto_atom_id_expected_molw[new_sto_atom_id] = new_molw
        else:
            self._sto_atom_id_expected_molw[new_sto_atom_id] = -1
        try:
            self._sto_atom_id_actual_molw[new_sto_atom_id] = self._parent_molw[sto_gen_id]
        except KeyError:
            self._sto_atom_id_actual_molw[new_sto_atom_id] = 0

        if is_nested_parent:
            self.parent_map[new_sto_atom_id] = [old_atom_id]

        return new_sto_atom_id

    def register_parent_atom_instances(self, sto_gen_id, old_atom_id: int, parent_sto_id_list: list[int], reuse_existing: bool = True):
        # reuse_existing=False registers a fresh instance chain even when
        # unterminated instances of the same gen ids exist (used by the -1
        # global transitions, whose arms are independent by design).
        parent_list = []

        for parent_sto_id in reversed(parent_sto_id_list):
            if parent_sto_id >= 0:
                parent_atom_id = None
                if reuse_existing and parent_sto_id in self._stochastic_gen_id_to_atom_id:
                    for existing_parent_atom_id in self._stochastic_gen_id_to_atom_id[parent_sto_id]:
                        if not self.is_terminated(existing_parent_atom_id):
                            parent_atom_id = existing_parent_atom_id
                            break
                if parent_atom_id is None:
                    if len(parent_list) > 0:
                        parent_expected_molw = self._sto_atom_id_expected_molw.get(parent_list[len(parent_list) - 1])
                        parent_atom_id = self.register_new_atom_instance(parent_sto_id, old_atom_id, parent_expected_molw, False)
                    else:
                        parent_atom_id = self.register_new_atom_instance(parent_sto_id, old_atom_id, None, False)
                parent_list.append(parent_atom_id)

        new_sto_atom_id = None

        if reuse_existing and sto_gen_id in self._stochastic_gen_id_to_atom_id:
            for a in self._stochastic_gen_id_to_atom_id[sto_gen_id]:
                if not self.is_terminated(a):
                    new_sto_atom_id = a
                    break
        if new_sto_atom_id is None:
            if len(parent_list) > 0:
                parent_expected_molw = self._sto_atom_id_expected_molw.get(parent_list[len(parent_list) - 1])
                new_sto_atom_id = self.register_new_atom_instance(sto_gen_id, old_atom_id, parent_expected_molw, False)
            else:
                new_sto_atom_id = self.register_new_atom_instance(sto_gen_id, old_atom_id, None, False)
        if len(parent_list) > 0:
            self.parent_map[new_sto_atom_id] = parent_list
            # Ancestors materialized implicitly above (growth entering a deeply
            # nested unit directly) need their own chains recorded: without
            # them add_molw never credits the outer levels and
            # pending-termination finalization treats the intermediate as
            # parentless, firing its continuation at the wrong level.
            for k, ancestor_id in enumerate(parent_list):
                if k > 0 and ancestor_id not in self.parent_map:
                    self.parent_map[ancestor_id] = parent_list[:k]

        return new_sto_atom_id, parent_list

    def add_molw(self, sto_atom_id, atomic_num, total_atom_bonds, stochastic_id_tree, num_explicit_h=-1, charge=0, aromatic=False):
        # A bracket atom that wrote its H count (e.g. [nH]) fixes num_H exactly;
        # other atoms infer it from the charge- and aromaticity-aware valence so
        # the tracked MW matches the RDKit molecule ([NH3+] binds more hydrogens
        # than the neutral valence implies, thiophene sulfur binds none).
        num_H = _infer_hydrogen_count(atomic_num, charge, total_atom_bonds, num_explicit_h, aromatic)
        added_mass = atomic_masses[atomic_num] + num_H * atomic_masses.get(1)
        self._sto_atom_id_actual_molw[sto_atom_id] += added_mass

        # Ancestors are credited the exact same mass as the instance itself: an
        # asymmetric (unclamped) hydrogen term here subtracted phantom mass from
        # every ancestor of over-coordinated atoms, so parents overshot their
        # target before should_terminate fired.
        if self.parent_map.get(sto_atom_id):
            for parent_sto_atom_id in self.parent_map.get(sto_atom_id):
                self._sto_atom_id_actual_molw[parent_sto_atom_id] += added_mass
        return num_H

    def credit_hydrogen_delta(self, sto_atom_id, delta_h):
        """Adjust an instance's tracked mass when a realized bond changes an
        atom's hydrogen count (owner and ancestors move together, mirroring
        add_molw)."""
        if not delta_h:
            return
        mass = delta_h * atomic_masses.get(1)
        self._sto_atom_id_actual_molw[sto_atom_id] += mass
        for parent_sto_atom_id in self.parent_map.get(sto_atom_id, []):
            self._sto_atom_id_actual_molw[parent_sto_atom_id] += mass

    def should_terminate(self, sto_atom_id, avg_termination_weight=0.0):
        condition = self._sto_atom_id_actual_molw[sto_atom_id] >= self._sto_atom_id_expected_molw[sto_atom_id] - avg_termination_weight
        return condition
        # return self.add_molw(sto_atom_id, 0, 0, 0, None)

    def _is_sto_gen_id_known(self, sto_gen_id):
        return sto_gen_id in self._sto_gen_id_distribution

    def is_terminated(self, sto_atom_id):
        if sto_atom_id not in self._stochastic_atom_id_to_gen_id:
            raise ValueError("Unknown atom id. it cannot be terminated")
        return sto_atom_id in self._terminated_sto_atom_ids

    def terminate(self, sto_atom_id):
        if self.is_terminated(sto_atom_id):
            raise RuntimeError("You cannot terminate an already terminated stochastic ID. This is a bug, please report on github.")
        try:
            self._parent_molw[self._stochastic_atom_id_to_gen_id[sto_atom_id]] = 0
        except KeyError:
            pass
        self._terminated_sto_atom_ids.add(sto_atom_id)

    def draw_mw(self, sto_gen_id, sto_atom_id=None, rng=None) -> None | float:
        if sto_gen_id is None:
            sto_gen_id = self._stochastic_atom_id_to_sto_gen_id[sto_atom_id]

        return self._sto_gen_id_distribution[sto_gen_id].draw_mw(rng)

    def get_unterminated_sto_atom_ids(self):
        unterminated_sto_atom_ids = []

        for sto_atom_id in reversed(self._stochastic_atom_id_to_gen_id):
            if sto_atom_id not in self._terminated_sto_atom_ids:
                unterminated_sto_atom_ids += [sto_atom_id]

        return unterminated_sto_atom_ids


@dataclass
class EnsembleData:
    """
    Full result of :meth:`EnsembleCreator.create_ensemble` with ``ensemble_info=True``.

    ``chains`` and ``sequences`` follow the requested ``output_format``;
    the ensemble aggregates are format-independent: ``units`` maps each
    derived unit_id (see :func:`g2rins.derive_unit_labels`) to
    ``{"psmiles", "g2rins", "frequency"}`` and ``bonds`` is a list of
    undirected linkage records ``{"between": ["I0.1", "R0.1"], "count": n}``
    whose endpoints are ``"<unit_id>.<bond_id>"`` strings (parse with
    ``endpoint.rsplit(".", 1)``), sorted so the same linkage always prints
    identically.
    """

    chains: list
    units: dict
    bonds: list
    sequences: list
    mol_weights: dict
    distributions: dict


def _bond_endpoint_sort_key(endpoint):
    unit_id, bond_id = endpoint.rsplit(".", 1)
    return unit_id[0], int(unit_id[1:]), int(bond_id)


def _bond_records(bond_counts, origin_endpoint):
    """
    Undirected linkage records from the growth-direction ``bond_counts``
    (origin_idx pairs): both orientations of a chemical linkage merge into one
    record with summed counts, endpoints sorted by (unit role, unit number,
    bond id).
    """
    merged = {}
    for (origin_u, origin_v), count in bond_counts.items():
        pair = tuple(sorted((origin_endpoint[origin_u], origin_endpoint[origin_v]), key=_bond_endpoint_sort_key))
        merged[pair] = merged.get(pair, 0) + count
    return [
        {"between": list(pair), "count": count}
        for pair, count in sorted(merged.items(), key=lambda item: tuple(_bond_endpoint_sort_key(endpoint) for endpoint in item[0]))
    ]


@contextmanager
def _no_main_reimport():
    # Spawned workers only re-import the caller's script so they can unpickle
    # script-defined objects; this pool ships none, so skip the re-import
    # (it is what makes unguarded Windows scripts recursively re-spawn).
    # The patch is process-global for the pool's lifetime and not reentrant:
    # anything else spawning workers in that window also skips its re-import.
    orig = multiprocessing.spawn.get_preparation_data

    def patched(name):
        data = orig(name)
        data.pop("init_main_from_path", None)
        data.pop("init_main_from_name", None)
        return data

    multiprocessing.spawn.get_preparation_data = patched
    try:
        yield
    finally:
        multiprocessing.spawn.get_preparation_data = orig


def _attempt_chain(atom_graph, collect_info, termination_flag, rng):
    """One sampling attempt, classified for the discard/retry contract shared
    by the serial loop and the parallel workers.

    Returns ``(sample, reasons, cause, deferred_warnings)``: the raw
    sample_mol_graph result (``None`` on a retryable discard), the
    discard-reason names, the rejecting :class:`DeadSamplingPath` (``None``
    for warning-only truncations; fatal errors transition instead), and the
    non-discard warnings for the caller to surface in its own process.
    """
    reasons = set()
    cause = None
    sample = None
    deferred_warnings = []
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            if collect_info:
                sample = atom_graph.sample_mol_graph(termination_flag=termination_flag, molecule_info=True, rng=rng)
            else:
                sample = atom_graph.sample_mol_graph(termination_flag=termination_flag, rng=rng)
        for caught in caught_warnings:
            if issubclass(caught.category, PossibleNonRepresentativePolymerChain):
                # Truncated chain (sampler ran out of growth/transition moves
                # below the target MW): discard it instead of keeping it.
                sample = None
                reasons.add(caught.category.__name__)
            else:
                deferred_warnings.append(caught)
    except DeadSamplingPath as error:
        # The path depends on an earlier choice or realized growth, so
        # reject only this attempt. Fatal model/input errors transition.
        reasons.add(type(error).__name__)
        cause = error
    return sample, reasons, cause, deferred_warnings


def _convert_chain(sample, molecule_format, collect_info):
    """Convert one accepted sample_mol_graph result into a chain record in the
    requested output format."""
    if collect_info:
        mol_graph, molecule_units, bonds, sequences, mol_weights, distributions = sample
    else:
        mol_graph = sample
        molecule_units = bonds = sequences = mol_weights = distributions = None

    if molecule_format == "mol":
        molecule = mol_graph_to_rdkit_mol(mol_graph)
    elif molecule_format == "smiles":
        molecule = mol_graph_to_smiles(mol_graph)
    else:
        molecule = mol_graph

    converted_sequences = None
    if collect_info:
        # Units are static-connected fragments with dangling inter-unit
        # valences, so convert them with kekulize=False (an aromatic ring
        # at a connection point can't be kekulized in isolation).
        if molecule_format == "mol":
            converted_sequences = [[mol_graph_to_rdkit_mol(unit, kekulize=False) for unit in sequence] for sequence in sequences]
        elif molecule_format == "smiles":
            converted_sequences = [[mol_graph_to_smiles(unit, kekulize=False) for unit in sequence] for sequence in sequences]
        else:
            converted_sequences = sequences

    return {
        "molecule": molecule,
        "molecule_units": molecule_units,
        "bonds": bonds,
        "sequences": converted_sequences,
        "mol_weights": mol_weights,
        "distributions": distributions,
    }


def _portable_warning(caught):
    """Make a caught warning safe to ship across the process boundary."""
    message = caught.message
    try:
        pickle.dumps(message)
    except Exception:
        message = str(message)
    return (message, caught.category, caught.filename, caught.lineno)


def _sample_chain_batch(atom_graph, chain_jobs, molecule_format, collect_info, max_discards, termination_flag):
    """Sample a batch of chains in one worker process (module level so
    ProcessPoolExecutor can pickle it).

    ``chain_jobs`` is a list of ``(chain_index, seed_sequence)`` pairs. Each
    chain builds its own Generator once, before its retry loop, so the stream
    a chain draws from is fixed by the chain index alone (independent of
    n_workers, chunking, and the process start method) and a discarded
    attempt advances the stream instead of redrawing the same rejection.

    A chain that exhausts ``max_discards`` consecutive discards yields a
    failure entry (record=None) with its first pickled cause; the
    ensemble-level verdict belongs to the parent. Fatal errors transition.
    """
    batch = []
    for chain_index, seed_sequence in chain_jobs:
        rng = np.random.default_rng(seed_sequence)
        discards = 0
        reasons = Counter()
        first_cause = None
        deferred_warnings = []
        record = None
        while True:
            sample, attempt_reasons, cause, attempt_warnings = _attempt_chain(atom_graph, collect_info, termination_flag, rng)
            deferred_warnings.extend(attempt_warnings)
            if sample is not None:
                record = _convert_chain(sample, molecule_format, collect_info)
                break
            discards += 1
            reasons.update(attempt_reasons)
            if first_cause is None and cause is not None:
                first_cause = _detach_tracebacks(cause)
            if discards >= max_discards:
                break
        batch.append(
            {
                "chain_index": chain_index,
                "record": record,
                "discards": discards,
                "reasons": tuple(reasons.items()),
                "first_cause": first_cause,
                "warnings": [_portable_warning(caught) for caught in deferred_warnings],
            }
        )
    return batch


class _PartialAtomGraph:
    _ATOM_ATTRS = {"atomic_num", _AROMATIC_NAME, "charge", "num_explicit_h"}
    _BOND_ATTRS = {_BOND_TYPE_NAME, _AROMATIC_NAME}
    # Defaults for optional node attributes so a generative_graph built before an attribute
    # existed still yields an EnsembleCreator (required attributes stay strict).
    _ATOM_ATTR_DEFAULTS = {"num_explicit_h": -1}

    def __init__(self, generative_graph, static_graph, source_node, stochastic_tracker, sto_atom_id, rng, collect_info=True):
        self._atom_id = 0
        self.generative_graph = generative_graph
        self.static_graph = static_graph
        self.stochastic_tracker = stochastic_tracker
        # Units/bonds/sequence bookkeeping costs two subgraph deepcopies per
        # grown unit; skip it entirely unless the caller asked for the info.
        self.collect_info = collect_info

        self.atom_graph = nx.Graph()
        self._open_half_bond_map: dict[int, list[_HalfAtomBond]] = {}
        self.add_static_sub_graph(source_node, sto_atom_id, rng)

        self.bonds_idx = {}
        self.units = {}
        self.sto_instance_molw_list = {}
        self.sequence = []
        self.terminal_units = []
        self.current_connection = 0

    def __deepcopy__(self, memo):
        # Snapshots must copy the mutable molecule state, but the generating and
        # static template graphs are never mutated during sampling — share them
        # instead of deep-copying them on every snapshot (they dominated the
        # cost). The tracker (including its forked rng) is still deep-copied.
        memo[id(self.generative_graph)] = self.generative_graph
        memo[id(self.static_graph)] = self.static_graph
        new_graph = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_graph
        for key, value in self.__dict__.items():
            setattr(new_graph, key, copy.deepcopy(value, memo))
        return new_graph

    def merge(self, other, self_idx, other_idx, bond_attr):
        # relabel other idx
        remapping_dict = {idx: idx + self._atom_id for idx in other.atom_graph.nodes}
        other_graph = nx.relabel_nodes(other.atom_graph, remapping_dict, copy=True)
        other_open_half_bond_map = {}
        for stochastic_id in other._open_half_bond_map:
            for half_bond in other._open_half_bond_map[stochastic_id]:
                new_half_bond = copy.copy(half_bond)
                new_half_bond.atom_idx += self._atom_id
                try:
                    other_open_half_bond_map[stochastic_id] += [new_half_bond]
                except KeyError:
                    other_open_half_bond_map[stochastic_id] = [new_half_bond]

        other_idx += self._atom_id

        # Now we can do the actual merging
        self._atom_id += other._atom_id

        # In-place merge: relabel above guarantees disjoint node ids, so we can skip
        # nx.union (which allocates a fresh graph and re-validates non-overlap).
        self.atom_graph.add_nodes_from(other_graph.nodes(data=True))
        self.atom_graph.add_edges_from(other_graph.edges(data=True))
        self.atom_graph.add_edge(self_idx, other_idx, **bond_attr)
        self._apply_realized_bond(self_idx, other_idx, bond_attr)
        for stochastic_id in other_open_half_bond_map:
            try:
                self._open_half_bond_map[stochastic_id] += other_open_half_bond_map[stochastic_id]
            except KeyError:
                self._open_half_bond_map[stochastic_id] = other_open_half_bond_map[stochastic_id]

    def get_open_half_bonds(self, sto_atom_id: int | tuple[int] | None, prefer_parent: bool = False) -> list[_HalfAtomBond]:

        if sto_atom_id is None:
            fetch_ids: tuple[int] = tuple(self._open_half_bond_map.keys())
        elif isinstance(sto_atom_id, Sequence):
            fetch_ids: tuple[int] = tuple(sto_atom_id)
        else:
            fetch_ids: tuple[int] = tuple([int(sto_atom_id)])
        open_half_bonds: list[_HalfAtomBond] = []
        for idx in fetch_ids:
            try:
                open_half_bonds += self._open_half_bond_map[idx]
            except KeyError:
                pass
        idx = range(len(open_half_bonds))
        if prefer_parent:
            parent_idx = []
            parent_bonds = []
            for i, bond in enumerate(open_half_bonds):
                if bond.parent >= 0:
                    parent_idx += [i]
                    parent_bonds += [bond]
            if len(parent_bonds) > 0:
                open_half_bonds = parent_bonds
                idx = parent_idx
        return idx, open_half_bonds

    def _apply_realized_bond(self, u_idx, v_idx, bond_attr):
        """A junction bond just became real: charge each endpoint's occupied
        valence by the bond order and re-infer its hydrogens, crediting the
        mass delta to the atom's owning stochastic instance (and ancestors).

        A phantom endpoint (split multi-connector atom placeholder) routes
        the delta to the real atom(s) behind it — the phantom collapses into
        exactly this bond at finalization.
        """
        order = bond_attr.get(_BOND_TYPE_NAME, 1)
        graph_nodes = self.atom_graph.nodes
        for endpoint, opposite in ((u_idx, v_idx), (v_idx, u_idx)):
            for real_idx in self._real_anchors(endpoint, exclude=opposite):
                data = graph_nodes[real_idx]
                data["occupied_valence"] += order
                new_h = _infer_hydrogen_count(
                    data["atomic_num"],
                    data["charge"],
                    data["occupied_valence"],
                    data.get("num_explicit_h", -1),
                    data.get(_AROMATIC_NAME, False),
                )
                delta = new_h - data["credited_h"]
                if delta:
                    data["credited_h"] = new_h
                    self.stochastic_tracker.credit_hydrogen_delta(data["owner_sto_atom_id"], delta)

    def _real_anchors(self, atom_idx, exclude):
        """Real atom(s) a junction at atom_idx ultimately binds: the atom
        itself if real, otherwise the real atoms reached through the phantom
        chain (never crossing back over the junction toward `exclude`)."""
        graph = self.atom_graph
        if graph.nodes[atom_idx].get("atomic_num", 0) > 0:
            return [atom_idx]
        anchors = []
        seen = {atom_idx, exclude}
        queue = [atom_idx]
        while queue:
            current = queue.pop()
            for neighbor in graph.neighbors(current):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                if graph.nodes[neighbor].get("atomic_num", 0) > 0:
                    anchors.append(neighbor)
                else:
                    queue.append(neighbor)
        return anchors

    def _compute_total_bond(self, node_idx: int) -> int:
        """Occupied valence an atom has the moment its static unit is
        instantiated: the bond orders of its static edges to REAL atoms, plus
        the aromatic increment.

        Connector sites and phantom placeholders are only POTENTIAL bonds:
        they contribute when a junction actually fires (merge applies the
        hydrogen delta via _apply_realized_bond), so an unfired site keeps its
        hydrogens. Counting potential sites here credited one hydrogen too few
        per unfired site, so the tracker ran light and should_terminate let
        chains grow past their target (P1-01).
        """
        total_bond = 0
        has_aromatic = False
        template_nodes = self.generative_graph.nodes

        for _u, v, attr in self.generative_graph.out_edges(node_idx, data=True):
            if attr.get("static") and template_nodes[v].get("atomic_num", 0) > 0:
                total_bond += attr.get("bond_type", 0)
                if attr.get("aromatic"):
                    has_aromatic = True

        if has_aromatic:
            total_bond += 1

        return total_bond

    def add_static_sub_graph(self, source, sto_atom_id, rng):
        atom_key_to_gen_key = {}
        gen_key_to_atom_key = {}

        def add_node(node_idx):
            data = self.gen_node_attr_to_atom_attr(self.generative_graph.nodes[node_idx])
            self.atom_graph.add_node(self._atom_id, **(data | {"origin_idx": str(node_idx)}))
            atom_key_to_gen_key[self._atom_id] = node_idx
            gen_key_to_atom_key[node_idx] = self._atom_id
            half_bond = _HalfAtomBond(
                self._atom_id,
                node_idx,
                self.generative_graph,
                self.stochastic_tracker,
                rng,
            )

            atom_total_bond = self._compute_total_bond(node_idx)

            stochastic_id_tree = self.generative_graph.nodes[node_idx]["stochastic_id_tree"]

            credited_h = self.stochastic_tracker.add_molw(
                sto_atom_id,
                data["atomic_num"],
                atom_total_bond,
                stochastic_id_tree,
                num_explicit_h=data.get("num_explicit_h", -1),
                charge=data["charge"],
                aromatic=data.get(_AROMATIC_NAME, False),
            )
            # Runtime accounting state: realized bonds added later (merge
            # junctions, phantom collapse) re-infer this atom's hydrogens and
            # credit the delta to its owning instance.
            runtime_attrs = self.atom_graph.nodes[self._atom_id]
            runtime_attrs["owner_sto_atom_id"] = sto_atom_id
            runtime_attrs["occupied_valence"] = atom_total_bond
            runtime_attrs["credited_h"] = credited_h
            self._atom_id += 1

            if half_bond.weight > 0 and half_bond.has_any_bonds():
                try:
                    self._open_half_bond_map[sto_atom_id] += [half_bond]
                except KeyError:
                    self._open_half_bond_map[sto_atom_id] = [half_bond]

        # Initiate with first node
        add_node(source)

        edges_data_map = {}

        for u, v, k in nx.edge_dfs(self.static_graph, source=source):
            for gen_atom_idx in (u, v):
                if gen_atom_idx not in gen_key_to_atom_key:
                    add_node(gen_atom_idx)

            u_atom_idx = gen_key_to_atom_key[u]
            v_atom_idx = gen_key_to_atom_key[v]

            if (u_atom_idx, v_atom_idx) not in edges_data_map and (
                v_atom_idx,
                u_atom_idx,
            ) not in edges_data_map:
                edges_data_map[(u_atom_idx, v_atom_idx)] = self.gen_edge_attr_to_bond_attr(self.static_graph.get_edge_data(u, v, k))

        for u_atom_idx, v_atom_idx in edges_data_map:
            self.atom_graph.add_edge(u_atom_idx, v_atom_idx, **edges_data_map[(u_atom_idx, v_atom_idx)])

    def gen_node_attr_to_atom_attr(self, attr: dict[str, bool | float | int], keys_to_copy: None | set[str] = None) -> dict[str, bool | float | int]:
        if keys_to_copy is None:
            keys_to_copy = self._ATOM_ATTRS
        return self._copy_some_dict_attr(attr, keys_to_copy)

    def gen_edge_attr_to_bond_attr(self, attr: dict[str, bool | int], keys_to_copy: None | set[str] = None) -> dict[str, bool | int]:
        if keys_to_copy is None:
            keys_to_copy = self._BOND_ATTRS
        return self._copy_some_dict_attr(attr, keys_to_copy)

    @staticmethod
    def _copy_some_dict_attr(dictionary: dict[str, Any], keys_to_copy: set[str]) -> dict[str, Any]:
        new_dict = {}
        for k in keys_to_copy:
            if k in dictionary:
                new_dict[k] = dictionary[k]
            else:
                # Missing optional attr -> its default; missing required attr -> KeyError.
                new_dict[k] = _PartialAtomGraph._ATOM_ATTR_DEFAULTS[k]
        return new_dict

    def pop_target_open_half_bond(self, sto_atom_idx, target_idx) -> _HalfAtomBond:
        found_target_index = None
        try:
            for target_index, half_bond in enumerate(self._open_half_bond_map[sto_atom_idx]):
                if half_bond.node_idx == target_idx:
                    if found_target_index is not None:
                        raise RuntimeError("A matching target index was found twice, that is a bug. Please report on github.")

                    found_target_index = target_index
        except KeyError:
            pass

        if found_target_index is None:
            possible_connections = self._find_origin_to_atom(target_idx)
            if len(possible_connections) != 1:
                raise RuntimeError("There should only be one possible connection left. Please report this bug on github.")
            return possible_connections[0]

        target_half_bond = self._open_half_bond_map[sto_atom_idx].pop(found_target_index)
        return target_half_bond.atom_idx

    def _pop_random_bond(self, half_bonds, sto_atom_id, sto_gen_id, rng):
        """Pop a half-bond carrying a transition edge at level `sto_gen_id`, weighted.

        Eligibility is decided BEFORE hierarchy selection and the draw: bonds
        without an edge at the requested level (or without transition edges at
        all) are never candidates, so returning None deterministically means
        "nothing in this bucket serves this level". Deciding after a single
        weighted draw let one mismatched pick report a dead end while a
        compatible bond sat in the bucket — and the caller's truncation path
        treats that verdict as terminal for the whole molecule.
        """
        if not half_bonds:
            raise ValueError("Cannot pop from empty list")
        eligible_half_bonds = [
            half_bond
            for half_bond in half_bonds
            if any(attr.get(_EDGE_STOCHASTIC_ID_NAME) == sto_gen_id for attr in half_bond._mode_attr_map.get(_TRANSITION_NAME, []))
        ]
        if not eligible_half_bonds:
            return None, []
        max_hierarchy = max(half_bond.gen_hierarchy for half_bond in eligible_half_bonds)
        hierarchical_half_bonds = [half_bond for half_bond in eligible_half_bonds if half_bond.gen_hierarchy == max_hierarchy]
        non_used_half_bonds = []
        old_transitions = {}
        new_transitions = {}
        current_sto_id = self.stochastic_tracker._stochastic_atom_id_to_gen_id[sto_atom_id]
        for half_bond in hierarchical_half_bonds:
            node_sto_id = half_bond.stochastic_id
            all_target_attr, all_target_idx, _all_target_molar_amounts = half_bond.get_mode_bonds(_TRANSITION_NAME)
            if node_sto_id != current_sto_id:
                old_transitions[half_bond] = all_target_idx
            else:
                new_transitions[half_bond] = all_target_idx

        for half_bond, all_target_idx in old_transitions.items():
            if not any(target_idx in new_all_target_idx for new_all_target_idx in new_transitions.values() for target_idx in all_target_idx):
                non_used_half_bonds.append(half_bond)

        filtered_half_bonds = [half_bond for half_bond in hierarchical_half_bonds if half_bond not in non_used_half_bonds]
        if not filtered_half_bonds:
            # Every candidate got filtered out (e.g. only converted old-SO bonds
            # remain in the bucket): report "no transition possible" instead of
            # crashing on an empty draw.
            return None, non_used_half_bonds
        if len(filtered_half_bonds) == 1:
            # Deterministic pick: skip the draw so it consumes no RNG state.
            chosen = filtered_half_bonds[0]
        else:
            weights = [half_bond.weight for half_bond in filtered_half_bonds]
            chosen = filtered_half_bonds[
                self.stochastic_tracker.choose(
                    rng,
                    len(filtered_half_bonds),
                    weights,
                    "transition bond draw",
                )
            ]
        half_bonds.remove(chosen)

        return chosen, non_used_half_bonds

    def _get_level_termination_bonds(
        self,
        owner_sto_atom_id,
        level_sto_atom_id,
        include_transition_bonds=False,
        require_transition_bonds=False,
    ):
        """Return open bonds owned by one instance that can terminate at a
        particular stochastic-object level.

        Normal end-group attachment asks for ``owner == level`` and excludes
        continuation sites.  A parked ancestor is the one exception: a
        finished descendant's continuation site is deliberately capped at the
        ancestor's level, so callers estimating that conditional cap must be
        able to include transition-capable bonds without mutating the graph.
        """
        # This helper scans ONE bucket (the owner's); terminate_graph composes
        # it over the owner plus terminated-descendant buckets. It never scans
        # sibling buckets: consuming a bond re-added into a terminated
        # sibling's bucket would double-terminate the same half-bond site
        # (over-valent atoms). The stochastic_id filter is still required:
        # unlike main, the generative graph keeps termination edges of every
        # SO level on a node, so an unfiltered scan would fire outer-owned
        # terminators at inner level.
        termination_bonds = []
        sto_gen_id = self.stochastic_tracker._stochastic_atom_id_to_gen_id[level_sto_atom_id]
        for _, half_bond in zip(*self.get_open_half_bonds(owner_sto_atom_id), strict=False):
            if require_transition_bonds and not half_bond.has_mode_bonds(_TRANSITION_NAME):
                continue
            if not include_transition_bonds and self._transition_blocks_termination(half_bond, sto_gen_id):
                # A bond whose continuation some level still needs is not an
                # end: capping it would either kill that continuation or, once
                # re-opened, over-bond the atom. Transition at the terminating
                # level itself, or at a level with no live instance left, dies
                # with the termination, so such a bond's terminators may fire.
                continue
            if half_bond.has_mode_bonds(_TERMINATION_NAME):
                if any(attr.get(_EDGE_STOCHASTIC_ID_NAME) == sto_gen_id for attr in half_bond._mode_attr_map[_TERMINATION_NAME]):
                    termination_bonds += [half_bond]
        return termination_bonds

    def _transition_blocks_termination(self, half_bond, sto_gen_id):
        """True if capping this bond would kill a continuation another level
        still needs: any -1 (global) transition edge — ALWAYS protected,
        because a -1 arm's source instance is terminated at birth so instance
        liveness cannot vouch for it — or a transition edge at a level other
        than ``sto_gen_id`` that still has a live instance."""
        tracker = self.stochastic_tracker
        for attr in half_bond._mode_attr_map.get(_TRANSITION_NAME, []):
            level = attr.get(_EDGE_STOCHASTIC_ID_NAME)
            if level == -1:
                return True
            if level != sto_gen_id and tracker.has_sto_gen_id_unterminated_sto_ids(level):
                return True
        return False

    def _get_average_level_termination_mw(
        self,
        owner_sto_atom_id,
        level_sto_atom_id,
        static_graph,
        rng,
        include_transition_bonds=False,
        require_transition_bonds=False,
    ):
        # Estimation is observational: constructing candidate cap fragments can
        # initialize half-bonds with stochastic special targets, but merely
        # checking a boundary must not advance the sample's RNG stream.
        estimator_rng = copy.deepcopy(rng)

        def _get_terminator_atom_graph(source):
            stochastic_object_tracker = _StochasticObjectTracker(
                self.generative_graph,
                estimator_rng,
                path_is_conditional=self.stochastic_tracker.path_is_conditional,
                zero_support_is_unavoidable=(
                    self.stochastic_tracker.zero_support_is_unavoidable
                ),
            )
            source_sto_gen_id = self.generative_graph.nodes[source]["stochastic_id_tree"][0]
            term_sto_atom_id = stochastic_object_tracker.register_new_atom_instance(source_sto_gen_id, self.generative_graph.nodes[source]["stochastic_id_tree"][1], None, False)
            terminator_atom_graph = _PartialAtomGraph(
                self.generative_graph,
                static_graph,
                source,
                stochastic_object_tracker,
                term_sto_atom_id,
                estimator_rng,
            )
            del stochastic_object_tracker
            return terminator_atom_graph

        termination_bonds = self._get_level_termination_bonds(
            owner_sto_atom_id,
            level_sto_atom_id,
            include_transition_bonds=include_transition_bonds,
            require_transition_bonds=require_transition_bonds,
        )
        avg_termination_mw = 0
        # The source endpoint's hydrogen loss only depends on the attach order,
        # not on which terminator fires: share it across candidates.
        source_delta_by_order = {}

        for termination_bond in termination_bonds:
            all_attributes, all_ids, all_molar = termination_bond.get_mode_bonds(_TERMINATION_NAME)
            gen_sto_id = self.stochastic_tracker._stochastic_atom_id_to_gen_id[level_sto_atom_id]
            # Same level filter as terminate_graph: the estimate must average over
            # the terminators that termination would actually attach.
            level_indices = [i for i, attr in enumerate(all_attributes) if attr.get(_EDGE_STOCHASTIC_ID_NAME) == gen_sto_id]
            target_attributes = [all_attributes[i] for i in level_indices]
            target_ids = [all_ids[i] for i in level_indices]
            all_molar_amounts = [all_molar[i] for i in level_indices]
            molar_amounts = [molar_amount[gen_sto_id] for molar_amount in all_molar_amounts]
            target_weight = np.asarray([attr[_TERMINATION_NAME] for attr in target_attributes])
            target_molar_amount = np.asarray(molar_amounts)
            target_weight *= target_molar_amount
            target_prob = self.stochastic_tracker.normalized_probabilities(
                target_weight,
                "termination MW estimate",
                record_branch=False,
            )
            source_delta_by_order.clear()
            for i, node_id in enumerate(target_ids):
                terminator_atom_graph = _get_terminator_atom_graph(node_id)
                attach_order = target_attributes[i].get(_BOND_TYPE_NAME, 1)
                # The estimate must be the NET tracker delta the attach would
                # realize, mirroring merge/_apply_realized_bond without mutating:
                # gross cap mass overstates the margin by the hydrogens the
                # existing endpoint sheds and, for split-dummy connectors, by
                # charging the attach order to the massless dummy instead of
                # its real anchors (so their hydrogens stayed uncounted).
                extra_occupied = {}
                for frag_node, data in terminator_atom_graph.atom_graph.nodes(data=True):
                    if data["origin_idx"] == str(node_id):
                        for anchor in terminator_atom_graph._real_anchors(frag_node, exclude=None):
                            extra_occupied[anchor] = extra_occupied.get(anchor, 0) + attach_order
                        break
                terminator_weight = 0
                for frag_node, data in terminator_atom_graph.atom_graph.nodes(data=True):
                    atomic_number = data["atomic_num"]
                    if atomic_number <= 0:
                        # Phantom placeholders carry no mass and no hydrogens.
                        continue
                    occupied = self._compute_total_bond(data["origin_idx"]) + extra_occupied.get(frag_node, 0)
                    num_H = _infer_hydrogen_count(
                        atomic_number,
                        data.get("charge", 0),
                        occupied,
                        data.get("num_explicit_h", -1),
                        data.get(_AROMATIC_NAME, False),
                    )
                    terminator_weight += atomic_masses[atomic_number] + num_H * atomic_masses.get(1)
                if attach_order not in source_delta_by_order:
                    delta = 0.0
                    for anchor in self._real_anchors(termination_bond.atom_idx, exclude=None):
                        anchor_data = self.atom_graph.nodes[anchor]
                        new_h = _infer_hydrogen_count(
                            anchor_data["atomic_num"],
                            anchor_data["charge"],
                            anchor_data["occupied_valence"] + attach_order,
                            anchor_data.get("num_explicit_h", -1),
                            anchor_data.get(_AROMATIC_NAME, False),
                        )
                        delta += (new_h - anchor_data["credited_h"]) * atomic_masses.get(1)
                    source_delta_by_order[attach_order] = delta
                avg_termination_mw += target_prob[i] * (terminator_weight + source_delta_by_order[attach_order])
        return avg_termination_mw

    def get_average_termination_mw(self, sto_atom_id, static_graph, rng):
        return self._get_average_level_termination_mw(
            sto_atom_id,
            sto_atom_id,
            static_graph,
            rng,
        )

    def get_average_junction_termination_mw(
        self,
        owner_sto_atom_id,
        level_sto_atom_id,
        static_graph,
        rng,
    ):
        """Expected net mass of the cap that replaces ``owner``'s
        continuation when ``level`` has already parked."""
        return self._get_average_level_termination_mw(
            owner_sto_atom_id,
            level_sto_atom_id,
            static_graph,
            rng,
            include_transition_bonds=True,
            require_transition_bonds=True,
        )

    def terminate_graph(self, sto_atom_id, rng):
        terminated_graph = self
        tracker = self.stochastic_tracker

        def get_termination_bonds():
            # The instance's stranded end sites can sit in a terminated
            # descendant's bucket (its frontier ended inside a nested
            # instance). Composing the per-bucket scan over those buckets is
            # safe: a consumed bond is popped from its holding bucket and
            # never re-added, so nothing double-caps.
            bucket_ids = [sto_atom_id] + [
                bucket_id
                for bucket_id in terminated_graph._open_half_bond_map
                if bucket_id != sto_atom_id and tracker.is_terminated(bucket_id) and sto_atom_id in tracker.parent_map.get(bucket_id, [])
            ]
            pairs = []
            for bucket_id in bucket_ids:
                for half_bond in terminated_graph._get_level_termination_bonds(bucket_id, sto_atom_id):
                    pairs.append((bucket_id, half_bond))
            return pairs

        def pop_next_termination_bond():
            pairs = get_termination_bonds()
            termination_weight = np.asarray([hb.weight for _bucket_id, hb in pairs])
            selected_idx = self.stochastic_tracker.choose(
                rng,
                len(pairs),
                termination_weight,
                "termination bond draw",
            )
            selected_bucket_id, selected_bond = pairs[selected_idx]
            bond_list = terminated_graph._open_half_bond_map[selected_bucket_id]
            for k, bond in enumerate(bond_list):
                if bond is selected_bond:
                    bond_list.pop(k)
                    break
            return selected_bond

        own_gen_sto_id = tracker._stochastic_atom_id_to_gen_id[sto_atom_id]
        transition_half_bonds = []
        foreign_termination_half_bonds = []

        for _i, half_bond in zip(*terminated_graph.get_open_half_bonds(sto_atom_id), strict=False):
            if half_bond.has_mode_bonds(_TRANSITION_NAME):
                transition_half_bonds.append(half_bond)
            elif half_bond.has_mode_bonds(_TERMINATION_NAME) and all(
                attr.get(_EDGE_STOCHASTIC_ID_NAME) != own_gen_sto_id for attr in half_bond._mode_attr_map[_TERMINATION_NAME]
            ):
                # An ancestor level's declared cap rides this instance's
                # frontier without a transition partner (e.g. a side port
                # whose terminator models a post-polymerization modification,
                # deliberately declared outside the inner MW distribution):
                # dropping it with the wipe silently un-caps that chemistry.
                # It fires when the owning ancestor level terminates.
                foreign_termination_half_bonds.append(half_bond)

        consumed_half_bonds = []

        while len(get_termination_bonds()) > 0:
            termination_bond = pop_next_termination_bond()
            consumed_half_bonds.append(termination_bond)
            all_attributes, all_ids, all_molar = termination_bond.get_mode_bonds(_TERMINATION_NAME)

            # The bond was selected because at least one termination attr matches
            # this SO's level, but it may also carry other levels' terminators —
            # draw only among this level's edges.
            gen_sto_id = self.stochastic_tracker._stochastic_atom_id_to_gen_id[sto_atom_id]
            level_indices = [i for i, attr in enumerate(all_attributes) if attr.get(_EDGE_STOCHASTIC_ID_NAME) == gen_sto_id]
            target_attributes = [all_attributes[i] for i in level_indices]
            target_ids = [all_ids[i] for i in level_indices]
            all_target_molar_amounts = [all_molar[i] for i in level_indices]

            target_weight = np.asarray([attr[_TERMINATION_NAME] for attr in target_attributes])
            target_molar_amounts = [molar_amounts[gen_sto_id] for molar_amounts in all_target_molar_amounts]
            target_amounts = np.asarray(target_molar_amounts)
            target_weight *= target_amounts
            selected_target_idx = self.stochastic_tracker.choose(
                rng,
                len(target_weight),
                target_weight,
                "terminator target selection",
            )
            selected_target = target_ids[selected_target_idx]
            selected_attr = self.gen_edge_attr_to_bond_attr(target_attributes[selected_target_idx])

            other_partial_graph = _PartialAtomGraph(terminated_graph.generative_graph, terminated_graph.static_graph, selected_target, self.stochastic_tracker, sto_atom_id, rng)
            other_half_bond_atom_idx = other_partial_graph.pop_target_open_half_bond(sto_atom_id, selected_target)
            pre_merge_watermark = terminated_graph._atom_id
            terminated_graph.merge(
                other_partial_graph,
                termination_bond.atom_idx,
                other_half_bond_atom_idx,
                selected_attr,
            )
            last_unit = terminated_graph.add_new_unit_and_bond(pre_merge_watermark)
            terminated_graph.add_unit_to_sequence(last_unit)

        terminated_graph._open_half_bond_map[sto_atom_id] = []
        for bond in transition_half_bonds + foreign_termination_half_bonds:
            # A half-bond is one valence slot: once consumed by termination it
            # must not be re-opened for transitions (-1 or otherwise).
            if any(bond is consumed for consumed in consumed_half_bonds):
                continue
            terminated_graph._open_half_bond_map[sto_atom_id] += [bond]

        terminated_graph.stochastic_tracker.terminate(sto_atom_id)

        return terminated_graph

    def cap_junction_bonds(self, owner_sto_atom_id, level_sto_atom_id, rng):
        """Fire level_sto_atom_id-level termination edges on the owner's
        leftover open bonds, consuming each capped bond.

        A finished nested instance whose junction would continue a PARKED
        ancestor (graft-through architectures) must end the ancestor's chain
        there instead of growing it: the ancestor's under/over rounding
        already fixed its budget, and firing the transition would add units
        past the decision — or loop forever, since each new unit spawns a
        fresh nested instance that keeps the parked ancestor unfinalizable.
        _get_level_termination_bonds cannot serve here: it deliberately skips
        transition-capable bonds, whose continuation is exactly what the
        parked decision overrides. Capped mass is credited to the LEVEL
        instance (its end group), like terminate_graph would from its own
        bucket.

        The junction is not necessarily in the owner's own bucket: with three
        or more nesting levels the finished child's frontier can end inside a
        terminated descendant, whose bucket then holds the ancestor-level
        junction bond (the termination flavor of the transition rescue)."""
        tracker = self.stochastic_tracker
        gen_sto_id = tracker._stochastic_atom_id_to_gen_id[level_sto_atom_id]
        bucket_ids = [owner_sto_atom_id] + [
            bucket_id
            for bucket_id in self._open_half_bond_map
            if bucket_id != owner_sto_atom_id and tracker.is_terminated(bucket_id) and owner_sto_atom_id in tracker.parent_map.get(bucket_id, [])
        ]
        for bucket_id in bucket_ids:
            bucket = self._open_half_bond_map.get(bucket_id, [])
            for half_bond in list(bucket):
                all_attributes, all_ids, all_molar = half_bond.get_mode_bonds(_TERMINATION_NAME)
                level_indices = [i for i, attr in enumerate(all_attributes) if attr.get(_EDGE_STOCHASTIC_ID_NAME) == gen_sto_id]
                if not level_indices:
                    continue
                target_attributes = [all_attributes[i] for i in level_indices]
                target_ids = [all_ids[i] for i in level_indices]
                target_molar_amounts = [all_molar[i][gen_sto_id] for i in level_indices]

                target_weight = np.asarray([attr[_TERMINATION_NAME] for attr in target_attributes])
                target_weight = target_weight * np.asarray(target_molar_amounts)
                selected_target_idx = self.stochastic_tracker.choose(
                    rng,
                    len(target_weight),
                    target_weight,
                    "junction terminator selection",
                )
                selected_target = target_ids[selected_target_idx]
                selected_attr = self.gen_edge_attr_to_bond_attr(target_attributes[selected_target_idx])

                other_partial_graph = _PartialAtomGraph(self.generative_graph, self.static_graph, selected_target, self.stochastic_tracker, level_sto_atom_id, rng)
                other_half_bond_atom_idx = other_partial_graph.pop_target_open_half_bond(level_sto_atom_id, selected_target)
                pre_merge_watermark = self._atom_id
                self.merge(
                    other_partial_graph,
                    half_bond.atom_idx,
                    other_half_bond_atom_idx,
                    selected_attr,
                )
                last_unit = self.add_new_unit_and_bond(pre_merge_watermark)
                self.add_unit_to_sequence(last_unit)
                bucket.remove(half_bond)

    def trigger_global_transitions(self, rng):
        """Fire ONE transition edge with stochastic_id == -1 after all SO instances are terminated.

        Picks one open half-bond with a -1 transition edge (weighted by bond weight), fires it
        (consuming that bond), and returns True so the while loop can grow the new SO arm before
        the next -1 fires.  Other bonds with -1 transitions remain open for subsequent calls.
        Returns False when no -1 transition bonds remain.
        """
        candidates = []
        for sid in list(self._open_half_bond_map.keys()):
            for half_bond in self._open_half_bond_map[sid]:
                if half_bond.has_mode_bonds(_TRANSITION_NAME):
                    all_target_attr, _, _ = half_bond.get_mode_bonds(_TRANSITION_NAME)
                    if any(attr.get(_EDGE_STOCHASTIC_ID_NAME) == -1 for attr in all_target_attr):
                        candidates.append((sid, half_bond))

        if not candidates:
            return False

        # Pick one half-bond weighted by the source atom's generating weight
        weights = np.asarray([hb.weight for _, hb in candidates], dtype=float)
        chosen_idx = self.stochastic_tracker.choose(
            rng,
            len(candidates),
            weights,
            "global transition bond draw",
        )
        bucket_id, half_bond = candidates[chosen_idx]

        self._open_half_bond_map[bucket_id].remove(half_bond)

        all_target_attr, all_target_idx, all_molar_amounts = half_bond.get_mode_bonds(_TRANSITION_NAME)
        minus_one_indices = [
            i for i, attr in enumerate(all_target_attr) if attr.get(_EDGE_STOCHASTIC_ID_NAME) == -1
        ]

        target_attr = [all_target_attr[i] for i in minus_one_indices]
        target_idx = [all_target_idx[i] for i in minus_one_indices]
        target_molar_amounts = [all_molar_amounts[i] for i in minus_one_indices]

        target_weights = []
        for attr, idx, molar in zip(target_attr, target_idx, target_molar_amounts):
            w = float(attr[_TRANSITION_NAME])
            target_sto_gen_id = self.generative_graph.nodes[idx]["stochastic_id_tree"][0]
            if target_sto_gen_id >= 0:
                try:
                    w *= molar[target_sto_gen_id]
                except (KeyError, IndexError, TypeError):
                    pass
            target_weights.append(w)

        target_weights = np.asarray(target_weights, dtype=float)
        chosen = self.stochastic_tracker.choose(
            rng,
            len(target_idx),
            target_weights,
            "global transition target selection",
        )

        selected_target_idx = target_idx[chosen]
        selected_attr = self.gen_edge_attr_to_bond_attr(target_attr[chosen])
        selected_target_sto_gen_id = self.generative_graph.nodes[selected_target_idx]["stochastic_id_tree"][0]
        selected_target_sto_parent_id = self.generative_graph.nodes[selected_target_idx]["stochastic_id_tree"][1:]

        # Each -1 arm is independent: register a fresh instance chain.
        new_sto_atom_id, _parent_list = self.stochastic_tracker.register_parent_atom_instances(
            selected_target_sto_gen_id, -1, selected_target_sto_parent_id, reuse_existing=False
        )

        other_graph = _PartialAtomGraph(
            self.generative_graph, self.static_graph, selected_target_idx,
            self.stochastic_tracker, new_sto_atom_id, rng,
        )
        other_half_bond_atom_idx = other_graph.pop_target_open_half_bond(new_sto_atom_id, selected_target_idx)
        pre_merge_watermark = self._atom_id
        self.merge(other_graph, half_bond.atom_idx, other_half_bond_atom_idx, selected_attr)
        last_unit = self.add_new_unit_and_bond(pre_merge_watermark)
        self.add_unit_to_sequence(last_unit)
        self.nested_transition(new_sto_atom_id, rng)

        return True

    def _find_origin_to_atom(self, origin_idx):
        atom_id_list = []
        for node_id, data in self.atom_graph.nodes(data=True):
            if data["origin_idx"] == origin_idx:
                atom_id_list += [node_id]
        return atom_id_list

    def _find_rescue_bucket(self, sto_atom_id, sto_gen_id, mode_name):
        """Bucket of a terminated descendant of ``sto_atom_id`` that still
        holds a bond with a ``mode_name`` edge at level ``sto_gen_id``, or
        None.

        With three or more nesting levels an ancestor-level junction bond can
        sit in a terminated grandchild's bucket (the finished child's frontier
        ended inside its own nested instance), where any scan keyed on the
        finished child alone never finds it."""
        tracker = self.stochastic_tracker
        for bucket_id, bonds in self._open_half_bond_map.items():
            if bucket_id == sto_atom_id or not tracker.is_terminated(bucket_id):
                continue
            if sto_atom_id not in tracker.parent_map.get(bucket_id, []):
                continue
            for half_bond in bonds:
                if any(attr.get(_EDGE_STOCHASTIC_ID_NAME) == sto_gen_id for attr in half_bond._mode_attr_map.get(mode_name, [])):
                    return bucket_id
        return None

    def transition_graph(self, sto_atom_id, sto_gen_id, rng):
        new_sto_atom_id, success = self._transition_graph_single_bucket(sto_atom_id, sto_gen_id, rng)
        if success or sto_gen_id < 0:
            return new_sto_atom_id, success
        # A single-bucket miss is only a dead end for the whole level if no
        # terminated descendant abandoned a junction bond for it. Rescue only
        # while the level is live: the root/inter-object continuation path
        # also lands here, and rescuing a finished level would resurrect it
        # and grow the chain far past its target.
        if not self.stochastic_tracker.has_sto_gen_id_unterminated_sto_ids(sto_gen_id):
            return new_sto_atom_id, success
        rescue_bucket_id = self._find_rescue_bucket(sto_atom_id, sto_gen_id, _TRANSITION_NAME)
        if rescue_bucket_id is None:
            return new_sto_atom_id, success
        rescued_id, rescued_success = self._transition_graph_single_bucket(rescue_bucket_id, sto_gen_id, rng)
        if rescued_success:
            return rescued_id, True
        return sto_atom_id, False

    def _transition_graph_single_bucket(self, sto_atom_id, sto_gen_id, rng):
        # Early exit if no transition necessary
        if len(self.get_open_half_bonds(sto_atom_id)[0]) < 1:
            return sto_atom_id, False

        half_bonds = self._open_half_bond_map[sto_atom_id]

        transition_bond, non_used_half_bonds = self._pop_random_bond(half_bonds, sto_atom_id, sto_gen_id, rng)

        if transition_bond is None:
            # No bond in this bucket carries an edge at the requested level:
            # a deterministic dead-end verdict (the selection is level-aware,
            # so no compatible bond can have been passed over).
            return sto_atom_id, False

        # only keep target_attr and target_idx of the requested stochastic level (sto_gen_id)
        all_target_attr, all_target_idx, all_molar_amounts = transition_bond.get_mode_bonds(_TRANSITION_NAME)

        filtered_indices = [i for i, attr in enumerate(all_target_attr) if attr.get(_EDGE_STOCHASTIC_ID_NAME) == sto_gen_id]

        if not filtered_indices:
            raise RuntimeError("A popped transition bond carries no edge at the level it was selected for. This is a bug, please report on github.")

        target_attr = [all_target_attr[i] for i in filtered_indices]
        target_idx = [all_target_idx[i] for i in filtered_indices]
        all_target_amounts = [all_molar_amounts[i] for i in filtered_indices]

        target_weights = np.asarray([attr[_TRANSITION_NAME] for attr in target_attr])
        if sto_gen_id >= 0:
            target_amounts = [molar_amount[sto_gen_id] for molar_amount in all_target_amounts]
        else:
            # -1 (global) level: unit_molar_amounts has no slot for it — a negative
            # index would silently read the last SO's slot (same guard as
            # trigger_global_transitions).
            target_amounts = [1.0] * len(all_target_amounts)
        molar_amounts = np.asarray(target_amounts)
        target_weights *= molar_amounts
        target_id = self.stochastic_tracker.choose(
            rng,
            len(target_idx),
            target_weights,
            "transition target selection",
        )

        selected_target_idx = target_idx[target_id]

        selected_attr = self.gen_edge_attr_to_bond_attr(target_attr[target_id])
        selected_target_sto_gen_id = self.generative_graph.nodes[selected_target_idx]["stochastic_id_tree"][0]

        selected_target_sto_parent_id = self.generative_graph.nodes[selected_target_idx]["stochastic_id_tree"][1:]

        new_sto_atom_id, parent_list = self.stochastic_tracker.register_parent_atom_instances(selected_target_sto_gen_id, sto_atom_id, selected_target_sto_parent_id)

        # Transfer non-used transition bonds to new stochastic bonds

        list_of_new_bonds = []
        list_of_bond_idx_to_delete = []

        for j, half_bond in enumerate(half_bonds):
            prop_attr_list = half_bond._mode_attr_map.get(_TRANSITION_NAME, [])
            fired_level_indices = [i for i, attr in enumerate(prop_attr_list) if attr.get(_EDGE_STOCHASTIC_ID_NAME) == sto_gen_id]
            if prop_attr_list and not fired_level_indices:
                # Transition edges only at OTHER levels (-1 continuations for
                # trigger_global_transitions, or a level a later call must be
                # able to fire): leave the bond untouched in this bucket.
                # Sweeping it into the delete/convert below handed those
                # continuations a mode-less copy and silently truncated them.
                continue

            # Shallow copy: all three mode maps are rebound below, the other
            # attributes are immutable, and deepcopy dragged a full copy of the
            # generative graph along via half_bond._graph.
            new_stochastic_bond = copy.copy(half_bond)
            new_stochastic_bond._mode_attr_map = {}
            new_stochastic_bond._mode_target_map = {}
            new_stochastic_bond._mode_target_molar_amounts_map = {}

            if fired_level_indices:
                list_of_bond_idx_to_delete.append(j)
                if half_bond.gen_hierarchy == transition_bond.gen_hierarchy and half_bond not in non_used_half_bonds:
                    bond_attr_list = copy.deepcopy([prop_attr_list[i] for i in fired_level_indices])
                    for bond_attr in bond_attr_list:
                        bond_attr[_PROPAGATION_NAME] = bond_attr[_TRANSITION_NAME]
                        bond_attr[_TRANSITION_NAME] = 0
                    new_stochastic_bond._mode_attr_map[_PROPAGATION_NAME] = bond_attr_list
                    new_stochastic_bond._mode_target_map[_PROPAGATION_NAME] = [half_bond._mode_target_map[_TRANSITION_NAME][i] for i in fired_level_indices]
                    new_stochastic_bond._mode_target_molar_amounts_map[_PROPAGATION_NAME] = [half_bond._mode_target_molar_amounts_map[_TRANSITION_NAME][i] for i in fired_level_indices]
                    retained_indices = [i for i in range(len(prop_attr_list)) if i not in fired_level_indices]
                    if retained_indices:
                        # A mixed bond also carries other-level transition
                        # edges: keep them as transition modes on the
                        # transferred copy so their continuations survive for
                        # their own level's call.
                        new_stochastic_bond._mode_attr_map[_TRANSITION_NAME] = [prop_attr_list[i] for i in retained_indices]
                        new_stochastic_bond._mode_target_map[_TRANSITION_NAME] = [half_bond._mode_target_map[_TRANSITION_NAME][i] for i in retained_indices]
                        new_stochastic_bond._mode_target_molar_amounts_map[_TRANSITION_NAME] = [half_bond._mode_target_molar_amounts_map[_TRANSITION_NAME][i] for i in retained_indices]

            new_stochastic_bond.parent = parent_list[len(parent_list)-1] if parent_list else -2
            list_of_new_bonds.append(new_stochastic_bond)

        self._open_half_bond_map[sto_atom_id] = [bond for j, bond in enumerate(half_bonds) if j not in list_of_bond_idx_to_delete]

        for bond in list_of_new_bonds:
            try:
                self._open_half_bond_map[new_sto_atom_id] += [bond]
            except KeyError:
                self._open_half_bond_map[new_sto_atom_id] = [bond]

        other_graph = _PartialAtomGraph(self.generative_graph, self.static_graph, selected_target_idx, self.stochastic_tracker, new_sto_atom_id, rng)

        other_target_idx = other_graph.pop_target_open_half_bond(new_sto_atom_id, selected_target_idx)
        pre_merge_watermark = self._atom_id
        self.merge(other_graph, transition_bond.atom_idx, other_target_idx, selected_attr)
        last_unit = self.add_new_unit_and_bond(pre_merge_watermark)
        self.add_unit_to_sequence(last_unit)
        self.nested_transition(new_sto_atom_id, rng)

        return new_sto_atom_id, True

    def propagate_graph(self, sto_atom_id, rng, prefer_parent_bonds):

        def pop_random_stochastic_bond():
            # Find a transition bond
            stochastic_idx = []
            propagation_weight = []
            for i, half_bond in zip(*self.get_open_half_bonds(sto_atom_id, prefer_parent=prefer_parent_bonds)):
                if half_bond.propagation_suitable:
                    # TODO carefully check if stochastic bonds have the right weight here!
                    propagation_weight += [half_bond.weight]
                    stochastic_idx += [i]
            propagation_weight = np.asarray(propagation_weight)
            stochastic_prob = None
            if len(propagation_weight) > 0:
                stochastic_prob = self.stochastic_tracker.normalized_probabilities(
                    propagation_weight,
                    "stochastic growth bond draw",
                )
            stochastic_half_bond = None

            # Select one of them
            if len(stochastic_idx) > 0:
                selected_stochastic_idx = rng.choice(stochastic_idx, p=stochastic_prob)
                stochastic_half_bond = self._open_half_bond_map[sto_atom_id].pop(selected_stochastic_idx)

            return stochastic_half_bond

        stochastic_bond = pop_random_stochastic_bond()

        if stochastic_bond is None:
            raise IncompleteStochasticGeneration(self)

        target_attr, target_idx, all_target_molar_amounts = stochastic_bond.get_mode_bonds(_PROPAGATION_NAME)
        target_weights = np.asarray([attr[_PROPAGATION_NAME] for attr in target_attr])
        gen_sto_id = self.stochastic_tracker._stochastic_atom_id_to_gen_id[sto_atom_id]
        target_molar_amounts = [molar_amount[gen_sto_id] for molar_amount in all_target_molar_amounts]
        molar_amounts = np.asarray(target_molar_amounts)
        target_weights *= molar_amounts
        target_id = self.stochastic_tracker.choose(
            rng,
            len(target_idx),
            target_weights,
            "stochastic growth target selection",
        )
        selected_target_idx = target_idx[target_id]
        selected_attr = self.gen_edge_attr_to_bond_attr(target_attr[target_id])
        selected_target_sto_gen_id = self.generative_graph.nodes[selected_target_idx]["stochastic_id_tree"][0]
        selected_target_sto_parent_id = self.generative_graph.nodes[selected_target_idx]["stochastic_id_tree"][1:]
        new_sto_atom_id = sto_atom_id
        if self.stochastic_tracker._stochastic_atom_id_to_gen_id[sto_atom_id] != selected_target_sto_gen_id:
            for existing_atom_id in reversed(self.stochastic_tracker.get_unterminated_sto_atom_ids()):
                if self.stochastic_tracker._stochastic_atom_id_to_gen_id[existing_atom_id] == selected_target_sto_gen_id:
                    new_sto_atom_id = existing_atom_id
                    break

            if new_sto_atom_id == sto_atom_id:
                new_sto_atom_id, parent_list = self.stochastic_tracker.register_parent_atom_instances(selected_target_sto_gen_id, sto_atom_id, selected_target_sto_parent_id)
                # new_sto_atom_id = self.stochastic_tracker.register_new_atom_instance(selected_target_sto_gen_id, sto_atom_id, None, False)
            # self.stochastic_tracker.terminate(sto_atom_id)

        other_graph = _PartialAtomGraph(self.generative_graph, self.static_graph, selected_target_idx, self.stochastic_tracker, new_sto_atom_id, rng)

        other_half_bond_atom_idx = other_graph.pop_target_open_half_bond(new_sto_atom_id, selected_target_idx)
        pre_merge_watermark = self._atom_id
        self.merge(other_graph, stochastic_bond.atom_idx, other_half_bond_atom_idx, selected_attr)
        last_unit = self.add_new_unit_and_bond(pre_merge_watermark)
        self.add_unit_to_sequence(last_unit)
        self.nested_transition(new_sto_atom_id, rng)
        return new_sto_atom_id

    def nested_transition(self, sto_atom_id, rng):

        def pop_nested_bonds():
            if sto_atom_id not in self._open_half_bond_map:
                return []
            if all(half_bond._special_target is None for half_bond in self._open_half_bond_map[sto_atom_id]):
                # Common case (no nested special targets): skip the rebuild.
                return []

            # Find a transition bond
            normal_bonds = []
            special_bonds = []

            for half_bond in self._open_half_bond_map[sto_atom_id]:
                if half_bond._special_target is not None:
                    special_bonds += [half_bond]
                else:
                    normal_bonds += [half_bond]
            self._open_half_bond_map[sto_atom_id] = normal_bonds

            return special_bonds

        for nested_transition_bond in pop_nested_bonds():
            selected_target_idx, selected_edge_attr = nested_transition_bond._special_target
            selected_attr = self.gen_edge_attr_to_bond_attr(selected_edge_attr)
            selected_target_sto_gen_id = self.generative_graph.nodes[selected_target_idx]["stochastic_id_tree"][0]

            new_sto_atom_id = sto_atom_id
            if self.stochastic_tracker._stochastic_atom_id_to_gen_id[sto_atom_id] != selected_target_sto_gen_id:
                for existing_atom_id in reversed(self.stochastic_tracker.get_unterminated_sto_atom_ids()):
                    if self.stochastic_tracker._stochastic_atom_id_to_gen_id[existing_atom_id] == selected_target_sto_gen_id:
                        new_sto_atom_id = existing_atom_id
                        break

                if new_sto_atom_id == sto_atom_id:
                    # Register with the FULL ancestor chain: add_molw credits only the
                    # ancestors listed in parent_map, so a single-parent entry starves
                    # grandparent SOs of grandchild mass and they overshoot their target.
                    selected_target_sto_parent_id = self.generative_graph.nodes[selected_target_idx]["stochastic_id_tree"][1:]
                    new_sto_atom_id, _parent_list = self.stochastic_tracker.register_parent_atom_instances(selected_target_sto_gen_id, sto_atom_id, selected_target_sto_parent_id)

            other_graph = _PartialAtomGraph(self.generative_graph, self.static_graph, selected_target_idx, self.stochastic_tracker, new_sto_atom_id, rng)

            other_half_bond_atom_idx = other_graph.pop_target_open_half_bond(new_sto_atom_id, selected_target_idx)

            pre_merge_watermark = self._atom_id
            self.merge(other_graph, nested_transition_bond.atom_idx, other_half_bond_atom_idx, selected_attr)
            last_unit = self.add_new_unit_and_bond(pre_merge_watermark)
            self.add_unit_to_sequence(last_unit)
            self.nested_transition(new_sto_atom_id, rng)

    def add_new_unit_and_bond(self, pre_merge_watermark):
        # `pre_merge_watermark` is self._atom_id captured BEFORE the most recent
        # merge: merge() relabels incoming nodes to ids >= that watermark, so the
        # newly added unit is exactly the nodes at or above it.
        # TODO: add bonds to units as in add_unit_to_sequence so mol_graph_to_rdkit_mol can be simpler
        if not self.collect_info:
            return None
        current_atom_graph = self.atom_graph
        new_nodes = [node for node in current_atom_graph.nodes() if node >= pre_merge_watermark]
        # Copy only the new unit (deepcopy of the whole molecule made generation
        # quadratic in chain length); the extra .copy() detaches the subgraph
        # view before deepcopy so the template graph is not dragged along.
        added_atom_graph = deepcopy(current_atom_graph.subgraph(new_nodes).copy())
        # A merge fires exactly one half-bond, so at most one edge crosses the
        # watermark; if that invariant ever breaks, this keeps only the last.
        edge_to_add = None
        for u, v in current_atom_graph.edges():
            if v >= pre_merge_watermark > u:
                edge_to_add = current_atom_graph.nodes[u]["origin_idx"], current_atom_graph.nodes[v]["origin_idx"]

        if edge_to_add is not None:
            if edge_to_add in self.bonds_idx:
                self.bonds_idx[edge_to_add] += 1
            else:
                self.bonds_idx[edge_to_add] = 1

        if added_atom_graph.number_of_nodes() > 0:
            old_unit = None
            for _node, data in added_atom_graph.nodes(data=True):
                for unit in self.units:
                    for _other_node, other_data in unit.nodes(data=True):
                        if data["origin_idx"] == other_data["origin_idx"]:
                            old_unit = unit
                            break
            if old_unit is None:
                self.units[added_atom_graph] = 1
            else:
                self.units[old_unit] += 1
        if added_atom_graph.number_of_nodes() > 0:
            return added_atom_graph
        else:
            return None

    def add_unit_to_sequence(self, last_unit):
        added_unit = deepcopy(last_unit)
        if added_unit is None:
            return
        if len(self.sequence) == 0:
            self.sequence.append([added_unit])
            return
        if len(self.sequence) == 1:
            self.sequence.append([added_unit])
            self.terminal_units.append(added_unit)
            initiator = self.sequence[0][0]
            for u, v in self.atom_graph.edges():
                if ((u, v) not in added_unit.edges()) and (v in added_unit.nodes()):
                    initiator.add_node("C" + str(self.current_connection))
                    initiator.add_edge(u, "C" + str(self.current_connection))
                    for attribute in self.atom_graph.edges[(u, v)]:
                        initiator.edges[(u, "C" + str(self.current_connection))][attribute] = self.atom_graph.edges[(u, v)][attribute]
                    for attribute in self.atom_graph.nodes[v]:
                        initiator.nodes["C" + str(self.current_connection)][attribute] = self.atom_graph.nodes[v][attribute]
                    initiator.nodes["C" + str(self.current_connection)]["atomic_num"] = 0
                    initiator.nodes["C" + str(self.current_connection)]["connection"] = self.current_connection
                    self.current_connection += 1
            return

        connection = None
        for u, v in self.atom_graph.edges():
            if ((u, v) not in added_unit.edges()) and (v in added_unit.nodes()):
                connection = (u, v)
        if connection is None:
            pass
        else:
            (u, v) = connection
            found_connection = False
            for sequence_idx in range(len(self.sequence)):
                sequence = self.sequence[sequence_idx]
                for unit in sequence:
                    for node in unit.nodes():
                        if node == u:
                            found_connection = True
                            if unit in self.terminal_units:
                                self.terminal_units.remove(unit)
                                self.terminal_units.append(added_unit)
                                self.sequence[sequence_idx].append(added_unit)
                            else:
                                unit.add_node("C" + str(self.current_connection))
                                unit.add_edge(u, "C" + str(self.current_connection))
                                for attribute in self.atom_graph.edges[(u, v)]:
                                    unit.edges[(u, "C" + str(self.current_connection))][attribute] = self.atom_graph.edges[(u, v)][attribute]
                                for attribute in self.atom_graph.nodes[v]:
                                    unit.nodes["C" + str(self.current_connection)][attribute] = self.atom_graph.nodes[v][attribute]
                                unit.nodes["C" + str(self.current_connection)]["atomic_num"] = 0
                                unit.nodes["C" + str(self.current_connection)]["connection"] = self.current_connection
                                self.current_connection += 1
                                self.terminal_units.append(added_unit)
                                self.sequence.append([added_unit])
                            break
                        if found_connection:
                            break
                    if found_connection:
                        break
                if found_connection:
                    break


class EnsembleCreator:

    def __init__(self, generative_graph):

        self._generative_graph = generative_graph.copy()

        # Sampling filters every non-static decision by the per-edge stochastic id;
        # a graph built against the older schema (per-edge 'hierarchy') would not
        # error but silently generate truncated, end-group-less molecules.
        for _u, _v, edge_data in self._generative_graph.edges(data=True):
            if _EDGE_STOCHASTIC_ID_NAME not in edge_data:
                raise IncompatibleGenerativeGraphSchema(_EDGE_STOCHASTIC_ID_NAME)

        self._static_graph = self._create_static_graph(self.generative_graph)
        self._static_proof_supported = all(
            u == v or self._static_graph.has_edge(v, u)
            for u, v in self._static_graph.edges()
        )

        # The static partition: a unit is one static-connected component.
        static_components = tuple(
            frozenset(component)
            for component in nx.connected_components(
                self._static_graph.to_undirected(as_view=True)
            )
        )
        self._static_components = static_components
        self._node_to_static_component = {
            node: component_id
            for component_id, component in enumerate(static_components)
            for node in component
        }
        self._statically_empty_nested_mw_sto_gen_ids = (
            self._find_statically_empty_nested_mw_sto_gen_ids()
        )
        if self._static_proof_supported:
            (
                self._provably_dead_construction_states,
                self._provably_immediate_zero_components,
            ) = self._find_provably_dead_construction_states()
            self._provably_zero_termination_states = (
                self._find_provably_zero_termination_states()
            )
        else:
            self._provably_dead_construction_states = frozenset()
            self._provably_immediate_zero_components = frozenset()
            self._provably_zero_termination_states = frozenset()
        self._source_provably_dead_cache = {}

        self._starting_node_idx, self._starting_node_weight = self._create_init_weights(self.generative_graph)

        self._repeat_unit_starting_node_idx, self._repeat_unit_starting_node_weight = self._create_repeat_units_as_source(self.generative_graph)

        # A sticky branch flag is insufficient on its own: several positive
        # sources (or repeat-unit choices) may all converge on the same dead
        # nested expansion.  This conservative proof overrides that flag only
        # when every source with nonzero selection probability is known dead.
        self._automatic_zero_support_is_unavoidable = {
            False: self._all_reachable_sources_are_provably_dead(
                self._starting_node_idx,
                self._starting_node_weight,
            ),
            True: self._all_reachable_sources_are_provably_dead(
                self._repeat_unit_starting_node_idx,
                self._repeat_unit_starting_node_weight,
            ),
        }
        # Both weight vectors are immutable after this point, so whether the
        # automatic source draw branches is a per-mode constant.
        self._automatic_source_is_conditional = {
            False: np.count_nonzero(
                np.asarray(self._starting_node_weight) > 0.0
            ) > 1,
            True: np.count_nonzero(
                np.asarray(self._repeat_unit_starting_node_weight) > 0.0
            ) > 1,
        }

    def _find_statically_empty_nested_mw_sto_gen_ids(self):
        """Find nested MW draws whose truncated support is empty on every chain.

        A nested instance draws its target MW truncated to [1, parent budget],
        and the parent budget never exceeds the parent distribution's support
        upper bound.  When the child support starts strictly above that static
        ceiling, every instantiation raises EmptyTruncatedDistributionSupport
        regardless of the drawn parent target.  Anything ambiguous (unknown
        bounds, infinite parent support, malformed trees) stays out of the
        set, keeping the failure chain-local.
        """
        try:
            first_node = next(iter(self._generative_graph.nodes))
            serial_vectors = tuple(
                self._generative_graph.nodes[first_node][
                    "molecular_weight_distribution"
                ]
            )
        except (StopIteration, KeyError, TypeError):
            return frozenset()

        bounds = {}
        for sto_gen_id, serial_vector in enumerate(serial_vectors):
            try:
                distribution = StochasticDistribution.from_serial_vector(
                    list(serial_vector)
                )
                frozen = distribution._distribution
                parameters = getattr(frozen, "kwds", {})
                scale = parameters.get("scale")
                if scale is not None and float(scale) == 0.0:
                    point = float(parameters.get("loc", 0.0))
                    if np.isfinite(point):
                        bounds[sto_gen_id] = (point, point)
                    continue
                support_lower, support_upper = frozen.support()
                bounds[sto_gen_id] = (
                    float(support_lower),
                    float(support_upper),
                )
            except (
                AttributeError,
                IndexError,
                TypeError,
                ValueError,
                OverflowError,
                RuntimeError,
            ):
                continue

        parent_ids = {}
        for _node, data in self._generative_graph.nodes(data=True):
            try:
                tree = data["stochastic_id_tree"]
                child = tree[0]
                parent = tree[1]
            except (KeyError, IndexError, TypeError):
                continue
            if not isinstance(child, (int, np.integer)) or child < 0:
                continue
            if not isinstance(parent, (int, np.integer)) or parent < 0:
                continue
            known = parent_ids.setdefault(int(child), int(parent))
            if known != int(parent):
                parent_ids[int(child)] = None

        empty_ids = set()
        for child, parent in parent_ids.items():
            if parent is None:
                continue
            child_bounds = bounds.get(child)
            parent_bounds = bounds.get(parent)
            if child_bounds is None or parent_bounds is None:
                continue
            child_lower = max(child_bounds[0], 1.0)
            parent_upper = parent_bounds[1]
            if (
                np.isfinite(child_lower)
                and np.isfinite(parent_upper)
                and child_lower > parent_upper
            ):
                empty_ids.add(child)

        return frozenset(empty_ids)

    def _find_provably_dead_construction_states(self):
        """Find ``(unit, consumed entry)`` states that must hit zero support.

        Special-target normalization happens while every static atom is built,
        so an empty group is fatal even if its half-bond is later discarded.
        Following a positive special target is different: the connection
        half-bond is consumed before ``nested_transition``, and nonpositive
        ``gen_weight`` half-bonds are never retained.  Keying the least fixed
        point by the consumed node preserves those runtime distinctions and
        prevents the conservative proof from declaring a viable unit fatal.
        Malformed data and unseeded cycles remain unknown (not dead).
        """
        graph = self._generative_graph
        groups_by_component = {
            component_id: []
            for component_id in range(len(self._static_components))
        }
        immediate_zero_components = set()
        seed_dead_states = set()

        for component_id, component in enumerate(self._static_components):
            for node in component:
                try:
                    source_sto_id = graph.nodes[node]["stochastic_id_tree"][0]
                except (KeyError, IndexError, TypeError):
                    continue

                targets = []
                group_found = False
                group_unknown = False
                for _u, target, data in graph.out_edges(node, data=True):
                    try:
                        transition_weight = float(data[_TRANSITION_NAME])
                        target_tree = graph.nodes[target]["stochastic_id_tree"]
                        target_sto_id = target_tree[0]
                        is_special = (
                            not data["static"]
                            and transition_weight > 0
                            and source_sto_id in target_tree[1:]
                            and data.get(_EDGE_STOCHASTIC_ID_NAME)
                            == target_sto_id
                            and target_sto_id != -1
                        )
                    except (KeyError, IndexError, TypeError, ValueError):
                        continue

                    if not is_special:
                        continue
                    group_found = True
                    try:
                        molar_amount = graph.nodes[target][
                            "unit_molar_amounts"
                        ][target_sto_id]
                        effective_weight = float(
                            transition_weight * molar_amount
                        )
                        target_component = self._node_to_static_component[target]
                    except (KeyError, IndexError, TypeError, ValueError):
                        group_unknown = True
                        continue

                    if not np.isfinite(effective_weight) or effective_weight < 0:
                        group_unknown = True
                    elif effective_weight > 0:
                        target_state = (target_component, target)
                        targets.append(target_state)
                        if (
                            target_sto_id
                            in self._statically_empty_nested_mw_sto_gen_ids
                        ):
                            # Instantiating this nested object dies at its
                            # truncated MW draw before any construction, so
                            # the entered state is dead a priori.
                            seed_dead_states.add(target_state)

                if group_found:
                    followable = None
                    try:
                        gen_weight = float(graph.nodes[node]["gen_weight"])
                        if np.isfinite(gen_weight):
                            followable = gen_weight > 0
                    except (KeyError, TypeError, ValueError):
                        pass
                    group = (
                        None
                        if group_unknown
                        else (node, followable, tuple(targets))
                    )
                    groups_by_component[component_id].append(group)
                    if group is not None and not targets:
                        immediate_zero_components.add(component_id)

        dead_states = set(seed_dead_states)
        changed = True
        while changed:
            changed = False
            for component_id, groups in groups_by_component.items():
                consumed_nodes = (None, *self._static_components[component_id])
                for consumed_node in consumed_nodes:
                    state = (component_id, consumed_node)
                    if state in dead_states:
                        continue
                    for group in groups:
                        if group is None:
                            continue
                        source_node, followable, targets = group
                        if not targets:
                            dead_states.add(state)
                            changed = True
                            break
                        if (
                            followable is True
                            and source_node != consumed_node
                            and all(target in dead_states for target in targets)
                        ):
                            dead_states.add(state)
                            changed = True
                            break

        return (
            frozenset(dead_states),
            frozenset(immediate_zero_components),
        )

    def _find_provably_zero_termination_states(self):
        """Find attached-unit states with a retained all-zero end-group draw.

        The normal termination estimator visits every retained termination
        half-bond, so one known all-zero target group is fatal.  This proof is
        intentionally limited to freshly attached static units: an entry
        half-bond is consumed, nonpositive generating weights are dropped, and
        any transition-capable half-bond is excluded by
        ``_get_level_termination_bonds`` (or consumed by nested transition).
        Ambiguous attributes remain unknown.
        """
        graph = self._generative_graph
        zero_states = set()

        for component_id, component in enumerate(self._static_components):
            for node in component:
                try:
                    gen_weight = float(graph.nodes[node]["gen_weight"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not np.isfinite(gen_weight) or gen_weight <= 0:
                    continue

                has_transition = False
                groups = {}
                unknown_levels = set()
                malformed = False
                for _u, target, data in graph.out_edges(node, data=True):
                    if data.get("static", False):
                        continue
                    try:
                        transition_weight = float(data[_TRANSITION_NAME])
                        termination_weight = float(data[_TERMINATION_NAME])
                    except (KeyError, TypeError, ValueError):
                        malformed = True
                        break
                    if (
                        not np.isfinite(transition_weight)
                        or transition_weight < 0
                        or not np.isfinite(termination_weight)
                        or termination_weight < 0
                    ):
                        malformed = True
                        break
                    if transition_weight > 0:
                        has_transition = True
                    if termination_weight <= 0:
                        continue

                    level = data.get(_EDGE_STOCHASTIC_ID_NAME)
                    if not isinstance(level, (int, np.integer)) or level < 0:
                        malformed = True
                        break
                    try:
                        molar_amount = graph.nodes[target][
                            "unit_molar_amounts"
                        ][level]
                        effective_weight = float(
                            termination_weight * molar_amount
                        )
                    except (KeyError, IndexError, TypeError, ValueError):
                        unknown_levels.add(level)
                        continue
                    if not np.isfinite(effective_weight) or effective_weight < 0:
                        unknown_levels.add(level)
                        continue
                    groups.setdefault(level, []).append(effective_weight)

                if malformed or has_transition:
                    continue

                for level, effective_weights in groups.items():
                    if (
                        level in unknown_levels
                        or any(
                            weight > 0 for weight in effective_weights
                        )
                    ):
                        continue
                    for consumed_node in (None, *component):
                        if consumed_node != node:
                            zero_states.add(
                                (component_id, consumed_node, level)
                            )

        return frozenset(zero_states)

    def _attached_target_is_provably_dead(self, target):
        """Known zero-support failure after attaching ``target``, or None."""
        try:
            component_id = self._node_to_static_component[target]
            sto_gen_id = self._generative_graph.nodes[target][
                "stochastic_id_tree"
            ][0]
        except (KeyError, IndexError, TypeError):
            return None
        if not isinstance(sto_gen_id, (int, np.integer)) or sto_gen_id < 0:
            return None
        return (
            (component_id, target)
            in self._provably_dead_construction_states
            or (
                component_id,
                target,
                sto_gen_id,
            ) in getattr(
                self,
                "_provably_zero_termination_states",
                frozenset(),
            )
        )

    def _global_source_is_provably_dead(self, component):
        """Prove every possible first ``-1`` arm is dead.

        The loop may stop after a non-growing arm, before later global bonds
        fire.  Therefore eventual failure of one arm is insufficient: every
        retained group must fail on its own first target/construction path.
        Unknown weights or target ownership disable the proof.
        """
        graph = self._generative_graph
        found_group = False
        for node in component:
            try:
                gen_weight = float(graph.nodes[node]["gen_weight"])
            except (KeyError, TypeError, ValueError):
                return False
            if not np.isfinite(gen_weight):
                return False
            if gen_weight <= 0:
                continue

            global_edges = []
            for _u, target, data in graph.out_edges(node, data=True):
                try:
                    transition_weight = float(data[_TRANSITION_NAME])
                except (KeyError, TypeError, ValueError):
                    return False
                if not np.isfinite(transition_weight) or transition_weight < 0:
                    return False
                if (
                    not data.get("static", False)
                    and transition_weight > 0
                    and data.get(_EDGE_STOCHASTIC_ID_NAME) == -1
                ):
                    global_edges.append((target, transition_weight))

            if not global_edges:
                continue
            found_group = True

            group_has_non_dead_target = False
            for target, transition_weight in global_edges:
                try:
                    target_sto_gen_id = graph.nodes[target][
                        "stochastic_id_tree"
                    ][0]
                    if (
                        not isinstance(target_sto_gen_id, (int, np.integer))
                        or target_sto_gen_id < 0
                    ):
                        return False
                    molar_amount = graph.nodes[target][
                        "unit_molar_amounts"
                    ][target_sto_gen_id]
                    effective_weight = float(
                        transition_weight * molar_amount
                    )
                except (KeyError, IndexError, TypeError, ValueError):
                    return False
                if not np.isfinite(effective_weight) or effective_weight < 0:
                    return False
                if effective_weight > 0:
                    target_is_dead = self._attached_target_is_provably_dead(
                        target
                    )
                    if target_is_dead is None:
                        return False
                    if not target_is_dead:
                        group_has_non_dead_target = True

            if not group_has_non_dead_target:
                continue
            return False

        return found_group

    def _source_is_provably_dead(self, source):
        """Whether this source's first growth must reach zero support.

        False means "not proved dead", not necessarily proved productive.  The
        runtime's initial max-hierarchy and owner filters are mirrored exactly
        where static data is sufficient; every ambiguity stays retryable.
        """
        cache = self._source_provably_dead_cache
        if source not in cache:
            cache[source] = self._compute_source_is_provably_dead(source)
        return cache[source]

    def _compute_source_is_provably_dead(self, source):
        if not self._static_proof_supported:
            return False

        graph = self._generative_graph
        zero_termination_states = getattr(
            self,
            "_provably_zero_termination_states",
            frozenset(),
        )
        try:
            component_id = self._node_to_static_component[source]
            component = self._static_components[component_id]
            sto_gen_id = graph.nodes[source]["stochastic_id_tree"][0]
        except (KeyError, IndexError, TypeError):
            return False

        if component_id in self._provably_immediate_zero_components:
            return True
        if not isinstance(sto_gen_id, (int, np.integer)):
            return False
        if sto_gen_id == -1:
            return self._global_source_is_provably_dead(component)
        if sto_gen_id < 0:
            return False

        eligible = []
        for node in component:
            try:
                gen_weight = float(graph.nodes[node]["gen_weight"])
            except (KeyError, TypeError, ValueError):
                return False
            if not np.isfinite(gen_weight):
                return False
            if gen_weight <= 0:
                continue

            level_edges = []
            for _u, target, data in graph.out_edges(node, data=True):
                try:
                    transition_weight = float(data[_TRANSITION_NAME])
                except (KeyError, TypeError, ValueError):
                    return False
                if not np.isfinite(transition_weight) or transition_weight < 0:
                    return False
                if (
                    not data.get("static", False)
                    and transition_weight > 0
                    and data.get(_EDGE_STOCHASTIC_ID_NAME) == sto_gen_id
                ):
                    level_edges.append((target, data))
            if level_edges:
                eligible.append((node, level_edges))

        if not eligible:
            return (
                component_id,
                None,
                sto_gen_id,
            ) in zero_termination_states

        hierarchy_by_node = {}
        for node, _edges in eligible:
            try:
                hierarchy = graph.nodes[node]["gen_hierarchy"]
                if not isinstance(
                    hierarchy,
                    (int, float, np.integer, np.floating),
                ):
                    raise TypeError
                hierarchy = float(hierarchy)
                if not np.isfinite(hierarchy):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                return False
            hierarchy_by_node[node] = hierarchy
        max_hierarchy = max(hierarchy_by_node.values())
        eligible = [
            (node, edges)
            for node, edges in eligible
            if hierarchy_by_node[node] == max_hierarchy
        ]

        for node, edges in eligible:
            try:
                if graph.nodes[node]["stochastic_id_tree"][0] != sto_gen_id:
                    return False
            except (KeyError, IndexError, TypeError):
                return False

            source_expansion_is_dead = (
                component_id,
                node,
            ) in self._provably_dead_construction_states
            source_termination_is_dead = (
                component_id,
                node,
                sto_gen_id,
            ) in zero_termination_states

            route_has_non_dead_target = False
            for target, data in edges:
                try:
                    effective_weight = float(
                        data[_TRANSITION_NAME]
                        * graph.nodes[target]["unit_molar_amounts"][sto_gen_id]
                    )
                    target_sto_gen_id = graph.nodes[target][
                        "stochastic_id_tree"
                    ][0]
                except (KeyError, IndexError, TypeError, ValueError):
                    return False
                if not np.isfinite(effective_weight) or effective_weight < 0:
                    return False
                if effective_weight <= 0:
                    continue
                if (
                    target_sto_gen_id == sto_gen_id
                    and (
                        source_expansion_is_dead
                        or source_termination_is_dead
                    )
                ):
                    continue
                target_is_dead = self._attached_target_is_provably_dead(target)
                if target_is_dead is None:
                    return False
                if (
                    not target_is_dead
                ):
                    route_has_non_dead_target = True

            if route_has_non_dead_target:
                return False

        return True

    def _all_reachable_sources_are_provably_dead(self, source_nodes, weights):
        """Combine only source candidates that automatic selection can reach."""
        if len(source_nodes) != len(weights):
            return False

        reachable_sources = []
        for source, weight in zip(source_nodes, weights, strict=True):
            try:
                probability = float(weight)
            except (TypeError, ValueError):
                return False
            if not np.isfinite(probability) or probability < 0:
                return False
            if probability > 0:
                reachable_sources.append(source)

        return bool(reachable_sources) and all(
            self._source_is_provably_dead(source)
            for source in reachable_sources
        )

    @staticmethod
    def _create_init_weights(graph):
        # TODO: check mixtures.
        from collections import defaultdict

        starting_node_idx = []
        starting_node_weight = []
        stochastic_id_trees = []
        molar_weights = []
        init_weights = []
        for node_idx, data in graph.nodes(data=True):
            if data["init_weight"] > 0:
                stochastic_id_tree = [stochastic_id for stochastic_id in data["stochastic_id_tree"] if stochastic_id >= 0]
                stochastic_id_trees += [stochastic_id_tree]
                molar_weight = [data["unit_molar_amounts"][id] for id in stochastic_id_tree]
                molar_weights += [molar_weight]
                init_weights += [data["init_weight"]]
                starting_node_idx += [node_idx]

        group_level_weights = defaultdict(lambda: defaultdict(float))
        for i, node_idx in enumerate(starting_node_idx):
            id_tree = stochastic_id_trees[i]
            weights = molar_weights[i]
            rev_ids = list(reversed(id_tree))
            rev_weights = list(reversed(weights))
            for k, group_id in enumerate(rev_ids):
                child_key = rev_ids[k + 1] if k < len(rev_ids) - 1 else node_idx
                group_level_weights[group_id][child_key] += rev_weights[k]

        for i, node_idx in enumerate(starting_node_idx):
            rev_ids = list(reversed(stochastic_id_trees[i]))
            init_weight = init_weights[i]
            prob = init_weight
            for k, group_id in enumerate(rev_ids):
                child_key = rev_ids[k + 1] if k < len(rev_ids) - 1 else node_idx
                level_total = sum(group_level_weights[group_id].values())
                # A |0| molar amount on every alternative of a group makes
                # routes through it unreachable, not a division error.
                if level_total > 0:
                    prob *= group_level_weights[group_id][child_key] / level_total
                else:
                    prob = 0.0
            starting_node_weight += [prob]

        starting_node_weight = np.asarray(starting_node_weight)
        total_weight = np.sum(starting_node_weight)
        if not total_weight > 0:
            # No initiation route carries probability mass; automatic source
            # selection then reports NoValidGenerationSource at draw time.
            return [], np.asarray([])
        starting_node_weight /= total_weight

        return starting_node_idx, starting_node_weight

    @staticmethod
    # TODO: consider nested stochastic object in the selection of starting nodes from repeat units
    def _create_repeat_units_as_source(generative_graph):
        # TODO fix this function, sometimes it brings errors.
        starting_node_idx = []
        starting_node_weight = []
        graph_transitions = []
        for u, v, data in generative_graph.edges(data=True):
            if data[_TRANSITION_NAME] > 0:
                graph_transitions.append((u, v))

        if not graph_transitions:
            for node_idx, data in generative_graph.nodes(data=True):
                if (data["init_weight"] == -1) and (data["gen_weight"] > 0):  # and (stochastic_tree_depth[node_idx]) == max_depth:
                    starting_node_idx.append(node_idx)
                    starting_node_weight.append(data["gen_weight"])
        else:
            list_of_repeat_units = []
            for u, _ in graph_transitions:
                visited = set([u])
                queue = deque([u])

                while queue:
                    node = queue.popleft()

                    # Outgoing edges
                    for _, nbr, _key, data in generative_graph.out_edges(node, keys=True, data=True):
                        if data.get(_TRANSITION_NAME, 0) > 0:
                            continue  # stop traversal in this direction
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)

                    # Incoming edges
                    for nbr, _, _key, data in generative_graph.in_edges(node, keys=True, data=True):
                        if data.get(_TRANSITION_NAME, 0) > 0:
                            continue  # stop traversal in this direction
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
                list_of_repeat_units.append(visited)
            repeat_units_to_remove = []
            for repeat_unit in list_of_repeat_units:
                for node_idx in repeat_unit:
                    for _u, v in graph_transitions:
                        if node_idx == v:
                            if repeat_unit not in repeat_units_to_remove:
                                repeat_units_to_remove.append(repeat_unit)
                                continue
            for repeat_unit in repeat_units_to_remove:
                if repeat_unit in list_of_repeat_units:
                    list_of_repeat_units.remove(repeat_unit)

            for node_idx, data in generative_graph.nodes(data=True):
                if (any(node_idx in repeat_unit for repeat_unit in list_of_repeat_units)) or not list_of_repeat_units:
                    if (data["init_weight"] == -1) and (data["gen_weight"] > 0):
                        starting_node_idx.append(node_idx)
                        starting_node_weight.append(data["gen_weight"])

        if starting_node_idx:
            starting_node_weight = np.asarray(starting_node_weight)
            starting_node_weight /= np.sum(starting_node_weight)

        return starting_node_idx, starting_node_weight

    @staticmethod
    def _create_static_graph(generative_graph):
        static_graph = generative_graph.copy()
        edges_to_delete = set()
        for u, v, k, d in static_graph.edges(keys=True, data=True):
            if not d["static"]:
                edges_to_delete.add((u, v, k))

        static_graph.remove_edges_from(edges_to_delete)
        return static_graph

    @property
    def generative_graph(self):
        return self._generative_graph.copy()

    def _get_random_start_node(self, rng, use_repeat_units_as_source=False):
        if use_repeat_units_as_source:
            candidates = self._repeat_unit_starting_node_idx
            probabilities = self._repeat_unit_starting_node_weight
        else:
            candidates = self._starting_node_idx
            probabilities = self._starting_node_weight
        if not candidates:
            raise NoValidGenerationSource(use_repeat_units_as_source)
        return rng.choice(candidates, p=probabilities)

    @staticmethod
    def get_dot_string(atom_graph, bond_type_colors=None, prefix="") -> str:
        if bond_type_colors is None:
            bond_type_colors = {1: "black", 2: "red", 3: "green", 4: "blue"}
        dot_str = "graph{\n"
        for node, data in atom_graph.nodes(data=True):
            label = atom_name_mapping[data["atomic_num"]]
            color = "#" + atom_color_mapping[data["atomic_num"]]

            extra_attr = f'style="filled", fillcolor="{color}", '
            if _determine_darkness_from_hex(color):
                extra_attr += "fontcolor=white, "
            dot_str += f'"{prefix}{node}" [{extra_attr} label="{label}"];\n'

        for u, v, d in atom_graph.edges(data=True):
            bond_type = d["bond_type"]
            color = bond_type_colors[bond_type]
            style = "solid"
            if d["aromatic"]:
                style = "dashed"
            dot_str += f'"{prefix}{u}" -- "{prefix}{v}" [color="{color}", style="{style}"];\n'
        dot_str += "}\n"
        return dot_str

    def sample_mol_graph(
        self,
        source: Optional[str] = None,
        use_repeat_units_as_source=False,
        rng=None,
        termination_flag: Optional[int] = None,
        tolerate_incomplete_stochastic_generation_with_no_more_than_X_open_bonds=0,
        molecule_info=False,
    ):
        # TODO: consider using repeat units as source not an option.
        if rng is None:
            rng = get_global_rng()

        automatic_source = source is None
        source_is_conditional = False
        zero_support_is_unavoidable = False
        if automatic_source:
            zero_support_is_unavoidable = (
                self._automatic_zero_support_is_unavoidable[
                    bool(use_repeat_units_as_source)
                ]
            )
            source_is_conditional = self._automatic_source_is_conditional[
                bool(use_repeat_units_as_source)
            ]
            source = self._get_random_start_node(rng, use_repeat_units_as_source)

        # The generative_graph property copies the whole template graph on every access:
        # take one copy per sample instead of one per use.
        generative_graph = self.generative_graph

        if source not in generative_graph.nodes():
            raise InvalidGenerationSource(source, generative_graph.nodes(), generative_graph)

        if not automatic_source:
            zero_support_is_unavoidable = self._source_is_provably_dead(source)

        if (source not in self._starting_node_idx) and not use_repeat_units_as_source:
            warnings.warn(
                UnvalidatedGenerationSource(source, self._starting_node_idx, generative_graph),
                stacklevel=2,
            )

        stochastic_object_tracker = _StochasticObjectTracker(
            generative_graph,
            rng,
            path_is_conditional=source_is_conditional,
            zero_support_is_unavoidable=zero_support_is_unavoidable,
        )

        source_stochastic_id_tree = generative_graph.nodes[source]["stochastic_id_tree"]
        source_sto_gen_id = source_stochastic_id_tree[0]
        source_parents_sto_gen_id = source_stochastic_id_tree[1:]
        sto_atom_id, _parent_list = stochastic_object_tracker.register_parent_atom_instances(source_sto_gen_id, source_stochastic_id_tree[1], source_parents_sto_gen_id)

        partial_atom_graph = _PartialAtomGraph(generative_graph, self._static_graph, source, stochastic_object_tracker, sto_atom_id, rng, collect_info=molecule_info)
        del stochastic_object_tracker

        if molecule_info:
            # Use a stable snapshot (deepcopy) as the units key. After the Phase 0
            # in-place merge change, partial_atom_graph.atom_graph is mutated as the
            # chain grows, so storing the live graph as a dict key would let it
            # accumulate all subsequently-added atoms and incorrectly match every
            # later unit by origin_idx in add_new_unit_and_bond.
            unit_to_add = deepcopy(partial_atom_graph.atom_graph)

            if unit_to_add.number_of_nodes() > 0:
                partial_atom_graph.units[unit_to_add] = 1
                partial_atom_graph.add_unit_to_sequence(unit_to_add)

        if source_sto_gen_id == -1:
            # Source is not a stochastic object. Terminate it immediately so the while
            # loop doesn't try to grow it via transition_graph (which would incorrectly
            # sweep sibling -1 bonds from other atoms into the first arm's stochastic
            # bucket). Its -1 transition bonds stay in _open_half_bond_map and are fired
            # independently, one per call, by trigger_global_transitions in the while
            # condition.
            partial_atom_graph.stochastic_tracker.terminate(sto_atom_id)
        else:
            partial_atom_graph.transition_graph(sto_atom_id, source_sto_gen_id, rng)

        # Pending-termination bookkeeping (P1-02 rework). An instance whose
        # PROJECTED final mass crosses its target stops growing at its own
        # level ("pending") but is only terminated once no live descendant
        # remains, so nested objects always finish their own declared
        # distributions instead of being truncated mid-growth. The projection
        #     proj(X) = actual(X) + sum(expected remaining mass of X's live
        #               descendant subtrees) + X's net termination-cap mass
        # is invariant while a descendant remains live (its credit to X
        # cancels against its shrinking remainder). Finalization exposes the
        # descendant's realized rounding residual and can therefore cross X;
        # the owner checkpoint spans that complete unit/subtree and remains the
        # correct under boundary. Rounding between `checkpoints[X]` and the
        # current state keeps E[final] on target.
        #
        # A checkpoint is a RAW deepcopy of the whole assembly taken right
        # before an OWNER-level growth step. Its epoch advances only when that
        # same owner adds a unit or continuation: descendant mutations leave
        # the ancestor epoch unchanged. It is usable only for the immediately
        # following owner-level step: an older owner state can be arbitrarily
        # far below target, so a checkpoint never survives a second owner step.
        checkpoints = {}
        owner_epochs = {}
        forced_overshoot_no_boundary = set()
        deferred_continuation = None
        mutations = 0

        def _commit_mutation():
            # Committed growth makes the state path-dependent (the drawn MW
            # and earlier draws shaped it), so a later dead end must reject
            # only this chain; pairing the mark with the count here keeps a
            # future growth site from silently skipping it.
            nonlocal mutations
            partial_atom_graph.stochastic_tracker.mark_path_conditional()
            mutations += 1
        pending_termination = set()
        max_step_gain = {}
        gain_floor = {}
        last_checked_proj = {}
        own_termination_cache = {}
        avg_termination_cache = {}

        def _live_forest_children(tracker, live_ids, live_set, sto_atom_id):
            """Live descendants of sto_atom_id with no live instance strictly
            between: their subtree's future mass reaches sto_atom_id exactly
            once (a live intermediate's remainder already contains its own
            subtree's)."""
            children = []
            for candidate in live_ids:
                if candidate == sto_atom_id:
                    continue
                ancestors = tracker.parent_map.get(candidate, [])
                if sto_atom_id not in ancestors:
                    continue
                between = ancestors[ancestors.index(sto_atom_id) + 1:]
                if all(ancestor not in live_set for ancestor in between):
                    children.append(candidate)
            return children

        def _conditional_junction_mw(
            graph,
            tracker,
            live_ids,
            live_set,
            sto_atom_id,
            deferred_junction=None,
        ):
            """Net caps that would replace live descendant continuations if
            ``sto_atom_id`` parked in this exact topology.

            A finalization-boundary snapshot has already terminated its child
            but has not yet fired the ancestor-level continuation.  Its
            deferred owner is therefore also a conditional junction even
            though it is no longer in ``live_ids``.
            """
            owners = _live_forest_children(tracker, live_ids, live_set, sto_atom_id)
            if deferred_junction is not None and deferred_junction[1] == sto_atom_id:
                owners.append(deferred_junction[0])
            return sum(
                graph.get_average_junction_termination_mw(
                    owner_sto_atom_id,
                    sto_atom_id,
                    self._static_graph,
                    rng,
                )
                for owner_sto_atom_id in dict.fromkeys(owners)
            )

        def _remaining_credit(tracker, sto_atom_id):
            """Signed correction that makes a live descendant contribute its
            drawn target, independent of its temporary or rounded actual mass.

            This applies after parking too.  Replacing it with only pending cap
            mass makes a child's over/under outcome move every ancestor's
            projection, even though no ancestor-level unit was added.
            """
            expected = tracker._sto_atom_id_expected_molw[sto_atom_id]
            if expected < 0:
                return 0.0
            # Keep this remainder SIGNED.  A child temporarily above its target
            # must reduce its remaining credit by the same amount its actual
            # mass just added to the ancestor.  Clamping at zero made an
            # ancestor appear to cross first and consume the child's fresh
            # rounding boundary, systematically biasing small nested objects.
            return expected - tracker._sto_atom_id_actual_molw[sto_atom_id]

        def _projected_molw(tracker, live_ids, sto_atom_id):
            """Projected final tracked mass of sto_atom_id EXCLUDING its own
            termination caps (callers add the fresh/cached cap estimate).

            Only LIVE descendants contribute their signed target correction.
            A terminated descendant's realized mass stands as-is: freezing its
            ``expected - actual`` residual into the ancestor (a previous
            "settled credit" design) made every level inherit its children's
            structural overshoot — a sub-unit-target child can only land
            above its target, and the compounded inheritance biased 3-level
            ensembles +7% — whereas accounting the realized mass lets the
            ancestor compensate with its own growth."""
            live_set = set(live_ids)
            projected = tracker._sto_atom_id_actual_molw[sto_atom_id]
            for child in _live_forest_children(tracker, live_ids, live_set, sto_atom_id):
                projected += _remaining_credit(tracker, child)
            return projected

        def _total_termination_mw(
            graph,
            tracker,
            live_ids,
            sto_atom_id,
            deferred_junction=None,
        ):
            live_set = set(live_ids)
            own_mw = graph.get_average_termination_mw(
                sto_atom_id,
                self._static_graph,
                rng,
            )
            conditional_mw = _conditional_junction_mw(
                graph,
                tracker,
                live_ids,
                live_set,
                sto_atom_id,
                deferred_junction,
            )
            return own_mw, own_mw + conditional_mw

        def _capture_checkpoint(checkpoint_owner, deferred_junction=None):
            """Capture both molecular topology and loop-control state.

            Restoring only the graph leaves pending ids and adaptive caches on
            the discarded timeline; newly-created ids can then be referenced
            after rollback and a finalized child can be lost altogether. Other
            checkpoints are immutable historical states, so shallow references
            preserve compatible ancestor boundaries without recursively
            copying whole molecular graphs. Sibling/descendant checkpoints are
            deliberately excluded: restoring this owner can remove its newly
            spawned subtree, while unrelated checkpoint histories would retain
            stale full-graph copies.
            """
            compatible_ancestors = set(
                partial_atom_graph.stochastic_tracker.parent_map.get(
                    checkpoint_owner,
                    [],
                )
            )
            return {
                "graph": copy.deepcopy(partial_atom_graph),
                "pending": set(pending_termination),
                "max_step_gain": dict(max_step_gain),
                "gain_floor": dict(gain_floor),
                "last_checked_proj": dict(last_checked_proj),
                "own_termination_cache": dict(own_termination_cache),
                "avg_termination_cache": dict(avg_termination_cache),
                "owner_epochs": dict(owner_epochs),
                "forced_overshoot_no_boundary": set(forced_overshoot_no_boundary),
                "checkpoints": {
                    owner: checkpoint
                    for owner, checkpoint in checkpoints.items()
                    if owner in compatible_ancestors
                },
                "owner": checkpoint_owner,
                "epoch": owner_epochs.get(checkpoint_owner, 0),
                "deferred_junction": deferred_junction,
            }

        def _advance_owner_epoch(sto_atom_id):
            """Record one successful composition step at exactly this level.

            A checkpoint survives arbitrary descendant mutations, but never a
            second owner-level step. Dropping it here is the stale-snapshot
            guard that prevents multi-unit rollback.
            """
            owner_epochs[sto_atom_id] = owner_epochs.get(sto_atom_id, 0) + 1
            checkpoint = checkpoints.get(sto_atom_id)
            if (
                checkpoint is not None
                and checkpoint["epoch"] != owner_epochs[sto_atom_id] - 1
            ):
                checkpoints.pop(sto_atom_id, None)

        def _finalize_pending(sto_atom_id):
            """Fire the parked instance's declared end groups and continue the
            chain at the nearest live ancestor's level (e.g. the diblock
            junction at the parent level). If that ancestor is itself parked,
            the junction must NOT continue: cap it with the ancestor's end
            groups instead (graft-through chains otherwise grow a decided
            level forever, one nested instance per continued unit)."""
            tracker = partial_atom_graph.stochastic_tracker
            partial_atom_graph.terminate_graph(sto_atom_id, rng)
            pending_termination.discard(sto_atom_id)
            nearest_live_ancestor = None
            for ancestor in reversed(tracker.parent_map.get(sto_atom_id, [])):
                if not tracker.is_terminated(ancestor):
                    nearest_live_ancestor = ancestor
                    break
            if nearest_live_ancestor is not None and nearest_live_ancestor in pending_termination:
                partial_atom_graph.cap_junction_bonds(sto_atom_id, nearest_live_ancestor, rng)
                return None
            if nearest_live_ancestor is None:
                continuation_gen_id = tracker._stochastic_atom_id_to_gen_id[sto_atom_id]
            else:
                continuation_gen_id = tracker._stochastic_atom_id_to_gen_id[nearest_live_ancestor]
            # The caller owns the continuation so it can preserve the exact
            # post-child/pre-continuation boundary used for ancestor rounding.
            return sto_atom_id, nearest_live_ancestor, continuation_gen_id

        while True:
            tracker = partial_atom_graph.stochastic_tracker
            unterminated_sto_atom_ids = tracker.get_unterminated_sto_atom_ids()
            if not unterminated_sto_atom_ids:
                if not partial_atom_graph.trigger_global_transitions(rng):
                    break
                # A -1 arm starts a fresh, independent owner timeline.
                checkpoints.clear()
                _commit_mutation()
                continue

            parent_map = tracker.parent_map

            # Finalize parked instances whose subtree finished, deepest first
            # (a child's caps credit its ancestors before those are decided).
            finalized = None
            if deferred_continuation is None:
                for sto_atom_id in sorted(pending_termination, key=lambda i: len(parent_map.get(i, [])), reverse=True):
                    if any(sto_atom_id in parent_map.get(d, []) for d in unterminated_sto_atom_ids):
                        continue
                    finalized = sto_atom_id
                    break
            if finalized is not None:
                if _DECISION_TRACE is not None:
                    _DECISION_TRACE.append({
                        "kind": "finalize", "id": finalized,
                        "gen": tracker._stochastic_atom_id_to_gen_id[finalized],
                        "expected": tracker._sto_atom_id_expected_molw[finalized],
                        "actual": tracker._sto_atom_id_actual_molw[finalized],
                    })
                # Finish the child's own level, then defer its ancestor-level
                # continuation for one pass.  The nearest live ancestor must be
                # tested in the post-child/pre-continuation state: finalization
                # can expose a structural residual that crosses the ancestor,
                # while its retained owner checkpoint is still the true state
                # before the unit which spawned this child.
                continuation = _finalize_pending(finalized)
                checkpoints.pop(finalized, None)
                if continuation is not None:
                    owner_sto_atom_id, continuation_level, continuation_gen_id = continuation
                    if continuation_level is None:
                        # No live owner can cross; this is an inter-object/root
                        # continuation, not an owner-level epoch.
                        _new_sto_atom_id, transition_success = partial_atom_graph.transition_graph(
                            owner_sto_atom_id,
                            continuation_gen_id,
                            rng,
                        )
                        if transition_success:
                            _commit_mutation()
                    else:
                        deferred_continuation = continuation
                continue

            growable = [i for i in unterminated_sto_atom_ids if i not in pending_termination]
            if not growable:
                # All live instances are parked: the deepest one has no live
                # descendants, so the next pass finalizes it.
                continue

            # A deferred child exit belongs to the nearest live ancestor and
            # must run before any unrelated growth. Otherwise active is the
            # deepest live non-parked instance (every descendant lists all of
            # its ancestors, so one pass suffices).
            if deferred_continuation is not None:
                active_sto_atom_id = deferred_continuation[1]
            else:
                active_sto_atom_id = growable[0]
                for sto_atom_id in growable:
                    if sto_atom_id in parent_map:
                        for ancestor in parent_map[sto_atom_id]:
                            if active_sto_atom_id == ancestor:
                                active_sto_atom_id = sto_atom_id

            if (
                deferred_continuation is None
                and len(partial_atom_graph.get_open_half_bonds(active_sto_atom_id)[1]) == 0
            ):
                # A live instance with no open half-bonds can never grow, transition,
                # or terminate: retire it so the remaining instances and -1 arms
                # continue instead of truncating the whole molecule. Only a
                # CHAIN-LEVEL (parentless) instance dying below its target makes the
                # chain non-representative — that warning drives create_ensemble's
                # discard. A nested arm (parented instance) that structurally
                # dead-ends below its own drawn target retires silently: the chain
                # completes on target regardless, and warning here made
                # create_ensemble discard every chain of such architectures.
                if (not partial_atom_graph.stochastic_tracker.parent_map.get(active_sto_atom_id)
                        and partial_atom_graph.stochastic_tracker._sto_atom_id_expected_molw[active_sto_atom_id] > 0
                        and partial_atom_graph.stochastic_tracker._sto_atom_id_actual_molw[active_sto_atom_id]
                        < partial_atom_graph.stochastic_tracker._sto_atom_id_expected_molw[active_sto_atom_id]):
                    warnings.warn(PossibleNonRepresentativePolymerChain(), stacklevel=1)
                if _DECISION_TRACE is not None:
                    _DECISION_TRACE.append({
                        "kind": "retire", "id": active_sto_atom_id,
                        "gen": tracker._stochastic_atom_id_to_gen_id[active_sto_atom_id],
                        "expected": tracker._sto_atom_id_expected_molw[active_sto_atom_id],
                        "actual": tracker._sto_atom_id_actual_molw[active_sto_atom_id],
                    })
                partial_atom_graph.stochastic_tracker.terminate(active_sto_atom_id)
                checkpoints.pop(active_sto_atom_id, None)
                continue

            # Track per-instance PROJECTED mass gains for every growable
            # instance: descendant growth leaves the projection invariant, so
            # recorded gains reflect X-level composition steps (new units, new
            # arms) — exactly what a crossing and the lookahead must bound.
            # An instance's FIRST observed projection doubles as its lookahead
            # gain floor: it is one unit's worth of content at that level
            # (including freshly spawned arm expectations), i.e. the scale of
            # a step the loop has not had the chance to observe yet. Observed
            # gains alone miss two real cases: an instance that idles behind a
            # growing sibling records only zero gains, and a backbone whose
            # only in-loop "gains" are arm rounding residuals never sees the
            # unit-with-arms jump that actually crosses it.
            proj_now = {}
            for sto_atom_id in growable:
                projected = _projected_molw(
                    tracker,
                    unterminated_sto_atom_ids,
                    sto_atom_id,
                )
                proj_now[sto_atom_id] = projected
                if sto_atom_id in last_checked_proj:
                    step_gain = projected - last_checked_proj[sto_atom_id]
                    max_step_gain[sto_atom_id] = max(step_gain, max_step_gain.get(sto_atom_id, 0.0))
                else:
                    gain_floor[sto_atom_id] = projected
                last_checked_proj[sto_atom_id] = projected

            # Resolve the deepest crossing first.  A descendant owns the
            # mutation boundary for its growth; ancestors see that subtree at
            # its signed expected-mass credit and therefore cannot legitimately
            # consume the descendant's checkpoint. The termination-MW estimate is only
            # recomputed once an instance is plausibly near its target (its
            # previous estimate serves as the margin; the active instance
            # keeps the exact per-iteration check).
            crossing_sto_atom_id = None
            crossing_projected = None
            current_deferred_junction = None
            crossing_candidates = (
                [deferred_continuation[1]]
                if deferred_continuation is not None
                else sorted(
                    growable,
                    key=lambda i: len(parent_map.get(i, [])),
                    reverse=True,
                )
            )
            for sto_atom_id in crossing_candidates:
                expected_i = tracker._sto_atom_id_expected_molw[sto_atom_id]
                if expected_i < 0:
                    continue
                cached_margin = avg_termination_cache.get(sto_atom_id)
                if (
                    sto_atom_id != active_sto_atom_id
                    and cached_margin is not None
                    and proj_now[sto_atom_id] + cached_margin < expected_i
                ):
                    continue
                current_deferred_junction = None
                if (
                    deferred_continuation is not None
                    and deferred_continuation[1] == sto_atom_id
                ):
                    current_deferred_junction = (
                        deferred_continuation[0],
                        sto_atom_id,
                    )
                own_termination_weight, avg_termination_weight = _total_termination_mw(
                    partial_atom_graph,
                    tracker,
                    unterminated_sto_atom_ids,
                    sto_atom_id,
                    current_deferred_junction,
                )
                own_termination_cache[sto_atom_id] = own_termination_weight
                avg_termination_cache[sto_atom_id] = avg_termination_weight
                if proj_now[sto_atom_id] + avg_termination_weight >= expected_i:
                    crossing_sto_atom_id = sto_atom_id
                    crossing_projected = proj_now[sto_atom_id] + avg_termination_weight
                    break

            if crossing_sto_atom_id is not None:
                # Choose the timeline (overshoot = current state, undershoot =
                # its immediately preceding owner-level boundary), then PARK
                # the instance. Its own level stops growing; live descendants
                # keep growing to
                # their own targets, and _finalize_pending fires its caps once
                # the subtree is done. termination_flag: 0 -> always overshoot,
                # 1 -> always undershoot, None -> stochastic rounding (unbiased
                # projected mean matching the target).
                expected_molw = tracker._sto_atom_id_expected_molw[crossing_sto_atom_id]
                caps_molw = avg_termination_cache[crossing_sto_atom_id]
                snapshot_valid = False
                projected_under = None
                checkpoint = checkpoints.get(crossing_sto_atom_id)
                if (
                    checkpoint is not None
                    and checkpoint["epoch"] + 1
                    == owner_epochs.get(crossing_sto_atom_id, 0)
                ):
                    snapshot_graph = checkpoint["graph"]
                    snapshot_tracker = snapshot_graph.stochastic_tracker
                    if (
                        crossing_sto_atom_id in snapshot_tracker._sto_atom_id_actual_molw
                        and not snapshot_tracker.is_terminated(crossing_sto_atom_id)
                    ):
                        snapshot_live_ids = snapshot_tracker.get_unterminated_sto_atom_ids()
                        _under_own_mw, under_caps_molw = _total_termination_mw(
                            snapshot_graph,
                            snapshot_tracker,
                            snapshot_live_ids,
                            crossing_sto_atom_id,
                            checkpoint["deferred_junction"],
                        )
                        projected_under = (
                            _projected_molw(
                                snapshot_tracker,
                                snapshot_live_ids,
                                crossing_sto_atom_id,
                            )
                            + under_caps_molw
                        )
                        snapshot_valid = projected_under < expected_molw
                if termination_flag == 0:
                    adopt_overshoot = True
                elif not snapshot_valid:
                    owner_epoch = owner_epochs.get(crossing_sto_atom_id, 0)
                    if termination_flag == 1 and (
                        owner_epoch == 0 or projected_under is not None
                    ):
                        # The first state (or the immediately preceding owner
                        # boundary) is already at/over target: no undershoot
                        # timeline exists for this instance. A nested residual
                        # can still be absorbed by an ancestor, and even a
                        # parked root can later move below target when a child
                        # adopts its own undershoot. Defer the warning until the
                        # final parentless mass proves the requested undershoot
                        # was actually impossible.
                        if not tracker.parent_map.get(crossing_sto_atom_id):
                            forced_overshoot_no_boundary.add(crossing_sto_atom_id)
                    elif owner_epoch > 0 and projected_under is None:
                        # At least one owner step occurred, so absence of its
                        # exact predecessor is a genuine lookahead miss.
                        warnings.warn(UndershootSnapshotMissed(), stacklevel=1)
                    adopt_overshoot = True
                elif termination_flag == 1:
                    adopt_overshoot = False
                else:
                    span = crossing_projected - projected_under
                    if span <= 0:
                        # The undone step was descendant-level (projection
                        # invariant) or otherwise massless: no real boundary.
                        adopt_overshoot = True
                    else:
                        p_over = max(0.0, min(1.0, (expected_molw - projected_under) / span))
                        adopt_overshoot = rng.random() < p_over
                if _DECISION_TRACE is not None:
                    _DECISION_TRACE.append({
                        "kind": "crossing", "id": crossing_sto_atom_id,
                        "gen": tracker._stochastic_atom_id_to_gen_id[crossing_sto_atom_id],
                        "expected": expected_molw, "caps": caps_molw,
                        "proj_over": crossing_projected, "proj_under": projected_under,
                        "snapshot_valid": snapshot_valid, "adopt_overshoot": adopt_overshoot,
                        "flag": termination_flag,
                    })
                adopted_deferred_junction = None
                if not adopt_overshoot:
                    adopted_deferred_junction = checkpoint["deferred_junction"]
                    partial_atom_graph = checkpoint["graph"]
                    # Deepcopy clones the tracker's generator.  Keep consuming
                    # the caller's already-advanced stream; rewinding it would
                    # replay the rejected over-step and can loop forever.
                    partial_atom_graph.stochastic_tracker._rng = rng
                    partial_atom_graph.stochastic_tracker.mark_path_conditional()
                    pending_termination = set(checkpoint["pending"])
                    max_step_gain = dict(checkpoint["max_step_gain"])
                    gain_floor = dict(checkpoint["gain_floor"])
                    last_checked_proj = dict(checkpoint["last_checked_proj"])
                    own_termination_cache = dict(checkpoint["own_termination_cache"])
                    avg_termination_cache = dict(checkpoint["avg_termination_cache"])
                    owner_epochs = dict(checkpoint["owner_epochs"])
                    forced_overshoot_no_boundary = set(
                        checkpoint["forced_overshoot_no_boundary"]
                    )
                    restored_tracker = partial_atom_graph.stochastic_tracker
                    restored_live = set(restored_tracker.get_unterminated_sto_atom_ids())
                    checkpoints = {
                        owner: prior
                        for owner, prior in checkpoint["checkpoints"].items()
                        if (
                            owner in restored_live
                            and prior["epoch"] + 1 == owner_epochs.get(owner, 0)
                        )
                    }
                else:
                    adopted_deferred_junction = current_deferred_junction
                    checkpoints.pop(crossing_sto_atom_id, None)
                if (
                    deferred_continuation is not None
                    and deferred_continuation[1] == crossing_sto_atom_id
                ):
                    # This child exit was either replaced by a cap on the over
                    # timeline or disappeared when an older under state won.
                    deferred_continuation = None
                # Parking changes control state only, not the mass epoch.
                pending_termination.add(crossing_sto_atom_id)
                if (
                    adopted_deferred_junction is not None
                    and adopted_deferred_junction[1] == crossing_sto_atom_id
                ):
                    # The undershoot timeline ends immediately before the
                    # ancestor continuation.  Parking redirects that exact
                    # junction to the ancestor's end group.
                    partial_atom_graph.cap_junction_bonds(
                        adopted_deferred_junction[0],
                        crossing_sto_atom_id,
                        rng,
                    )
                    _commit_mutation()

            else:
                if deferred_continuation is not None:
                    owner_sto_atom_id, continuation_level, continuation_gen_id = deferred_continuation
                    deferred_junction = (owner_sto_atom_id, continuation_level)
                    observed_gain = max_step_gain.get(continuation_level)
                    lookahead_gain = max(
                        observed_gain or 0.0,
                        gain_floor.get(continuation_level, 0.0),
                    )
                    expected_i = tracker._sto_atom_id_expected_molw[continuation_level]
                    need_snapshot = (
                        termination_flag != 0
                        and expected_i >= 0
                        and (
                            observed_gain is None
                            or lookahead_gain <= 0.0
                            or proj_now[continuation_level]
                            + _LOOKAHEAD_MARGIN * lookahead_gain
                            >= expected_i - avg_termination_cache.get(continuation_level, 0.0)
                        )
                    )
                    previous_checkpoint = checkpoints.get(continuation_level)
                    if need_snapshot:
                        checkpoints[continuation_level] = _capture_checkpoint(
                            continuation_level,
                            deferred_junction,
                        )
                    _new_sto_atom_id, transition_success = partial_atom_graph.transition_graph(
                        owner_sto_atom_id,
                        continuation_gen_id,
                        rng,
                    )
                    deferred_continuation = None
                    if transition_success:
                        _advance_owner_epoch(continuation_level)
                        _commit_mutation()
                    elif need_snapshot:
                        if previous_checkpoint is None:
                            checkpoints.pop(continuation_level, None)
                        else:
                            checkpoints[continuation_level] = previous_checkpoint
                    continue

                # Lazy snapshot: only deepcopy when the NEXT step could cross
                # the active owner's threshold and an undershoot boundary is
                # actually needed (never for termination_flag == 0).
                # Adaptive: before any observed step for an instance we always
                # snapshot; afterwards a margin on its largest observed
                # projected gain is kept. A residual miss is surfaced by
                # UndershootSnapshotMissed. Owner epochs, rather than global
                # mutation age, keep the checkpoint through descendant work but
                # invalidate it immediately after a second owner-level step.
                need_snapshot = False
                if termination_flag != 0:
                    expected_i = tracker._sto_atom_id_expected_molw[active_sto_atom_id]
                    observed_gain = max_step_gain.get(active_sto_atom_id)
                    lookahead_gain = max(
                        observed_gain or 0.0,
                        gain_floor.get(active_sto_atom_id, 0.0),
                    )
                    need_snapshot = expected_i >= 0 and (
                        observed_gain is None
                        or lookahead_gain <= 0.0
                        or proj_now[active_sto_atom_id]
                        + _LOOKAHEAD_MARGIN * lookahead_gain
                        >= expected_i - avg_termination_cache.get(active_sto_atom_id, 0.0)
                    )
                if need_snapshot:
                    checkpoints[active_sto_atom_id] = _capture_checkpoint(active_sto_atom_id)
                if _DECISION_TRACE is not None:
                    _DECISION_TRACE.append({
                        "kind": "grow", "active": active_sto_atom_id,
                        "mutations": mutations, "need_snapshot": need_snapshot,
                        "proj": dict(proj_now),
                        "gains": {k: max_step_gain.get(k) for k in growable},
                    })
                try:
                    partial_atom_graph.propagate_graph(active_sto_atom_id, rng, True)
                    _advance_owner_epoch(active_sto_atom_id)
                    _commit_mutation()
                except IncompleteStochasticGeneration:
                    active_sto_gen_id = partial_atom_graph.stochastic_tracker._stochastic_atom_id_to_gen_id[active_sto_atom_id]
                    sto_atom_id, transition_success = partial_atom_graph.transition_graph(active_sto_atom_id, active_sto_gen_id, rng)
                    if transition_success:
                        _advance_owner_epoch(active_sto_atom_id)
                        _commit_mutation()
                    else:
                        # TODO: raise an error if this goes for too long in create_ensemble (indication of ill-defined string)
                        expected_mol_weights = partial_atom_graph.stochastic_tracker.sto_atom_id_expected_molw
                        actual_mol_weights = partial_atom_graph.stochastic_tracker.sto_atom_id_actual_molw
                        highest_order_sto_id = max(expected_mol_weights, key=expected_mol_weights.get)
                        highest_expected_mol_weight = expected_mol_weights[highest_order_sto_id]
                        highest_actual_mol_weight = actual_mol_weights[highest_order_sto_id]
                        if highest_actual_mol_weight < highest_expected_mol_weight:
                            warnings.warn(PossibleNonRepresentativePolymerChain(), stacklevel=1)
                        # Truncated chain: no growth or transition is possible
                        # any more. Fire the caps of every parked instance (its
                        # still-live descendants first, deepest first) so the
                        # return shape matches a completed chain, then stop and
                        # fall through to the normal finalization so the caller
                        # gets a phantom-free graph (the raw early return used
                        # to leak placeholder atoms and break create_ensemble's
                        # tuple unpack).
                        cleanup_tracker = partial_atom_graph.stochastic_tracker
                        for parked in sorted(pending_termination, key=lambda i: len(cleanup_tracker.parent_map.get(i, [])), reverse=True):
                            live_now = cleanup_tracker.get_unterminated_sto_atom_ids()
                            descendants = [d for d in live_now if parked in cleanup_tracker.parent_map.get(d, [])]
                            descendants.sort(key=lambda i: len(cleanup_tracker.parent_map.get(i, [])), reverse=True)
                            for descendant in descendants:
                                partial_atom_graph.terminate_graph(descendant, rng)
                            if not cleanup_tracker.is_terminated(parked):
                                partial_atom_graph.terminate_graph(parked, rng)
                        pending_termination.clear()
                        break

        # TODO: replace this legacy clique closure in a focused follow-up. It
        # selects placeholders by ``atomic_num`` below but attempts to traverse
        # them via the absent ``num`` attribute. Pairwise closure happens to
        # recover today's single-bond cases, but a placeholder-to-placeholder
        # junction can inherit a static placeholder edge's attributes instead
        # of the realized junction edge's attributes.
        def find_non_phantom_endpoints(G, phantom_node):

            visited = set()
            non_phantom_endpoints = []
            queue = [phantom_node]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                for neighbor in G.neighbors(current):
                    if neighbor in visited:
                        continue
                    attr = G[phantom_node][neighbor]
                    if G.nodes[neighbor].get("num") == 0:
                        # Continue traversing through zero nodes
                        queue.append(neighbor)
                    else:
                        # Found a non-zero endpoint
                        non_phantom_endpoints.append(neighbor)

            return non_phantom_endpoints, attr

        phantom_nodes = [node for node, data in partial_atom_graph.atom_graph.nodes(data=True) if data.get("atomic_num") == 0]
        processed = set()

        for phantom_node in phantom_nodes:
            if phantom_node in processed:
                continue

            # Find all non-zero endpoints from this zero node
            endpoints, attr = find_non_phantom_endpoints(partial_atom_graph.atom_graph, phantom_node)

            # Mark all zero nodes in this chain as processed
            visited = set()
            queue = [phantom_node]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                processed.add(current)

                for neighbor in partial_atom_graph.atom_graph.neighbors(current):
                    if neighbor not in visited and partial_atom_graph.atom_graph.nodes[neighbor].get("num") == 0:
                        queue.append(neighbor)

            # Connect all non-zero endpoints to each other
            for i, ep1 in enumerate(endpoints):
                for ep2 in endpoints[i + 1 :]:
                    partial_atom_graph.atom_graph.add_edge(ep1, ep2, **attr)

        # Remove all zero nodes
        partial_atom_graph.atom_graph.remove_nodes_from(phantom_nodes)

        # Reconcile hydrogen credits with the FINAL realized connectivity:
        # phantom collapse can close cliques with extra bonds, and any bond
        # created outside merge() would otherwise leave stale credits. After
        # this pass the tracked mass equals the returned molecule's mass by
        # construction.
        final_graph = partial_atom_graph.atom_graph
        for _node, data, edges in ((n, d, final_graph.edges(n, data=True)) for n, d in final_graph.nodes(data=True)):
            occupied = 0
            has_aromatic = False
            for _u, _v, edge_data in edges:
                occupied += edge_data.get(_BOND_TYPE_NAME, 1)
                if edge_data.get(_AROMATIC_NAME):
                    has_aromatic = True
            if has_aromatic:
                occupied += 1
            new_h = _infer_hydrogen_count(
                data["atomic_num"],
                data["charge"],
                occupied,
                data.get("num_explicit_h", -1),
                data.get(_AROMATIC_NAME, False),
            )
            delta = new_h - data["credited_h"]
            if delta:
                data["credited_h"] = new_h
                data["occupied_valence"] = occupied
                partial_atom_graph.stochastic_tracker.credit_hydrogen_delta(data["owner_sto_atom_id"], delta)

        # Sequence fragments are independent snapshots made before the whole
        # molecule's phantom collapse. Normalize them once here, before any of
        # create_ensemble's mol-graph/RDKit/SMILES conversion paths. Synthetic
        # sequence stubs copy every attribute of the far-side atom, so for a
        # split connector they even copy the placeholder's origin_idx; only the
        # ``connection`` key distinguishes a retained stub from an internal
        # phantom placeholder.
        self._contract_sequence_phantoms(partial_atom_graph.sequence)

        # Only report an unavailable explicit undershoot when it survives all
        # descendant rounding and final hydrogen reconciliation. Nested
        # first-step overshoots are structural quantization that a live
        # ancestor can compensate; warning at the provisional crossing made
        # correctly undershooting final chains look like failures.
        final_tracker = partial_atom_graph.stochastic_tracker
        if termination_flag == 1:
            for sto_atom_id in sorted(forced_overshoot_no_boundary):
                if (
                    sto_atom_id in final_tracker._sto_atom_id_actual_molw
                    and not final_tracker.parent_map.get(sto_atom_id)
                    and final_tracker._sto_atom_id_actual_molw[sto_atom_id]
                    > final_tracker._sto_atom_id_expected_molw[sto_atom_id] + 1e-9
                ):
                    warnings.warn(ForcedOvershootNoBoundary(), stacklevel=1)

        if not molecule_info:
            return partial_atom_graph.atom_graph
        actual_mol_weights = {}
        for instance_id in partial_atom_graph.stochastic_tracker.sto_atom_id_actual_molw:
            stochastic_id = partial_atom_graph.stochastic_tracker._stochastic_atom_id_to_gen_id[instance_id]
            try:
                actual_mol_weights[stochastic_id] += [partial_atom_graph.stochastic_tracker.sto_atom_id_actual_molw[instance_id]]
            except KeyError:
                actual_mol_weights[stochastic_id] = [partial_atom_graph.stochastic_tracker.sto_atom_id_actual_molw[instance_id]]
        distributions = {}
        for stochastic_id, distribution in partial_atom_graph.stochastic_tracker._sto_gen_id_distribution.items():
            try:
                distributions[stochastic_id] = distribution.generate_string(True)
            except AttributeError:
                break
        return (partial_atom_graph.atom_graph, partial_atom_graph.units, partial_atom_graph.bonds_idx, partial_atom_graph.sequence, actual_mol_weights, distributions)


    @staticmethod
    def _contract_sequence_phantoms(sequences):
        """Remove internal split-atom placeholders from sequence unit graphs.

        Mapped sequence connection stubs are retained and, when they hang from
        a placeholder, reattached directly to that placeholder's real anchor
        with the realized junction edge attributes preserved.
        """
        for sequence in sequences:
            for unit_graph in sequence:
                phantom_nodes = {
                    node
                    for node, data in unit_graph.nodes(data=True)
                    if data.get("atomic_num") == 0 and "connection" not in data
                }
                for component in list(nx.connected_components(unit_graph.subgraph(phantom_nodes))):
                    real_anchors = set()
                    connection_edges = []
                    for phantom_node in component:
                        for neighbor in list(unit_graph.neighbors(phantom_node)):
                            if neighbor in component:
                                continue
                            neighbor_data = unit_graph.nodes[neighbor]
                            if neighbor_data.get("atomic_num", 0) > 0:
                                real_anchors.add(neighbor)
                            elif neighbor_data.get("atomic_num") == 0 and "connection" in neighbor_data:
                                connection_edges.append((neighbor, deepcopy(unit_graph[phantom_node][neighbor])))
                            else:
                                raise RuntimeError("A sequence phantom has an unsupported boundary node. Please report this bug.")

                    if len(real_anchors) != 1:
                        raise RuntimeError(
                            f"A sequence phantom component must have exactly one real anchor, found {len(real_anchors)}. Please report this bug."
                        )
                    real_anchor = next(iter(real_anchors))
                    for connection_node, edge_data in connection_edges:
                        if unit_graph.has_edge(real_anchor, connection_node):
                            raise RuntimeError("A sequence connection is already attached to its phantom's real anchor. Please report this bug.")
                        unit_graph.add_edge(real_anchor, connection_node, **edge_data)
                    unit_graph.remove_nodes_from(component)

    @staticmethod
    def _unit_graph_with_stars(unit_graph, origin_bond_id):
        """
        Copy of `unit_graph` with one mapped star for every template bond id.
        A split-atom phantom already is the connection star, while a real
        connection atom receives a new star. Bond ids deliberately remain on
        placeholder nodes in the generative graph and its JSON export.
        """
        star_graph = unit_graph.copy()
        for node, data in unit_graph.nodes(data=True):
            bond_id = origin_bond_id.get(data["origin_idx"])
            if bond_id is not None:
                # The converter renders map numbers as connection + 1.
                if data["atomic_num"] == 0:
                    star_graph.nodes[node]["connection"] = bond_id - 1
                else:
                    star_node = ("star", node)
                    star_graph.add_node(star_node, **{"atomic_num": 0, _AROMATIC_NAME: False, "charge": 0, "connection": bond_id - 1})
                    star_graph.add_edge(node, star_node, **{_BOND_TYPE_NAME: 1, _AROMATIC_NAME: False})
        return star_graph

    @staticmethod
    def _validate_unit_psmiles_mol(unit_id, star_mol, expected_maps, expected_real_atom_count):
        """Validate a rendered unit against independent template-derived data."""
        dummy_atoms = [atom for atom in star_mol.GetAtoms() if atom.GetAtomicNum() == 0]
        actual_maps = sorted(atom.GetAtomMapNum() for atom in dummy_atoms)
        invalid_dummy_degrees = tuple(
            (atom.GetIdx(), atom.GetAtomMapNum(), atom.GetDegree())
            for atom in dummy_atoms
            if atom.GetDegree() != 1
        )
        actual_real_atom_count = sum(atom.GetAtomicNum() > 0 for atom in star_mol.GetAtoms())
        expected_maps = sorted(expected_maps)
        if actual_maps != expected_maps or invalid_dummy_degrees or actual_real_atom_count != expected_real_atom_count:
            raise InvalidUnitPSmiles(
                unit_id,
                expected_maps,
                actual_maps,
                invalid_dummy_degrees,
                expected_real_atom_count,
                actual_real_atom_count,
            )

    def create_ensemble(self, n_samples, output_format="mol_graph", ensemble_info=False, max_number_of_discarded_chains: int = 100, termination_flag: Optional[int] = None, json_file: Optional[str] = None, json_max_chains: Optional[int] = None, parallel: bool = False, n_workers: Optional[int] = None, seed: Optional[int] = None):
        """Sample an ensemble while rejecting explicitly chain-local failures.

        ``max_number_of_discarded_chains`` limits consecutive rejected paths
        (per chain in parallel mode). If the limit is reached before any
        success, the first ``DeadSamplingPath`` is re-raised (warning-only
        truncations return ``None``); after one or more successes, the
        accepted shorter ensemble is preserved.  Fatal input/model errors are
        never retried, in either mode.

        With ``ensemble_info=True`` the full :class:`EnsembleData` is returned
        (``None`` on failure); otherwise just the list of molecules.

        ``json_file`` writes the originating G2RINS string, the generative graph (with
        derived unit/bond annotations) and the ensemble data to that path as
        JSON; chains and sequences are written as SMILES regardless of
        ``output_format``. ``json_max_chains``
        caps only the number of chains stored in the file (default ``None`` =
        all); statistics always cover every sampled chain.

        ``parallel=False`` (the default) samples everything in this process.
        ``parallel=True`` samples chains in ``n_workers`` subprocesses:
        ``None`` picks ``max(1, cpu_count - 2)`` capped by ``n_samples``, and
        ``n_workers=1`` is a deliberate escape hatch that runs the serial path
        with no multiprocessing overhead. Passing ``n_workers`` without
        ``parallel=True`` raises ``ValueError``. Workers are fresh interpreter
        processes; the pool skips re-importing the calling script, so
        unguarded scripts are safe on spawn platforms (Windows/macOS).
        Starting the pool costs a few seconds, so parallel pays off once the
        ensemble needs more than a few seconds of serial work; on
        power-limited laptop CPUs (hybrid P/E cores, shared thermal budget)
        the useful worker count is often below the default — tune with
        ``n_workers``.

        ``seed=None`` keeps the historical randomness (serial: the
        library-global RNG; parallel: fresh entropy). An integer seed derives
        one independent stream per chain index from ``SeedSequence(seed)``, in
        both modes: the same seed reproduces the same ensemble across modes
        and worker counts — given the same parsed graph object. A fresh parse
        relabels the graph nodes and reproduces statistics, not bytes. On
        budget exhaustion the modes keep different survivors (serial stops at
        the failure and keeps the prefix, parallel keeps every succeeding
        chain), so the cross-mode equality applies to failure-free runs.
        """

        supported_formats = {"mol", "smiles", "mol_graph"}
        molecule_format = output_format.lower()
        if molecule_format not in supported_formats:
            raise ValueError(f"Unsupported format: '{output_format}'. " f"Please choose from {list(supported_formats)}.")

        if n_workers is not None and not parallel:
            raise ValueError("n_workers only applies to parallel=True; the default mode is serial.")
        if parallel and n_workers is not None and n_workers < 1:
            raise ValueError(f"n_workers must be a positive integer, got {n_workers}.")
        if parallel and n_workers is None:
            n_workers = max(1, min((os.cpu_count() or 1) - 2, n_samples))

        # The JSON dump needs unit/sequence info even when the caller did not
        # ask for the returned ensemble information.
        collect_info = ensemble_info or json_file is not None

        total_discards = 0
        discard_reasons = Counter()
        first_discard_cause = None

        if parallel and n_workers > 1:
            # One independent stream per CHAIN INDEX (not per worker): the
            # ensemble is fixed by the seed alone, independent of n_workers,
            # chunking, and the process start method. seed=None draws a fresh
            # entropy root.
            chain_jobs = list(enumerate(np.random.SeedSequence(seed).spawn(n_samples)))
            chunk_size = max(1, n_samples // n_workers)
            chunks = [chain_jobs[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]
            with _no_main_reimport(), concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(_sample_chain_batch, self, chunk, molecule_format, collect_info, max_number_of_discarded_chains, termination_flag)
                    for chunk in chunks
                ]
                # Collected in submission order: records come back sorted by
                # chain index, and a fatal worker error re-raises here exactly
                # like on the serial path.
                chain_results = [chain_result for future in futures for chain_result in future.result()]

            records = []
            for chain_result in chain_results:
                for message, category, filename, lineno in chain_result["warnings"]:
                    warnings.warn_explicit(message, category, filename, lineno)
                total_discards += chain_result["discards"]
                discard_reasons.update(dict(chain_result["reasons"]))
                if chain_result["record"] is not None:
                    records.append(chain_result["record"])
                elif first_discard_cause is None and chain_result["first_cause"] is not None:
                    first_discard_cause = chain_result["first_cause"]

            if len(records) < n_samples:
                # Some chain exhausted its per-chain discard budget; the
                # ensemble-level verdict mirrors the serial rules (preserve
                # partial successes, re-raise the first cause on total loss).
                warnings.warn(TooManyDiscardedChains(max_number_of_discarded_chains), stacklevel=1)
                if not records:
                    warnings.warn(
                        DiscardedSamplingPaths(total_discards, tuple(discard_reasons.items())),
                        stacklevel=2,
                    )
                    if first_discard_cause is not None:
                        raise first_discard_cause
                    return None
        else:
            # Serial path (also parallel=True with n_workers=1: the documented
            # no-multiprocessing escape hatch). Without a seed the chains draw
            # from the library-global RNG as always; with one they use the
            # same per-chain streams as the parallel path, so equal seeds give
            # equal ensembles across modes and worker counts.
            chain_rngs = None
            if seed is not None:
                chain_rngs = [np.random.default_rng(seed_sequence) for seed_sequence in np.random.SeedSequence(seed).spawn(n_samples)]

            records = []
            consecutive_discards = 0
            while len(records) < n_samples:
                rng = chain_rngs[len(records)] if chain_rngs is not None else None
                sample, attempt_reasons, discard_cause, deferred_warnings = _attempt_chain(self, collect_info, termination_flag, rng)
                for caught in deferred_warnings:
                    warnings.warn_explicit(caught.message, caught.category, caught.filename, caught.lineno)
                if sample is None:
                    consecutive_discards += 1
                    total_discards += 1
                    discard_reasons.update(attempt_reasons)
                    if first_discard_cause is None and discard_cause is not None:
                        first_discard_cause = _detach_tracebacks(discard_cause)
                    if consecutive_discards >= max_number_of_discarded_chains:
                        warnings.warn(TooManyDiscardedChains(max_number_of_discarded_chains), stacklevel=1)
                        if not records:
                            # The reason breakdown must reach the caller even when
                            # no chain succeeded; the loop's tail summary is only
                            # reached through success or the partial-preserve break.
                            warnings.warn(
                                DiscardedSamplingPaths(total_discards, tuple(discard_reasons.items())),
                                stacklevel=2,
                            )
                            if first_discard_cause is not None:
                                raise first_discard_cause
                            return None
                        # Preserve already accepted chains. The warning makes the
                        # short result explicit instead of silently replacing it
                        # with None when a later chain exhausts its retry budget.
                        break
                    continue
                consecutive_discards = 0
                first_discard_cause = None
                records.append(_convert_chain(sample, molecule_format, collect_info))

        if total_discards:
            warnings.warn(
                DiscardedSamplingPaths(total_discards, tuple(discard_reasons.items())),
                stacklevel=2,
            )

        list_of_molecules = [record["molecule"] for record in records]

        if not collect_info:
            return list_of_molecules

        bond_counts = {}
        units = {}
        origin_idx_to_unit = {}
        mol_weight_lists = {}
        ensemble_distributions = {}
        list_of_sequences = []
        for record in records:
            # Units from the same template unit share their origin_idx set,
            # so a frozenset of it keys the per-unit tally directly.
            for unit, count in record["molecule_units"].items():
                origin_key = frozenset(data["origin_idx"] for _node, data in unit.nodes(data=True))
                if origin_key in origin_idx_to_unit:
                    units[origin_idx_to_unit[origin_key]] += count
                else:
                    origin_idx_to_unit[origin_key] = unit
                    units[unit] = count

            for bond, count in record["bonds"].items():
                bond_counts[bond] = bond_counts.get(bond, 0) + count

            for stochastic_id, mol_weight_list in record["mol_weights"].items():
                for mol_weight in mol_weight_list:
                    try:
                        mol_weight_lists[stochastic_id] += [mol_weight]
                    except KeyError:
                        mol_weight_lists[stochastic_id] = [mol_weight]

            for stochastic_id, distribution in record["distributions"].items():
                if stochastic_id not in ensemble_distributions:
                    ensemble_distributions[stochastic_id] = distribution

            list_of_sequences.append(record["sequences"])

        # Ensemble aggregates are format-independent: units and bonds are keyed
        # by the derived labels (which also keeps chemically identical units --
        # e.g. two Br terminators -- apart), only chains/sequences follow
        # output_format.
        labels = derive_unit_labels(self._generative_graph)
        origin_unit_id = {str(node): unit_id for node, unit_id in labels.unit_id.items()}
        origin_bond_id = {str(node): bond_id for node, bond_id in labels.bond_id.items()}
        origin_endpoint = {origin: f"{origin_unit_id[origin]}.{bond_id}" for origin, bond_id in origin_bond_id.items()}

        # The public unit representation is validated against the immutable
        # template rather than the sampled unit snapshot it renders. Besides
        # renderer defects, this catches snapshots that lost a real atom or
        # connection site at a merge watermark. This template-only contract
        # could move to a pre-flight check in a future change.
        template_unit_bond_ids = {unit_id: [] for unit_id in set(labels.unit_id.values())}
        template_unit_real_atom_counts = {unit_id: 0 for unit_id in template_unit_bond_ids}
        for node, data in self._generative_graph.nodes(data=True):
            unit_id = labels.unit_id[node]
            if data["atomic_num"] > 0:
                template_unit_real_atom_counts[unit_id] += 1
            bond_id = labels.bond_id.get(node)
            if bond_id is not None:
                template_unit_bond_ids[unit_id].append(bond_id)

        # unit_g2rins was composed against the same derivation at parse time;
        # if the graph was mutated since, omit the texts rather than mislabel.
        unit_g2rins = self._generative_graph.graph.get("unit_g2rins", {})
        if not set(unit_g2rins).issubset(origin_unit_id.values()):
            unit_g2rins = {}

        canonical_units = {}
        for unit_graph, frequency in units.items():
            # Unit fragments have dangling inter-unit valences: kekulize=False
            # (an aromatic ring at a connection point can't be kekulized in
            # isolation).
            star_mol = mol_graph_to_rdkit_mol(self._unit_graph_with_stars(unit_graph, origin_bond_id), kekulize=False)
            unit_id = origin_unit_id[next(iter(unit_graph.nodes(data=True)))[1]["origin_idx"]]
            self._validate_unit_psmiles_mol(
                unit_id,
                star_mol,
                template_unit_bond_ids[unit_id],
                template_unit_real_atom_counts[unit_id],
            )
            canonical_units[unit_id] = {"psmiles": rdkit_mol_to_smiles(star_mol), "g2rins": unit_g2rins.get(unit_id, ""), "frequency": frequency}
        canonical_units = dict(sorted(canonical_units.items(), key=lambda item: (item[0][0], int(item[0][1:]))))

        bond_records = _bond_records(bond_counts, origin_endpoint)

        if json_file is not None:
            if molecule_format == "smiles":

                def _chain_smiles(molecule):
                    return molecule

                def _unit_smiles(unit):
                    return unit

            elif molecule_format == "mol":

                def _chain_smiles(molecule):
                    return rdkit_mol_to_smiles(molecule)

                def _unit_smiles(unit):
                    return rdkit_mol_to_smiles(unit)

            else:

                def _chain_smiles(molecule):
                    return mol_graph_to_smiles(molecule)

                def _unit_smiles(unit):
                    return mol_graph_to_smiles(unit, kekulize=False)

            saved_chains = list_of_molecules if json_max_chains is None else list_of_molecules[:json_max_chains]
            json_data = {"string": self._generative_graph.graph.get("g2rins_string", "")}
            json_data.update(generative_graph_json_data(self._generative_graph))
            json_data["ensemble"] = {
                "units": canonical_units,
                "chains": [_chain_smiles(molecule) for molecule in saved_chains],
                "bonds": bond_records,
                "mol_weights": mol_weight_lists,
                "distributions": ensemble_distributions,
                "sequences": [[[_unit_smiles(unit) for unit in sequence] for sequence in chain_sequences] for chain_sequences in list_of_sequences],
            }
            with open(json_file, "w") as file_handle:
                json.dump(json_data, file_handle, indent=2)

        if ensemble_info:
            return EnsembleData(
                chains=list_of_molecules,
                units=canonical_units,
                bonds=bond_records,
                sequences=list_of_sequences,
                mol_weights=mol_weight_lists,
                distributions=ensemble_distributions,
            )
        return list_of_molecules
