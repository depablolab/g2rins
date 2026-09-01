# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import concurrent.futures
import copy
import faulthandler
import functools
import inspect
import json
import multiprocessing.spawn
import os
import pickle
import threading
import warnings
from collections import Counter, OrderedDict, deque
from collections.abc import Sequence
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

import networkx as nx
import numpy as np
from rdkit import Chem, rdBase

from ._version import version as _G2RINS_VERSION
from .convergence import ConvergenceTracker
from .nx_rdkit_mol import (
    mol_graph_to_rdkit_mol,
    mol_graph_to_smiles,
    rdkit_mol_to_smiles,
    rdkit_mol_weight,
)
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
    NoValidGenerationSource,
    PossibleNonRepresentativePolymerChain,
    TooManyDiscardedChains,
    UndershootSnapshotMissed,
    UnvalidatedGenerationSource,
    WorkerProcessFailure,
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

# Temporary parity escape hatch while the journal path is validated.
_USE_LEGACY_CHECKPOINTS = False
_VERIFY_TERMINATION_MW_CACHE = False
_USE_STATIC_SOURCE_TEMPLATES = True
_USE_DIRECT_GRAPH_MERGE = True

# Last state remains available to Python-level diagnostics and tests. Native
# crashes are covered by the optional durable JSONL sink.
_LAST_NATIVE_STATE = None
_WORKER_ENSEMBLE_CREATOR = None
_NATIVE_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_MAX_TASKS_PER_CHILD = 500
_SUPPORTS_MAX_TASKS_PER_CHILD = (
    "max_tasks_per_child"
    in inspect.signature(concurrent.futures.ProcessPoolExecutor).parameters
)
# Worker spawning temporarily changes process-global state. Serialize the full
# pool lifetime so concurrent callers cannot interleave save/restore operations
# and leave either multiprocessing's preparation hook or native-thread limits
# permanently modified.
_SPAWN_CONFIGURATION_LOCK = threading.RLock()
_ACTIVE_PARALLEL_SCHEDULER = ContextVar(
    "g2rins_active_parallel_scheduler",
    default=None,
)


def _enable_native_faulthandler():
    if not faulthandler.is_enabled():
        faulthandler.enable(all_threads=True)


def _seed_diagnostic(seed_sequence):
    if seed_sequence is None:
        return "global"
    entropy = seed_sequence.entropy
    if isinstance(entropy, np.ndarray):
        entropy = entropy.tolist()
    elif isinstance(entropy, np.generic):
        entropy = entropy.item()
    return {
        "entropy": entropy,
        "spawn_key": list(seed_sequence.spawn_key),
    }


def _native_state_publisher(chain_index, seed_sequence, mol_graph, diagnostics_path):
    """Return a stage callback that records compact pre-native-call state."""
    base_state = {
        "chain_index": chain_index,
        "seed": _seed_diagnostic(seed_sequence),
        "atom_count": mol_graph.number_of_nodes(),
        "bond_count": mol_graph.number_of_edges(),
        "g2rins_version": _G2RINS_VERSION,
        "rdkit_version": rdBase.rdkitVersion,
        "worker_pid": os.getpid(),
    }
    path = os.fspath(diagnostics_path) if diagnostics_path is not None else None

    def publish(stage):
        global _LAST_NATIVE_STATE
        state = {**base_state, "native_stage": stage}
        _LAST_NATIVE_STATE = state
        if path is None:
            return
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        payload = (json.dumps(state, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(
            path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    return publish


@contextmanager
def _single_native_thread_environment():
    """Make spawned workers import numerical libraries with one native thread."""
    with _SPAWN_CONFIGURATION_LOCK:
        previous = {
            name: os.environ.get(name) for name in _NATIVE_THREAD_ENVIRONMENT
        }
        try:
            for name in _NATIVE_THREAD_ENVIRONMENT:
                os.environ[name] = "1"
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _initialize_sampling_worker(ensemble_creator):
    """Bind one creator to a worker for all compact chain jobs it executes."""
    global _WORKER_ENSEMBLE_CREATOR
    for name in _NATIVE_THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    _enable_native_faulthandler()
    _WORKER_ENSEMBLE_CREATOR = ensemble_creator


def _sample_chain_job(
    chain_job,
    molecule_format,
    collect_info,
    max_discards,
    termination_flag,
    include_sequences=True,
    native_diagnostics_path=None,
    defer_conversion=False,
    use_repeat_units_as_source=False,
):
    """Execute one compact chain job using the creator initialized in-worker."""
    if _WORKER_ENSEMBLE_CREATOR is None:
        raise RuntimeError("sampling worker was not initialized")
    return _sample_chain_batch(
        _WORKER_ENSEMBLE_CREATOR,
        [chain_job],
        molecule_format,
        collect_info,
        max_discards,
        termination_flag,
        include_sequences,
        native_diagnostics_path,
        defer_conversion,
        use_repeat_units_as_source,
    )[0]


def _last_native_diagnostic(diagnostics_path):
    if diagnostics_path is None:
        return None
    try:
        with open(diagnostics_path, "rb") as file_handle:
            file_handle.seek(0, os.SEEK_END)
            position = file_handle.tell()
            leading_fragment = b""
            while position:
                chunk_size = min(8192, position)
                position -= chunk_size
                file_handle.seek(position)
                parts = (file_handle.read(chunk_size) + leading_fragment).split(
                    b"\n"
                )
                leading_fragment = parts[0]
                for line in reversed(parts[1:]):
                    if not line.strip():
                        continue
                    try:
                        return json.loads(line)
                    except (ValueError, TypeError, UnicodeDecodeError):
                        continue
            if leading_fragment.strip():
                try:
                    return json.loads(leading_fragment)
                except (ValueError, TypeError, UnicodeDecodeError):
                    pass
        return None
    except (OSError, ValueError, TypeError):
        return None


class _ParallelChainScheduler:
    """Own one bounded, restartable worker pool across submission batches."""

    def __init__(
        self,
        ensemble_creator,
        n_workers,
        max_worker_restarts,
        native_diagnostics_path,
    ):
        self.ensemble_creator = ensemble_creator
        self.n_workers = n_workers
        self.max_worker_restarts = max_worker_restarts
        self.native_diagnostics_path = native_diagnostics_path
        self.restart_count = 0
        self.executor = None
        self.pool_broken = False
        self._contexts = None

    def __enter__(self):
        self._contexts = ExitStack()
        try:
            self._contexts.enter_context(_single_native_thread_environment())
            self._contexts.enter_context(_no_main_reimport())
        except BaseException:
            self.close(pool_broken=True)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(pool_broken=isinstance(exc_value, BrokenProcessPool))

    def _create_executor(self):
        executor_options = {
            "max_workers": self.n_workers,
            "initializer": _initialize_sampling_worker,
            "initargs": (self.ensemble_creator,),
        }
        if _SUPPORTS_MAX_TASKS_PER_CHILD:
            executor_options["max_tasks_per_child"] = _MAX_TASKS_PER_CHILD
        self.executor = concurrent.futures.ProcessPoolExecutor(**executor_options)

    def close(self, pool_broken=False):
        if self.executor is not None:
            self.executor.shutdown(
                wait=not (pool_broken or self.pool_broken),
                cancel_futures=pool_broken or self.pool_broken,
            )
            self.executor = None
        self.pool_broken = False
        if self._contexts is not None:
            self._contexts.close()
            self._contexts = None

    def _restart(self, error):
        self.pool_broken = True
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
        self.restart_count += 1
        if self.restart_count > self.max_worker_restarts:
            raise WorkerProcessFailure(
                self.restart_count,
                _last_native_diagnostic(self.native_diagnostics_path),
            ) from error
        self._create_executor()
        self.pool_broken = False

    def records(
        self,
        chain_jobs,
        molecule_format,
        collect_info,
        max_discards,
        termination_flag,
        include_sequences,
        native_diagnostics_path,
        defer_conversion,
        use_repeat_units_as_source=False,
    ):
        if self.executor is None:
            self._create_executor()
        ordered_indices = [chain_index for chain_index, _seed in chain_jobs]
        jobs_by_index = dict(chain_jobs)
        pending = deque(chain_jobs)
        completed = {}
        next_position = 0
        max_inflight = max(1, 2 * self.n_workers)

        while next_position < len(ordered_indices):
            pool_broken = None
            inflight = {}
            try:
                while pending or inflight:
                    while pending and len(inflight) < max_inflight:
                        chain_job = pending.popleft()
                        future = self.executor.submit(
                            _sample_chain_job,
                            chain_job,
                            molecule_format,
                            collect_info,
                            max_discards,
                            termination_flag,
                            include_sequences,
                            native_diagnostics_path,
                            defer_conversion,
                            use_repeat_units_as_source,
                        )
                        inflight[future] = chain_job

                    done, _not_done = concurrent.futures.wait(
                        inflight,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    fatal_error = None
                    for future in done:
                        chain_job = inflight.pop(future)
                        try:
                            completed[chain_job[0]] = future.result()
                        except BrokenProcessPool as error:
                            pool_broken = error
                            self.pool_broken = True
                        except BaseException as error:
                            fatal_error = error

                    while (
                        next_position < len(ordered_indices)
                        and ordered_indices[next_position] in completed
                    ):
                        yield completed.pop(ordered_indices[next_position])
                        next_position += 1

                    if fatal_error is not None:
                        raise fatal_error
                    if pool_broken is not None:
                        break
            except BrokenProcessPool as error:
                pool_broken = error
                self.pool_broken = True

            if pool_broken is None:
                return
            self._restart(pool_broken)
            pending = deque(
                (chain_index, jobs_by_index[chain_index])
                for chain_index in ordered_indices[next_position:]
                if chain_index not in completed
            )


def _parallel_chain_records(
    ensemble_creator,
    chain_jobs,
    molecule_format,
    collect_info,
    max_discards,
    termination_flag,
    include_sequences,
    native_diagnostics_path,
    n_workers,
    max_worker_restarts,
    defer_conversion=False,
    scheduler=None,
    use_repeat_units_as_source=False,
):
    """Yield ordered records from bounded, restartable compact worker jobs."""
    if scheduler is not None:
        yield from scheduler.records(
            chain_jobs,
            molecule_format,
            collect_info,
            max_discards,
            termination_flag,
            include_sequences,
            native_diagnostics_path,
            defer_conversion,
            use_repeat_units_as_source,
        )
        return
    with _ParallelChainScheduler(
        ensemble_creator,
        n_workers,
        max_worker_restarts,
        native_diagnostics_path,
    ) as local_scheduler:
        yield from local_scheduler.records(
            chain_jobs,
            molecule_format,
            collect_info,
            max_discards,
            termination_flag,
            include_sequences,
            native_diagnostics_path,
            defer_conversion,
            use_repeat_units_as_source,
        )


def _with_parallel_convergence_scheduler(method):
    signature = inspect.signature(method)

    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if (
            not values["parallel"]
            or values["batch_size"] < 1
            or values["max_samples"] < 1
            or (
                values["n_workers"] is not None
                and values["n_workers"] < 2
            )
        ):
            return method(*args, **kwargs)
        n_workers = values["n_workers"]
        if n_workers is None:
            n_workers = max(
                1,
                min(
                    (os.cpu_count() or 1) - 2,
                    values["batch_size"],
                    values["max_samples"],
                ),
            )
        if n_workers < 2:
            return method(*args, **kwargs)
        with _ParallelChainScheduler(
            values["self"],
            n_workers,
            values["max_worker_restarts"],
            values["native_diagnostics_path"],
        ) as scheduler:
            token = _ACTIVE_PARALLEL_SCHEDULER.set(scheduler)
            try:
                return method(*args, **kwargs)
            finally:
                _ACTIVE_PARALLEL_SCHEDULER.reset(token)

    return wrapped


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


def _static_total_bond(generative_graph, node_idx: int) -> int:
    """Occupied valence contributed by a template unit's static real bonds."""
    total_bond = 0
    has_aromatic = False
    template_nodes = generative_graph.nodes
    for _u, v, attr in generative_graph.out_edges(node_idx, data=True):
        if attr.get("static") and template_nodes[v].get("atomic_num", 0) > 0:
            total_bond += attr.get(_BOND_TYPE_NAME, 0)
            if attr.get(_AROMATIC_NAME):
                has_aromatic = True
    return total_bond + int(has_aromatic)


def _static_fragment_graph(static_graph, source):
    """Undirected topology instantiated by add_static_sub_graph(source)."""
    fragment = nx.Graph()
    fragment.add_node(source)
    for u, v, _key in nx.edge_dfs(static_graph, source=source):
        fragment.add_edge(u, v)
    return fragment


def _static_real_anchors(fragment, node_idx):
    if fragment.nodes[node_idx].get("atomic_num", 0) > 0:
        return [node_idx]
    anchors = []
    seen = {node_idx}
    queue = [node_idx]
    while queue:
        current = queue.pop()
        for neighbor in fragment.neighbors(current):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            if fragment.nodes[neighbor].get("atomic_num", 0) > 0:
                anchors.append(neighbor)
            else:
                queue.append(neighbor)
    return anchors


def _collapse_phantom_nodes(graph):
    """Replace each connected phantom component with its realized bond(s)."""
    phantom_nodes = {
        node
        for node, data in graph.nodes(data=True)
        if data.get("atomic_num") == 0
    }
    processed = set()

    for phantom_node in phantom_nodes:
        if phantom_node in processed:
            continue

        component = set()
        endpoints = []
        queue = deque([phantom_node])
        while queue:
            current = queue.popleft()
            if current in component:
                continue
            component.add(current)
            processed.add(current)
            for neighbor in graph.neighbors(current):
                if neighbor in phantom_nodes:
                    if neighbor not in component:
                        queue.append(neighbor)
                elif neighbor not in endpoints:
                    endpoints.append(neighbor)

        # A realized junction normally joins two phantom placeholders. Its
        # edge attributes describe the bond that remains after the placeholders
        # disappear; static real-to-phantom anchor edges must not override it.
        candidate_attrs = [
            dict(attrs)
            for _u, _v, attrs in graph.subgraph(component).edges(data=True)
        ]
        if not candidate_attrs and len(endpoints) >= 2:
            candidate_attrs = [
                dict(graph[phantom][endpoint])
                for phantom in component
                for endpoint in graph.neighbors(phantom)
                if endpoint not in phantom_nodes
            ]

        if len(endpoints) >= 2:
            bond_attrs = candidate_attrs[0] if candidate_attrs else {}
            if any(attrs != bond_attrs for attrs in candidate_attrs[1:]):
                raise RuntimeError(
                    "Cannot collapse a phantom component with inconsistent bond attributes"
                )
            for index, endpoint in enumerate(endpoints):
                for other_endpoint in endpoints[index + 1 :]:
                    if endpoint != other_endpoint:
                        graph.add_edge(endpoint, other_endpoint, **bond_attrs)

    graph.remove_nodes_from(phantom_nodes)


def _prepare_termination_fragment_masses(generative_graph, static_graph):
    """Precompute invariant net cap-fragment masses by target and bond order."""
    keys = {
        (target, attr.get(_BOND_TYPE_NAME, 1))
        for _source, target, attr in generative_graph.edges(data=True)
        if attr.get(_TERMINATION_NAME, 0) > 0
    }
    masses = {}
    fragment_cache = {}
    for target, attach_order in keys:
        if target not in fragment_cache:
            fragment_cache[target] = _static_fragment_graph(static_graph, target)
            nx.set_node_attributes(
                fragment_cache[target],
                {
                    node: dict(generative_graph.nodes[node])
                    for node in fragment_cache[target]
                },
            )
        fragment = fragment_cache[target]
        anchors = set(_static_real_anchors(fragment, target))
        mass = 0.0
        for node, data in fragment.nodes(data=True):
            atomic_number = data.get("atomic_num", 0)
            if atomic_number <= 0:
                continue
            occupied = _static_total_bond(generative_graph, node)
            if node in anchors:
                occupied += attach_order
            num_h = _infer_hydrogen_count(
                atomic_number,
                data.get("charge", 0),
                occupied,
                data.get("num_explicit_h", -1),
                data.get(_AROMATIC_NAME, False),
            )
            mass += atomic_masses[atomic_number] + num_h * atomic_masses[1]
        masses[(target, attach_order)] = mass
    return masses

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


@dataclass(frozen=True)
class _HalfBondTemplate:
    node_idx: Any
    weight: float
    molar_amounts: Any
    gen_hierarchy: int
    stochastic_id: int
    parent: int
    mode_entries: tuple[tuple[str, tuple, tuple, tuple], ...]
    special_targets: tuple[tuple[Any, dict], ...]
    special_weights: tuple[float, ...]


@dataclass(frozen=True)
class _StaticNodeTemplate:
    origin_idx: Any
    atom_attrs: tuple[tuple[str, Any], ...]
    static_total_bond: int
    stochastic_id_tree: tuple[int, ...]
    half_bond: _HalfBondTemplate


@dataclass(frozen=True)
class _StaticSourceTemplate:
    nodes: tuple[_StaticNodeTemplate, ...]
    edges: tuple[tuple[int, int, tuple[tuple[str, Any], ...]], ...]


class _HalfAtomBond:
    @staticmethod
    def prepare_template(node_idx, graph):
        node_data = graph.nodes[node_idx]
        stochastic_id = node_data["stochastic_id_tree"][0]
        mode_attrs = {}
        mode_targets = {}
        mode_molar_amounts = {}
        special_targets = []
        special_weights = []

        for _u, target, edge_data in graph.out_edges(node_idx, data=True):
            if edge_data["static"]:
                continue
            for mode in _NON_STATIC_ATTR:
                if edge_data[mode] > 0:
                    mode_attrs.setdefault(mode, []).append(edge_data)
                    mode_targets.setdefault(mode, []).append(target)
                    mode_molar_amounts.setdefault(mode, []).append(
                        graph.nodes[target]["unit_molar_amounts"]
                    )

            if edge_data[_TRANSITION_NAME] > 0:
                target_tree = graph.nodes[target]["stochastic_id_tree"]
                target_stochastic_id = target_tree[0]
                if (
                    stochastic_id in target_tree[1:]
                    and edge_data.get(_EDGE_STOCHASTIC_ID_NAME)
                    == target_stochastic_id
                    and target_stochastic_id != -1
                ):
                    special_targets.append((target, edge_data))
                    special_weights.append(
                        edge_data[_TRANSITION_NAME]
                        * graph.nodes[target]["unit_molar_amounts"][
                            target_stochastic_id
                        ]
                    )

        mode_entries = tuple(
            (
                mode,
                tuple(mode_attrs[mode]),
                tuple(mode_targets[mode]),
                tuple(mode_molar_amounts[mode]),
            )
            for mode in mode_attrs
        )
        return _HalfBondTemplate(
            node_idx=node_idx,
            weight=node_data["gen_weight"],
            molar_amounts=node_data["unit_molar_amounts"],
            gen_hierarchy=node_data["gen_hierarchy"],
            stochastic_id=stochastic_id,
            parent=node_data["stochastic_id_tree"][1],
            mode_entries=mode_entries,
            special_targets=tuple(special_targets),
            special_weights=tuple(special_weights),
        )

    def __init__(
        self,
        atom_idx: int,
        node_idx: str,
        graph,
        stochastic_tracker,
        rng,
        template=None,
    ):
        if template is None:
            template = self.prepare_template(node_idx, graph)
        self.atom_idx: int = atom_idx
        self.node_idx: str = template.node_idx
        self.weight: float = template.weight
        self.molar_amounts: float = template.molar_amounts
        self.gen_hierarchy: int = template.gen_hierarchy
        self.stochastic_id: int = template.stochastic_id
        self.parent: int = template.parent
        self._graph = graph

        # Lists and maps remain instance-owned: transition promotion replaces
        # and filters them later in the sample lifecycle.
        self._mode_attr_map = {
            mode: list(attrs) for mode, attrs, _targets, _molar in template.mode_entries
        }
        self._mode_target_map = {
            mode: list(targets) for mode, _attrs, targets, _molar in template.mode_entries
        }
        self._mode_target_molar_amounts_map = {
            mode: list(molar) for mode, _attrs, _targets, molar in template.mode_entries
        }

        self._special_target = None
        if template.special_targets:
            chosen = stochastic_tracker.choose(
                rng,
                len(template.special_targets),
                np.asarray(template.special_weights, dtype=float),
                "nested special-target selection",
            )
            self._special_target = template.special_targets[chosen]

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
        prepared_distributions=None,
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
        self._active_transactions = []

        if prepared_distributions is None:
            for _node_idx, data in generative_graph.nodes(data=True):
                for index, stochastic_vector in enumerate(
                    data["molecular_weight_distribution"]
                ):
                    distribution = StochasticDistribution.from_serial_vector(
                        list(stochastic_vector)
                    )
                    if distribution is not None:
                        self._register_sto_gen_id(index, distribution)
                break
        else:
            for sto_gen_id, distribution in prepared_distributions.items():
                self._register_sto_gen_id(sto_gen_id, distribution)

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
        self._journal_scalar("_path_is_conditional")
        self._path_is_conditional = True

    def _journal_scalar(self, name):
        for transaction in self._active_transactions:
            transaction.record_tracker_scalar(name, getattr(self, name))

    def _journal_mapping_key(self, name, key):
        mapping = getattr(self, name)
        for transaction in self._active_transactions:
            transaction.record_tracker_mapping_key(name, key, mapping)

    def _journal_set_member(self, name, value):
        values = getattr(self, name)
        for transaction in self._active_transactions:
            transaction.record_tracker_set_member(name, value, value in values)

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
            self._journal_scalar("_path_is_conditional")
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

        self._journal_mapping_key("_stochastic_atom_id_to_gen_id", new_sto_atom_id)
        self._journal_mapping_key("_stochastic_gen_id_to_atom_id", sto_gen_id)
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
            self._journal_mapping_key("_sto_atom_id_expected_molw", new_sto_atom_id)
            self._sto_atom_id_expected_molw[new_sto_atom_id] = new_molw
        else:
            self._journal_mapping_key("_sto_atom_id_expected_molw", new_sto_atom_id)
            self._sto_atom_id_expected_molw[new_sto_atom_id] = -1
        self._journal_mapping_key("_sto_atom_id_actual_molw", new_sto_atom_id)
        try:
            self._sto_atom_id_actual_molw[new_sto_atom_id] = self._parent_molw[sto_gen_id]
        except KeyError:
            self._sto_atom_id_actual_molw[new_sto_atom_id] = 0

        if is_nested_parent:
            self._journal_mapping_key("parent_map", new_sto_atom_id)
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
            self._journal_mapping_key("parent_map", new_sto_atom_id)
            self.parent_map[new_sto_atom_id] = parent_list
            # Ancestors materialized implicitly above (growth entering a deeply
            # nested unit directly) need their own chains recorded: without
            # them add_molw never credits the outer levels and
            # pending-termination finalization treats the intermediate as
            # parentless, firing its continuation at the wrong level.
            for k, ancestor_id in enumerate(parent_list):
                if k > 0 and ancestor_id not in self.parent_map:
                    self._journal_mapping_key("parent_map", ancestor_id)
                    self.parent_map[ancestor_id] = parent_list[:k]

        return new_sto_atom_id, parent_list

    def add_molw(self, sto_atom_id, atomic_num, total_atom_bonds, stochastic_id_tree, num_explicit_h=-1, charge=0, aromatic=False):
        # A bracket atom that wrote its H count (e.g. [nH]) fixes num_H exactly;
        # other atoms infer it from the charge- and aromaticity-aware valence so
        # the tracked MW matches the RDKit molecule ([NH3+] binds more hydrogens
        # than the neutral valence implies, thiophene sulfur binds none).
        num_H = _infer_hydrogen_count(atomic_num, charge, total_atom_bonds, num_explicit_h, aromatic)
        added_mass = atomic_masses[atomic_num] + num_H * atomic_masses.get(1)
        self._journal_mapping_key("_sto_atom_id_actual_molw", sto_atom_id)
        self._sto_atom_id_actual_molw[sto_atom_id] += added_mass

        # Ancestors are credited the exact same mass as the instance itself: an
        # asymmetric (unclamped) hydrogen term here subtracted phantom mass from
        # every ancestor of over-coordinated atoms, so parents overshot their
        # target before should_terminate fired.
        if self.parent_map.get(sto_atom_id):
            for parent_sto_atom_id in self.parent_map.get(sto_atom_id):
                self._journal_mapping_key("_sto_atom_id_actual_molw", parent_sto_atom_id)
                self._sto_atom_id_actual_molw[parent_sto_atom_id] += added_mass
        return num_H

    def credit_hydrogen_delta(self, sto_atom_id, delta_h):
        """Adjust an instance's tracked mass when a realized bond changes an
        atom's hydrogen count (owner and ancestors move together, mirroring
        add_molw)."""
        if not delta_h:
            return
        mass = delta_h * atomic_masses.get(1)
        self._journal_mapping_key("_sto_atom_id_actual_molw", sto_atom_id)
        self._sto_atom_id_actual_molw[sto_atom_id] += mass
        for parent_sto_atom_id in self.parent_map.get(sto_atom_id, []):
            self._journal_mapping_key("_sto_atom_id_actual_molw", parent_sto_atom_id)
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
            sto_gen_id = self._stochastic_atom_id_to_gen_id[sto_atom_id]
            self._journal_mapping_key("_parent_molw", sto_gen_id)
            self._parent_molw[sto_gen_id] = 0
        except KeyError:
            pass
        self._journal_set_member("_terminated_sto_atom_ids", sto_atom_id)
        self._terminated_sto_atom_ids.add(sto_atom_id)

    def draw_mw(self, sto_gen_id, sto_atom_id=None, rng=None) -> None | float:
        if sto_gen_id is None:
            sto_gen_id = self._stochastic_atom_id_to_gen_id[sto_atom_id]
        if rng is None:
            rng = self._rng

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

    ``chains`` and ``sequences`` follow the requested ``output_format``; the
    ensemble aggregates are template-level and format-independent. ``units``
    maps each derived unit_id (see :func:`g2rins.derive_unit_labels`) to
    ``{"psmiles", "g2rins", "subgraph", "count"}``, where ``subgraph`` is a
    detached copy of the unit's static subgraph of the generative graph
    (original node ids, static edges only, ``unit_id`` stamped on the copy's
    nodes). ``bonds`` is a list of undirected linkage records
    ``{"labels": ["I0.1", "R0.1"], "nodes": [id, id], "count": n}``:
    ``labels`` endpoints are ``"<unit_id>.<bond_id>"`` strings (parse with
    ``endpoint.rsplit(".", 1)``) sorted so the same linkage always prints
    identically, and ``nodes`` holds the generative-graph node ids of the
    same two connection atoms, aligned with ``labels``. Labels survive a
    fresh parse of the same string; node ids are only valid for this parsed
    graph.
    """

    chains: list
    units: dict
    bonds: list
    sequences: list
    mol_weights: dict
    distributions: dict
    molecular_weights: list


@dataclass
class ConvergedEnsembleData(EnsembleData):
    """Result of :meth:`EnsembleCreator.create_ensemble_until_converged`."""

    converged: bool
    n_batches: int
    convergence_settings: dict
    convergence_trace: list
    number_average_molecular_weight: float
    weight_average_molecular_weight: float
    dispersity: float


@dataclass
class ConvergenceCheckpoint:
    """Serializable state for resuming seeded convergence at a batch boundary."""

    next_chain_index: int
    accepted_count: int
    batch_index: int
    seed: int
    mass_sum: float
    mass_square_sum: float
    contact_counts: dict
    contact_total: int
    aggregate: EnsembleData
    convergence_history: list
    retained_records: tuple
    retained_seen: int
    reservoir_rng_state: dict | None
    settings: dict
    policy: str = "full"


class _MetadataLevel(IntEnum):
    """Amount of metadata carried between private sampling stages."""

    NONE = 0
    COUNTS = 1
    COMPACT_SEQUENCES = 2
    FULL_LEGACY = 3


def _metadata_level(value):
    """Accept historical private booleans while using explicit levels internally."""
    if isinstance(value, _MetadataLevel):
        return value
    if value is None or isinstance(value, bool):
        return _MetadataLevel.FULL_LEGACY if value else _MetadataLevel.NONE
    return _MetadataLevel(value)


@dataclass
class _CompactMetadata:
    unit_counts: dict
    bond_counts: dict
    labeled_bond_counts: dict
    occurrences: tuple
    sequences: tuple
    unit_prototypes: dict

    def materialize_sequences(self):
        def materialize_unit(occurrence_id):
            occurrence = self.occurrences[occurrence_id]
            unit = deepcopy(self.unit_prototypes[occurrence.prototype_key])
            for (
                parent_origin,
                connection_id,
                node_attributes,
                edge_attributes,
            ) in occurrence.connections:
                parent_node = next(
                    node
                    for node, data in unit.nodes(data=True)
                    if str(data["origin_idx"]) == parent_origin
                )
                placeholder = "C" + str(connection_id)
                unit.add_node(placeholder, **node_attributes)
                unit.nodes[placeholder]["atomic_num"] = 0
                unit.nodes[placeholder]["connection"] = connection_id
                unit.add_edge(parent_node, placeholder, **edge_attributes)
            return unit

        return [
            [materialize_unit(occurrence_id) for occurrence_id in sequence]
            for sequence in self.sequences
        ]


@dataclass
class _SampledMolecule:
    graph: nx.Graph
    metadata: _CompactMetadata
    mol_weights: dict
    distributions: dict
    legacy_units: dict | None = None
    legacy_sequences: list | None = None

    def __getitem__(self, index):
        legacy = (
            self.graph,
            self.legacy_units,
            self.metadata.bond_counts,
            self.legacy_sequences,
            self.mol_weights,
            self.distributions,
        )
        return legacy[index]


@dataclass(frozen=True)
class _ChainRecord:
    molecule: Any
    unit_counts: dict | None
    bonds: dict | None
    contact_counts: dict | None
    sequences: list | None
    mol_weights: dict | None
    distributions: dict | None
    molecular_weight: float | None
    legacy_units: dict | None = None

    def __getitem__(self, name):
        """Preserve the historical callback/test mapping interface."""
        if name == "molecule_units":
            return self.legacy_units
        return getattr(self, name)

    def callback_record(self):
        return {
            "molecule": self.molecule,
            "molecule_units": self.legacy_units,
            "bonds": self.bonds,
            "sequences": self.sequences,
            "mol_weights": self.mol_weights,
            "distributions": self.distributions,
            "molecular_weight": self.molecular_weight,
        }


@dataclass(frozen=True)
class _DeferredChainRecord:
    sample: _SampledMolecule
    molecular_weight: float
    chain_index: int | None
    seed_sequence: Any
    native_diagnostics_path: Any


def _bond_endpoint_sort_key(endpoint):
    unit_id, bond_id = endpoint.rsplit(".", 1)
    return unit_id[0], int(unit_id[1:]), int(bond_id)


def _labeled_bond_counts(bond_counts, origin_endpoint):
    """Canonical endpoint-label/node pairs with both directions merged."""
    merged = {}
    for (origin_u, origin_v), count in bond_counts.items():
        pair = tuple(
            sorted(
                ((origin_endpoint[origin_u], origin_u), (origin_endpoint[origin_v], origin_v)),
                key=lambda endpoint: _bond_endpoint_sort_key(endpoint[0]),
            )
        )
        merged[pair] = merged.get(pair, 0) + count
    return merged


def _bond_records_from_labeled(labeled_bond_counts):
    return [
        {"labels": [label for label, _node in pair], "nodes": [node for _label, node in pair], "count": count}
        for pair, count in sorted(labeled_bond_counts.items(), key=lambda item: tuple(_bond_endpoint_sort_key(label) for label, _node in item[0]))
    ]


def _bond_records(bond_counts, origin_endpoint):
    """
    Undirected linkage records from growth-direction origin-index pairs.
    """
    return _bond_records_from_labeled(
        _labeled_bond_counts(bond_counts, origin_endpoint)
    )


def _merge_ensemble_data(target, batch):
    """Merge a batch into cumulative :class:`EnsembleData` in place."""
    target.chains.extend(batch.chains)
    target.sequences.extend(batch.sequences)
    target.molecular_weights.extend(batch.molecular_weights)

    for unit_id, unit_data in batch.units.items():
        if unit_id in target.units:
            target.units[unit_id]["count"] += unit_data["count"]
        else:
            target.units[unit_id] = unit_data

    bonds = {tuple(record["labels"]): record for record in target.bonds}
    for record in batch.bonds:
        key = tuple(record["labels"])
        if key in bonds:
            bonds[key]["count"] += record["count"]
        else:
            copied = dict(record)
            target.bonds.append(copied)
            bonds[key] = copied
    target.bonds.sort(key=lambda record: tuple(_bond_endpoint_sort_key(label) for label in record["labels"]))

    for stochastic_id, weights in batch.mol_weights.items():
        target.mol_weights.setdefault(stochastic_id, []).extend(weights)
    for stochastic_id, distribution in batch.distributions.items():
        target.distributions.setdefault(stochastic_id, distribution)


def _contact_frequencies(bonds):
    total = sum(record["count"] for record in bonds)
    if not total:
        return {}
    return {"|".join(record["labels"]): record["count"] / total for record in bonds}


def _unit_subgraphs(generative_graph, unit_id_by_node):
    """
    Detached static subgraph per unit: the unit's nodes with their static
    edges only (non-static edges are generative rules, not template
    structure -- an induced subgraph would drag intra-unit self-transitions
    along). Node ids are kept; ``unit_id`` is stamped on the copy's nodes,
    never on the generative graph itself.
    """
    unit_nodes = {}
    for node, unit_id in unit_id_by_node.items():
        unit_nodes.setdefault(unit_id, []).append(node)
    subgraphs = {}
    for unit_id, nodes in unit_nodes.items():
        subgraph = deepcopy(generative_graph.subgraph(nodes).copy())
        subgraph.graph.clear()
        subgraph.remove_edges_from([(u, v, key) for u, v, key, data in subgraph.edges(keys=True, data=True) if not data["static"]])
        for node in subgraph.nodes:
            subgraph.nodes[node]["unit_id"] = unit_id
        subgraphs[unit_id] = subgraph
    return subgraphs


@contextmanager
def _no_main_reimport():
    # Spawned workers only re-import the caller's script so they can unpickle
    # script-defined objects; this pool ships none, so skip the re-import
    # (it is what makes unguarded Windows scripts recursively re-spawn).
    # The patch is process-global for the pool's lifetime and not reentrant:
    # anything else spawning workers in that window also skips its re-import.
    with _SPAWN_CONFIGURATION_LOCK:
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


def _attempt_chain(
    atom_graph,
    collect_info,
    termination_flag,
    rng,
    use_repeat_units_as_source=False,
):
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
            metadata_mode = _metadata_level(collect_info)
            if metadata_mode is not _MetadataLevel.NONE:
                sample = atom_graph.sample_mol_graph(
                    termination_flag=termination_flag,
                    use_repeat_units_as_source=use_repeat_units_as_source,
                    rng=rng,
                    _metadata_mode=metadata_mode,
                )
            else:
                sample = atom_graph.sample_mol_graph(
                    termination_flag=termination_flag,
                    use_repeat_units_as_source=use_repeat_units_as_source,
                    rng=rng,
                )
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


def _convert_chain(
    sample,
    molecule_format,
    collect_info,
    include_sequences=True,
    chain_index=None,
    seed_sequence=None,
    native_diagnostics_path=None,
    molecular_weight=None,
):
    """Convert one accepted sample_mol_graph result into a chain record in the
    requested output format."""
    metadata_mode = _metadata_level(collect_info)
    if metadata_mode is not _MetadataLevel.NONE:
        mol_graph = sample.graph
        unit_counts = sample.metadata.unit_counts
        bonds = sample.metadata.bond_counts
        contact_counts = sample.metadata.labeled_bond_counts
        mol_weights = sample.mol_weights
        distributions = sample.distributions
        legacy_units = sample.legacy_units
    else:
        mol_graph = sample
        unit_counts = bonds = contact_counts = mol_weights = distributions = legacy_units = None

    rdkit_mol = None
    publish_native_stage = None
    needs_weight = (
        metadata_mode is not _MetadataLevel.NONE
        and molecular_weight is None
    )
    if needs_weight or molecule_format == "smiles":
        _enable_native_faulthandler()
        publish_native_stage = _native_state_publisher(
            chain_index,
            seed_sequence,
            mol_graph,
            native_diagnostics_path,
        )
        rdkit_mol = mol_graph_to_rdkit_mol(
            mol_graph,
            native_stage_callback=lambda stage: publish_native_stage(
                f"molecule-{stage}"
            ),
        )
    if needs_weight:
        molecular_weight = rdkit_mol_weight(
            rdkit_mol,
            native_stage_callback=lambda stage: publish_native_stage(
                f"molecule-{stage}"
            ),
        )

    if molecule_format == "smiles":
        molecule = rdkit_mol_to_smiles(
            rdkit_mol,
            native_stage_callback=lambda stage: publish_native_stage(
                f"molecule-{stage}"
            ),
        )
    else:
        molecule = mol_graph

    converted_sequences = None
    if metadata_mode >= _MetadataLevel.COMPACT_SEQUENCES and include_sequences:
        sequences = (
            sample.legacy_sequences
            if sample.legacy_sequences is not None
            else sample.metadata.materialize_sequences()
        )
        # Units are static-connected fragments with dangling inter-unit
        # valences, so convert them with kekulize=False (an aromatic ring
        # at a connection point can't be kekulized in isolation).
        if molecule_format == "smiles":
            converted_sequences = [
                [
                    mol_graph_to_smiles(
                        unit,
                        kekulize=False,
                        native_stage_callback=lambda stage: publish_native_stage(
                            f"sequence-{stage}"
                        ),
                    )
                    for unit in sequence
                ]
                for sequence in sequences
            ]
        else:
            converted_sequences = sequences

    return _ChainRecord(
        molecule=molecule,
        unit_counts=unit_counts,
        bonds=bonds,
        contact_counts=contact_counts,
        sequences=converted_sequences,
        mol_weights=mol_weights,
        distributions=distributions,
        molecular_weight=molecular_weight,
        legacy_units=legacy_units,
    )


def _defer_chain_conversion(
    sample,
    collect_info,
    chain_index=None,
    seed_sequence=None,
    native_diagnostics_path=None,
):
    """Compute mandatory convergence statistics but defer output conversion."""
    metadata_mode = _metadata_level(collect_info)
    if metadata_mode is _MetadataLevel.NONE:
        raise ValueError("deferred conversion requires chain metadata")
    publish_native_stage = _native_state_publisher(
        chain_index,
        seed_sequence,
        sample.graph,
        native_diagnostics_path,
    )
    _enable_native_faulthandler()
    rdkit_mol = mol_graph_to_rdkit_mol(
        sample.graph,
        native_stage_callback=lambda stage: publish_native_stage(
            f"molecule-{stage}"
        ),
    )
    molecular_weight = rdkit_mol_weight(
        rdkit_mol,
        native_stage_callback=lambda stage: publish_native_stage(
            f"molecule-{stage}"
        ),
    )
    return _DeferredChainRecord(
        sample,
        molecular_weight,
        chain_index,
        seed_sequence,
        native_diagnostics_path,
    )


def _materialize_deferred_chain(
    deferred,
    molecule_format,
    collect_info,
    include_sequences,
):
    return _convert_chain(
        deferred.sample,
        molecule_format,
        collect_info,
        include_sequences,
        deferred.chain_index,
        deferred.seed_sequence,
        deferred.native_diagnostics_path,
        molecular_weight=deferred.molecular_weight,
    )


def _portable_warning(caught):
    """Make a caught warning safe to ship across the process boundary."""
    message = caught.message
    try:
        pickle.dumps(message)
    except Exception:
        message = str(message)
    return (message, caught.category, caught.filename, caught.lineno)


def _sample_chain_batch(
    atom_graph,
    chain_jobs,
    molecule_format,
    collect_info,
    max_discards,
    termination_flag,
    include_sequences=True,
    native_diagnostics_path=None,
    defer_conversion=False,
    use_repeat_units_as_source=False,
):
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
            sample, attempt_reasons, cause, attempt_warnings = _attempt_chain(
                atom_graph,
                collect_info,
                termination_flag,
                rng,
                use_repeat_units_as_source,
            )
            deferred_warnings.extend(attempt_warnings)
            if sample is not None:
                if defer_conversion:
                    record = _defer_chain_conversion(
                        sample,
                        collect_info,
                        chain_index,
                        seed_sequence,
                        native_diagnostics_path,
                    )
                else:
                    record = _convert_chain(
                        sample,
                        molecule_format,
                        collect_info,
                        include_sequences,
                        chain_index,
                        seed_sequence,
                        native_diagnostics_path,
                    )
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


@dataclass
class _UnitOccurrence:
    """Compact metadata for one realized template unit in a sampled chain."""

    unit_id: str
    prototype_key: tuple
    nodes: tuple[int, ...]
    incoming_connection: tuple[int, int] | None
    connections: list[tuple[str, int, dict, dict]]


class _PartialAtomGraph:
    _ATOM_ATTRS = {"atomic_num", _AROMATIC_NAME, "charge", "num_explicit_h", "atom_chiral_token"}
    _BOND_ATTRS = {_BOND_TYPE_NAME, _AROMATIC_NAME, "bond_symbol_raw"}
    _MISSING_REQUIRED = object()
    _SKIP_OPTIONAL = object()
    # Defaults for optional node attributes so a generative_graph built before an attribute
    # existed still yields an EnsembleCreator (required attributes stay strict).
    _ATTR_DEFAULTS = {
        "num_explicit_h": -1,
        "atom_chiral_token": _SKIP_OPTIONAL,
        "bond_symbol_raw": _SKIP_OPTIONAL,
    }

    def __init__(
        self,
        generative_graph,
        static_graph,
        source_node,
        stochastic_tracker,
        sto_atom_id,
        rng,
        collect_info=True,
        unit_id_by_origin=None,
        termination_fragment_masses=None,
        static_source_templates=None,
    ):
        self._atom_id = 0
        self.generative_graph = generative_graph
        self.static_graph = static_graph
        self.stochastic_tracker = stochastic_tracker
        # Sampling metadata stays compact so exact-rounding checkpoints do not
        # recursively copy graph-valued unit keys and sequence fragments.
        self.collect_info = collect_info
        self._unit_id_by_origin = unit_id_by_origin or {}
        self._termination_fragment_masses = (
            termination_fragment_masses
            if termination_fragment_masses is not None
            else _prepare_termination_fragment_masses(generative_graph, static_graph)
        )
        self._static_source_templates = static_source_templates or {}

        self.atom_graph = nx.Graph()
        self._open_half_bond_map: dict[int, list[_HalfAtomBond]] = {}
        self._active_transactions = []
        self._last_merge_connection = None
        self.add_static_sub_graph(source_node, sto_atom_id, rng)

        self._bond_counts = Counter()
        self._unit_counts = Counter()
        self._unit_occurrences: list[_UnitOccurrence] = []
        self._unit_prototypes: dict[tuple, nx.Graph] = {}
        self._atom_to_unit_occurrence: dict[int, int] = {}
        self.sto_instance_molw_list = {}
        self._sequences: list[list[int]] = []
        self._terminal_unit_occurrences: list[int] = []
        self.current_connection = 0

    def __deepcopy__(self, memo):
        # Snapshots must copy the mutable molecule state, but the generating and
        # static template graphs are never mutated during sampling — share them
        # instead of deep-copying them on every snapshot (they dominated the
        # cost). The tracker (including its forked rng) is still deep-copied.
        memo[id(self.generative_graph)] = self.generative_graph
        memo[id(self.static_graph)] = self.static_graph
        memo[id(self._unit_id_by_origin)] = self._unit_id_by_origin
        memo[id(self._termination_fragment_masses)] = self._termination_fragment_masses
        memo[id(self._static_source_templates)] = self._static_source_templates
        # Prototypes are an append-only derived cache keyed entirely by unit
        # identity and remaining connector origins. Occurrences, counts, and
        # sequences are still copied per timeline; sharing this bounded cache
        # cannot make a restored timeline reference a discarded occurrence.
        memo[id(self._unit_prototypes)] = self._unit_prototypes
        new_graph = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_graph
        for key, value in self.__dict__.items():
            setattr(new_graph, key, copy.deepcopy(value, memo))
        return new_graph

    def merge(self, other, self_idx, other_idx, bond_attr):
        offset = self._atom_id
        other_open_half_bond_map = {}
        for stochastic_id in other._open_half_bond_map:
            for half_bond in other._open_half_bond_map[stochastic_id]:
                new_half_bond = copy.copy(half_bond)
                new_half_bond.atom_idx += offset
                try:
                    other_open_half_bond_map[stochastic_id] += [new_half_bond]
                except KeyError:
                    other_open_half_bond_map[stochastic_id] = [new_half_bond]

        other_idx += offset

        # Now we can do the actual merging
        self._atom_id += other._atom_id

        if _USE_DIRECT_GRAPH_MERGE:
            self.atom_graph.add_nodes_from(
                (node_idx + offset, data)
                for node_idx, data in other.atom_graph.nodes(data=True)
            )
            self.atom_graph.add_edges_from(
                (u_idx + offset, v_idx + offset, data)
                for u_idx, v_idx, data in other.atom_graph.edges(data=True)
            )
        else:
            remapping = {
                node_idx: node_idx + offset for node_idx in other.atom_graph.nodes
            }
            other_graph = nx.relabel_nodes(other.atom_graph, remapping, copy=True)
            self.atom_graph.add_nodes_from(other_graph.nodes(data=True))
            self.atom_graph.add_edges_from(other_graph.edges(data=True))
        self.atom_graph.add_edge(self_idx, other_idx, **bond_attr)
        self._last_merge_connection = (self_idx, other_idx)
        self._apply_realized_bond(self_idx, other_idx, bond_attr)
        for stochastic_id in other_open_half_bond_map:
            self._journal_frontier_bucket(stochastic_id)
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
                self._journal_runtime_node(real_idx)
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

    def _journal_runtime_node(self, node_idx):
        for transaction in self._active_transactions:
            transaction.record_runtime_node(node_idx)

    def _journal_frontier_bucket(self, bucket_id):
        for transaction in self._active_transactions:
            transaction.record_frontier_bucket(bucket_id)

    def _journal_metadata_counter(self, name, key):
        for transaction in self._active_transactions:
            transaction.record_metadata_counter(name, key)

    def _journal_occurrence_connections(self, occurrence_id):
        for transaction in self._active_transactions:
            transaction.record_occurrence_connections(occurrence_id)

    def _journal_sequence(self, sequence_index):
        for transaction in self._active_transactions:
            transaction.record_sequence(sequence_index)

    def _journal_terminal_occurrences(self):
        for transaction in self._active_transactions:
            transaction.record_terminal_occurrences()

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
        return _static_total_bond(self.generative_graph, node_idx)

    def add_static_sub_graph(self, source, sto_atom_id, rng):
        if _USE_STATIC_SOURCE_TEMPLATES and source in self._static_source_templates:
            self._add_static_source_template(
                self._static_source_templates[source],
                sto_atom_id,
                rng,
            )
            return

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

    def _add_static_source_template(self, template, sto_atom_id, rng):
        for node in template.nodes:
            atom_idx = self._atom_id
            data = dict(node.atom_attrs)
            self.atom_graph.add_node(
                atom_idx,
                **(data | {"origin_idx": str(node.origin_idx)}),
            )
            half_bond = _HalfAtomBond(
                atom_idx,
                node.origin_idx,
                self.generative_graph,
                self.stochastic_tracker,
                rng,
                template=node.half_bond,
            )
            credited_h = self.stochastic_tracker.add_molw(
                sto_atom_id,
                data["atomic_num"],
                node.static_total_bond,
                node.stochastic_id_tree,
                num_explicit_h=data.get("num_explicit_h", -1),
                charge=data["charge"],
                aromatic=data.get(_AROMATIC_NAME, False),
            )
            runtime_attrs = self.atom_graph.nodes[atom_idx]
            runtime_attrs["owner_sto_atom_id"] = sto_atom_id
            runtime_attrs["occupied_valence"] = node.static_total_bond
            runtime_attrs["credited_h"] = credited_h
            self._atom_id += 1

            if half_bond.weight > 0 and half_bond.has_any_bonds():
                self._open_half_bond_map.setdefault(sto_atom_id, []).append(half_bond)

        for u_idx, v_idx, edge_attrs in template.edges:
            self.atom_graph.add_edge(u_idx, v_idx, **dict(edge_attrs))

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
                default_value = _PartialAtomGraph._ATTR_DEFAULTS.get(k, _PartialAtomGraph._MISSING_REQUIRED)
                if default_value is _PartialAtomGraph._MISSING_REQUIRED:
                    raise KeyError(k)
                if default_value is _PartialAtomGraph._SKIP_OPTIONAL:
                    continue
                new_dict[k] = default_value
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

        self._journal_frontier_bucket(sto_atom_idx)
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
                attach_order = target_attributes[i].get(_BOND_TYPE_NAME, 1)
                # The estimate must be the NET tracker delta the attach would
                # realize, mirroring merge/_apply_realized_bond without mutating:
                # gross cap mass overstates the margin by the hydrogens the
                # existing endpoint sheds and, for split-dummy connectors, by
                # charging the attach order to the massless dummy instead of
                # its real anchors (so their hydrogens stayed uncounted).
                terminator_weight = self._termination_fragment_masses[(node_id, attach_order)]
                if _VERIFY_TERMINATION_MW_CACHE:
                    legacy_weight = self._legacy_termination_fragment_mass(
                        node_id,
                        attach_order,
                        static_graph,
                        rng,
                    )
                    if not np.isclose(terminator_weight, legacy_weight, rtol=0, atol=1e-12):
                        raise AssertionError(
                            f"termination fragment cache mismatch for {(node_id, attach_order)}: "
                            f"{terminator_weight} != {legacy_weight}"
                        )
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

    def _legacy_termination_fragment_mass(self, source, attach_order, static_graph, rng):
        """Slow construction oracle used only when cache parity is requested."""
        estimator_rng = copy.deepcopy(rng)
        tracker = _StochasticObjectTracker(
            self.generative_graph,
            estimator_rng,
            path_is_conditional=self.stochastic_tracker.path_is_conditional,
            zero_support_is_unavoidable=self.stochastic_tracker.zero_support_is_unavoidable,
            prepared_distributions=self.stochastic_tracker._sto_gen_id_distribution,
        )
        source_tree = self.generative_graph.nodes[source]["stochastic_id_tree"]
        sto_atom_id = tracker.register_new_atom_instance(
            source_tree[0], source_tree[1], None, False
        )
        fragment = _PartialAtomGraph(
            self.generative_graph,
            static_graph,
            source,
            tracker,
            sto_atom_id,
            estimator_rng,
            collect_info=False,
            termination_fragment_masses=self._termination_fragment_masses,
            static_source_templates=self._static_source_templates,
        )
        extra_occupied = {}
        for frag_node, data in fragment.atom_graph.nodes(data=True):
            if data["origin_idx"] == str(source):
                for anchor in fragment._real_anchors(frag_node, exclude=None):
                    extra_occupied[anchor] = extra_occupied.get(anchor, 0) + attach_order
                break
        weight = 0.0
        for frag_node, data in fragment.atom_graph.nodes(data=True):
            atomic_number = data["atomic_num"]
            if atomic_number <= 0:
                continue
            occupied = self._compute_total_bond(data["origin_idx"]) + extra_occupied.get(frag_node, 0)
            num_h = _infer_hydrogen_count(
                atomic_number,
                data.get("charge", 0),
                occupied,
                data.get("num_explicit_h", -1),
                data.get(_AROMATIC_NAME, False),
            )
            weight += atomic_masses[atomic_number] + num_h * atomic_masses[1]
        return weight

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
            terminated_graph._journal_frontier_bucket(selected_bucket_id)
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

            other_partial_graph = _PartialAtomGraph(
                terminated_graph.generative_graph,
                terminated_graph.static_graph,
                selected_target,
                self.stochastic_tracker,
                sto_atom_id,
                rng,
                collect_info=False,
                termination_fragment_masses=self._termination_fragment_masses,
                static_source_templates=self._static_source_templates,
            )
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

        terminated_graph._journal_frontier_bucket(sto_atom_id)
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

                other_partial_graph = _PartialAtomGraph(
                    self.generative_graph,
                    self.static_graph,
                    selected_target,
                    self.stochastic_tracker,
                    level_sto_atom_id,
                    rng,
                    collect_info=False,
                    termination_fragment_masses=self._termination_fragment_masses,
                    static_source_templates=self._static_source_templates,
                )
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
                self._journal_frontier_bucket(bucket_id)
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

        self._journal_frontier_bucket(bucket_id)
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
            collect_info=False,
            termination_fragment_masses=self._termination_fragment_masses,
            static_source_templates=self._static_source_templates,
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

    def promote_level_transitions(self, child_sto_atom_id, owner_sto_atom_id, sto_gen_id):
        """Hand a finished instance's level-``sto_gen_id`` transition sites to
        that level's live owner as ordinary propagation options, so the
        owner's next growth step is one weighted draw over ALL its options
        (chain continuation and unfired entry ports alike) instead of a
        deterministic continuation fire — the multifunctional initiation
        principle generalized to every level of the nested tree.

        Promotion is per-edge and lazy: only edges declared at the owner's
        level convert; transition edges of shallower levels ride along
        unchanged and are promoted, in turn, at their own level's hand-off.
        Termination modes ride along too, so a site the owner never draws is
        capped by the owner's own terminate pass (a parked owner included).
        The finished level's own propagation modes are dropped: that level is
        decided. Scans the child's bucket plus its terminated descendants'
        buckets (a frontier can end inside a deeper instance), the same
        custody set terminate_graph composes.
        """
        tracker = self.stochastic_tracker
        bucket_ids = [child_sto_atom_id] + [
            bucket_id
            for bucket_id in self._open_half_bond_map
            if bucket_id != child_sto_atom_id and tracker.is_terminated(bucket_id) and child_sto_atom_id in tracker.parent_map.get(bucket_id, [])
        ]
        owner_parents = tracker.parent_map.get(owner_sto_atom_id, [])
        # Pool-native parent tag: a promoted option must rank exactly like
        # the owner's own bonds under the prefer_parent filter.
        native_parent = tracker._stochastic_atom_id_to_gen_id[owner_parents[-1]] if owner_parents else -2
        promoted_bonds = []
        for bucket_id in bucket_ids:
            self._journal_frontier_bucket(bucket_id)
            kept_bonds = []
            for half_bond in self._open_half_bond_map.get(bucket_id, []):
                attr_list = half_bond._mode_attr_map.get(_TRANSITION_NAME, [])
                level_indices = [i for i, attr in enumerate(attr_list) if attr.get(_EDGE_STOCHASTIC_ID_NAME) == sto_gen_id]
                if not level_indices:
                    kept_bonds.append(half_bond)
                    continue
                promoted_bond = copy.copy(half_bond)
                promoted_bond._mode_attr_map = {}
                promoted_bond._mode_target_map = {}
                promoted_bond._mode_target_molar_amounts_map = {}
                bond_attr_list = copy.deepcopy([attr_list[i] for i in level_indices])
                for bond_attr in bond_attr_list:
                    bond_attr[_PROPAGATION_NAME] = bond_attr[_TRANSITION_NAME]
                    bond_attr[_TRANSITION_NAME] = 0
                promoted_bond._mode_attr_map[_PROPAGATION_NAME] = bond_attr_list
                promoted_bond._mode_target_map[_PROPAGATION_NAME] = [half_bond._mode_target_map[_TRANSITION_NAME][i] for i in level_indices]
                promoted_bond._mode_target_molar_amounts_map[_PROPAGATION_NAME] = [half_bond._mode_target_molar_amounts_map[_TRANSITION_NAME][i] for i in level_indices]
                retained_indices = [i for i in range(len(attr_list)) if i not in level_indices]
                if retained_indices:
                    promoted_bond._mode_attr_map[_TRANSITION_NAME] = [attr_list[i] for i in retained_indices]
                    promoted_bond._mode_target_map[_TRANSITION_NAME] = [half_bond._mode_target_map[_TRANSITION_NAME][i] for i in retained_indices]
                    promoted_bond._mode_target_molar_amounts_map[_TRANSITION_NAME] = [half_bond._mode_target_molar_amounts_map[_TRANSITION_NAME][i] for i in retained_indices]
                if half_bond.has_mode_bonds(_TERMINATION_NAME):
                    promoted_bond._mode_attr_map[_TERMINATION_NAME] = list(half_bond._mode_attr_map[_TERMINATION_NAME])
                    promoted_bond._mode_target_map[_TERMINATION_NAME] = list(half_bond._mode_target_map[_TERMINATION_NAME])
                    promoted_bond._mode_target_molar_amounts_map[_TERMINATION_NAME] = list(half_bond._mode_target_molar_amounts_map[_TERMINATION_NAME])
                promoted_bond.parent = native_parent
                promoted_bonds.append(promoted_bond)
            self._open_half_bond_map[bucket_id] = kept_bonds
        for promoted_bond in promoted_bonds:
            self._journal_frontier_bucket(owner_sto_atom_id)
            try:
                self._open_half_bond_map[owner_sto_atom_id] += [promoted_bond]
            except KeyError:
                self._open_half_bond_map[owner_sto_atom_id] = [promoted_bond]
        return len(promoted_bonds)

    def _transition_graph_single_bucket(self, sto_atom_id, sto_gen_id, rng):
        # Early exit if no transition necessary
        if len(self.get_open_half_bonds(sto_atom_id)[0]) < 1:
            return sto_atom_id, False

        self._journal_frontier_bucket(sto_atom_id)
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

        new_sto_atom_id, _parent_list = self.stochastic_tracker.register_parent_atom_instances(selected_target_sto_gen_id, sto_atom_id, selected_target_sto_parent_id)

        # Transfer the source bucket's remaining same-level transition bonds,
        # converted to propagation, to the instance OF THE FIRED LEVEL. When
        # the target lies inside a nested stochastic object (own instance, own
        # MW draw) that owner differs from new_sto_atom_id: filing the bonds
        # under the landing instance pooled every sibling junction into its
        # one draw, and the terminate-time wipe then destroyed the unfired
        # ones (graft architectures could never reach the outer target). For
        # targets at the fired level itself the two owners coincide (star
        # initiators, outer-level graft-from), keeping that behavior intact.
        if sto_gen_id < 0 or selected_target_sto_gen_id == sto_gen_id:
            owner_sto_atom_id = new_sto_atom_id
        elif self.stochastic_tracker._stochastic_atom_id_to_gen_id[sto_atom_id] == sto_gen_id:
            owner_sto_atom_id = sto_atom_id
        else:
            # Nearest ancestor instance at the fired level; fall back to the
            # landing instance (pre-fix behavior) if none exists.
            owner_sto_atom_id = new_sto_atom_id
            for ancestor in reversed(self.stochastic_tracker.parent_map.get(sto_atom_id, [])):
                if self.stochastic_tracker._stochastic_atom_id_to_gen_id[ancestor] == sto_gen_id:
                    owner_sto_atom_id = ancestor
                    break

        # Converted bonds become ordinary members of the owner's pool: tag
        # them with the owner level's native parent value so prefer_parent
        # ranks them exactly like the owner's own bonds.
        owner_parents = self.stochastic_tracker.parent_map.get(owner_sto_atom_id, [])
        owner_native_parent = self.stochastic_tracker._stochastic_atom_id_to_gen_id[owner_parents[-1]] if owner_parents else -2

        list_of_new_bonds = []
        list_of_bond_idx_to_delete = []

        for j, half_bond in enumerate(half_bonds):
            prop_attr_list = half_bond._mode_attr_map.get(_TRANSITION_NAME, [])
            fired_level_indices = [i for i, attr in enumerate(prop_attr_list) if attr.get(_EDGE_STOCHASTIC_ID_NAME) == sto_gen_id]
            if not fired_level_indices:
                # No transition edge at the fired level: leave the bond
                # untouched in its bucket. Other levels' continuations (-1
                # arms for trigger_global_transitions, levels a later call
                # must fire) would be silently truncated by a mode-less copy,
                # and termination-only bonds (e.g. outer-declared side-port
                # caps) must keep their custody for the terminated-descendant
                # scan — a stripped duplicate of them in an ancestor bucket
                # masked that bucket's real growth bonds via prefer_parent
                # and stalled the molecule before its terminators fired.
                continue

            # Shallow copy: all three mode maps are rebound below, the other
            # attributes are immutable, and deepcopy dragged a full copy of the
            # generative graph along via half_bond._graph.
            new_stochastic_bond = copy.copy(half_bond)
            new_stochastic_bond._mode_attr_map = {}
            new_stochastic_bond._mode_target_map = {}
            new_stochastic_bond._mode_target_molar_amounts_map = {}

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
                if half_bond.has_mode_bonds(_TERMINATION_NAME):
                    # Termination modes ride along with the converted copy, as
                    # in promote_level_transitions: the copy holds the same
                    # valence slot under the owner's custody, so a site the
                    # owner never grows is capped by its declared terminators
                    # instead of finalizing bare. The owner's own average-cap
                    # estimate reads the same bucket and level stamp that its
                    # terminate pass fires, so the moved cap stays priced.
                    new_stochastic_bond._mode_attr_map[_TERMINATION_NAME] = list(half_bond._mode_attr_map[_TERMINATION_NAME])
                    new_stochastic_bond._mode_target_map[_TERMINATION_NAME] = list(half_bond._mode_target_map[_TERMINATION_NAME])
                    new_stochastic_bond._mode_target_molar_amounts_map[_TERMINATION_NAME] = list(half_bond._mode_target_molar_amounts_map[_TERMINATION_NAME])

            if not new_stochastic_bond.has_any_bonds():
                # Hierarchy-filtered or non-used old-SO bonds: their
                # fired-level edges are deliberately dropped with the source
                # entry; a mode-less duplicate serves no consumer and pollutes
                # the owner bucket.
                continue
            new_stochastic_bond.parent = owner_native_parent
            list_of_new_bonds.append(new_stochastic_bond)

        self._open_half_bond_map[sto_atom_id] = [bond for j, bond in enumerate(half_bonds) if j not in list_of_bond_idx_to_delete]

        for bond in list_of_new_bonds:
            self._journal_frontier_bucket(owner_sto_atom_id)
            try:
                self._open_half_bond_map[owner_sto_atom_id] += [bond]
            except KeyError:
                self._open_half_bond_map[owner_sto_atom_id] = [bond]

        other_graph = _PartialAtomGraph(
            self.generative_graph,
            self.static_graph,
            selected_target_idx,
            self.stochastic_tracker,
            new_sto_atom_id,
            rng,
            collect_info=False,
            termination_fragment_masses=self._termination_fragment_masses,
            static_source_templates=self._static_source_templates,
        )

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
                self._journal_frontier_bucket(sto_atom_id)
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

        other_graph = _PartialAtomGraph(
            self.generative_graph,
            self.static_graph,
            selected_target_idx,
            self.stochastic_tracker,
            new_sto_atom_id,
            rng,
            collect_info=False,
            termination_fragment_masses=self._termination_fragment_masses,
            static_source_templates=self._static_source_templates,
        )

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
            self._journal_frontier_bucket(sto_atom_id)
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

            other_graph = _PartialAtomGraph(
                self.generative_graph,
                self.static_graph,
                selected_target_idx,
                self.stochastic_tracker,
                new_sto_atom_id,
                rng,
                collect_info=False,
                termination_fragment_masses=self._termination_fragment_masses,
                static_source_templates=self._static_source_templates,
            )

            other_half_bond_atom_idx = other_graph.pop_target_open_half_bond(new_sto_atom_id, selected_target_idx)

            pre_merge_watermark = self._atom_id
            self.merge(other_graph, nested_transition_bond.atom_idx, other_half_bond_atom_idx, selected_attr)
            last_unit = self.add_new_unit_and_bond(pre_merge_watermark)
            self.add_unit_to_sequence(last_unit)
            self.nested_transition(new_sto_atom_id, rng)

    def add_new_unit_and_bond(self, pre_merge_watermark):
        # `pre_merge_watermark` is self._atom_id captured BEFORE the most recent
        # merge: merge() relabels incoming nodes to ids >= that watermark, so the
        # newly added unit is exactly the contiguous suffix ending at _atom_id.
        connection = self._last_merge_connection
        self._last_merge_connection = None
        if not self.collect_info:
            return None
        current_atom_graph = self.atom_graph
        new_nodes = tuple(range(pre_merge_watermark, self._atom_id))
        if connection is not None:
            u, v = connection
            bond_key = (
                current_atom_graph.nodes[u]["origin_idx"],
                current_atom_graph.nodes[v]["origin_idx"],
            )
            self._journal_metadata_counter("_bond_counts", bond_key)
            self._bond_counts[bond_key] += 1

        return self._record_unit_occurrence(new_nodes, connection)

    def _record_unit_occurrence(self, nodes, incoming_connection=None):
        if not self.collect_info or not nodes:
            return None
        origin = self.atom_graph.nodes[nodes[0]]["origin_idx"]
        unit_id = self._unit_id_by_origin.get(origin, self._unit_id_by_origin.get(str(origin), str(origin)))
        prototype_key = (
            unit_id,
            tuple(sorted(str(self.atom_graph.nodes[node]["origin_idx"]) for node in nodes)),
        )
        if prototype_key not in self._unit_prototypes:
            self._unit_prototypes[prototype_key] = deepcopy(
                self.atom_graph.subgraph(nodes).copy()
            )
        occurrence_id = len(self._unit_occurrences)
        self._unit_occurrences.append(
            _UnitOccurrence(
                unit_id,
                prototype_key,
                tuple(nodes),
                incoming_connection,
                [],
            )
        )
        self._journal_metadata_counter("_unit_counts", unit_id)
        self._unit_counts[unit_id] += 1
        for node in nodes:
            self._atom_to_unit_occurrence[node] = occurrence_id
        return occurrence_id

    def _connection_to_occurrence(self, occurrence_id):
        return self._unit_occurrences[occurrence_id].incoming_connection

    def _add_sequence_connection(self, occurrence_id, parent_node, child_node):
        parent_origin = str(self.atom_graph.nodes[parent_node]["origin_idx"])
        self._journal_occurrence_connections(occurrence_id)
        self._unit_occurrences[occurrence_id].connections.append(
            (
                parent_origin,
                self.current_connection,
                dict(self.atom_graph.nodes[child_node]),
                dict(self.atom_graph.edges[(parent_node, child_node)]),
            )
        )
        self.current_connection += 1

    def add_unit_to_sequence(self, last_unit):
        if last_unit is None:
            return
        if len(self._sequences) == 0:
            self._sequences.append([last_unit])
            return
        connection = self._connection_to_occurrence(last_unit)
        if len(self._sequences) == 1:
            self._sequences.append([last_unit])
            self._journal_terminal_occurrences()
            self._terminal_unit_occurrences.append(last_unit)
            if connection is not None:
                u, v = connection
                self._add_sequence_connection(self._sequences[0][0], u, v)
            return

        if connection is None:
            return
        u, v = connection
        parent_occurrence = self._atom_to_unit_occurrence.get(u)
        if parent_occurrence is None:
            return
        if parent_occurrence in self._terminal_unit_occurrences:
            self._journal_terminal_occurrences()
            self._terminal_unit_occurrences.remove(parent_occurrence)
            self._terminal_unit_occurrences.append(last_unit)
            for sequence_index, sequence in enumerate(self._sequences):
                if parent_occurrence in sequence:
                    self._journal_sequence(sequence_index)
                    sequence.append(last_unit)
                    break
        else:
            self._add_sequence_connection(parent_occurrence, u, v)
            self._journal_terminal_occurrences()
            self._terminal_unit_occurrences.append(last_unit)
            self._sequences.append([last_unit])

    def _materialize_sequence_unit(self, occurrence_id):
        occurrence = self._unit_occurrences[occurrence_id]
        unit = deepcopy(self._unit_prototypes[occurrence.prototype_key])
        for parent_origin, connection_id, node_attributes, edge_attributes in occurrence.connections:
            parent_node = next(
                node
                for node, data in unit.nodes(data=True)
                if str(data["origin_idx"]) == parent_origin
            )
            placeholder = "C" + str(connection_id)
            unit.add_node(placeholder, **node_attributes)
            unit.nodes[placeholder]["atomic_num"] = 0
            unit.nodes[placeholder]["connection"] = connection_id
            unit.add_edge(parent_node, placeholder, **edge_attributes)
        return unit

    def materialize_legacy_metadata(self):
        """Build the historical graph-valued tuple only at the public boundary."""
        representative_units = {}
        units = {}
        for occurrence in self._unit_occurrences:
            if occurrence.unit_id in representative_units:
                continue
            unit = deepcopy(self._unit_prototypes[occurrence.prototype_key])
            representative_units[occurrence.unit_id] = unit
            units[unit] = self._unit_counts[occurrence.unit_id]
        sequences = [
            [self._materialize_sequence_unit(occurrence_id) for occurrence_id in sequence]
            for sequence in self._sequences
        ]
        return units, dict(self._bond_counts), sequences

    def compact_metadata(self, include_sequences=False):
        """Return stable IDs and optionally the compact sequence recipe."""
        return _CompactMetadata(
            unit_counts=dict(self._unit_counts),
            bond_counts=dict(self._bond_counts),
                labeled_bond_counts={},
            occurrences=(tuple(self._unit_occurrences) if include_sequences else ()),
            sequences=(
                tuple(tuple(sequence) for sequence in self._sequences)
                if include_sequences
                else ()
            ),
            unit_prototypes=(dict(self._unit_prototypes) if include_sequences else {}),
        )


class _GrowthTransaction:
    """Rollback state for one owner-level stochastic growth step.

    Capture itself is constant-size. Mutators record the first old value for
    every touched tracker key, frontier bucket, runtime atom, and compact
    metadata entry in every live compatible transaction. This is necessary
    because descendant work can continue while an ancestor boundary remains
    eligible for rollback.
    """

    _MISSING = object()

    def __init__(self, graph, owner, epoch):
        self.graph = graph
        self.owner = owner
        self.epoch = epoch
        self.atom_watermark = graph._atom_id
        self.atom_id = graph._atom_id
        self.node_runtime = {}
        self.frontier_buckets = {}
        self.tracker_scalars = {}
        self.tracker_mapping_keys = {}
        self.tracker_set_members = {}
        self.metadata_counters = {}
        self.occurrence_connections = {}
        self.sequence_lengths = {}
        self.terminal_unit_occurrences = None
        self.unit_occurrence_watermark = len(graph._unit_occurrences)
        self.sequence_watermark = len(graph._sequences)
        self.last_merge_connection = graph._last_merge_connection
        self.current_connection = graph.current_connection
        graph._active_transactions.append(self)
        graph.stochastic_tracker._active_transactions.append(self)

    def record_runtime_node(self, node):
        if node not in self.node_runtime:
            data = self.graph.atom_graph.nodes[node]
            self.node_runtime[node] = (
                data.get("occupied_valence"),
                data.get("credited_h"),
            )

    def record_frontier_bucket(self, bucket_id):
        if bucket_id in self.frontier_buckets:
            return
        if bucket_id not in self.graph._open_half_bond_map:
            self.frontier_buckets[bucket_id] = self._MISSING
            return
        memo = {id(self.graph.generative_graph): self.graph.generative_graph}
        self.frontier_buckets[bucket_id] = copy.deepcopy(
            self.graph._open_half_bond_map[bucket_id], memo
        )

    def record_tracker_scalar(self, name, value):
        self.tracker_scalars.setdefault(name, value)

    def record_tracker_mapping_key(self, name, key, mapping):
        journal_key = (name, key)
        if journal_key in self.tracker_mapping_keys:
            return
        self.tracker_mapping_keys[journal_key] = (
            copy.deepcopy(mapping[key]) if key in mapping else self._MISSING
        )

    def record_tracker_set_member(self, name, value, was_present):
        self.tracker_set_members.setdefault((name, value), was_present)

    def record_metadata_counter(self, name, key):
        journal_key = (name, key)
        if journal_key not in self.metadata_counters:
            self.metadata_counters[journal_key] = getattr(self.graph, name)[key]

    def record_occurrence_connections(self, occurrence_id):
        if (
            occurrence_id < self.unit_occurrence_watermark
            and occurrence_id not in self.occurrence_connections
        ):
            self.occurrence_connections[occurrence_id] = list(
                self.graph._unit_occurrences[occurrence_id].connections
            )

    def record_sequence(self, sequence_index):
        if sequence_index < self.sequence_watermark:
            self.sequence_lengths.setdefault(
                sequence_index, len(self.graph._sequences[sequence_index])
            )

    def record_terminal_occurrences(self):
        if self.terminal_unit_occurrences is None:
            self.terminal_unit_occurrences = list(
                self.graph._terminal_unit_occurrences
            )

    def _restore_tracker(self, tracker):
        for name, value in self.tracker_scalars.items():
            setattr(tracker, name, value)
        for (name, key), value in self.tracker_mapping_keys.items():
            mapping = getattr(tracker, name)
            if value is self._MISSING:
                mapping.pop(key, None)
            else:
                mapping[key] = copy.deepcopy(value)
        for (name, value), was_present in self.tracker_set_members.items():
            values = getattr(tracker, name)
            if was_present:
                values.add(value)
            else:
                values.discard(value)

    def _snapshot_tracker(self):
        tracker = self.graph.stochastic_tracker
        snapshot = copy.copy(tracker)
        snapshot._active_transactions = []
        for name in (
            "_stochastic_gen_id_to_atom_id",
            "_stochastic_atom_id_to_gen_id",
            "_sto_atom_id_actual_molw",
            "_sto_atom_id_expected_molw",
            "parent_map",
            "_parent_molw",
        ):
            setattr(snapshot, name, copy.deepcopy(getattr(tracker, name)))
        snapshot._terminated_sto_atom_ids = set(tracker._terminated_sto_atom_ids)
        self._restore_tracker(snapshot)
        return snapshot

    def _snapshot_frontier(self):
        frontier = dict(self.graph._open_half_bond_map)
        for bucket_id, value in self.frontier_buckets.items():
            if value is self._MISSING:
                frontier.pop(bucket_id, None)
            else:
                frontier[bucket_id] = value
        return frontier

    def _restore_runtime_attributes(self):
        for node, (occupied_valence, credited_h) in self.node_runtime.items():
            if node not in self.graph.atom_graph:
                continue
            data = self.graph.atom_graph.nodes[node]
            data["occupied_valence"] = occupied_valence
            data["credited_h"] = credited_h

    @contextmanager
    def snapshot_view(self):
        """Expose the under-boundary topology for observational calculations."""
        current_runtime = {
            node: (
                self.graph.atom_graph.nodes[node].get("occupied_valence"),
                self.graph.atom_graph.nodes[node].get("credited_h"),
            )
            for node in self.node_runtime
            if node in self.graph.atom_graph
        }
        self._restore_runtime_attributes()
        snapshot = copy.copy(self.graph)
        snapshot._atom_id = self.atom_id
        snapshot.atom_graph = nx.subgraph_view(
            self.graph.atom_graph,
            filter_node=lambda node: node < self.atom_watermark,
        )
        snapshot._open_half_bond_map = self._snapshot_frontier()
        snapshot.stochastic_tracker = self._snapshot_tracker()
        try:
            yield snapshot
        finally:
            for node, (occupied_valence, credited_h) in current_runtime.items():
                data = self.graph.atom_graph.nodes[node]
                data["occupied_valence"] = occupied_valence
                data["credited_h"] = credited_h

    def rollback(self, rng):
        """Adopt the undershoot timeline without rewinding the caller RNG."""
        self.graph.atom_graph.remove_nodes_from(
            node for node in tuple(self.graph.atom_graph) if node >= self.atom_watermark
        )
        self._restore_runtime_attributes()
        self.graph._atom_id = self.atom_id
        self.graph._last_merge_connection = self.last_merge_connection
        for bucket_id, value in self.frontier_buckets.items():
            if value is self._MISSING:
                self.graph._open_half_bond_map.pop(bucket_id, None)
            else:
                self.graph._open_half_bond_map[bucket_id] = value
        tracker = self.graph.stochastic_tracker
        self._restore_tracker(tracker)
        tracker._rng = rng
        for (name, key), value in self.metadata_counters.items():
            counter = getattr(self.graph, name)
            if value:
                counter[key] = value
            else:
                counter.pop(key, None)
        for occurrence_id, connections in self.occurrence_connections.items():
            self.graph._unit_occurrences[occurrence_id].connections = connections
        del self.graph._unit_occurrences[self.unit_occurrence_watermark:]
        for node in tuple(self.graph._atom_to_unit_occurrence):
            if node >= self.atom_watermark:
                self.graph._atom_to_unit_occurrence.pop(node, None)
        for sequence_index, length in self.sequence_lengths.items():
            del self.graph._sequences[sequence_index][length:]
        del self.graph._sequences[self.sequence_watermark:]
        if self.terminal_unit_occurrences is not None:
            self.graph._terminal_unit_occurrences = self.terminal_unit_occurrences
        self.graph.current_connection = self.current_connection
        return self.graph


class EnsembleCreator:

    def __init__(self, generative_graph):

        self._generative_graph = generative_graph.copy()
        self._prepared_distributions = self._prepare_distributions()
        labels = derive_unit_labels(self._generative_graph)
        self._unit_id_by_origin = {
            node: unit_id for node, unit_id in labels.unit_id.items()
        }
        self._unit_id_by_origin.update(
            {str(node): unit_id for node, unit_id in labels.unit_id.items()}
        )
        self._origin_unit_id = {
            str(node): unit_id for node, unit_id in labels.unit_id.items()
        }
        self._origin_bond_id = {
            str(node): bond_id for node, bond_id in labels.bond_id.items()
        }
        self._origin_endpoint = {
            origin: f"{self._origin_unit_id[origin]}.{bond_id}"
            for origin, bond_id in self._origin_bond_id.items()
        }

        # Sampling filters every non-static decision by the per-edge stochastic id;
        # a graph built against the older schema (per-edge 'hierarchy') would not
        # error but silently generate truncated, end-group-less molecules.
        for _u, _v, edge_data in self._generative_graph.edges(data=True):
            if _EDGE_STOCHASTIC_ID_NAME not in edge_data:
                raise IncompatibleGenerativeGraphSchema(_EDGE_STOCHASTIC_ID_NAME)

        self._static_graph = self._create_static_graph(self.generative_graph)
        self._static_source_templates = self._prepare_static_source_templates()
        self._canonical_unit_info = self._prepare_canonical_unit_info(labels)
        self._termination_fragment_masses = _prepare_termination_fragment_masses(
            self._generative_graph,
            self._static_graph,
        )
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

    def _prepare_distributions(self):
        """Deserialize immutable distribution templates once per creator."""
        try:
            first_node = next(iter(self._generative_graph.nodes))
            serial_vectors = self._generative_graph.nodes[first_node][
                "molecular_weight_distribution"
            ]
        except (StopIteration, KeyError, TypeError):
            return {}

        prepared = {}
        for sto_gen_id, serial_vector in enumerate(serial_vectors):
            distribution = StochasticDistribution.from_serial_vector(
                list(tuple(serial_vector))
            )
            if distribution is not None:
                prepared[sto_gen_id] = distribution
        return prepared

    def _prepare_static_source_templates(self):
        """Capture exact per-source static traversal and half-bond descriptors."""
        templates = {}
        for source in self._generative_graph.nodes:
            ordered_origins = [source]
            origin_to_local = {source: 0}
            edges = {}

            for u_idx, v_idx, key in nx.edge_dfs(
                self._static_graph,
                source=source,
            ):
                for origin_idx in (u_idx, v_idx):
                    if origin_idx not in origin_to_local:
                        origin_to_local[origin_idx] = len(ordered_origins)
                        ordered_origins.append(origin_idx)

                u_local = origin_to_local[u_idx]
                v_local = origin_to_local[v_idx]
                if (u_local, v_local) not in edges and (v_local, u_local) not in edges:
                    edge_data = self._static_graph.get_edge_data(u_idx, v_idx, key)
                    edge_attrs = _PartialAtomGraph._copy_some_dict_attr(
                        edge_data,
                        _PartialAtomGraph._BOND_ATTRS,
                    )
                    edges[(u_local, v_local)] = tuple(edge_attrs.items())

            nodes = []
            for origin_idx in ordered_origins:
                node_data = self._generative_graph.nodes[origin_idx]
                atom_attrs = _PartialAtomGraph._copy_some_dict_attr(
                    node_data,
                    _PartialAtomGraph._ATOM_ATTRS,
                )
                nodes.append(
                    _StaticNodeTemplate(
                        origin_idx=origin_idx,
                        atom_attrs=tuple(atom_attrs.items()),
                        static_total_bond=_static_total_bond(
                            self._generative_graph,
                            origin_idx,
                        ),
                        stochastic_id_tree=tuple(node_data["stochastic_id_tree"]),
                        half_bond=_HalfAtomBond.prepare_template(
                            origin_idx,
                            self._generative_graph,
                        ),
                    )
                )

            templates[source] = _StaticSourceTemplate(
                nodes=tuple(nodes),
                edges=tuple(
                    (u_idx, v_idx, attrs)
                    for (u_idx, v_idx), attrs in edges.items()
                ),
            )
        return templates

    def _prepare_canonical_unit_info(self, labels):
        """Prepare format-independent public unit metadata once per creator."""
        unit_subgraphs = _unit_subgraphs(self._generative_graph, labels.unit_id)
        unit_g2rins = self._generative_graph.graph.get("unit_g2rins", {})
        if not set(unit_g2rins).issubset(unit_subgraphs):
            unit_g2rins = {}

        representative_sources = {}
        for source in self._generative_graph.nodes:
            unit_id = self._unit_id_by_origin[source]
            representative_sources.setdefault(unit_id, source)

        canonical = {}
        self._canonical_unit_prototypes = {}
        for unit_id, source in representative_sources.items():
            template = self._static_source_templates[source]
            unit_graph = nx.Graph()
            for local_id, node in enumerate(template.nodes):
                unit_graph.add_node(
                    local_id,
                    **dict(node.atom_attrs),
                    origin_idx=node.origin_idx,
                )
            for left, right, attributes in template.edges:
                unit_graph.add_edge(left, right, **dict(attributes))
            self._canonical_unit_prototypes[unit_id] = unit_graph
            try:
                with rdBase.BlockLogs():
                    star_mol = mol_graph_to_rdkit_mol(
                        self._unit_graph_with_stars(
                            unit_graph,
                            self._origin_bond_id,
                        ),
                        kekulize=False,
                    )
                    psmiles = rdkit_mol_to_smiles(star_mol)
            except Exception:
                # Historically unit conversion happened only after a chain was
                # accepted. Do not reject a model at creator construction for
                # an unreachable/dead template; retry if it reaches output.
                psmiles = None
            canonical[unit_id] = {
                "psmiles": psmiles,
                "g2rins": unit_g2rins.get(unit_id, ""),
                "subgraph": unit_subgraphs[unit_id],
            }
        return canonical

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
        bounds = {}
        for sto_gen_id, distribution in self._prepared_distributions.items():
            try:
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
        _metadata_mode=None,
    ):
        # TODO: consider using repeat units as source not an option.
        internal_request = _metadata_mode is not None
        metadata_mode = _metadata_level(
            _metadata_mode if internal_request else molecule_info
        )
        collect_info = metadata_mode is not _MetadataLevel.NONE
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
            prepared_distributions=self._prepared_distributions,
        )

        source_stochastic_id_tree = generative_graph.nodes[source]["stochastic_id_tree"]
        source_sto_gen_id = source_stochastic_id_tree[0]
        source_parents_sto_gen_id = source_stochastic_id_tree[1:]
        sto_atom_id, _parent_list = stochastic_object_tracker.register_parent_atom_instances(source_sto_gen_id, source_stochastic_id_tree[1], source_parents_sto_gen_id)

        partial_atom_graph = _PartialAtomGraph(
            generative_graph,
            self._static_graph,
            source,
            stochastic_object_tracker,
            sto_atom_id,
            rng,
            collect_info=collect_info,
            unit_id_by_origin=self._unit_id_by_origin,
            termination_fragment_masses=self._termination_fragment_masses,
            static_source_templates=self._static_source_templates,
        )
        del stochastic_object_tracker

        if collect_info:
            source_occurrence = partial_atom_graph._record_unit_occurrence(
                tuple(partial_atom_graph.atom_graph.nodes)
            )
            partial_atom_graph.add_unit_to_sequence(source_occurrence)

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
        ):
            """Net caps that would replace live descendant continuations if
            ``sto_atom_id`` parked in this exact topology. A finished child's
            own promoted continuation sites are not conditional junctions:
            they sit in the owner's pool with their termination modes intact,
            so the owner's own average-cap estimate already prices them."""
            owners = _live_forest_children(tracker, live_ids, live_set, sto_atom_id)
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
            )
            return own_mw, own_mw + conditional_mw

        def _capture_checkpoint(checkpoint_owner):
            """Capture molecular and loop-control state for one owner step.

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
                "graph": (
                    copy.deepcopy(partial_atom_graph)
                    if _USE_LEGACY_CHECKPOINTS
                    else None
                ),
                "transaction": (
                    None
                    if _USE_LEGACY_CHECKPOINTS
                    else _GrowthTransaction(
                        partial_atom_graph,
                        checkpoint_owner,
                        owner_epochs.get(checkpoint_owner, 0),
                    )
                ),
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
            }

        @contextmanager
        def _checkpoint_snapshot(checkpoint):
            if checkpoint["graph"] is not None:
                yield checkpoint["graph"]
            else:
                with checkpoint["transaction"].snapshot_view() as snapshot:
                    yield snapshot

        def _sync_transactions():
            """Retain journals reachable from the current checkpoint forest."""
            live_transactions = set()
            visited = set()

            def visit(checkpoint):
                checkpoint_id = id(checkpoint)
                if checkpoint_id in visited:
                    return
                visited.add(checkpoint_id)
                transaction = checkpoint.get("transaction")
                if transaction is not None:
                    live_transactions.add(transaction)
                for ancestor in checkpoint.get("checkpoints", {}).values():
                    visit(ancestor)

            for checkpoint in checkpoints.values():
                visit(checkpoint)
            partial_atom_graph._active_transactions[:] = [
                transaction
                for transaction in partial_atom_graph._active_transactions
                if transaction in live_transactions
            ]
            partial_atom_graph.stochastic_tracker._active_transactions[:] = [
                transaction
                for transaction in partial_atom_graph.stochastic_tracker._active_transactions
                if transaction in live_transactions
            ]

        def _advance_owner_epoch(sto_atom_id):
            """Record one successful composition step at exactly this level.

            A checkpoint survives arbitrary descendant mutations, but never a
            second owner-level step. Dropping it here is the stale-snapshot
            guard that prevents multi-unit rollback.
            """
            owner_epochs[sto_atom_id] = owner_epochs.get(sto_atom_id, 0) + 1
            checkpoint = checkpoints.get(sto_atom_id)
            if checkpoint is not None and checkpoint["epoch"] != owner_epochs[sto_atom_id] - 1:
                checkpoints.pop(sto_atom_id, None)
                _sync_transactions()

        def _finalize_pending(sto_atom_id):
            """Fire the parked instance's declared end groups, then hand its
            remaining continuation sites to the nearest live ancestor as
            ordinary growth options (the multifunctional initiation principle
            generalized to every level of the nested tree): the ancestor's
            next step is one weighted draw over chain continuation and
            unfired entry ports alike, instead of a deterministic
            continuation fire. If that ancestor is itself parked, the
            junction must NOT continue: cap it with the ancestor's end groups
            instead (graft-through chains otherwise grow a decided level
            forever, one nested instance per continued unit). With no live
            ancestor at all the continuation is inter-object/root and the
            caller fires it directly."""
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
                return sto_atom_id, None, tracker._stochastic_atom_id_to_gen_id[sto_atom_id]
            partial_atom_graph.promote_level_transitions(
                sto_atom_id,
                nearest_live_ancestor,
                tracker._stochastic_atom_id_to_gen_id[nearest_live_ancestor],
            )
            return None

        while True:
            tracker = partial_atom_graph.stochastic_tracker
            unterminated_sto_atom_ids = tracker.get_unterminated_sto_atom_ids()
            if not unterminated_sto_atom_ids:
                if not partial_atom_graph.trigger_global_transitions(rng):
                    break
                # A -1 arm starts a fresh, independent owner timeline.
                checkpoints.clear()
                _sync_transactions()
                _commit_mutation()
                continue

            parent_map = tracker.parent_map

            # Finalize parked instances whose subtree finished, deepest first
            # (a child's caps credit its ancestors before those are decided).
            finalized = None
            for sto_atom_id in sorted(pending_termination, key=lambda i: len(parent_map.get(i, [])), reverse=True):
                if any(sto_atom_id in parent_map.get(d, []) for d in unterminated_sto_atom_ids):
                    continue
                finalized = sto_atom_id
                break
            if finalized is not None:
                if _DECISION_TRACE is not None:
                    _DECISION_TRACE.append(
                        {
                            "kind": "finalize",
                            "id": finalized,
                            "gen": tracker._stochastic_atom_id_to_gen_id[finalized],
                            "expected": tracker._sto_atom_id_expected_molw[finalized],
                            "actual": tracker._sto_atom_id_actual_molw[finalized],
                        }
                    )
                # Finish the child's own level. A live-ancestor continuation
                # is promoted into that ancestor's pool inside
                # _finalize_pending and fires later as an ordinary owner-level
                # propagation draw; only the inter-object/root case (no live
                # ancestor) still fires a transition directly here.
                continuation = _finalize_pending(finalized)
                checkpoints.pop(finalized, None)
                _sync_transactions()
                if continuation is not None:
                    origin_sto_atom_id, _continuation_level, continuation_gen_id = continuation
                    # No live owner can cross; this is an inter-object/root
                    # continuation, not an owner-level epoch.
                    _new_sto_atom_id, transition_success = partial_atom_graph.transition_graph(
                        origin_sto_atom_id,
                        continuation_gen_id,
                        rng,
                    )
                    if transition_success:
                        _commit_mutation()
                continue

            growable = [i for i in unterminated_sto_atom_ids if i not in pending_termination]
            if not growable:
                # All live instances are parked: the deepest one has no live
                # descendants, so the next pass finalizes it.
                continue

            # Active is the deepest live non-parked instance (every
            # descendant lists all of its ancestors, so one pass suffices).
            active_sto_atom_id = growable[0]
            for sto_atom_id in growable:
                if sto_atom_id in parent_map:
                    for ancestor in parent_map[sto_atom_id]:
                        if active_sto_atom_id == ancestor:
                            active_sto_atom_id = sto_atom_id

            if len(partial_atom_graph.get_open_half_bonds(active_sto_atom_id)[1]) == 0:
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
                _sync_transactions()
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
            crossing_candidates = sorted(
                growable,
                key=lambda i: len(parent_map.get(i, [])),
                reverse=True,
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
                own_termination_weight, avg_termination_weight = _total_termination_mw(
                    partial_atom_graph,
                    tracker,
                    unterminated_sto_atom_ids,
                    sto_atom_id,
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
                    with _checkpoint_snapshot(checkpoint) as snapshot_graph:
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
                if not adopt_overshoot:
                    if checkpoint["graph"] is not None:
                        partial_atom_graph = checkpoint["graph"]
                    else:
                        partial_atom_graph = checkpoint["transaction"].rollback(rng)
                    # Keep consuming the caller's already-advanced stream;
                    # rewinding it would replay the rejected over-step and can
                    # loop forever.
                    partial_atom_graph.stochastic_tracker._rng = rng
                    partial_atom_graph.stochastic_tracker.mark_path_conditional()
                    pending_termination = set(checkpoint["pending"])
                    max_step_gain = dict(checkpoint["max_step_gain"])
                    gain_floor = dict(checkpoint["gain_floor"])
                    last_checked_proj = dict(checkpoint["last_checked_proj"])
                    own_termination_cache = dict(checkpoint["own_termination_cache"])
                    avg_termination_cache = dict(checkpoint["avg_termination_cache"])
                    owner_epochs = dict(checkpoint["owner_epochs"])
                    forced_overshoot_no_boundary = set(checkpoint["forced_overshoot_no_boundary"])
                    restored_tracker = partial_atom_graph.stochastic_tracker
                    restored_live = set(restored_tracker.get_unterminated_sto_atom_ids())
                    checkpoints = {owner: prior for owner, prior in checkpoint["checkpoints"].items() if (owner in restored_live and prior["epoch"] + 1 == owner_epochs.get(owner, 0))}
                    _sync_transactions()
                else:
                    checkpoints.pop(crossing_sto_atom_id, None)
                    _sync_transactions()
                # Parking changes control state only, not the mass epoch. An
                # unconsumed promoted continuation site in the parked
                # instance's pool needs no redirect: it carries its
                # termination modes and is capped by the instance's own
                # terminate pass at finalization.
                pending_termination.add(crossing_sto_atom_id)

            else:
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
                    _sync_transactions()
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

        checkpoints.clear()
        _sync_transactions()

        _collapse_phantom_nodes(partial_atom_graph.atom_graph)

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

        if not collect_info:
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
        include_sequences = metadata_mode >= _MetadataLevel.COMPACT_SEQUENCES
        compact = partial_atom_graph.compact_metadata(include_sequences)
        compact.labeled_bond_counts = _labeled_bond_counts(
            compact.bond_counts,
            self._origin_endpoint,
        )
        if internal_request:
            legacy_units = None
            legacy_sequences = None
            if metadata_mode is _MetadataLevel.FULL_LEGACY:
                legacy_units, _legacy_bonds, legacy_sequences = (
                    partial_atom_graph.materialize_legacy_metadata()
                )
            return _SampledMolecule(
                partial_atom_graph.atom_graph,
                compact,
                actual_mol_weights,
                distributions,
                legacy_units,
                legacy_sequences,
            )

        units, bonds, sequences = partial_atom_graph.materialize_legacy_metadata()
        return (
            partial_atom_graph.atom_graph,
            units,
            bonds,
            sequences,
            actual_mol_weights,
            distributions,
        )


    @staticmethod
    def _unit_graph_with_stars(unit_graph, origin_bond_id):
        """
        Copy of `unit_graph` with a star atom bonded to every atom whose origin
        is a connection atom (has a derived bond id); the star's map number is
        that bond id, so the P-SMILES prints numbered stars ``[*:n]``.
        """
        star_graph = unit_graph.copy()
        for node, data in unit_graph.nodes(data=True):
            bond_id = origin_bond_id.get(data["origin_idx"])
            if bond_id is not None:
                star_node = ("star", node)
                # The converter renders map numbers as connection + 1.
                star_graph.add_node(star_node, **{"atomic_num": 0, _AROMATIC_NAME: False, "charge": 0, "connection": bond_id - 1})
                star_graph.add_edge(node, star_node, **{_BOND_TYPE_NAME: 1, _AROMATIC_NAME: False})
        return star_graph

    def _iter_chain_records(
        self,
        n_samples,
        molecule_format,
        collect_info,
        max_discards,
        termination_flag,
        parallel,
        n_workers,
        seed,
        start_index=0,
        include_sequences=True,
        native_diagnostics_path=None,
        max_worker_restarts=2,
        defer_conversion=False,
        parallel_scheduler=None,
        use_repeat_units_as_source=False,
    ):
        """Yield ordered per-chain success/failure records.

        Chain-local retry and RNG policy lives here so fixed-size and
        convergence-driven creation consume the same sampling engine.
        """
        seed_sequences = None
        if seed is not None or (parallel and n_workers > 1):
            seed_sequences = np.random.SeedSequence(seed).spawn(n_samples)

        if parallel and n_workers > 1:
            chain_jobs = [
                (start_index + local_index, seed_sequence)
                for local_index, seed_sequence in enumerate(seed_sequences)
            ]
            yield from _parallel_chain_records(
                self,
                chain_jobs,
                molecule_format,
                collect_info,
                max_discards,
                termination_flag,
                include_sequences,
                native_diagnostics_path,
                n_workers,
                max_worker_restarts,
                defer_conversion,
                parallel_scheduler,
                use_repeat_units_as_source,
            )
            return

        for local_index in range(n_samples):
            chain_index = start_index + local_index
            rng = (
                np.random.default_rng(seed_sequences[local_index])
                if seed_sequences is not None
                else None
            )
            discards = 0
            reasons = Counter()
            first_cause = None
            deferred_warnings = []
            record = None
            while True:
                sample, attempt_reasons, cause, attempt_warnings = _attempt_chain(
                    self,
                    collect_info,
                    termination_flag,
                    rng,
                    use_repeat_units_as_source,
                )
                deferred_warnings.extend(attempt_warnings)
                if sample is not None:
                    seed_sequence = (
                        seed_sequences[local_index]
                        if seed_sequences is not None
                        else None
                    )
                    if defer_conversion:
                        record = _defer_chain_conversion(
                            sample,
                            collect_info,
                            chain_index,
                            seed_sequence,
                            native_diagnostics_path,
                        )
                    else:
                        record = _convert_chain(
                            sample,
                            molecule_format,
                            collect_info,
                            include_sequences,
                            chain_index,
                            seed_sequence,
                            native_diagnostics_path,
                        )
                    break
                discards += 1
                reasons.update(attempt_reasons)
                if first_cause is None and cause is not None:
                    first_cause = _detach_tracebacks(cause)
                if discards >= max_discards:
                    break
            yield {
                "chain_index": chain_index,
                "record": record,
                "discards": discards,
                "reasons": tuple(reasons.items()),
                "first_cause": first_cause,
                "warnings": [
                    _portable_warning(caught) for caught in deferred_warnings
                ],
            }
            if record is None:
                # Serial sampling preserves the accepted prefix and stops at
                # the first chain whose consecutive-discard budget expires.
                return

    def _materialize_unit_counts(self, unit_counts):
        if not hasattr(self, "_canonical_unit_info"):
            return {
                unit_id: {"count": count}
                for unit_id, count in unit_counts.items()
            }
        canonical_units = {}
        for unit_id, count in unit_counts.items():
            info = self._canonical_unit_info[unit_id]
            psmiles = info["psmiles"]
            if psmiles is None:
                star_mol = mol_graph_to_rdkit_mol(
                    self._unit_graph_with_stars(
                        self._canonical_unit_prototypes[unit_id],
                        self._origin_bond_id,
                    ),
                    kekulize=False,
                )
                psmiles = rdkit_mol_to_smiles(star_mol)
                info["psmiles"] = psmiles
            canonical_units[unit_id] = {
                **info,
                "psmiles": psmiles,
                "subgraph": deepcopy(info["subgraph"]),
                "count": count,
            }
        return dict(
            sorted(
                canonical_units.items(),
                key=lambda item: (item[0][0], int(item[0][1:])),
            )
        )

    def _records_to_ensemble_data(self, records, materialize_units=True):
        """Materialize public aggregate metadata from accepted chain records."""
        molecules = [record["molecule"] for record in records]
        molecular_weights = [record["molecular_weight"] for record in records]
        bond_counts = Counter()
        labeled_bond_counts = Counter()
        unit_counts = Counter()
        mol_weight_lists = {}
        ensemble_distributions = {}
        sequences = []

        for record in records:
            if isinstance(record, _ChainRecord):
                unit_counts.update(record.unit_counts or {})
                labeled_bond_counts.update(record.contact_counts or {})
            else:
                for unit, count in (record["molecule_units"] or {}).items():
                    unit_id = self._unit_id_by_origin[
                        next(iter(unit.nodes(data=True)))[1]["origin_idx"]
                    ]
                    unit_counts[unit_id] += count
                bond_counts.update(record["bonds"])
            for stochastic_id, weights in record["mol_weights"].items():
                mol_weight_lists.setdefault(stochastic_id, []).extend(weights)
            for stochastic_id, distribution in record["distributions"].items():
                ensemble_distributions.setdefault(stochastic_id, distribution)
            sequences.append(record["sequences"])

        canonical_units = (
            self._materialize_unit_counts(unit_counts)
            if materialize_units
            else {
                unit_id: {"count": count}
                for unit_id, count in unit_counts.items()
            }
        )
        return EnsembleData(
            chains=molecules,
            units=canonical_units,
            bonds=(
                _bond_records_from_labeled(labeled_bond_counts)
                if labeled_bond_counts
                else _bond_records(bond_counts, self._origin_endpoint)
            ),
            sequences=sequences,
            mol_weights=mol_weight_lists,
            distributions=ensemble_distributions,
            molecular_weights=molecular_weights,
        )

    def create_ensemble(self, n_samples, output_format="mol_graph", ensemble_info=False, max_number_of_discarded_chains: int = 100, termination_flag: Optional[int] = None, json_file: Optional[str] = None, json_max_chains: Optional[int] = None, parallel: bool = False, n_workers: Optional[int] = None, seed: Optional[int] = None, native_diagnostics_path=None, max_worker_restarts=2):
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
        JSON. The file's chains follow ``output_format`` -- SMILES strings, or
        node-link graph dicts for ``"mol_graph"`` (roughly two orders of
        magnitude larger; picking the format is picking the file size) -- as
        recorded in ``format.chain_format``; sequences are written as SMILES
        regardless. ``json_max_chains`` caps only the number of chains stored
        in the file (default ``None`` = all); statistics always cover every
        sampled chain.

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

        ``native_diagnostics_path`` optionally names an append-only JSONL file
        updated immediately before each accepted-chain RDKit native stage.
        ``max_worker_restarts`` limits transparent process-pool rebuilds after
        worker death; completed ordered chain results are preserved.
        """

        supported_formats = {"smiles", "mol_graph"}
        molecule_format = output_format.lower()
        if molecule_format not in supported_formats:
            raise ValueError(f"Unsupported format: '{output_format}'. " f"Please choose from {list(supported_formats)}.")

        if n_workers is not None and not parallel:
            raise ValueError("n_workers only applies to parallel=True; the default mode is serial.")
        if parallel and n_workers is not None and n_workers < 1:
            raise ValueError(f"n_workers must be a positive integer, got {n_workers}.")
        if max_worker_restarts < 0:
            raise ValueError(
                f"max_worker_restarts must be non-negative, got {max_worker_restarts}."
            )
        if parallel and n_workers is None:
            n_workers = max(1, min((os.cpu_count() or 1) - 2, n_samples))

        # The JSON dump needs unit/sequence info even when the caller did not
        # ask for the returned ensemble information.
        collect_info = (
            _MetadataLevel.COMPACT_SEQUENCES
            if ensemble_info or json_file is not None
            else _MetadataLevel.NONE
        )

        total_discards = 0
        discard_reasons = Counter()
        first_discard_cause = None

        records = []
        failed = False
        for chain_result in self._iter_chain_records(
            n_samples=n_samples,
            molecule_format=molecule_format,
            collect_info=collect_info,
            max_discards=max_number_of_discarded_chains,
            termination_flag=termination_flag,
            parallel=parallel,
            n_workers=n_workers,
            seed=seed,
            native_diagnostics_path=native_diagnostics_path,
            max_worker_restarts=max_worker_restarts,
        ):
            for message, category, filename, lineno in chain_result["warnings"]:
                warnings.warn_explicit(message, category, filename, lineno)
            total_discards += chain_result["discards"]
            discard_reasons.update(dict(chain_result["reasons"]))
            if chain_result["record"] is not None:
                records.append(chain_result["record"])
            else:
                failed = True
                if first_discard_cause is None and chain_result["first_cause"] is not None:
                    first_discard_cause = chain_result["first_cause"]

        if failed:
            warnings.warn(TooManyDiscardedChains(max_number_of_discarded_chains), stacklevel=1)
            if not records:
                warnings.warn(
                    DiscardedSamplingPaths(total_discards, tuple(discard_reasons.items())),
                    stacklevel=2,
                )
                if first_discard_cause is not None:
                    raise first_discard_cause
                return None

        if total_discards:
            warnings.warn(
                DiscardedSamplingPaths(total_discards, tuple(discard_reasons.items())),
                stacklevel=2,
            )

        list_of_molecules = [record["molecule"] for record in records]
        if not collect_info:
            return list_of_molecules
        ensemble_data = self._records_to_ensemble_data(records)
        canonical_units = ensemble_data.units
        bond_records = ensemble_data.bonds
        mol_weight_lists = ensemble_data.mol_weights
        ensemble_distributions = ensemble_data.distributions
        list_of_sequences = ensemble_data.sequences

        if json_file is not None:
            # The file's chains follow output_format (the caller's format
            # choice decides the file size); sequences are always SMILES.
            if molecule_format == "smiles":

                def _chain_json(molecule):
                    return molecule

                def _sequence_unit_smiles(unit):
                    return unit

            else:

                def _chain_json(molecule):
                    return nx.node_link_data(molecule, edges="edges")

                def _sequence_unit_smiles(unit):
                    return mol_graph_to_smiles(unit, kekulize=False)

            saved_chains = list_of_molecules if json_max_chains is None else list_of_molecules[:json_max_chains]
            json_data = {"string": self._generative_graph.graph.get("g2rins_string", "")}
            json_data.update(generative_graph_json_data(self._generative_graph))
            json_data["format"]["chain_format"] = molecule_format
            json_data["ensemble"] = {
                "units": {unit_id: {**info, "subgraph": nx.node_link_data(info["subgraph"], edges="edges")} for unit_id, info in canonical_units.items()},
                "chains": [_chain_json(molecule) for molecule in saved_chains],
                "bonds": bond_records,
                "mol_weights": mol_weight_lists,
                "distributions": ensemble_distributions,
                "sequences": [[[_sequence_unit_smiles(unit) for unit in sequence] for sequence in chain_sequences] for chain_sequences in list_of_sequences],
            }
            with open(json_file, "w") as file_handle:
                json.dump(json_data, file_handle, indent=2)

        if ensemble_info:
            return ensemble_data
        return list_of_molecules

    @_with_parallel_convergence_scheduler
    def create_ensemble_until_converged(
        self,
        batch_size=25,
        max_samples=1500,
        window=4,
        mass_tolerance=0.002,
        contact_tolerance=0.01,
        output_format="mol_graph",
        max_number_of_discarded_chains=100,
        termination_flag=None,
        parallel=False,
        n_workers=None,
        seed=None,
        progress_callback=None,
        retain_chains=True,
        retain_sequences=True,
        metadata=True,
        reservoir_size=None,
        sample_callback=None,
        checkpoint=None,
        checkpoint_callback=None,
        native_diagnostics_path=None,
        max_worker_restarts=2,
        checkpoint_policy="full",
        use_repeat_units_as_source=False,
    ):
        """Sample batches until cumulative mass and contact statistics stabilize.

        This is an opt-in alternative to :meth:`create_ensemble`; the existing
        fixed-size API and its return type remain unchanged. Convergence requires
        ``window`` consecutive batch-to-batch transitions whose relative Mn/Mw
        changes and absolute contact-frequency changes satisfy the supplied
        tolerances. Sampling stops without convergence at ``max_samples``.

        An integer ``seed`` follows the established batch convention: batch
        ``i`` uses ``seed + i * batch_size``. Results are reproducible for a
        fixed batch size, execution mode, and worker count.

        Statistics always include every accepted chain. ``retain_chains`` and
        ``retain_sequences`` control which sample-level outputs are kept;
        ``metadata`` controls returned aggregate unit/contact metadata.
        ``reservoir_size`` bounds retained samples with independent reservoir
        sampling, without changing generation RNG streams. ``sample_callback``
        receives ``(global_chain_index, record)`` for every accepted chain.
        To resume at a batch boundary, pass a :class:`ConvergenceCheckpoint`
        previously received by ``checkpoint_callback``. Exact resume requires
        an integer seed. ``checkpoint_policy='full'`` preserves retained
        payloads; ``'statistics'`` omits them from checkpoints and therefore
        returns no retained sample payload after resume. ``native_diagnostics_path`` has the same durable
        native-stage logging semantics as :meth:`create_ensemble`, as does the
        ``max_worker_restarts`` recovery policy.
        Set ``use_repeat_units_as_source=True`` to seed each chain from a
        repeat unit when the polymer has no initiator.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if max_samples < 1:
            raise ValueError(f"max_samples must be positive, got {max_samples}.")
        if reservoir_size is not None and reservoir_size < 0:
            raise ValueError(
                f"reservoir_size must be non-negative, got {reservoir_size}."
            )
        if sample_callback is not None and not callable(sample_callback):
            raise TypeError("sample_callback must be callable or None.")
        if checkpoint_callback is not None and not callable(checkpoint_callback):
            raise TypeError("checkpoint_callback must be callable or None.")
        if checkpoint_policy not in {"full", "statistics"}:
            raise ValueError(
                "checkpoint_policy must be 'full' or 'statistics'."
            )
        if (checkpoint is not None or checkpoint_callback is not None) and seed is None:
            raise ValueError("seed is required for resumable convergence checkpoints.")
        if checkpoint is not None and not isinstance(
            checkpoint, ConvergenceCheckpoint
        ):
            raise TypeError("checkpoint must be a ConvergenceCheckpoint or None.")
        supported_formats = {"smiles", "mol_graph"}
        molecule_format = output_format.lower()
        if molecule_format not in supported_formats:
            raise ValueError(
                f"Unsupported format: '{output_format}'. "
                f"Please choose from {list(supported_formats)}."
            )
        if n_workers is not None and not parallel:
            raise ValueError(
                "n_workers only applies to parallel=True; the default mode is serial."
            )
        if parallel and n_workers is not None and n_workers < 1:
            raise ValueError(f"n_workers must be a positive integer, got {n_workers}.")
        if max_worker_restarts < 0:
            raise ValueError(
                f"max_worker_restarts must be non-negative, got {max_worker_restarts}."
            )
        use_default_workers = parallel and n_workers is None

        tracker = ConvergenceTracker(
            window=window,
            mass_tolerance=mass_tolerance,
            contact_tolerance=contact_tolerance,
        )
        retain_samples = retain_chains or retain_sequences
        if sample_callback is not None:
            metadata_mode = _MetadataLevel.FULL_LEGACY
        elif retain_sequences:
            metadata_mode = _MetadataLevel.COMPACT_SEQUENCES
        else:
            metadata_mode = _MetadataLevel.COUNTS
        checkpoint_settings = {
            "batch_size": batch_size,
            "output_format": molecule_format,
            "window": window,
            "mass_tolerance": mass_tolerance,
            "contact_tolerance": contact_tolerance,
            "max_number_of_discarded_chains": max_number_of_discarded_chains,
            "termination_flag": termination_flag,
            "parallel": parallel,
            "n_workers": n_workers,
            "max_worker_restarts": max_worker_restarts,
            "retain_chains": retain_chains,
            "retain_sequences": retain_sequences,
            "metadata": metadata,
            "reservoir_size": reservoir_size,
            "checkpoint_policy": checkpoint_policy,
            "use_repeat_units_as_source": bool(use_repeat_units_as_source),
        }
        reservoir_rng = None
        if retain_samples and reservoir_size is not None:
            retention_seed = np.random.SeedSequence(seed).spawn(2)[1]
            reservoir_rng = np.random.default_rng(retention_seed)

        if checkpoint is None:
            aggregate_unit_counts = Counter()
            aggregate_bond_counts = Counter()
            aggregate_distributions = {}
            batch_index = 0
            accepted_count = 0
            next_chain_index = 0
            mass_sum = 0.0
            mass_square_sum = 0.0
            contact_counts = Counter()
            contact_total = 0
            retained_records = []
            retained_seen = 0
        else:
            if checkpoint.seed != seed:
                raise ValueError("checkpoint seed does not match seed.")
            saved_settings = dict(checkpoint.settings)
            saved_settings.setdefault(
                "checkpoint_policy",
                getattr(checkpoint, "policy", "full"),
            )
            saved_settings.setdefault("use_repeat_units_as_source", False)
            if saved_settings != checkpoint_settings:
                raise ValueError("checkpoint settings do not match this convergence run.")
            aggregate_unit_counts = Counter(
                {
                    unit_id: unit_data["count"]
                    for unit_id, unit_data in checkpoint.aggregate.units.items()
                }
            )
            aggregate_bond_counts = Counter(
                {
                    tuple(zip(record["labels"], record["nodes"])): record["count"]
                    for record in checkpoint.aggregate.bonds
                }
            )
            aggregate_distributions = dict(checkpoint.aggregate.distributions)
            batch_index = checkpoint.batch_index
            accepted_count = checkpoint.accepted_count
            next_chain_index = checkpoint.next_chain_index
            mass_sum = checkpoint.mass_sum
            mass_square_sum = checkpoint.mass_square_sum
            contact_counts = Counter(checkpoint.contact_counts)
            contact_total = checkpoint.contact_total
            tracker.history = deepcopy(checkpoint.convergence_history)
            retained_records = list(deepcopy(checkpoint.retained_records))
            retained_seen = checkpoint.retained_seen
            if reservoir_rng is not None and checkpoint.reservoir_rng_state is not None:
                reservoir_rng.bit_generator.state = deepcopy(
                    checkpoint.reservoir_rng_state
                )

        retain_checkpoint_payloads = not (
            checkpoint is not None and checkpoint_policy == "statistics"
        )

        converged = tracker.converged()
        progress_header_pending = True

        while accepted_count < max_samples and not converged:
            current_batch_size = min(batch_size, max_samples - accepted_count)
            batch_seed = None if seed is None else seed + batch_index * batch_size
            batch_workers = n_workers
            if use_default_workers:
                batch_workers = max(
                    1,
                    min((os.cpu_count() or 1) - 2, current_batch_size),
                )
            batch_accepted = 0
            total_discards = 0
            discard_reasons = Counter()
            first_discard_cause = None
            failed = False
            for chain_result in self._iter_chain_records(
                n_samples=current_batch_size,
                molecule_format=molecule_format,
                collect_info=metadata_mode,
                max_discards=max_number_of_discarded_chains,
                termination_flag=termination_flag,
                parallel=parallel,
                n_workers=batch_workers,
                seed=batch_seed,
                start_index=next_chain_index,
                include_sequences=retain_sequences or sample_callback is not None,
                native_diagnostics_path=native_diagnostics_path,
                max_worker_restarts=max_worker_restarts,
                defer_conversion=True,
                parallel_scheduler=_ACTIVE_PARALLEL_SCHEDULER.get(),
                use_repeat_units_as_source=use_repeat_units_as_source,
            ):
                next_chain_index = max(
                    next_chain_index,
                    chain_result["chain_index"] + 1,
                )
                for message, category, filename, lineno in chain_result["warnings"]:
                    warnings.warn_explicit(message, category, filename, lineno)
                total_discards += chain_result["discards"]
                discard_reasons.update(dict(chain_result["reasons"]))
                if chain_result["record"] is not None:
                    deferred = chain_result["record"]
                    if not isinstance(deferred, _DeferredChainRecord):
                        raise TypeError(
                            "convergence sampling requires a deferred chain record"
                        )
                    chain_index = chain_result["chain_index"]
                    sample = deferred.sample
                    metadata_record = sample.metadata
                    molecular_weight = deferred.molecular_weight
                    accepted_count += 1
                    batch_accepted += 1
                    mass_sum += molecular_weight
                    mass_square_sum += molecular_weight * molecular_weight
                    for pair, count in metadata_record.labeled_bond_counts.items():
                        contact_key = "|".join(label for label, _node in pair)
                        contact_counts[contact_key] += count
                        contact_total += count
                    if metadata:
                        aggregate_unit_counts.update(metadata_record.unit_counts)
                        aggregate_bond_counts.update(
                            metadata_record.labeled_bond_counts
                        )
                        for stochastic_id, distribution in sample.distributions.items():
                            aggregate_distributions.setdefault(
                                stochastic_id,
                                distribution,
                            )

                    materialized = None
                    if sample_callback is not None:
                        materialized = _materialize_deferred_chain(
                            deferred,
                            molecule_format,
                            metadata_mode,
                            True,
                        )
                        sample_callback(
                            chain_index,
                            materialized.callback_record(),
                        )
                    if retain_samples:
                        retained_seen += 1
                        retained_position = None
                        if reservoir_size is None:
                            retained_position = len(retained_records)
                        elif retained_seen <= reservoir_size:
                            retained_position = retained_seen - 1
                        elif reservoir_size:
                            replacement = int(
                                reservoir_rng.integers(retained_seen)
                            )
                            if replacement < reservoir_size:
                                retained_position = replacement
                        if (
                            retained_position is not None
                            and retain_checkpoint_payloads
                        ):
                            if materialized is None:
                                materialized = _materialize_deferred_chain(
                                    deferred,
                                    molecule_format,
                                    metadata_mode,
                                    retain_sequences,
                                )
                            retained_entry = (chain_index, materialized)
                            if retained_position == len(retained_records):
                                retained_records.append(retained_entry)
                            else:
                                retained_records[retained_position] = retained_entry
                else:
                    failed = True
                    if first_discard_cause is None:
                        first_discard_cause = chain_result["first_cause"]

            if failed:
                warnings.warn(
                    TooManyDiscardedChains(max_number_of_discarded_chains),
                    stacklevel=1,
                )
            if total_discards:
                warnings.warn(
                    DiscardedSamplingPaths(
                        total_discards,
                        tuple(discard_reasons.items()),
                    ),
                    stacklevel=2,
                )
            if not batch_accepted:
                if first_discard_cause is not None:
                    raise first_discard_cause
                break

            mn = mass_sum / accepted_count
            mw = mass_square_sum / mass_sum
            contacts = {
                key: count / contact_total
                for key, count in contact_counts.items()
            } if contact_total else {}
            tracker.record(accepted_count, mn, mw, contacts)
            batch_index += 1

            if checkpoint_callback is not None:
                checkpoint_callback(
                    ConvergenceCheckpoint(
                        next_chain_index=next_chain_index,
                        accepted_count=accepted_count,
                        batch_index=batch_index,
                        seed=seed,
                        mass_sum=mass_sum,
                        mass_square_sum=mass_square_sum,
                        contact_counts=dict(contact_counts),
                        contact_total=contact_total,
                        aggregate=EnsembleData(
                            [],
                            {
                                unit_id: {"count": count}
                                for unit_id, count in aggregate_unit_counts.items()
                            },
                            _bond_records_from_labeled(aggregate_bond_counts),
                            [],
                            {},
                            dict(aggregate_distributions),
                            [],
                        ),
                        convergence_history=deepcopy(tracker.history),
                        retained_records=(
                            tuple(deepcopy(retained_records))
                            if checkpoint_policy == "full"
                            else ()
                        ),
                        retained_seen=retained_seen,
                        reservoir_rng_state=(
                            deepcopy(reservoir_rng.bit_generator.state)
                            if reservoir_rng is not None
                            else None
                        ),
                        settings=dict(checkpoint_settings),
                        policy=checkpoint_policy,
                    )
                )

            if progress_callback is not None:
                progress_row = (
                    f"{batch_index:5d} | {accepted_count:7d} | "
                    f"{mn:10.1f} | {mw:10.1f} | {tracker.progress()}"
                )
                if progress_header_pending:
                    progress_row = (
                        "Batch | Samples |         Mn |         Mw | Status\n"
                        "----- | ------- | ---------- | ---------- | ------\n"
                        f"{progress_row}"
                    )
                    progress_header_pending = False
                progress_callback(progress_row)
            if tracker.converged():
                converged = True
                break
            if batch_accepted < current_batch_size:
                break

        if not accepted_count:
            return None
        retained_records.sort(key=lambda item: item[0])
        retained = [record for _index, record in retained_records]
        chains = [record.molecule for record in retained] if retain_chains else []
        sequences = (
            [record.sequences for record in retained]
            if retain_sequences
            else []
        )
        masses = (
            [record.molecular_weight for record in retained]
            if retain_samples
            else []
        )
        retained_mol_weights = {}
        if metadata:
            for record in retained:
                for stochastic_id, weights in record.mol_weights.items():
                    retained_mol_weights.setdefault(stochastic_id, []).extend(weights)
        mn = mass_sum / accepted_count
        mw = mass_square_sum / mass_sum
        return ConvergedEnsembleData(
            chains=chains,
            units=(
                self._materialize_unit_counts(
                    aggregate_unit_counts
                )
                if metadata
                else {}
            ),
            bonds=(
                _bond_records_from_labeled(aggregate_bond_counts)
                if metadata
                else []
            ),
            sequences=sequences,
            mol_weights=retained_mol_weights if metadata else {},
            distributions=aggregate_distributions if metadata else {},
            molecular_weights=masses,
            converged=converged,
            n_batches=batch_index,
            convergence_settings={
                "batch_size": batch_size,
                "max_samples": max_samples,
                "window": window,
                "mass_tolerance": mass_tolerance,
                "contact_tolerance": contact_tolerance,
                "retain_chains": retain_chains,
                "retain_sequences": retain_sequences,
                "metadata": metadata,
                "reservoir_size": reservoir_size,
                "checkpoint_policy": checkpoint_policy,
                "use_repeat_units_as_source": bool(
                    use_repeat_units_as_source
                ),
            },
            convergence_trace=tracker.history,
            number_average_molecular_weight=mn,
            weight_average_molecular_weight=mw,
            dispersity=mw / mn,
        )
