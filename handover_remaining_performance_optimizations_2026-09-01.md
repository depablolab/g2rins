# G2RINS remaining performance optimization handover — 2026-09-01

## Purpose

This document hands the next optimization pass to another agent. It supplements:

- `handover_code_improvements_2026-08-31.txt`
- `handover_further_performance_improvements_2026-08-31.md`
- `benchmarks/PROFILE_2026-08-31.md`

Phases 1–10 from the earlier handover are implemented. Do not repeat them. The next work should be a sequence of small, independently benchmarked changes rather than a C/Cython rewrite.

## Validated environment

- Repository: `/gpfs/projects/descmod_polymer_representations/software/g2rins`
- Interpreter: `/gpfs/users/home/settelv/.conda/envs/poly_catalog_env/bin/python`
- Run source validation with `PYTHONPATH=src`.
- Inspect `git status --short`, staged changes, and unstaged changes before editing. Do not overwrite unrelated work.

Recommended initial commands:

```text
git status --short
git diff --stat
git diff --cached --stat
PYTHONPATH=src /gpfs/users/home/settelv/.conda/envs/poly_catalog_env/bin/python -m pytest -q
```

## Non-negotiable behavior

Every optimization must preserve:

- Canonical chemistry and normalized molecular topology.
- Public NetworkX graph type and atom/bond attributes.
- `origin_idx` provenance.
- Molecular weights and hydrogen-credit reconciliation.
- Unit counts, labeled contact counts, sequence contents, and ordering.
- Nested-parent mass accounting and end-group selection.
- Warning, discard, fatal, and retryable classifications.
- Exact stochastic under/overshoot probabilities.
- RNG advancement after rollback; rollback must not restore RNG state.
- Seeded serial/parallel equivalence and chain ordering.
- Existing fixed-size and convergence APIs.
- Worker crash recovery and bounded in-flight scheduling.
- Full-checkpoint resume behavior and statistics-checkpoint behavior.

Raw NetworkX node IDs need not remain identical where existing normalized references permit that, but output ordering and seeded references must remain unchanged.

## Current measured baseline

The user reran the deterministic fresh-process benchmark with three repetitions:

```text
PYTHONPATH=src /gpfs/users/home/settelv/.conda/envs/poly_catalog_env/bin/python \
  -m benchmarks.benchmark_sampling \
  --case large-linear --case branched --case nested --case high-discard \
  --runs 3 --metadata both --termination exact --summary-only
```

Medians:

| Case | Atoms | No metadata | Full legacy metadata | RSS no metadata | RSS full | Transactions | Rollbacks |
|---|---:|---:|---:|---:|---:|---:|---:|
| large-linear | 8,525 | 0.8146 s | 1.2403 s | 142.9 MiB | 151.1 MiB | 4 | 0 |
| branched | 110 | 0.0725 s | 0.0941 s | 126.5 MiB | 126.5 MiB | 5 | 1 |
| nested | 42 | 0.0258 s | 0.0327 s | 126.5 MiB | 126.5 MiB | 9 | 1 |
| high-discard | 10 | 0.7902 s | 0.9280 s | 126.5 MiB | 126.5 MiB | 366 | 35 |

Other important baseline facts:

- Large-linear IPC payload: 619,080 bytes without metadata and 1,472,638 bytes with full legacy metadata.
- Large-linear has 2,843 unit occurrences and sequence lengths `[1, 2842]`.
- High-discard performs 52 rejected attempts before success.
- All configurations report zero whole-molecule checkpoint copies.
- cProfile is for attribution only; fresh-process medians decide acceptance.

A fresh cProfile review of the current tree found:

- Large-linear/no metadata: 1.468 s profiled total; propagation 1.142 s, merge 0.372 s, temporary partial-graph construction 0.341 s, static-template realization 0.275 s, propagation-bond selection 0.232 s.
- Large-linear/full legacy: 3.214 s profiled total; legacy metadata materialization 0.965 s, sequence-unit materialization 0.955 s, `deepcopy` 0.943 s, propagation 1.900 s.
- High-discard/full legacy: 1.546 s profiled total; `deepcopy` 0.769 s and first-write frontier bucket journaling 0.533 s.
- Nested/no metadata: 0.051 s profiled total; bounded discrete distribution sampling 0.018 s.

Profiler timings vary under machine load. Retain attribution, call counts, atom counts, transactions, and rollbacks beside every result.

## Review of the proposed implementation plan

The proposed plan is sound after the following corrections:

1. Keep sequence materialization, live-instance tracking, and frontier journaling as separate commits. They have different correctness risks and benchmark targets.
2. Implement the ordered live-instance index before broad forest caching. Existing scaling proves historical-ID scanning is quadratic, while parent/child forest helpers were not significant.
3. Do not combine direct template insertion with frontier-journal changes. Both alter propagation state, making regressions difficult to localize.
4. Treat the shallow-copy sequence experiment as evidence, not production-ready proof. Attribute values must be audited and mutation isolation tested before replacing `deepcopy`.
5. Do not optimize distribution inversion until the nested scaling benchmark shows it matters after the live-index fix.
6. Do not begin a C/Cython/Rust backend unless a post-Python profile shows a sufficiently large residual dominated by flat, native-friendly work.

## Phase 12 — Replace legacy sequence `deepcopy` with a structural clone

### Evidence

`_PartialAtomGraph._materialize_sequence_unit()` deep-copies one NetworkX unit prototype for every occurrence. The large-linear case makes 2,843 copies.

A process-local experiment changed only sequence-unit prototype cloning from `deepcopy(prototype)` to `prototype.copy()`. It reduced legacy materialization from approximately 0.965 s cumulative to 0.140 s and reduced total profiled time from 3.21 s to 2.73 s despite profile noise.

### Targets

- `_PartialAtomGraph._materialize_sequence_unit()`
- `_PartialAtomGraph.materialize_legacy_metadata()`
- `_CompactMetadata.materialize_sequences()` if it uses the same cloning pattern
- `_PartialAtomGraph._record_unit_occurrence()` only if prototype creation itself remains measurable

### Work

1. Audit all node, edge, and graph attributes stored in unit prototypes.
2. Establish whether values are immutable scalars/tuples or whether nested mutable values exist.
3. Add one internal clone helper that creates a detached graph with independent graph, node-attribute, and edge-attribute dictionaries.
4. Share only values proven immutable. Deep-copy nested mutable values selectively if required.
5. Use the helper for representative units and sequence units.
6. Keep every materialized occurrence as a separately mutable NetworkX graph.

Do not return the prototype itself and do not share node/edge attribute dictionaries between occurrences.

### Required tests

- Mutating one materialized sequence graph does not alter another occurrence, the representative unit, the prototype, or later output.
- Graph-level, node-level, and edge-level nested mutable attributes are detached where supported.
- Existing sequence signatures and unit counts remain identical for all seeded benchmark cases.
- JSON and pickle round trips remain unchanged in public shape.
- Compact/full metadata parity remains exact.

### Acceptance

- Full seeded-reference suite passes unchanged.
- Large-linear full-legacy fresh-process median improves materially.
- No-metadata timing remains statistically unchanged.
- Full output remains independently mutable.

## Phase 13 — Maintain an ordered live stochastic-instance index

### Evidence

`_StochasticObjectTracker.get_unterminated_sto_atom_ids()` currently scans all historical IDs in reverse registration order and filters the terminated set.

The Phase 11 scaling gate from the previous profile established:

| Outer target | Atoms | Live-ID calls | Max registered IDs | Max live IDs | Historical IDs visited |
|---:|---:|---:|---:|---:|---:|
| 800 | 51 | 58 | 11 | 3 | 375 |
| 4,000 | 298 | 288 | 47 | 3 | 7,107 |
| 12,000 | 912 | 878 | 137 | 3 | 61,550 |
| 40,000 | 3,008 | 2,911 | 449 | 3 | 658,045 |
| 120,000 | 9,009 | 8,751 | 1,343 | 3 | 5,885,363 |

The largest direct probe attributed about 0.523 s to this helper. The live set never exceeded three IDs.

### Targets

- `_StochasticObjectTracker.__init__()`
- `_StochasticObjectTracker.register_new_atom_instance()`
- `_StochasticObjectTracker.terminate()`
- `_StochasticObjectTracker.get_unterminated_sto_atom_ids()`
- `_GrowthTransaction` tracker journaling and restoration
- Legacy checkpoint snapshot parity

### Preferred design

Maintain an ordered live-ID structure preserving registration order, then expose IDs in exactly the same reverse-registration order as today.

A plain `dict[int, None]` is a suitable starting point because insertion order is guaranteed. Termination removes the ID. Registration appends it. Re-registration of an existing ID should never occur without rollback; rollback must restore exact position/order.

Do not derive ordering from a `set`.

### Transaction requirements

- Registering an ID must be journaled in every compatible active transaction.
- Terminating an existing ID must record membership and original order sufficiently to restore the exact historical view.
- Snapshot creation and rollback must produce the same `get_unterminated_sto_atom_ids()` result as the old implementation.
- Legacy `_USE_LEGACY_CHECKPOINTS` behavior must include the new index.
- Avoid copying the entire live structure on every mutation; that would replace one quadratic path with another.

### Required tests

- Registration and termination preserve the old reverse-registration order.
- Rollback restores a terminated old ID and removes newly registered IDs.
- Nested descendant mutations are visible to every compatible ancestor journal.
- Multiple simultaneously active ancestor transactions restore the same order.
- Seeded journal-versus-legacy parity in exact, forced-under, and forced-over modes.
- The 120,000-target nested scaling case returns the same normalized result and visits work proportional to live IDs rather than historical IDs.

### Acceptance

- `get_unterminated_sto_atom_ids()` scales with live IDs.
- The 9,009-atom nested scaling case improves materially.
- Existing small nested cases do not regress significantly.
- Broad parent/child/depth caches remain deferred unless reprofiled evidence supports them.

## Phase 14 — Replace generic frontier-bucket `deepcopy`

### Evidence

In high-discard/full metadata, `_GrowthTransaction.record_frontier_bucket()` accounts for approximately 0.533 s cumulative. It deep-copies a complete list of `_HalfAtomBond` objects on first write.

`_HalfAtomBond` contains many immutable scalar/template-derived references plus mutable mode maps and lists. Generic `deepcopy` traverses more state than rollback requires.

### Targets

- `_HalfAtomBond`
- `_GrowthTransaction.record_frontier_bucket()`
- `_GrowthTransaction._snapshot_frontier()`
- `_GrowthTransaction.rollback()`
- Promotion/rebinding methods that mutate `_HalfAtomBond`

### Safer first implementation

1. Add a purpose-built `_HalfAtomBond.clone_for_rollback()`.
2. Copy scalar fields directly.
3. Share the immutable generative graph and immutable edge-attribute dictionaries only if mutation audits prove safety.
4. Copy mutable mode dictionaries and their lists explicitly.
5. Copy `_special_target` only to the depth required by actual mutators.
6. Replace bucket `deepcopy` with `[bond.clone_for_rollback() for bond in bucket]`.

Only after measuring this version should operation-level list journaling or copy-on-write buckets be considered.

### Follow-up opportunity

`record_tracker_mapping_key()` deep-copies mapping values generically. In particular, changes to `_stochastic_gen_id_to_atom_id[sto_gen_id]` may copy a growing historical set. Prefer membership-level journaling for that mapping rather than cloning the complete set, but keep this as a separate subcommit from half-bond cloning.

### Required tests

- Clone independence for every mutable `_HalfAtomBond` field.
- Promotion, filtering, parent reassignment, and special-target state restore exactly.
- Existing checkpoint-capture tests still prove journals are empty at capture and grow only on touched state.
- High-discard rollback leaves no discarded atom, instance, occurrence, frontier, sequence, or parent-map references.
- RNG state remains advanced after rollback.
- Seeded journal-versus-legacy parity across all reference cases and boundary modes.

### Acceptance

- High-discard `deepcopy` and frontier-journal cumulative time fall substantially.
- High-discard fresh-process wall time improves without changing 52 discards, 366 transactions, or 35 rollbacks.
- Large-linear and branched cases do not regress.

## Phase 15 — Remove temporary `_PartialAtomGraph` construction per unit

### Evidence

Large-linear/no-metadata cProfile currently attributes approximately:

- 1.142 s to propagation.
- 0.372 s to merge.
- 0.341 s to temporary `_PartialAtomGraph` construction.
- 0.275 s to static-template realization.

Each propagation builds a temporary NetworkX graph from an already prepared static source template and then copies its nodes and edges into the destination graph.

### Targets

- `_PartialAtomGraph.propagate_graph()`
- `_PartialAtomGraph.transition_graph()` and nested transition paths
- `_PartialAtomGraph._add_static_source_template()`
- `_PartialAtomGraph.merge()`
- `_StaticSourceTemplate`

### Design

Add a destination-oriented helper that realizes a `_StaticSourceTemplate` directly into the existing `atom_graph` at the current `_atom_id` watermark. It should return:

- The local-to-destination atom mapping or translated target atom.
- Newly created half-bonds with destination atom IDs.
- The exact contiguous node interval for metadata occurrence recording.

It must still:

- Allocate fresh runtime atom IDs.
- Credit the current stochastic instance and all ancestors.
- Perform special-target draws in exactly the current order with the same RNG.
- Create independent mutable half-bond mode maps.
- Apply the realized junction bond and hydrogen delta once.
- Preserve the current metadata occurrence and sequence ordering.

Keep a private fallback/parity flag until seeded equivalence is proven. Do not remove the current temporary-graph path immediately.

### Required tests

- Optimized-versus-fallback parity for every seeded case.
- Special-target draw order parity.
- Rings, aromatic bonds, association edges, dummy atoms, counterions, and multiple connector sites.
- Nested transition and global `-1` arm behavior.
- Rollback after recursively expanded nested targets.
- Exact/forced-under/forced-over parity.

### Acceptance

- Per-unit temporary `_PartialAtomGraph` and temporary NetworkX graph construction disappear from the optimized propagation path.
- Large-linear/no-metadata improves materially.
- No seeded decision trace changes.

## Phase 16 — Propagation selection fast paths

Only start after Phase 15 is measured.

### Evidence

The nested `pop_random_stochastic_bond()` inside `propagate_graph()` consumed approximately 0.232 s cumulative in the current large-linear profile. It repeatedly creates lists and small NumPy arrays, even for one eligible bond.

### Work

- Add a one-candidate path that makes no `rng.choice()` call when current behavior also makes no meaningful random choice. Verify RNG advancement before doing this; changing whether NumPy is called can change every later seeded result.
- Cache immutable candidate weights in templates where dynamic molar weighting does not alter them.
- Avoid repeated concatenation in `get_open_half_bonds()`.
- Consider indexed propagation-eligible frontier entries only if maintenance and rollback costs are lower than scanning.

### Critical RNG warning

Do not assume selecting one candidate can skip a random call. First prove whether the existing NumPy call consumes generator state for each one-candidate path. Exact seeded parity has priority over this optimization.

## Deferred work

### Bounded discrete distribution inversion

The small nested profile attributes about 0.018 s to bounded discrete inversion, including repeated SciPy `logcdf`/`logsf` calls. Defer optimization until the large nested scaling case is reprofiled after Phase 13.

If justified, compare distribution-specific bounded inverses against the current stable extreme-tail implementation. Preserve empty-support classification, exact support bounds, one RNG draw, and extreme-tail correctness. Do not replace the current method with naive `cdf` subtraction.

### Public streaming fixed-size API

`create_ensemble()` necessarily retains all returned molecules. A public iterator could bound library-managed memory, but it is an API feature rather than a transparent optimization. Convergence already supports bounded retention and statistics-only checkpoints. Design streaming separately and document lifetime, warnings, failure semantics, ordering, and parallel cleanup.

### Lazy imports

Fresh-process import probes measured approximately:

- Empty interpreter: 10.6 MiB.
- NumPy + NetworkX: 40.6 MiB.
- NumPy + NetworkX + `scipy.stats`: 90.3 MiB.
- NumPy + NetworkX + SciPy + RDKit: 104.1 MiB.
- `import g2rins`: 108.4 MiB.

Lazy SciPy/RDKit loading could reduce parse-only startup RSS, but ordinary ensemble sampling eventually needs both. This is lower priority than reducing retained/output graph memory and should not complicate the public import surface without a parse-only use-case benchmark.

## Native-code decision

Do not translate the current NetworkX/object-heavy implementation directly into C, Cython, Rust, or Numba.

Reasons:

- NumPy, SciPy, and RDKit already execute their numerical/chemistry kernels in native code.
- Remaining hotspots are Python object graphs, dictionaries, lists, NetworkX mutation, deep-copy behavior, and transaction semantics.
- Cython calling NetworkX and Python containers still pays Python object costs.
- Numba cannot efficiently compile the current NetworkX/custom-object state.
- RDKit conversion is already C++ and is not the principal sampling bottleneck.

A native backend becomes reasonable only as a compact internal representation with:

- Contiguous atom records.
- Flat attribute arrays or structs.
- Append-only edge storage.
- Indexed adjacency and frontier records.
- O(1) truncation to a transaction watermark.
- Conversion to NetworkX only at the public boundary.

Such a backend is a separate architectural project, not a translation task. Reconsider it only after Phases 12–16 and a fresh profile. Require a prototype to demonstrate at least a compelling end-to-end gain after conversion costs; NetworkX represented only about 19–22% of current profiled self time in the reviewed large cases.

## Validation sequence after every phase

1. Run the most focused new tests.
2. Run seeded references:

```text
PYTHONPATH=src /gpfs/users/home/settelv/.conda/envs/poly_catalog_env/bin/python \
  -m pytest tests/test_seeded_performance_references.py -q
```

3. Run generation, convergence, and parallel tests relevant to the change.
4. Run the full suite.
5. Run the four-case three-repetition benchmark shown above.
6. Run cProfile for:
   - large-linear, none/full
   - high-discard, full
   - nested, none
7. For Phase 13, rerun the outer-target nested scaling series through 120,000.
8. Record atom counts, accepted/discard counts, transaction counts, rollbacks, IPC bytes, and RSS with timings.
9. Compare normalized seeded output and decision traces before accepting a speedup.

Do not update `benchmarks/seeded_references.json` to make an unexplained optimization change pass.

## Suggested test locations

- `tests/test_seeded_performance_references.py`: seeded output, cache/fallback, transaction, rollback, and boundary parity.
- `tests/test_generation_regressions.py`: chemistry, nested behavior, phantom collapse, and molecular-weight accounting.
- `tests/test_convergence.py`: compact/full metadata, retention, callbacks, and checkpoints.
- `tests/test_parallel_ensemble.py`: serial/parallel ordering, payload behavior, pool recovery, and diagnostics.
- `tests/test_benchmark_sampling.py`: benchmark/profile reporting.

## Estimated implementation budget

Approximate AI-token budgets including investigation, code, tests, profiling, and documentation:

| Phase | Estimate |
|---|---:|
| Phase 12: structural sequence clone | 20k–40k |
| Phase 13: ordered live-instance index | 35k–70k |
| Phase 14: specialized frontier journal | 70k–140k |
| Phase 15: direct template insertion | 90k–180k |
| Phase 16: propagation fast paths | 35k–70k |

Recommended first tranche: Phases 12–14, approximately 125k–250k tokens. Reprofile before funding Phases 15–16.

## Immediate next action

Implement Phase 12 only. Add mutation-isolation tests first, replace generic sequence-unit `deepcopy` with a proven structural clone, run seeded/full validation, and record a new three-run benchmark. Commit or hand off that phase independently before touching live-instance or transaction state.
