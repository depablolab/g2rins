# G2RINS performance references

This directory establishes deterministic measurements and behavioral guardrails before sampler internals are optimized.

## Benchmark protocol

Use the `poly_catalog_env` environment and run from the repository root:

```bash
conda run -n poly_catalog_env python -m benchmarks.benchmark_sampling \
  --case small-linear --case branched --runs 3
```

Each configuration runs in a fresh subprocess. The JSON output contains every run plus medians, and reports:

- parser, graph-creator, and ensemble-creator construction time;
- `sample_mol_graph()` time with no metadata and full legacy metadata;
- RDKit construction, sanitization, and canonical serialization time;
- IPC pickle time and payload size;
- peak RSS, final atom/bond count, and molecular weight;
- checkpoint-copy and rollback counts;
- unit/sequence sizes when metadata is enabled;
- discard attempts, discard classifications, and warnings.

Add `--summary-only` for compact JSON containing only configuration medians;
raw runs are still executed in isolated subprocesses but are omitted from the
printed report.

Use the sampling-only CPU profiler for attribution, not wall-time comparison:

```bash
conda run -n poly_catalog_env python -m benchmarks.profile_sampling \
  --case large-linear --metadata full --limit 25
```

It excludes parsing and export, then reports self time grouped into G2RINS,
NetworkX, RDKit's Python layer, and standard-library/other code, plus the top
cumulative functions.

Available cases cover small and large linear chains, branching, nested stochastic objects, grafting, Flory-Schulz heavy tails, and a known all-discard architecture. Use fixed cases and seeds when comparing revisions, and always report atom counts beside timing because sampled chain sizes vary.

Exact stochastic rounding is the default. Compare forced boundaries explicitly with `--termination overshoot` or `--termination undershoot`. Wall-clock results are descriptive, not absolute CI limits; automated performance gates should compare relative medians on the same machine.

## Guarded metadata-copy runtime review

The [October 26 report](GRAPH_COPY_2026-10-26.md) compares legacy deep copying
with order-preserving guarded prototype copying, using alternating fresh-process
pairs. Sampling and public fixed-size API timings are separate; disabled/counts
paths do not perform the optimized copying. Reproduce with
[benchmark_graph_copy.py](benchmark_graph_copy.py), selecting `--workflow sample`
or `--workflow fixed`, repeated `--case` options, `--runs 5`, and `--warmups 1`.
Use new output paths rather than overwriting the recorded evidence.

## Compare a source tree with the original checkout

`compare_versions` runs the common public `create_ensemble()` API against two
source trees using the same interpreter, inputs, seeds, and one thread per
numeric runtime. Every repetition runs in a fresh subprocess. It reports
sampling and end-to-end wall time, peak RSS immediately after sampling, and
node-ID-independent output parity:

```bash
conda run -n poly_catalog_env python -m benchmarks.compare_versions \
  --case small-linear --case large-linear --metadata both --runs 3 \
  --source original=/path/to/original/g2rins \
  --source optimized=/path/to/optimized/g2rins \
  --include-runs --output comparison.json
```

Parity covers canonical SMILES, molecular weight, attributed topology,
provenance grouping, warnings, and—when requested—unit counts and labels,
contact counts, sequence contents, stochastic masses, and distributions. Raw
parser UUIDs and internal stochastic-instance IDs are normalized while their
grouping relationships are retained. The command exits nonzero if either
version is not repeatable or any cross-version output differs.

The August 31 original-versus-optimized measurements are summarized in
[COMPARISON_ORIGINAL_2026-08-31.md](COMPARISON_ORIGINAL_2026-08-31.md); the
machine-readable medians, all raw repetitions, and normalized output evidence
are retained in
[COMPARISON_ORIGINAL_2026-08-31.json](COMPARISON_ORIGINAL_2026-08-31.json).

For the seeded 8,525-atom `large-linear` case on the reference development machine, three exact-rounding runs after all planned optimization phases produced these medians:

| Metadata | Sampling | Peak RSS | Transactions | Whole-molecule copies |
|---|---:|---:|---:|---:|
| None | 0.992 s | 168.1 MiB | 4 | 0 |
| Full legacy output | 1.559 s | 177.1 MiB | 4 | 0 |

The full mode includes compatibility materialization of all graph-valued sequences. Keep atom count and checkpoint count with future comparisons.

The complete matrix, sampling-only CPU attribution, boundary-mode comparison,
and decision not to replace NetworkX yet are recorded in
[PROFILE_2026-08-31.md](PROFILE_2026-08-31.md).

The exact August 29 CO2/polycarbonate catalog string is not present in this checkout. Add it as a non-reference case in [cases.py](cases.py) once recovered rather than substituting a chemically different input.

## Bond-sequence current-tree toggle

`benchmark_bond_sequences` compares the public serial convergence API with
bond-sequence options **omitted** and `retain_bond_sequences=True` on identical fixed inputs. It uses
fresh subprocesses, alternating pair order, one unreported warm-up pair per
case, multiple paired seeds, and a fixed accepted-chain count. Shared metadata
and unit-path statistics remain enabled in both modes. The enabled mode adds
only `retain_bond_sequences=True`; no retention record-count or byte budgets
are configured or required. Disabled and historical-baseline calls omit all
feature kwargs, including when comparing an API that predates the feature.
Normal library calls also default to disabled;
ordinary graph/SMILES/JSON output does not activate topology or motif export.

Topology `occurrences` map record-local instance IDs to stable building-block
type labels such as `R1`. Those IDs are local connectivity references, not new
building-block types. A type-only edge list cannot preserve which instances
connect or how branches are joined.

Repeat `--workload` to select the production-output dimensions:

| Workload | Retained output | Timed file output |
| --- | --- | --- |
| `stats` (default) | None; existing aggregates only | None |
| `retained-graph` | Whole-chain and sequence-unit graphs | None |
| `retained-smiles` | Whole-chain and sequence-unit SMILES | None |
| `production-json` | Whole-chain and sequence-unit SMILES | Benchmark-specific JSON projection, serialization, temporary-file write, flush and fsync |

The JSON workload exercises output cost, but is **not** the Synthesizer catalog
pipeline or a public convergence JSON API. It does not measure clustering,
descriptor calculation, parallel workers, checkpoints, or natural convergence.
All chains are retained, so selected indices are ordered accepted indices and
coverage weights are one. For catalog-specific claims, benchmark the actual
wrapper and frozen production strings separately.

```bash
PYTHONPATH=src python -m benchmarks.benchmark_bond_sequences \
  --case small-linear --case branched --case large-linear \
  --runs 10 --warmups 1 --samples 10 \
  --output benchmarks/bond_sequence_toggle.json
```

Use a new output path for new measurements; the October 17 artifacts are frozen
historical evidence from the earlier harness, not regenerated with this version.
Existing reports, JSON, and archives preserve their original budget settings;
they are historical evidence, not the current budget-free calling policy.
For an expanded workload matrix:

```bash
PYTHONPATH=src python -m benchmarks.benchmark_bond_sequences \
  --case small-linear --case branched --case large-linear \
  --workload stats --workload retained-graph --workload retained-smiles \
  --workload production-json --runs 5 --warmups 1 --samples 2 \
  --output benchmarks/bond_sequence_production.json
```

Add `--baseline-source /explicit/path/to/pre-feature/checkout` only when an
auditable historical source is available. The same harness and frozen input
strings run against that source, with all feature kwargs omitted. Package,
submodule import paths, source-tree hashes, and source revisions are recorded;
identical baseline/current hashes are rejected. A different hash alone does not
prove pre-feature provenance: that remains the caller's responsibility. No
historical source is searched for or reconstructed by the harness.

RSS unit conversion is shared through [_common.py](_common.py), which imports
no G2RINS code and does not affect source routing. It uses Unix `resource` or
the Windows peak-working-set API without adding dependencies. New feature-benchmark source
identities include `benchmark_common_sha256` alongside the existing harness
and case hashes. JSON normalization and digest policies remain local to each
harness; historical reports and archives are not rewritten.

The JSON reports paired historical-disabled overhead separately from current
on/off overhead. Default diagnostic thresholds are 1% disabled versus baseline
and 5% retained versus current off; both ensemble and end-to-end median deltas
must fit. Without a baseline the historical check is explicitly **unverified**.
These statuses are descriptive diagnostics, not universal release acceptance;
use at least five pairs, inspect spread, and avoid concurrent test/benchmark load.
Exit status checks correctness/count parity, not timing thresholds.

The JSON stores every paired run, seed, separate import/setup/ensemble/file-write
and end-to-end timings, RSS, output sizes, exact source identities, and component
parity digests. RSS is sampled while results are live **before** post-timing
topology sizing, canonical chemistry normalization, and hashing. Those later
checks compare chain/sequence chemistry, units, contacts, masses, traces,
retained indices and weights, and path/junction statistics without parse UUIDs.
Statistics-only comparisons intentionally exclude extra retained-only mass rows.

The October 17 results and their limitations are described in
[BOND_SEQUENCE_TOGGLE_2026-10-17.md](BOND_SEQUENCE_TOGGLE_2026-10-17.md). This
is a **current-tree regression reference**, not a comparison to the missing
pre-feature source and not evidence that the original release gates passed.

The expanded [October 19 report](BOND_SEQUENCE_PRODUCTION_2026-10-19.md)
combines the retained-output matrix with freshly rerun statistics rows after
fixing a retained-only mass-bucket parity false positive. It preserves the
initial raw failures rather than rewriting them. All selected pairs agree on
correctness/counts, but the large-linear statistics recheck has **+7.98%**
ensemble overhead and fails the current-tree 5% diagnostic. No blanket
performance acceptance or historical-disabled result is implied.

### Building-block-statistics toggle

The same harness accepts `--feature bond-sequences|building-block-statistics`.
Omitting it preserves the existing `bond-sequences` behavior. The selector is
passed to each fresh worker subprocess and recorded in the report configuration
and per-run records. The existing mode name `retained` means **feature on**;
for building-block statistics it does not imply topology or chain retention.

For `--feature building-block-statistics`, only retained/on API requests receive
`building_block_statistics=True`. Off and historical-baseline API requests omit
all feature kwargs entirely, including `retain_bond_sequences`; they do not
pass explicit `False` placeholders. The `stats` workload uses `metadata=False`,
`retain_chains=False`, and `retain_sequences=False` in **both off/on modes**
(and baseline), avoiding legacy path enumeration. The other three full-output
workloads keep their existing `metadata=True`, chain/sequence retention, and
graph/SMILES/JSON output settings. No new retention budgets are introduced.

Statistics are read from `result.building_block_statistics`; production-json
includes this dictionary only when present, so its serialization/write cost is
timed. The dictionary is deliberately **not a parity input**, since it exists
only in enabled runs. Metadata-free stats runs can have no unit/contact/path
aggregates in either mode; accepted counts, molecular-weight aggregates, and
convergence traces still provide common sampling parity checks. This parity
check does not validate the new counter's mathematical correctness. Topology
size/count metrics remain topology-specific and are zero without bond export.

The [API and result schema](../README.md#building-block-population-statistics)
describe root-incident multistars and occupied-endpoint probability semantics.
Use a new output artifact for new measurements; the frozen reports and JSON
above remain evidence for their original bond-sequence workloads only. This
selector alone makes no performance acceptance claim. The separate
[2026-10-19 statistics report](BUILDING_BLOCK_STATISTICS_2026-10-19.md) records
45 paired runs over three inputs and three workloads: all pairs match sampling
parity, and all nine paired-median ensemble/end-to-end diagnostics satisfy 5%
in that series. No historical-disabled baseline was provided. The same baseline
provenance requirements, paired-run/no-concurrent-load guidance, and
correctness-only exit status continue to apply.

The frozen statistics series measures its original **serial source snapshot**,
before the subsequent parallel-statistics update. The current harness still runs
serial workloads; those numbers do not establish parallel overhead or speedup.
The public API now accepts `building_block_statistics=True, parallel=True`;
see the [parallel usage and semantics](../README.md#building-block-population-statistics).
The separate [parallel validation receipt](BUILDING_BLOCK_PARALLEL_2026-10-19.md)
records process-boundary/parity/recovery regressions, not timing acceptance.

## Behavioral references

[seeded_references.json](seeded_references.json) records node-ID-independent outputs for the compact reference cohort:

- canonical SMILES;
- attributed topology hashes and atom counts;
- molecular weights;
- unit and contact counts;
- normalized sequence SMILES;
- warnings;
- under/overshoot decision traces.

The test suite rebuilds these values from source and compares them exactly. Regenerate only for an intentional, reviewed behavior change:

```bash
conda run -n poly_catalog_env python -m benchmarks.reference
python -m pytest tests/test_seeded_performance_references.py -q
```

Do not update the fixture to make an unexplained optimization regression pass.
