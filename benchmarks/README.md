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
