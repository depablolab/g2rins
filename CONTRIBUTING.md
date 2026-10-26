# Contributing to G2RINS

Thanks for your interest in improving G2RINS!

## Development setup

Requires Python ≥ 3.10.

```bash
git clone https://github.com/depablolab/g2rins.git
cd g2rins
python -m pip install -e ".[test]"
```

For the notebook and repository-local plotting examples, install
`.[test,notebook]` instead. DOT rendering additionally requires the Graphviz
`dot` executable on `PATH`; the Python extra does not install system tools.

## Running tests

```bash
python -m pytest tests/
```

CI tests Python 3.10 (minimum supported) and 3.14 (highest currently tested).
Pull requests to `main` and `next` use a faster Linux-only matrix. Pushes to
either branch and the monthly scheduled run use the full Linux/Windows/macOS
matrix. Manual workflow dispatch can test any branch carrying the trigger;
use it to check a release candidate across platforms before integration.
Passing PR checks is the pre-merge regression gate, not proof that the full
cross-platform matrix or feature-specific performance gates have passed.

## Contribution workflow

Use short-lived branches and pull requests, not Git Flow; there is no `develop`
branch. `main` is the main integration branch. `next` collects reviewed changes
as the release-candidate line ahead of the next release. Base work on, and
target the PR at, the appropriate integration line; use `next` for changes
belonging to that release candidate rather than assuming every PR targets `main`.

1. Create a focused branch: `feature/<topic>` or `fix/<topic>`.
2. Make a focused change; avoid bundling unrelated edits.
3. Add or update tests for any behavioral change.
4. Update the [changelog](CHANGELOG.md), relevant docs, and docstrings when user-facing behavior changes.
5. Open a pull request against the selected integration branch and make sure CI passes.

## Documentation layout

- [README.md](README.md) stays focused on installation, version scope, a short
  quickstart, and links. Do not imply unreleased APIs exist in the v1.0.0 tag.
- [docs/index.md](docs/index.md) indexes references and the feature support matrix.
- [docs/topology.md](docs/topology.md) documents sequences, topology formats,
  hydrogen policy, and structural motif analytics.
- [docs/building-block-statistics.md](docs/building-block-statistics.md) owns
  the population-statistics schema and probability semantics.
- [docs/migration.md](docs/migration.md) owns detailed upgrade/safety conditions;
  [CHANGELOG.md](CHANGELOG.md) summarizes user-visible changes and links to it.
- [benchmarks/README.md](benchmarks/README.md) indexes methods and frozen reports.
  Keep historical evidence unchanged; new measurements need new artifacts.

Use G2RINS in prose and `g2rins` for package/code identifiers; preserve published
paper titles/authors and historical artifact bytes. Keep existing README
anchors as short forwarding sections when moving reference material. Maintain
the support matrix against source guards, especially metadata, parallel, and
checkpoint restrictions; do not turn descriptive benchmarks into release claims.

## Grammar and test support

[src/g2rins/data/g2rins.lark](src/g2rins/data/g2rins.lark) is the canonical
grammar loaded by the parser and included in distributions. Update the
[root mirror](g2rins.lark) in the same change; a regression test checks their
byte-for-byte agreement. Preserve parser entry points and parse/serialize
round trips when changing grammar rules.

Put shared recipes, builders, and independent oracles in
[tests/_support.py](tests/_support.py), rather than importing one test module
from another. Do not replace independent expected-value calculations with
the production implementation they are meant to verify.

Keep export/provenance regressions in
[tests/test_ensemble_export.py](tests/test_ensemble_export.py) and convergence
checkpoint/resume regressions in
[tests/test_convergence_checkpoints.py](tests/test_convergence_checkpoints.py).
Cheap convergence payload/copy/order contracts belong in
[tests/test_convergence_assembly.py](tests/test_convergence_assembly.py).
When moving tests, preserve bodies, decorators, parametrized cases, and
disabled-feature sentinels; compare collection before and after the move.

## Linting

The repository uses [Trunk](https://docs.trunk.io) for linting and formatting; it runs automatically in CI. Running it locally before pushing is optional:

```bash
./trunk check
```

## Bugs and feature requests

Please use the issue forms on the [issue tracker](https://github.com/depablolab/g2rins/issues). For bugs, a minimal reproducer and the G2RINS input string (when applicable) make fixes much faster.
