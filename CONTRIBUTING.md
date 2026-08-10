# Contributing to G²RINS

Thanks for your interest in improving G²RINS!

## Development setup

Requires Python ≥ 3.10.

```bash
git clone https://github.com/depablolab/g2rins.git
cd g2rins
python -m pip install -e ".[test]"
```

## Running tests

```bash
python -m pytest tests/
```

CI runs the same test suite on Linux, Windows, and macOS using Python 3.10 (the minimum supported version) and Python 3.14 (the highest version currently tested in CI). Pull requests use a faster Linux-only matrix for these two Python versions, while the full cross-platform matrix runs after changes are merged to `main` and on the monthly scheduled CI run. Passing the pull-request checks provides the pre-merge regression gate; the post-merge matrix provides the final cross-platform compatibility check.

## Contribution workflow

We use plain GitHub flow — short-lived branches off `main`, merged back via pull request. There is no `develop` branch and no Git Flow.

1. Create a focused branch: `feature/<topic>` or `fix/<topic>`.
2. Make a focused change; avoid bundling unrelated edits.
3. Add or update tests for any behavioral change.
4. Update documentation (README, docstrings, `CHANGELOG.md`) when user-facing behavior changes.
5. Open a pull request against `main` and make sure CI passes.

## Linting

The repository uses [Trunk](https://docs.trunk.io) for linting and formatting; it runs automatically in CI. Running it locally before pushing is optional:

```bash
./trunk check
```

## Bugs and feature requests

Please use the issue forms on the [issue tracker](https://github.com/depablolab/g2rins/issues). For bugs, a minimal reproducer and the G²RINS input string (when applicable) make fixes much faster.
