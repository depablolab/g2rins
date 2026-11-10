# Changelog

Notable, user-visible changes to G²RINS. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions correspond to the repository's `v*` git tags.

## [Unreleased]

### Added

- `CONTRIBUTING.md`, `CITATION.cff`, this changelog, issue forms, and a pull request template.
- GitHub Release automation for future `v*` tags: build, verify, attach wheel/sdist, and generate release notes.

### Changed

- Pull requests now run a faster Linux-only Python 3.10/3.14 test matrix, while the full Linux/Windows/macOS compatibility matrix runs after merges to `main` and on the monthly schedule. Python 3.14 replaces 3.13 as the highest version tested in CI (RDKit ≥ 2026.3 publishes Python 3.14 wheels).
- Updated official GitHub Actions to current stable majors and tightened workflow permissions.
- Packaging and installation workflows fetch full Git history and tags so `setuptools-scm` can derive versions reliably.
- Simplified the `setuptools-scm` configuration in `pyproject.toml` while preserving `g2rins.__version__`.
- A string can hold up to 50 stochastic objects, nested ones included, instead of 10. The per-node `molecular_weight_distribution`, `unit_molar_amounts` and `stochastic_id_tree` vectors of the generative graph have 50 entries instead of 10, so the graph and its JSON output change shape. `EnsembleCreator` takes the length from the graph and still samples graphs built with 10 entries.
- Ensemble sampling is faster, about twice on most of the guide's examples and up to seven times, with unchanged chains: each stochastic object's molecular-weight distribution is deserialized once per `EnsembleCreator` instead of once per chain and once per terminator candidate of every termination-mass estimate.
- Checkpoints taken while sampling share the distribution table and the per-node molar-amount vectors with the generative graph instead of copying them. Sampling is a further 3% to 20% faster on the guide's examples, and the number of vector entries no longer costs sampling time.

### Fixed

- CI explicitly installs the `[test]` extra so pytest is available in test jobs (#1).
- Removed a machine-local `.trunk/plugins/trunk` artifact from version control.
- Graph construction no longer enumerates every simple path of the atom graph to skip over bond connectors, only the paths that run through bond connectors. The graphs are unchanged; construction time no longer doubles with each further sequential stochastic object (twelve objects: 15 s before, 0.1 s now).

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
