# Changelog

Notable, user-visible changes to G²RINS. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions correspond to the repository's `v*` git tags.

## [Unreleased]

### Added

- `CONTRIBUTING.md`, `CITATION.cff`, this changelog, issue forms, and a pull request template.
- GitHub Release automation for future `v*` tags: build, verify, attach wheel/sdist, and generate release notes.
- Warnings `IsomerSymbolsNotInGraph` (from `get_generative_graph`) and `IsomerSymbolsIgnoredInGeneration` (from `get_ensemble_creator`) when a string writes isotope, chirality or cis/trans symbols: they parse, but the generative graph does not encode them and generation does not apply them yet (available soon).

### Changed

- Pull requests now run a faster Linux-only Python 3.10/3.14 test matrix, while the full Linux/Windows/macOS compatibility matrix runs after merges to `main` and on the monthly schedule. Python 3.14 replaces 3.13 as the highest version tested in CI (RDKit ≥ 2026.3 publishes Python 3.14 wheels).
- Updated official GitHub Actions to current stable majors and tightened workflow permissions.
- Packaging and installation workflows fetch full Git history and tags so `setuptools-scm` can derive versions reliably.
- Simplified the `setuptools-scm` configuration in `pyproject.toml` while preserving `g2rins.__version__`.
- Bonds of different order leading to connecting bond connectors (e.g. `[<]=CC[>]`) raise `IncompatibleBondTypeBondConnector`, now a `ParsingError`, when the generative graph is requested; previously only two written symbols were compared and the mismatch was a warning that dropped the pairing.
- A chirality specification whose order lies outside its class (e.g. `[C@OH45]`) raises `InvalidChiralityOrder`.

### Fixed

- CI explicitly installs the `[test]` extra so pytest is available in test jobs (#1).
- Removed a machine-local `.trunk/plugins/trunk` artifact from version control.
- The cis/trans bond symbols `/` and `\` leading to connecting bond connectors are single bonds and no longer count as a bond-type mismatch, which dropped every pairing of e.g. `[<]/C=C\[>]` and crashed generation; paths through atoms no longer emit spurious `IncompatibleBondTypeBondConnector` warnings.
- A bond symbol written on a ring-closure digit (e.g. `C=1CCCCC1`) is applied to the ring bond instead of being dropped; the two digits of a ring closure writing different symbols raise `DoubleBondSymbolDefinition`.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
