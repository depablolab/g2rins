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

### Fixed

- Aromatic selenium and arsenic symbols now map to atomic numbers 34 and 33 instead of negative graph-object IDs, through the aromatic aliases of `atom_name_num`.
- `ParsingError` and `TooManyTokens` preserve their context when pickled or deep-copied. Atoms without a symbol are rejected during construction with `MissingAtomSymbol`, a `ParsingError` subclass.
- Bare wildcard atoms (`*`) retain their symbol when a parsed G2RINS string is serialized or exported, instead of incorrectly becoming `None`.
- Hyperbranched units with several connection sites on one atom no longer emit extra unmapped dummy atoms in ensemble unit pSMILES or sequence fragments. Split-atom placeholders now become the mapped pSMILES stars, and sequence-only connection stubs are reattached directly to their real atom during finalization. Completed chain structures, template/JSON bond ids on placeholder nodes, and bond records such as `R0.2` and `R0.3` are unchanged. Malformed unit pSMILES now raises an internal generation error instead of being exported.
- CI explicitly installs the `[test]` extra so pytest is available in test jobs (#1).
- Removed a machine-local `.trunk/plugins/trunk` artifact from version control.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
