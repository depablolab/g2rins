# Changelog

Notable, user-visible changes to G²RINS. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions correspond to the repository's `v*` git tags.

## [Unreleased]

### Added

- `CONTRIBUTING.md`, `CITATION.cff`, this changelog, issue forms, and a pull request template.
- GitHub Release automation for future `v*` tags: build, verify, attach wheel/sdist, and generate release notes.
- Warnings that report how each open site of a generative graph will be capped: `ShadowedTerminationDeclaration`, `InheritedTermination`, `ForeignControlledTermination`, and `MissingTermination`. The canonical configuration — a site capped in the step that grows it — stays silent.

### Changed

- Pull requests now run a faster Linux-only Python 3.10/3.14 test matrix, while the full Linux/Windows/macOS compatibility matrix runs after merges to `main` and on the monthly schedule. Python 3.14 replaces 3.13 as the highest version tested in CI (RDKit ≥ 2026.3 publishes Python 3.14 wheels).
- Updated official GitHub Actions to current stable majors and tightened workflow permissions.
- Packaging and installation workflows fetch full Git history and tags so `setuptools-scm` can derive versions reliably.
- Simplified the `setuptools-scm` configuration in `pyproject.toml` while preserving `g2rins.__version__`.
- When a nested stochastic object finishes, its continuation and the level's remaining entry sites compete in one weighted draw at the owning level, instead of the continuation firing unconditionally.
- Which terminator caps an open site is resolved at graph construction: the declaration nearest the site wins, and termination edges that can never fire are removed from the generative graph. The stochastic object that owns the site's bond descriptor fires the cap, and the cap's mass counts toward that object's molecular weight target.
- A nested stochastic object used as an initiator now inherits the enclosing object's terminators for its exposed chain ends, as one used as a repeat unit already did.

### Fixed

- CI explicitly installs the `[test]` extra so pytest is available in test jobs (#1).
- Removed a machine-local `.trunk/plugins/trunk` artifact from version control.
- Nested stochastic objects used as repeat units could not grow their own instances after a transition fired; chains fell short of the outer target and were discarded.
- Open sites handed to another level's custody lost their termination modes, silently dropping declared end groups from finished molecules.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
