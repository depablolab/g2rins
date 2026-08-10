# Changelog

Notable, user-visible changes to G²RINS. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions correspond to the repository's `v*` git tags.

## [Unreleased]

### Added

- `CONTRIBUTING.md`, `CITATION.cff`, this changelog, issue forms, and a pull request template.
- GitHub Release automation for future `v*` tags: build, verify, attach wheel/sdist, and generate release notes.

### Changed

- CI test matrix upper bound raised from Python 3.13 to 3.14 (RDKit ≥ 2026.3 publishes Python 3.14 wheels).
- All workflows updated to current GitHub Action majors, with least-privilege permissions and full-history checkouts so `setuptools-scm` derives the real tag version.
- Simplified the `setuptools-scm` configuration in `pyproject.toml` to the documented pattern (no behavioral change to `g2rins.__version__`).

### Fixed

- CI installs the `[test]` extra so pytest is available in the test jobs (#1).
- Distribution builds previously ran on a shallow, tag-less checkout, producing a fallback development version instead of the tag-derived one.
- Removed a machine-local `.trunk/plugins/trunk` symlink from version control.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/releases/tag/v1.0.0
