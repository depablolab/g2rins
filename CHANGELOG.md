# Changelog

Notable, user-visible changes to G²RINS. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions correspond to the repository's `v*` git tags.

## [Unreleased]

### Added

- `CONTRIBUTING.md`, `CITATION.cff`, this changelog, issue forms, and a pull request template.
- GitHub Release automation for future `v*` tags: build, verify, attach wheel/sdist, and generate release notes.
- Unit records in the ensemble output carry the unit's static subgraph of the generative graph (`subgraph`: original node ids, static edges only, `unit_id` stamped on the copy's nodes; node-link encoded in JSON files).
- Bond records carry the generative-graph node ids of the two connection atoms (`nodes`, positionally aligned with `labels`; labels survive a fresh parse, node ids are only valid for the graph they came from).
- Ensemble JSON files follow the requested `output_format` for their stored chains — SMILES strings or node-link graph dicts — and record the choice in `format.chain_format`. Sequences are written as SMILES regardless.

### Changed

- Pull requests now run a faster Linux-only Python 3.10/3.14 test matrix, while the full Linux/Windows/macOS compatibility matrix runs after merges to `main` and on the monthly schedule. Python 3.14 replaces 3.13 as the highest version tested in CI (RDKit ≥ 2026.3 publishes Python 3.14 wheels).
- Updated official GitHub Actions to current stable majors and tightened workflow permissions.
- Packaging and installation workflows fetch full Git history and tags so `setuptools-scm` can derive versions reliably.
- Simplified the `setuptools-scm` configuration in `pyproject.toml` while preserving `g2rins.__version__`.
- Exported graph and ensemble JSON is format version 2: the unit record key `frequency` is renamed to `count`, the bond record key `between` is renamed to `labels`, and unit records list `psmiles`, `g2rins`, `subgraph`, `count` in that order.

### Removed

- The `mol` output format of `create_ensemble`. Request `mol_graph` and convert chains with `g2rins.mol_graph_to_rdkit_mol`, or parse the SMILES output.

### Fixed

- CI explicitly installs the `[test]` extra so pytest is available in test jobs (#1).
- Removed a machine-local `.trunk/plugins/trunk` artifact from version control.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
