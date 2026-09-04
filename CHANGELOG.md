# Changelog

Notable, user-visible changes to G²RINS. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions correspond to the repository's `v*` git tags.

## [Unreleased]

### Added

- `CONTRIBUTING.md`, `CITATION.cff`, this changelog, issue forms, and a pull request template.
- GitHub Release automation for future `v*` tags: build, verify, attach wheel/sdist, and generate release notes.
- Conditional connectivity: bond connector symbols accept a per-symbol group suffix (`[<[<]1]` ladder, `[>,>1[]1]` exclusion, `[>[all]1]` all) that parses, round-trips and is validated (errors `MixedRulesInGroup`, `MixedOuterSymbolsInGroup`, `RepeatedGroupInSite`, `IncompatibleGroupPair`, `GroupPartnerNotPlain`, `GroupRuleOnTerminalBondConnector`, `GroupRuleOnNestedObjectBondConnector`; warnings `SingleMemberGroup`, `IndistinguishableSymbolsInSite`); `GroupRule`, `GroupSuffix` and `RuleKeyword` are exported. Every generative-graph edge now carries `source_group`, `target_group` (-1 = none) and `source_rule`, `target_rule` (0 NONE, 1 LADDER, 2 EXCLUSION, 3 ALL), one edge per distinct group annotation of a compatible bond connector pair. A bond across nesting levels carries the rule of the unit it leaves or of the unit it reaches (terminal bond connectors and the bond connectors that attach a nested stochastic object stay plain); a bond ruled at both ends raises `GroupRulesOnBothPathEnds` from `get_generative_graph`. Generation of strings that use group rules is not implemented yet and raises `NotImplementedError` from `EnsembleCreator`.

### Changed

- Nested bond connector notation such as `[<1[<1]1]` is parsed as a group suffix instead of raising `UnsupportedBigSMILES("ladder_bond_connector")`.
- Pull requests now run a faster Linux-only Python 3.10/3.14 test matrix, while the full Linux/Windows/macOS compatibility matrix runs after merges to `main` and on the monthly schedule. Python 3.14 replaces 3.13 as the highest version tested in CI (RDKit ≥ 2026.3 publishes Python 3.14 wheels).
- Updated official GitHub Actions to current stable majors and tightened workflow permissions.
- Packaging and installation workflows fetch full Git history and tags so `setuptools-scm` can derive versions reliably.
- Simplified the `setuptools-scm` configuration in `pyproject.toml` while preserving `g2rins.__version__`.

### Fixed

- CI explicitly installs the `[test]` extra so pytest is available in test jobs (#1).
- Removed a machine-local `.trunk/plugins/trunk` artifact from version control.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
