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
- When a nested stochastic object finishes, its continuation competes in one weighted draw with all of the owning level's growth options — remaining entry sites and other frontier bonds alike — instead of firing unconditionally. A continuation the owner never draws before reaching its target is capped by the owner's declared end groups, or retired unfired when its channel declares none; architectures whose owning level has several simultaneous growth options therefore sample different sequences than before. At each step of the owning level the site to grow is drawn among all of its open sites in proportion to their bond-descriptor weights, normalized over the sites open at that moment; a promoted continuation enters that draw with the weight of the site it sits on, and molar amounts act only on the choice of the incoming unit. An owner with a single growth option still draws it deterministically.
- Known limitation: plain SMILES written directly after a nested stochastic object inside a unit (an exit through the object's terminal bond connector, encoded as a transition) is delivered only when the owner draws it before parking; it previously fired as soon as the nested object finished unless the owner had already parked. Firing such exits unconditionally is planned as a separate change.

### Fixed

- CI explicitly installs the `[test]` extra so pytest is available in test jobs (#1).
- Removed a machine-local `.trunk/plugins/trunk` artifact from version control.
- Nested stochastic objects used as repeat units could not grow their own instances after a transition fired; chains fell short of the outer target and were discarded.
- Open sites handed to another level's custody lost their termination modes, silently dropping declared end groups from finished molecules.
- The transition sweep no longer transfers mode-less copies of bonds it does not convert; such copies could mask a bucket's real growth bonds and end the chain before its terminators fired.
- Open sites converted after a root-level continuation could be filed under the already-terminated source instance, silently dropping the arms and end groups they carried. When no live instance of the fired level exists at all, the sampler now raises instead of filing them under another level's bucket.
- Average termination-mass estimates now price end groups held in terminated descendants' custody, matching what termination actually attaches; heavy declared end groups no longer systematically overshoot the target mass.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
