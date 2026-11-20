# Changelog

Notable, user-visible changes to G²RINS. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions correspond to the repository's `v*` git tags.

## [Unreleased]

### Added

- `CONTRIBUTING.md`, `CITATION.cff`, this changelog, issue forms, and a pull request template.
- GitHub Release automation for future `v*` tags: build, verify, attach wheel/sdist, and generate release notes.
- `EnsembleCreator.get_distributions()` returns the molecular-weight distributions the sampler draws from, keyed by stochastic object id as in the ensemble output, with count laws bound to their repeat-unit and end-group masses. The guide plots its reference distributions from them.

### Changed

- `flory_schulz(a)` samples the number fraction of the most probable distribution, `a (1-a)^(N-1)` over repeat-unit counts `N` (number-average count `1/a`, dispersity approaching `2 - a`). It previously sampled the weight fraction and read the count as a molecular weight in g/mol.
- `prob_mw` of a count law is a probability density per g/mol (the nearest count's probability over one repeat-unit mass), comparable with the continuous laws.
- New `CountLaw` base class for `poisson` and `flory_schulz` (`bind_repeat_unit_mass`, `mean_count`), and `mean_mw`/`support_mw` on every molecular-weight distribution.
- Pull requests now run a faster Linux-only Python 3.10/3.14 test matrix, while the full Linux/Windows/macOS compatibility matrix runs after merges to `main` and on the monthly schedule. Python 3.14 replaces 3.13 as the highest version tested in CI (RDKit ≥ 2026.3 publishes Python 3.14 wheels).
- Updated official GitHub Actions to current stable majors and tightened workflow permissions.
- Packaging and installation workflows fetch full Git history and tags so `setuptools-scm` can derive versions reliably.
- Simplified the `setuptools-scm` configuration in `pyproject.toml` while preserving `g2rins.__version__`.

### Fixed

- `poisson(Mn)` and `flory_schulz(a)` are laws over the number of repeat units but were sampled as if that count were a molecular weight in g/mol, which left every `poisson` ensemble essentially monodisperse. The ensemble creator now estimates from the generative graph each stochastic object's mean repeat-unit mass (how often each repeat unit appears follows from the molar amounts and bond-connector weights) and its end-group mass (initiator and the caps of the ports it exposes, nested objects included), binds both to the count law and converts the drawn counts into target masses. With several ports growing at once, the repeat-unit frequencies also follow the sampler's draw of the port that grows next (by bond-connector weight; a port with weight 0 never opens, so it neither grows nor gets an end group). `poisson(Mn)` keeps `Mn` as the number-average molecular weight of the whole object and now yields the dispersity `1 + M_unit (Mn - M_end)/Mn^2`, which is `1 + M_unit/Mn` for light end groups. A stochastic object none of whose repeat units can be placed keeps its counts in g/mol and warns (`CountLawUnresolved`). A `poisson(Mn)` with `Mn` at or below that end-group mass is read with zero repeat units and warns (`CountLawBelowEndGroups`).
- CI explicitly installs the `[test]` extra so pytest is available in test jobs (#1).
- Removed a machine-local `.trunk/plugins/trunk` artifact from version control.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
