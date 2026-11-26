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
- Bonds of different order leading to connecting bond connectors (e.g. `[<]=CC[>]`) raise `IncompatibleBondTypeBondConnector`, now a `ParsingError`, when the generative graph is built without its bond connectors (the default of `get_generative_graph`); previously only two written symbols were compared and the mismatch was a warning that dropped the pairing. A warning filter that names the class (`category=IncompatibleBondTypeBondConnector`) is no longer valid, because the class is not a warning: remove the filter and catch the exception around `get_generative_graph`.
- The same rule covers a bond symbol written at the boundary of a stochastic object: the unwritten bond of a bond connector is single, and a symbol written next to the object or between two objects has to agree with the bonds of the bond connectors it reaches. `C={[<] [<]CC[>];; [>]}|gauss(400,10)|=C`, which built before, raises `IncompatibleBondTypeBondConnector`; a bond of another order than the one the bond connectors write has to be spelled with the multiply bonded atom outside the stochastic object (`C=C{[<] [<]CC[>];; [>]}|gauss(400,10)|C=C`). The aromatic symbol `:` is an order of its own: it pairs with neither a single bond nor an unwritten one, so it has to be written on both bond connectors.
- A chirality specification whose order lies outside its class (e.g. `[C@OH45]`) raises `InvalidChiralityOrder`.
- A string can hold up to 50 stochastic objects, nested ones included, instead of 10. The per-node `molecular_weight_distribution`, `unit_molar_amounts` and `stochastic_id_tree` vectors of the generative graph have 50 entries instead of 10, so the graph and its JSON output change shape. `EnsembleCreator` takes the length from the graph and still samples graphs built with 10 entries.
- Ensemble sampling is faster, about twice on most of the guide's examples and up to seven times, with unchanged chains: each stochastic object's molecular-weight distribution is deserialized once per `EnsembleCreator` instead of once per chain and once per terminator candidate of every termination-mass estimate.
- Sampling checkpoints reuse the ensemble creator's distribution table and the template graph's per-node molar-amount vectors. Sampling is a further 3% to 20% faster on the guide's examples, and the number of vector entries no longer costs sampling time.

### Fixed

- CI explicitly installs the `[test]` extra so pytest is available in test jobs (#1).
- Removed a machine-local `.trunk/plugins/trunk` artifact from version control.
- The cis/trans bond symbols `/` and `\` leading to connecting bond connectors are single bonds and no longer count as a bond-type mismatch, which dropped every pairing of e.g. `[<]/C=C\[>]` and crashed generation.
- A bond symbol written on a ring-closure digit (e.g. `C=1CCCCC1`) is applied to the ring bond instead of being dropped; the two digits of a ring closure writing contradicting symbols raise `DoubleBondSymbolDefinition` (the directional symbols `/` and `\` are read along the string, so on the two digits they agree when opposite, e.g. `C/1CCCCC\1`; a directional symbol on one digit and `-` on the other, e.g. `C/1CCCCC-1`, raise as well).
- Graph construction enumerates connector paths on the subgraph containing the two endpoint atoms and the bond connectors. Bonds inside a unit are excluded from connector-order checks. This avoids the exponential path search for sequential stochastic objects.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
