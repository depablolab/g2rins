# Changelog

Notable, user-visible changes to G²RINS. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions correspond to the repository's `v*` git tags.

## [Unreleased]

### Added

- Stereochemistry and isotopes written in a G²RINS string survive into the generative graph, the sampled molecular graphs, the RDKit molecules and the per-unit P-SMILES: tetrahedral chirality (`@`, `@@`, `@TH1`, `@TH2`), square-planar centers (`@SP1`-`@SP3`), double-bond geometry (`/`, `\`) and isotopes (`[13C]`). Allene, trigonal-bipyramidal and octahedral marks are carried in the graphs and reported with a `ChiralityNotApplied` warning at RDKit conversion. Both graphs gained the node attributes `chiral` (0 none, 1 `@TH1`, 2 `@TH2`, 3 `@AL1`, 4 `@AL2`, 5-7 `@SP1`-`@SP3`, 8-27 `@TB1`-`@TB20`, 28-57 `@OH1`-`@OH30`; a bare `@`/`@@` is resolved by the atom's written degree) and `isotope` (mass number, 0 when unspecified), and the edge attributes `cis_trans_mark` (0 none, 1 `/`, 2 `\`, read from u to v) and `chiral_neighbor_rank` (position of the bond among the written neighbors of each end). Isotopic masses enter the molecular-weight accounting. On a bond that joins two units, `cis_trans_mark` is the mark written in the source unit and the new edge attribute `cis_trans_flip` records that the target unit wrote the opposite mark on its side; that unit is then placed with all its marks inverted, so the geometry written in both units is kept: `[<]/C=C\[>]` and `[$]\C=C/[$]` give all-cis chains, `[$]/C=C/[$]` an all-trans chain.
- A chirality order outside its class (`@OH45`) raises `InvalidChiralityOrder`; a class that contradicts the atom's written degree (`@TB1` on a four-coordinate atom) raises `InconsistentChiralityClass`.
- `CONTRIBUTING.md`, `CITATION.cff`, this changelog, issue forms, and a pull request template.
- GitHub Release automation for future `v*` tags: build, verify, attach wheel/sdist, and generate release notes.

### Changed

- Bonds of different order leading to bond connectors that bond to each other (`[<]=CC[>]`) raise `IncompatibleBondTypeBondConnector` instead of warning and silently dropping the pairing. The cis/trans symbols `/` and `\` count as single bonds, and bond symbols are no longer compared along paths through atoms, which removed spurious warnings.
- Pull requests now run a faster Linux-only Python 3.10/3.14 test matrix, while the full Linux/Windows/macOS compatibility matrix runs after merges to `main` and on the monthly schedule. Python 3.14 replaces 3.13 as the highest version tested in CI (RDKit ≥ 2026.3 publishes Python 3.14 wheels).
- Updated official GitHub Actions to current stable majors and tightened workflow permissions.
- Packaging and installation workflows fetch full Git history and tags so `setuptools-scm` can derive versions reliably.
- Simplified the `setuptools-scm` configuration in `pyproject.toml` while preserving `g2rins.__version__`.

### Fixed

- A repeat unit whose two junction bonds carry different cis/trans symbols (`[<]/C=C\[>]`) was rejected as a bond-type mismatch and produced a single repeat unit, and `[$]`-paired alkenes could come out with mixed or inverted geometry; every unit now keeps its written geometry at every junction.
- A bond symbol written on a ring-closure digit (`C1CCCCC=1`, `C/1CCCCC/C=C1`) was dropped; the bond order and the cis/trans mark now reach the graph, and different symbols on the two digits of one ring closure raise `DoubleBondSymbolDefinition`.
- CI explicitly installs the `[test]` extra so pytest is available in test jobs (#1).
- Removed a machine-local `.trunk/plugins/trunk` artifact from version control.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
