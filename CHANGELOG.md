# Changelog

Notable, user-visible changes to G2RINS. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); released versions correspond to the repository's `v*` git tags.

## [Unreleased]

Development-tree changes, not a release announcement or a v1.0.0 API guarantee.
See [migration and safety details](docs/migration.md) and the
[feature support matrix](docs/index.md#feature-support-matrix).

### Breaking changes

- **Graph JSON v2 requires explicit zero-node identity.** Export and ensemble construction reject `atomic_num=0` nodes lacking boolean `is_connector_placeholder`: `True` means internal placeholder, `False` user wildcard. Graphs without zero-number nodes may omit it. Changing `format.version` alone is insufficient. Prefer rebuilding from source strings. Use `mark_legacy_connector_placeholders` only with known provenance: pendant user wildcards have the same one-static-neighbor shape as placeholders. Its count-only warning does not verify identities; audit changed flags. Without that provenance, recover identities from the producer first. See [full migration conditions](docs/migration.md#generative-graph-json-format-v2).
- **Ensemble output v2** shares `format.version=2`: unit `frequency` becomes `count`, bond `between` becomes `labels`, units add NetworkX `subgraph`, and bonds add aligned connection-atom `nodes`. Unit records are not directly JSON-serializable; template attributes must be deep-copyable. Default `mol_graph` JSON now stores node-link chains, not SMILES; request `smiles` for compact strings. Equality is structural. See [output migration](docs/migration.md#ensemble-output-format-v2).
- **User wildcards cannot generate ensembles.** `*`/`[*]` still parse/export but raise `UnsupportedWildcardGeneration` at creator construction. Use explicit atoms/end groups. Atomic-number/placeholder validation and unsupported multi-neighbor descriptors now fail early; see [validation and diagnostics](docs/migration.md#wildcards-and-construction-validation).

### Building-block population statistics

- Independent, keyword-only `building_block_statistics=True` supports fixed-size/convergence and serial/parallel runs, including companion single/catalog APIs and CLIs. Fixed-size needs `ensemble_info=True` or `json_file`; convergence works with `metadata=False` and no retained payload. Checkpoint/resume, including callbacks, is unsupported.
- Detached `building-block-statistics/v1` results count all accepted samples: exact type/motif occurrences, once-per-chain presence, and directed site endpoints. Root-incident covalent multistars distinguish parallel arms, repeated same-type neighbors, self-links, and isolates; association is excluded. Probabilities condition on occupied source-site endpoints, not eligible sites or input weights. Workers supply raw counters to the ordered parent, never averaged probabilities or representative weights. See [complete schema and semantics](docs/building-block-statistics.md).
- Counting reuses occurrence connectivity without solely-for-counts graphs/SMILES/canonicalization. It neither enables topology/motif retention nor replaces instance graphs or implements exact whole-topology counts. The [serial measurements](benchmarks/BUILDING_BLOCK_STATISTICS_2026-10-19.md) are for the earlier source snapshot; the [parallel receipt](benchmarks/BUILDING_BLOCK_PARALLEL_2026-10-19.md) is validation, not speedup evidence. See [benchmark selector/methods](benchmarks/README.md#building-block-statistics-toggle).

### Added

- Optional `notebook` extra for guide plotting/display dependencies. Repository-local guide helpers use public `sequence_unit_ids` for unit identities; DOT rendering still requires the Graphviz executable separately.
- Experimental serial convergence `retain_bond_sequences=True`, requiring `metadata=True`, with compact instance topology, branches/paths renderers, aligned retained indices/weights, and all/cover/reservoir/topology-only selection. Mandatory retention budgets were removed; local occurrence references preserve stable type labels and branching. Companion APIs/CLIs provide chain-index joins and independent topology compression. See [topology contracts](docs/topology.md).
- Independent experimental `retain_motif_membership=True` in serial fixed-size/convergence runs: retained-chain membership and complete-chain masses plus exact all-accepted path-motif occurrence/presence totals, also in fixed-size JSON-only exports. These are not chemical motifs or motif masses. Parallel/checkpoint modes remain unsupported; see [requirements](docs/topology.md#independent-structural-motif-analytics).
- **Topology limitations remain:** no fixed-size topology export, exact all-accepted topology counting (`count_bond_sequences=True`), parallel execution, checkpoint/resume, or chemical/stereo-valid motif rendering. Unsupported enabled modes raise, not silently fall back. Coverage weights are not exact topology counts.
- Automatic repeat-unit initiation when no initiator is declared anywhere, with flow-aware source selection, connector/nesting-weighted probabilities, and one `RepeatUnitInitiation` warning at creator construction. Zero-weight candidates still fail. **`sample_mol_graph` and `NoValidGenerationSource` still accept `use_repeat_units_as_source`; explicit repeat mode remains separate and no positional arguments shifted.** See [source selection](docs/migration.md#automatic-initiation-and-explicit-source-modes).
- Positional terminal connector-list pairing with equal-length/single-terminal checks and `MismatchedBondConnectorLists`; explicit legacy placeholder migration; `UnitLabels.unit_nodes`; static unit subgraphs and aligned bond node IDs; output-format-aware, detached, normalized, hash-seed-independent JSON. See [migration details](docs/migration.md).
- Termination diagnostics: `ShadowedTerminationDeclaration`, `InheritedTermination`, `ForeignControlledTermination`, and `MissingTermination`; canonical local capping stays silent.
- Contribution/citation guidance, issue forms, PR template, and future-`v*` release automation to build/verify/attach wheel and sdist and generate notes. Development documentation now separates the short README, reference/schema pages, and migration guide.

#### Benchmark evidence, not release acceptance

Paired topology toggles and expanded graph/SMILES/JSON workloads are documented
in [benchmark methods](benchmarks/README.md#bond-sequence-current-tree-toggle),
the [October 17 report](benchmarks/BOND_SEQUENCE_TOGGLE_2026-10-17.md), and the
[October 19 report](benchmarks/BOND_SEQUENCE_PRODUCTION_2026-10-19.md).
The latter preserves the stats-only parity false positives and corrected
rechecks: large-linear stats overhead **+7.98% fails the 5% diagnostic**.
The original pre-feature gates and historical-disabled regression remain
**unverified** without an auditable baseline. Current-tree toggles are not
universal runtime guarantees; frozen reports retain their original scope/settings.

### Changed

- Readability cleanup extracts private retry/warning, accepted-record assembly, convergence payload/checkpoint packaging, metadata-equality, JSON-projection, and SMILES recovery helpers, with immutable internal conversion settings. Export/checkpoint/assembly regressions have focused modules; benchmark RSS measurement is shared. Sampling parameters, RNG/retry policies, ensemble JSON layouts, and checkpoint class locations are preserved. See the [cleanup audit](docs/cleanup-review.md) for the remaining orchestration boundary.
- Runtime review adds conservative, order-preserving unit-prototype copying with deep-copy fallback for extension metadata; paired local measurements and scope are in the [runtime report](benchmarks/GRAPH_COPY_2026-10-26.md). Grammar resource loading supports Python 3.10's single-child Traversable interface, and benchmark peak-RSS measurement works without Unix `resource` on Windows. The Python minimum and core dependencies are unchanged.
- Initiation warnings now describe automatic fallback; objects without an initiator get only `NoExplicitInitiation`. Nested completion and remaining entry sites compete in one weighted draw. Nearest termination declarations win at graph construction, unreachable cap edges are removed, and the descriptor-owning object's target includes cap mass. Nested initiators inherit enclosing terminators. See [growth/termination semantics](docs/migration.md#connector-lists-nested-growth-and-termination).
- Private graph copies normalize placeholder flags and remove stale derived labels; extra graph information drops `0: "None"`. Structural ensemble equality covers graphs, supported NumPy/container types, and safe unequal outcomes for cyclic/ambiguous metadata; see [precise output contracts](docs/migration.md#ensemble-output-format-v2).
- PR CI uses Linux Python 3.10/3.14 (3.14 replaces 3.13; RDKit ≥ 2026.3 provides wheels). The full Linux/Windows/macOS matrix runs on integration pushes/monthly schedules and manual dispatch. CI, Trunk, and release-build workflows include `next`, the release-candidate line. Official Actions majors/permissions were updated; packaging/install workflows fetch full history/tags, and simplified `setuptools-scm` configuration preserves `g2rins.__version__`. See [contribution workflow](CONTRIBUTING.md).

### Removed

- `create_ensemble` output format `mol`. Use `mol_graph` plus `g2rins.mol_graph_to_rdkit_mol`, or parse SMILES. Sequence fragment conversion needs `kekulize=False`; see [conversion guidance](docs/migration.md#chain-formats-and-json-normalization). Source-mode arguments were **not** removed.

### Fixed

- Default grammar loading supports non-filesystem package resources. Guide setup again resets the drawing RNG on reruns; example helpers reject nonpositive chunk requests and handle empty plotting/graft/nitrogen data safely.
- Connector `generable` attributes now return booleans, consistently with atoms and the base-class contract. Access `connector.generable` as a property, not as a method.
- Ordinary terminal H caps normalize to `hydrogen_termination="implicit/v1"` for topology/path/branch and motif output without changing sampling, SMILES, masses, legacy sequences, or raw unit/contact counts. Isotopic/charged H, H initiators/repeats, isolated H/H-H, and other explicit exceptions remain. Historical records are not rewritten and offline renderers operate as-is; see [hydrogen policy](docs/topology.md#hydrogen-termination-policy).
- Fixed-size motif analytics now count before conversion, preserving membership, totals, mass alignment, empty/single-unit cases, and JSON-only output. Disabled feature work/buffers remain off; see [consumer migration](docs/migration.md#topology-and-motif-consumers).
- Nested entry lists wire every entry; repeat-source nested growth starts immediately with its own instance/draw; nested repeat instances grow after transitions; custody transfer preserves termination modes/end groups. See [initiation](docs/migration.md#automatic-initiation-and-explicit-source-modes) and [nested growth](docs/migration.md#connector-lists-nested-growth-and-termination).
- Split-site unit/sequence fragments avoid duplicate/inactive stars, preserve active junction order, validate pSMILES via `InvalidUnitPSmiles`, and use neutral non-aromatic connection stubs. Completed chains/template bonds are unchanged; isolated fragment masses need not sum to chain mass. See [fragment contracts](docs/migration.md#unit-psmiles-and-sequence-fragments).
- Early atomic-number/placeholder and multi-neighbor descriptor diagnostics; safe NumPy JSON normalization, collision/non-finite rejection, detached payloads; verified optional `unit_node_ids` provenance; aromatic Se/As mapping; wildcard serialization; pickle/deepcopy context for `ParsingError`/`TooManyTokens`; `MissingAtomSymbol` at construction. See [migration details and remaining sampling limitations](docs/migration.md).
- CI installs `[test]` so pytest is available (#1); removed a machine-local Trunk plugin artifact from version control.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
