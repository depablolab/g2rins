# Changelog

Notable, user-visible changes to G2RINS. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); released versions correspond to the repository's `v*` git tags.

## [Unreleased]

Development-tree changes, not a release announcement or a v1.0.0 API guarantee.
See [migration and safety details](docs/migration.md) and the
[feature support matrix](docs/index.md#feature-support-matrix).

### Breaking changes

- **2026-10-30 — convergence checkpoints fail closed on model identity.**
	New full/statistics checkpoints include `g2rins-model-fingerprint/v1` for
	chemistry, distribution/weight/hierarchy data and operational graph order,
	plus a separate origin-ID digest. All are checked before RNG restoration,
	workers or sampling. Persist/restore the original creator; fresh source parsing
	does not preserve origin-indexed metadata. Checkpoints without a supported
	fingerprint cannot resume; restart instead. Non-checkpoint sampling is unchanged.
	See [identity and migration contract](docs/checkpoint-model-fingerprint.md).
- **2026-10-30 — direct mass MWD replaces experimental NB.** `mwd(Mw,Mn)`
	now draws Gamma/Schulz–Zimm number-weighted target masses directly, without
	recipe calibration or conversion to a polymerization count. Equal moments
	give a point mass. The standard graph sampler is used; target moments need
	not equal actual final molecular moments. The NB compilers/count assembly,
	`ShiftedNegativeBinomial`, long DSL alias and `nb_realization` selector have
	been removed. Old NB v3/14-wide graph artifacts and checkpoints fail closed;
	reviewed source must be rebuilt and sampling restarted. Gamma uses v4/16-wide
	vectors with the retired NB positions reserved. Unversioned `mwd` source text
	cannot reveal its old meaning; review it explicitly before reuse. Existing
	ordinary distributions and nested mass-stop semantics remain unchanged.
	See [MWD contract and migration](docs/mwd.md). The NB entries and receipts
	below record the superseded development implementation, not current support.
- **Graph JSON v2 requires explicit zero-node identity.** Export and ensemble construction reject `atomic_num=0` nodes lacking boolean `is_connector_placeholder`: `True` means internal placeholder, `False` user wildcard. Graphs without zero-number nodes may omit it. Changing `format.version` alone is insufficient. Prefer rebuilding from source strings. Use `mark_legacy_connector_placeholders` only with known provenance: pendant user wildcards have the same one-static-neighbor shape as placeholders. Its count-only warning does not verify identities; audit changed flags. Without that provenance, recover identities from the producer first. See [full migration conditions](docs/migration.md#generative-graph-json-format-v2).
- **Ensemble output v2** shares `format.version=2`: unit `frequency` becomes `count`, bond `between` becomes `labels`, units add NetworkX `subgraph`, and bonds add aligned connection-atom `nodes`. Unit records are not directly JSON-serializable; template attributes must be deep-copyable. Default `mol_graph` JSON now stores node-link chains, not SMILES; request `smiles` for compact strings. Equality is structural. See [output migration](docs/migration.md#ensemble-output-format-v2).
- **User wildcards cannot generate ensembles.** `*`/`[*]` still parse/export but raise `UnsupportedWildcardGeneration` at creator construction. Use explicit atoms/end groups. Atomic-number/placeholder validation and unsupported multi-neighbor descriptors now fail early; see [validation and diagnostics](docs/migration.md#wildcards-and-construction-validation).

### Building-block population statistics

- Independent, keyword-only `building_block_statistics=True` supports fixed-size/convergence and serial/parallel runs, including companion single/catalog APIs and CLIs. Fixed-size needs `ensemble_info=True` or `json_file`; convergence works with `metadata=False` and no retained payload. Checkpoint/resume, including callbacks, is unsupported.
- Detached `building-block-statistics/v1` results count all accepted samples: exact type/motif occurrences, once-per-chain presence, and directed site endpoints. Root-incident covalent multistars distinguish parallel arms, repeated same-type neighbors, self-links, and isolates; association is excluded. Probabilities condition on occupied source-site endpoints, not eligible sites or input weights. Workers supply raw counters to the ordered parent, never averaged probabilities or representative weights. See [complete schema and semantics](docs/building-block-statistics.md).
- Counting reuses occurrence connectivity without solely-for-counts graphs/SMILES/canonicalization. It neither enables topology/motif retention nor replaces instance graphs or implements exact whole-topology counts. The [serial measurements](benchmarks/BUILDING_BLOCK_STATISTICS_2026-10-19.md) are for the earlier source snapshot; the [parallel receipt](benchmarks/BUILDING_BLOCK_PARALLEL_2026-10-19.md) is validation, not speedup evidence. See [benchmark selector/methods](benchmarks/README.md#building-block-statistics-toggle).

### Added

- Experimental `mwd(Mw,Mn)` with recipe-bound affine calibration and explicit
	`nb_realization="physical_compound"` count-led assembly. Physical C1/C2 linear
	and B1/B2 closed-moment branching include composition/random-cap variance;
	one whole-molecule `N=1+K` means one seed plus `K` real additions, not a count
	per arm, followed by actual caps. Default `target_mass` stays affine/two-port
	linear; the C2 32-state limit is not a branching-frontier bound. B1/B2
	first-two-moment closure is not a full i.i.d. molecular-mass law.
- **2026-10-29 current extension — B3:** if the exact B1/B2 closure theorem
	does not apply, `correlated_frontier` evaluates deterministic fixed-`N`
	frontier moments and finite computed count-mixture levels with analytic
	polynomial Chernoff tail remainder bounds, **not formal floating-point
	interval arithmetic**. Invalid chemistry/numerical errors do not select this
	fallback. Moment-solver limits are 16 classes, 20,000 states, 5,000,000
	transitions and 4096 computed count levels; sampled NB count support remains
	**untruncated**. Inverse fitting is bounded/local, not global uniqueness.
	With unbounded random child rewards, conditional RMS/Minkowski envelopes
	bound first/second mass moments, **not mass support**. Chernoff count-tail
	guards are unchanged; numerical certification is float-inflated, not intervals.
- **2026-10-29 expanded extension — generalized independent nested macros:**
	pure child-as-entire-parent-repeat, anchored backbone insertions, heterogeneous
	parent/child repeats, sibling and recursive backbone/pendant combinations,
	and B1/B2/B3 roots with per-selected-unit rewards are implemented. Private
	**real representative chemistry scaffolds**, never mean-mass atoms, are fully
	substituted by independently count-drawn actual children per selected actual
	parent repeat; no private scaffold/cap origins escape final graphs. Calibration
	retains frontier/composition covariance and full child variance, with a count
	law per level including rewards, not a discarded plain-parent fit first.
	Each level requires `physical_compound` `mwd` for the entire isolated final
	child. Ten declarations total include siblings. Oriented f2 child entry/exit
	sets must each remain one endpoint in every reachable child state, with
	identity-independent external source/target laws. Declared shared-carbon split
	sites contract to real atoms on the general macro path; bare nonnested
	same-atom multiports remain unsupported. Arbitrary multiboundary/random-affinity/
	coupled interfaces are not supported: this is not unconstrained backbone
	chemistry. Failures do not redraw counts or rewind RNG; legacy mass-stop
	nested RNG-order parity is not promised.
- **2026-10-29 current extension — NB convergence checkpoints:** seeded
	`checkpoint` / `checkpoint_callback` support both full and statistics policies.
	Source-ordered model fingerprints fail closed before RNG construction,
	callbacks or sampling, binding chemistry, laws, actual compiled arrays/adapters,
	nested local source order and assembly; nonsemantic diagnostics/caches are
	excluded. General macros bind the temporary-cap namespace and actual local
	scaffold/source/replacement state; only original top origins in retained public
	data are rebased. This is **not graph-isomorphism canonicalization**: changed
	operational order or fresh-recompile floating summation/parameter last bits
	can require rejection; fail-closed behavior is intentional.
	Source-free JSON/pickle is supported subject to that check. Legacy non-NB
	checkpoint compatibility is unchanged; ordinary noncheckpoint sampling never
	fingerprints. The **historical** 2226-pass/3-skipped full run, two subsequent
	3000-molecule population tests and [October 29 runtime receipts](benchmarks/NB_EXTENSIONS_2026-10-29.md)
	cover the earlier bounded pendant/B3 source, **not the generalized macro source**.
	**Verified validation evidence:** the [macro receipt](benchmarks/NB_MACROS_2026-10-29.md)
	records 300/300 valid connected RDKit-parsed SMILES from the three exact,
	unchanged public blocks with unpatched counts, and 360/360 fixed-count
	molecules passing independent chemistry/owner/metadata checks in the post-test
	[final runtime capture](benchmarks/NB_MACROS_FINAL_2026-10-29.json).
	The [permanent validation record](benchmarks/NB_MACROS_VALIDATION_2026-10-29.json)
	confirms **2747 passed, 3 skipped, 62 warnings in 1123.59 s**, all 2750 collected
	cases with zero failures/errors and unchanged timed package identity. Final
	documented receipt reconciliation is complete in the
	[expanded acceptance snapshot](docs/negative-binomial-implementation-status.md#current-expanded-macro-snapshot--validation-evidence).
	All 30 fresh legacy parity pairs pass; small/large fail the proposed 1%
	pipeline gate, nested passes only for its fixture. No full enabled-law or
	release acceptance; noise does not waive failed gates.
	Functionality-one repeats, external starters, inter-unit cycles/fusion,
	association, isotope masses and stereo-coupled junctions remain outside the
	supported NB chemistry; inter-unit junctions require covalent single bonds.
	Internal monomer rings are not blanket excluded. See
	[current scope and real macro examples](docs/negative-binomial-implementation-status.md).
	This is not full-plan completion or release/performance acceptance.
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

**Historical B1/B2 snapshot (2026-10-29, before B3/C3/checkpoints and the cap
fix):** the recorded production suite passed **1948 tests, 3 skipped, 38 warnings
in 609.66 s**. A subsequently added, unpatched 3000-chain variable-functionality
($f\in\{2,3,4\}$) H/F-cap population test passed separately in **8.37 s**, using independent
fourth-order MGF/count six-standard-error bounds and actual RDKit masses.
It is **not included** in the 1948-pass full-suite total; at that snapshot,
production code had not changed since that full run. Subsequent feature work
has changed it; these are not current full-suite results. See the
[validation record](docs/negative-binomial-implementation-status.md#acceptance-status).

The [October 29 branching diagnostics](benchmarks/README.md#branching-b1b2-october-29)
measure fixed-count native assembly with only count/mass draws patched, not
full enabled-feature overhead. The fresh archived-pre-feature legacy rerun has
exact graph/count/vector-width/export parity, but small/nested end-to-end
changes fail the proposed 1% gate on a shared host. Older October 28 receipts
predate both naming and branching; they remain historical evidence, not current
performance acceptance. These B1/B2 receipts also predate B3, C3, recursive
pendants, NB checkpoints and the integer-origin cap fix. The separate
[October 29 NB_EXTENSIONS receipts](benchmarks/NB_EXTENSIONS_2026-10-29.md)
exist for that **earlier bounded pendant/B3 source**, including its failed
legacy performance gates. They do **not** time the new generalized macro,
heterogeneous-repeat or B3-with-rewards source. The new
[macro runtime/public-example receipt](benchmarks/NB_MACROS_2026-10-29.md) does:
its [final legacy pairs](benchmarks/NB_MACROS_LEGACY_2026-10-29.json) all pass
parity, but median paired pipeline changes **+1.1619496% / +1.2824360% /
−0.1078471%** for small/large/nested fail the proposed 1% gate for small and
large; nested passes for this fixture only. Fixed-count diagnostics do not
establish a full enabled-law budget. The first expanded full run had **2745
passed, 2 failed, 3 skipped, 62 warnings in 1138.37 s**; targeted repairs passed
**5 tests in 3.68 s**, without relaxing production tolerances. The final full
rerun verified **2747 passed, 3 skipped, 62 warnings in 1123.59 s**: all 2750
collected cases, zero failures/errors, including 519 cases across the six new
macro/checkpoint/solver/nesting/boundary/reward files. The
[permanent validation record](benchmarks/NB_MACROS_VALIDATION_2026-10-29.json)
binds that result to the unchanged timed package. The post-test
[final runtime raw data](benchmarks/NB_MACROS_FINAL_2026-10-29.json) is authoritative;
the corrected descriptive formula label and final tables are in the macro receipt.
Saved-source integrity and final documented receipt reconciliation are complete;
frozen source files lacking terminal newlines were reported, not normalized.
The earlier 145-test and 438-test selections remain historical. No sampling
timing table is copied here; historical audits and plans retain their scope.

### Changed

- **Convergence representative cover defaults to typed-medoid selection.** With `representative_distance`, keyword-only `representative_selection="typed-medoid"` also compares log counts per building-block type, assigns each accepted chain to the nearest cluster whose members all lie within the threshold (complete linkage), and retains each cluster's medoid. Medoids are converted in a seeded replay of their batches, so SMILES/graph work is spent only on representatives; unseeded runs generate a seed, reported with the selection in `convergence_settings`. Retained chains and `representative_counts` can differ from earlier runs; pass `representative_selection="first-match"` for the historical first-fit cover. Feature state for every accepted chain is kept in memory and in checkpoints, so assignment cost grows with sample count. Older cover checkpoints resume only with `"first-match"`.
- The preferred recipe-calibrated distribution spelling is `mwd(Mw,Mn)`;
	`shifted_negative_binomial(Mw,Mn)` remains an accepted alias without warnings.
	Rendered model/distribution strings and metadata now use `mwd(...)`, also when
	reading the older spelling or numeric vectors. Argument order (`Mw`, then
	`Mn`), the `ShiftedNegativeBinomial` Python class and numeric graph
	serialization layouts/schema identifiers are unchanged. The naming-only
	change did not alter calibration or sampling; the separate B1/B2 extension
	above adds an opt-in grouped frontier selector and sparse per-step journals.
	Branch selection preserves the weighted-port probability law but consumes RNG
	differently from the legacy live-weight list. The recorded full suite covers
	the earlier B1/B2 linear/legacy snapshot, not seeded nested RNG-order parity
	for the new independent-pendant implementation.
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
