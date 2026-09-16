# Changelog

Notable, user-visible changes to G²RINS. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions correspond to the repository's `v*` git tags.

## [Unreleased]

### Breaking changes

**Generative-graph JSON format v2 requires explicit identity on zero-number nodes.** Graph export and ensemble construction now raise `IncompatibleGenerativeGraphSchema` if any node with `atomic_num=0` lacks the boolean `is_connector_placeholder` attribute. `True` identifies an internal split-atom placeholder; `False` identifies a user wildcard, which can still be parsed and exported but cannot be used for ensemble generation. Graphs without zero-number nodes may omit this attribute. Changing the JSON `format.version` field alone does not migrate node identity.

- **Preferred upgrade:** rebuild legacy graphs from their original G2RINS strings using `G2rins.make(text).get_graph_creator().get_generative_graph()` before export or generation.
- **Migration without source strings requires known provenance:** use `mark_legacy_connector_placeholders(graph)` only when every unmarked zero-number node with exactly one static neighbor is known to be an internal placeholder. A pendant user wildcard such as `CC(*)O` has that same topology and will otherwise be marked `True`, incorrectly treating it as an internal placeholder. The helper cannot recover this distinction from graph structure. If the assumption cannot be established, recover the node identities from the graph producer before migrating.
- **A migration warning is not verification:** `G2RINSWarning` reports counts, not affected node IDs, and does not establish that the assumption is correct. The helper returns a copy and preserves existing valid flags; compare the input and returned node flags to audit which identities it assigned.

**Ensemble output format v2.** `EnsembleData.units` records are `{"psmiles", "g2rins", "subgraph", "count"}` (`frequency` renamed to `count`) and `EnsembleData.bonds` records are `{"labels", "nodes", "count"}` (`between` renamed to `labels`); the `ensemble` section of JSON files uses the same keys and shares `format.version` 2 with the generative-graph JSON format above. The `mol` output format of `create_ensemble` is removed. JSON files written with the default `output_format="mol_graph"` now store node-link chain graphs instead of SMILES strings; pass `output_format="smiles"` to keep the compact chains of version 1. `EnsembleData` equality is structural.

### Added

- `mark_legacy_connector_placeholders(graph)` provides explicit migration for legacy datasets with known placeholder semantics; see [Breaking changes](#breaking-changes) for its assumptions and limitations. It marks unmarked zero-number nodes whose static degree is not one (a terminal `[<]*`, a backbone `C*C`) as user wildcards. Migration and ensemble validation count incoming and outgoing static neighbors, including when a reverse non-static edge exists. Export and generation require explicit identity on zero-number nodes.
- `UnitLabels.unit_nodes` exposes the per-unit node partition behind `unit_id` and `bond_id`, so consumers no longer rebuild it by scanning every node.
- `CONTRIBUTING.md`, `CITATION.cff`, this changelog, issue forms, and a pull request template.
- GitHub Release automation for future `v*` tags: build, verify, attach wheel/sdist, and generate release notes.
- Unit records in the ensemble output carry the unit's static subgraph of the generative graph (`subgraph`: original node ids, static edges only, nodes in derivation order and edges in template order, `unit_id` stamped on the copy's nodes, no graph-level attributes; node-link encoded in JSON files).
- Bond records carry the generative-graph node ids of the two connection atoms (`nodes`, positionally aligned with `labels`; labels survive a fresh parse, node ids are only valid for the graph they came from).
- Ensemble JSON files follow the requested `output_format` for their stored chains — SMILES strings or node-link graph dicts — and record the choice in `format.chain_format`. Sequences are written as SMILES regardless. The whole `ensemble` section is normalized like the graph section (NumPy values become JSON values, non-finite values raise `ValueError` before the file is opened), and the file bytes do not depend on the interpreter's hash seed.

### Changed

- User-written wildcard atoms (`*` and `[*]`) remain supported for parsing and generative-graph export, but now raise `UnsupportedWildcardGeneration` when an ensemble creator is built. They are no longer silently removed from generated chains or reported as internal rendering failures. Diagnostics include the node ID, derived unit ID, and verified source unit text when available, and survive pickling and deep copying; the node diagnostic remains available if unit-label derivation fails. Use explicit atoms and end groups, such as `[<][H]` for a hydrogen terminator.
- The extra graph information returned by `get_generative_graph(return_extra_graph_info=True)` no longer includes the unused `0: "None"` entry. Zero denotes a wildcard or a flagged connector placeholder; negative entries still label non-atom graph objects.
- Pull requests now run a faster Linux-only Python 3.10/3.14 test matrix, while the full Linux/Windows/macOS compatibility matrix runs after merges to `main` and on the monthly schedule. Python 3.14 replaces 3.13 as the highest version tested in CI (RDKit ≥ 2026.3 publishes Python 3.14 wheels).
- The CI, trunk and release-build workflows also run for pushes and pull requests to the `next` branch, the release-candidate line that collects reviewed changes ahead of the next release.
- The full Linux/Windows/macOS compatibility matrix can be run on demand, through a manual workflow dispatch, on any branch that carries the trigger, so a release candidate is checked on all three platforms before it lands.
- Updated official GitHub Actions to current stable majors and tightened workflow permissions.
- Packaging and installation workflows fetch full Git history and tags so `setuptools-scm` can derive versions reliably.
- Simplified the `setuptools-scm` configuration in `pyproject.toml` while preserving `g2rins.__version__`.
- The ensemble output format — the `ensemble` section of JSON files and `EnsembleData` — is version 2, sharing the `format.version` field with the generative-graph JSON format v2 above: the unit record key `frequency` is renamed to `count`, the bond record key `between` is renamed to `labels`, and unit records list `psmiles`, `g2rins`, `subgraph`, `count` in that order.
- The ensemble creator's private copy of the generative graph carries `is_connector_placeholder` on every node, filling `False` where a legacy graph omits it on a real atom, and drops the derived `unit_id`/`bond_id` node attributes that a graph loaded back from a JSON export carries, so unit subgraphs always expose the flag and carry the current `unit_id` only.
- `EnsembleData` equality compares graph-valued members (unit subgraphs, and molecule-graph chains or sequences) by structure instead of object identity, with NumPy arrays compared by value, so two ensembles with the same content compare equal; the comparison is reflexive, never broadcasts a NumPy scalar against a container, compares object arrays element-wise, and compares structured NumPy data only with structured data of the same dtype.

### Removed

- The `mol` output format of `create_ensemble`. Request `mol_graph` and convert chains with `g2rins.mol_graph_to_rdkit_mol`, or parse the SMILES output. Sequence fragments have dangling inter-unit valences, so convert them with `g2rins.mol_graph_to_rdkit_mol(unit, kekulize=False)`.

### Fixed

- Split atoms with several connection sites now render one dummy per active site in ensemble unit pSMILES and sequence fragments, eliminating duplicate dummy atoms. Split placeholders become mapped pSMILES stars; recorded sequence stubs attach directly to the real atom and preserve the realized junction bond order. Active split sites without a recorded stub retain their unmapped `*`, including in carbanion fragments. Partnerless split sites, identified as internal placeholders without a template bond ID, are omitted from both representations so they do not consume implicit-hydrogen valence. The generative graph, JSON graph export, and DOT view retain these inactive sites without bond IDs. Completed chains and template bond records are unchanged. Sequence fragments are still hydrogen-capped in isolation, so their masses need not sum to the chain mass.
- Unit pSMILES is validated against template map numbers, connection-star degrees, and real-atom counts before export. Inconsistent output raises `InvalidUnitPSmiles` with normalized diagnostics that survive pickling and deep copying.
- Sequence connection stubs and their attachment bonds are non-aromatic, and stubs are neutral even when their far-side atom is charged. This prevents aromatic inter-unit bonds from producing non-ring aromatic dummy atoms that fail fragment sanitization with `AtomKekulizeException`. Stubs carry `is_connector_placeholder=False` and retain origin and connection provenance without copying sampling bookkeeping.
- Descriptors with multiple atom neighbors now raise the user-facing `UnsupportedBondDescriptor` when descriptors are removed for generation, JSON export, or the default DOT rendering. This includes descriptors embedded between atoms and placements where ring closures or branches attach additional atom neighbors; these previously produced incorrect graphs. The raw parsed graph with `include_bond_connectors=True` remains available for inspection.
- Ensemble creators validate atomic numbers and placeholder structure at construction: atomic numbers must be non-negative integers, excluding booleans, and each connector placeholder must have exactly one static neighbor that is a real atom. Schema errors distinguish missing attributes from invalid values and identify the affected node. NumPy integer atomic numbers and boolean placeholder flags are normalized in the creator's own graph copy without changing the caller's input. This sampling support is limited to those fields: NumPy charge and aromatic values still fail during sampling, as in v1.0.0.
- JSON export normalizes supported NumPy integer, real floating-point, boolean, and string scalars throughout the payload, including nested arrays in node, edge, and graph metadata. This includes charge and aromatic values that are not normalized for sampling. Export validates placeholder identity without imposing generation's atomic-number restrictions; raw descriptor graphs retain negative atomic numbers and export unavailable descriptor charges as `null`. Unsupported types, other non-finite values, and dictionary keys that would collide in JSON raise errors identifying their location. All exported containers are detached from the input, so editing a payload no longer changes its source graph.
- Parsed graphs preserve optional unit membership in the graph-level `unit_node_ids` mapping. Wildcard diagnostics and ensemble metadata use it to verify source unit text after units are removed or labels are reassigned, including when JSON nodes are reordered. Unit text is omitted when membership differs or a saved record is missing or malformed. Legacy graphs without the mapping retain their `unit_g2rins` texts when every saved unit ID is still derived; a present but empty or malformed map does not activate this fallback. Generation remains supported without parser provenance.
- Aromatic selenium and arsenic symbols now map to atomic numbers 34 and 33 instead of negative graph-object IDs, through the aromatic aliases of `atom_name_num`.
- `ParsingError` and `TooManyTokens` preserve their context when pickled or deep-copied. Atoms without a symbol are rejected during construction with `MissingAtomSymbol`, a `ParsingError` subclass.
- Bare wildcard atoms (`*`) retain their symbol when a parsed G2RINS string is serialized or exported, instead of incorrectly becoming `None`.
- CI explicitly installs the `[test]` extra so pytest is available in test jobs (#1).
- Removed a machine-local `.trunk/plugins/trunk` artifact from version control.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
