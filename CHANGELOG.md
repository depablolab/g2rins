# Changelog

Notable, user-visible changes to G²RINS. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions correspond to the repository's `v*` git tags.

## [Unreleased]

### Added

- Unit source text survives graphs written before the per-unit membership map existed. Such a graph keeps its `unit_g2rins` texts if every saved unit id is still derived, instead of silently reporting empty text for every unit. A present but empty or malformed map is treated as no evidence and never activates the legacy fallback.
- `mark_legacy_connector_placeholders(graph)` migrates a copy of a legacy dataset when the caller knows its unmarked zero-number nodes are internal placeholders. It warns about that assumption, preserves existing flags, and leaves normal loading strict. Unmarked zero-number nodes whose static degree is not one (a terminal `[<]*`, a backbone `C*C`) are provably user wildcards and are marked as such, so only pendant wildcards remain indistinguishable from partnerless placeholders.
- `UnitLabels.unit_nodes` exposes the per-unit node partition behind `unit_id` and `bond_id`, so consumers no longer rebuild it by scanning every node.
- `CONTRIBUTING.md`, `CITATION.cff`, this changelog, issue forms, and a pull request template.
- GitHub Release automation for future `v*` tags: build, verify, attach wheel/sdist, and generate release notes.

### Changed

- User-written wildcard atoms (`*` and `[*]`) remain supported for parsing and generative-graph export, but now raise `UnsupportedWildcardGeneration` when an ensemble creator is built. They are no longer silently removed from generated chains or reported as internal rendering failures. Use explicit atoms and end groups, such as `[<][H]` for a hydrogen terminator.
- Generative graphs and their JSON exports now identify internal split-atom placeholders with the boolean node attribute `is_connector_placeholder`. JSON format version 2 rejects ambiguous older graphs containing unmarked zero-number nodes: rebuild parsed graphs from their original G2RINS strings before export or generation, or explicitly migrate datasets whose placeholder semantics are known. Older graphs without zero-number nodes may omit this new attribute.
- The extra graph information returned by `get_generative_graph(return_extra_graph_info=True)` no longer includes the unused `0: "None"` entry. Zero denotes a wildcard or a flagged connector placeholder; negative entries still label non-atom graph objects.
- Pull requests now run a faster Linux-only Python 3.10/3.14 test matrix, while the full Linux/Windows/macOS compatibility matrix runs after merges to `main` and on the monthly schedule. Python 3.14 replaces 3.13 as the highest version tested in CI (RDKit ≥ 2026.3 publishes Python 3.14 wheels).
- Updated official GitHub Actions to current stable majors and tightened workflow permissions.
- Packaging and installation workflows fetch full Git history and tags so `setuptools-scm` can derive versions reliably.
- Simplified the `setuptools-scm` configuration in `pyproject.toml` while preserving `g2rins.__version__`.

### Fixed

- JSON export normalizes NumPy integer, real floating-point, boolean, and string scalars throughout node, edge, and graph metadata, including nested arrays. Raw descriptor graphs retain negative atomic numbers and export unavailable descriptor charges as `null`. Unsupported types, other non-finite values, and dictionary keys that would collide in JSON raise errors identifying their location. Exported containers remain detached from the input.
- Legacy placeholder migration and ensemble construction now use the same static-neighbor rule, considering incoming and outgoing static edges even when a reverse non-static edge exists. Constructor identity checks share one node pass.
- Descriptors embedded between two atoms of a unit now raise the user-facing `UnsupportedBondDescriptor` when descriptors are removed for generation, JSON export, or the default DOT rendering, before either sampling mode can produce invalid chains or an internal rendering error. The raw parsed graph with `include_bond_connectors=True` remains available for inspection.
- Ensemble creators reject, at construction, connector placeholders whose sole static neighbor is not a real atom, placeholders with a static degree other than one, and nodes with negative atomic numbers. These inputs previously failed inside sampling with an internal error or silently realized wrong bonds depending on the output format.
- Parsed graphs preserve optional unit membership in the graph-level `unit_node_ids` mapping. Wildcard diagnostics and ensemble metadata use it to verify the source unit text after units are removed or labels are reassigned, including when JSON nodes are reordered. Unit text is omitted when membership differs or the saved record is missing or malformed; generation remains supported without parser provenance.
- Sequence connection stubs are neutral and non-aromatic, including when their far-side atom is charged or aromatic, and so is the bond that attaches them. An aromatic inter-unit bond previously produced an aromatic bond to a non-ring dummy atom, so sanitizing the affected sequence fragments raised `AtomKekulizeException`. Stub provenance and the realized bond order are still preserved.
- Generative-graph export payloads are detached from the input, including nested node, edge, and graph attributes. Editing a payload no longer changes its source graph.
- Aromatic selenium and arsenic symbols now map to atomic numbers 34 and 33 instead of negative graph-object IDs, through the aromatic aliases of `atom_name_num`.
- `ParsingError` and `TooManyTokens` preserve their context when pickled or deep-copied. Atoms without a symbol are rejected during construction with `MissingAtomSymbol`, a `ParsingError` subclass, and wildcard diagnostics remain available even if unit-label derivation fails.
- `InvalidUnitPSmiles` preserves its normalized diagnostics when pickled or deep-copied, and includes all diagnostic fields in its error message.
- Schema errors distinguish missing attributes from invalid placeholder flags and identify the affected node. Wildcard generation errors include the node ID, derived unit ID, and source unit text when available; these diagnostics survive pickling and deep copying.
- Ensemble creators accept NumPy boolean placeholder flags and NumPy integer atomic numbers, normalizing both in their own graph copy, preserving the caller's input. A NumPy atomic number previously passed validation and then failed inside RDKit during sampling. Non-integer atomic numbers, including booleans, are now rejected at construction with advice that fits the value. JSON export validates placeholder identity and normalizes NumPy values without imposing generation's atom-number restrictions. Sequence connection stubs carry `is_connector_placeholder=False` and retain provenance without copying sampling bookkeeping.
- Bare wildcard atoms (`*`) retain their symbol when a parsed G2RINS string is serialized or exported, instead of incorrectly becoming `None`.
- Partnerless descriptors on split atoms no longer abort ensemble metadata generation or JSON export. Their inactive placeholders are omitted from unit pSMILES, consistently with partnerless descriptors on single-site atoms. The generative graph, JSON export, and DOT view retain the encoded split sites without bond IDs; unsplit atoms need no extra node.
- Sequence fragments keep the split-atom placeholders that this release had begun deleting, so an interior split unit is no longer rendered as a complete small molecule: a divalent carbanion stays `*[CH-]*` instead of becoming methanide. A split site whose connection the sampler recorded carries a mapped stub; any other split site keeps its unmapped `*`. This restores the previous split-site marker behavior only, and does not restore sequence fragments wholesale: stub charge and aromaticity changed deliberately in this release. Note also that a connection atom carrying a single descriptor has no split-atom placeholder, but can still receive a mapped stub when the sampler records a connection. Fragment masses still do not sum to the chain mass, because each fragment is hydrogen-capped when read in isolation.
- Hyperbranched units with several connection sites on one atom no longer emit duplicate dummy atoms in ensemble unit pSMILES or sequence fragments. Split-atom placeholders now become the mapped pSMILES stars, and sequence-only connection stubs are reattached directly to their real atom during finalization. Completed chain structures, template/JSON bond ids on placeholder nodes, and bond records such as `R0.2` and `R0.3` are unchanged. Unit pSMILES with inconsistent map numbers, connection-star degrees, or real-atom counts now raises an internal generation error instead of being exported.
- CI explicitly installs the `[test]` extra so pytest is available in test jobs (#1).
- Removed a machine-local `.trunk/plugins/trunk` artifact from version control.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
