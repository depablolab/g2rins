# Changelog

Notable, user-visible changes to G²RINS. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions correspond to the repository's `v*` git tags.

## [Unreleased]

### Added

- Generative-graph exports now preserve optional stereochemistry provenance:
  `atom_chiral_token` on chiral bracket atoms and `bond_symbol_raw` for
  slash/backslash directional single bonds.
- Convergence-driven ensemble generation accepts
  `use_repeat_units_as_source=True`, enabling iterative sampling of polymers
  without an initiator.
- Convergence checkpoints support an opt-in `statistics` policy that stores
  resumable convergence counters, history, aggregate metadata, and reservoir
  RNG state without embedding retained chain graphs. The compatible default
  remains `full`.
- Sampling benchmarks support concise median-only matrices and sampling-only CPU attribution by implementation category; the post-optimization profile documents why NetworkX remains the internal graph backend for now.
- Convergence sampling supports bounded chain/sequence reservoirs, per-chain streaming callbacks, optional returned metadata, and serializable seeded checkpoints for exact batch-boundary resume.
- `CONTRIBUTING.md`, `CITATION.cff`, this changelog, issue forms, and a pull request template.
- GitHub Release automation for future `v*` tags: build, verify, attach wheel/sdist, and generate release notes.
- Warnings that report how each open site of a generative graph will be capped: `ShadowedTerminationDeclaration`, `InheritedTermination`, `ForeignControlledTermination`, and `MissingTermination`. The canonical configuration — a site capped in the step that grows it — stays silent.
- Unit records in the ensemble output carry the unit's static subgraph of the generative graph (`subgraph`: original node ids, static edges only, `unit_id` stamped on the copy's nodes; node-link encoded in JSON files).
- Bond records carry the generative-graph node ids of the two connection atoms (`nodes`, positionally aligned with `labels`; labels survive a fresh parse, node ids are only valid for the graph they came from).
- Ensemble JSON files follow the requested `output_format` for their stored chains — SMILES strings or node-link graph dicts — and record the choice in `format.chain_format`. Sequences are written as SMILES regardless.

### Changed

- `mol_graph_to_rdkit_mol` now applies preserved chirality and directional-bond
  metadata before stereochemistry assignment, and tolerates duplicate directed
  edges by ignoring already-added atom pairs.
- Double-bond directional annotations that are incomplete, ambiguous, or
  conflicting now emit explicit runtime warnings. Ambiguous/conflicting
  directional stereoinformation is discarded, leaving E/Z stereochemistry
  unspecified instead of silently forcing an interpretation.
- Generative graphs now retain `double_bond_stereo_defined` provenance for each
  double bond and parallel ensemble creation aggregates directional stereochemistry
  warnings across worker failures and accepted chains without dropping the
  underlying signal.
- Native crash-diagnostic recovery scans JSONL logs backward in bounded chunks
  instead of loading the complete file, and accepted private chain records are
  frozen before checkpoint retention.
- Parallel convergence now keeps one bounded, restartable worker pool across
  successful batches instead of paying process startup per batch. Broken pools
  are replaced without losing completed ordered chains, and nested callback
  sampling uses an independent scheduler.
- Convergence now aggregates each accepted chain immediately and defers
  sequence/requested-format conversion until after callback or reservoir
  selection. Unretained chains are discarded without batch/final compatibility
  conversion, while molecular-weight moments, contacts, units, checkpoints,
  callbacks, and seeded reservoir behavior remain unchanged.
- Sampling now uses explicit private metadata levels for graph-only, counts,
  compact-sequence, and full legacy records. Convergence without retained
  sequences uses stable unit IDs and endpoint counts without materializing
  graph-valued units, while callbacks and public outputs retain their full
  compatibility payloads.
- Sampling prepares source-specific static-unit and half-bond templates once
  per ensemble creator, and merges incoming graph data directly by atom-ID
  offset instead of repeatedly traversing and relabeling static graphs.
- Sampling prepares stochastic distributions once per ensemble creator and
  reuses those immutable templates across attempts and termination estimates.
- Sampling precomputes invariant termination-fragment masses per ensemble
  creator; boundary lookahead now evaluates only live endpoint hydrogen loss
  and dynamic target probabilities instead of constructing temporary graphs.
- Parallel ensembles now initialize one persistent creator per worker, submit compact chain jobs with at most twice the worker count in flight, cap numerical-library threads, and recycle workers where supported. Broken pools preserve completed ordered results and restart up to `max_worker_restarts`, then raise `WorkerProcessFailure` with the last valid native diagnostic state.
- Accepted chains now construct and sanitize one RDKit molecule and reuse it for molecular weight and requested canonical SMILES. Optional durable native-stage diagnostics include chain/seed context and library versions, `faulthandler` is enabled automatically, and enlarged-stack protection covers construction, sanitization, descriptors, and SMILES generation for large molecules.
- Fixed-size and convergence-driven ensemble creation now share one ordered per-chain sampling engine. Convergence updates molecular-weight moments and contact frequencies online instead of rescanning all accumulated samples after every batch.
- Ensemble sampling now tracks unit counts, contacts, and branching sequences with compact IDs and direct atom indexes, materializing the legacy graph-valued metadata only when returning it. This removes per-unit molecule scans and per-occurrence NetworkX copies without changing the public output.
- Exact stochastic rounding now restores rejected growth steps through sparse first-write mutation journals and append watermarks instead of copying the partial molecule, frontier, stochastic tracker, and compact metadata at checkpoint capture. The consumed random stream is deliberately not rewound.
- Pull requests now run a faster Linux-only Python 3.10/3.14 test matrix, while the full Linux/Windows/macOS compatibility matrix runs after merges to `main` and on the monthly schedule. Python 3.14 replaces 3.13 as the highest version tested in CI (RDKit ≥ 2026.3 publishes Python 3.14 wheels).
- Updated official GitHub Actions to current stable majors and tightened workflow permissions.
- Packaging and installation workflows fetch full Git history and tags so `setuptools-scm` can derive versions reliably.
- Simplified the `setuptools-scm` configuration in `pyproject.toml` while preserving `g2rins.__version__`.
- When a nested stochastic object finishes, its continuation and the level's remaining entry sites compete in one weighted draw at the owning level, instead of the continuation firing unconditionally.
- Which terminator caps an open site is resolved at graph construction: the declaration nearest the site wins, and termination edges that can never fire are removed from the generative graph. The stochastic object that owns the site's bond descriptor fires the cap, and the cap's mass counts toward that object's molecular weight target.
- A nested stochastic object used as an initiator now inherits the enclosing object's terminators for its exposed chain ends, as one used as a repeat unit already did.
- Exported graph and ensemble JSON is format version 2: the unit record key `frequency` is renamed to `count`, the bond record key `between` is renamed to `labels`, and unit records list `psmiles`, `g2rins`, `subgraph`, `count` in that order.

### Removed

- Matplotlib is no longer a runtime dependency; an unused internal plotting
  helper had loaded it during ordinary parsing.
- The `mol` output format of `create_ensemble`. Request `mol_graph` and convert chains with `g2rins.mol_graph_to_rdkit_mol`, or parse the SMILES output.

### Fixed

- Valid initiator-free homopolymers are now accepted when a single repeat unit
  is genuinely self-initiating through a matching opposite bond-descriptor pair
  such as `[<]`/`[>]`, `[<1]`/`[>1]`, or `[$]`/`[$]`. Empty or mismatched
  repeat-unit source sets still raise `NoValidGenerationSource` rather than
  silently falling back to another source mode.
- Bond-connector removal no longer enumerates exponentially many impossible
  atom paths and invalid consecutive-transition routes when constructing
  branched, ring-containing initiator-free polymer graphs.
- Adjacent phantom connector nodes are now traversed as one component and
  collapsed using the realized junction's bond attributes, preserving the
  intended bond between their real-atom endpoints.
- Large, ring-rich cyclic polymers now recover from RDKit's open-ring labeling
  overflow during SMILES serialization by retrying with non-canonical traversal
  and a root-aware atom ordering selection, preventing failures such as
  "Too many rings open at once. SMILES cannot be generated." for cyclic monomer
  inputs and long polymer chains.
- Added explicit regression coverage for long cyclic-monomer generation and
  RDKit serialization failures to prevent the issue from returning when new
  generation or serialization logic is added.
- CI explicitly installs the `[test]` extra so pytest is available in test jobs (#1).
- Removed a machine-local `.trunk/plugins/trunk` artifact from version control.
- Nested stochastic objects used as repeat units could not grow their own instances after a transition fired; chains fell short of the outer target and were discarded.
- Open sites handed to another level's custody lost their termination modes, silently dropping declared end groups from finished molecules.

## [1.0.0] - 2026-08-08

Initial public release.

[Unreleased]: https://github.com/depablolab/g2rins/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/depablolab/g2rins/tree/v1.0.0
