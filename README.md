# G<sup>2</sup>RINS

[![CI](https://github.com/depablolab/g2rins/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/depablolab/g2rins/actions/workflows/ci.yml)
[![License: GPL-3.0-only](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)
[![ChemRxiv](https://img.shields.io/badge/paper-ChemRxiv-b31b1b.svg)](https://doi.org/10.26434/chemrxiv.15007504/v1)

G²RINS is a generative string-and-graph representation for describing complex polymer structures. This package validates G²RINS strings, constructs generative graphs, and generates simulation-ready molecular ensembles.

---

## About

**G<sup>2</sup>RINS** stands for **G**enerative **G**raph **R**epresentation of **I**ntegrated **N**ested Big**SMILES** and is pronounced “grins”. It is a compact string- and graph-based polymer representation designed to support computational materials discovery, enable machine-learning workflows, and connect computational predictions with experimental polymer design.

G²RINS encodes repeat units, end groups, connectivity, architecture, composition, and molecular-weight distributions, enabling automated generation of diverse, simulation-ready polymer ensembles.

---

## Highlights

- Represents branched polymers, prepolymers, repeat units, and end groups
- Encodes polymer connectivity, architecture, composition, and molecular-weight distributions
- Converts G²RINS strings into generative graphs
- Generates molecular ensembles as SMILES strings, RDKit molecules, or molecular graphs

---

## Paper

G²RINS is described in our ChemRxiv preprint:

> Gervasio Zaldivar<sup>†</sup>, Yuan Tian<sup>†</sup>, Ge Sun, Chryssa Kappatou, Philipp Eiden, Niklas B. Wulkow, Volker Settels, and Juan J. de Pablo, _G2RINS: A Generative String-and-Graph Polymer Representation to Assist Computational Materials Discovery_, ChemRxiv (2026), version 1. [doi:10.26434/chemrxiv.15007504/v1](https://doi.org/10.26434/chemrxiv.15007504/v1)
>
> <sup>†</sup> Equal contribution.

---

## Installation

Requires Python ≥ 3.10.

Install the tagged v1.0.0 release directly from GitHub:

```bash
pip install "g2rins @ git+https://github.com/depablolab/g2rins.git@v1.0.0"
```

To install the latest development version from `main`:

```bash
pip install "g2rins @ git+https://github.com/depablolab/g2rins.git@main"
```

For local development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/depablolab/g2rins.git
cd g2rins
pip install -e ".[test]"
```

---

## Quickstart

```python
import g2rins

g2rins_string = "{[] [<]CC([>])c1ccccc1; [>][H]; [<][H] []}|gauss(1000, 120)|"
g2rins_object = g2rins.G2rins.make(g2rins_string)

graph_creator = g2rins_object.get_graph_creator()
generative_graph = graph_creator.get_generative_graph()

ensemble_creator = graph_creator.get_ensemble_creator()
ensemble = ensemble_creator.create_ensemble(100, output_format="smiles", ensemble_info=True)
```

`generative_graph` is the graph representation constructed from the encoded polymer. With `ensemble_info=True`, `ensemble` is an `EnsembleData` object: `ensemble.chains` contains the generated SMILES strings, while the remaining fields describe units, bonds, sequences, molecular weights, and distributions.

When sequence metadata is retained, `ensemble.sequence_unit_ids` has the same
chain/segment/unit shape as `ensemble.sequences`, but contains the exact sampled
unit labels (`I0`, `R0`, `T0`, etc.) regardless of `output_format`. This sidecar
also appears in fixed-size JSON exports. Growth segments are not independently
initiated chains: do not infer unit roles from a segment's first position or
attempt to recover ambiguous identities from fragment SMILES. Sequence retention
disabled (or unavailable metadata) yields `sequence_unit_ids=None`.

For experimental serial convergence topology retention, use
`ensemble_creator.create_ensemble_until_converged(metadata=True, retain_bond_sequences=True)`;
no record-count or byte-budget options are required. Each record's `occurrences`
maps local instance IDs to stable building-block type labels such as `R1`.
Occurrence IDs are local references, not additional building-block types; an
edge list containing only type labels loses which instances connect and how
branches are joined. Fixed-size topology export, exact topology counting,
parallel execution, and checkpoint/resume remain unsupported. See the
[changelog](CHANGELOG.md) for scope and limitations.

New topology exports declare `hydrogen_termination="implicit/v1"`. Explicit
ordinary hydrogen-only terminal units are omitted from the representation,
matching assumed valence hydrogen, before retained topology and all-accepted
centered next-neighbor motif statistics are produced. Compact, branches and
paths therefore agree for the same realized chemistry regardless of whether
H termination was specified. Isotopic/charged H, non-H end groups, hydrogen
initiators/repeat units, isolated H and H-H molecules remain explicit. Sampling,
SMILES, masses, legacy sequences, and raw unit/contact statistics are unchanged;
their explicit-H counts need not match normalized topology/motif totals.
Historical records without this policy are not automatically rewritten. The
offline renderers below render their supplied topology as-is, since they do
not receive canonical unit chemistry.

For a sequence-like view without repeated trunks, add
`bond_sequence_format="branches"`. `bond_sequences` contains a list of segments
per chain, with singleton roots, e.g.
`[["null.I0.null"], ["null.I0.1","1.R1.3","1.T1.null"], ["1.R1.2","1.R2.null"]]`.
The aligned `bond_sequence_parents` entry is `[null,[0,0],[1,1]]`: each non-null
pair references `[parent_segment, junction_position]`, both zero-based. The first
token of that segment references the existing junction, not a new building-block
copy. Thus the arm above starts at the `R1` in segment 1, position 1, via site 2.
Only junction references repeat; every other occurrence is introduced once.
`null` means a segment endpoint, not a chemically terminated or unreacted site.
The earliest surviving occurrence ID is continued at a junction; other children
start separate segments. This deterministic growth-order convention does not
claim to chemically identify a backbone. Parent references preserve tree
connectivity and occupied sites even with identical unit labels; bond categories
and aromaticity still require `compact`. Cycles are rejected explicitly.
The public `g2rins.render_bond_sequence_branches(topology)` returns
`(segments, parents)` for existing compact records without resampling.

For the older repeated root-to-leaf view, use `bond_sequence_format="paths"`. Each retained
chain is a list of root-to-leaf paths, for example
`[["null.I0.2", "1.R0.3", "1.T0.null"]]`. The token `1.R0.3` means entering
unit `R0` through canonical pSMILES attachment site 1 and leaving through site 3.
`null` marks a path endpoint, not an unreacted site. Shared trunks repeat across
paths; occurrence identity and bond order/aromaticity are not encoded, so this
view must not be used as an exact topology key. Cycles raise an explicit error;
use `bond_sequence_format="compact"` (the Python API default) for lossless graphs.
Rendering is post-retention and does not change sampling or coverage weights.
The public `g2rins.render_bond_sequence_paths(topology, max_tokens=1_000_000)`
also renders existing compact records offline. The token budget bounds repeated
trunk expansion and raises explicitly rather than truncating output.

Worked examples are in [`G2RINS_guide.ipynb`](G2RINS_guide.ipynb).

---

## Building-block population statistics

The independent, keyword-only `building_block_statistics=True` option is
available on both `EnsembleCreator.create_ensemble` and
`EnsembleCreator.create_ensemble_until_converged`. It counts building-block
types, rooted neighborhoods, and directed attachment-site connections over
**all accepted samples**, not just selected representatives. It does not enable
`retain_bond_sequences` or `retain_motif_membership`.

For a statistics-only parallel convergence run, reuse the creator above:

```python
population = ensemble_creator.create_ensemble_until_converged(
  batch_size=25,
  max_samples=1000,
  seed=7,
  metadata=False,
  retain_chains=False,
  retain_sequences=False,
  building_block_statistics=True,
  parallel=True,
  n_workers=4,
)
statistics = population.building_block_statistics
```

`metadata=False` avoids legacy path enumeration and sequence work; compact
occurrence data still supplies these independent statistics. No retained chains,
sequences, or topology records are required. For fixed-size sampling, request
`create_ensemble(100, ensemble_info=True, building_block_statistics=True)` and
read the same result field. Fixed-size requests require `ensemble_info=True`
or a `json_file` destination; JSON-only requests can leave `ensemble_info=False`
and write the dictionary under `ensemble.building_block_statistics`. Fixed-size
sampling also accepts `parallel=True, n_workers=4` with this option.

Sampling and neighborhood counting run in the existing worker pool. Each
successful job returns raw counters; the ordered parent loop merges accepted
results, then assigns motif IDs and computes probabilities from the combined
endpoint totals. Counts are not averaged across workers or weighted by retained
representatives. Convergence decisions and retention selection remain in the
parent. The existing worker-crash recovery and seeded chain ordering apply.
Serial execution remains the default; `n_workers=1` uses the existing serial
escape hatch, and automatic worker selection may also choose one worker for a
small batch. Process startup/transport can outweigh the benefit for small runs.

Convergence checkpoint/resume (including checkpoint callbacks) is still
unsupported for these statistics and raises an error. This does not enable
parallel bond-topology retention or motif-membership export; those independent
features retain their existing restrictions.

See the [parallel validation receipt](benchmarks/BUILDING_BLOCK_PARALLEL_2026-10-19.md)
for real-worker, seeded-parity, and crash-recovery coverage.

### Neighborhood and probability semantics

Each unit occurrence is a center, identified by its existing stable unit-type
label (for example, `R1`), not a new chemical-equivalence class. Its motif is a
**root-incident covalent multistar**, **not an induced radius-one graph**:
only covalent links incident to the root are included, with no links between
neighbors. Connections are grouped by the actual neighboring occurrence. Thus
parallel links to one neighbor remain one group with several arms; two distinct
neighbors of the same type remain two groups. A self-link forms a group marked
`is_center=True` and includes both oriented arms. Isolated centers are counted
with an empty neighbor list. Association bonds (category `0`) are excluded;
covalent categories `1`–`4`, attachment sites, and aromatic flags are preserved.
Occurrence addresses are temporary grouping references, not motif identities.
Type/site labels are scoped to the input model; do not join `R1` from unrelated
polymer recipes without an explicit mapping between their building blocks.

For each row, `occurrence_count` is the exact population total and
`chain_presence_count` increments **once per accepted chain** containing that
type or motif, regardless of its multiplicity. These are not representative
coverage weights. Empty accepted samples still increment `accepted_count`.

Site connections are directed endpoint observations. Their `probability` is
`endpoint_count / occupied_endpoint_count`, where the denominator sums occupied
endpoints for that `(source_unit, source_site)` over all accepted samples. This
is conditional on **occupied source-site endpoints**, not all eligible sites,
unit occurrences, chains, or input sampling weights. Every physical covalent
bond contributes two oriented endpoints, including homotypic bonds and
self-links; site-row `chain_presence_count` still increments only once per chain.

### Result dictionary: building-block-statistics/v1

The detached, JSON-safe `result.building_block_statistics` dictionary contains:

| Field | Contents |
| --- | --- |
| `schema` | `"building-block-statistics/v1"` |
| `population` | `"all-accepted-samples"` |
| `accepted_count` | Number of accepted samples included |
| `neighborhood_definition` | `"root-incident-covalent-multistar/v1"` |
| `probability_definition` | `"conditional-on-occupied-source-site-endpoints"` |
| `building_blocks` | Rows with `unit_id`, `occurrence_count`, `chain_presence_count` |
| `neighborhoods` | Rows with `motif_id`, `center_unit`, grouped `neighbors`, `occurrence_count`, `chain_presence_count` |
| `site_connections` | Rows with `source_unit`, `source_site`, `target_unit`, `target_site`, `bond_type`, `aromatic`, `endpoint_count`, `chain_presence_count`, `occupied_endpoint_count`, `probability` |

Each neighbor group has `unit_id`, `is_center`, and `connections`; each
connection has `source_site`, `target_site`, `bond_type`, and `aromatic`.
Sites are canonical positive decimal strings and bond categories are strings.
Rows and arms are deterministically sorted. Neighborhood `motif_id` values
(`bbn:1`, etc.) are snapshot-local sorted ordinals, not durable IDs across
populations that introduce new motifs. Disabled results leave the field `None`;
fixed-size JSON omits it when disabled. See the
[schema implementation](src/g2rins/_building_block_statistics.py) for details.

These counts consume the existing occurrence connectivity: no new chain graph,
SMILES, hashing, or chemical canonicalization is needed solely to count them.
They are structural population summaries, not chemical motif structures or a
complete-chain topology export. Keep the existing selected-chain bond-sequence
topology for GNN inputs that need instance connectivity, branching, and sites;
type-level counts cannot replace that graph.

The [benchmark selector](benchmarks/README.md#building-block-statistics-toggle)
compares this feature independently. The separate
[45-pair measurement report](benchmarks/BUILDING_BLOCK_STATISTICS_2026-10-19.md)
records parity and workload-specific overhead for the earlier **serial source
snapshot**; it does not measure parallel performance or the updated source.
It is not a historical-disabled regression test or universal runtime guarantee.
Existing frozen bond-sequence reports measure different work.

---

## Background

G²RINS builds on G-BigSMILES, which extends the [BigSMILES line notation](https://olsenlabmit.github.io/BigSMILES/docs/line_notation.html):

> Schneider, Walsh, Olsen, de Pablo, _Generative BigSMILES: an extension for polymer informatics, computer simulations & ML/AI_, Digital Discovery **3**, 51–61 (2024). [doi:10.1039/D3DD00147D](https://doi.org/10.1039/D3DD00147D)

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

Report bugs or request features through the [issue tracker](https://github.com/depablolab/g2rins/issues). Notable changes are documented in [CHANGELOG.md](CHANGELOG.md) and on the [releases page](https://github.com/depablolab/g2rins/releases).

---

## Citing

If you use G²RINS in your work, please cite the [G²RINS paper](https://doi.org/10.26434/chemrxiv.15007504/v1):

```bibtex
@article{zaldivar2026g2rins,
  title = {{G2RINS}: A Generative String-and-Graph Polymer Representation to Assist Computational Materials Discovery},
  author = {Zaldivar, Gervasio and Tian, Yuan and Sun, Ge and Kappatou, Chryssa and Eiden, Philipp and Wulkow, Niklas B. and Settels, Volker and de Pablo, Juan J.},
  journal = {ChemRxiv},
  year = {2026},
  doi = {10.26434/chemrxiv.15007504/v1},
  url = {https://doi.org/10.26434/chemrxiv.15007504/v1},
  note = {Version 1, preprint}
}
```

The paper citation is also available through [CITATION.cff](CITATION.cff) and GitHub’s "Cite this repository" button.

---

## License

G²RINS is distributed under the [GNU General Public License v3.0](LICENSE).
