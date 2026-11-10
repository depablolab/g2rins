# G2RINS

[![CI](https://github.com/depablolab/g2rins/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/depablolab/g2rins/actions/workflows/ci.yml)
[![License: GPL-3.0-only](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)
[![ChemRxiv](https://img.shields.io/badge/paper-ChemRxiv-b31b1b.svg)](https://doi.org/10.26434/chemrxiv.15007504/v1)

G2RINS is a generative string-and-graph representation for describing complex polymer structures. The `g2rins` package validates G2RINS strings, constructs generative graphs, and generates simulation-ready molecular ensembles.

---

## About

**G2RINS** stands for **G**enerative **G**raph **R**epresentation of **I**ntegrated **N**ested Big**SMILES** and is pronounced “grins”. It connects computational materials discovery, machine-learning workflows, and experimental polymer design.

G2RINS encodes repeat units, end groups, connectivity, architecture, composition, and molecular-weight distributions.

---

## Highlights

- Represents branched polymers, prepolymers, repeat units, and end groups
- Encodes polymer connectivity, architecture, composition, and molecular-weight distributions
- Converts G2RINS strings into generative graphs
- Generates molecular ensembles as SMILES strings, RDKit molecules, or molecular graphs

---

## Paper

G2RINS is described in our ChemRxiv preprint:

> Gervasio Zaldivar<sup>†</sup>, Yuan Tian<sup>†</sup>, Ge Sun, Chryssa Kappatou, Philipp Eiden, Niklas B. Wulkow, Volker Settels, and Juan J. de Pablo, _G2RINS: A Generative String-and-Graph Polymer Representation to Assist Computational Materials Discovery_, ChemRxiv (2026), version 1. [doi:10.26434/chemrxiv.15007504/v1](https://doi.org/10.26434/chemrxiv.15007504/v1)
>
> <sup>†</sup> Equal contribution.

---

## Installation

Requires Python ≥ 3.10.

**Version scope:** this README and the [documentation index](docs/index.md)
describe the current development tree. The tagged **v1.0.0** install below is
the initial release, not a promise of the development API. Convergence,
`sequence_unit_ids`, topology/motif exports, building-block statistics, and
format-v2 migration guidance here describe unreleased development behavior.
See [CHANGELOG.md](CHANGELOG.md) before upgrading; no later release is asserted.

Install the tagged v1.0.0 release directly from GitHub:

```bash
pip install "g2rins @ git+https://github.com/depablolab/g2rins.git@v1.0.0"
```

To install the development version on `main` (not a tagged release):

```bash
pip install "g2rins @ git+https://github.com/depablolab/g2rins.git@main"
```

The `next` branch is the release-candidate integration line; use the branch
containing the reviewed feature you need. Development branches can differ from
each other and from this checkout.

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
graph_creator = g2rins.G2rins.make(g2rins_string).get_graph_creator()
ensemble_creator = graph_creator.get_ensemble_creator()
ensemble = ensemble_creator.create_ensemble(100, output_format="smiles", ensemble_info=True)
print(ensemble.chains[0])
```

With `ensemble_info=True`, the result is an `EnsembleData` object with chains,
units, bonds, sequences, molecular weights, and distributions. Otherwise the
fixed-size API returns a list of chains. Use
`graph_creator.get_generative_graph()` to inspect the generative graph.
The [migration guide](docs/migration.md#ensemble-output-format-v2) describes
development output fields and JSON changes.

Worked examples are in [G2RINS_guide.ipynb](G2RINS_guide.ipynb); check its APIs
against your installed revision.

## Documentation

- [Documentation index and feature support matrix](docs/index.md): fixed-size
  versus convergence, metadata, parallel execution, and checkpoint restrictions.
- [Sequences and topology](docs/topology.md): exact unit labels, compact graphs,
  hydrogen policy, branches/paths, and independent motif analytics.
- [Building-block population statistics](docs/building-block-statistics.md):
  usage, schema, neighborhoods, and probability denominators.
- [Migration guide](docs/migration.md): format v2, wildcard/placeholder safety,
  source selection, termination, and validation changes.
- [Changelog](CHANGELOG.md), [contribution guide](CONTRIBUTING.md), and
  [benchmark methods and frozen evidence](benchmarks/README.md).

---

## Building-block population statistics

`building_block_statistics=True` independently counts types, rooted covalent
neighborhoods, and directed site endpoints over **all accepted samples**. It
supports fixed-size/convergence and serial/parallel runs, but not convergence
checkpoint/resume. See the [usage and complete reference](docs/building-block-statistics.md)
and [support matrix](docs/index.md#feature-support-matrix).

### Neighborhood and probability semantics

Moved to [neighborhood and probability semantics](docs/building-block-statistics.md#neighborhood-and-probability-semantics).
This heading remains for existing links.

### Result dictionary: building-block-statistics/v1

Moved to the [complete result dictionary](docs/building-block-statistics.md#result-dictionary-building-block-statisticsv1).
This heading remains for existing links.

---

## Background

G2RINS builds on G-BigSMILES, which extends the [BigSMILES line notation](https://olsenlabmit.github.io/BigSMILES/docs/line_notation.html):

> Schneider, Walsh, Olsen, de Pablo, _Generative BigSMILES: an extension for polymer informatics, computer simulations & ML/AI_, Digital Discovery **3**, 51–61 (2024). [doi:10.1039/D3DD00147D](https://doi.org/10.1039/D3DD00147D)

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

Report bugs or request features through the [issue tracker](https://github.com/depablolab/g2rins/issues). Notable changes are documented in [CHANGELOG.md](CHANGELOG.md) and on the [releases page](https://github.com/depablolab/g2rins/releases).

---

## Citing

If you use G2RINS in your work, please cite the [G2RINS paper](https://doi.org/10.26434/chemrxiv.15007504/v1):

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

G2RINS is distributed under the [GNU General Public License v3.0](LICENSE).
