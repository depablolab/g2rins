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

Worked examples are in [`G2RINS_guide.ipynb`](G2RINS_guide.ipynb).

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
