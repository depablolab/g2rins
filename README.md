# G<sup>2</sup>RINS

Implementation of G2RINS, a line-and-graph representation of polymers. Grammar validation, generative graph construction, and molecular ensemble generation.

---

## About

**G<sup>2</sup>RINS** stands for **G**enerative **G**raph **R**epresentation of **I**ntegrated **N**ested Big**SMILES** and is pronounced “grins”. It is a compact string- and graph-based polymer representation designed to support computational materials discovery, enable machine-learning workflows, and connect computational predictions with experimental polymer design.

G²RINS encodes repeat units, end groups, connectivity, architecture, composition, and molecular-weight distributions, enabling automated generation of diverse, simulation-ready polymer ensembles.

---

## Installation

Requires Python ≥ 3.10.

```bash
pip install "g2rins @ git+https://github.com/depablolab/g2rins.git@v1.0.0"
```

For development, clone the repository and install it editable:

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

Worked examples are in [`G2RINS_guide.ipynb`](G2RINS_guide.ipynb).

---

## Heritage

G²RINS is the evolution of **G-BigSMILES** ([latest repository](https://github.com/gervasiozaldivar/G-BigSMILES), [original repository](https://github.com/InnocentBug/G-BigSMILES)), which extends the [BigSMILES line notation](https://olsenlabmit.github.io/BigSMILES/docs/line_notation.html):

> Schneider, Walsh, Olsen, de Pablo, *Generative BigSMILES: an extension for polymer informatics, computer simulations & ML/AI*, Digital Discovery **3**, 51–61 (2024). [doi:10.1039/D3DD00147D](https://doi.org/10.1039/D3DD00147D)

A publication describing G²RINS is in preparation.

---

## License

[GPL-3.0](LICENSE)
