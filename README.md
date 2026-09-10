# G<sup>2</sup>RINS

[![CI](https://github.com/depablolab/g2rins/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/depablolab/g2rins/actions/workflows/ci.yml)
[![License: GPL-3.0-only](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)

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

### Molar-mass distributions

Poisson and Flory–Schulz support explicit number- and weight-average molar
masses, ordered consistently as `(Mw, Mn)`:

```text
|poisson(1100, 1000)|
|flory_schulz(1400, 1000)|
```

These forms sample molar-mass targets whose equally weighted chain population
has the requested `Mn` and `Mw`. The one-argument forms `poisson(N)` and
`flory_schulz(a)` remain available with their historical behavior for existing
models and serialized graphs.

For the two-argument molar-mass form, G²RINS uses a zero-truncated Poisson
law on strictly positive repeat-unit counts. In other words, the count variable
is modeled as

```
N | N > 0 ~ Poisson(lambda),
M = q N,
```

with `q` chosen so that the target distribution has the requested `Mn` and
`Mw`. The zero-count event is excluded from the realized chain population
because a chemically meaningful polymer chain must have positive mass; in the
ensemble engine that boundary case is interpreted as an immediate termination or
molecular boundary event rather than as a real chain mass.

This makes the mass-form Poisson model physically consistent with a polymer
population while preserving the target-moment definitions. The zero-truncated
Poisson law naturally spans only a narrow dispersity range,
`1 <= Mw/Mn <= 1.298...`; for broader dispersity, prefer the Flory–Schulz,
Schulz–Zimm, or log-normal forms.

Sampled targets and realized molecular masses need not be identical. Repeat-unit
granularity, end groups, topology, and stochastic boundary selection still
affect the constructed polymer ensemble.

Fixed-size sampling remains the default. To sample batches until cumulative
Mn, Mw, and bond-contact frequencies stabilize, opt in explicitly:

```python
ensemble = ensemble_creator.create_ensemble_until_converged(
	batch_size=25,
	max_samples=1500,
	window=4,
	mass_tolerance=0.01,
	contact_tolerance=0.01,
	output_format="smiles",
	seed=7,
)

print(ensemble.converged, len(ensemble.chains), ensemble.convergence_trace)
```

Convergence requires every adjacent cumulative snapshot in the trailing
window to satisfy both tolerances. If the hard sample limit is reached first,
the result is returned with `converged == False`.
Both convergence thresholds (`mass_tolerance`, `contact_tolerance`) and the
maximum number of generated chains (`max_samples`) are user-configurable.

SMILES output defaults to `smiles_policy="fast"`, producing deterministic
non-canonical text without canonical ranking. This is substantially faster for
large polymers and retains the complete molecular structure for later parsing,
selection, or canonicalization. Use `smiles_policy="auto"` to request canonical
serialization with deterministic recovery for large ring-rich molecules, or
`smiles_policy="canonical"` to require strict canonical output and disable
overflow fallbacks.

For polymers without an initiator, pass
`use_repeat_units_as_source=True` to seed each iteratively generated chain
from a repeat unit.

For long runs, statistics can cover every accepted chain while only a
feature-diverse representative set is converted and retained. A recommended
starting distance is `0.15`:

```python
checkpoints = []
ensemble = ensemble_creator.create_ensemble_until_converged(
	batch_size=25,
	max_samples=100_000,
	output_format="smiles",
	seed=7,
	representative_distance=0.15,
	retain_sequences=False,
	checkpoint_callback=checkpoints.append,
)

# Counts are positionally aligned with the representative SMILES.
for smiles, count in zip(ensemble.chains, ensemble.representative_counts):
	print(count, smiles)

# Resume from a serialized batch-boundary checkpoint.
ensemble = ensemble_creator.create_ensemble_until_converged(
	batch_size=25,
	max_samples=100_000,
	output_format="smiles",
	seed=7,
	representative_distance=0.15,
	retain_sequences=False,
	checkpoint=checkpoints[-1],
)
```

The distance is the maximum of four dimensionless differences: log molecular
weight, log building-block count, total-variation distance between
building-block compositions, and total-variation distance between contact
frequencies. Thus `0.15` allows roughly a 16% size ratio and at most 0.15
total-variation distance in either normalized distribution. Every accepted
chain is within the threshold of the representative whose count it increments.
Selection is deterministic and online, but the resulting cover is not
guaranteed to be the smallest possible cover.

Set both `retain_chains=False` and `retain_sequences=False` for no retained
sample records. `metadata=False` omits returned unit/contact metadata without
changing the statistics used for convergence. Leave
`representative_distance=None` to retain every sample. Alternatively,
`reservoir_size=100` enables the previous fixed-size uniform reservoir;
`reservoir_size` and `representative_distance` cannot be combined. Reservoir
retention uses an independent random stream, so it never changes generated
chemistry, but it does not provide representative population counts.
Checkpoints are serializable with `pickle` and require an integer `seed` for
exact resume. The default `checkpoint_policy="full"` embeds retained chains
and sequences for exact output reconstruction. For much smaller checkpoints,
pass `checkpoint_policy="statistics"` to both the original and resumed calls.
This preserves convergence counters, aggregate metadata, history, and
reservoir RNG state, but a resumed result deliberately contains no retained
chains, sequences, or per-chain molecular weights from that run.

For diagnosing rare native-library failures, pass
`native_diagnostics_path="rdkit-state.jsonl"` to either ensemble creation
method. Immediately before each RDKit build, sanitization, property-cache,
descriptor, or SMILES operation, G²RINS durably records the chain index and
seed, atom and bond counts, process ID, operation stage, and library versions.
The last JSON line for a failed worker identifies its last native stage.
Python fault handling is enabled automatically, and large-molecule RDKit
operations run with enlarged-stack protection.

Parallel sampling initializes one reusable `EnsembleCreator` per worker and
submits only compact chain-index/seed jobs. At most twice the worker count is
queued at once, numerical libraries are restricted to one native thread per
process, and workers are periodically recycled where the Python runtime
supports it. A broken process pool is rebuilt without dropping completed
ordered results. `max_worker_restarts` controls the restart budget (default
`2`); exhaustion raises `WorkerProcessFailure`, whose `native_state` contains
the latest valid record from `native_diagnostics_path` when available.

### Stereochemistry

G2RINS preserves stereochemistry metadata through generative-graph export,
sampling, and RDKit conversion:

- Chiral bracket-atom tokens are carried as `atom_chiral_token`.
- Directional bond tokens `/` and `\\` are carried as `bond_symbol_raw`.
- `mol_graph_to_rdkit_mol` applies these tokens before RDKit stereo assignment.

Currently supported atom-chirality tokens are `@`, `@@`, `@TH1`, and `@TH2`.
Unsupported chirality tokens are left unset and emit a `RuntimeWarning`.

For directional markers around double bonds, incomplete, ambiguous, or
conflicting marker patterns emit a `RuntimeWarning`.
When markers are ambiguous or conflicting, directional stereoinformation is
discarded and E/Z stereochemistry is intentionally left unspecified instead
of being guessed.

Worked examples are in [`G2RINS_guide.ipynb`](G2RINS_guide.ipynb).

---

## Heritage

G²RINS is the evolution of **G-BigSMILES** ([latest repository](https://github.com/gervasiozaldivar/G-BigSMILES), [original repository](https://github.com/InnocentBug/G-BigSMILES)), which extends the [BigSMILES line notation](https://olsenlabmit.github.io/BigSMILES/docs/line_notation.html):

> Schneider, Walsh, Olsen, de Pablo, _Generative BigSMILES: an extension for polymer informatics, computer simulations & ML/AI_, Digital Discovery **3**, 51–61 (2024). [doi:10.1039/D3DD00147D](https://doi.org/10.1039/D3DD00147D)

A publication describing G²RINS is in preparation.

---

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and feature requests go through the [issue tracker](https://github.com/depablolab/g2rins/issues); notable changes are tracked in [CHANGELOG.md](CHANGELOG.md) and on the [releases page](https://github.com/depablolab/g2rins/releases).

---

## Citing

If you use G²RINS in your work, please cite the software using the metadata in [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button). The citation for the publication describing G²RINS will be added here once it is available.

---

## License

[GPL-3.0](LICENSE)
