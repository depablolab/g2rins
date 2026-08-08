# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only


"""
Implementation of G2RINS, a line-and-graph representation of polymers.

Grammar validation, generative graph construction, and molecular ensemble generation.
"""

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError as exc:
    raise RuntimeError("Please make sure to install this module correctly via setuptools with setuptools_scm activated to generate a `_version.py` file.") from exc

from .atom import (
    AliphaticOrganic,
    AromaticOrganic,
    AromaticSymbol,
    Atom,
    AtomCharge,
    AtomClass,
    AtomSymbol,
    BracketAtom,
    Chiral,
    HCount,
    Isotope,
)
from .g2rins_molecule import G2rins, G2rinsMolecule, DotGeneration, DotSystemSize
from .bond import (
    BondConnector,
    BondConnectorGeneration,
    BondConnectorList,
    BondConnectorSymbol,
    BondConnectorSymbolIdx,
    BondSymbol,
    InnerBondConnector,
    RingBond,
    SimpleBondConnector,
    TerminalBondConnector,
    TerminalBondConnectorList,
)
from .core import G2rinsBase
from .distribution import (
    FlorySchulz,
    Gauss,
    LogNormal,
    Poisson,
    SchulzZimm,
    StochasticDistribution,
    Uniform,
)
from .ensemble_creator import EnsembleCreator
from .generative_graph import GraphCreator, UnitLabels, derive_unit_labels, generative_graph_json_data
from .nx_rdkit_mol import mol_graph_to_rdkit_mol, mol_graph_to_smiles, rdkit_mol_to_smiles
from .parser import get_global_parser
from .smiles import (
    AtomAssembly,
    Branch,
    BranchedAtom,
    Counterion,
    CounterionAssembly,
    CounterionBranchedAtom,
    Dot,
    MolarAmount,
    Smiles,
)
from .stochastic import StochasticObject
from .transformer import G2RINSTransformer, get_global_transformer
from .util import camel_to_snake, get_global_rng, snake_to_camel

__all__ = [
    "__version__",
    "version_tuple",
    "Atom",
    "BracketAtom",
    "Isotope",
    "AtomSymbol",
    "Chiral",
    "HCount",
    "AtomCharge",
    "AtomClass",
    "AromaticSymbol",
    "AliphaticOrganic",
    "AromaticOrganic",
    "BondSymbol",
    "RingBond",
    "BondConnectorSymbol",
    "BondConnectorSymbolIdx",
    "BondConnectorGeneration",
    "InnerBondConnector",
    "BondConnector",
    "SimpleBondConnector",
    "TerminalBondConnector",
    "TerminalBondConnectorList",
    "BondConnectorList",
    "G2rinsBase",
    "camel_to_snake",
    "snake_to_camel",
    "get_global_rng",
    "G2RINSTransformer",
    "get_global_transformer",
    "get_global_parser",
    "FlorySchulz",
    "SchulzZimm",
    "Gauss",
    "LogNormal",
    "Poisson",
    "StochasticDistribution",
    "StochasticObject",
    "Uniform",
    "Branch",
    "BranchedAtom",
    "AtomAssembly",
    "Counterion",
    "CounterionAssembly",
    "CounterionBranchedAtom",
    "Dot",
    "MolarAmount",
    "Smiles",
    "G2rins",
    "G2rinsMolecule",
    "DotGeneration",
    "DotSystemSize",
    "mol_graph_to_rdkit_mol",
    "mol_graph_to_smiles",
    "rdkit_mol_to_smiles",
    "UnitLabels",
    "derive_unit_labels",
    "generative_graph_json_data",
    "GraphCreator",
    "EnsembleCreator",
]
