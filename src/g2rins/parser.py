# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

from importlib.resources import files

from lark import Lark


_DEFAULT_START_TOKENS = (
    "g2rins",
    "g2rins_molecule",
    "stochastic_object",
    "atom",
    "bond_symbol",
    "ring_bond",
    "aliphatic_organic",
    "aromatic_organic",
    "branch",
    "smiles",
    "molar_amount",
    "bond_connector_symbol",
    "bond_connector_generation",
    "ladder_bond_connector",
    "non_covalent_bond_connector",
    "bond_connector",
    "simple_bond_connector",
    "terminal_bond_connector",
    "terminal_bond_connector_list",
    "bond_connector_list",
    "stochastic_generation",
    "stochastic_distribution",
    "dot_system_size",
    "dot_generation",
    "dot",
    "counterion",
    "isotope",
    "atom_symbol",
    "aromatic_symbol",
    "bracket_atom",
    "flory_schulz",
    "uniform",
    "schulz_zimm",
    "log_normal",
    "gauss",
    "poisson",
)


def _make_parser(filename=None, start_tokens=None):
    if filename is None:
        # Package resources are Traversables, not necessarily filesystem paths.
        grammar_text = files("g2rins").joinpath("data").joinpath("g2rins.lark").read_text(encoding="utf-8")
    else:
        with open(filename, "r", encoding="utf-8") as file_handle:
            grammar_text = file_handle.read()

    if start_tokens is None:
        start_tokens = list(_DEFAULT_START_TOKENS)
    parser = Lark(rf"{grammar_text}", start=start_tokens, keep_all_tokens=True)
    return parser


_GLOBAL_PARSER: None | Lark = None


def get_global_parser():
    global _GLOBAL_PARSER
    if _GLOBAL_PARSER is None:
        _GLOBAL_PARSER = _make_parser()

    return _GLOBAL_PARSER
