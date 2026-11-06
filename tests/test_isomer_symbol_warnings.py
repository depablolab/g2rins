# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""The two warnings that say isotope, chirality and cis/trans symbols are parsed but not yet encoded in the generative graph
nor applied during generation. This file is deleted together with the warnings once those features land."""

import warnings

import pytest

import g2rins
from g2rins.exception import IsomerSymbolsIgnoredInGeneration, IsomerSymbolsNotInGraph

CIS_ALKENE = r"{[] [<]/C=C\[>]; [H][>]; [<][H] []}|gauss(400,10)|"
ISOMER_WARNINGS = (IsomerSymbolsNotInGraph, IsomerSymbolsIgnoredInGeneration)


def _isomer_warnings(func):
    """The isomer-symbol warnings `func()` emits, in order."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        func()
    return [w.message for w in caught if isinstance(w.message, ISOMER_WARNINGS)]


@pytest.mark.parametrize(
    ("text", "kinds"),
    [
        ("[13CH3]C", ["isotope"]),
        ("C[C@H](F)Cl", ["chirality"]),
        (r"C/C=C\C", ["cis/trans"]),
        (r"C/1CCCCC/C=C1", ["cis/trans"]),
        (CIS_ALKENE, ["cis/trans"]),
        ("[H]{[>][<]C[13C@H](C)[>];;[<]}|uniform(170,170)|[H]", ["isotope", "chirality"]),
        (r"[2H]/C=C\[C@H](F)Cl", ["isotope", "chirality", "cis/trans"]),
    ],
)
def test_the_graph_warning_names_the_written_kinds(text, kinds):
    creator = g2rins.G2rins.make(text).get_graph_creator()
    emitted = _isomer_warnings(creator.get_generative_graph)
    assert [(type(w), w.token) for w in emitted] == [(IsomerSymbolsNotInGraph, kinds)]
    assert all(kind in str(emitted[0]) for kind in kinds)
    assert "available soon" in str(emitted[0])


def test_the_ensemble_creator_warns_once_about_generation_only():
    creator = g2rins.G2rins.make(CIS_ALKENE).get_graph_creator()
    emitted = _isomer_warnings(creator.get_ensemble_creator)
    assert [(type(w), w.token) for w in emitted] == [(IsomerSymbolsIgnoredInGeneration, ["cis/trans"])]
    assert "available soon" in str(emitted[0])


@pytest.mark.parametrize("text", ["CCO", "[CH3]C", "C=1CCCCC1", "{[] [<]C=C[>]; [H][>]; [<][H] []}|gauss(400,10)|"])
def test_strings_without_isomer_symbols_warn_nothing(text):
    creator = g2rins.G2rins.make(text).get_graph_creator()
    assert _isomer_warnings(creator.get_generative_graph) == []
    assert _isomer_warnings(creator.get_ensemble_creator) == []
