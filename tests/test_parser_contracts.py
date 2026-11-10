# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

from pathlib import Path
from zipfile import Path as ZipPath, ZipFile

import pytest

from g2rins import parser as parser_module


DEFAULT_ENTRYPOINTS = [
    ("g2rins", "C"),
    ("g2rins_molecule", "C"),
    ("stochastic_object", "{[$] [$]CC[$]; ; [$]}"),
    ("atom", "[13CH3+]"),
    ("bond_symbol", "="),
    ("ring_bond", "%12"),
    ("aliphatic_organic", "Cl"),
    ("aromatic_organic", "c"),
    ("branch", "(=O)"),
    ("smiles", "CC(=O)O"),
    ("molar_amount", "|2|"),
    ("bond_connector_symbol", "$"),
    ("bond_connector_generation", "|1 2|"),
    ("ladder_bond_connector", "[$[$]]"),
    ("non_covalent_bond_connector", "[$:1]"),
    ("bond_connector", "[$]"),
    ("simple_bond_connector", "[<1]"),
    ("terminal_bond_connector", "[]"),
    ("terminal_bond_connector_list", "[<]|[>1]"),
    ("bond_connector_list", "[$]|[>1]"),
    ("stochastic_generation", "|gauss(100, 10)|"),
    ("stochastic_distribution", "poisson(10)"),
    ("dot_system_size", "|2|"),
    ("dot_generation", ".|2|"),
    ("dot", "."),
    ("counterion", ".[Na+]"),
    ("isotope", "13"),
    ("atom_symbol", "Cl"),
    ("aromatic_symbol", "se"),
    ("bracket_atom", "[13CH3+]"),
    ("flory_schulz", "flory_schulz(0.5)"),
    ("uniform", "uniform(10, 20)"),
    ("schulz_zimm", "schulz_zimm(100, 50)"),
    ("log_normal", "log_normal(100, 50)"),
    ("gauss", "gauss(100, 10)"),
    ("poisson", "poisson(10)"),
]


@pytest.fixture(scope="module")
def default_parser():
    return parser_module._make_parser()


def test_default_start_tokens_are_immutable_and_unchanged(default_parser):
    expected = tuple(start for start, _text in DEFAULT_ENTRYPOINTS)

    assert type(parser_module._DEFAULT_START_TOKENS) is tuple
    assert parser_module._DEFAULT_START_TOKENS == expected
    assert default_parser.options.start == list(expected)
    assert default_parser.options.keep_all_tokens is True


@pytest.mark.parametrize(("start", "text"), DEFAULT_ENTRYPOINTS)
def test_default_grammar_entrypoints(default_parser, start, text):
    tree = default_parser.parse(text, start=start)

    assert tree.data == start


@pytest.mark.parametrize("explicit_filename", [False, True])
@pytest.mark.parametrize("start_tokens", ["atom", ["atom"], ["atom", "inner_bond_connector"]])
def test_parser_overrides_use_actual_grammar(explicit_filename, start_tokens):
    filename = Path(parser_module.__file__).with_name("data") / "g2rins.lark" if explicit_filename else None
    parser = parser_module._make_parser(filename=filename, start_tokens=start_tokens)
    expected = [start_tokens] if isinstance(start_tokens, str) else start_tokens

    assert parser.options.start == expected
    assert parser.parse("C", start="atom").data == "atom"
    if "inner_bond_connector" in expected:
        assert parser.parse("$1", start="inner_bond_connector").data == "inner_bond_connector"
    assert parser_module._DEFAULT_START_TOKENS == tuple(start for start, _text in DEFAULT_ENTRYPOINTS)


def test_default_grammar_can_be_read_from_non_filesystem_resource(monkeypatch, tmp_path):
    grammar = Path(parser_module.__file__).with_name("data") / "g2rins.lark"
    archive = tmp_path / "grammar.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("g2rins/data/g2rins.lark", grammar.read_text(encoding="utf-8"))

    with ZipFile(archive) as handle:
        resource = ZipPath(handle, "g2rins/")
        monkeypatch.setattr(parser_module, "files", lambda _package: resource)
        parser = parser_module._make_parser(start_tokens="atom")
        assert parser.parse("[13CH3+]").data == "atom"


def test_default_grammar_supports_single_argument_resource_joinpath(monkeypatch, tmp_path):
    grammar = Path(parser_module.__file__).with_name("data") / "g2rins.lark"
    archive = tmp_path / "grammar.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("g2rins/data/g2rins.lark", grammar.read_text(encoding="utf-8"))

    children = []
    encodings = []

    class SingleArgumentResource:
        """Wrap a real resource without exposing a filesystem path."""

        def __init__(self, resource):
            self.resource = resource

        def joinpath(self, child):
            children.append(child)
            return SingleArgumentResource(self.resource.joinpath(child))

        def read_text(self, *, encoding):
            encodings.append(encoding)
            return self.resource.read_text(encoding=encoding)

    with ZipFile(archive) as handle:
        resource = SingleArgumentResource(ZipPath(handle, "g2rins/"))
        monkeypatch.setattr(parser_module, "files", lambda _package: resource)
        parser = parser_module._make_parser(start_tokens="atom")

        assert parser.parse("[13CH3+]").data == "atom"
        assert children == ["data", "g2rins.lark"]
        assert encodings == ["utf-8"]