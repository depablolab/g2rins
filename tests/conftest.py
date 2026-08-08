import json
import os
from importlib.resources import files

import lark
import networkx as nx
import pytest

# TODO: add nested examples to smi.json


@pytest.fixture(scope="session")
def smi_dict():
    path = os.path.dirname(__file__)
    with open(os.path.join(path, "smi.json"), "r") as file_handle:
        data = json.load(file_handle)
    return data


@pytest.fixture(scope="session")
def graph_validation_dict():
    path = os.path.dirname(__file__)
    with open(os.path.join(path, "graph_validation.json"), "r") as file_handle:
        raw_data = json.load(file_handle)
    data = {}
    for string in raw_data:
        data[string] = nx.adjacency_graph(raw_data[string])
    return data


@pytest.fixture(scope="session")
def chembl_smi_list(smi_dict):
    return smi_dict["chembl_smiles"]


@pytest.fixture(scope="session")
def g2rins_list(smi_dict):
    return smi_dict["g2rins"]


@pytest.fixture(scope="session")
def invalid_g2rins_list(smi_dict):
    return smi_dict["invalid_g2rins"]


@pytest.fixture(scope="session")
def big_smiles_features_unsupported(smi_dict):
    return smi_dict["big_smiles_features_unsupported_by_g2rins"]


@pytest.fixture(scope="session")
def big_smiles_features_to_rewrite(smi_dict):
    return smi_dict["todo_big_smiles_features_to_rewrite"]


@pytest.fixture(scope="session")
def grammar_text():
    grammar_file = files("g2rins").joinpath("data", "g2rins.lark")
    with open(grammar_file, "r") as file_handle:
        grammar_text = file_handle.read()
    return grammar_text


@pytest.fixture(scope="session")
def grammar_parser(grammar_text):
    parser = lark.Lark(rf"{grammar_text}", start="g2rins")
    return parser
