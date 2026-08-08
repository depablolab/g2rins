# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import networkx as nx

import g2rins

# graph_validation.json holds nx.adjacency_data of get_generative_graph(include_bond_connectors=False)
# for every tests/smi.json g2rins string; regenerate it whenever the generative-graph
# schema deliberately changes (see dev-clean 2026-07 regeneration).


def node_match(node1_attrs, node2_attrs):
    return node1_attrs == node2_attrs


# Define a custom edge matcher
def edge_match(edge1_attrs, edge2_attrs):
    return edge1_attrs == edge2_attrs


def test_generative_graph_generation(graph_validation_dict):
    for g2rins_string in graph_validation_dict:
        print(g2rins_string)
        graph_creator = g2rins.G2rins.make(g2rins_string).get_graph_creator()
        generative_graph = graph_creator.get_generative_graph(include_bond_connectors=False)
        assert nx.is_isomorphic(generative_graph, graph_validation_dict[g2rins_string], node_match=node_match, edge_match=edge_match)

        dot_string_A = graph_creator.get_dot_string(include_bond_connectors=True)
        dot_string_B = graph_creator.get_dot_string(include_bond_connectors=False, node_prefix="bc-")
        dot_string_A = dot_string_A[: dot_string_A.rfind("}")]
        dot_string_B = dot_string_B[len("digraph{") :]
        dot_string = dot_string_A + dot_string_B
        assert len(dot_string) > 0


def test_generative_graph_json_data_format_block():
    smi = "{[] [<]CC([>])c1ccccc1; [>][H]; [<][H] []}|gauss(1000, 45)|"
    generative_graph = g2rins.G2rins.make(smi).get_graph_creator().get_generative_graph(include_bond_connectors=False)
    data = g2rins.generative_graph_json_data(generative_graph)

    assert data["format"]["version"] == 1
    declared = set(data["format"]["derived_node_fields"])
    stored_attrs = {key for _node, attrs in generative_graph.nodes(data=True) for key in attrs}
    injected = {key for node_dict in data["graph"]["nodes"] for key in node_dict} - stored_attrs - {"id"}
    # The declared derived fields are exactly what the writer injects, and they
    # are annotations only -- never stored on the graph itself.
    assert injected == declared
    assert not declared & stored_attrs

    labels = g2rins.derive_unit_labels(generative_graph)
    for node_dict in data["graph"]["nodes"]:
        assert node_dict["unit_id"] == labels.unit_id[node_dict["id"]]
        assert node_dict.get("bond_id") == labels.bond_id.get(node_dict["id"])
