"""Parser provenance resolves same-atom and context-inactive motif ports."""

import pytest
from g2rins import G2rins
from rdkit import Chem

from g2rins.motif_chemistry import (
    _source_descriptors,
    motif_chemical_strings,
    native_source_descriptors,
)


FAILED_MODELS = [
    pytest.param(
        "{[] [>1]CC([<1])C(=O)O|65|, [>1]CC([<1])(C)C(=O)O|35|; "
        "{[] [<]C(C)([<1|0.1|])CO[>]; CCCCCO[>]; [<][H] "
        "[<1]|[>1]}|poisson(1698.71, 1518.14)|[>1]|[<1]; [<][H] "
        "[]}|schulz_zimm(20431, 9824.52)|",
        {"R0": {1: "[>]", 2: "[<1]", 3: "[<]"}}, id="a3d5adb3",
    ),
    pytest.param(
        "{[] [>1]CC([<1])C(=O)OCC1CO1; {[] [<]C([<1|0.1|])(COC)CO[>]; "
        "{[] [>]OC([<1|0.05|])C([<1|0.05|])[<]; "
        "[>]OCC(O[>])CO[>]|0.8|, [>]OCC(CC)(CO[>])CO[>]|0.2|; "
        "CCCCCCCCCCCC[<], [<][H], [<][H], [<][H], [<][H], [<][H] "
        "[<]}|poisson(8031.43, 7138.95)|[>]; "
        "CCCCCCCCCCCC[<], [<][H], [<][H], [<][H], [<][H], [<][H] "
        "[<1]|[>1]}|poisson(8411.67, 7982.92)|[>1]|[<1]; [<][H] "
        "[]}|schulz_zimm(38387.1, 17813.9)|",
        {"R1": {1: "[>]", 2: "[<1]", 3: "[<]"},
         "R0": {1: "[>]", 2: "[<]"}}, id="24ea1add",
    ),
]


@pytest.mark.parametrize("source,expected", FAILED_MODELS)
def test_failed_models_use_exact_native_connector_provenance(source, expected):
    graph_creator = G2rins.make(source).get_graph_creator()
    creator = graph_creator.get_ensemble_creator()
    descriptors = native_source_descriptors(graph_creator.g, creator._generative_graph)
    for unit, mapping in expected.items():
        assert descriptors[unit] == mapping
        info = creator._canonical_unit_info[unit]
        # These valid models really cannot be resolved by chemistry alone.
        with pytest.raises(ValueError, match="ambiguous connector identity"):
            _source_descriptors(info["psmiles"], info["g2rins"])
        for site, descriptor in mapping.items():
            fragment = motif_chemical_strings(
                {"units": [unit], "edges": [], "boundary": [[0, site, "1", False]]},
                {unit: {"psmiles": info["psmiles"], "bigsmiles": info["g2rins"]}},
                unit_descriptors=descriptors,
            )
            assert fragment["bigsmiles"].count(descriptor) == 1
            assert "*" not in fragment["bigsmiles"]
            assert Chem.MolFromSmiles(fragment["psmiles"]) is not None
    for unit, info in creator._canonical_unit_info.items():
        assert sorted(descriptors[unit]) == info["expected_maps"]


def test_graphs_from_different_parses_cannot_supply_provenance():
    source = "{[] [<]CC[>]; C[>]; [<][H] []}|poisson(150, 140)|"
    left = G2rins.make(source).get_graph_creator()
    right = G2rins.make(source).get_graph_creator()
    native = right.get_ensemble_creator()._generative_graph
    with pytest.raises(ValueError, match="no parser provenance"):
        native_source_descriptors(left.g, native)


@pytest.mark.parametrize("mapping", [{}, {1: "[>]"}, {1: "[>]", 2: "invalid"}])
def test_incomplete_or_invalid_provenance_cannot_fall_back_to_guessing(mapping):
    with pytest.raises(ValueError, match="provenance does not match"):
        motif_chemical_strings(
            {"units": ["R0"], "edges": [], "boundary": []},
            {"R0": {"psmiles": "[*:1]CO[*:2]", "bigsmiles": "[<]CO[>]"}},
            unit_descriptors={"R0": mapping},
        )