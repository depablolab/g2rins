# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import pytest

import g2rins

# TODO: add nested structures

test_args = [
    (
        "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F;; [<]}|gauss(1000,100)|CC{[>] [<]CC([>])c1ccccc1;;[<]}|gauss(500,50)|C(C)CC(c1ccccc1)c1ccccc1",
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F; ;  [<]}CC{[>] [<]CC([>])c1ccccc1; ;  [<]}C(C)CC(c1ccccc1)c1ccccc1",
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F; ;  [<]}|gauss(1000.0, 100.0)|CC{[>] [<]CC([>])c1ccccc1; ;  [<]}|gauss(500.0, 50.0)|C(C)CC(c1ccccc1)c1ccccc1",
    ),
    (
        "{[] CC([$])=NCC[$]; [H][$];[H][$][]}|schulz_zimm(1000, 900)|",
        "{[] CC([$])=NCC[$]; [H][$]; [H][$] []}",
        "{[] CC([$])=NCC[$]; [H][$]; [H][$] []}|schulz_zimm(1000.0, 900.0)|",
    ),
    (
        "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F;;[<]}|schulz_zimm(1500, 1400)|CC.|60000|",
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F; ;  [<]}CC.",
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F; ;  [<]}|schulz_zimm(1500.0, 1400.0)|CC.|60000.0|",
    ),
    (
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F;;[<]}|schulz_zimm(1500, 1000)|CC{[>][>]CC([<])c1ccccc1;;[<]}|schulz_zimm(1500, 1000)|C(C)CC(c1ccccc1)c1ccccc1.|60000|",
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F; ;  [<]}CC{[>] [>]CC([<])c1ccccc1; ;  [<]}C(C)CC(c1ccccc1)c1ccccc1.",
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F; ;  [<]}|schulz_zimm(1500.0, 1000.0)|CC{[>] [>]CC([<])c1ccccc1; ;  [<]}|schulz_zimm(1500.0, 1000.0)|C(C)CC(c1ccccc1)c1ccccc1.|60000.0|",
    ),
    (
        "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F; ;[<]}|gauss(1000,10)|CC{[>][<]CC([>])c1ccccc1;;[<]}|gauss(500, 10)|C(C)CC(c1ccccc1)c1ccccc1.|60000|",
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F; ;  [<]}CC{[>] [<]CC([>])c1ccccc1; ;  [<]}C(C)CC(c1ccccc1)c1ccccc1.",
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F; ;  [<]}|gauss(1000.0, 10.0)|CC{[>] [<]CC([>])c1ccccc1; ;  [<]}|gauss(500.0, 10.0)|C(C)CC(c1ccccc1)c1ccccc1.|60000.0|",
    ),
    (
        "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F;;[<]}|schulz_zimm(1500, 1000)|CC{[>][<]CC([>])c1ccccc1;;[<]}|schulz_zimm(1500, 1000)|C(C)CC(c1ccccc1)c1ccccc1.|5000|",
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F; ;  [<]}CC{[>] [<]CC([>])c1ccccc1; ;  [<]}C(C)CC(c1ccccc1)c1ccccc1.",
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F; ;  [<]}|schulz_zimm(1500.0, 1000.0)|CC{[>] [<]CC([>])c1ccccc1; ;  [<]}|schulz_zimm(1500.0, 1000.0)|C(C)CC(c1ccccc1)c1ccccc1.|5000.0|",
    ),
    (
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F ;;[<]}|schulz_zimm(1000, 950)|CC{[>][<]CC([>])c1ccccc1;;[<]}|schulz_zimm(500, 400)|C(C)CC(c1ccccc1)c1ccccc1.|5e7|",
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; ;  [<]}CC{[>] [<]CC([>])c1ccccc1; ;  [<]}C(C)CC(c1ccccc1)c1ccccc1.",
        "[H]{[>] CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; ;  [<]}|schulz_zimm(1000.0, 950.0)|CC{[>] [<]CC([>])c1ccccc1; ;  [<]}|schulz_zimm(500.0, 400.0)|C(C)CC(c1ccccc1)c1ccccc1.|50000000.0|",
    ),
    (
        "{[]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N];[<]}|schulz_zimm(1000, 450)|{[>][<]CC([>])c1ccccc1;; [<][H][]}|schulz_zimm(400, 300)|.|5e7|",
        "{[] CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N];  [<]}{[>] [<]CC([>])c1ccccc1; ; [<][H] []}.",
        "{[] CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N];  [<]}|schulz_zimm(1000.0, 450.0)|{[>] [<]CC([>])c1ccccc1; ; [<][H] []}|schulz_zimm(400.0, 300.0)|.|50000000.0|",
    ),
    (
        # A counterion trailing the repeat unit of a stochastic object embedded
        # in a full molecule line survives parsing and both regenerated forms.
        "[H]{[>][<]CC(C[NH3+])[>].[Cl-];;[<]}|poisson(400)|C.|60000|",
        "[H]{[>] [<]CC(C[NH3+])[>].[Cl-]; ;  [<]}C.",
        "[H]{[>] [<]CC(C[NH3+])[>].[Cl-]; ;  [<]}|poisson(400.0)|C.|60000.0|",
    ),
]


@pytest.mark.parametrize(("text", "big", "ref"), test_args)
def test_molecule(text, big, ref):
    g2rins_object = g2rins.G2rins.make(text)
    assert str(g2rins_object) == ref
    assert g2rins_object.generate_string(False) == big

    graph_creator = g2rins_object.get_graph_creator()
    graph_creator.get_generative_graph(include_bond_connectors=False)
    graph_creator.get_generative_graph(include_bond_connectors=True)


if __name__ == "__main__":
    test_molecule()
