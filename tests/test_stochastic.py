# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import pytest

import g2rins

# trunk-disable-all(cspell/error)
# trunk-ignore-all(bandit/B101)

test_args = [
    (
        "{[]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H]; [<][H] []}|gauss(1500, 50)|",
        "{[] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H]; [<][H] []}",
        "{[] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H]; [<][H] []}|gauss(1500.0, 50.0)|",
    ),
    (
        "{[]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H]; [<][H][]}|schulz_zimm(4500, 3500)|",
        "{[] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H]; [<][H] []}",
        "{[] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H]; [<][H] []}|schulz_zimm(4500.0, 3500.0)|",
    ),
    (
        "{[>][<]CCO[>|0.8 0. 0.2 0.|],[<]CC(O[>|0.2 0. 0.8 0.|])C;;[<][H][]}|gauss(1500.0, 50.0)|",
        "{[>] [<]CCO[>], [<]CC(O[>])C; ; [<][H] []}",
        "{[>] [<]CCO[>|0.8 0.0 0.2 0.0|], [<]CC(O[>|0.2 0.0 0.8 0.0|])C; ; [<][H] []}|gauss(1500.0, 50.0)|",
    ),
    (
        "{[>][<]CCO[>|0.8 0. 0.2 0.|],[<]CC(O[>|0.2 0. 0.8 0.|])C;;[<][H][]}|poisson(300)|",
        "{[>] [<]CCO[>], [<]CC(O[>])C; ; [<][H] []}",
        "{[>] [<]CCO[>|0.8 0.0 0.2 0.0|], [<]CC(O[>|0.2 0.0 0.8 0.0|])C; ; [<][H] []}|poisson(300.0)|",
    ),
    (
        "{[][<,<1]CCN([>1])[>]; N([>])([>])[>1];[<]|[<1]}|flory_schulz(0.0011)|",
        "{[] [<, <1]CCN([>1])[>]; N([>])([>])[>1];  [<]|[<1]}",
        "{[] [<, <1]CCN([>1])[>]; N([>])([>])[>1];  [<]|[<1]}|flory_schulz(0.0011)|",
    ),
    (
        "{[][<,<1]CCN([>1])[>]; N([>])([>])[>1];[<]|[<1]}|flory_schulz(9e-4)|",
        "{[] [<, <1]CCN([>1])[>]; N([>])([>])[>1];  [<]|[<1]}",
        "{[] [<, <1]CCN([>1])[>]; N([>])([>])[>1];  [<]|[<1]}|flory_schulz(0.0009)|",
    ),
    (
        # Counterions on a repeat unit AND on an end group: both re-emit right
        # after their residue in the regenerated string.
        "{[][<]CC(C[NH3+])[>].[Cl-]; C[NH3+][>].[Cl-]; [<][H][]}|poisson(500)|",
        "{[] [<]CC(C[NH3+])[>].[Cl-]; C[NH3+][>].[Cl-]; [<][H] []}",
        "{[] [<]CC(C[NH3+])[>].[Cl-]; C[NH3+][>].[Cl-]; [<][H] []}|poisson(500.0)|",
    ),
]


@pytest.mark.parametrize(("text", "big", "ref"), test_args)
def test_stochastic(text, big, ref):
    stochastic = g2rins.StochasticObject.make(text)

    assert str(stochastic) == ref
    assert stochastic.generate_string(False) == big

    stochastic.get_graph_creator()


if __name__ == "__main__":
    test_stochastic()
