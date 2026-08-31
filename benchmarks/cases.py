# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Seeded polymer cases shared by performance and behavioral references."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingCase:
    g2rins: str
    seed: int = 0
    reference: bool = True
    description: str = ""


CASES = {
    "small-linear": SamplingCase(
        "C{[>][<]CC(C)[>];;[<]}|poisson(900)|[H]",
        description="Small linear polypropylene-like chain.",
    ),
    "branched": SamplingCase(
        "{[] [<]CCN([>])[>]; [<][H]; O[>], [<][H] []}|poisson(1200)|",
        description="Hyperbranched AB2 poly(ethyleneimine).",
    ),
    "nested": SamplingCase(
        "{[] [<]CC({[<] [<]NN({[<] [<]C(C)O[>];; [>]}|poisson(80)|[H])[>];; [>]}|poisson(200)|[H])CC[>]; [<][H]; []}|poisson(800)|",
        description="Three-level nested stochastic object.",
    ),
    "grafted": SamplingCase(
        "{[] [<1]CC(C(=O)O)[>1]; {[] [<]C(C)(C(=O)OCCOC(=O)C(C)(C)[>1])C[>0]; [H][>]; [<]Br [<1]}|gauss(1800, 200)|[>1]; [<1]Br []}|gauss(3500, 350)|",
        description="Nested Gaussian graft polymer.",
    ),
    "heavy-tail": SamplingCase(
        "{[] [<]C(=O)CCCCC(=O)[<], [>]NCCCCCCN[>]; [<][H]; [>]O []}|flory_schulz(0.02)|",
        description="Seeded Flory-Schulz chain with unbounded support.",
    ),
    "large-linear": SamplingCase(
        "C{[>][<]CC(C)[>];;[<]}|poisson(120000)|[H]",
        reference=False,
        description="Large exact-rounding checkpoint stress case.",
    ),
    "large-heavy-tail": SamplingCase(
        "{[] [<]C(=O)CCCCC(=O)[<], [>]NCCCCCCN[>]; [<][H]; [>]O []}|flory_schulz(0.001)|",
        reference=False,
        description="Large heavy-tailed Flory-Schulz case.",
    ),
    "high-discard": SamplingCase(
        "{[] [<|9.0|]CC(C)O[>|9.0|], [<|6.0|]CC(CC)O[>|6.0|]; {[] [<|7.0|]CCO[>|7.0|], [<|4.0|]CC(CC)O[>|4.0|]; CCCCO[>]; [<] []}|gauss(680.0, 215.0)|[>]; [<][H] []}|gauss(1649.0, 521.5)|",
        reference=False,
        description="Known truncating architecture for discard-path accounting.",
    ),
}

REFERENCE_CASES = tuple(name for name, case in CASES.items() if case.reference)
