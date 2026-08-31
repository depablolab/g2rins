# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Convergence tracking for incrementally sampled molecular ensembles."""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class ConvergenceTracker:
    """Track stability of cumulative Mn, Mw, and contact frequencies.

    ``window`` adjacent batch-to-batch transitions must all satisfy both
    tolerances. Consequently, convergence can first be declared after
    ``window + 1`` batches.
    """

    window: int = 4
    mass_tolerance: float = 0.002
    contact_tolerance: float = 0.01
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.window < 1:
            raise ValueError(f"window must be positive, got {self.window}.")
        if self.mass_tolerance < 0:
            raise ValueError(
                f"mass_tolerance must be non-negative, got {self.mass_tolerance}."
            )
        if self.contact_tolerance < 0:
            raise ValueError(
                "contact_tolerance must be non-negative, "
                f"got {self.contact_tolerance}."
            )

    def record(
        self,
        n_samples: int,
        mn: float,
        mw: float,
        contact_frequencies: Mapping[str, float],
    ):
        """Append one cumulative convergence snapshot."""
        self.history.append(
            {
                "n_samples": n_samples,
                "mn": mn,
                "mw": mw,
                "contact_frequencies": dict(contact_frequencies),
            }
        )

    def deltas(self):
        """Maximum adjacent deltas in the trailing window, or ``None``."""
        if len(self.history) <= self.window:
            return None
        mass_relative = 0.0
        contact_delta = 0.0
        trailing = self.history[-self.window - 1 :]
        for previous, current in zip(trailing, trailing[1:]):
            mass_relative = max(
                mass_relative,
                abs(current["mn"] - previous["mn"])
                / max(abs(previous["mn"]), 1e-9),
                abs(current["mw"] - previous["mw"])
                / max(abs(previous["mw"]), 1e-9),
            )
            contact_keys = set(previous["contact_frequencies"]) | set(
                current["contact_frequencies"]
            )
            for key in contact_keys:
                contact_delta = max(
                    contact_delta,
                    abs(
                        current["contact_frequencies"].get(key, 0.0)
                        - previous["contact_frequencies"].get(key, 0.0)
                    ),
                )
        return {
            "mass_relative": mass_relative,
            "contact_delta": contact_delta,
        }

    def converged(self):
        """Whether every transition in the trailing window is stable."""
        deltas = self.deltas()
        return deltas is not None and (
            deltas["mass_relative"] <= self.mass_tolerance
            and deltas["contact_delta"] <= self.contact_tolerance
        )

    def progress(self):
        """Return a concise human-readable convergence status."""
        deltas = self.deltas()
        if deltas is None:
            remaining = self.window - len(self.history) + 1
            return (
                "warming up window "
                f"({remaining} more batch(es) before convergence can be checked)"
            )
        return (
            f"mass_delta={deltas['mass_relative']:.4f}/"
            f"{self.mass_tolerance:.4f} "
            f"contact_delta={deltas['contact_delta']:.4f}/"
            f"{self.contact_tolerance:.4f}"
        )