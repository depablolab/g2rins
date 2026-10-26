# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Internal conversion configuration, never part of worker/checkpoint payloads."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class _ConversionMode(Enum):
    IMMEDIATE = False
    DEFERRED = True
    MATERIALIZE = "materialize"

    @classmethod
    def from_legacy(cls, value):
        """Name existing wire values without tightening legacy truthiness rules."""
        if isinstance(value, cls):
            return value
        if not value:
            return cls.IMMEDIATE
        return cls.MATERIALIZE if value == "materialize" else cls.DEFERRED


@dataclass(frozen=True, slots=True)
class _ChainConversionOptions:
    """One immutable record per iterator/batch, not per growth step or chain."""

    molecule_format: str
    collect_info: Any
    include_sequences: bool
    native_diagnostics_path: Any
    mode: _ConversionMode
    strip_unresolved_directional_markers: bool
    smiles_policy: str