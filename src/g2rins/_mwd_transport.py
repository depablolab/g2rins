"""Small wire-version guard for direct Gamma target-mass distributions."""

import numbers

import numpy as np

from .distribution import Mwd, StochasticDistribution

MWD_SCHEMA = "g2rins-distribution-vector/v4"


def validate_mwd_transport(graph):
    """Reject retired NB data and validate replicated Gamma distribution slots.

    Legacy 10/12-wide graphs keep their historical decoding path. New MWD
    graphs have no recipe context, only an explicit distribution version.
    """
    schema = graph.graph.get("molecular_weight_distribution_schema")
    if "nb_recipe_context" in graph.graph or schema == "g2rins-distribution-vector/v3":
        raise ValueError("Retired NB graph is incompatible with Gamma MWD; recreate it from a reviewed recipe.")
    if schema not in (None, MWD_SCHEMA):
        raise ValueError(f"Unsupported molecular-weight distribution schema: {schema!r}")
    reference = None
    has_mwd = False
    checked = set()
    for _, data in graph.nodes(data=True):
        raw = data.get("molecular_weight_distribution", ())
        if id(raw) in checked:
            continue
        checked.add(id(raw))
        widths = {len(vector) for vector in raw}
        if 14 in widths:
            raise ValueError("Retired 14-wide NB vectors are incompatible with Gamma MWD.")
        if schema is None:
            if widths - {10, 12}:
                raise ValueError("Extended MWD vectors require the Gamma v4 schema.")
            continue
        if len(raw) != 10 or widths != {16}:
            raise ValueError("Gamma v4 requires ten common-width 16 distribution slots.")
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real)
               for vector in raw for x in vector):
            raise ValueError("MWD vectors must contain finite real numbers, not booleans.")
        vectors = np.asarray(raw, dtype=float)
        if not np.isfinite(vectors).all():
            raise ValueError("MWD vectors must be finite.")
        if reference is None:
            reference = vectors
            distributions = [StochasticDistribution.from_serial_vector(v) for v in raw]
            has_mwd = any(isinstance(distribution, Mwd) for distribution in distributions)
        elif not np.array_equal(reference, vectors):
            raise ValueError("Replicated molecular_weight_distribution arrays disagree.")
    if schema == MWD_SCHEMA and not has_mwd:
        raise ValueError("Gamma v4 requires a direct MWD declaration.")