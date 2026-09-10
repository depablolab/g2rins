# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""
This module defines base classes for handling stochastic generation based on
various statistical distributions.
"""
import math
import re
from abc import abstractmethod
from typing import Any, ClassVar, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from lark.exceptions import UnexpectedInput, VisitError
from scipy import special, stats

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .core import G2rinsBase
from .exception import EmptyTruncatedDistributionSupport, UnknownDistribution
from .util import RememberAdd, get_global_rng

_T = TypeVar("_T", bound="StochasticDistribution")
_S = TypeVar("_S", bound="StochasticGeneration")

_SERIAL_SENTINEL = -1.0
_LEGACY_SERIAL_LAYOUT = (
    ("flory_schulz", 1),
    ("schulz_zimm", 2),
    ("gauss", 2),
    ("uniform", 2),
    ("log_normal", 2),
    ("poisson", 1),
)
_CURRENT_SERIAL_LAYOUT = (
    ("flory_schulz", 2),
    ("schulz_zimm", 2),
    ("gauss", 2),
    ("uniform", 2),
    ("log_normal", 2),
    ("poisson", 2),
)


_GROUPED_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _strip_grouping_separators_in_numeric_literals(text: str) -> str:
    """Remove thousands separators from grouped numeric literals.

    This supports external generators that may emit locale-formatted values
    such as ``11,963.3`` inside distribution argument lists.
    """

    def _degroup(match: re.Match[str]) -> str:
        return match.group(0).replace(",", "")

    return _GROUPED_NUMBER_RE.sub(_degroup, text)


def _mass_tolerance(mass: float, quantum: float) -> float:
    """Return a small absolute tolerance suitable for mass-lattice comparisons."""
    scale = max(1.0, abs(mass), abs(quantum))
    return 8.0 * math.ulp(scale)


def _mass_floor_count(mass: float, quantum: float) -> int:
    ratio = mass / quantum
    nearest = int(np.rint(ratio))
    if abs(mass - nearest * quantum) <= _mass_tolerance(mass, quantum):
        return nearest
    return math.floor(ratio)


def _mass_ceil_count(mass: float, quantum: float) -> int:
    ratio = mass / quantum
    nearest = int(np.rint(ratio))
    if abs(mass - nearest * quantum) <= _mass_tolerance(mass, quantum):
        return nearest
    return math.ceil(ratio)


def _mass_lattice_count(mass: float, quantum: float) -> Optional[int]:
    if not math.isfinite(mass):
        return None
    nearest = int(np.rint(mass / quantum))
    if abs(mass - nearest * quantum) <= _mass_tolerance(mass, quantum):
        return nearest
    return None


def _log_difference(log_larger: float, log_smaller: float) -> float:
    """Return ``log(exp(log_larger) - exp(log_smaller))`` stably."""
    if math.isnan(log_larger) or math.isnan(log_smaller):
        return math.nan
    if log_larger == -math.inf:
        return -math.inf
    if log_smaller == -math.inf:
        return log_larger
    if log_smaller >= log_larger:
        return -math.inf
    return log_larger + math.log(-math.expm1(log_smaller - log_larger))


def _open_unit_draw(rng: np.random.Generator) -> float:
    """Draw from (0, 1), keeping inverse transforms away from infinities."""
    value = float(rng.random())
    return min(max(value, np.nextafter(0.0, 1.0)), np.nextafter(1.0, 0.0))


def _discrete_inverse(
    distribution,
    lower: int,
    upper: Optional[int],
    log_probability: float,
    boundary_log_probability: float,
    use_survival: bool,
    kwargs: Any,
) -> int:
    """Invert a discrete CDF/SF without SciPy's ``1 - q`` cancellation."""

    def reached(value: int) -> bool:
        if use_survival:
            current = float(distribution.logsf(value, **kwargs))
            # Preserve the open boundary at lower - 1 even if the random
            # probability rounded back to that boundary.
            return current <= log_probability and current < boundary_log_probability
        current = float(distribution.logcdf(value, **kwargs))
        return current >= log_probability and current > boundary_log_probability

    high = lower if reached(lower) else upper
    if high is None:
        step = 1
        high = lower + step
        for _ in range(63):
            if reached(high):
                break
            step *= 2
            high = lower + step
        else:
            raise RuntimeError("Could not bracket a finite discrete truncated-distribution quantile")
    elif not reached(high):
        raise RuntimeError("Could not invert a discrete truncated-distribution quantile inside its bounds")

    low = lower
    while low < high:
        midpoint = (low + high) // 2
        if reached(midpoint):
            high = midpoint
        else:
            low = midpoint + 1
    return low


class StochasticGeneration(G2rinsBase):
    """
    Base class for stochastic generation components in G2RINS.
    """

    pass


class StochasticDistribution(StochasticGeneration):
    """
    Base class for stochastic distributions used in G2RINS.

    Subclasses should implement specific distributions and register themselves
    in the `_known_distributions` class attribute.
    """

    _known_distributions: ClassVar[List[Type["StochasticDistribution"]]] = list()
    _distribution: Optional[stats.rv_discrete] = None

    def __init__(self, children: List[Any]):
        """
        Initializes a StochasticDistribution object.

        Args:
            children (List[Any]): List of parsed child elements.
        """
        super().__init__(children)

    def __bool__(self) -> bool:
        """
        Returns True if this object can generate targets, including point masses.
        """
        return self.generable

    @classmethod
    def make(cls: Type[_T], text: str) -> _T:
        """
        Creates a specific StochasticDistribution subclass instance from a text representation.

        It iterates through the registered `_known_distributions` and attempts
        to create an instance if the distribution's token name (snake case)
        is found in the input text.

        Args:
            text (str): The textual representation of the stochastic distribution.

        Returns:
            _T: An instance of the appropriate StochasticDistribution subclass.

        Raises:
            UnknownDistribution: If no known distribution's token name is found in the text.
        """
        for known_distr in cls._known_distributions:
            if known_distr.token_name_snake_case in text:
                try:
                    return known_distr.make(text)
                except UnexpectedInput:
                    normalized_text = _strip_grouping_separators_in_numeric_literals(text)
                    if normalized_text == text:
                        raise
                    return known_distr.make(normalized_text)
        raise UnknownDistribution(text)

    def draw_mw(self, rng: Optional[np.random.Generator] = None, lower=None, upper=None, **kwargs: Any) -> Any:
        # TODO: revise this method to handle default lower and upper bounds correctly
        """
        Draws a sample from the molecular weight distribution.

        Args:
            rng (Optional[np.random.Generator]): Numpy random number generator for sampling.
                                                 If None, the global RNG is used.
            lower (float): The lower bound for the sampling range. Defaults to None for non-truncated sampling.
            upper (float): The upper bound for the sampling range. Defaults to None for non-truncated sampling.
            **kwargs (Any): Keyword arguments to pass to the distribution's sampling method.

        Returns:
            Any: A sample drawn from the distribution.

        Raises:
            NotImplementedError: If the `_distribution` attribute is None.
        """
        distribution = self._distribution

        if distribution is None:
            raise NotImplementedError

        if rng is None:
            rng = get_global_rng()

        if lower is None and upper is None:
            # Return the honest draw: clamping negatives to 0 made them
            # indistinguishable from a genuine zero target (a valid value --
            # some stochastic objects generate nothing); the caller decides
            # how to treat a negative.
            return float(distribution.rvs(random_state=rng, **kwargs))

        return self._draw_bounded_mw(distribution, rng, lower, upper, kwargs)

    def _draw_bounded_mw(self, distribution, rng, lower, upper, kwargs) -> float:
        """Draw over the requested interval using stable CDF/SF transforms."""
        try:
            requested_lower = -math.inf if lower is None else float(lower)
            requested_upper = math.inf if upper is None else float(upper)
        except (TypeError, ValueError, OverflowError) as error:
            raise EmptyTruncatedDistributionSupport(type(self).__name__, math.nan, math.nan) from error

        def empty_support() -> EmptyTruncatedDistributionSupport:
            return EmptyTruncatedDistributionSupport(type(self).__name__, requested_lower, requested_upper)

        if (
            math.isnan(requested_lower)
            or math.isnan(requested_upper)
            or requested_lower > requested_upper
        ):
            raise empty_support()

        # Molecular weights live on [0, +inf). Intersect before sampling so a
        # negative draw is never moved outside an already validated interval by
        # a post-hoc clamp.
        interval_lower = max(0.0, requested_lower)
        interval_upper = requested_upper
        if interval_lower > interval_upper or interval_lower == math.inf:
            raise empty_support()

        underlying = getattr(distribution, "dist", distribution)
        is_discrete = isinstance(underlying, stats.rv_discrete)

        # scipy exposes a zero-scale frozen continuous distribution with NaN
        # CDF/support values even though it represents a useful point mass.
        frozen_parameters = getattr(distribution, "kwds", {})
        if not is_discrete and frozen_parameters.get("scale") == 0:
            point = float(frozen_parameters.get("loc", 0.0))
            if math.isfinite(point) and interval_lower <= point <= interval_upper:
                return point
            raise empty_support()

        try:
            support_lower, support_upper = (
                float(value) for value in distribution.support(**kwargs)
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError(f"Could not determine support for {type(self).__name__}") from error
        if math.isnan(support_lower) or math.isnan(support_upper):
            raise RuntimeError(f"{type(self).__name__} returned NaN distribution support")

        interval_lower = max(interval_lower, support_lower)
        interval_upper = min(interval_upper, support_upper)
        if interval_lower > interval_upper or interval_lower == math.inf:
            raise empty_support()

        if is_discrete:
            return self._draw_bounded_discrete(
                distribution, rng, interval_lower, interval_upper, empty_support, kwargs
            )
        return self._draw_bounded_continuous(
            distribution, rng, interval_lower, interval_upper, empty_support, kwargs
        )

    def _draw_bounded_discrete(
        self, distribution, rng, interval_lower, interval_upper, empty_support, kwargs
    ) -> float:
        """Sample an inclusive integer interval using log-CDF/log-SF search."""
        lower_integer = math.ceil(interval_lower)
        upper_integer = None if interval_upper == math.inf else math.floor(interval_upper)
        if upper_integer is not None and lower_integer > upper_integer:
            raise empty_support()

        lower_edge = lower_integer - 1
        upper_edge = math.inf if upper_integer is None else upper_integer
        log_cdf_before = float(distribution.logcdf(lower_edge, **kwargs))
        log_cdf_upper = float(distribution.logcdf(upper_edge, **kwargs))
        log_sf_before = float(distribution.logsf(lower_edge, **kwargs))
        log_sf_upper = float(distribution.logsf(upper_edge, **kwargs))
        log_cdf_mass = _log_difference(log_cdf_upper, log_cdf_before)
        log_sf_mass = _log_difference(log_sf_before, log_sf_upper)

        cdf_before = float(distribution.cdf(lower_edge, **kwargs))
        prefer_survival = not math.isfinite(cdf_before) or cdf_before >= 0.5
        if prefer_survival and math.isfinite(log_sf_mass):
            use_survival = True
        elif not prefer_survival and math.isfinite(log_cdf_mass):
            use_survival = False
        elif math.isfinite(log_sf_mass):
            use_survival = True
        elif math.isfinite(log_cdf_mass):
            use_survival = False
        else:
            # In sufficiently remote tails even sf itself underflows to zero.
            # logpmf can remain finite, so normalize individual support weights
            # in log space instead of falsely reporting empty support. For a
            # one-sided tail, the fallback grows a finite numerical bracket.
            return self._draw_bounded_discrete_logpmf(
                distribution,
                rng,
                lower_integer,
                upper_integer,
                empty_support,
                kwargs,
            )

        unit_draw = _open_unit_draw(rng)
        if use_survival:
            log_probability = float(
                np.logaddexp(log_sf_upper, math.log(unit_draw) + log_sf_mass)
            )
            boundary_log_probability = log_sf_before
        else:
            log_probability = float(
                np.logaddexp(log_cdf_before, math.log(unit_draw) + log_cdf_mass)
            )
            boundary_log_probability = log_cdf_before

        sample_integer = _discrete_inverse(
            distribution,
            lower_integer,
            upper_integer,
            log_probability,
            boundary_log_probability,
            use_survival,
            kwargs,
        )
        if upper_integer is not None and sample_integer > upper_integer:
            raise RuntimeError("Discrete truncated-distribution inverse escaped its interval")
        return float(sample_integer)

    def _draw_bounded_discrete_logpmf(
        self, distribution, rng, lower_integer, upper_integer, empty_support, kwargs
    ) -> float:
        """Sample a finite discrete interval whose cumulative tails underflow."""
        if upper_integer is not None:
            count = upper_integer - lower_integer + 1
            if count <= 0:
                raise empty_support()
            if count > 1_000_000:
                raise RuntimeError(
                    "A numerically underflowed discrete interval is too wide for per-value inversion"
                )
            values = np.arange(lower_integer, upper_integer + 1, dtype=np.int64)
            log_weights = np.asarray(distribution.logpmf(values, **kwargs), dtype=float)
        else:
            # Cumulative functions have already underflowed, which places this
            # path in a remote tail. Grow until a sustained decreasing run is
            # negligible relative to the largest enumerated log weight. If an
            # unusual distribution does not establish such a bracket, report a
            # numerical inversion failure rather than misclassifying it as an
            # empty chain-local interval.
            count = 128
            while True:
                values = np.arange(lower_integer, lower_integer + count, dtype=np.int64)
                log_weights = np.asarray(distribution.logpmf(values, **kwargs), dtype=float)
                if np.isnan(log_weights).any() or np.isposinf(log_weights).any():
                    raise RuntimeError("Discrete distribution returned invalid log-PMF values")
                finite_weights = log_weights[np.isfinite(log_weights)]
                if finite_weights.size:
                    tail = finite_weights[-64:]
                    decreasing = tail.size == 64 and np.all(np.diff(tail) <= 0.0)
                    negligible = tail[-1] <= np.max(finite_weights) - 50.0
                    ended = np.isneginf(log_weights[-64:]).all()
                    if (decreasing and negligible) or ended:
                        break
                if count >= 1_000_000:
                    raise RuntimeError(
                        "Could not bracket a numerically underflowed one-sided discrete tail"
                    )
                count = min(2 * count, 1_000_000)

        if np.isnan(log_weights).any() or np.isposinf(log_weights).any():
            raise RuntimeError("Discrete distribution returned invalid log-PMF values")
        finite = np.isfinite(log_weights)
        if not finite.any():
            raise empty_support()

        log_normalization = float(special.logsumexp(log_weights[finite]))
        if not math.isfinite(log_normalization):
            raise RuntimeError("Could not normalize a finite discrete truncated interval")
        weights = np.zeros(log_weights.shape, dtype=float)
        weights[finite] = np.exp(log_weights[finite] - log_normalization)
        weights /= weights.sum()
        index = int(rng.choice(values.size, p=weights))
        return float(values[index])

    def _draw_bounded_continuous(
        self, distribution, rng, interval_lower, interval_upper, empty_support, kwargs
    ) -> float:
        """Sample a continuous interval, choosing its stable probability tail."""
        log_cdf_lower = float(distribution.logcdf(interval_lower, **kwargs))
        log_cdf_upper = float(distribution.logcdf(interval_upper, **kwargs))
        log_sf_lower = float(distribution.logsf(interval_lower, **kwargs))
        log_sf_upper = float(distribution.logsf(interval_upper, **kwargs))
        log_cdf_mass = _log_difference(log_cdf_upper, log_cdf_lower)
        log_sf_mass = _log_difference(log_sf_lower, log_sf_upper)

        cdf_lower = float(distribution.cdf(interval_lower, **kwargs))
        prefer_survival = not math.isfinite(cdf_lower) or cdf_lower >= 0.5
        if prefer_survival and math.isfinite(log_sf_mass):
            use_survival = True
        elif not prefer_survival and math.isfinite(log_cdf_mass):
            use_survival = False
        elif math.isfinite(log_sf_mass):
            use_survival = True
        elif math.isfinite(log_cdf_mass):
            use_survival = False
        else:
            raise empty_support()

        unit_draw = _open_unit_draw(rng)
        if use_survival:
            log_probability = float(
                np.logaddexp(log_sf_upper, math.log(unit_draw) + log_sf_mass)
            )
            probability = float(math.exp(log_probability))
            probability = min(
                max(probability, np.nextafter(0.0, 1.0)),
                np.nextafter(1.0, 0.0),
            )
            sample_mw = float(distribution.isf(probability, **kwargs))
        else:
            log_probability = float(
                np.logaddexp(log_cdf_lower, math.log(unit_draw) + log_cdf_mass)
            )
            probability = float(math.exp(log_probability))
            probability = min(
                max(probability, np.nextafter(0.0, 1.0)),
                np.nextafter(1.0, 0.0),
            )
            sample_mw = float(distribution.ppf(probability, **kwargs))

        # Positive probability was established above.  A non-finite or remote
        # inverse result is therefore a numerical/inverse failure, not empty
        # chain-local support, and must not be silently converted to a discard.
        if not math.isfinite(sample_mw):
            raise RuntimeError(f"{type(self).__name__} returned a non-finite truncated-distribution quantile")
        if not interval_lower <= sample_mw <= interval_upper:
            tolerance = 1e-10 * max(1.0, abs(sample_mw))
            if interval_lower - tolerance <= sample_mw <= interval_upper + tolerance:
                sample_mw = min(max(sample_mw, interval_lower), interval_upper)
            else:
                raise RuntimeError(
                    f"{type(self).__name__} returned truncated quantile {sample_mw:g} "
                    f"outside [{interval_lower:g}, {interval_upper:g}]"
                )
        return sample_mw

    def prob_mw(self, mw: Union[float, "RememberAdd"], **kwargs: Any) -> float:
        """
        Calculates the probability (PMF or CDF difference) for a given molecular weight.

        Args:
            mw (Union[float, RememberAdd]): The molecular weight to calculate the probability for.
                                           If a RememberAdd object, calculates the probability
                                           within the range defined by its previous and current values.
            **kwargs (Any): Keyword arguments to pass to the distribution's probability method.

        Returns:
            float: The probability of the given molecular weight(s).

        Raises:
            NotImplementedError: If the `_distribution` attribute is None.
        """
        if self._distribution is None:
            raise NotImplementedError

        if isinstance(mw, RememberAdd):
            return self._distribution.cdf(mw.value, **kwargs) - self._distribution.cdf(mw.previous, **kwargs)

        if hasattr(self._distribution, "pdf"):
            return self._distribution.pdf(mw, **kwargs)
        if hasattr(self._distribution, "pmf"):
            return self._distribution.pmf(k=int(mw), **kwargs)
        raise NotImplementedError

    def reference_mw(self) -> float:
        """Return the exact number-average target mass when the form defines one."""
        raise NotImplementedError(f"{type(self).__name__} does not define a reference molecular weight")

    def support_mw(self) -> Tuple[float, float]:
        """Return the lower and upper support bounds in molar-mass units."""
        if self._distribution is None:
            raise NotImplementedError
        parameters = getattr(self._distribution, "kwds", {})
        if parameters.get("scale") == 0:
            point = float(parameters.get("loc", 0.0))
            return point, point
        lower, upper = self._distribution.support()
        return float(lower), float(upper)

    def _draw_scaled_discrete(
        self,
        distribution,
        quantum: float,
        rng: np.random.Generator,
        lower,
        upper,
        kwargs: Any,
        *,
        minimum_count: int,
        deterministic_mass: Optional[float] = None,
    ) -> float:
        """Draw an integer count law through inclusive molar-mass bounds."""
        try:
            requested_lower = -math.inf if lower is None else float(lower)
            requested_upper = math.inf if upper is None else float(upper)
        except (TypeError, ValueError, OverflowError) as error:
            raise EmptyTruncatedDistributionSupport(type(self).__name__, math.nan, math.nan) from error

        def empty_support() -> EmptyTruncatedDistributionSupport:
            return EmptyTruncatedDistributionSupport(
                type(self).__name__, requested_lower, requested_upper
            )

        if (
            math.isnan(requested_lower)
            or math.isnan(requested_upper)
            or requested_lower > requested_upper
            or requested_lower == math.inf
            or requested_upper == -math.inf
        ):
            raise empty_support()

        if deterministic_mass is not None:
            tolerance = _mass_tolerance(deterministic_mass, quantum)
            if requested_lower - tolerance <= deterministic_mass <= requested_upper + tolerance:
                return float(deterministic_mass)
            raise empty_support()

        if lower is None and upper is None:
            return float(quantum * distribution.rvs(random_state=rng, **kwargs))

        lower_count = (
            minimum_count
            if requested_lower == -math.inf
            else max(minimum_count, _mass_ceil_count(requested_lower, quantum))
        )
        upper_count = (
            math.inf
            if requested_upper == math.inf
            else _mass_floor_count(requested_upper, quantum)
        )
        if lower_count > upper_count:
            raise empty_support()

        count = self._draw_bounded_discrete(
            distribution,
            rng,
            lower_count,
            upper_count,
            empty_support,
            kwargs,
        )
        mass = float(quantum * count)
        tolerance = _mass_tolerance(mass, quantum)
        if not requested_lower - tolerance <= mass <= requested_upper + tolerance:
            raise RuntimeError(
                f"{type(self).__name__} returned scaled discrete mass {mass:g} "
                f"outside [{requested_lower:g}, {requested_upper:g}]"
            )
        return mass

    @classmethod
    def _default_serialize(cls: Type["StochasticDistribution"], n: int) -> Tuple[float, ...]:
        """
        Internal helper method to create a tuple of default serialization values (-1.0).

        Args:
            n (int): The number of default values to generate.

        Returns:
            Tuple[float, ...]: A tuple containing n -1.0 values.
        """
        return tuple((-1.0 for _ in range(n)))

    @classmethod
    def default_serialize(cls: Type["StochasticDistribution"]) -> Tuple[float, ...]:
        """
        Returns the default serialization vector for this distribution type (an empty tuple).
        """
        return cls._default_serialize(0)

    @classmethod
    def get_empty_serial_vector(cls: Type["StochasticDistribution"]) -> List[float]:
        """
        Returns an empty serialization vector with the correct length to hold
        the default serialization of all known stochastic distributions.
        """
        return [
            _SERIAL_SENTINEL
            for _token_name, width in _CURRENT_SERIAL_LAYOUT
            for _ in range(width)
        ]

    def get_serial_vector(self) -> List[float]:
        """
        Returns the serialization vector for this specific stochastic distribution instance.

        The vector contains the serialized parameters of this instance, with default
        serialization values for other known distribution types.
        """
        vector: List[float] = []
        own_token = type(self).token_name_snake_case
        for token_name, width in _CURRENT_SERIAL_LAYOUT:
            if own_token == token_name:
                serialized = tuple(self.serialize())
                if len(serialized) != width:
                    raise ValueError(
                        f"{type(self).__name__} serialized {len(serialized)} values; expected {width}."
                    )
                vector.extend(serialized)
            else:
                vector.extend((_SERIAL_SENTINEL,) * width)
        return vector

    @classmethod
    def from_serialized(cls: Type[_T], params: Tuple[float, ...]) -> Optional[_T]:
        """Reconstruct a distribution from a class-local serialized parameter tuple."""
        if params is None:
            return None
        params = tuple(params)
        if not params:
            return None
        if all(float(value) == -1.0 for value in params):
            return None

        if len(params) == 1:
            return cls.make(f"{cls.token_name_snake_case}({params[0]})")

        if len(params) == 2:
            return cls.make(f"{cls.token_name_snake_case}({params[0]}, {params[1]})")

        raise ValueError(f"Unsupported serialized parameter tuple for {cls.__name__}: {params!r}")

    @classmethod
    def from_serial_vector(cls: Type[_T], vector: List[float]) -> Optional[_T]:
        """Decode either the legacy 10-slot layout or the newer 12-slot layout."""
        values = list(vector)
        distribution_types = {
            distribution_type.token_name_snake_case: distribution_type
            for distribution_type in cls._known_distributions
        }

        def decode(layout):
            candidates: List[Tuple[float, ...]] = []
            type_candidates: List[Type[_T]] = []
            index = 0
            for token_name, width in layout:
                distr_type = distribution_types[token_name]
                segment = tuple(values[index:index + width])
                index += width
                default_serial = (_SERIAL_SENTINEL,) * width
                if segment != default_serial:
                    candidates.append(segment)
                    type_candidates.append(distr_type)
            if not candidates:
                return None
            if len(candidates) != 1:
                raise ValueError("The passed vector did not contain only one candidate for the distribution.")
            return type_candidates[0].from_serialized(candidates[0])

        if len(values) == 10:
            return decode(_LEGACY_SERIAL_LAYOUT)

        if len(values) == 12:
            return decode(_CURRENT_SERIAL_LAYOUT)

        raise ValueError(
            f"Unrecognized stochastic-distribution serialization length {len(values)}. "
            "Expected the legacy length 10 or the current length 12."
        )

    @abstractmethod
    def serialize(self) -> Tuple[float, ...]:
        """
        Abstract method to serialize the parameters of this distribution into a tuple of floats.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Abstract class method to return the default serialization (e.g., a tuple of -1.0s)
        representing the absence of this distribution's parameters in a serial vector.
        """
        raise NotImplementedError


class FlorySchulz(StochasticDistribution):
    """Flory–Schulz target distribution.

    ``flory_schulz(a)`` preserves the legacy weight-fraction count law.
    ``flory_schulz(Mw, Mn)`` samples equally weighted chain targets in molar-mass
    units using ``M = q*N``, where ``N`` is geometric, ``a = 2 - Mw/Mn``, and
    ``q = Mn*a``. The mass form represents ``1 <= Mw/Mn < 2``.
    """

    class flory_schulz_gen(stats.rv_discrete):
        """Legacy weight-fraction Flory–Schulz chain-length law."""

        def _rvs(self, fls_a, size=None, random_state=None):
            return random_state.negative_binomial(2, fls_a, size=size) + 1

        def _pmf(self, k, fls_a):
            return fls_a**2 * k * (1 - fls_a) ** (k - 1)

        def _sf(self, k, fls_a):
            k = np.asarray(k)
            finite_k = np.where(np.isfinite(k), k, 0.0)
            tail_k = np.maximum(np.floor(finite_k), 0)
            tail = (1 - fls_a) ** tail_k * (1 + fls_a * tail_k)
            return np.where(np.isposinf(k), 0.0, np.where(np.isneginf(k), 1.0, tail))

        def _logsf(self, k, fls_a):
            k = np.asarray(k)
            finite_k = np.where(np.isfinite(k), k, 0.0)
            tail_k = np.maximum(np.floor(finite_k), 0)
            log_tail = tail_k * np.log1p(-fls_a) + np.log1p(fls_a * tail_k)
            return np.where(
                np.isposinf(k), -math.inf, np.where(np.isneginf(k), 0.0, log_tail)
            )

    _fls_a: Optional[float] = None
    _Mw: Optional[float] = None
    _Mn: Optional[float] = None
    _q: Optional[float] = None
    _mode: Optional[str] = None

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        try:
            return G2rinsBase.make.__func__(cls, text)
        except VisitError as error:
            raise error.orig_exc from error

    def __init__(self, children: List[Any]):
        super().__init__(children)

        numbers: List[float] = []
        for child in self._children:
            if isinstance(child, float):
                numbers.append(float(child))

        if len(numbers) == 1:
            fls_a = numbers[0]
            if not 0 < fls_a < 1:
                raise RuntimeError(f"The legacy Flory-Schulz distribution needs a parameter between 0 and 1. Got {fls_a}.")
            self._mode = "legacy"
            self._fls_a = fls_a
            self._distribution = self.flory_schulz_gen(name="Flory-Schulz", a=1)(fls_a=self._fls_a)
            return

        if len(numbers) == 2:
            self._Mw, self._Mn = numbers
            if not np.isfinite(self._Mw) or not np.isfinite(self._Mn):
                raise ValueError("Flory–Schulz molar-mass parameters must be finite.")
            if not (self._Mn > 0 and self._Mw >= self._Mn):
                raise ValueError("For flory_schulz(Mw, Mn), require finite Mw >= Mn > 0.")
            dispersity = self._Mw / self._Mn
            if not (1.0 <= dispersity < 2.0):
                raise ValueError(
                    "For flory_schulz(Mw, Mn), the representable dispersity range is "
                    "1 <= Mw/Mn < 2; use Schulz-Zimm or log-normal for larger dispersity."
                )
            self._mode = "molar_mass"
            self._fls_a = 2.0 - dispersity
            self._q = self._Mn * self._fls_a
            self._distribution = stats.geom(p=self._fls_a)
            return

        raise ValueError("flory_schulz accepts either one legacy parameter or two molar-mass parameters: (Mw, Mn).")

    def _mass_to_count(self, mass: float) -> Optional[int]:
        if self._q is None or self._q <= 0:
            return None
        count = _mass_lattice_count(mass, self._q)
        return count if count is not None and count >= 1 else None

    def draw_mw(self, rng: Optional[np.random.Generator] = None, lower=None, upper=None, **kwargs):
        if self._mode == "legacy":
            return super().draw_mw(rng=rng, lower=lower, upper=upper, **kwargs)
        if rng is None:
            rng = get_global_rng()
        return super()._draw_scaled_discrete(
            self._distribution,
            self._q,
            rng,
            lower,
            upper,
            kwargs,
            minimum_count=1,
            deterministic_mass=float(self._Mn) if self._Mn == self._Mw else None,
        )

    def generate_string(self, extension: bool) -> str:
        if extension:
            if self._mode == "legacy":
                return f"|flory_schulz({self._fls_a})|"
            return f"|flory_schulz({self._Mw}, {self._Mn})|"
        return ""

    @property
    def generable(self) -> bool:
        return self._distribution is not None

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        return cls._default_serialize(2)

    def serialize(self) -> Tuple[float, ...]:
        if self._mode == "legacy":
            return (self._fls_a, -1.0)
        return (self._Mw, self._Mn)

    @classmethod
    def from_serialized(cls: Type[Self], params: Tuple[float, ...]) -> Optional[Self]:
        if tuple(params) == (_SERIAL_SENTINEL, _SERIAL_SENTINEL):
            return None
        if len(params) == 1 or (len(params) == 2 and params[1] == _SERIAL_SENTINEL):
            return cls.make(f"flory_schulz({params[0]})")
        if len(params) == 2:
            return cls.make(f"flory_schulz({params[0]}, {params[1]})")
        raise ValueError(f"Unsupported serialized parameter tuple for {cls.__name__}: {params!r}")

    def reference_mw(self) -> float:
        if self._mode == "legacy":
            return float(2.0 / self._fls_a - 1.0)
        return float(self._Mn)

    def support_mw(self) -> Tuple[float, float]:
        if self._mode == "legacy":
            return super().support_mw()
        if self._Mn == self._Mw:
            return float(self._Mn), float(self._Mn)
        lower, upper = self._distribution.support()
        return float(self._q * lower), float(self._q * upper)

    def prob_mw(self, mw):
        if self._mode == "legacy":
            return super().prob_mw(mw)

        if isinstance(mw, RememberAdd):
            if self._Mn == self._Mw:
                previous = mw.previous
                value = mw.value
                if previous < self._Mn <= value:
                    return 1.0
                return 0.0
            lower = mw.previous
            upper = mw.value
            if lower > upper:
                lower, upper = upper, lower
            lower_count = _mass_floor_count(lower, self._q) if np.isfinite(lower) else -math.inf
            upper_count = _mass_floor_count(upper, self._q) if np.isfinite(upper) else math.inf
            lower_cdf = self._distribution.cdf(lower_count) if np.isfinite(lower_count) else 0.0
            upper_cdf = self._distribution.cdf(upper_count) if np.isfinite(upper_count) else 1.0
            return float(upper_cdf - lower_cdf)

        if self._Mn == self._Mw:
            point = float(self._Mn)
            tol = _mass_tolerance(point, self._q)
            return 1.0 if abs(float(mw) - point) <= tol else 0.0

        count = self._mass_to_count(float(mw))
        if count is None:
            return 0.0
        return float(self._distribution.pmf(count))


StochasticDistribution._known_distributions.append(FlorySchulz)


class SchulzZimm(StochasticDistribution):
    r"""
    Schulz-Zimm distribution of molecular weights.

    :math:`P(M) = \frac{z^{z+1}}{\Gamma(z+1)} \left(\frac{M}{M_n}\right)^{z-1} \frac{1}{M_n} \exp\left(-\frac{zM}{M_n}\right)`
    :math:`z = \frac{M_n}{M_w - M_n}`

    where :math:`\Gamma` is the Gamma function, :math:`M_w` is the weight-average
    molecular weight, and :math:`M_n` is the number-average molecular weight.
    P. C. Hiemenz, T. P. Lodge, Polymer Chemistry, CRC Press, Boca Raton, FL 2007.

    The textual representation of this distribution is: `schulz_zimm(Mw, Mn)`
    """

    class schulz_zimm_gen(stats.rv_continuous):
        """Schulz-Zimm distribution."""

        # The Schulz-Zimm number distribution is a Gamma with shape z and scale
        # Mn/z (matching the class docstring), so Mn = z*(Mn/z) and Mw/Mn =
        # (z+1)/z with z = Mn/(Mw-Mn). An earlier version used shape z+1, which
        # kept Mn correct but gave Mw/Mn = (z+2)/(z+1) — the requested dispersity
        # was not reproduced.
        def _pdf(self, M, z, Mn):
            prefactor = z ** (z + 1) / special.gamma(z + 1)
            return prefactor * (M ** (z - 1) / Mn**z) * np.exp(-z * M / Mn)

        def _cdf(self, M, z, Mn):
            # Regularized lower incomplete gamma: P(z, z*M/Mn)
            return special.gammainc(z, z * M / Mn)

        def _sf(self, M, z, Mn):
            # Evaluate the complemented function directly; 1 - gammainc
            # loses the upper tail once the CDF rounds to one.
            return special.gammaincc(z, z * M / Mn)

        def _logsf(self, M, z, Mn):
            return np.log(special.gammaincc(z, z * M / Mn))

        def _ppf(self, q, z, Mn):
            # M = Mn/z * gammaincinv(z, q)
            return (Mn / z) * special.gammaincinv(z, q)

        def _isf(self, q, z, Mn):
            return (Mn / z) * special.gammainccinv(z, q)

        def _get_support(self, z, Mn):
            return (0, np.inf)

    _Mw: Optional[float] = None
    _Mn: Optional[float] = None
    _z: Optional[float] = None

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        """
        Creates a SchulzZimm instance from its textual representation.

        Args:
            text (str): The textual representation, e.g., 'schulz_zimm(1000, 500)'.

        Returns:
            Self: A SchulzZimm instance.
        """
        # We use G2rinsBase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return G2rinsBase.make.__func__(cls, text)

    def __init__(self, children: List[Any]):
        """
        Initialization of Schulz-Zimm distribution object.

        Args:
            children (List[Any]): List of parsed children, expected to contain Mw and Mn as floats.
        """
        super().__init__(children)

        numbers: List[float] = []
        for child in self._children:
            if isinstance(child, float):
                numbers.append(child)

        self._Mw, self._Mn = numbers
        self._z = self._Mn / (self._Mw - self._Mn) if self._Mw > self._Mn else None
        self._distribution = self.schulz_zimm_gen(name="Schulz-Zimm", a=0)(z=self._z, Mn=self._Mn)

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Returns the default serialization for SchulzZimm (a tuple with two -1.0s).
        """
        return cls._default_serialize(2)

    def serialize(self) -> Tuple[float, ...]:
        """
        Serializes the Mw and Mn parameters of the SchulzZimm distribution.
        """
        return (self._Mw, self._Mn)

    def generate_string(self, extension: bool) -> str:
        """
        Generates the textual representation of the Schulz-Zimm distribution.

        Args:
            extension (bool): Whether to include the '|' delimiters.

        Returns:
            str: The textual representation, e.g., '|schulz_zimm(1000, 500)|'.
        """
        if extension:
            return f"|schulz_zimm({self._Mw}, {self._Mn})|"
        return ""

    @property
    def generable(self) -> bool:
        """
        Returns True if the distribution is initialized (i.e., Mw and Mn are set and valid).
        """
        return self._distribution is not None and self._z is not None

    def draw_mw(self, rng: Optional[np.random.Generator] = None, lower=None, upper=None) -> Any:
        """
        Draws a sample from the Schulz-Zimm distribution.
        """
        return super().draw_mw(rng=rng, lower=lower, upper=upper)

    def prob_mw(self, mw: Union[float, "RememberAdd"]) -> float:
        """
        Calculates the probability for a given molecular weight using the Schulz-Zimm distribution.
        """
        return super().prob_mw(mw)


StochasticDistribution._known_distributions.append(SchulzZimm)


class Gauss(StochasticDistribution):
    r"""
    Gauss (Normal) distribution of molecular weights.

    :math:`G(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2} \left(\frac{x-\mu}{\sigma}\right)^2\right)`

    where :math:`\mu` is the mean and :math:`\sigma` is the standard deviation.
    The textual representation is: `gauss(mu, sigma)`
    """

    _mu: Optional[float] = None
    _sigma: Optional[float] = None

    def __init__(self, children: List[Any]):
        """
        Initialization of Gaussian distribution object.

        Args:
            children (List[Any]): List of parsed children, expected to contain mean (mu) and
                                 standard deviation (sigma) as floats.
        """
        super().__init__(children)

        numbers: List[float] = []
        for child in self._children:
            if isinstance(child, float):
                numbers.append(child)

        self._mu, self._sigma = numbers
        self._distribution = stats.norm(loc=self._mu, scale=self._sigma)

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Returns the default serialization for Gauss (a tuple with two -1.0s).
        """
        return cls._default_serialize(2)

    def serialize(self) -> Tuple[float, ...]:
        """
        Serializes the mean (mu) and standard deviation (sigma) of the Gauss distribution.
        """
        return (self._mu, self._sigma)

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        """
        Creates a Gauss instance from its textual representation.

        Args:
            text (str): The textual representation, e.g., 'gauss(100, 10)'.

        Returns:
            Self: A Gauss instance.
        """
        # We use G2rinsBase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return G2rinsBase.make.__func__(cls, text)

    def generate_string(self, extension: bool) -> str:
        """
        Generates the textual representation of the Gauss distribution.

        Args:
            extension (bool): Whether to include the '|' delimiters.

        Returns:
            str: The textual representation, e.g., '|gauss(100, 10)|'.
        """
        if extension:
            return f"|gauss({self._mu}, {self._sigma})|"
        return ""

    @property
    def generable(self) -> bool:
        """
        Returns True if the distribution is initialized (i.e., mu and sigma are set).
        """
        return self._distribution is not None

    def prob_mw(self, mw: Union[float, "RememberAdd"]) -> float:
        """
        Calculates the probability density for a given molecular weight using the Gauss distribution.

        Args:
            mw (Union[float, RememberAdd]): The molecular weight to calculate the probability for.
                                           If a RememberAdd object, this method might not be directly
                                           meaningful for a continuous distribution.

        Returns:
            float: The probability density at the given molecular weight.
        """
        if self._sigma is not None and self._sigma < 1e-6 and self._mu is not None and abs(self._mu - mw) < 1e-6:
            return 1.0
        return super().prob_mw(mw)


StochasticDistribution._known_distributions.append(Gauss)


class Uniform(StochasticDistribution):
    # TODO: implement prob_mw()
    """
    Uniform distribution of different lengths, usually useful for short chains.

    The textual representation of this distribution is: `uniform(low, high)`
    """

    _low: Optional[float] = None
    _high: Optional[float] = None

    def __init__(self, children: List[Any]):
        """
        Initialization of Uniform distribution object.

        Args:
            children (List[Any]): List of parsed children, expected to contain the lower (low) and
                                 upper (high) bounds as floats.
        """
        super().__init__(children)

        numbers: List[float] = []
        for child in self._children:
            if isinstance(child, float):
                numbers.append(child)

        self._low, self._high = numbers
        self._distribution = stats.uniform(loc=self._low, scale=(self._high - self._low) if self._low is not None and self._high is not None else 0)

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Returns the default serialization for Uniform (a tuple with two -1.0s).
        """
        return cls._default_serialize(2)

    def serialize(self) -> Tuple[float, ...]:
        """
        Serializes the lower (low) and upper (high) bounds of the Uniform distribution.
        """
        return (self._low, self._high)

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        """
        Creates a Uniform instance from its textual representation.

        Args:
            text (str): The textual representation, e.g., 'uniform(1, 5)'.

        Returns:
            Self: A Uniform instance.
        """
        # We use G2rinsBase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return G2rinsBase.make.__func__(cls, text)

    def generate_string(self, extension: bool) -> str:
        """
        Generates the textual representation of the Uniform distribution.

        Args:
            extension (bool): Whether to include the '|' delimiters.

        Returns:
            str: The textual representation, e.g., '|uniform(1, 5)|'.
        """
        if extension:
            return f"|uniform({self._low}, {self._high})|"
        return ""

    @property
    def generable(self) -> bool:
        """
        Returns True if the distribution is initialized (i.e., low and high are set).
        """
        return self._distribution is not None


StochasticDistribution._known_distributions.append(Uniform)


class LogNormal(StochasticDistribution):
    # TODO: revise why the truncated sampling doesn't work for LogNormal

    r"""
    LogNormal distribution of molecular weights.

    :math:`f(x; S, \sigma) = \frac{1}{x \sigma \sqrt{2\pi}} \exp\left(-\frac{(\ln x - S)^2}{2\sigma^2}\right)`

    where :math:`S` is the shape parameter and :math:`\sigma` is the scale parameter.
    In the context of the original code, it seems :math:`M_n` (number average MW)
    and :math:`D` (polydispersity) are used as parameters. The provided PDF in the
    original docstring doesn't directly match the standard log-normal PDF.
    Assuming the original intent was to use :math:`M_n` and :math:`D`:

    The textual representation of this distribution is: `log_normal(Mn, D)`
    """

    class log_normal_gen(stats.rv_continuous):
        """Log-Normal distribution (parameterized by Mn and D)."""

        def _pdf(self, m, Mn, D):
            prefactor = 1 / (m * np.sqrt(2 * np.pi * np.log(D)))
            value = prefactor * np.exp(-((np.log(m / Mn) + np.log(D) / 2) ** 2) / (2 * np.log(D)))
            return value

        def _cdf(self, m, Mn, D):
            standard_normal = (np.log(m / Mn) + np.log(D) / 2) / np.sqrt(np.log(D))
            return special.ndtr(standard_normal)

        def _logcdf(self, m, Mn, D):
            standard_normal = (np.log(m / Mn) + np.log(D) / 2) / np.sqrt(np.log(D))
            return special.log_ndtr(standard_normal)

        def _sf(self, m, Mn, D):
            z = (np.log(m / Mn) + np.log(D) / 2) / np.sqrt(2 * np.log(D))
            return 0.5 * special.erfc(z)

        def _logsf(self, m, Mn, D):
            standard_normal = (np.log(m / Mn) + np.log(D) / 2) / np.sqrt(np.log(D))
            return special.log_ndtr(-standard_normal)

        def _ppf(self, q, Mn, D):
            standard_normal = special.ndtri(q)
            log_m = (
                standard_normal * np.sqrt(np.log(D))
                - np.log(D) / 2
                + np.log(Mn)
            )
            return np.exp(log_m)

        def _isf(self, q, Mn, D):
            z = special.erfcinv(2 * q)
            log_m = z * np.sqrt(2 * np.log(D)) - np.log(D) / 2 + np.log(Mn)
            return np.exp(log_m)

        def _get_support(self, Mn: float, D: float) -> Tuple[float, float]:
            """Returns the support of the distribution."""
            return (0, np.inf)

    _M: Optional[float] = None  # Assuming this corresponds to Mn
    _D: Optional[float] = None  # Assuming this corresponds to D

    def __init__(self, children: List[Any]):
        """
        Initialization of LogNormal distribution object.

        Args:
            children (List[Any]): List of parsed children, expected to contain Mn and D as floats.
        """
        super().__init__(children)

        numbers: List[float] = []
        for child in self._children:
            if isinstance(child, float):
                numbers.append(child)

        self._M, self._D = numbers
        if self._M is not None and self._D is not None and self._D > 0:
            self._distribution = self.log_normal_gen(name="Log-Normal")
        else:
            self._distribution = None

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Returns the default serialization for LogNormal (a tuple with two -1.0s).
        """
        return cls._default_serialize(2)

    def serialize(self) -> Tuple[float, ...]:
        """
        Serializes the Mn and D parameters of the LogNormal distribution.
        """
        return (self._M, self._D)

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        """
        Creates a LogNormal instance from its textual representation.

        Args:
            text (str): The textual representation, e.g., 'log_normal(500, 1.1)'.

        Returns:
            Self: A LogNormal instance.
        """
        # We use G2rinsBase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return G2rinsBase.make.__func__(cls, text)

    def generate_string(self, extension: bool) -> str:
        """
        Generates the textual representation of the LogNormal distribution.

        Args:
            extension (bool): Whether to include the '|' delimiters.

        Returns:
            str: The textual representation, e.g., '|log_normal(500, 1.1)|'.
        """
        if extension:
            return f"|log_normal({self._M}, {self._D})|"
        return ""

    @property
    def generable(self) -> bool:
        """
        Returns True if the distribution is initialized (i.e., Mn and D are set and valid).
        """
        return self._distribution is not None

    def draw_mw(self, rng: Optional[np.random.Generator] = None, lower=None, upper=None) -> Any:
        """
        Draws a sample from the LogNormal distribution.
        """
        return super().draw_mw(rng=rng, lower=lower, upper=upper, Mn=self._M, D=self._D)

    def prob_mw(self, mw: Union[float, "RememberAdd"]) -> float:
        """
        Calculates the probability density for a given molecular weight using the LogNormal distribution.

        Args:
            mw (Union[float, RememberAdd]): The molecular weight to calculate the probability for.
                                           If a RememberAdd object, this method might not be directly
                                           meaningful for a continuous distribution.

        Returns:
            float: The probability density at the given molecular weight.
        """
        return super().prob_mw(mw, Mn=self._M, D=self._D)


StochasticDistribution._known_distributions.append(LogNormal)


class Poisson(StochasticDistribution):
    """Poisson target distribution.

    ``poisson(N)`` preserves the legacy unscaled behavior. The molar-mass form
    ``poisson(Mw, Mn)`` is treated as a zero-truncated Poisson law on strictly
    positive repeat-unit counts, so each generated chain has a physically valid
    positive mass. The underlying count law is therefore
    ``N | N > 0 ~ Poisson(lambda)`` conditioned on ``N >= 1`` and
    ``M = q N`` with ``q > 0`` chosen so that the target distribution matches
    ``Mn`` and ``Mw``. This excludes the mathematically allowed but
    non-polymeric zero-mass event from the realized chain population.

    The zero-truncated form is mathematically consistent with the idealized
    polymer interpretation: a zero draw represents a boundary/termination event,
    not a real chain mass. It also constrains the attainable dispersity to the
    narrow Poisson window ``1 <= Mw/Mn <= 1.298...``.
    """

    _N: Optional[float] = None
    _Mw: Optional[float] = None
    _Mn: Optional[float] = None
    _q: Optional[float] = None
    _lambda: Optional[float] = None
    _mode: Optional[str] = None

    def __init__(self, children: List[Any]):
        super().__init__(children)
        numbers: List[float] = []
        for child in self._children:
            if isinstance(child, float):
                numbers.append(float(child))

        if len(numbers) == 1:
            self._N = numbers[0]
            self._mode = "legacy"
            self._distribution = stats.poisson(mu=self._N)
            return

        if len(numbers) == 2:
            self._Mw, self._Mn = numbers
            if not np.isfinite(self._Mw) or not np.isfinite(self._Mn):
                raise ValueError("Poisson molar-mass parameters must be finite.")
            if not (self._Mn > 0 and self._Mw >= self._Mn):
                raise ValueError("For poisson(Mw, Mn), require finite Mw >= Mn > 0.")
            if self._Mw == self._Mn:
                self._mode = "point_mass"
                self._q = 1.0
                self._distribution = None
                return

            target_ratio = float(self._Mw / self._Mn)
            if target_ratio <= 1.0:
                raise ValueError("For zero-truncated poisson(Mw, Mn), require Mw > Mn > 0.")

            def zero_truncated_ratio(lam: float) -> float:
                if lam <= 0.0:
                    return 1.0
                return ((1.0 + lam) * (1.0 - math.exp(-lam))) / lam

            lam_low = 1e-12
            lam_high = 2.0
            max_ratio = zero_truncated_ratio(lam_high)
            if target_ratio > max_ratio + 1e-12:
                raise ValueError(
                    "For zero-truncated poisson(Mw, Mn), the target dispersity exceeds the "
                    f"supported Poisson range: Mw/Mn must be <= {max_ratio:.6f}."
                )

            for _ in range(200):
                lam_mid = 0.5 * (lam_low + lam_high)
                if zero_truncated_ratio(lam_mid) < target_ratio:
                    lam_low = lam_mid
                else:
                    lam_high = lam_mid

            self._mode = "molar_mass"
            self._lambda = 0.5 * (lam_low + lam_high)
            normalizer = 1.0 - math.exp(-self._lambda)
            self._q = self._Mn * normalizer / self._lambda
            self._distribution = stats.poisson(mu=self._lambda)
            return

        raise ValueError("poisson accepts either one legacy parameter or two molar-mass parameters: (Mw, Mn).")

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        return cls._default_serialize(2)

    def serialize(self) -> Tuple[float, ...]:
        if self._mode == "legacy":
            return (self._N, -1.0)
        if self._mode == "point_mass":
            return (self._Mn, self._Mn)
        return (self._Mw, self._Mn)

    @classmethod
    def from_serialized(cls: Type[Self], params: Tuple[float, ...]) -> Optional[Self]:
        if tuple(params) == (_SERIAL_SENTINEL, _SERIAL_SENTINEL):
            return None
        if len(params) == 1 or (len(params) == 2 and params[1] == _SERIAL_SENTINEL):
            return cls.make(f"poisson({params[0]})")
        if len(params) == 2:
            return cls.make(f"poisson({params[0]}, {params[1]})")
        raise ValueError(f"Unsupported serialized parameter tuple for {cls.__name__}: {params!r}")

    def reference_mw(self) -> float:
        if self._mode == "legacy":
            return float(self._N)
        return float(self._Mn)

    def support_mw(self) -> Tuple[float, float]:
        if self._mode == "point_mass":
            return float(self._Mn), float(self._Mn)
        if self._mode == "legacy":
            return super().support_mw()
        return float(self._q), math.inf

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        try:
            return G2rinsBase.make.__func__(cls, text)
        except VisitError as error:
            raise error.orig_exc from error

    def generate_string(self, extension: bool) -> str:
        if extension:
            if self._mode == "legacy":
                return f"|poisson({self._N})|"
            if self._mode == "point_mass":
                return f"|poisson({self._Mn}, {self._Mn})|"
            return f"|poisson({self._Mw}, {self._Mn})|"
        return ""

    @property
    def generable(self) -> bool:
        return self._distribution is not None or self._mode == "point_mass"

    def draw_mw(self, rng: Optional[np.random.Generator] = None, lower=None, upper=None, **kwargs):
        if self._mode == "legacy":
            return super().draw_mw(rng=rng, lower=lower, upper=upper, **kwargs)
        if self._mode == "point_mass":
            if rng is None:
                rng = get_global_rng()
            mass = float(self._Mn)
            lower_bound = -math.inf if lower is None else float(lower)
            upper_bound = math.inf if upper is None else float(upper)
            if math.isnan(lower_bound) or math.isnan(upper_bound) or lower_bound > upper_bound:
                raise EmptyTruncatedDistributionSupport(type(self).__name__, lower_bound, upper_bound)
            tolerance = _mass_tolerance(mass, 1.0)
            if lower_bound - tolerance <= mass <= upper_bound + tolerance:
                return mass
            raise EmptyTruncatedDistributionSupport(type(self).__name__, lower_bound, upper_bound)
        if rng is None:
            rng = get_global_rng()

        if lower is None and upper is None:
            while True:
                count = int(self._distribution.rvs(random_state=rng, **kwargs))
                if count > 0:
                    return float(self._q * count)

        requested_lower = -math.inf if lower is None else float(lower)
        requested_upper = math.inf if upper is None else float(upper)
        if math.isnan(requested_lower) or math.isnan(requested_upper) or requested_lower > requested_upper:
            raise EmptyTruncatedDistributionSupport(type(self).__name__, requested_lower, requested_upper)

        lower_count = 1 if requested_lower <= 0.0 else max(1, _mass_ceil_count(requested_lower, self._q))
        upper_count = math.inf if requested_upper == math.inf else _mass_floor_count(requested_upper, self._q)
        if upper_count < lower_count:
            raise EmptyTruncatedDistributionSupport(type(self).__name__, requested_lower, requested_upper)

        for _ in range(100_000):
            count = int(self._distribution.rvs(random_state=rng, **kwargs))
            if count < lower_count or count > upper_count:
                continue
            if count > 0:
                mass = float(self._q * count)
                tolerance = _mass_tolerance(mass, self._q)
                if requested_lower - tolerance <= mass <= requested_upper + tolerance:
                    return mass
        raise EmptyTruncatedDistributionSupport(type(self).__name__, requested_lower, requested_upper)

    def prob_mw(self, mw):
        if self._mode == "legacy":
            return super().prob_mw(mw)

        if isinstance(mw, RememberAdd):
            if self._mode == "point_mass":
                previous = float(mw.previous)
                value = float(mw.value)
                if previous < self._Mn <= value:
                    return 1.0
                return 0.0
            lower = float(mw.previous)
            upper = float(mw.value)
            if lower > upper:
                lower, upper = upper, lower
            lower_count = _mass_floor_count(lower, self._q) if np.isfinite(lower) else 0
            upper_count = _mass_floor_count(upper, self._q) if np.isfinite(upper) else math.inf
            if upper_count <= 0:
                return 0.0
            lower_cdf = 0.0 if lower_count <= 0 else self._distribution.cdf(lower_count)
            upper_cdf = self._distribution.cdf(upper_count) if np.isfinite(upper_count) else 1.0
            normalizer = 1.0 - self._distribution.pmf(0)
            return float((upper_cdf - lower_cdf) / normalizer)

        if self._mode == "point_mass":
            tol = _mass_tolerance(self._Mn, self._q)
            return 1.0 if abs(float(mw) - self._Mn) <= tol else 0.0

        count = _mass_lattice_count(float(mw), self._q)
        if count is None or count < 1:
            return 0.0
        normalizer = 1.0 - self._distribution.pmf(0)
        return float(self._distribution.pmf(count) / normalizer)


StochasticDistribution._known_distributions.append(Poisson)
