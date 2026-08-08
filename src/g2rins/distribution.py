# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""
This module defines base classes for handling stochastic generation based on
various statistical distributions.
"""
import math
from abc import abstractmethod
from typing import Any, ClassVar, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
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
        Returns True if a statistical distribution is associated with this object.
        """
        return self._distribution is not None

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
                return known_distr.make(text)
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
        vector: List[float] = []
        for distr_type in cls._known_distributions:
            vector += list(distr_type.default_serialize())
        return vector

    def get_serial_vector(self) -> List[float]:
        """
        Returns the serialization vector for this specific stochastic distribution instance.

        The vector contains the serialized parameters of this instance, with default
        serialization values for other known distribution types.
        """
        vector: List[float] = []
        for distr_type in self._known_distributions:
            if type(self) is distr_type:
                vector += list(self.serialize())
            else:
                vector += list(distr_type.default_serialize())
        return vector

    @classmethod
    def from_serial_vector(cls: Type[_T], vector: List[float]) -> Optional[_T]:
        """
        Creates a StochasticDistribution instance from a serialization vector.

        It iterates through known distributions, extracts the corresponding
        segment from the vector, and if it's not the default serialization,
        creates an instance of that distribution with the deserialized parameters.

        Args:
            vector (List[float]): The serialization vector.

        Returns:
            Optional[_T]: An instance of a StochasticDistribution subclass if
                           the vector contains non-default serialization for one,
                           otherwise None.

        Raises:
            ValueError: If the vector contains non-default serialization for more
                        than one known distribution.
        """
        candidates: List[Tuple[float, ...]] = []
        type_candidates: List[Type[_T]] = []
        for distr_type in cls._known_distributions:
            default_serial = distr_type.default_serialize()
            given_serial = tuple((vector.pop(0) for _ in default_serial))
            if default_serial != given_serial:
                candidates.append(given_serial)
                type_candidates.append(distr_type)

        if not candidates:
            return None

        if len(candidates) != 1:
            raise ValueError("The passed vector did not contain only one candidate for the distribution.")
        distr_type = type_candidates[0]
        params = candidates[0]
        return distr_type.make(distr_type.token_name_snake_case + str(params))

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


# TODO: Flory Schulz samples chain length rather than molecular weight. We need to implement it differently in ensemble_creator.py
class FlorySchulz(StochasticDistribution):
    """
    Flory-Schulz distribution of molecular weights for geometrically distributed chain lengths.

    :math:`W_a(N) = a^2 N (1-a)^{N-1}`

    where :math:`0<a<1` is the experimentally determined constant of remaining monomers and :math:`N` is the chain length.

    The textual representation of this distribution is: `flory_schulz(a)`
    """

    class flory_schulz_gen(stats.rv_discrete):
        """Flory Schulz distribution."""

        def _rvs(self, fls_a, size=None, random_state=None):
            # If X ~ NegativeBinomial(2, a), then N = X + 1 has
            # P(N=n) = a**2 * n * (1-a)**(n-1).  Sampling this equivalent
            # form avoids SciPy's generic discrete PPF, which can stall while
            # inverting the infinite support.
            return random_state.negative_binomial(2, fls_a, size=size) + 1

        def _pmf(self, k, fls_a):
            return fls_a**2 * k * (1 - fls_a) ** (k - 1)

        def _sf(self, k, fls_a):
            # Sum_{n=k+1..inf} a^2*n*(1-a)^(n-1)
            #   = (1-a)^k * (1 + a*k).
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

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        """
        Creates a FlorySchulz instance from its textual representation.

        Args:
            text (str): The textual representation, e.g., 'flory_schulz(0.9)'.

        Returns:
            Self: A FlorySchulz instance.
        """
        # We use G2rinsBase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return G2rinsBase.make.__func__(cls, text)

    def __init__(self, children: List[Any]):
        """
        Initialization of Flory-Schulz distribution object.

        Args:
            children (List[Any]): List of parsed children, expected to contain the 'a' parameter as a float.
        """
        super().__init__(children)

        fls_a: Optional[float] = None
        for child in self._children:
            if isinstance(child, float):
                fls_a = child

        if not 0 < fls_a < 1:
            raise RuntimeError(f"The Flory-Schulz distribution needs an a parameter between 0, and 1. But got {fls_a}.")

        self._fls_a = fls_a
        self._distribution = self.flory_schulz_gen(name="Flory-Schulz", a=1)(fls_a=self._fls_a)

    def generate_string(self, extension: bool) -> str:
        """
        Generates the textual representation of the Flory-Schulz distribution.

        Args:
            extension (bool): Whether to include the '|' delimiters.

        Returns:
            str: The textual representation, e.g., '|flory_schulz(0.9)|'.
        """
        if extension:
            return f"|flory_schulz({self._fls_a})|"
        return ""

    @property
    def generable(self) -> bool:
        """
        Returns True if the distribution is initialized (i.e., the 'a' parameter is set).
        """
        return self._distribution is not None

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Returns the default serialization for FlorySchulz (a tuple with one -1.0).
        """
        return cls._default_serialize(1)

    def serialize(self) -> Tuple[float, ...]:
        """
        Serializes the 'a' parameter of the FlorySchulz distribution.
        """
        return (self._fls_a,)

    def prob_mw(self, mw):
        return super().prob_mw(mw)


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
    # TODO: implement prob_mw()
    """
    Poisson distribution of molecular weights for chain lengths.
    Flory, P. J. Molecular size distribution in ethylene oxide polymers. Journal of the American chemical society 1940, 62, 1561–1565.

    The textual representation of this distribution is: `poisson(N)`
    """

    _N: Optional[float] = None  # Mean number of repeating units

    def __init__(self, children: List[Any]):
        """
        Initialization of Poisson distribution object.

        Args:
            children (List[Any]): List of parsed children, expected to contain the mean (N) as a float.
        """
        super().__init__(children)
        N: Optional[float] = None
        for child in self._children:
            if isinstance(child, float):
                N = child

        self._N = N
        self._distribution = stats.poisson(mu=self._N)

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Returns the default serialization for Poisson (a tuple with one -1.0).
        """
        return cls._default_serialize(1)

    def serialize(self) -> Tuple[float, ...]:
        """
        Serializes the mean (N) of the Poisson distribution.
        """
        return (self._N,)

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        """
        Creates a Poisson instance from its textual representation.

        Args:
            text (str): The textual representation, e.g., 'poisson(10)'.

        Returns:
            Self: A Poisson instance.
        """
        # We use G2rinsBase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return G2rinsBase.make.__func__(cls, text)

    def generate_string(self, extension: bool) -> str:
        """
        Generates the textual representation of the Poisson distribution.

        Args:
            extension (bool): Whether to include the '|' delimiters.

        Returns:
            str: The textual representation, e.g., '|poisson(10)|'.
        """
        if extension:
            return f"|poisson({self._N})|"
        return ""

    @property
    def generable(self) -> bool:
        """
        Returns True if the distribution is initialized (i.e., N is set).
        """
        return self._distribution is not None


StochasticDistribution._known_distributions.append(Poisson)
