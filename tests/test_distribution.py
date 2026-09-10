# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only
import math

import numpy as np
import pytest
from lark.exceptions import UnexpectedInput
from scipy import stats

import g2rins
from g2rins.exception import EmptyTruncatedDistributionSupport
from g2rins.util import RememberAdd

EPSILON = 0.15
NSTAT = 2000


def _zero_truncated_poisson_draw(rng, mu):
    while True:
        count = int(rng.poisson(mu))
        if count > 0:
            return count


def test_empty_serialize():
    vector = g2rins.StochasticDistribution.get_empty_serial_vector()
    original = vector.copy()
    instance = g2rins.StochasticDistribution.from_serial_vector(vector)
    assert instance is None
    assert vector == original


@pytest.mark.parametrize("a", [0.01, 0.05, 0.1, 0.3, 0.5])
def test_flory_schulz(a):
    def mean(a):
        return 2 / a - 1

    def variance(a):
        return (2 - 2 * a) / a**2

    def skew(a):
        return (2 - a) / np.sqrt(2 - 2 * a)

    value = int(1 / a + 1)
    if value % 2:
        flory_schulz = g2rins.StochasticDistribution.make(f"flory_schulz({a})")
    else:
        flory_schulz = g2rins.FlorySchulz.make(f"flory_schulz({a})")

    assert isinstance(flory_schulz, g2rins.FlorySchulz)

    random_mw = flory_schulz.draw_mw()
    assert flory_schulz.prob_mw(random_mw) > 0

    data = np.asarray([flory_schulz.draw_mw() for i in range(4 * NSTAT)])

    assert np.abs((np.mean(data) - mean(a)) / mean(a)) < EPSILON
    assert np.abs((np.var(data) - variance(a)) / variance(a)) < EPSILON
    assert np.abs((stats.skew(data) - skew(a)) / skew(a)) < EPSILON
    assert str(flory_schulz) == f"|flory_schulz({a})|"
    assert flory_schulz.generable

    serial_vector = flory_schulz.get_serial_vector()
    new_instance = g2rins.StochasticDistribution.from_serial_vector(serial_vector)
    assert str(new_instance) == str(flory_schulz)


def test_flory_schulz_propagates_deterministic_sampling_failure(monkeypatch):
    """A deterministic inverse failure must be reported after one attempt.

    Retrying every RuntimeError recursively masks the original diagnostic and
    eventually exhausts the Python stack.
    """
    flory_schulz = g2rins.StochasticDistribution.make("flory_schulz(0.2)")
    expected_error = RuntimeError("deterministic inverse failure")
    calls = 0

    def fail_once_then_reject_retry(self, distribution, rng, lower, upper, kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("Flory-Schulz retried a deterministic failure")
        raise expected_error

    monkeypatch.setattr(
        g2rins.StochasticDistribution,
        "_draw_bounded_mw",
        fail_once_then_reject_retry,
    )

    with pytest.raises(RuntimeError) as caught:
        flory_schulz.draw_mw(
            rng=np.random.default_rng(0),
            lower=10.0,
            upper=20.0,
        )

    assert caught.value is expected_error
    assert calls == 1


def test_flory_schulz_unbounded_draw_uses_stable_exact_sampler():
    """Unbounded draws must not depend on SciPy's generic discrete inverse."""
    flory_schulz = g2rins.StochasticDistribution.make("flory_schulz(0.01)")
    actual_rng = np.random.default_rng(0)
    expected_rng = np.random.default_rng(0)

    actual = [flory_schulz.draw_mw(rng=actual_rng) for _ in range(100)]
    expected = [
        1 + expected_rng.negative_binomial(2, 0.01)
        for _ in range(100)
    ]

    np.testing.assert_array_equal(actual, expected)


def test_legacy_poisson_preserves_seeded_draws():
    poisson = g2rins.StochasticDistribution.make("poisson(25)")
    actual_rng = np.random.default_rng(42)
    expected_rng = np.random.default_rng(42)

    actual = [poisson.draw_mw(rng=actual_rng) for _ in range(100)]
    expected = [expected_rng.poisson(25) for _ in range(100)]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("mu", "sigma"), [(100.0, 10.0), (200.0, 100.0), (500.0, 1.0), (600.0, 0.0)])
def test_gauss(mu, sigma):
    def mean(mu, sigma):
        return mu

    def variance(mu, sigma):
        return sigma**2

    def skew(mu, sigma):
        return 0

    gauss = g2rins.StochasticDistribution.make(f"gauss({mu}, {sigma})")
    assert isinstance(gauss, g2rins.Gauss)

    example = gauss.draw_mw()
    assert gauss.prob_mw(example) > 0

    data = np.asarray([gauss.draw_mw() for i in range(NSTAT)])

    assert np.abs((np.mean(data) - mean(mu, sigma)) / mean(mu, sigma)) < EPSILON
    if sigma > 0:
        assert np.abs((np.var(data) - variance(mu, sigma)) / variance(mu, sigma)) < EPSILON

    assert str(gauss) == f"|gauss({mu}, {sigma})|"
    assert gauss.generable

    serial_vector = gauss.get_serial_vector()
    new_instance = g2rins.StochasticDistribution.from_serial_vector(serial_vector)
    assert str(new_instance) == str(gauss)


@pytest.mark.parametrize(("low", "high"), [(10.0, 100.0), (200.0, 1000.0), (50.0, 100.0), (0.0, 600.0)])
def test_uniform(low, high):
    def mean(low, high):
        return 0.5 * (low + high)

    def variance(low, high):
        return 1 / 12.0 * (high - low) ** 2

    def skew(low, high):
        return 0

    uniform = g2rins.StochasticDistribution.make(f"uniform({low}, {high})")
    assert isinstance(uniform, g2rins.Uniform)

    assert uniform.prob_mw(uniform.draw_mw()) > 0

    data = np.asarray([uniform.draw_mw() for i in range(NSTAT)])

    assert np.abs((np.mean(data) - mean(low, high)) / mean(low, high)) < EPSILON
    assert np.abs((np.var(data) - variance(low, high)) / variance(low, high)) < EPSILON

    assert str(uniform) == f"|uniform({low}, {high})|"
    assert uniform.generable

    serial_vector = uniform.get_serial_vector()
    new_instance = g2rins.StochasticDistribution.from_serial_vector(serial_vector)
    assert str(new_instance) == str(uniform)


@pytest.mark.parametrize(("Mw", "factor"), [(11.3e3, 4)])
def test_schulz_zimm(Mw, factor):
    def mean(Mn, z):
        return Mn

    def variance(Mn, z):
        return Mn**2 / z

    Mn = Mw / (1 / factor + 1)
    schulz_zimm = g2rins.StochasticDistribution.make(f"schulz_zimm({Mw}, {Mn})")
    assert isinstance(schulz_zimm, g2rins.SchulzZimm)
    z = schulz_zimm._z

    data = []
    for _i in range(100 * NSTAT):
        data.append(schulz_zimm.draw_mw())
    data = np.asarray(data)

    # x = np.linspace(1e3, 40e3, 1000).astype(int)
    # plt.plot(x, schulz_zimm._distribution.pmf(x, z=schulz_zimm._z, Mn=schulz_zimm._Mn))
    # plt.show()

    assert np.abs((np.mean(data) - mean(Mn, z)) / mean(Mn, z)) < EPSILON
    assert np.abs((np.var(data) ** 0.5 - variance(Mn, z) ** 0.5) / variance(Mn, z) ** 0.5) < EPSILON
    assert str(schulz_zimm) == f"|schulz_zimm({Mw}, {Mn})|"
    assert schulz_zimm.generable

    serial_vector = schulz_zimm.get_serial_vector()
    new_instance = g2rins.StochasticDistribution.from_serial_vector(serial_vector)
    assert str(new_instance) == str(schulz_zimm)


@pytest.mark.parametrize(("M", "D"), [(11.3e3, 1.1), (5.3e3, 1.5), (20.3e3, 2.0)])
def test_log_normal(M, D):
    def mean(M, D):
        return M

    log_normal = g2rins.StochasticDistribution.make(f"log_normal({M}, {D})")
    assert isinstance(log_normal, g2rins.LogNormal)

    data = []
    for _i in range(NSTAT):
        d = log_normal.draw_mw()
        data.append(d)
    data = np.asarray(data)

    # import matplotlib.pyplot as plt
    # x = np.linspace(1e3, 40e3, 1000)
    # plt.plot(x, log_normal._distribution.pdf(x, M=log_normal._M, D=log_normal._D))
    # plt.show()

    assert np.abs((np.mean(data) - mean(M, D))) / mean(M, D) < EPSILON
    assert str(log_normal) == f"|log_normal({M}, {D})|"
    assert log_normal.generable

    serial_vector = log_normal.get_serial_vector()
    new_instance = g2rins.StochasticDistribution.from_serial_vector(serial_vector)
    assert str(new_instance) == str(log_normal)


@pytest.mark.parametrize("M", [11.3e3, 5.3e3, 20.3e3])
def test_poisson(M):
    def mean(M):
        return M

    def variance(M):
        return M

    poisson = g2rins.StochasticDistribution.make(f"poisson({M})")
    assert isinstance(poisson, g2rins.Poisson)

    data = []
    for _i in range(NSTAT):
        d = poisson.draw_mw()
        data.append(d)
    data = np.asarray(data)

    assert np.abs((np.mean(data) - mean(M))) / mean(M) < EPSILON
    assert np.abs((np.var(data) - variance(M))) / variance(M) < EPSILON
    assert str(poisson) == f"|poisson({M})|"
    assert poisson.generable

    serial_vector = poisson.get_serial_vector()
    new_instance = g2rins.StochasticDistribution.from_serial_vector(serial_vector)
    assert str(new_instance) == str(poisson)


def test_poisson_and_flory_molar_mass_forms_round_trip():
    poisson = g2rins.StochasticDistribution.make("poisson(1100.0, 1000.0)")
    assert isinstance(poisson, g2rins.Poisson)
    assert poisson._Mw == 1100.0
    assert poisson._Mn == 1000.0
    assert str(poisson) == "|poisson(1100.0, 1000.0)|"

    flory = g2rins.StochasticDistribution.make("flory_schulz(1.4, 1.0)")
    assert isinstance(flory, g2rins.FlorySchulz)
    assert flory._Mw == 1.4
    assert flory._Mn == 1.0
    assert str(flory) == "|flory_schulz(1.4, 1.0)|"

    assert str(g2rins.StochasticDistribution.from_serial_vector(poisson.get_serial_vector())) == str(poisson)
    assert str(g2rins.StochasticDistribution.from_serial_vector(flory.get_serial_vector())) == str(flory)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("poisson(11,963.3, 10,751.9)", "|poisson(11963.3, 10751.9)|"),
        ("flory_schulz(11,963.3, 7519.6)", "|flory_schulz(11963.3, 7519.6)|"),
    ],
)
def test_poisson_and_flory_accept_grouped_thousands_separators(text, expected):
    assert str(g2rins.StochasticDistribution.make(text)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "poisson()",
        "poisson(1,2,3)",
        "flory_schulz()",
        "flory_schulz(1,2,3)",
    ],
)
def test_poisson_and_flory_reject_invalid_arity_at_parse_time(text):
    with pytest.raises(UnexpectedInput):
        g2rins.StochasticDistribution.make(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("poisson(500,)", "|poisson(500.0)|"),
        ("poisson(600,500,)", "|poisson(600.0, 500.0)|"),
        ("flory_schulz(0.2,)", "|flory_schulz(0.2)|"),
        ("flory_schulz(600,500,)", "|flory_schulz(600.0, 500.0)|"),
    ],
)
def test_poisson_and_flory_preserve_trailing_comma_compatibility(text, expected):
    assert str(g2rins.StochasticDistribution.make(text)) == expected


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("poisson(0.0, 1.0)", ValueError),
        ("poisson(1.0, 0.0)", ValueError),
        ("poisson(1e309, 1.0)", ValueError),
        ("flory_schulz(0.5, 1.0)", ValueError),
        ("flory_schulz(1.0, 0.0)", ValueError),
        ("flory_schulz(2.0, 1.0)", ValueError),
        ("flory_schulz(1e309, 1.0)", ValueError),
    ],
)
def test_two_argument_distributions_reject_invalid_molar_mass_parameters(text, error):
    with pytest.raises(error):
        g2rins.StochasticDistribution.make(text)


@pytest.mark.parametrize(
    ("text", "expected_q", "expected_parameter"),
    [
        ("poisson(1100, 1000)", 891.916362205061, 0.233299496020568),
        ("flory_schulz(1400, 1000)", 600.0, 0.6),
    ],
)
def test_molar_mass_forms_have_requested_analytic_moments(text, expected_q, expected_parameter):
    distribution = g2rins.StochasticDistribution.make(text)

    assert distribution._q == pytest.approx(expected_q)
    assert distribution.reference_mw() == 1000.0
    if isinstance(distribution, g2rins.Poisson):
        assert distribution._distribution.kwds["mu"] == pytest.approx(expected_parameter)
    else:
        assert distribution._fls_a == pytest.approx(expected_parameter)

    if isinstance(distribution, g2rins.Poisson):
        mu = float(distribution._distribution.kwds["mu"])
        upper_prob = 1.0 - math.exp(-mu)
        mean_count = mu / upper_prob
        second_count_moment = (mu + mu**2) / upper_prob
        mean_mass = distribution._q * mean_count
        weight_average_mass = distribution._q * second_count_moment / mean_count
        assert mean_mass == pytest.approx(1000.0)
        assert weight_average_mass == pytest.approx(1100.0)
        support_lower, support_upper = distribution.support_mw()
        assert support_lower == pytest.approx(expected_q)
        assert support_upper == np.inf
    else:
        mean_count = float(distribution._distribution.mean())
        second_count_moment = float(distribution._distribution.var() + mean_count**2)
        mean_mass = distribution._q * mean_count
        weight_average_mass = distribution._q * second_count_moment / mean_count
        assert mean_mass == pytest.approx(1000.0)
        assert weight_average_mass == pytest.approx(1400.0)
        support_lower, support_upper = distribution.support_mw()
        assert support_lower == pytest.approx(expected_q)
        assert support_upper == np.inf


@pytest.mark.parametrize(
    ("text", "count_draw"),
    [
        ("poisson(1100, 1000)", lambda rng, mu=0.233299496020568: _zero_truncated_poisson_draw(rng, mu)),
        ("flory_schulz(1400, 1000)", lambda rng: rng.geometric(0.6)),
    ],
)
def test_molar_mass_forms_match_exact_numpy_count_draws(text, count_draw):
    distribution = g2rins.StochasticDistribution.make(text)
    actual_rng = np.random.default_rng(918)
    expected_rng = np.random.default_rng(918)

    actual = [distribution.draw_mw(actual_rng) for _ in range(100)]
    expected = [distribution._q * count_draw(expected_rng) for _ in range(100)]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name", ["poisson", "flory_schulz"])
def test_molar_mass_forms_are_covariant_in_mass_units(name):
    original = g2rins.StochasticDistribution.make(f"{name}(1100, 1000)" if name == "poisson" else "flory_schulz(1400,1000)")
    scaled = g2rins.StochasticDistribution.make(f"{name}(11000, 10000)" if name == "poisson" else "flory_schulz(14000,10000)")
    original_rng = np.random.default_rng(33)
    scaled_rng = np.random.default_rng(33)

    original_draws = np.asarray([original.draw_mw(original_rng) for _ in range(100)])
    scaled_draws = np.asarray([scaled.draw_mw(scaled_rng) for _ in range(100)])
    np.testing.assert_allclose(scaled_draws, 10.0 * original_draws)
    assert scaled._q == pytest.approx(10.0 * original._q)


@pytest.mark.parametrize("name", ["poisson", "flory_schulz"])
def test_molar_mass_point_masses(name):
    distribution = g2rins.StochasticDistribution.make(f"{name}(1000, 1000)")

    assert distribution.draw_mw(np.random.default_rng(0)) == 1000.0
    assert distribution.draw_mw(np.random.default_rng(0), lower=1000, upper=1000) == 1000.0
    assert distribution.prob_mw(1000.0) == 1.0
    assert distribution.prob_mw(1001.0) == 0.0
    assert distribution.reference_mw() == 1000.0
    with pytest.raises(EmptyTruncatedDistributionSupport):
        distribution.draw_mw(np.random.default_rng(0), upper=999.0)


@pytest.mark.parametrize("name", ["poisson", "flory_schulz"])
def test_scaled_discrete_bounds_are_inclusive_and_tolerant(name):
    distribution = g2rins.StochasticDistribution.make(f"{name}(1.1, 1.0)")
    quantum = distribution._q
    count = 3 if name == "poisson" else 11
    mass = count * quantum
    rounded_bound = float(f"{mass:.15g}")

    draws = [
        distribution.draw_mw(
            np.random.default_rng(seed),
            lower=rounded_bound,
            upper=rounded_bound,
        )
        for seed in range(5)
    ]
    np.testing.assert_allclose(draws, mass)

    with pytest.raises(EmptyTruncatedDistributionSupport):
        distribution.draw_mw(np.random.default_rng(0), lower=mass + 0.25 * quantum, upper=mass + 0.75 * quantum)
    with pytest.raises(EmptyTruncatedDistributionSupport):
        distribution.draw_mw(np.random.default_rng(0), lower=np.nan, upper=mass)


@pytest.mark.parametrize("name", ["poisson", "flory_schulz"])
def test_scaled_discrete_one_sided_and_remote_tail_bounds(name):
    text = "poisson(1100, 1000)" if name == "poisson" else "flory_schulz(1400, 1000)"
    distribution = g2rins.StochasticDistribution.make(text)
    quantum = distribution._q

    lower_draw = distribution.draw_mw(np.random.default_rng(4), lower=3 * quantum)
    upper_draw = distribution.draw_mw(np.random.default_rng(5), upper=5 * quantum)
    if name == "poisson":
        tail_lower, tail_upper = 1.0 * quantum, 3.0 * quantum
    else:
        tail_lower, tail_upper = 1000.0 * quantum, 1002.0 * quantum
    tail_draw = distribution.draw_mw(
        np.random.default_rng(6),
        lower=tail_lower,
        upper=tail_upper,
    )

    assert lower_draw >= 3 * quantum
    assert upper_draw <= 5 * quantum
    assert tail_lower <= tail_draw <= tail_upper


@pytest.mark.parametrize(
    ("text", "minimum_count"),
    [("poisson(1100, 1000)", 1), ("flory_schulz(1400, 1000)", 1)],
)
def test_scaled_discrete_probability_uses_mass_lattice(text, minimum_count):
    distribution = g2rins.StochasticDistribution.make(text)
    quantum = distribution._q
    count = minimum_count + 2
    mass = quantum * count

    expected = distribution._distribution.pmf(count) / (1.0 - distribution._distribution.pmf(0)) if isinstance(distribution, g2rins.Poisson) else distribution._distribution.pmf(count)
    assert distribution.prob_mw(mass) == pytest.approx(expected)
    assert distribution.prob_mw(np.nextafter(mass, np.inf)) == pytest.approx(expected)
    assert distribution.prob_mw(mass + 0.25 * quantum) == 0.0

    interval = RememberAdd(mass - quantum)
    interval += quantum
    assert distribution.prob_mw(interval) == pytest.approx(expected)


def test_poisson_mass_form_excludes_zero_mass_targets():
    distribution = g2rins.StochasticDistribution.make("poisson(1100, 1000)")
    assert distribution.prob_mw(0.0) == 0.0
    assert distribution.draw_mw(np.random.default_rng(0)) > 0.0


@pytest.mark.parametrize("name", ["poisson", "flory_schulz"])
def test_point_mass_probability_uses_interval_convention(name):
    distribution = g2rins.StochasticDistribution.make(f"{name}(1000,1000)")
    selected = RememberAdd(999.0)
    selected += 1.0
    excluded = RememberAdd(1000.0)
    excluded += 1.0

    assert distribution.prob_mw(selected) == 1.0
    assert distribution.prob_mw(excluded) == 0.0


def test_legacy_poisson_zero_remains_a_point_mass():
    distribution = g2rins.StochasticDistribution.make("poisson(0)")
    assert distribution.draw_mw(np.random.default_rng(0)) == 0.0
    assert str(distribution) == "|poisson(0.0)|"


def test_serialized_distribution_layouts_are_versioned_and_strict():
    assert g2rins.StochasticDistribution.get_empty_serial_vector() == [-1.0] * 12
    assert g2rins.StochasticDistribution.from_serial_vector([-1.0] * 10) is None
    assert g2rins.StochasticDistribution.from_serial_vector([-1.0] * 12) is None

    legacy_flory = [0.2] + [-1.0] * 9
    legacy_poisson = [-1.0] * 9 + [500.0]
    assert str(g2rins.StochasticDistribution.from_serial_vector(legacy_flory)) == "|flory_schulz(0.2)|"
    assert str(g2rins.StochasticDistribution.from_serial_vector(legacy_poisson)) == "|poisson(500.0)|"

    for malformed_length in (0, 1, 9, 11, 13):
        with pytest.raises(ValueError, match="serialization length"):
            g2rins.StochasticDistribution.from_serial_vector([-1.0] * malformed_length)


@pytest.mark.parametrize(
    "text",
    [
        "flory_schulz(0.2)",
        "flory_schulz(1400,1000)",
        "schulz_zimm(1400,1000)",
        "gauss(1000,100)",
        "uniform(100,1000)",
        "log_normal(1000,1.4)",
        "poisson(500)",
        "poisson(1100,1000)",
    ],
)
def test_every_distribution_round_trips_through_current_layout(text):
    distribution = g2rins.StochasticDistribution.make(text)
    reconstructed = g2rins.StochasticDistribution.from_serial_vector(distribution.get_serial_vector())
    assert str(reconstructed) == str(distribution)


@pytest.mark.parametrize("distribution", ["poisson(380,300)", "flory_schulz(400,300)"])
def test_molar_mass_forms_round_trip_through_graph_and_generate_ensemble(distribution):
    text = f"{{[] [<]CC[>]; C[>]; [<][H] []}}|{distribution}|"
    graph_creator = g2rins.G2rins.make(text).get_graph_creator()
    graph = graph_creator.get_generative_graph()
    first_node = next(iter(graph.nodes))
    serial_vectors = graph.nodes[first_node]["molecular_weight_distribution"]

    assert len(serial_vectors) == 10
    reconstructed = g2rins.StochasticDistribution.from_serial_vector(serial_vectors[0])
    assert str(reconstructed) == str(g2rins.StochasticDistribution.make(distribution))
    assert reconstructed.draw_mw(np.random.default_rng(7)) >= 0.0

    ensemble = graph_creator.get_ensemble_creator().create_ensemble(
        3,
        ensemble_info=True,
        seed=7,
    )
    assert len(ensemble.chains) == 3
    assert all(np.isfinite(mass) and mass > 0.0 for mass in ensemble.molecular_weights)
