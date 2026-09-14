"""Scalar moment range cases with independently known means and standard errors."""

import math

from dpt.statistics import mean_standard_error


def test_raw_sum_overflow_does_not_destroy_a_finite_mean() -> None:
    assert mean_standard_error((1e308, 1e308)) == (1e308, 0)
    mean, error = mean_standard_error((-1.7e308, 1.7e308))
    assert mean == 0
    assert math.isclose(error, 1.7e308, rel_tol=1e-15)


def test_uniform_subnormal_mean_survives_replicate_normalisation() -> None:
    tiny = math.ulp(0.0)
    assert mean_standard_error((tiny,) * 16) == (tiny, 0)


def test_cancelling_extremes_do_not_erase_a_representable_small_mean() -> None:
    mean, error = mean_standard_error((1e308, -1e308, 1e-300))
    assert mean == 1e-300 / 3
    assert math.isfinite(error) and error > 0


def test_standard_error_centres_before_rounding_a_subnormal_mean() -> None:
    tiny = math.ulp(0.0)
    mean, error = mean_standard_error((0.0, 2 * tiny))
    assert mean == tiny
    assert error == tiny


def test_centred_deviation_retains_adjacent_large_values() -> None:
    left = 2.0**500
    right = left + 4 * math.ulp(left)
    mean, error = mean_standard_error((left, right))
    assert mean == left + 2 * math.ulp(left)
    assert math.isclose(error, 2 * math.ulp(left), rel_tol=1e-15)
