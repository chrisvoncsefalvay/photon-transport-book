"""Independent exact cases for the manuscript's analytic figure payloads."""

import json
import math
from dataclasses import asdict
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.validation.book_illustrations import (
    aperture_difference,
    charge_assignment,
    gaussian_overlap,
    lattice_values,
    moving_interval,
    nuisance_information_fraction,
    quadratic_midpoint,
    slab_interval,
    sphere_chord,
    transmission_difference,
)

approx: Any = pytest.approx  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def test_hat_and_clamp_have_distinct_outer_half_cells() -> None:
    outer = lattice_values((1.0, 0.0), -0.75)
    assert outer.zero_hat_mm_inverse == 0.25
    assert outer.half_cell_clamp_mm_inverse == 0
    interior = lattice_values((1.0, 0.0), -0.25)
    assert interior.zero_hat_mm_inverse == 0.75
    assert interior.half_cell_clamp_mm_inverse == 1
    assert lattice_values((0.0, 2.0), 1.5).half_cell_clamp_mm_inverse == 2


def test_finite_slab_endpoints_and_parallel_miss() -> None:
    bounds = ((-0.5, -0.5, -0.5), (1.5, 1.5, 1.5))
    crossing = slab_interval((-1.5, 0.5, 0.5), (2.5, 0.5, 0.5), *bounds)
    assert (crossing.entry, crossing.exit, crossing.chord_mm) == (0.25, 0.75, 2)
    assert crossing.axis_intervals[1:] == (None, None)
    tangent = slab_interval((-1.5, 0.5, 0.5), (0.5, 2.5, 0.5), *bounds)
    assert (tangent.entry, tangent.exit, tangent.chord_mm, tangent.state) == (
        0.5,
        0.5,
        0,
        "tangent",
    )
    miss = slab_interval((-1.5, 2.0, 0.5), (2.5, 2.0, 0.5), *bounds)
    assert miss.state == "miss" and miss.entry is None
    inside = slab_interval((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), *bounds)
    assert inside.chord_mm == 0.5
    json.dumps(asdict(miss), allow_nan=False)


@pytest.mark.parametrize("count", [10, 20, 40])
def test_quadratic_worked_errors_against_rational_sum(count: int) -> None:
    result = quadratic_midpoint(100, 0.01, 0, 1e-6, count)
    step = Fraction(100, count)
    midpoint = sum(
        (Fraction(1, 100) + ((i + Fraction(1, 2)) * step) ** 2 / 1_000_000) * step
        for i in range(count)
    )
    expected = Fraction(4, 3) - midpoint
    assert result.midpoint == approx(float(midpoint), rel=2e-15)
    assert result.signed_error == approx(float(expected), abs=4e-16)
    assert result.predicted_error == approx(float(expected), rel=2e-15)


def test_quadratic_large_coordinates_do_not_square_before_small_coefficient() -> None:
    constant = quadratic_midpoint(1e200, 1e-200, 0, 0, 4)
    assert constant.exact == approx(1.0)
    assert constant.midpoint == approx(1.0)
    result = quadratic_midpoint(1e200, 0, 0, 1e-300, 4)
    with localcontext() as context:
        context.prec = 80
        length, coefficient = Decimal.from_float(1e200), Decimal.from_float(1e-300)
        expected = float(coefficient * length**3 / 3)
    assert result.exact == approx(expected, rel=2e-15)
    assert math.isfinite(result.midpoint)


def test_moving_indicator_reports_crossings_without_inventing_a_derivative() -> None:
    regular = moving_interval(2, 8, 10, 0.5, 5)
    assert (regular.exact, regular.midpoint, regular.exact_entry_derivative_per_mm) == (3, 3, -0.5)
    assert regular.branch_entry_derivative_per_mm == 0
    crossing = moving_interval(3, 8, 10, 0.5, 5)
    assert crossing.branch_entry_derivative_per_mm is None
    assert crossing.branch_state == "sample-crossing"


def test_transmission_central_and_taylor_truncation_orders() -> None:
    coarse = transmission_difference(1, 0.7, 0.01)
    fine = transmission_difference(1, 0.7, 0.005)
    assert coarse.derivative_absolute_error / fine.derivative_absolute_error == approx(4, rel=1e-4)
    assert coarse.taylor_remainder / fine.taylor_remainder == approx(4, rel=0.002)
    assert abs(coarse.central_derivative - float(coarse.reference_central_derivative)) < 1e-14


def test_transmission_large_step_keeps_representable_subnormal_difference() -> None:
    result = transmission_difference(2, 1e-308, 1e308)
    with localcontext() as context:
        context.prec = 100
        scale, step = Decimal.from_float(1e-308), Decimal.from_float(1e308)
        exact = ((-Decimal(2) - scale * step).exp() - (-Decimal(2) + scale * step).exp()) / (
            2 * step
        )
    target = float(exact)
    assert result.central_derivative != 0
    assert abs(result.central_derivative - target) <= 2 * math.ulp(target)


def test_sphere_chord_known_triangle_and_tangent_state() -> None:
    result = sphere_chord(5, 0.1, 3)
    assert result.optical_depth == approx(0.8)
    assert result.derivative_per_mm == approx(-0.15)
    tangent = sphere_chord(5, 0.1, 5)
    assert tangent.derivative_per_mm is None and tangent.derivative_state == "inward-divergence"
    assert sphere_chord(5, 0, 5).derivative_per_mm == 0
    assert sphere_chord(5, 0.1, 6).derivative_state == "outside"
    json.dumps(asdict(tangent), allow_nan=False)


def test_sphere_near_tangency_large_radius_against_decimal() -> None:
    radius, mu = 1e200, 1e-200
    impact = math.nextafter(radius, 0)
    result = sphere_chord(radius, mu, impact)
    with localcontext() as context:
        context.prec = 100
        r, b, attenuation = map(Decimal.from_float, (radius, impact, mu))
        root = (r * r - b * b).sqrt()
        depth = float(2 * attenuation * root)
        derivative = float(-2 * attenuation * b / root)
    assert result.optical_depth == approx(depth, rel=3e-15)
    assert result.derivative_per_mm == approx(derivative, rel=3e-15)


def test_gaussian_overlap_is_even_and_loss_derivative_odd() -> None:
    positive, negative = gaussian_overlap(1, 2), gaussian_overlap(1, -2)
    assert positive.overlap == approx(math.exp(-1))
    assert positive.derivative_per_pixel == approx(math.exp(-1))
    assert positive.discrepancy == negative.discrepancy
    assert positive.derivative_per_pixel == -negative.derivative_per_pixel
    assert gaussian_overlap(1e-300, 1e300).derivative_per_pixel == 0


def test_nuisance_fraction_known_and_nearly_parallel_cases() -> None:
    assert nuisance_information_fraction(0.9) == approx(0.19)
    assert nuisance_information_fraction(0.99) == approx(0.0199)
    assert nuisance_information_fraction(1) == 0
    value = math.nextafter(1, 0)
    with localcontext() as context:
        context.prec = 80
        reference = float(1 - Decimal.from_float(value) ** 2)
    assert abs(nuisance_information_fraction(value) - reference) <= math.ulp(reference)


def test_charge_laws_share_mean_but_not_covariance() -> None:
    moments = charge_assignment(1000, (0.25, 0.75))
    assert moments.mean == (250, 750)
    assert moments.shared_covariance == ((62.5, 187.5), (187.5, 562.5))
    assert moments.exclusive_covariance == ((250, 0), (0, 750))
    assert sum(map(sum, moments.shared_covariance)) == 1000
    assert sum(map(sum, moments.exclusive_covariance)) == 1000


def test_aperture_moments_against_bernoulli_law() -> None:
    result = aperture_difference(10, 5, 1, 1000, 100)
    probability = Fraction(1, 5)
    contribution = Fraction(1000, 2)
    variance = float(contribution**2 * probability * (1 - probability) / 100)
    assert result.derivative_per_mm == 100
    assert result.standard_error_per_mm**2 == approx(variance)
    assert result.relative_standard_error == approx(0.2)
    assert result.zero_contribution_probability == approx(float((1 - probability) ** 100))


def test_aperture_scale_extremes_remain_finite() -> None:
    result = aperture_difference(1e100, 5e99, 1e-200, 1e100, 1_000_000_000)
    assert result.derivative_per_mm == 1
    assert result.relative_standard_error == approx(math.sqrt(5e290), rel=1e-13)
    assert result.zero_contribution_probability == 1
    json.dumps(asdict(result), allow_nan=False)


def test_invalid_domains_and_unrepresentable_outputs_fail_explicitly() -> None:
    with pytest.raises(ContractError):
        sphere_chord(0, 1, 0)
    with pytest.raises(ContractError):
        nuisance_information_fraction(1.0001)
    with pytest.raises(ContractError):
        charge_assignment(100, (0.2, 0.2))
    with pytest.raises(ContractError):
        aperture_difference(10, 5, 5, 100, 10)
    with pytest.raises(ContractError):
        transmission_difference(1, 2, 1)
    with pytest.raises(ContractError):
        quadratic_midpoint(10, 1, 0, 0, True)
    with pytest.raises(NumericalError):
        sphere_chord(1e308, 1e308, 0)
