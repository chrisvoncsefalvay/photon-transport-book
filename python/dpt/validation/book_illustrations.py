"""Closed-form manuscript illustrations, separate from production transport.

These CPU reference functions evaluate the formulas stated beside the book's
figure placeholders. They neither sample photon histories nor infer anatomy,
material properties or measurement data. Units and finite domains are explicit;
a mathematically singular endpoint is represented by a labelled absent value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext

from dpt.contracts import ContractError, NumericalError, finite_scalar, integer


def _positive(value: float, name: str) -> float:
    result = finite_scalar(value, name, minimum=0)
    if result == 0:
        raise ContractError(f"{name} must be strictly positive")
    return result


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise NumericalError("analytic result exceeds binary64 range")
    return value


def _product(*values: float) -> float:
    """Retain finite compensated-scale products without intermediate overflow."""
    if any(value == 0 for value in values):
        return 0.0
    mantissa, exponent = 1.0, 0
    for value in values:
        coefficient, power = math.frexp(value)
        mantissa *= coefficient
        exponent += power
    try:
        return _finite(math.ldexp(mantissa, exponent))
    except OverflowError as error:
        raise NumericalError("analytic product exceeds binary64 range") from error


def _exp(value: float) -> float:
    try:
        return _finite(math.exp(value))
    except OverflowError as error:
        raise NumericalError("analytic exponential exceeds binary64 range") from error


@dataclass(frozen=True, slots=True)
class LatticeValues:
    grid_coordinate: float
    cell_mm_inverse: float
    zero_hat_mm_inverse: float
    half_cell_clamp_mm_inverse: float


def lattice_values(coefficients: tuple[float, ...], coordinate: float) -> LatticeValues:
    """One-dimensional reference fields in grid units; centre i is integer i.

    Cell faces use [i-1/2, i+1/2), hats have zero exterior coefficients, and the
    clamped interpolant includes both outer faces, matching GridSpec. Isolated
    endpoint values do not affect their line integrals.
    """
    if not coefficients or len(coefficients) > 4096:
        raise ContractError("illustration lattice needs 1..4096 coefficients")
    values = tuple(finite_scalar(x, "attenuation", minimum=0) for x in coefficients)
    position = finite_scalar(coordinate, "grid coordinate")
    index = math.floor(position + 0.5)
    cell = values[index] if 0 <= index < len(values) else 0.0
    hat = math.fsum(value * max(1.0 - abs(position - i), 0.0) for i, value in enumerate(values))
    clamped = 0.0
    if -0.5 <= position <= len(values) - 0.5:
        point = min(max(position, 0.0), len(values) - 1)
        low = math.floor(point)
        fraction = point - low
        clamped = (1 - fraction) * values[low] + fraction * values[min(low + 1, len(values) - 1)]
    return LatticeValues(position, cell, _finite(hat), _finite(clamped))


@dataclass(frozen=True, slots=True)
class SlabInterval:
    axis_intervals: tuple[tuple[float, float] | None, ...]
    axis_states: tuple[str, ...]
    entry: float | None
    exit: float | None
    chord_mm: float
    state: str


def slab_interval(
    source_mm: tuple[float, float, float],
    endpoint_mm: tuple[float, float, float],
    lower_mm: tuple[float, float, float],
    upper_mm: tuple[float, float, float],
) -> SlabInterval:
    """Independent scalar slab construction; None marks an unbounded parallel slab."""
    for vector in (source_mm, endpoint_mm, lower_mm, upper_mm):
        if len(vector) != 3:
            raise ContractError("slab coordinates need three components")
        for value in vector:
            finite_scalar(value, "coordinate in mm")
    if any(a >= b for a, b in zip(lower_mm, upper_mm, strict=True)):
        raise ContractError("every slab must have positive physical width")
    length = _positive(math.dist(source_mm, endpoint_mm), "finite ray length")
    intervals: list[tuple[float, float] | None] = []
    states: list[str] = []
    entry, exit_parameter = 0.0, 1.0
    miss = False
    for source, end, low, high in zip(source_mm, endpoint_mm, lower_mm, upper_mm, strict=True):
        direction = _finite(end - source)
        if direction == 0:
            inside = low <= source <= high
            states.append("parallel-inside" if inside else "parallel-miss")
            intervals.append(None)
            miss |= not inside
        else:
            first = _finite((low - source) / direction)
            last = _finite((high - source) / direction)
            interval = (min(first, last), max(first, last))
            intervals.append(interval)
            states.append("crossing")
            entry, exit_parameter = max(entry, interval[0]), min(exit_parameter, interval[1])
    if miss or entry > exit_parameter:
        return SlabInterval(tuple(intervals), tuple(states), None, None, 0.0, "miss")
    state = "tangent" if entry == exit_parameter else "crossing"
    return SlabInterval(
        tuple(intervals),
        tuple(states),
        entry,
        exit_parameter,
        _product(length, exit_parameter - entry),
        state,
    )


@dataclass(frozen=True, slots=True)
class QuadraticIntegral:
    samples: int
    exact: float
    midpoint: float
    signed_error: float
    predicted_error: float


def quadratic_midpoint(
    length_mm: float,
    intercept_mm_inverse: float,
    linear_mm_inverse_squared: float,
    curvature_mm_inverse_cubed: float,
    samples: int,
) -> QuadraticIntegral:
    """Section 4.5: nonnegative polynomial on a positive physical segment."""
    length = _positive(length_mm, "length_mm")
    a, b, c = (
        finite_scalar(x, "polynomial coefficient", minimum=0)
        for x in (intercept_mm_inverse, linear_mm_inverse_squared, curvature_mm_inverse_cubed)
    )
    count = integer(samples, "samples", minimum=1, maximum=1_000_000)
    step = length / count
    if step == 0:
        raise NumericalError("quadrature spacing underflowed")
    exact = _finite(
        math.fsum(
            (
                _product(a, length),
                _product(b, length, length, 0.5),
                _product(c, length, length, length, 1 / 3),
            )
        )
    )
    midpoint = _finite(
        math.fsum(
            _product(
                step,
                _finite(
                    a
                    + _product(b, (i + 0.5) * step)
                    + _product(c, (i + 0.5) * step, (i + 0.5) * step)
                ),
            )
            for i in range(count)
        )
    )
    prediction = _product(c, length, length, length, 1 / (12 * count * count))
    return QuadraticIntegral(count, exact, midpoint, exact - midpoint, prediction)


@dataclass(frozen=True, slots=True)
class IntervalIntegral:
    entry_mm: float
    exact: float
    midpoint: float
    exact_entry_derivative_per_mm: float
    branch_entry_derivative_per_mm: float | None
    branch_state: str


def moving_interval(
    entry_mm: float,
    exit_mm: float,
    ray_length_mm: float,
    attenuation_mm_inverse: float,
    samples: int,
) -> IntervalIntegral:
    """Fixed ray nodes sampling an indicator; not the production clipped quadrature."""
    length = _positive(ray_length_mm, "ray_length_mm")
    entry = finite_scalar(entry_mm, "entry_mm", minimum=0)
    end = finite_scalar(exit_mm, "exit_mm", minimum=0)
    mu = finite_scalar(attenuation_mm_inverse, "attenuation_mm_inverse", minimum=0)
    count = integer(samples, "samples", minimum=1, maximum=1_000_000)
    if not entry < end <= length:
        raise ContractError("require 0 <= entry < exit <= ray length")
    step = length / count
    nodes = tuple((i + 0.5) * step for i in range(count))
    hits = sum(entry <= node < end for node in nodes)
    at_crossing = entry in nodes and mu > 0
    return IntervalIntegral(
        entry,
        _product(mu, end - entry),
        _product(mu, step, hits),
        -mu,
        None if at_crossing else 0.0,
        "sample-crossing" if at_crossing else "constant-branch",
    )


@dataclass(frozen=True, slots=True)
class TransmissionDifference:
    step: float
    forward_difference: float
    central_derivative: float
    exact_derivative: float
    derivative_absolute_error: float
    taylor_remainder: float
    reference_central_derivative: str
    reference_taylor_remainder: str


def transmission_difference(
    depth: float, depth_scale: float, step: float
) -> TransmissionDifference:
    """Binary64 evaluation and 100-digit reference for exp(-(depth + scale*z))."""
    optical_depth = finite_scalar(depth, "optical depth", minimum=0)
    scale = _positive(depth_scale, "depth scale per dimensionless coordinate")
    h = _positive(step, "dimensionless step")
    if scale * h > optical_depth:
        raise ContractError("both perturbed depths must be nonnegative")
    base = math.exp(-optical_depth)
    plus, minus = math.exp(-(optical_depth + scale * h)), math.exp(-(optical_depth - scale * h))
    derivative = -_product(scale, base)
    numerator, numerator_power = math.frexp(plus - minus)
    denominator, denominator_power = math.frexp(h)
    try:
        central = _finite(
            math.ldexp(numerator / denominator, numerator_power - denominator_power - 1)
        )
    except OverflowError as error:
        raise NumericalError("central difference exceeds binary64 range") from error
    remainder = abs(math.fsum((plus, -base, -h * derivative)))
    with localcontext() as context:
        context.prec = 100
        d, s, q = map(Decimal.from_float, (optical_depth, scale, h))
        reference_base = (-d).exp()
        reference_plus, reference_minus = (-(d + s * q)).exp(), (-(d - s * q)).exp()
        reference_central = (reference_plus - reference_minus) / (2 * q)
        reference_remainder = abs(reference_plus - reference_base + q * s * reference_base)
    return TransmissionDifference(
        h,
        abs(plus - base),
        central,
        derivative,
        abs(central - derivative),
        _finite(remainder),
        str(reference_central),
        str(reference_remainder),
    )


@dataclass(frozen=True, slots=True)
class SphereChord:
    impact_mm: float
    optical_depth: float
    derivative_per_mm: float | None
    derivative_state: str


def sphere_chord(radius_mm: float, attenuation_mm_inverse: float, impact_mm: float) -> SphereChord:
    """Sharp-sphere chord, with a labelled inward singularity at tangency."""
    radius = _positive(radius_mm, "radius_mm")
    mu = finite_scalar(attenuation_mm_inverse, "attenuation_mm_inverse", minimum=0)
    impact = finite_scalar(impact_mm, "impact_mm", minimum=0)
    if impact > radius:
        return SphereChord(impact, 0.0, 0.0, "outside")
    if impact == radius:
        return SphereChord(
            impact, 0.0, None if mu > 0 else 0.0, "inward-divergence" if mu > 0 else "zero-field"
        )
    # (R-b)/R avoids subtracting nearly equal squared radii; the dimensionless
    # factors avoid overflow in R*R and retain small chords near tangency.
    root = math.sqrt((radius - impact) / radius) * math.sqrt(1 + impact / radius)
    depth = _product(2.0, mu, radius, root)
    derivative = -_product(2.0, mu, impact / radius, 1 / root)
    return SphereChord(impact, depth, derivative, "interior")


@dataclass(frozen=True, slots=True)
class GaussianOverlap:
    shift_pixels: float
    overlap: float
    discrepancy: float
    derivative_per_pixel: float


def gaussian_overlap(width_pixels: float, shift_pixels: float) -> GaussianOverlap:
    """Section 6.4's uncentred, infinite-domain overlap; no recovery claim."""
    width = _positive(width_pixels, "width_pixels")
    shift = finite_scalar(shift_pixels, "shift_pixels")
    scaled = (shift / width) / 2
    exponent = -(scaled * scaled)
    overlap = math.exp(exponent)
    discrepancy = -math.expm1(exponent)
    if shift == 0:
        derivative = 0.0
    else:
        log_derivative = math.log(abs(shift)) - math.log(2.0) - 2 * math.log(width) + exponent
        derivative = math.copysign(_exp(log_derivative), shift)
    return GaussianOverlap(shift, overlap, discrepancy, derivative)


def nuisance_information_fraction(cosine: float) -> float:
    """Section 7.6: one nuisance direction in already whitened coordinates."""
    value = finite_scalar(cosine, "derivative cosine")
    if abs(value) > 1:
        raise ContractError("a derivative cosine must lie in [-1, 1]")
    return (1 - abs(value)) * (1 + abs(value))


@dataclass(frozen=True, slots=True)
class ChargeAssignment:
    mean: tuple[float, ...]
    shared_covariance: tuple[tuple[float, ...], ...]
    exclusive_covariance: tuple[tuple[float, ...], ...]
    fractions: tuple[float, ...]


def charge_assignment(arrival_mean: float, fractions: tuple[float, ...]) -> ChargeAssignment:
    """Two stipulated event laws with equal means, in arbitrary unit-event signal.

    Shared: every event deposits the given fraction vector. Exclusive: one pixel
    receives a unit score with probability equal to its fraction. These are
    analytic event models, not a calibrated charge-sharing detector.
    """
    rate = finite_scalar(arrival_mean, "Poisson arrival mean", minimum=0)
    if not 1 <= len(fractions) <= 32:
        raise ContractError("declare between one and 32 response fractions")
    values = tuple(finite_scalar(x, "response fraction", minimum=0) for x in fractions)
    if any(x > 1 for x in values) or math.fsum(values) != 1:
        raise ContractError("response fractions must sum to one; no renormalisation is performed")
    mean = tuple(_product(rate, x) for x in values)
    shared = tuple(tuple(_product(rate, a, b) for b in values) for a in values)
    exclusive = tuple(
        tuple(mean[i] if i == j else 0.0 for j in range(len(values))) for i in range(len(values))
    )
    return ChargeAssignment(mean, shared, exclusive, values)


@dataclass(frozen=True, slots=True)
class ApertureDifference:
    half_step_mm: float
    lower_edge_mm: float
    upper_edge_mm: float
    lower_mean: float
    upper_mean: float
    derivative_per_mm: float
    standard_error_per_mm: float
    relative_standard_error: float
    zero_contribution_probability: float


def aperture_difference(
    width_mm: float,
    edge_mm: float,
    half_step_mm: float,
    source_population: float,
    histories: int,
) -> ApertureDifference:
    """Exact moments of the paired uniform-strip estimator, equations 10.11/10.17.

    This computes a probability law, not sampled transport or an implemented
    moving-boundary gradient. Relative standard error is independent of source
    population; the no-hit probability describes the whole original-history batch.
    """
    width = _positive(width_mm, "strip width_mm")
    edge = _positive(edge_mm, "edge_mm")
    h = _positive(half_step_mm, "half_step_mm")
    population = _positive(source_population, "source_population")
    count = integer(histories, "histories", minimum=1, maximum=2**53)
    if not edge < width or not h < min(edge, width - edge):
        raise ContractError("both perturbed aperture edges must lie strictly inside the strip")
    # sqrt((w-2h)/(2hN)); logs avoid forming h*N or w/h outside binary64 range.
    relative = _exp(0.5 * (math.log(width - 2 * h) - math.log(2.0) - math.log(h) - math.log(count)))
    derivative = _finite(population / width)
    probability = min(1.0, (h / width) * 2)
    no_hit = 0.0 if probability == 1 else math.exp(count * math.log1p(-probability))
    return ApertureDifference(
        h,
        edge - h,
        edge + h,
        _product(population, (edge - h) / width),
        _product(population, (edge + h) / width),
        derivative,
        _product(derivative, relative),
        relative,
        no_hit,
    )
