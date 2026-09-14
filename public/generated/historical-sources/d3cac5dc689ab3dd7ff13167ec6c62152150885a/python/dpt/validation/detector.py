"""Independent dense-matrix and moment oracles for detector validation.

Dense matrices deliberately belong only here: production blur uses its compact
finite stencil and transpose, never an N² detector matrix.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal, localcontext


def poisson_log_mass(count: int, rate: float) -> Decimal:
    """Direct 80-digit log mass, independent of the device deviance evaluation.

    Moderate factorials are exact integers. Above 1000, an eight-term
    Stirling expansion supplies log(count!) with next-term error below 2e-52.
    Direct subtraction is safe at this precision for the declared rate range;
    it deliberately does not reproduce the production cancellation avoidance.
    """
    if isinstance(count, bool) or count < 0 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Poisson reference requires integer count >= 0 and finite rate > 0")
    with localcontext() as context:
        context.prec = 80
        k, mean = Decimal(count), Decimal.from_float(rate)
        if count <= 1000:
            log_factorial = Decimal(math.factorial(count)).ln()
        else:
            pi = Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078164")
            log_factorial = (k + Decimal("0.5")) * k.ln() - k + (2 * pi).ln() / 2
            bernoulli_coefficients = (
                (1, 12),
                (-1, 360),
                (1, 1260),
                (-1, 1680),
                (1, 1188),
                (-691, 360360),
                (7, 1092),
                (-3617, 122400),
            )
            for index, (numerator, denominator) in enumerate(bernoulli_coefficients):
                log_factorial += Decimal(numerator) / Decimal(denominator) / k ** (2 * index + 1)
        return +(k * mean.ln() - mean - log_factorial)


def spatial_response_matrix(
    height: int,
    width: int,
    kernel_height: int,
    kernel_width: int,
    weights: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """Construct the defining finite linear map by source/destination coordinates."""
    if min(height, width, kernel_height, kernel_width) <= 0:
        raise ValueError("matrix and stencil dimensions must be positive")
    if kernel_height % 2 != 1 or kernel_width % 2 != 1:
        raise ValueError("stencil dimensions must be odd")
    if len(weights) != kernel_height * kernel_width:
        raise ValueError("weight count does not match stencil dimensions")
    pixels = height * width
    matrix = [[0.0 for _ in range(pixels)] for _ in range(pixels)]
    for destination in range(pixels):
        row, col = divmod(destination, width)
        for source in range(pixels):
            sr, sc = divmod(source, width)
            kr = sr - row + kernel_height // 2
            kc = sc - col + kernel_width // 2
            if 0 <= kr < kernel_height and 0 <= kc < kernel_width:
                matrix[destination][source] = float(weights[kr * kernel_width + kc])
    return tuple(tuple(row) for row in matrix)


def matrix_product(
    matrix: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    transpose: bool = False,
) -> tuple[float, ...]:
    if not matrix:
        return ()
    rows, columns = len(matrix), len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("matrix must be rectangular")
    if len(values) != (rows if transpose else columns):
        raise ValueError("matrix/vector shape mismatch")
    if transpose:
        return tuple(
            math.fsum(matrix[r][c] * values[r] for r in range(rows)) for c in range(columns)
        )
    return tuple(
        math.fsum(value * weight for value, weight in zip(values, row, strict=True))
        for row in matrix
    )


def compound_poisson_moments(
    rates: Sequence[float],
    scores: Sequence[float],
) -> tuple[float, float, float]:
    """Mean, variance and fourth central moment from independent Poisson cumulants."""
    if len(rates) != len(scores):
        raise ValueError("one deterministic photon score is required per rate")
    if any(not math.isfinite(v) or v < 0 for v in (*rates, *scores)):
        raise ValueError("rates and scores must be finite and non-negative")
    mean = math.fsum(rate * score for rate, score in zip(rates, scores, strict=True))
    variance = math.fsum(rate * score**2 for rate, score in zip(rates, scores, strict=True))
    fourth_cumulant = math.fsum(rate * score**4 for rate, score in zip(rates, scores, strict=True))
    return mean, variance, fourth_cumulant + 3 * variance**2


def spread_covariance(
    matrix: Sequence[Sequence[float]],
    independent_variances: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """B diag(v) Bᵀ: spatially spread independent events create correlated pixels."""
    if any(len(row) != len(independent_variances) for row in matrix):
        raise ValueError("covariance dimensions do not match the response matrix")
    return tuple(
        tuple(
            math.fsum(
                a * b * variance
                for a, b, variance in zip(left, right, independent_variances, strict=True)
            )
            for right in matrix
        )
        for left in matrix
    )
