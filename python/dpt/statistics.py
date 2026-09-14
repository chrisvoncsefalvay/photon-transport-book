"""Range-aware scalar summaries shared by independent-replicate controllers.

The caller establishes independence and the sampling unit. These arithmetic
helpers cannot infer independence, normality or confidence-interval coverage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

from dpt.contracts import NumericalError


def mean_standard_error(values: Sequence[float]) -> tuple[float, float]:
    """Finite binary64 mean and sample standard error without raw squared sums.

    These are small host checkpoints, not history/image reductions. Exact binary
    rationals form the mean and centred differences so cancelling extreme values
    cannot erase a representable residual. Only bounded normalised deviations
    enter the floating-point norm; scaling is restored after normalisation.
    """
    count = len(values)
    if count < 2 or not all(map(math.isfinite, values)):
        raise NumericalError("replicate checkpoint requires at least two finite values")
    scale = max(map(abs, values))
    if scale == 0:
        return 0.0, 0.0
    exact = tuple(Fraction.from_float(float(value)) for value in values)
    exact_mean = sum(exact, Fraction(0)) / count
    exact_scale = Fraction.from_float(scale)
    mean = float(exact_mean)
    deviations = tuple(float((value - exact_mean) / exact_scale) for value in exact)
    error = scale * (math.hypot(*deviations) / math.sqrt(count * (count - 1)))
    if not math.isfinite(mean) or not math.isfinite(error):
        raise NumericalError("replicate mean or standard error exceeds binary64 range")
    return mean, error
