"""CPU-only, scale-aware directional derivative reports for independent checks."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DirectionalRecord:
    step: float
    numerical: float
    predicted: float
    absolute_error: float
    relative_error: float
    scheme: str


def directional_sweep(
    evaluate: Callable[[tuple[float, ...]], float],
    point: Sequence[float],
    gradient: Sequence[float],
    direction: Sequence[float],
    steps: Sequence[float],
    *,
    scales: Sequence[float] | None = None,
    admissible: Callable[[tuple[float, ...]], bool] | None = None,
) -> tuple[DirectionalRecord, ...]:
    """Compare independently evaluated perturbations against a supplied derivative.

    direction is dimensionless; scales set physical units per coordinate.
    Central differences are used when both perturbations are admissible, else
    a labelled one-sided difference. No tolerance or universal best step is
    inferred. Non-smooth cases need their own classification outside this check.
    """
    count = len(point)
    if not count or len(gradient) != count or len(direction) != count:
        raise ValueError("point, gradient and direction must have equal nonzero length")
    units = tuple(scales) if scales is not None else (1.0,) * count
    if len(units) != count or any(not math.isfinite(x) or x <= 0 for x in units):
        raise ValueError("scales must be positive finite coordinate units")
    if any(not math.isfinite(x) for values in (point, gradient, direction) for x in values):
        raise ValueError("directional inputs must be finite")
    delta = tuple(direction[i] * units[i] for i in range(count))
    if not any(x != 0 for x in delta):
        raise ValueError("direction must be nonzero")
    predicted = math.fsum(gradient[i] * delta[i] for i in range(count))
    baseline = float(evaluate(tuple(point)))
    if not math.isfinite(baseline):
        raise ValueError("baseline evaluation is non-finite")
    records: list[DirectionalRecord] = []
    for step in steps:
        if not math.isfinite(step) or step <= 0:
            raise ValueError("steps must be positive finite values")
        plus = tuple(point[i] + step * delta[i] for i in range(count))
        minus = tuple(point[i] - step * delta[i] for i in range(count))
        valid_plus, valid_minus = (
            (True, True) if admissible is None else (admissible(plus), admissible(minus))
        )
        if valid_plus and valid_minus:
            numerical = (evaluate(plus) - evaluate(minus)) / (2 * step)
            scheme = "central"
        elif valid_plus:
            numerical = (evaluate(plus) - baseline) / step
            scheme = "forward"
        elif valid_minus:
            numerical = (baseline - evaluate(minus)) / step
            scheme = "backward"
        else:
            raise ValueError(f"neither perturbation is admissible at step {step}")
        if not math.isfinite(numerical):
            raise ValueError("perturbed evaluation produced a non-finite derivative")
        error = abs(numerical - predicted)
        denominator = max(abs(numerical), abs(predicted), math.ulp(0.0))
        records.append(
            DirectionalRecord(step, numerical, predicted, error, error / denominator, scheme)
        )
    return tuple(records)
