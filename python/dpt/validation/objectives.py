"""Independent Decimal objective references for explicitly supplied scalar inputs."""

from __future__ import annotations

from decimal import Decimal, localcontext


def scalar_objective(
    prediction: float, observation: float, *, poisson: bool
) -> tuple[Decimal, Decimal]:
    """Evaluate at the exact binary input values, with no production helper calls."""
    with localcontext() as context:
        context.prec = 100
        predicted = Decimal.from_float(prediction)
        observed = Decimal.from_float(observation)
        if poisson:
            if predicted < 0 or observed < 0 or observed != observed.to_integral_value():
                raise ValueError("invalid counting-domain input")
            if observed == 0:
                return predicted, Decimal(1)
            if predicted == 0:
                raise ValueError("positive observed count has zero predicted probability")
            return (
                predicted - observed + observed * (observed / predicted).ln(),
                Decimal(1) - observed / predicted,
            )
        difference = predicted - observed
        return difference * difference / 2, difference
