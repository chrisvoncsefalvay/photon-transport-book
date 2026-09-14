"""Independent analytic transport oracles; never called by the CUDA operator.

These use complete-path limits and Decimal arithmetic rather than replaying the
production voxel traversal. They are mathematical validation cases, not material
assets or independently measured physical data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext


@dataclass(frozen=True, slots=True)
class AbsorbingSlabReference:
    transmission: Decimal
    log_density_derivative: tuple[Decimal, ...]
    variance_per_history: Decimal


def absorbing_layers(
    layer_optical_depths: tuple[float, ...],
) -> AbsorbingSlabReference:
    """Unit-score survival and per-layer log-density derivatives in pure absorption."""
    if not layer_optical_depths or any(
        not math.isfinite(value) or value < 0 for value in layer_optical_depths
    ):
        raise ValueError("supply finite nonnegative layer optical depths")
    with localcontext() as context:
        context.prec = 70
        depths = tuple(Decimal.from_float(value) for value in layer_optical_depths)
        transmission = (-sum(depths, Decimal(0))).exp()
        derivatives = tuple(-depth * transmission for depth in depths)
        return AbsorbingSlabReference(transmission, derivatives, transmission * (1 - transmission))


def compton_energy(incident_kev: float, cosine: float) -> Decimal:
    """Energy conservation for a stationary free electron, evaluated in Decimal."""
    if not math.isfinite(incident_kev) or incident_kev <= 0 or not -1 <= cosine <= 1:
        raise ValueError("positive incident energy and a cosine in [-1,1] are required")
    with localcontext() as context:
        context.prec = 70
        incident = Decimal.from_float(incident_kev)
        return incident / (
            1 + incident / Decimal("510.99895069") * (1 - Decimal.from_float(cosine))
        )


def klein_nishina_total_ratio(energy_kev: float) -> Decimal:
    """Total Klein-Nishina cross section divided by the Thomson cross section.

    Closed form from integration of the free-electron differential cross section;
    independent of the rejection sampler. Decimal resolves the cancellation in
    the low-energy limit. This is a conditional-law normalisation oracle, not a
    replacement for the user's sourced macroscopic scattering coefficients.
    """
    if not math.isfinite(energy_kev) or energy_kev <= 0:
        raise ValueError("energy must be positive and finite")
    with localcontext() as context:
        context.prec = 90
        alpha = Decimal.from_float(energy_kev) / Decimal("510.99895069")
        one = Decimal(1)
        log = (one + 2 * alpha).ln()
        return (
            Decimal(3)
            / 4
            * (
                (one + alpha) / alpha**3 * (2 * alpha * (one + alpha) / (one + 2 * alpha) - log)
                + log / (2 * alpha)
                - (one + 3 * alpha) / (one + 2 * alpha) ** 2
            )
        )


@dataclass(frozen=True, slots=True)
class BernoulliInverseReference:
    expected_signal: float
    expected_loss: float
    true_amplitude_gradient: float
    independent_gradient_expectation: float
    reused_history_gradient_expectation: float


def bernoulli_inverse(
    probability: float,
    amplitude: float,
    observed: float,
) -> BernoulliInverseReference:
    """Enumerate two independent Bernoulli outcomes; expose same-history bias exactly.

    Signal is amplitude times Z. The true amplitude derivative of its expectation
    is P(Z=1). The double sum is an independent finite probability-space oracle for
    the complete loss-gradient construction, not a comparison against itself.
    """
    if not 0 <= probability <= 1 or not all(map(math.isfinite, (amplitude, observed))):
        raise ValueError("invalid Bernoulli inverse parameters")
    outcomes = ((0.0, 1.0 - probability), (1.0, probability))
    independent = math.fsum(
        left_mass * right_mass * (amplitude * left - observed) * right
        for left, left_mass in outcomes
        for right, right_mass in outcomes
    )
    reused = math.fsum(mass * (amplitude * value - observed) * value for value, mass in outcomes)
    signal = amplitude * probability
    return BernoulliInverseReference(
        signal,
        0.5 * (signal - observed) ** 2,
        (signal - observed) * probability,
        independent,
        reused,
    )
