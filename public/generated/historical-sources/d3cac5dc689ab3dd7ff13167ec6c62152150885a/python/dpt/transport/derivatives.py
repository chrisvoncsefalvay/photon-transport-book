"""First-order expected-score derivatives with a reviewed parameter boundary.

For analogue sampling and log density eta_m, the path-density derivative is
N_m - integral_path rho_m (mu_a,m(E) + mu_s,m(E)) ds. Conditional interaction-type
and angular/energy distributions do not depend on rho_m, so their *combined*
collision density contributes one per collision. The integral includes every
survival segment, including the final escape. Ignoring that final term biases
transmission derivatives even when no collision occurs.

Continuous absorption instead samples only scattering events. For a fixed
scattering path its likelihood derivative is N_s,m - integral rho_m mu_s,m(E) ds;
the explicit absorption weight contributes -integral rho_m mu_a,m(E) ds.
The resulting expected-score derivative is X times their sum. Conditional
Compton angles and the energy sequence are independent of density at a fixed
path, so energy-dependent coefficients enter each segment's integral without an
extra angular derivative. This assumes fixed coefficient zeros, positive active
densities, fixed source/geometry and sufficient integrability to differentiate
the expectation. No derivative of a realised sampled collision location is used.

This is a likelihood-ratio estimator, not automatic differentiation through a
realised random trace. The same counter addresses permit deterministic replay;
pathwise support/boundary derivatives are not implied by that replay facility.
"""

from __future__ import annotations

# Workspace internals are shared only within this package.
# pyright: reportPrivateUsage=false
from dataclasses import dataclass
from typing import Any, Literal

from ._constants import LOG_SOURCE_AMPLITUDE, SOURCE_AMPLITUDE
from .forward import TransportWorkspace, _validate_call
from .model import TransportError
from .rng import HistoryBatch


@dataclass(frozen=True, slots=True)
class DerivativeCapability:
    parameter: str
    supported: bool
    density_term: str
    pathwise_term: str
    support_assumptions: str


CAPABILITIES = (
    DerivativeCapability(
        "log-material-density",
        True,
        "analogue: collisions minus total depth; "
        "continuous absorption: scatterings minus scattering depth",
        "analogue: zero; continuous absorption: minus absorption depth times weighted score",
        "fixed geometry; positive density; fixed partial-coefficient ratios and conditional laws",
    ),
    DerivativeCapability(
        "source-amplitude",
        True,
        "zero",
        "base detector score, including at zero amplitude",
        "fixed source sampling and importance weights; amplitude multiplies measurement only",
    ),
    DerivativeCapability(
        "log-source-amplitude",
        True,
        "zero",
        "amplitude times base detector score",
        "strictly positive amplitude; fixed source distribution and importance weights",
    ),
    DerivativeCapability(
        "moving-geometry",
        False,
        "not implemented",
        "not implemented",
        "requires a boundary-aware estimator for material visibility and detector edges",
    ),
    DerivativeCapability(
        "energy-or-angular-law",
        False,
        "not implemented",
        "not implemented",
        "requires derivatives of the conditional law, coefficient interpolation and energy score",
    ),
    DerivativeCapability(
        "source-distribution",
        False,
        "not implemented",
        "not implemented",
        "sampling-density and source-support terms must be derived before activation",
    ),
)


@dataclass(frozen=True, slots=True)
class TransportParameter:
    kind: Literal["log-material-density", "source-amplitude", "log-source-amplitude"]
    material: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "log-material-density":
            if type(self.material) is not int or self.material < 0:
                raise TransportError("a log-density derivative requires a nonnegative material ID")
        elif self.kind in ("source-amplitude", "log-source-amplitude"):
            if self.material is not None:
                raise TransportError("source amplitude is global and has no material ID")
        else:
            raise TransportError(
                f"unsupported transport derivative: {self.kind!r}; see CAPABILITIES"
            )


# region book:transport-derivative-contract
def derivative_histories(
    positions: Any,
    directions: Any,
    weights: Any,
    density: Any,
    *,
    parameter: TransportParameter,
    batch: HistoryBatch,
    workspace: TransportWorkspace,
    out_pixel: Any,
    out_derivative: Any,
    out_status: Any,
    source_amplitude: float = 1.0,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Replay complete histories and write their sparse expected-score derivatives.

    One selected parameter costs one complete-history launch and O(H) caller
    outputs; no history-by-event tape or material-by-history scratch is retained.
    Several parameters can use sequential calls and reuse these buffers. This
    trades recomputation for an explicit bounded memory footprint; its useful
    parameter-count range requires profiling. Binary64 signed scores are kept
    before reduction so uncertainty remains defined at the original-history level.

    Forward execution is not a prerequisite. Supplying an earlier batch identity
    and unchanged inputs reproduces its paths; independent score and derivative
    estimates for nonlinear losses must instead use disjoint batches.
    The prepared estimator selects both the flight law and derivative measure.
    Each output combines original factors and absorption log weight independently
    so forward-score underflow cannot erase a representable derivative.
    """
    material = SOURCE_AMPLITUDE if parameter.material is None else parameter.material
    if parameter.kind == "log-source-amplitude":
        if source_amplitude <= 0:
            raise TransportError("log-amplitude derivatives require strictly positive amplitude")
        material = LOG_SOURCE_AMPLITUDE
    if material >= workspace.spec.grid.materials:
        raise TransportError("active material is outside this workspace")
    wp = workspace.context.wp
    _validate_call(
        workspace,
        batch,
        positions,
        directions,
        weights,
        density,
        [
            ("out_pixel", out_pixel, wp.int32),
            ("out_derivative", out_derivative, wp.float64),
            ("out_status", out_status, wp.int32),
        ],
        source_amplitude,
        stream,
        validate,
    )
    workspace._launch(
        workspace._kernels.derivative_histories,
        batch.count,
        [
            positions,
            directions,
            weights,
            density,
            *workspace._model_inputs(),
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            material,
            source_amplitude,
            out_pixel,
            out_derivative,
            out_status,
            workspace._status,
        ],
    )
    if validate:
        workspace.check_status()


# endregion book:transport-derivative-contract
