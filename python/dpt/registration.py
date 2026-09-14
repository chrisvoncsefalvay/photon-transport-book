"""Safeguarded limited-memory optimisation in a fixed, scaled parameter chart.

The evaluator owns the CUDA composition. It returns only a scalar objective and
the small parameter gradient, never a detector image. Parameterisations (pose
anchor, units, constrained nuisance variables) remain fixed throughout a solve.
Starting a new chart requires a new solve and therefore empty curvature history.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from dpt.contracts import ContractError, NumericalError, finite_scalar, integer

Vector = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Objective and derivative with respect to the supplied dimensionless chart."""

    loss: float
    gradient: Vector


class Evaluator(Protocol):
    def __call__(self, parameters: Vector, /) -> Evaluation:
        """Evaluate current parameters; invalid trials raise a numerical/domain error."""
        ...


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    max_iterations: int = 200
    max_evaluations: int = 2000
    memory: int = 10
    line_search_evaluations: int = 32
    gradient_tolerance: float = 1e-6
    relative_loss_tolerance: float = 1e-12
    step_tolerance: float = 1e-10
    armijo: float = 1e-4
    curvature: float = 0.9
    maximum_step: float = 16.0
    minimum_step: float = 1e-14

    def __post_init__(self) -> None:
        for name in ("max_iterations", "max_evaluations", "memory", "line_search_evaluations"):
            integer(getattr(self, name), name, minimum=1)
        for name in ("gradient_tolerance", "relative_loss_tolerance", "step_tolerance"):
            finite_scalar(getattr(self, name), name, minimum=0.0)
        for name in ("armijo", "curvature", "minimum_step", "maximum_step"):
            finite_scalar(getattr(self, name), name, minimum=0.0)
        if not 0.0 < self.armijo < self.curvature < 1.0:
            raise ContractError("strong-Wolfe constants must satisfy 0 < c1 < c2 < 1")
        if not 0.0 < self.minimum_step < self.maximum_step:
            raise ContractError("line-search step interval must be positive and nonempty")


StopReason = Literal[
    "gradient_tolerance",
    "step_tolerance",
    "loss_stagnation",
    "line_search_failed",
    "evaluation_budget",
    "iteration_budget",
    "cancelled",
]


@dataclass(frozen=True, slots=True)
class IterationRecord:
    iteration: int
    loss: float
    gradient_infinity_norm: float
    step: float
    evaluations: int
    curvature_pairs: int


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    parameters: Vector
    evaluation: Evaluation
    reason: StopReason
    evaluations: int
    history: tuple[IterationRecord, ...]

    @property
    def stationary(self) -> bool:
        """Only the gradient criterion establishes numerical stationarity here.

        Small loss/step changes indicate stagnation; they do not prove the pose
        was recovered or that the current point is an identifiable minimum.
        """
        return self.reason == "gradient_tolerance"


def _dot(a: Vector, b: Vector) -> float:
    try:
        return math.fsum(x * y for x, y in zip(a, b, strict=True))
    except (OverflowError, ValueError):
        # Curvature and line-search callers reject an unrepresentable product;
        # it cannot certify a direction or replace the accepted iterate.
        return math.nan


def _norm(a: Vector) -> float:
    return max(abs(x) for x in a)


def _checked(evaluation: Evaluation, dimension: int) -> Evaluation:
    if len(evaluation.gradient) != dimension:
        raise ContractError("evaluator gradient dimension differs from its parameter chart")
    if not math.isfinite(evaluation.loss) or not all(map(math.isfinite, evaluation.gradient)):
        raise NumericalError("non-finite objective or parameter gradient")
    return evaluation


# region book:registration-lbfgs-direction
def _direction(gradient: Vector, pairs: list[tuple[Vector, Vector, float]]) -> Vector:
    """Two-loop recursion; curvature vectors all belong to the same scaled chart."""
    if not pairs:
        return _steepest_direction(gradient)
    q = gradient
    coefficients: list[float] = []
    for displacement, change, reciprocal in reversed(pairs):
        coefficient = reciprocal * _dot(displacement, q)
        coefficients.append(coefficient)
        q = tuple(a - coefficient * b for a, b in zip(q, change, strict=True))
    scale = 1.0
    if pairs:
        displacement, change, _ = pairs[-1]
        denominator = _dot(change, change)
        if not math.isfinite(denominator) or denominator <= 0.0:
            return _steepest_direction(gradient)
        scale = _dot(displacement, change) / denominator
        if not math.isfinite(scale) or scale <= 0.0:
            return _steepest_direction(gradient)
    result = tuple(scale * value for value in q)
    for (displacement, change, reciprocal), coefficient in zip(
        pairs, reversed(coefficients), strict=True
    ):
        beta = reciprocal * _dot(change, result)
        result = tuple(
            a + (coefficient - beta) * b for a, b in zip(result, displacement, strict=True)
        )
    return tuple(-value for value in result)


# endregion book:registration-lbfgs-direction


def _steepest_direction(gradient: Vector) -> Vector:
    """Bound the initial chart displacement without squaring a large gradient."""
    scale = max(1.0, _norm(gradient))
    return tuple(-value / scale for value in gradient)


class _BudgetExhaustedError(Exception):
    pass


def _line_search(
    evaluate: Callable[[Vector], Evaluation],
    x: Vector,
    base: Evaluation,
    direction: Vector,
    policy: RecoveryPolicy,
    *,
    initial_step: float = 1.0,
) -> tuple[float, Evaluation] | None:
    """Seek strong Wolfe, accepting strict Armijo decrease at the step bound.

    Bisection sacrifices polynomial interpolation speed to keep invalid-domain
    endpoints and nonfinite trials unambiguous. No failed or unchecked trial is
    returned as an accepted iterate. The last accepted state lives in the caller.
    """
    slope = _dot(base.gradient, direction)
    if not math.isfinite(slope) or slope >= 0:
        return None
    attempts = 0

    def trial(alpha: float) -> Evaluation | None:
        nonlocal attempts
        attempts += 1
        point = tuple(value + alpha * step for value, step in zip(x, direction, strict=True))
        if not all(map(math.isfinite, point)):
            return None
        try:
            return evaluate(point)
        except (NumericalError, OverflowError):
            # Domain invalidity may be signalled as NumericalError by a
            # parameterised evaluator. ContractError remains a programming error.
            return None

    def armijo(alpha: float, candidate: Evaluation) -> bool:
        return candidate.loss <= base.loss + policy.armijo * alpha * slope

    def zoom(low: float, high: float, low_value: Evaluation) -> tuple[float, Evaluation] | None:
        while attempts < policy.line_search_evaluations:
            alpha = 0.5 * (low + high)
            if abs(high - low) < policy.minimum_step or alpha == low or alpha == high:
                return None
            candidate = trial(alpha)
            if (
                candidate is None
                or not armijo(alpha, candidate)
                or candidate.loss >= low_value.loss
            ):
                high = alpha
                continue
            derivative = _dot(candidate.gradient, direction)
            if abs(derivative) <= -policy.curvature * slope:
                return alpha, candidate
            if derivative * (high - low) >= 0:
                high = low
            low, low_value = alpha, candidate
        return None

    previous_alpha, previous_value = 0.0, base
    alpha = min(initial_step, policy.maximum_step)
    while attempts < policy.line_search_evaluations:
        candidate = trial(alpha)
        if (
            candidate is None
            or not armijo(alpha, candidate)
            or (previous_alpha > 0 and candidate.loss >= previous_value.loss)
        ):
            return zoom(previous_alpha, alpha, previous_value)
        derivative = _dot(candidate.gradient, direction)
        if abs(derivative) <= -policy.curvature * slope:
            return alpha, candidate
        if alpha == policy.maximum_step and candidate.loss < base.loss:
            return alpha, candidate
        if derivative >= 0:
            return zoom(alpha, previous_alpha, candidate)
        if alpha == policy.maximum_step:
            return None
        previous_alpha, previous_value = alpha, candidate
        alpha = min(2.0 * alpha, policy.maximum_step)
    return None


# region book:registration-safeguarded-loop
def recover_parameters(
    evaluator: Evaluator,
    initial: Vector,
    *,
    policy: RecoveryPolicy | None = None,
    observe: Callable[[IterationRecord], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> RecoveryResult:
    """Minimise a deterministic objective in an immutable, dimensionless chart.

    The evaluator must use the same observation, forward model and parameter
    coordinates for every call. It may reuse device scratch, but must complete
    numerical-status checks before returning. Rejected trials never replace the
    accepted parameters in the result. Observer exceptions are not swallowed.
    """
    selected = policy or RecoveryPolicy()
    x: Vector = tuple(finite_scalar(value, "initial parameter") for value in initial)
    if not x:
        raise ContractError("recovery requires at least one active parameter")
    calls = 0

    def evaluate(point: Vector) -> Evaluation:
        nonlocal calls
        if calls >= selected.max_evaluations:
            raise _BudgetExhaustedError
        calls += 1
        return _checked(evaluator(point), len(x))

    current = evaluate(x)
    history: list[IterationRecord] = []
    pairs: list[tuple[Vector, Vector, float]] = []
    reason: StopReason = "iteration_budget"
    history.append(IterationRecord(0, current.loss, _norm(current.gradient), 0.0, calls, 0))
    for iteration in range(1, selected.max_iterations + 1):
        if cancelled is not None and cancelled():
            reason = "cancelled"
            break
        if _norm(current.gradient) <= selected.gradient_tolerance:
            reason = "gradient_tolerance"
            break
        direction = _direction(current.gradient, pairs)
        slope = _dot(direction, current.gradient)
        if not all(map(math.isfinite, direction)) or not math.isfinite(slope) or slope >= 0:
            pairs.clear()
            direction = _steepest_direction(current.gradient)
        try:
            accepted = _line_search(evaluate, x, current, direction, selected)
            if accepted is None:
                # A bad inverse-Hessian estimate need not terminate a sound
                # forward model: try steepest descent once with empty memory.
                pairs.clear()
                direction = _steepest_direction(current.gradient)
                accepted = _line_search(
                    evaluate,
                    x,
                    current,
                    direction,
                    selected,
                    initial_step=max(
                        selected.minimum_step, 2.0 ** -min(16, selected.line_search_evaluations)
                    ),
                )
        except _BudgetExhaustedError:
            reason = "evaluation_budget"
            break
        if accepted is None:
            reason = "line_search_failed"
            break
        alpha, new_value = accepted
        step: Vector = tuple(alpha * value for value in direction)
        new_x: Vector = tuple(a + b for a, b in zip(x, step, strict=True))
        change = tuple(a - b for a, b in zip(new_value.gradient, current.gradient, strict=True))
        curvature = _dot(step, change)
        threshold = 1e-12 * math.hypot(*step) * math.hypot(*change)
        if (
            math.isfinite(curvature)
            and curvature > max(0.0, threshold)
            and math.isfinite(1.0 / curvature)
        ):
            pairs.append((step, change, 1.0 / curvature))
            if len(pairs) > selected.memory:
                del pairs[0]
        old_loss = current.loss
        x, current = new_x, new_value
        record = IterationRecord(
            iteration, current.loss, _norm(current.gradient), alpha, calls, len(pairs)
        )
        history.append(record)
        if observe is not None:
            observe(record)
        if _norm(current.gradient) <= selected.gradient_tolerance:
            reason = "gradient_tolerance"
            break
        if _norm(step) <= selected.step_tolerance * max(1.0, _norm(x)):
            reason = "step_tolerance"
            break
        if abs(old_loss - current.loss) <= selected.relative_loss_tolerance * max(
            1.0, abs(old_loss), abs(current.loss)
        ):
            reason = "loss_stagnation"
            break
    return RecoveryResult(x, current, reason, calls, tuple(history))


# endregion book:registration-safeguarded-loop
