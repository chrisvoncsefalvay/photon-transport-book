"""Independent-batch squared-expected-signal optimisation in a fixed chart.

The acceptance band uses an estimated standard error, not a distribution-free
confidence bound. Sampling uncertainty can force batch growth or exhaustion;
it never turns an unresolved trial into an accepted decrease. General nonlinear
losses need different estimators and are outside this interface's contract.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal, Protocol, TypeVar, cast

from dpt.contracts import ContractError, NumericalError, TrialDomainError, finite_scalar, integer
from dpt.registration import Vector
from dpt.statistics import mean_standard_error
from dpt.stochastic_models import (
    Matrix,
    QuadraticProposal,
    quadratic_proposal,
    quadratic_reduction,
    validate_curvature,
)
from dpt.transport.rng import HistoryBatch


class IndependentSquaredOracle(Protocol):
    """Original histories are IID within each caller-specified batch.

    Source sampling must be keyed by the provided history identity too. The two
    independent batches supply distinct mean and derivative estimates. Replaying
    an original history to obtain its derivative does not create a new sample.
    """

    def gradient_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> Vector:
        """Return (estimated mean - observed) times an independent mean derivative."""
        ...

    def change_replicate(
        self, before: Vector, after: Vector, first: HistoryBatch, second: HistoryBatch
    ) -> float:
        """Return an unbiased squared-expected-signal loss difference.

        Each pose uses the product of independent residual-mean estimates.
        Reuse the specified streams between poses for common random numbers;
        batches first and second must stay independent within either pose.
        """
        ...


class QuadraticSquaredOracle(IndependentSquaredOracle, Protocol):
    def model_replicate(
        self, parameters: Vector, mean_batch: HistoryBatch, derivative_batch: HistoryBatch
    ) -> tuple[Vector, Matrix]:
        """Return independent-product gradient and PSD estimated-Jacobian metric."""
        ...


@dataclass(frozen=True, slots=True)
class StochasticPolicy:
    iterations: int = 100
    replicates: int = 8
    initial_batch: int = 1024
    maximum_batch: int = 1_048_576
    unique_history_budget: int = 100_000_000
    initial_radius: float = 0.1
    minimum_radius: float = 1e-7
    maximum_radius: float = 1.0
    acceptance_fraction: float = 0.1
    standard_error_multiplier: float = 2.0
    gradient_tolerance: float = 1e-5
    relative_gradient_uncertainty: float = 0.5
    proposal: Literal["linear", "quadratic"] = "linear"
    numerical_gradient_allowance: float = 0.0
    final_validation_batch: int | None = None
    damping_relative: float = 1e-12
    rank_tolerance: float = 1e-10
    radius_growth_agreement: float = 0.75

    def __post_init__(self) -> None:
        for name in ("iterations", "initial_batch", "maximum_batch", "unique_history_budget"):
            integer(getattr(self, name), name, minimum=1, maximum=2**63 - 1)
        integer(self.replicates, "replicates", minimum=2)
        if self.initial_batch > self.maximum_batch or self.maximum_batch >= 2**31:
            raise ContractError("batch sizes must be ordered positive signed-32-bit counts")
        for name in (
            "initial_radius",
            "minimum_radius",
            "maximum_radius",
            "acceptance_fraction",
            "standard_error_multiplier",
            "gradient_tolerance",
            "relative_gradient_uncertainty",
        ):
            if finite_scalar(getattr(self, name), name, minimum=0.0) == 0:
                raise ContractError(f"{name} must be positive")
        if not self.minimum_radius <= self.initial_radius <= self.maximum_radius:
            raise ContractError("trust radii must satisfy minimum <= initial <= maximum")
        if not 0 < self.acceptance_fraction < 1:
            raise ContractError("acceptance fraction must lie in (0,1)")
        if self.proposal not in ("linear", "quadratic"):
            raise ContractError("proposal must be linear or quadratic")
        finite_scalar(self.numerical_gradient_allowance, "numerical_gradient_allowance", minimum=0)
        finite_scalar(self.damping_relative, "damping_relative", minimum=0)
        finite_scalar(self.rank_tolerance, "rank_tolerance", minimum=0)
        finite_scalar(self.radius_growth_agreement, "radius_growth_agreement", minimum=0)
        if not 0 < self.rank_tolerance < 1 or not 0 < self.radius_growth_agreement <= 1:
            raise ContractError("rank tolerance and radius agreement must lie in (0,1] (rank < 1)")
        if self.final_validation_batch is not None:
            integer(self.final_validation_batch, "final_validation_batch", minimum=1)
            if self.final_validation_batch > self.maximum_batch:
                raise ContractError("final validation batch exceeds prepared batch capacity")


@dataclass(frozen=True, slots=True)
class StochasticStep:
    iteration: int
    accepted: bool
    mean_change: float
    change_standard_error: float
    predicted_decrease: float
    radius: float
    batch_size: int
    first_history: int
    histories_used: int


@dataclass(frozen=True, slots=True)
class GradientAttempt:
    """Host statistics for one complete set of independent gradient replicates."""

    iteration: int
    parameters: Vector
    batch_size: int
    first_history: int
    histories_used: int
    gradient: Vector
    standard_error: Vector
    gradient_norm: float
    standard_error_norm: float
    decision: Literal["zero_sample", "gradient_band", "resolved", "relative_uncertainty"]
    model_id: int | None = None
    pool: str = "proposal"
    band_method: str = "heuristic_euclidean_marginal_se"
    numerical_allowance: float = 0.0
    replicate_gradients: tuple[Vector, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkReservation:
    """Random identities reserved, distinct from completed oracle calls/work.

    A failed call may have executed partial CUDA work. Only the operator's work
    counters can report that; completed_replicates does not infer replay costs.
    No identity from a failed reservation is reused.
    """

    operation: str
    iteration: int
    first_history: int
    required_histories: int
    reserved_histories: int
    checkpoint_histories: int
    pairs: tuple[tuple[HistoryBatch, HistoryBatch], ...]
    status: str
    attempted_replicates: int = 0
    completed_replicates: int = 0


@dataclass(frozen=True, slots=True)
class LocalModel:
    model_id: int
    parameters: Vector
    gradient: Vector
    curvature: Matrix
    first_history: int
    histories_used: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class AcceptanceAttempt:
    iteration: int
    incumbent: Vector
    candidate: Vector
    model_id: int | None
    step: Vector
    predicted_decrease: float
    radius: float
    batch_size: int
    first_history: int
    histories_used: int
    required_histories: int
    replicate_changes: tuple[float, ...]
    mean_change: float | None
    standard_error: float | None
    band_method: str
    look_index: int
    threshold: float
    outcome: str
    agreement: float | None
    boundary: bool
    proposal_diagnostics: QuadraticProposal | None = None


@dataclass(frozen=True, slots=True)
class StochasticRecoveryResult:
    parameters: Vector
    reason: Literal[
        "gradient_band", "sampling_unresolved", "history_budget", "radius_limit", "iteration_budget"
    ]
    unique_histories: int
    steps: tuple[StochasticStep, ...]
    domain_rejections: tuple[DomainRejection, ...] = ()
    gradient_attempts: tuple[GradientAttempt, ...] = ()
    termination_detail: str = ""
    diagnostic_version: int = 2
    acceptance_attempts: tuple[AcceptanceAttempt, ...] = ()
    reservations: tuple[WorkReservation, ...] = ()
    local_models: tuple[LocalModel, ...] = ()
    final_validation_attempts: tuple[GradientAttempt, ...] = ()
    sampling_classification: str = "stochastic"
    proposal_policy: str = "linear"
    uncertainty_policy: str = "heuristic_euclidean_marginal_se"
    numerical_gradient_allowance: float = 0.0
    final_validation_batch: int | None = None
    final_validation_trigger: str | None = None


@dataclass(frozen=True, slots=True)
class DomainRejection:
    iteration: int
    radius: float
    message: str


_Replicate = TypeVar("_Replicate")


def _gradient_attempt(
    replicates: tuple[Vector, ...],
    parameters: Vector,
    selected: StochasticPolicy,
    deterministic: bool,
    *,
    iteration: int,
    batch_size: int,
    first_history: int,
    histories_used: int,
    model_id: int | None,
    pool: str = "proposal",
) -> GradientAttempt:
    if any(len(value) != len(parameters) for value in replicates):
        raise ContractError("gradient replicate dimension differs from the active chart")
    statistics = tuple(
        mean_standard_error(tuple(row[j] for row in replicates)) for j in range(len(parameters))
    )
    gradient = tuple(mean for mean, _ in statistics)
    norm = math.hypot(*gradient)
    empirical_error = math.hypot(*(error for _, error in statistics))
    if not math.isfinite(norm) or not math.isfinite(empirical_error):
        raise NumericalError("gradient or uncertainty norm exceeds the finite chart range")
    if deterministic and empirical_error > selected.numerical_gradient_allowance:
        raise NumericalError("verified deterministic oracle exceeds its numerical allowance")
    error_norm = 0.0 if deterministic else empirical_error
    allowance = selected.numerical_gradient_allowance if deterministic else 0.0
    decision: Literal["zero_sample", "gradient_band", "resolved", "relative_uncertainty"]
    if not deterministic and norm == 0.0 and error_norm == 0.0:
        decision = "zero_sample"
    elif (
        norm + selected.standard_error_multiplier * error_norm + allowance
        <= selected.gradient_tolerance
    ):
        decision = "gradient_band"
    elif norm > 0 and error_norm + allowance <= selected.relative_gradient_uncertainty * norm:
        decision = "resolved"
    else:
        decision = "relative_uncertainty"
    return GradientAttempt(
        iteration,
        parameters,
        batch_size,
        first_history,
        histories_used,
        gradient,
        (0.0,) * len(parameters) if deterministic else tuple(error for _, error in statistics),
        norm,
        error_norm,
        decision,
        model_id,
        pool,
        "deterministic_numerical_allowance" if deterministic else "heuristic_euclidean_marginal_se",
        allowance,
        replicates,
    )


# region book:stochastic-independent-acceptance
def recover_expected_signal(
    oracle: IndependentSquaredOracle,
    initial: Vector,
    *,
    seed: int,
    policy: StochasticPolicy | None = None,
) -> StochasticRecoveryResult:
    """Fresh proposal/acceptance pools with an optional held-out final checkpoint.

    Quadratic proposals cache a PSD metric at the unchanged incumbent; rejected
    radii use fresh acceptance identities. Final sample size is fixed here before
    observing any samples. A failed final checkpoint terminates unresolved and
    never feeds a new proposal. ``linear`` with no explicit final batch retains
    the original trajectory for ablation, including its heuristic stopping rule.

    The explicit ``deterministic_sampling`` oracle flag must be established from
    immutable physical/source properties. Empirical variance never establishes
    that classification. Its numerical allowance requires independent validation
    by the caller. Stochastic norm bands are heuristic, not simultaneous-vector
    or repeated-look confidence bounds. Unique identities are not replay costs.
    """
    selected: StochasticPolicy = policy or StochasticPolicy()
    quadratic = selected.proposal == "quadratic"
    integer(seed, "seed", maximum=2**64 - 1)
    parameters: Vector = tuple(finite_scalar(value, "initial parameter") for value in initial)
    if not parameters:
        raise ContractError("a stochastic inverse problem needs active parameters")
    if quadratic and len(parameters) > 16:
        raise ContractError("dense stochastic models support at most 16 active parameters")
    if quadratic and not callable(getattr(oracle, "model_replicate", None)):
        raise ContractError("quadratic proposals require an oracle model_replicate operation")
    classification = getattr(oracle, "deterministic_sampling", False)
    if type(classification) is not bool:
        raise ContractError("deterministic_sampling must be an explicit boolean contract")
    deterministic = classification
    final_size = selected.final_validation_batch
    if quadratic and final_size is None:
        final_size = selected.initial_batch
    final_cost = 0 if final_size is None else 2 * final_size * selected.replicates
    first_history = 0
    radius = selected.initial_radius
    batch_size = selected.initial_batch
    damping_relative = selected.damping_relative
    rank_tolerance = selected.rank_tolerance
    steps: list[StochasticStep] = []
    domain_rejections: list[DomainRejection] = []
    gradient_attempts: list[GradientAttempt] = []
    acceptance_attempts: list[AcceptanceAttempt] = []
    reservations: list[WorkReservation] = []
    local_models: list[LocalModel] = []
    validation_attempts: list[GradientAttempt] = []
    validation_trigger: str | None = None
    cached: list[LocalModel] = []
    band_method = (
        "deterministic_numerical_allowance" if deterministic else "heuristic_euclidean_marginal_se"
    )

    def reserve(size: int, operation: str, iteration: int, holdback: int = 0) -> int | None:
        nonlocal first_history
        required = 2 * size * selected.replicates
        start = first_history
        if first_history + required + holdback > selected.unique_history_budget:
            reservations.append(
                WorkReservation(
                    operation, iteration, start, required, 0, holdback, (), "budget_denied"
                )
            )
            return None
        pairs: list[tuple[HistoryBatch, HistoryBatch]] = []
        for _ in range(selected.replicates):
            left = HistoryBatch(seed, first_history, size, f"inverse-source-{first_history}")
            first_history += size
            right = HistoryBatch(seed, first_history, size, f"inverse-source-{first_history}")
            first_history += size
            pairs.append((left, right))
        reservations.append(
            WorkReservation(
                operation, iteration, start, required, required, holdback, tuple(pairs), "reserved"
            )
        )
        return len(reservations) - 1

    def execute(
        index: int, operation: Callable[[HistoryBatch, HistoryBatch], _Replicate]
    ) -> tuple[_Replicate, ...]:
        outputs: list[_Replicate] = []
        for left, right in reservations[index].pairs:
            reservations[index] = replace(
                reservations[index], attempted_replicates=len(outputs) + 1
            )
            try:
                outputs.append(operation(left, right))
            except Exception as error:
                reservations[index] = replace(
                    reservations[index],
                    status="domain_error"
                    if isinstance(error, TrialDomainError)
                    else "oracle_error",
                    completed_replicates=len(outputs),
                )
                raise
            reservations[index] = replace(reservations[index], completed_replicates=len(outputs))
        reservations[index] = replace(reservations[index], status="complete")
        return tuple(outputs)

    def result(
        reason: Literal[
            "gradient_band",
            "sampling_unresolved",
            "history_budget",
            "radius_limit",
            "iteration_budget",
        ],
        detail: str = "",
    ) -> StochasticRecoveryResult:
        return StochasticRecoveryResult(
            parameters,
            reason,
            first_history,
            tuple(steps),
            tuple(domain_rejections),
            tuple(gradient_attempts),
            detail or reason,
            acceptance_attempts=tuple(acceptance_attempts),
            reservations=tuple(reservations),
            local_models=tuple(local_models),
            final_validation_attempts=tuple(validation_attempts),
            sampling_classification="verified_deterministic_expectation"
            if deterministic
            else "stochastic",
            proposal_policy=selected.proposal,
            uncertainty_policy=band_method,
            numerical_gradient_allowance=selected.numerical_gradient_allowance,
            final_validation_batch=final_size,
            final_validation_trigger=validation_trigger,
        )

    def record_exception(error: Exception) -> None:
        # Preserve the normal numerical exception contract, with a serialisable
        # snapshot for callers recording a failed run. Partial device work is
        # deliberately unknown here; the production operator records its cost.
        error.__dict__["recovery_diagnostics"] = result("sampling_unresolved", "oracle_exception")
        error.add_note("Recovery reservation/attempt snapshot is in recovery_diagnostics.")

    def log_acceptance(
        base: AcceptanceAttempt,
        changes: list[float],
        outcome: str,
        change: float | None = None,
        error: float | None = None,
    ) -> None:
        ratio = None if change is None else -change / base.predicted_decrease
        acceptance_attempts.append(
            replace(
                base,
                outcome=outcome,
                replicate_changes=tuple(changes),
                mean_change=change,
                standard_error=error,
                agreement=ratio if ratio is None or math.isfinite(ratio) else None,
            )
        )

    def finish_stationarity(
        iteration: int, trigger: str = "proposal_gradient_band"
    ) -> StochasticRecoveryResult:
        nonlocal validation_trigger
        if final_size is None:
            return result("gradient_band")
        validation_trigger = trigger
        index = reserve(final_size, "final_validation", iteration)
        if index is None:
            return result("history_budget", "final_validation_reservation_failed")
        try:
            values = execute(index, partial(oracle.gradient_replicate, parameters))
            attempt = _gradient_attempt(
                values,
                parameters,
                selected,
                deterministic,
                iteration=iteration,
                batch_size=final_size,
                first_history=reservations[index].first_history,
                histories_used=reservations[index].reserved_histories,
                model_id=None,
                pool="final_validation",
            )
        except Exception as error:
            record_exception(error)
            raise
        validation_attempts.append(attempt)
        if attempt.decision == "gradient_band":
            return result("gradient_band", "held_out_gradient_band")
        return result("sampling_unresolved", "final_validation_failed")

    def finish_normal(
        reason: Literal["sampling_unresolved", "radius_limit"], detail: str, iteration: int
    ) -> StochasticRecoveryResult:
        # A development decision can be unresolved even when the fixed incumbent
        # is stationary. Spend its already reserved, independent checkpoint once.
        # Final failure terminates; none of these samples select another point.
        if final_size is not None and first_history + final_cost <= selected.unique_history_budget:
            return finish_stationarity(iteration, detail or reason)
        return result(reason, detail)

    for iteration in range(selected.iterations):
        if not cached:
            # Fresh-batch growth is deliberately retained: there is no pooling
            # across adaptive epochs or different parameter values.
            while True:
                checkpoint = final_cost
                if final_size is not None:
                    checkpoint += 2 * batch_size * selected.replicates
                index = reserve(
                    batch_size,
                    "model" if quadratic else "gradient",
                    iteration,
                    checkpoint,
                )
                if index is None:
                    if final_size is not None and any(value.accepted for value in steps):
                        return finish_stationarity(
                            iteration, "proposal_checkpoint_reservation_failed"
                        )
                    return result(
                        "history_budget",
                        "proposal_checkpoint_reservation_failed"
                        if checkpoint
                        else "history_budget",
                    )
                model_id = len(local_models) if quadratic else None
                try:
                    curvature: Matrix = ()
                    if quadratic:
                        model_oracle = cast(QuadraticSquaredOracle, oracle)
                        models = execute(index, partial(model_oracle.model_replicate, parameters))
                        replicates = tuple(value[0] for value in models)
                        size = len(parameters)
                        if any(
                            len(value[1]) != size or any(len(row) != size for row in value[1])
                            for value in models
                        ):
                            raise ContractError(
                                "model curvature dimension differs from the active chart"
                            )
                        curvature = tuple(
                            tuple(
                                mean_standard_error(tuple(value[1][i][j] for value in models))[0]
                                for j in range(size)
                            )
                            for i in range(size)
                        )
                        validate_curvature(curvature, size)
                    else:
                        replicates = execute(index, partial(oracle.gradient_replicate, parameters))
                    attempt = _gradient_attempt(
                        replicates,
                        parameters,
                        selected,
                        deterministic,
                        iteration=iteration,
                        batch_size=batch_size,
                        first_history=reservations[index].first_history,
                        histories_used=reservations[index].reserved_histories,
                        model_id=model_id,
                    )
                except Exception as error:
                    record_exception(error)
                    raise
                gradient_attempts.append(attempt)
                if model_id is not None:
                    local_models.append(
                        LocalModel(
                            model_id,
                            parameters,
                            attempt.gradient,
                            curvature,
                            attempt.first_history,
                            attempt.histories_used,
                            batch_size,
                        )
                    )
                if attempt.decision == "zero_sample":
                    if batch_size == selected.maximum_batch:
                        return finish_normal(
                            "sampling_unresolved",
                            "zero_gradient_and_variance_at_maximum_batch",
                            iteration,
                        )
                    batch_size = min(2 * batch_size, selected.maximum_batch)
                    continue
                if attempt.decision == "gradient_band":
                    return finish_stationarity(iteration)
                if attempt.decision == "resolved":
                    gradient, norm = attempt.gradient, attempt.gradient_norm
                    if model_id is not None:
                        cached[:] = [local_models[-1]]
                    break
                if batch_size == selected.maximum_batch:
                    return finish_normal(
                        "sampling_unresolved", "gradient_uncertainty_at_maximum_batch", iteration
                    )
                batch_size = min(2 * batch_size, selected.maximum_batch)
        else:
            gradient, norm = cached[0].gradient, math.hypot(*cached[0].gradient)
        proposal: QuadraticProposal | None = None
        active_curvature: Matrix = ()
        if quadratic:
            assert cached
            active_curvature = cached[0].curvature
            proposal = quadratic_proposal(
                gradient,
                active_curvature,
                radius,
                damping_relative=damping_relative,
                rank_tolerance=rank_tolerance,
            )
            step, predicted, boundary = (
                proposal.step,
                proposal.predicted_decrease,
                proposal.boundary,
            )
        else:
            step = tuple(-radius * (value / norm) for value in gradient)
            predicted, boundary = radius * norm, True
        candidate: Vector = tuple(a + b for a, b in zip(parameters, step, strict=True))
        if not math.isfinite(predicted) or predicted <= 0 or not all(map(math.isfinite, candidate)):
            raise NumericalError("stochastic proposal exceeds the finite chart range")
        if candidate == parameters:
            return finish_normal(
                "sampling_unresolved", "proposal_below_chart_resolution", iteration
            )
        # Addition in a large finite chart can round the requested displacement.
        # Log the realised move and assess its actual quadratic prediction. The
        # legacy linear prediction/decisions remain unchanged for the ablation.
        step = tuple(b - a for a, b in zip(parameters, candidate, strict=True))
        if proposal is not None:
            actual_norm = math.hypot(*step)
            if actual_norm > radius * (1 + 1e-12):
                return finish_normal(
                    "sampling_unresolved", "chart_rounding_exceeds_trust_radius", iteration
                )
            predicted = quadratic_reduction(gradient, active_curvature, step)
            boundary = proposal.boundary and actual_norm >= radius * (1 - 1e-12)
            proposal = replace(proposal, step=step, predicted_decrease=predicted, boundary=boundary)
        look = 0
        while True:
            look += 1
            start = first_history
            index = reserve(batch_size, "acceptance", iteration, final_cost)
            changes: list[float] = []
            threshold = -selected.acceptance_fraction * predicted

            base = AcceptanceAttempt(
                iteration,
                parameters,
                candidate,
                None if not cached else cached[0].model_id,
                step,
                predicted,
                radius,
                batch_size,
                start,
                first_history - start,
                2 * batch_size * selected.replicates,
                (),
                None,
                None,
                band_method,
                look,
                threshold,
                "reserved",
                None,
                boundary,
                proposal,
            )

            if index is None:
                log_acceptance(base, changes, "budget_denied")
                return result(
                    "history_budget",
                    "acceptance_checkpoint_reservation_failed" if final_cost else "history_budget",
                )

            def change_operation(
                a: HistoryBatch,
                b: HistoryBatch,
                *,
                before: Vector = parameters,
                after: Vector = candidate,
                outputs: list[float] = changes,
            ) -> float:
                value = oracle.change_replicate(before, after, a, b)
                if not math.isfinite(value):
                    raise NumericalError("objective-change replicate is nonfinite")
                outputs.append(value)
                return value

            try:
                execute(index, change_operation)
                change, standard_error = mean_standard_error(changes)
                band = selected.standard_error_multiplier * standard_error
                if not math.isfinite(band):
                    raise NumericalError("objective-change uncertainty band exceeds finite range")
            except TrialDomainError as error:
                log_acceptance(base, changes, "domain_error")
                domain_rejections.append(DomainRejection(iteration, radius, str(error)))
                radius *= 0.5
                if radius < selected.minimum_radius:
                    return finish_normal("radius_limit", "radius_limit", iteration)
                break
            except Exception as error:
                log_acceptance(base, changes, "oracle_error")
                record_exception(error)
                raise
            if change + band < threshold:
                log_acceptance(base, changes, "accepted", change, standard_error)
                steps.append(
                    StochasticStep(
                        iteration,
                        True,
                        change,
                        standard_error,
                        predicted,
                        radius,
                        batch_size,
                        start,
                        first_history - start,
                    )
                )
                parameters = candidate
                cached.clear()
                if not quadratic or (
                    boundary and -change / predicted >= selected.radius_growth_agreement
                ):
                    radius = min(2 * radius, selected.maximum_radius)
                break
            if change - band >= threshold:
                log_acceptance(base, changes, "rejected", change, standard_error)
                steps.append(
                    StochasticStep(
                        iteration,
                        False,
                        change,
                        standard_error,
                        predicted,
                        radius,
                        batch_size,
                        start,
                        first_history - start,
                    )
                )
                radius *= 0.5
                if radius < selected.minimum_radius:
                    return finish_normal("radius_limit", "radius_limit", iteration)
                break
            log_acceptance(base, changes, "ambiguous", change, standard_error)
            if batch_size == selected.maximum_batch:
                return finish_normal(
                    "sampling_unresolved", "acceptance_uncertainty_at_maximum_batch", iteration
                )
            batch_size = min(2 * batch_size, selected.maximum_batch)
    if final_size is not None:
        return finish_stationarity(selected.iterations, "iteration_budget")
    return result("iteration_budget")


# endregion book:stochastic-independent-acceptance
