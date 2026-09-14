"""Composition of the canonical projector, transmission, objective and optimiser.

The fixed volume, observation and reusable image buffers remain on CUDA. Each
evaluation explicitly uploads a twelve-value rigid transform and downloads one
loss and six local pose derivatives, plus explicit numerical-status checkpoints.
This is a deterministic primary-model
adapter; spectral/nuisance adapters use the same scaled optimiser contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from dpt._runtime import require_no_tape
from dpt.contracts import ContractError, NumericalError, finite_scalar
from dpt.geometry import DetectorGeometry, RigidTransform, chart_gradient, compose_pose
from dpt.objectives import (
    ObjectiveSpec,
    evaluate_objective,
    evaluate_primary_objective,
    prepare_objective,
)
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.registration import Evaluation, RecoveryPolicy, RecoveryResult, Vector, recover_parameters
from dpt.transmission import TransmissionSpec, prepare_transmission, transmission_vjp, transmit
from dpt.volumes import GridSpec


@dataclass(frozen=True, slots=True)
class PoseChart:
    """Dimensionless optimiser coordinates mapped into one fixed SE(3) chart."""

    anchor: RigidTransform = field(default_factory=RigidTransform)
    scales: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 0.01, 0.01, 0.01)
    rotation_radius_radians: float = math.pi

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, RigidTransform):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ContractError("pose anchor must be a validated RigidTransform")
        if len(self.scales) != 6:
            raise ContractError("pose scales must contain three mm and three radian values")
        for value in self.scales:
            if finite_scalar(value, "pose scale", minimum=0.0) == 0:
                raise ContractError("pose scales must be strictly positive")
        object.__setattr__(self, "scales", tuple(float(value) for value in self.scales))
        radius = finite_scalar(self.rotation_radius_radians, "rotation radius", minimum=0.0)
        if not 0 < radius <= math.pi:
            raise ContractError("fixed pose chart rotation radius must lie in (0, pi]")

    def increment(self, parameters: Vector) -> Vector:
        if len(parameters) != 6:
            raise ContractError("a rigid-pose evaluator needs six chart parameters")
        values = tuple(a * b for a, b in zip(parameters, self.scales, strict=True))
        if (
            not all(map(math.isfinite, values))
            or math.hypot(*values[3:]) >= self.rotation_radius_radians
        ):
            raise NumericalError("trial pose lies outside the declared fixed rotation chart")
        return values

    def pose(self, parameters: Vector) -> RigidTransform:
        increment = self.increment(parameters)
        try:
            return compose_pose(self.anchor, increment)
        except (ValueError, OverflowError) as error:
            raise NumericalError("trial transform exceeds the finite rigid-pose range") from error

    def gradient(self, parameters: Vector, local_gradient: Vector) -> Vector:
        gradient = chart_gradient(self.increment(parameters), local_gradient)
        return tuple(a * b for a, b in zip(gradient, self.scales, strict=True))


@dataclass(frozen=True, slots=True)
class PrimaryPoseProblem:
    grid: GridSpec
    geometry: DetectorGeometry
    attenuation: Any
    observation: Any
    objective: ObjectiveSpec = field(default_factory=lambda: ObjectiveSpec(domain="counts"))
    open_beam: float = 1.0
    weights: Any = None
    samples_per_ray: int = 256
    precision: Literal["float32", "float64"] = "float64"
    integration: Literal["midpoint", "cell_gauss"] = "midpoint"


@dataclass(frozen=True, slots=True)
class PoseRecoveryResult:
    pose: RigidTransform
    optimisation: RecoveryResult


class PrimaryPoseEvaluator:
    """Prepared single-stream composition; never share concurrently.

    Attenuation, observations, weights and chart are immutable for a solve.
    The first evaluation checks their device values; later calls rely on that
    immutability and check numerical output flags at explicit completion points.
    FP64 depth, signal and depth cotangents are retained by default so a pose
    line search can resolve the loss changes indicated by its gradient. Explicit
    precision="float32" reproduces the earlier composed storage contract.
    Scratch contents after return may correspond to a rejected trial; the result
    returned by recover_pose identifies the accepted pose and objective. Call
    evaluate explicitly at that pose if a final predicted image is needed.
    """

    def __init__(
        self,
        problem: PrimaryPoseProblem,
        chart: PoseChart,
        *,
        device: str = "cuda:0",
        stream: Any = None,
    ) -> None:
        self.problem, self.chart = problem, chart
        if problem.objective.domain not in ("counts", "log_transmission"):
            raise ContractError("primary pose recovery requires counts or log-transmission data")
        self.projection = prepare_projection(
            problem.grid,
            problem.geometry,
            ProjectionSpec(
                problem.samples_per_ray,
                precision=problem.precision,
                integration=problem.integration,
            ),
            device=device,
            stream=stream,
        )
        self.context = self.projection.context
        ctx, wp, count = self.context, self.context.wp, problem.geometry.pixels
        ctx.array(
            problem.attenuation, "attenuation", dtype=wp.float32, shape=(problem.grid.voxels,)
        )
        ctx.array(problem.observation, "observation", dtype=wp.float32, shape=(count,))
        if problem.objective.weighted:
            ctx.array(problem.weights, "weights", dtype=wp.float32, shape=(count,))
        elif problem.weights is not None:
            raise ContractError("weights require an explicitly weighted objective")
        self.transmission = (
            prepare_transmission(
                TransmissionSpec(beam="scalar" if problem.objective.domain == "counts" else "none"),
                max_pixels=count,
                device=device,
                stream=ctx.stream,
            )
            if problem.precision == "float32"
            else None
        )
        self.objective = prepare_objective(
            problem.objective, max_pixels=count, device=device, stream=ctx.stream
        )
        with ctx.scope():
            self.pose_device = wp.empty(12, dtype=wp.float64, device=ctx.device)
            self.optical_depth = wp.empty(count, dtype=self.projection.dtype, device=ctx.device)
            self.prediction = wp.empty(count, dtype=self.projection.dtype, device=ctx.device)
            self.image_seed = wp.empty(
                count if problem.precision == "float32" else 0, dtype=wp.float32, device=ctx.device
            )
            self.depth_seed = wp.empty(count, dtype=self.projection.dtype, device=ctx.device)
            self.pose_seed = wp.empty(6, dtype=wp.float64, device=ctx.device)
            self.loss = wp.empty(1, dtype=wp.float64, device=ctx.device)
        # Pinned host staging has a persistent owner and a fixed-size NumPy view.
        # These are explicit control-plane transfers, not per-pixel Python work.
        self._pose_host = wp.empty(12, dtype=wp.float64, device="cpu", pinned=True)
        self._loss_host = wp.empty(1, dtype=wp.float64, device="cpu", pinned=True)
        self._gradient_host = wp.empty(6, dtype=wp.float64, device="cpu", pinned=True)
        self._pose_view = self._pose_host.numpy()
        self._loss_view = self._loss_host.numpy()
        self._gradient_view = self._gradient_host.numpy()
        self._inputs_checked = False

    # region book:recovery-device-composition
    def __call__(self, parameters: Vector) -> Evaluation:
        require_no_tape()
        try:
            return self._evaluate(parameters)
        finally:
            # Even a failed launch can follow an enqueued asynchronous H2D copy.
            # Release its read of pinned staging before a retry may overwrite it.
            self.context.wp.synchronize_stream(self.context.stream)

    def _evaluate(self, parameters: Vector) -> Evaluation:
        pose = self.chart.pose(parameters)
        ctx, wp, problem = self.context, self.context.wp, self.problem
        validate = not self._inputs_checked
        self.projection.clear_status()
        if self.transmission is not None:
            self.transmission.clear_status()
        self.objective.clear_status()
        with ctx.scope():
            self._pose_view[:] = pose.packed()
            wp.copy(self.pose_device, self._pose_host, stream=ctx.stream)
            project_optical_depth(
                problem.attenuation,
                self.pose_device,
                out_L=self.optical_depth,
                workspace=self.projection,
                stream=ctx.stream,
                validate=validate,
            )
            if problem.precision == "float64":
                evaluate_primary_objective(
                    self.optical_depth,
                    problem.observation,
                    open_beam=problem.open_beam,
                    weights=problem.weights,
                    out_prediction=self.prediction,
                    out_loss=self.loss,
                    out_depth_seed=self.depth_seed,
                    workspace=self.objective,
                    stream=ctx.stream,
                    validate=validate,
                )
            else:
                assert self.transmission is not None
                counts = problem.objective.domain == "counts"
                transmit(
                    self.optical_depth,
                    problem.open_beam if counts else None,
                    out_counts=self.prediction if counts else None,
                    out_log_T=None if counts else self.prediction,
                    workspace=self.transmission,
                    stream=ctx.stream,
                    validate=validate,
                )
                evaluate_objective(
                    self.prediction,
                    problem.observation,
                    weights=problem.weights,
                    out_loss=self.loss,
                    out_seed=self.image_seed,
                    workspace=self.objective,
                    stream=ctx.stream,
                    validate=validate,
                )
                transmission_vjp(
                    self.optical_depth,
                    problem.open_beam if counts else None,
                    seed_counts=self.image_seed if counts else None,
                    seed_log_T=None if counts else self.image_seed,
                    out_grad_L=self.depth_seed,
                    workspace=self.transmission,
                    stream=ctx.stream,
                    validate=validate,
                )
            projection_vjp(
                problem.attenuation,
                self.pose_device,
                adj_L=self.depth_seed,
                out_pose=self.pose_seed,
                workspace=self.projection,
                stream=ctx.stream,
                validate=validate,
            )
            self.projection.check_status()
            if self.transmission is not None:
                self.transmission.check_status()
            self.objective.check_status()
            self._inputs_checked = True
            wp.copy(self._loss_host, self.loss, stream=ctx.stream)
            wp.copy(self._gradient_host, self.pose_seed, stream=ctx.stream)
            wp.synchronize_stream(ctx.stream)
        gradient = tuple(float(value) for value in self._gradient_view)
        return Evaluation(float(self._loss_view[0]), self.chart.gradient(parameters, gradient))

    # endregion book:recovery-device-composition


def recover_pose(
    evaluator: PrimaryPoseEvaluator,
    *,
    initial: Vector = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    policy: RecoveryPolicy | None = None,
) -> PoseRecoveryResult:
    """Run the canonical safeguarded driver without rebasing its curvature chart."""
    result = recover_parameters(evaluator, initial, policy=policy)
    return PoseRecoveryResult(evaluator.chart.pose(result.parameters), result)
