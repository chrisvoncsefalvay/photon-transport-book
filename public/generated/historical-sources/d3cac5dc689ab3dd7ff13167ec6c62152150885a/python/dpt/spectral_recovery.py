"""Multi-view spectral pose recovery with constrained shared acquisition parameters.

This adapter composes the canonical material projector, spectral response,
spatial response, calibration and objective. The optimiser sees a fixed SE(3)
chart followed by named nuisance blocks. Detector-sized data stay on CUDA;
pinned control buffers move only pose, nuisance values, losses, gradients and
status flags. No observed image is synthesised or normalised by this adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from dpt._runtime import prepare_context, require_no_tape
from dpt.contracts import ContractError, NumericalError, binary32_scalar, finite_scalar, integer
from dpt.detector import (
    BlurSpec,
    BlurWorkspace,
    CalibrationSpec,
    DetectorWorkspace,
    blur,
    blur_transpose,
    calibrate,
    calibration_vjp,
    prepare_blur,
    prepare_detector,
)
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_projection import (
    MaterialProjectionWorkspace,
    material_projection_vjp,
    prepare_material_projection,
    project_material_paths,
)
from dpt.materials import Provenance
from dpt.objectives import ObjectiveSpec, ObjectiveWorkspace, evaluate_objective, prepare_objective
from dpt.projection import ProjectionSpec
from dpt.recovery import PoseChart
from dpt.registration import Evaluation, RecoveryPolicy, RecoveryResult, Vector, recover_parameters
from dpt.spectral import (
    SpectralSpec,
    SpectralWorkspace,
    prepare_spectral,
    spectral_signal,
    spectral_vjp,
)
from dpt.volumes import GridSpec

# Public Python boundaries validate runtime inputs.
# pyright: reportUnnecessaryIsInstance=false


def _physical_parameter(value: float, name: str, *, positive: bool = False) -> float:
    """Translate shared scalar storage checks into rejectable trial-domain errors."""
    try:
        rounded = binary32_scalar(value, f"trial {name}")
    except ContractError as error:
        raise NumericalError(str(error)) from error
    if positive and rounded <= 0:
        raise NumericalError(f"trial {name} lies outside its representable physical domain")
    return rounded


def _scaled_product(*values: float) -> float:
    """Preserve exponent range until the final binary64 chart product is formed."""
    mantissa, exponent = 1.0, 0
    for value in values:
        if not math.isfinite(value):
            raise NumericalError("non-finite physical derivative or chart scale")
        part, power = math.frexp(value)
        mantissa *= part
        exponent += power
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError as error:
        raise NumericalError("chart derivative overflows binary64") from error


# region book:calibration-parameter-chart
@dataclass(frozen=True, slots=True)
class CalibrationBlock:
    """Named acquisition group sharing one gain/exposure/offset across its views.

    Exactly one multiplicative scale may be fitted. Its positive chart is
    reference*exp(scale_step*z); the other scale is fixed to remove the gain /
    exposure gauge. An active offset is reference+offset_step*z in output units.
    Optional Gaussian priors act on these dimensionless chart coordinates.

    Device parameters are rounded to binary32. The VJP differentiates the smooth
    physical chart at those represented values, not the discontinuous rounding
    map. Shared physical derivatives stay binary64 until the chart product is
    formed, so a large positive scale can preserve a small logarithmic partial.
    """

    name: str
    gain: float = 1.0
    exposure: float = 1.0
    offset: float = 0.0
    fit_scale: Literal["none", "gain", "exposure"] = "none"
    fit_offset: bool = False
    scale_step: float = 1.0
    offset_step: float = 1.0
    scale_prior_precision: float = 0.0
    offset_prior_precision: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ContractError("calibration groups need non-empty names")
        if self.fit_scale not in ("none", "gain", "exposure") or type(self.fit_offset) is not bool:
            raise ContractError("invalid calibration activity specification")
        for name in ("gain", "exposure", "scale_step", "offset_step"):
            if finite_scalar(getattr(self, name), name, minimum=0) <= 0:
                raise ContractError(f"{name} must be strictly positive in the recovery chart")
        finite_scalar(self.offset, "offset")
        for name in ("scale_prior_precision", "offset_prior_precision"):
            finite_scalar(getattr(self, name), name, minimum=0)
        if self.fit_scale == "none" and self.scale_prior_precision:
            raise ContractError("a scale prior requires an active scale")
        if not self.fit_offset and self.offset_prior_precision:
            raise ContractError("an offset prior requires an active offset")
        for name in ("gain", "exposure", "offset"):
            try:
                _physical_parameter(getattr(self, name), name, positive=name != "offset")
            except NumericalError as error:
                raise ContractError(str(error)) from error

    @property
    def dimension(self) -> int:
        return int(self.fit_scale != "none") + int(self.fit_offset)

    def decode(self, coordinates: Vector) -> tuple[float, float, float]:
        if len(coordinates) != self.dimension:
            raise ContractError("nuisance coordinates differ from the declared group chart")
        if not all(math.isfinite(value) for value in coordinates):
            raise NumericalError("trial nuisance coordinates are non-finite")
        gain, exposure, offset = self.gain, self.exposure, self.offset
        cursor = 0
        if self.fit_scale != "none":
            try:
                scale = math.exp(self.scale_step * coordinates[cursor])
            except OverflowError as error:
                raise NumericalError("trial logarithmic calibration scale overflowed") from error
            if self.fit_scale == "gain":
                gain *= scale
            else:
                exposure *= scale
            cursor += 1
        if self.fit_offset:
            offset += self.offset_step * coordinates[cursor]
        return (
            _physical_parameter(gain, "gain", positive=True),
            _physical_parameter(exposure, "exposure", positive=True),
            _physical_parameter(offset, "offset"),
        )

    def gradient(self, coordinates: Vector, physical_gradient: Vector) -> Vector:
        if len(physical_gradient) != 3:
            raise ContractError("a calibration gradient needs gain, exposure and offset partials")
        gain, exposure, _ = self.decode(coordinates)
        gradient: list[float] = []
        if self.fit_scale != "none":
            index, value = (0, gain) if self.fit_scale == "gain" else (1, exposure)
            gradient.append(_scaled_product(self.scale_step, value, physical_gradient[index]))
        if self.fit_offset:
            gradient.append(_scaled_product(self.offset_step, physical_gradient[2]))
        return tuple(gradient)

    def prior(self, coordinates: Vector) -> Evaluation:
        if len(coordinates) != self.dimension:
            raise ContractError("prior coordinates differ from calibration chart")
        precisions = () if self.fit_scale == "none" else (self.scale_prior_precision,)
        precisions += (self.offset_prior_precision,) if self.fit_offset else ()
        return Evaluation(
            math.fsum(0.5 * p * z * z for p, z in zip(precisions, coordinates, strict=True)),
            tuple(p * z for p, z in zip(precisions, coordinates, strict=True)),
        )


# endregion book:calibration-parameter-chart


@dataclass(frozen=True, slots=True)
class SpectralView:
    name: str
    geometry: DetectorGeometry
    spectral: SpectralSpec
    coefficients: Any
    weights: Any
    response: Any
    observation: Any
    calibration_group: str
    observation_provenance: Provenance
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    objective_weights: Any = None
    objective_weight: float = 1.0
    spatial_response: BlurSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation_provenance, Provenance):
            raise ContractError("each observation requires acquisition provenance")
        if not self.name.strip() or not self.calibration_group.strip():
            raise ContractError("view and calibration-group names are required")
        if finite_scalar(self.objective_weight, "view objective weight", minimum=0) <= 0:
            raise ContractError("view objective weight must be strictly positive")
        if self.objective.domain == "log_transmission":
            raise ContractError(
                "spectral recovery compares calibrated signal or counts, not log data"
            )
        if (
            self.spatial_response is not None
            and (self.spatial_response.height, self.spatial_response.width) != self.geometry.shape
        ):
            raise ContractError("spatial response and detector shapes differ")
        if not self.spectral.active_paths:
            raise ContractError("spectral pose recovery needs active material paths")
        if (
            self.spectral.active_weights
            or self.spectral.active_response
            or self.spectral.active_coefficients
        ):
            raise ContractError(
                "this recovery adapter keeps spectrum, response and coefficients fixed"
            )
        if not self.objective.weighted and self.objective_weights is not None:
            raise ContractError("objective weights require an explicitly weighted objective")


@dataclass(frozen=True, slots=True)
class SpectralPoseProblem:
    grid: GridSpec
    materials: int
    fields: Any
    fields_provenance: Provenance
    views: tuple[SpectralView, ...]
    calibration: tuple[CalibrationBlock, ...]
    samples_per_ray: int = 256

    def __post_init__(self) -> None:
        integer(self.materials, "materials", minimum=1, maximum=32)
        integer(self.samples_per_ray, "samples_per_ray", minimum=1)
        if not isinstance(self.fields_provenance, Provenance):
            raise ContractError("fixed material fields require provenance")
        if not isinstance(self.views, tuple) or not self.views:
            raise ContractError(
                "a spectral recovery problem needs an immutable non-empty view tuple"
            )
        if not isinstance(self.calibration, tuple) or not self.calibration:
            raise ContractError("calibration blocks must be an immutable non-empty tuple")
        if len({view.name for view in self.views}) != len(self.views):
            raise ContractError("view names must be unique")
        groups = {block.name: block for block in self.calibration}
        if len(groups) != len(self.calibration):
            raise ContractError("calibration group names must be unique")
        if set(groups) != {view.calibration_group for view in self.views}:
            raise ContractError("every named calibration group must be defined and used")
        for group in groups:
            units = {
                view.spectral.output_unit for view in self.views if view.calibration_group == group
            }
            if len(units) != 1:
                raise ContractError(
                    "shared calibration groups require identical detector output units"
                )
        for view in self.views:
            if view.objective.domain == "counts" and view.spectral.output_unit != "counts":
                raise ContractError("count-domain views must declare spectral output_unit='counts'")
            if view.spectral.materials != self.materials:
                raise ContractError("all views must use the same fixed material basis")
            block = groups[view.calibration_group]
            if view.objective.kind == "poisson" and (
                block.gain != 1.0
                or block.offset != 0.0
                or block.fit_scale == "gain"
                or block.fit_offset
                or view.spatial_response is not None
            ):
                raise ContractError(
                    "Poisson counts require unit gain, zero offset and no spatial blur"
                )


@dataclass(slots=True)
class _PreparedView:
    view: SpectralView
    group: int
    material: MaterialProjectionWorkspace
    spectral: SpectralWorkspace
    detector: DetectorWorkspace
    objective: ObjectiveWorkspace
    blur: BlurWorkspace | None
    calibration_spec: CalibrationSpec
    mean: Any
    spread: Any
    prediction: Any
    image_seed: Any
    spread_seed: Any
    mean_seed: Any
    loss: Any
    pose_gradient: Any
    nuisance_gradients: tuple[Any, Any, Any]


@dataclass(frozen=True, slots=True)
class CalibrationEstimate:
    name: str
    gain: float
    exposure: float
    offset: float


@dataclass(frozen=True, slots=True)
class SpectralRecoveryResult:
    pose: RigidTransform
    calibration: tuple[CalibrationEstimate, ...]
    optimisation: RecoveryResult


class SpectralPoseEvaluator:
    """Prepared single-stream multi-view objective with manual first-order products.

    Fields, input spectra, responses, observations and weights are immutable for
    this object's lifetime. Preparation checks their current values. Numerical
    flags are checked through each workspace after evaluation. Every path,
    image and adjoint buffer is allocated once. Concurrent/re-entrant evaluation
    and ambient tape recording are rejected before any staging buffer is changed.
    """

    def __init__(
        self,
        problem: SpectralPoseProblem,
        chart: PoseChart,
        *,
        device: str = "cuda:0",
        stream: Any = None,
    ) -> None:
        require_no_tape()
        self.problem, self.chart = problem, chart
        self.context = prepare_context(device=device, stream=stream)
        ctx, wp = self.context, self.context.wp
        ctx.array(
            problem.fields,
            "fields",
            dtype=wp.float32,
            shape=(problem.materials * problem.grid.voxels,),
        )
        self._group_slices: tuple[slice, ...] = self._parameter_slices()
        self.dimension = 6 + sum(block.dimension for block in problem.calibration)
        self._running = False
        self.last_view_losses: tuple[float, ...] | None = None
        views = len(problem.views)
        with ctx.scope():
            self._pose_device = wp.empty(12, dtype=wp.float64, device=ctx.device)
            self._parameters_device = wp.empty(
                3 * len(problem.calibration), dtype=wp.float32, device=ctx.device
            )
            self._control_device = wp.empty(7 * views, dtype=wp.float64, device=ctx.device)
            self._nuisance_device = wp.zeros(3 * views, dtype=wp.float64, device=ctx.device)
        self._parameter_views = tuple(
            tuple(self._parameters_device[3 * i + j : 3 * i + j + 1] for j in range(3))
            for i in range(len(problem.calibration))
        )
        self._pose_host = wp.empty(12, dtype=wp.float64, device="cpu", pinned=True)
        self._parameters_host = wp.empty(
            3 * len(problem.calibration), dtype=wp.float32, device="cpu", pinned=True
        )
        self._control_host = wp.empty(7 * views, dtype=wp.float64, device="cpu", pinned=True)
        self._nuisance_host = wp.empty(3 * views, dtype=wp.float64, device="cpu", pinned=True)
        self._pose_values = self._pose_host.numpy()
        self._parameter_values = self._parameters_host.numpy()
        self._control_values = self._control_host.numpy()
        self._nuisance_values = self._nuisance_host.numpy()
        self._prepared = tuple(
            self._prepare_view(index, view) for index, view in enumerate(problem.views)
        )
        self._validate_fixed_inputs()

    def _parameter_slices(self) -> tuple[slice, ...]:
        cursor = 6
        slices: list[slice] = []
        for block in self.problem.calibration:
            slices.append(slice(cursor, cursor + block.dimension))
            cursor += block.dimension
        return tuple(slices)

    def _prepare_view(self, index: int, view: SpectralView) -> _PreparedView:
        ctx, wp = self.context, self.context.wp
        group = next(
            i
            for i, block in enumerate(self.problem.calibration)
            if block.name == view.calibration_group
        )
        block = self.problem.calibration[group]
        pixels, spec = view.geometry.pixels, view.spectral
        ctx.array(
            view.coefficients,
            "coefficients",
            dtype=wp.float32,
            shape=(spec.materials * spec.energies,),
        )
        ctx.array(
            view.weights,
            "weights",
            dtype=wp.float32,
            shape=(spec.energies * (1 if spec.shared_weights else pixels),),
        )
        ctx.array(
            view.response,
            "response",
            dtype=wp.float32,
            shape=(spec.energies * (1 if spec.shared_response else pixels),),
        )
        ctx.array(view.observation, "observation", dtype=wp.float32, shape=(pixels,))
        if view.objective.weighted:
            ctx.array(
                view.objective_weights, "objective_weights", dtype=wp.float32, shape=(pixels,)
            )
        with ctx.scope():
            paths = wp.empty(spec.materials * pixels, dtype=wp.float32, device=ctx.device)
            adj_paths = wp.empty_like(paths)
            mean = wp.empty(pixels, dtype=wp.float32, device=ctx.device)
            mean_seed = wp.empty_like(mean)
            spread = wp.empty_like(mean) if view.spatial_response is not None else mean
            spread_seed = wp.empty_like(mean) if view.spatial_response is not None else mean_seed
            prediction, image_seed = wp.empty_like(mean), wp.empty_like(mean)
        material = prepare_material_projection(
            self.problem.grid,
            view.geometry,
            ProjectionSpec(self.problem.samples_per_ray),
            materials=self.problem.materials,
            fields=self.problem.fields,
            out_paths=paths,
            adj_paths=adj_paths,
            device=ctx.device,
            stream=ctx.stream,
        )
        spectral = prepare_spectral(spec, max_pixels=pixels, device=ctx.device, stream=ctx.stream)
        detector = prepare_detector(max_pixels=pixels, device=ctx.device, stream=ctx.stream)
        objective = prepare_objective(
            view.objective, max_pixels=pixels, device=ctx.device, stream=ctx.stream
        )
        spatial = (
            prepare_blur(view.spatial_response, device=ctx.device, stream=ctx.stream)
            if view.spatial_response is not None
            else None
        )
        calibration_spec = CalibrationSpec(
            spec.output_unit,
            active_gain=block.fit_scale == "gain",
            active_exposure=block.fit_scale == "exposure",
            active_offset=block.fit_offset,
        )
        return _PreparedView(
            view,
            group,
            material,
            spectral,
            detector,
            objective,
            spatial,
            calibration_spec,
            mean,
            spread,
            prediction,
            image_seed,
            spread_seed,
            mean_seed,
            self._control_device[7 * index : 7 * index + 1],
            self._control_device[7 * index + 1 : 7 * index + 7],
            tuple(self._nuisance_device[3 * index + j : 3 * index + j + 1] for j in range(3)),
        )

    def _workspaces(
        self,
    ) -> tuple[
        MaterialProjectionWorkspace
        | SpectralWorkspace
        | DetectorWorkspace
        | ObjectiveWorkspace
        | BlurWorkspace,
        ...,
    ]:
        """Use public workspace diagnostics; status encodings stay with operators."""
        return tuple(
            workspace
            for item in self._prepared
            for workspace in (
                item.material,
                item.spectral,
                item.detector,
                item.objective,
                item.blur,
            )
            if workspace is not None
        )

    def _clear_statuses(self) -> None:
        for workspace in self._workspaces():
            workspace.clear_status()

    def _check_statuses(self) -> None:
        for workspace in self._workspaces():
            workspace.check_status()

    def _validate_fixed_inputs(self) -> None:
        for item in self._prepared:
            item.material.validate_inputs(stream=self.context.stream)
            item.spectral.validate_inputs(
                item.view.coefficients,
                item.view.weights,
                item.view.response,
                pixels=item.view.geometry.pixels,
                probability_response=item.view.objective.kind == "poisson",
                stream=self.context.stream,
            )
            item.objective.validate_observation(
                item.view.observation,
                weights=item.view.objective_weights,
                stream=self.context.stream,
            )

    # region book:spectral-recovery-composition
    def __call__(self, parameters: Vector) -> Evaluation:
        require_no_tape()
        if self._running:
            raise ContractError("a spectral evaluator cannot be used concurrently or recursively")
        if len(parameters) != self.dimension:
            raise ContractError("parameter count differs from pose and calibration charts")
        pose = self.chart.pose(parameters[:6])
        decoded = tuple(
            block.decode(parameters[section])
            for block, section in zip(self.problem.calibration, self._group_slices, strict=True)
        )
        ctx, wp = self.context, self.context.wp
        self._running = True
        self.last_view_losses = None
        try:
            with ctx.scope():
                try:
                    self._clear_statuses()
                    self._pose_values[:] = pose.packed()
                    self._parameter_values[:] = tuple(value for block in decoded for value in block)
                    wp.copy(self._pose_device, self._pose_host, stream=ctx.stream)
                    wp.copy(self._parameters_device, self._parameters_host, stream=ctx.stream)
                    for item in self._prepared:
                        view = item.view
                        gain, exposure, offset = self._parameter_views[item.group]
                        project_material_paths(
                            self._pose_device,
                            workspace=item.material,
                            stream=ctx.stream,
                            validate=False,
                        )
                        spectral_signal(
                            item.material.paths,
                            view.coefficients,
                            view.weights,
                            view.response,
                            out_mean=item.mean,
                            workspace=item.spectral,
                            stream=ctx.stream,
                            validate=False,
                        )
                        if item.blur is not None:
                            blur(
                                item.mean,
                                out_signal=item.spread,
                                workspace=item.blur,
                                stream=ctx.stream,
                                validate=False,
                            )
                        calibrate(
                            item.spread,
                            gain,
                            exposure,
                            offset,
                            out_signal=item.prediction,
                            spec=item.calibration_spec,
                            workspace=item.detector,
                            stream=ctx.stream,
                            validate=False,
                        )
                        evaluate_objective(
                            item.prediction,
                            view.observation,
                            out_loss=item.loss,
                            out_seed=item.image_seed,
                            weights=view.objective_weights,
                            workspace=item.objective,
                            stream=ctx.stream,
                            validate=False,
                        )
                        calibration_vjp(
                            item.spread,
                            gain,
                            exposure,
                            offset,
                            seed=item.image_seed,
                            out_grad_mean=item.spread_seed,
                            out_grad_gain=item.nuisance_gradients[0]
                            if item.calibration_spec.active_gain
                            else None,
                            out_grad_exposure=item.nuisance_gradients[1]
                            if item.calibration_spec.active_exposure
                            else None,
                            out_grad_offset=item.nuisance_gradients[2]
                            if item.calibration_spec.active_offset
                            else None,
                            spec=item.calibration_spec,
                            workspace=item.detector,
                            stream=ctx.stream,
                            validate=False,
                        )
                        if item.blur is not None:
                            blur_transpose(
                                item.spread_seed,
                                out_grad_signal=item.mean_seed,
                                workspace=item.blur,
                                stream=ctx.stream,
                                validate=False,
                            )
                        spectral_vjp(
                            item.material.paths,
                            view.coefficients,
                            view.weights,
                            view.response,
                            seed=item.mean_seed,
                            out_grad_paths=item.material.adj_paths,
                            workspace=item.spectral,
                            stream=ctx.stream,
                            validate=False,
                        )
                        material_projection_vjp(
                            self._pose_device,
                            workspace=item.material,
                            out_pose=item.pose_gradient,
                            stream=ctx.stream,
                            validate=False,
                        )
                    wp.copy(self._control_host, self._control_device, stream=ctx.stream)
                    wp.copy(self._nuisance_host, self._nuisance_device, stream=ctx.stream)
                finally:
                    # Rejected evaluations must finish using pinned staging before
                    # the optimiser can write the next trial into the same buffers.
                    wp.synchronize_stream(ctx.stream)
            self._check_statuses()
            return self._assemble(parameters)
        finally:
            self._running = False

    # endregion book:spectral-recovery-composition

    def _assemble(self, parameters: Vector) -> Evaluation:
        losses = tuple(float(self._control_values[7 * i]) for i in range(len(self._prepared)))
        pose_gradient = tuple(
            math.fsum(
                item.view.objective_weight * float(self._control_values[7 * i + 1 + j])
                for i, item in enumerate(self._prepared)
            )
            for j in range(6)
        )
        gradient = list(self.chart.gradient(parameters[:6], pose_gradient))
        prior_losses: list[float] = []
        for group, (block, section) in enumerate(
            zip(self.problem.calibration, self._group_slices, strict=True)
        ):
            physical = tuple(
                math.fsum(
                    item.view.objective_weight * float(self._nuisance_values[3 * i + j])
                    for i, item in enumerate(self._prepared)
                    if item.group == group
                )
                for j in range(3)
            )
            prior = block.prior(parameters[section])
            prior_losses.append(prior.loss)
            derivative = block.gradient(parameters[section], physical)
            gradient.extend(a + b for a, b in zip(derivative, prior.gradient, strict=True))
        loss = math.fsum(
            [
                *(
                    item.view.objective_weight * value
                    for item, value in zip(self._prepared, losses, strict=True)
                ),
                *prior_losses,
            ]
        )
        if not math.isfinite(loss) or not all(math.isfinite(value) for value in gradient):
            raise NumericalError("multi-view loss or chart gradient is non-finite")
        self.last_view_losses = losses
        return Evaluation(loss, tuple(gradient))

    def calibration_estimates(self, parameters: Vector) -> tuple[CalibrationEstimate, ...]:
        if len(parameters) != self.dimension:
            raise ContractError("parameter count differs from evaluator chart")
        return tuple(
            CalibrationEstimate(block.name, *block.decode(parameters[section]))
            for block, section in zip(self.problem.calibration, self._group_slices, strict=True)
        )

    @property
    def predicted_images(self) -> tuple[Any, ...]:
        """Borrow the current device predictions; they may belong to a rejected trial.

        Re-evaluate accepted parameters before explicit export. No host copy is
        made by this accessor, and callers must not mutate these borrowed buffers.
        """
        return tuple(item.prediction for item in self._prepared)


def recover_spectral_pose(
    evaluator: SpectralPoseEvaluator,
    *,
    initial: Vector | None = None,
    policy: RecoveryPolicy | None = None,
) -> SpectralRecoveryResult:
    """Use the same safeguarded optimiser and immutable chart as primary recovery."""
    start = (0.0,) * evaluator.dimension if initial is None else initial
    optimisation = recover_parameters(evaluator, start, policy=policy)
    return SpectralRecoveryResult(
        evaluator.chart.pose(optimisation.parameters[:6]),
        evaluator.calibration_estimates(optimisation.parameters),
        optimisation,
    )
