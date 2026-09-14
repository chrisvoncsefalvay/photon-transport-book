"""Matrix-free material reconstruction of declared count or mean-signal data.

This is a fixed-geometry, fixed-spectrum primary-transmission model. It receives
only fitting measurements and calibration, never reference material anatomy.
Inputs are explicit FP32 host arrays; preparation uploads private copies once.
Repeated evaluations retain CUDA fields, observations, channel buffers and
cached Warp views. Scalar acceptance decisions and requested exports reach the
host. Legacy defaults retain independent Poisson counts and fraction-simplex
fields. Explicit signal WLS accepts fixed masks and deterministic objective
weights; an equivalent-basis domain permits nonnegative fields without a sum
cap. Basis units/scales and the observation law belong to supplied provenance.
The module imports without optional GPU/numpy dependencies.
"""

from __future__ import annotations

# Public runtime contracts deliberately validate callers beyond static typing.
# pyright: reportUnnecessaryIsInstance=false
import importlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from dpt._reductions import prepare_reduction
from dpt._runtime import load_kernels, prepare_context, require_no_tape
from dpt.contracts import ContractError, NumericalError, finite_scalar, integer
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_projection import (
    material_projection_vjp,
    prepare_material_projection,
    project_material_paths,
)
from dpt.objectives import ObjectiveSpec, evaluate_objective, prepare_objective
from dpt.projection import ProjectionSpec
from dpt.spectral import SpectralSpec, prepare_spectral, spectral_signal, spectral_vjp
from dpt.volumes import GridSpec


@dataclass(frozen=True, slots=True)
class MaterialReconstructionView:
    """Independent channels at one fixed geometry; no observations are generated.

    ``counts`` has shape (C,H,W), integer detected counts represented in FP32.
    ``weights`` and ``response`` have shape (C,E); weights are already integrated
    expected open-beam photon populations per pixel, response is a detection
    probability in [0,1]. Independent exposures or disjoint Poisson-thinned
    channels must be justified by the caller. Overlapping cumulative thresholds
    are not independent channels. No normalisation or inverse-square adjustment
    is inferred. Inputs must be finite, native-endian, contiguous FP32 arrays.
    """

    geometry: DetectorGeometry
    pose: RigidTransform
    counts: Any
    weights: Any
    response: Any


@dataclass(frozen=True, slots=True)
class MaterialReconstructionSignalView:
    """Fixed-weight mean-signal observations; no independent-count noise claim.

    Observations and deterministic objective weights have shape (C,H,W),
    native contiguous FP32. ``valid`` is an immutable uint8 0/1 mask of that
    shape; masked targets may be nonfinite. Spectral weights have shape (C,E)
    or (C,E,H,W) according to shared_weights. They already include any spatial
    open-beam scale exactly once. Response is shared (C,E). Fixed objective
    weights are not statistical precision unless independently justified.
    """

    geometry: DetectorGeometry
    pose: RigidTransform
    observations: Any
    weights: Any
    response: Any
    objective_weights: Any
    valid: Any


@dataclass(frozen=True, slots=True)
class MaterialReconstructionSettings:
    """Dimensionless projected steps; fraction-gradient penalty beta is mm^-1.

    The stopping test is ||(f-P(f-alpha*g))/alpha||_infinity at the fixed
    ``mapping_step``. The relative threshold is fixed by its initial value.
    Trial steps grow from the last accepted step and are checked by Armijo;
    neither objective stagnation nor an iteration budget means stationarity.
    ``step_selection="bb"`` optionally proposes the metric Barzilai--Borwein
    step (s^T H s)/(s^T y), using consecutive accepted fields and gradients.
    Nonpositive or nonfinite curvature falls back to the geometric proposal.
    Its finite positive proposal is clipped to ``secant_minimum_step`` and
    ``maximum_step`` before the same constrained Armijo search.
    ``acceleration="inertial"`` adds a feasible momentum proposal using the
    preceding accepted field. A failed momentum proposal restarts the ordinary
    projected search at the same step. Every accepted state still decreases the
    objective and meets Armijo using the true gradient and rounded displacement.
    This safeguard is not a convergence-rate claim for the nonlinear objective.
    """

    iterations: int = 100
    initial_step: float = 1.0
    maximum_step: float = 1.0
    regularisation_mm_inverse: float = 0.0
    gradient_mapping_tolerance: float = 1.0e-5
    relative_gradient_mapping_tolerance: float = 1.0e-4
    mapping_step: float = 1.0
    backtracking_factor: float = 0.5
    armijo: float = 1.0e-4
    maximum_backtracks: int = 40
    step_selection: Literal["geometric", "bb"] = "geometric"
    secant_minimum_step: float = 1.0e-16
    acceleration: Literal["none", "inertial"] = "none"

    def __post_init__(self) -> None:
        integer(self.iterations, "iterations", minimum=1)
        integer(self.maximum_backtracks, "maximum_backtracks", minimum=1)
        for name in ("initial_step", "maximum_step", "mapping_step"):
            if finite_scalar(getattr(self, name), name, minimum=0) <= 0:
                raise ContractError(f"{name} must be positive")
        if self.maximum_step < self.initial_step:
            raise ContractError("maximum_step must be at least initial_step")
        if self.step_selection not in ("geometric", "bb"):
            raise ContractError("step_selection must be geometric or bb")
        if self.acceleration not in ("none", "inertial"):
            raise ContractError("acceleration must be none or inertial")
        if finite_scalar(self.secant_minimum_step, "secant_minimum_step", minimum=0) <= 0:
            raise ContractError("secant_minimum_step must be positive")
        if self.step_selection == "bb" and self.secant_minimum_step > self.maximum_step:
            raise ContractError("secant_minimum_step must not exceed maximum_step")
        for name in (
            "regularisation_mm_inverse",
            "gradient_mapping_tolerance",
            "relative_gradient_mapping_tolerance",
        ):
            finite_scalar(getattr(self, name), name, minimum=0)
        for name in ("backtracking_factor", "armijo"):
            if not 0 < finite_scalar(getattr(self, name), name) < 1:
                raise ContractError(f"{name} must lie strictly between zero and one")


class MaterialReconstruction:
    """Persistent CUDA composition, supporting one to eight fraction fields.

    ``fractions`` and ``gradient`` expose owned flat Warp arrays for explicit
    inspection. Do not mutate them during evaluation, solve or callbacks. Use
    ``set_fractions`` between solves for independently prepared initial fields.
    ``solve`` never accesses any external reference or evaluation data.

    An optional ``material_metric=(h00,h01,h11)`` supplies a fixed symmetric
    positive-definite two-material update metric, with condition at most 1e6.
    It changes the update and its matching metric projection, never the
    objective, canonical Euclidean derivative or stopping diagnostic. It is an
    algorithmic metric; no full voxel-Hessian or uncertainty claim is implied.

    ``field_domain="nonnegative"`` treats the dimensionless arrays as equivalent
    basis coefficients with no artificial sum cap. Any density normalisation
    must be declared in the supplied spectral specification. The legacy names
    ``initial_fractions``, ``fractions`` and ``fractions_numpy`` retain their API
    meaning as the actual stored field values in either explicit domain.
    ``observation_model="signal_wls"`` requires signal views; objective weights
    are fixed numerical scale factors, not an inferred statistical noise law.
    """

    def __init__(
        self,
        *,
        grid: GridSpec,
        views: tuple[MaterialReconstructionView | MaterialReconstructionSignalView, ...],
        coefficients: Any,
        spectral_spec: SpectralSpec,
        initial_fractions: Any,
        settings: MaterialReconstructionSettings,
        samples_per_ray: int,
        device: str = "cuda:0",
        material_metric: tuple[float, float, float] | None = None,
        voxel_metric_scale: Any = None,
        field_domain: Literal["fractions", "nonnegative"] = "fractions",
        observation_model: Literal["poisson", "signal_wls"] = "poisson",
    ) -> None:
        if field_domain not in ("fractions", "nonnegative"):
            raise ContractError("field_domain must be fractions or nonnegative")
        if observation_model not in ("poisson", "signal_wls"):
            raise ContractError("observation_model must be poisson or signal_wls")
        self.field_domain, self.observation_model = field_domain, observation_model
        signal_wls = observation_model == "signal_wls"
        if not isinstance(grid, GridSpec) or not isinstance(
            settings, MaterialReconstructionSettings
        ):
            raise ContractError("supply GridSpec and MaterialReconstructionSettings")
        if not isinstance(spectral_spec, SpectralSpec):
            raise ContractError("supply the spectral specification with input provenance")
        integer(spectral_spec.materials, "materials", minimum=1, maximum=8)
        if (
            not spectral_spec.active_paths
            or (not signal_wls and not spectral_spec.shared_weights)
            or not spectral_spec.shared_response
            or spectral_spec.active_weights
            or spectral_spec.active_response
            or spectral_spec.active_coefficients
            or (not signal_wls and spectral_spec.output_unit != "counts")
        ):
            raise ContractError(
                "fixed spectra/coefficients and shared response required; "
                "Poisson needs shared count spectra"
            )
        if not isinstance(views, tuple) or not views:
            raise ContractError("supply a nonempty tuple of fitting views")
        samples_per_ray = integer(samples_per_ray, "samples_per_ray", minimum=1)
        self.grid, self.settings, self.spec = grid, settings, spectral_spec
        self.precision = spectral_spec.precision
        self.materials, self.energies = spectral_spec.materials, spectral_spec.energies
        self.material_metric, self._metric_inverse, self._metric_condition = _prepare_metric(
            material_metric, self.materials
        )
        self.np: Any = importlib.import_module("numpy")
        self._shape = (self.materials, *grid.shape)
        self.size = integer(self.materials * grid.voxels, "fraction field size", minimum=1)
        initial = self._fraction_input(initial_fractions)
        coefficients = self._array(coefficients, "coefficients", (self.materials, self.energies))
        if (coefficients < 0).any():
            raise ContractError("coefficients must be nonnegative linear attenuation in mm^-1")
        prepared: list[tuple[Any, Any, Any, Any, Any, Any]] = []
        effective_observations = 0
        expected_view = (
            MaterialReconstructionSignalView if signal_wls else MaterialReconstructionView
        )
        for view in views:
            if not isinstance(view, expected_view):
                raise ContractError("view type must match the declared observation model")
            if not isinstance(view.geometry, DetectorGeometry) or not isinstance(
                view.pose, RigidTransform
            ):
                raise ContractError("view geometry and pose must use canonical contracts")
            source = (
                view.observations
                if isinstance(view, MaterialReconstructionSignalView)
                else view.counts
            )
            if not hasattr(source, "shape") or len(source.shape) != 3:
                raise ContractError("observations must have shape (channels,height,width)")
            channels = integer(int(source.shape[0]), "channels", minimum=1)
            shape = (channels, *view.geometry.shape)
            counts = self._array(source, "observations", shape, finite=not signal_wls)
            weight_shape = (channels, self.energies)
            if not spectral_spec.shared_weights:
                weight_shape = (*weight_shape, *view.geometry.shape)
            weights = self._array(view.weights, "weights", weight_shape)
            response = self._array(view.response, "response", (channels, self.energies))
            objective_weights, valid = None, None
            if isinstance(view, MaterialReconstructionSignalView):
                objective_weights = self._array(view.objective_weights, "objective_weights", shape)
                if (objective_weights < 0).any():
                    raise ContractError("objective weights must be nonnegative")
                valid = view.valid
                if (
                    not isinstance(valid, self.np.ndarray)
                    or valid.dtype != self.np.dtype("uint8")
                    or tuple(valid.shape) != shape
                    or not valid.flags.c_contiguous
                    or (valid > 1).any()
                ):
                    raise ContractError(
                        "valid must be a contiguous uint8 0/1 array matching observations"
                    )
                if not self.np.isfinite(counts[valid == 1]).all():
                    raise ContractError("unmasked observations must be finite")
                effective_observations += int(((valid == 1) & (objective_weights > 0)).sum())
            elif (counts < 0).any() or (counts > 2**24).any() or (counts % 1 != 0).any():
                raise ContractError("counts must be integers in [0,2^24]")
            if (weights < 0).any() or (response < 0).any() or (response > 1).any():
                raise ContractError(
                    "weights must be nonnegative and response probabilities in [0,1]"
                )
            product = weights.astype(self.np.float64) * (
                response if spectral_spec.shared_weights else response[:, :, None, None]
            )
            open_beam = self.np.sum(product, axis=1)
            required = (
                open_beam
                if valid is None or spectral_spec.shared_weights
                else open_beam[valid == 1]
            )
            if (required <= 0).any():
                raise ContractError("each observed channel/ray must have a positive open-beam mean")
            prepared.append((view, counts, weights, response, objective_weights, valid))
        if signal_wls and effective_observations == 0:
            raise ContractError(
                "a signal fit requires at least one valid, positive-weight observation"
            )
        self.context = prepare_context(device=device)
        self.wp = self.context.wp
        self.stream = self.context.stream
        self.device = self.context.device
        require_no_tape()
        self.kernels = load_kernels("dpt.kernels.material_reconstruction")
        self._project_kernel = self.kernels.projected_step(
            self.materials, field_domain == "nonnegative"
        )
        self._metric_arguments: tuple[Any, ...] = ()
        if self.material_metric is not None:
            assert self._metric_inverse is not None
            self._metric_arguments = (
                self.wp.vec3d(*self.material_metric),
                self.wp.vec3d(*self._metric_inverse),
            )
        maximum_pixels = max(view.geometry.pixels for view in views)
        channels_total = sum(int(counts.shape[0]) for _, counts, _, _, _, _ in prepared)
        beta = settings.regularisation_mm_inverse
        self._penalty_coefficients = tuple(
            beta * math.prod(grid.spacing_mm) / h**2 for h in grid.spacing_mm
        )
        if any(not math.isfinite(value) for value in self._penalty_coefficients):
            raise ContractError("regularisation or grid spacing overflows its physical scale")
        self.views: list[dict[str, Any]] = []
        with self.context.scope():
            wp = self.wp
            value_dtype = wp.float64 if self.precision == "float64" else wp.float32
            self.fractions = wp.array(initial.reshape(-1), dtype=wp.float32, device=self.device)
            self.trial = wp.empty_like(self.fractions)
            self.gradient = wp.empty_like(self.fractions)
            self._components = wp.empty(self.size, dtype=wp.float64, device=self.device)
            self._mappings = wp.empty(grid.voxels, dtype=wp.float64, device=self.device)
            self._status = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._summary = wp.zeros(3, dtype=wp.float64, device=self.device)
            self._data_loss, self._prior_loss, self._total_loss = (
                self._summary[i : i + 1] for i in range(3)
            )
            self._slope = wp.zeros(1, dtype=wp.float64, device=self.device)
            self._losses = wp.empty(channels_total, dtype=wp.float64, device=self.device)
            self._coefficients = wp.array(
                coefficients.reshape(-1), dtype=wp.float32, device=self.device
            )
            self._reduction = prepare_reduction(self.context, max(self.size, channels_total))
            self._max_buffers: list[Any] = []
            count = grid.voxels
            while count > 1:
                count = (count + 255) // 256
                self._max_buffers.append(wp.empty(count, dtype=wp.float64, device=self.device))
            self._paths = wp.empty(
                self.materials * maximum_pixels, dtype=value_dtype, device=self.device
            )
            self._path_total = wp.empty(
                self.materials * maximum_pixels, dtype=wp.float64, device=self.device
            )
            self._path_seed = (
                self._path_total if self.precision == "float64" else wp.empty_like(self._paths)
            )
            self._channel_seed = wp.empty_like(self._paths)
            self._image_seed = wp.empty(maximum_pixels, dtype=value_dtype, device=self.device)
            self._add_paths = self.kernels.add_paths(self.precision)
            self._spectral = prepare_spectral(
                spectral_spec, max_pixels=maximum_pixels, device=device, stream=self.stream
            )
            self._objective = prepare_objective(
                ObjectiveSpec(
                    kind="squared_error",
                    domain="signal",
                    weighted=True,
                    masked=True,
                    precision=self.precision,
                )
                if signal_wls
                else ObjectiveSpec(kind="poisson", domain="counts", precision=self.precision),
                max_pixels=maximum_pixels,
                device=device,
                stream=self.stream,
            )
            loss_index = 0
            fixed_uploads: dict[tuple[int, int], Any] = {}
            for view, counts, weights, response, objective_weights, valid in prepared:
                pixels = view.geometry.pixels
                paths = self._paths[: self.materials * pixels]
                path_seed = self._path_seed[: self.materials * pixels]
                item: dict[str, Any] = {
                    "pose": wp.array(view.pose.packed(), dtype=wp.float64, device=self.device),
                    "pixels": pixels,
                    "shape": tuple(counts.shape),
                    "paths": paths,
                    "path_seed": path_seed,
                    "path_total": self._path_total[: self.materials * pixels],
                    "channel_seed": self._channel_seed[: self.materials * pixels],
                    "image_seed": self._image_seed[:pixels],
                    "channels": [],
                }
                for name, fields in (("accepted", self.fractions), ("trial", self.trial)):
                    item[name] = prepare_material_projection(
                        grid,
                        view.geometry,
                        ProjectionSpec(
                            samples_per_ray,
                            active_pose=False,
                            active_volume=True,
                            precision=self.precision,
                        ),
                        materials=self.materials,
                        fields=fields,
                        field_domain=field_domain,
                        out_paths=paths,
                        adj_paths=path_seed,
                        out_fields=self.gradient,
                        device=device,
                        stream=self.stream,
                    )
                for channel in range(counts.shape[0]):
                    row = {
                        "counts": wp.array(
                            counts[channel].reshape(-1), dtype=wp.float32, device=self.device
                        ),
                        "weights": self._fixed_upload(weights, channel, fixed_uploads),
                        "response": self._fixed_upload(response, channel, fixed_uploads),
                        "mean": wp.empty(pixels, dtype=value_dtype, device=self.device),
                        "loss": self._losses[loss_index : loss_index + 1],
                    }
                    if signal_wls:
                        row["objective_weights"] = wp.array(
                            objective_weights[channel].reshape(-1),
                            dtype=wp.float32,
                            device=self.device,
                        )
                        row["valid"] = wp.array(
                            valid[channel].reshape(-1), dtype=wp.uint8, device=self.device
                        )
                        self._objective.validate_observation(
                            row["counts"],
                            weights=row["objective_weights"],
                            valid=row["valid"],
                            stream=self.stream,
                        )
                    self._spectral.validate_inputs(
                        self._coefficients,
                        row["weights"],
                        row["response"],
                        pixels=pixels,
                        probability_response=True,
                        stream=self.stream,
                    )
                    item["channels"].append(row)
                    loss_index += 1
                self.views.append(item)
        self._last_loss: dict[str, float] = {}
        self.evaluations = 0
        self._prediction_state = "uninitialised"
        # Optional secant storage is allocated once, outside the solve loop.
        self._secant_fields: Any = None
        self._secant_gradient: Any = None
        self._secant_products: Any = None
        self._secant_totals: Any = None
        self._secant_numerator: Any = None
        self._secant_denominator: Any = None
        self._secant_metric: Any = None
        self._secant_host: Any = None
        self._secant_host_status: Any = None
        self._secant_host_view: Any = None
        self._secant_host_status_view: Any = None
        self._inertial_previous: Any = None
        self.voxel_metric_scale: Any = None
        self._voxel_metric_summary: dict[str, float] | None = None
        if voxel_metric_scale is not None:
            self.set_voxel_metric_scale(voxel_metric_scale)

    def set_voxel_metric_scale(self, values: Any) -> None:
        """Copy a fixed positive voxel scale D for update blocks D_i H.

        Supply a contiguous native FP32 host grid or a flat FP32 array on this
        workspace's CUDA device. Preparation checks values on device and reads
        only scalar bounds. Use this between solves; it never changes the
        objective, gradient or Euclidean stationarity diagnostic.
        """
        require_no_tape()
        if isinstance(values, self.np.ndarray):
            values = self._array(values, "voxel metric scale", self.grid.shape)
            if (values <= 0).any():
                raise ContractError("voxel metric scale must be strictly positive")
            with self.context.scope():
                candidate = self.wp.array(
                    values.reshape(-1), dtype=self.wp.float32, device=self.device
                )
        else:
            if (
                not isinstance(values, self.wp.array)
                or values.dtype != self.wp.float32
                or values.device != self.device
                or values.shape != (self.grid.voxels,)
                or not values.is_contiguous
            ):
                raise ContractError(
                    "supply a flat contiguous FP32 voxel scale on the solver device"
                )
            with self.context.scope():
                candidate = self.wp.empty_like(values)
                self.wp.copy(candidate, values, stream=self.stream)
        with self.context.scope():
            self._status.zero_()
            self.wp.launch(
                self.kernels.check_voxel_metric_scale,
                dim=self.grid.voxels,
                inputs=[candidate, self._components, self._mappings, self._status],
                stream=self.stream,
                record_tape=False,
            )
            try:
                self._check()
            except NumericalError as error:
                raise ContractError(
                    "voxel metric scale must be finite and strictly positive"
                ) from error
            bounds: list[float] = []
            for source in (self._components, self._mappings):
                previous, count = source, self.grid.voxels
                for following in self._max_buffers:
                    self.wp.launch_tiled(
                        self.kernels.max_partials,
                        dim=(count + 255) // 256,
                        inputs=[previous, count, following],
                        block_dim=256,
                        stream=self.stream,
                        record_tape=False,
                    )
                    previous, count = following, (count + 255) // 256
                self.wp.synchronize_stream(self.stream)
                bounds.append(float(previous.numpy()[0]))
        maximum, minimum = bounds[0], -bounds[1]
        if maximum / minimum > 1e6 * (1 + 8 * 2**-24):
            raise ContractError("voxel metric scale range must not exceed 1e6")
        self.voxel_metric_scale = candidate
        self._voxel_metric_summary = {
            "minimum": minimum,
            "maximum": maximum,
            "maximum_over_minimum": maximum / minimum,
        }

    def _fixed_upload(self, source: Any, channel: int, cache: dict[tuple[int, int], Any]) -> Any:
        key = (id(source), channel)
        if key not in cache:
            cache[key] = self.wp.array(
                source[channel].reshape(-1), dtype=self.wp.float32, device=self.device
            )
        return cache[key]

    def _array(self, value: Any, name: str, shape: tuple[int, ...], *, finite: bool = True) -> Any:
        if not isinstance(value, self.np.ndarray) or value.dtype != self.np.dtype("float32"):
            raise ContractError(f"{name} must be a native FP32 numpy array; no implicit casts")
        if tuple(value.shape) != shape or not value.flags.c_contiguous:
            raise ContractError(f"{name} must be contiguous with shape {shape}")
        if finite and not self.np.isfinite(value).all():
            raise ContractError(f"{name} must contain finite values")
        return value

    def _fraction_input(self, value: Any) -> Any:
        value = self._array(value, "fractions", self._shape)
        if (value < 0).any() or (
            self.field_domain == "fractions"
            and (self.np.sum(value, axis=0, dtype=self.np.float64) > 1 + 2**-24).any()
        ):
            raise ContractError("fields must satisfy the declared nonnegative or fraction domain")
        return value

    def set_fractions(self, values: Any) -> None:
        """Upload a new supplied initial state; no reference inputs are inferred."""
        values = self._fraction_input(values)
        with self.context.scope():
            self.fractions.assign(values.reshape(-1))
        self._prediction_state = "uninitialised"

    def _check(self) -> None:
        self.wp.synchronize_stream(self.stream)
        if int(self._status.numpy()[0]):
            raise NumericalError("material reconstruction arithmetic is not representable")

    def evaluate(self, *, gradient: bool = True, trial: bool = False) -> float:
        """Evaluate all fitting channels; trial evaluation preserves accepted gradient.

        All immutable inputs were validated/uploaded during preparation. Variable
        fractions are validated once; unchecked canonical launches retain their
        error flags until this evaluation's explicit completion checks.
        """
        if trial and gradient:
            raise ContractError("trial evaluations must preserve the accepted gradient")
        require_no_tape()
        selected = "trial" if trial else "accepted"
        fields = self.trial if trial else self.fractions
        self._prediction_state = "uninitialised"
        with self.context.scope():
            self._status.zero_()
            self._spectral.clear_status()
            self._objective.clear_status()
            self.views[0][selected].validate_inputs(stream=self.stream)
            if gradient:
                self.gradient.zero_()
            for view in self.views:
                projection = view[selected]
                projection.clear_status()
                project_material_paths(
                    view["pose"], workspace=projection, stream=self.stream, validate=False
                )
                if gradient:
                    view["path_total"].zero_()
                for channel in view["channels"]:
                    spectral_signal(
                        view["paths"],
                        self._coefficients,
                        channel["weights"],
                        channel["response"],
                        out_mean=channel["mean"],
                        workspace=self._spectral,
                        stream=self.stream,
                        validate=False,
                    )
                    evaluate_objective(
                        channel["mean"],
                        channel["counts"],
                        out_loss=channel["loss"],
                        out_seed=view["image_seed"] if gradient else None,
                        weights=channel.get("objective_weights"),
                        valid=channel.get("valid"),
                        workspace=self._objective,
                        stream=self.stream,
                        validate=False,
                    )
                    if gradient:
                        spectral_vjp(
                            view["paths"],
                            self._coefficients,
                            channel["weights"],
                            channel["response"],
                            seed=view["image_seed"],
                            out_grad_paths=view["channel_seed"],
                            workspace=self._spectral,
                            stream=self.stream,
                            validate=False,
                        )
                        self.wp.launch(
                            self._add_paths,
                            dim=self.materials * view["pixels"],
                            inputs=[view["channel_seed"], view["path_total"], self._status],
                            stream=self.stream,
                            record_tape=False,
                        )
                if gradient:
                    if self.precision == "float32":
                        self.wp.launch(
                            self.kernels.narrow_paths,
                            dim=self.materials * view["pixels"],
                            inputs=[view["path_total"], view["path_seed"], self._status],
                            stream=self.stream,
                            record_tape=False,
                        )
                    material_projection_vjp(
                        view["pose"],
                        workspace=projection,
                        accumulate=True,
                        stream=self.stream,
                        validate=False,
                    )
            self._reduction.finish(
                int(self._losses.size), self._data_loss, self._status, source=self._losses
            )
            self.wp.launch(
                self.kernels.regularise,
                dim=self.size,
                inputs=[
                    fields,
                    self.grid.shape[2],
                    self.grid.shape[1],
                    self.grid.shape[0],
                    *self._penalty_coefficients,
                    gradient,
                    self.gradient,
                    self._components,
                    self._status,
                ],
                stream=self.stream,
                record_tape=False,
            )
            self._reduction.finish(
                self.size, self._prior_loss, self._status, source=self._components
            )
            self.wp.launch(
                self.kernels.scalar_sum,
                dim=1,
                inputs=[self._data_loss, self._prior_loss, self._total_loss],
                stream=self.stream,
                record_tape=False,
            )
            self._check()
            self._spectral.check_status()
            self._objective.check_status()
            for view in self.views:
                view[selected].check_status()
            summary = self._summary.numpy()
        loss = float(summary[2])
        if not math.isfinite(loss):
            raise NumericalError("material reconstruction objective is nonfinite")
        self._last_loss = {
            "data_objective": float(summary[0]),
            "regularisation_objective": float(summary[1]),
        }
        self.evaluations += 1
        self._prediction_state = selected
        return loss

    def projected_trial(self, step: float, *, use_metric: bool = True) -> tuple[float, float]:
        """Prepare the constrained step and return its mapping norm/actual slope.

        With a metric, the trial minimises g^T(u-f)+(u-f)^T H(u-f)/(2*step)
        over the declared fraction simplex or nonnegative orthant.
        ``use_metric=False`` selects the ordinary Euclidean projection; solve
        always uses that mode for stationarity.
        """
        return self._projected_trial(step, use_metric=use_metric)

    def _projected_trial(
        self, step: float, *, use_metric: bool = True, momentum: float = 0.0
    ) -> tuple[float, float]:
        step = finite_scalar(step, "step", minimum=0)
        if step <= 0:
            raise ContractError("step must be positive")
        if type(use_metric) is not bool:
            raise ContractError("use_metric must be a boolean")
        with self.context.scope():
            self._status.zero_()
            selected_kernel = self._project_kernel
            metric_arguments: tuple[Any, ...] = ()
            if use_metric and self.material_metric is not None:
                selected_kernel = (
                    self.kernels.projected_metric_step
                    if self.field_domain == "fractions"
                    else self.kernels.projected_metric_nonnegative_step
                )
                metric_arguments = self._metric_arguments
            self.wp.launch(
                selected_kernel,
                dim=self.grid.voxels,
                inputs=[
                    self.fractions,
                    self.gradient,
                    self.grid.voxels,
                    step,
                    self._inertial_previous if momentum else self.fractions,
                    momentum,
                    use_metric and self.voxel_metric_scale is not None,
                    self.voxel_metric_scale
                    if self.voxel_metric_scale is not None
                    else self.fractions,
                    *metric_arguments,
                    self.trial,
                    self._components,
                    self._mappings,
                    self._status,
                ],
                stream=self.stream,
                record_tape=False,
            )
            self._reduction.finish(self.size, self._slope, self._status, source=self._components)
            previous, count = self._mappings, self.grid.voxels
            for following in self._max_buffers:
                self.wp.launch_tiled(
                    self.kernels.max_partials,
                    dim=(count + 255) // 256,
                    inputs=[previous, count, following],
                    block_dim=256,
                    stream=self.stream,
                    record_tape=False,
                )
                previous, count = following, (count + 255) // 256
            self._check()
            mapping, slope = float(previous.numpy()[0]), float(self._slope.numpy()[0])
        if not math.isfinite(mapping) or not math.isfinite(slope):
            raise NumericalError("projected update or slope is nonfinite")
        return mapping, slope

    def solve(
        self, callback: Callable[[int, dict[str, Any]], None] | None = None
    ) -> dict[str, Any]:
        """Projected Armijo solve; callbacks run only after accepted updates.

        Value evaluations of rejected trials cannot overwrite the accepted
        gradient or fractions. Numerical failure rejects that trial; failures
        in an accepted-state gradient evaluation propagate to the caller.
        Every search failure and rejected numerical evaluation is recorded.
        """
        started = time.perf_counter()
        policy = self.settings
        history: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        reason = "iteration_budget"
        step = policy.initial_step
        initial_mapping = 0.0
        tolerance = policy.gradient_mapping_tolerance
        accepted = 0
        use_secant = policy.step_selection == "bb"
        if use_secant:
            self._prepare_secant()
        use_inertia = policy.acceleration == "inertial"
        inertial_t = 1.0
        if use_inertia:
            self._prepare_inertia()
        for iteration in range(policy.iterations):
            loss = self.evaluate(gradient=True)
            base_losses = dict(self._last_loss)
            mapping, _ = self.projected_trial(policy.mapping_step, use_metric=False)
            if iteration == 0:
                initial_mapping = mapping
                tolerance += policy.relative_gradient_mapping_tolerance * initial_mapping
            if mapping <= tolerance:
                reason = "projected_gradient_tolerance"
                break
            secant_step = None
            if use_secant:
                if iteration > 0:
                    secant_step = self._secant_step(policy)
                    if secant_step is not None:
                        step = secant_step
                self._remember_secant()
            proposed_step = step
            next_inertial_t = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * inertial_t**2))
            momentum = (inertial_t - 1.0) / next_inertial_t if use_inertia else 0.0
            inertial_restart = None
            found = False
            backtracks = 0
            while backtracks < policy.maximum_backtracks:
                if step == 0:
                    break
                trial_loss = math.inf
                trial_failure = "armijo"
                try:
                    _, slope = self._projected_trial(step, momentum=momentum)
                    if slope >= 0:
                        trial_failure = "non_descent"
                    else:
                        trial_loss = self.evaluate(gradient=False, trial=True)
                except NumericalError as error:
                    slope = 0.0
                    trial_failure = "nonfinite"
                    rejected.append(
                        {
                            "iteration": iteration + 1,
                            "step": step,
                            "backtracks": backtracks,
                            "error": str(error),
                            "momentum": momentum,
                        }
                    )
                if slope < 0 and trial_loss < loss and trial_loss <= loss + policy.armijo * slope:
                    with self.context.scope():
                        if use_inertia:
                            self.wp.copy(
                                self._inertial_previous, self.fractions, stream=self.stream
                            )
                        self.wp.copy(self.fractions, self.trial, stream=self.stream)
                    if use_inertia:
                        inertial_t = 1.0 if inertial_restart is not None else next_inertial_t
                    self._prediction_state = "accepted"
                    record: dict[str, Any] = {
                        "iteration": iteration + 1,
                        "accepted": True,
                        "objective_before": loss,
                        "objective": trial_loss,
                        **self._last_loss,
                        "gradient_mapping_before": mapping,
                        "step": step,
                        "backtracks": backtracks,
                        "proposed_step": proposed_step,
                        "secant_step": secant_step,
                        "momentum": momentum,
                        "inertial_restart": inertial_restart,
                        "wall_seconds": time.perf_counter() - started,
                    }
                    history.append(record)
                    accepted += 1
                    # Callback failures belong to the recorder/caller; this
                    # update has already been accepted and cannot be retried.
                    if callback is not None:
                        callback(iteration + 1, dict(record))
                    step = min(step / policy.backtracking_factor, policy.maximum_step)
                    found = True
                    break
                if momentum:
                    # Discard only the proposal. The old accepted field and its
                    # gradient remain intact; retry the same alpha without inertia.
                    inertial_restart = trial_failure
                    momentum = 0.0
                    continue
                step *= policy.backtracking_factor
                backtracks += 1
            if not found:
                reason = "line_search_failed"
                history.append(
                    {
                        "iteration": iteration + 1,
                        "accepted": False,
                        "objective": loss,
                        **base_losses,
                        "gradient_mapping_before": mapping,
                        "step": step,
                        "wall_seconds": time.perf_counter() - started,
                        "failure": reason,
                    }
                )
                break
        final_loss = self.evaluate(gradient=True)
        final_mapping, _ = self.projected_trial(policy.mapping_step, use_metric=False)
        if final_mapping <= tolerance:
            reason = "projected_gradient_tolerance"
        return {
            "termination": reason,
            "field_domain": self.field_domain,
            "observation_model": self.observation_model,
            "accepted_steps": accepted,
            "evaluations": self.evaluations,
            "final_objective": final_loss,
            **self._last_loss,
            "initial_gradient_mapping": initial_mapping,
            "final_gradient_mapping": final_mapping,
            "mapping_step": policy.mapping_step,
            "gradient_mapping_metric": "euclidean",
            "material_metric": self.material_metric,
            "material_metric_condition": self._metric_condition,
            "voxel_metric_scale": self._voxel_metric_summary,
            "gradient_mapping_threshold": tolerance,
            "step_selection": policy.step_selection,
            "acceleration": policy.acceleration,
            "wall_seconds": time.perf_counter() - started,
            "history": history,
            "rejected_numerical_trials": rejected,
        }

    def _prepare_inertia(self) -> None:
        """Allocate and initialise one preceding accepted field outside the loop."""
        with self.context.scope():
            if self._inertial_previous is None:
                self._inertial_previous = self.wp.empty_like(self.fractions)
            self.wp.copy(self._inertial_previous, self.fractions, stream=self.stream)

    def _prepare_secant(self) -> None:
        """Keep two preceding FP32 vectors and secant reduction scratch resident."""
        if self._secant_fields is not None:
            return
        with self.context.scope():
            wp = self.wp
            self._secant_fields = wp.empty_like(self.fractions)
            self._secant_gradient = wp.empty_like(self.gradient)
            self._secant_products = wp.empty(self.size, dtype=wp.float64, device=self.device)
            self._secant_totals = wp.empty(2, dtype=wp.float64, device=self.device)
            self._secant_numerator = self._secant_totals[0:1]
            self._secant_denominator = self._secant_totals[1:2]
            self._secant_metric = wp.vec3d(*(self.material_metric or (1.0, 0.0, 1.0)))
            self._secant_host = wp.empty(2, dtype=wp.float64, device="cpu", pinned=True)
            self._secant_host_status = wp.empty(1, dtype=wp.int32, device="cpu", pinned=True)
            self._secant_host_view = self._secant_host.numpy()
            self._secant_host_status_view = self._secant_host_status.numpy()

    def _remember_secant(self) -> None:
        with self.context.scope():
            self.wp.copy(self._secant_fields, self.fractions, stream=self.stream)
            self.wp.copy(self._secant_gradient, self.gradient, stream=self.stream)

    def _secant_step(self, policy: MaterialReconstructionSettings) -> float | None:
        """A secant affects the proposal only, never acceptance or stationarity."""
        with self.context.scope():
            self._status.zero_()
            self.wp.launch(
                self.kernels.secant_products,
                dim=self.size,
                inputs=[
                    self.fractions,
                    self.gradient,
                    self._secant_fields,
                    self._secant_gradient,
                    self.grid.voxels,
                    self.material_metric is not None,
                    self._secant_metric,
                    self.voxel_metric_scale is not None,
                    self.voxel_metric_scale
                    if self.voxel_metric_scale is not None
                    else self.fractions,
                    self._components,
                    self._secant_products,
                ],
                stream=self.stream,
                record_tape=False,
            )
            self._reduction.finish(
                self.size, self._secant_numerator, self._status, source=self._components
            )
            self._reduction.finish(
                self.size, self._secant_denominator, self._status, source=self._secant_products
            )
            self.wp.copy(self._secant_host, self._secant_totals, stream=self.stream)
            self.wp.copy(self._secant_host_status, self._status, stream=self.stream)
            self.wp.synchronize_stream(self.stream)
            numerator = float(self._secant_host_view[0])
            denominator = float(self._secant_host_view[1])
            # Invalid secant arithmetic is an unusable proposal, not a change
            # to the already checked objective/gradient or a convergence claim.
            invalid = int(self._secant_host_status_view[0]) != 0
        if invalid or not all(math.isfinite(v) and v > 0 for v in (numerator, denominator)):
            return None
        step = numerator / denominator
        if not math.isfinite(step) or step <= 0:
            return None
        return min(policy.maximum_step, max(policy.secant_minimum_step, step))

    def fractions_numpy(self) -> Any:
        """Explicit synchronised export of accepted material-major fractions."""
        self.wp.synchronize_stream(self.stream)
        return self.fractions.numpy().reshape(self._shape)

    def predictions_numpy(self) -> tuple[Any, ...]:
        """Export fitting predictions only after evaluating/accepting that state."""
        if self._prediction_state != "accepted":
            raise ContractError("evaluate the accepted state before exporting predictions")
        self.wp.synchronize_stream(self.stream)
        return tuple(
            self.np.stack([channel["mean"].numpy() for channel in view["channels"]]).reshape(
                view["shape"]
            )
            for view in self.views
        )


def _prepare_metric(
    metric: tuple[float, float, float] | None, materials: int
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None, float | None]:
    """Validate the small fixed metric without importing optional dependencies."""
    if metric is None:
        return None, None, None
    if materials != 2:
        raise ContractError("a coupled material_metric requires exactly two materials")
    if not isinstance(metric, tuple) or len(metric) != 3:
        raise ContractError("material_metric must be the symmetric tuple (h00,h01,h11)")
    a, b, d = (finite_scalar(value, "material_metric entry") for value in metric)
    scale = max(abs(a), abs(b), abs(d))
    if scale == 0 or a <= 0 or d <= 0:
        raise ContractError("material_metric must be positive definite")
    a_scaled, b_scaled, d_scaled = a / scale, b / scale, d / scale
    determinant = a_scaled * d_scaled - b_scaled * b_scaled
    maximum = 0.5 * (a_scaled + d_scaled + math.hypot(a_scaled - d_scaled, 2 * b_scaled))
    if determinant <= 0 or maximum <= 0:
        raise ContractError("material_metric must be positive definite")
    minimum = determinant / maximum
    condition = maximum / minimum
    if not math.isfinite(condition) or condition > 1.0e6:
        raise ContractError("material_metric condition must not exceed 1e6")
    inverse = (
        d_scaled / determinant / scale,
        -b_scaled / determinant / scale,
        a_scaled / determinant / scale,
    )
    if not all(math.isfinite(value) for value in inverse):
        raise ContractError("material_metric inverse must be representable in FP64")
    return (a, b, d), inverse, condition
