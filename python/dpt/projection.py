"""Finite-segment optical-depth projection with caller-owned device buffers.

Coordinates and quadrature accumulators are FP64; field samples remain FP32.
Depth and depth-cotangent storage is explicitly FP32 (default) or FP64.
Numerical acceptance and performance must be established on CUDA;
this interface does not equate a finite result with an accurate projection.
"""

from __future__ import annotations

# Mathematical output names follow the optical-depth contract.
# ruff: noqa: N803
# Module-owned workspace scratch stays private to this implementation.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from dpt._reductions import ReductionTree, prepare_reduction
from dpt._runtime import DeviceContext, ensure_tape, load_kernels, prepare_context, require_no_tape
from dpt.contracts import ContractError, NumericalError, integer
from dpt.geometry import DetectorGeometry
from dpt.volumes import GridSpec

_BLOCK = 128


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    """Declared ray integration and explicitly supported active inputs.

    Midpoint uses samples_per_ray. Cell Gauss ignores this count and splits at
    every interpolation plane, integrating each cubic segment with two nodes.

    Grid/acquisition calibration is fixed. Pose derivatives include active
    intersection bounds. At ties, grazing rays and interpolation knots the
    returned branch derivative does not assert differentiability.
    """

    samples_per_ray: int
    active_pose: bool = True
    active_volume: bool = False
    precision: Literal["float32", "float64"] = "float32"
    integration: Literal["midpoint", "cell_gauss"] = "midpoint"

    def __post_init__(self) -> None:
        integer(self.samples_per_ray, "samples_per_ray", minimum=1)
        if self.integration not in ("midpoint", "cell_gauss"):
            raise ContractError("projection integration must be midpoint or cell_gauss")
        if self.precision not in ("float32", "float64"):
            raise ContractError("projection precision must be float32 or float64")
        if type(self.active_pose) is not bool or type(self.active_volume) is not bool:
            raise ContractError("active input flags must be booleans")


@dataclass(slots=True)
class ProjectionWorkspace:
    """Persistent O(12 P/128) FP64 reduction storage, one owning CUDA stream.

    Original volume and pose buffers must remain immutable through backward.
    One recorded forward may be outstanding per workspace. Use another
    workspace for concurrent/re-entrant evaluations. No sample history is kept.
    """

    grid: GridSpec
    geometry: DetectorGeometry
    spec: ProjectionSpec
    context: DeviceContext
    _kernels: Any = field(repr=False)
    _geometry_kernels: Any = field(repr=False)
    _configuration: Any = field(repr=False)
    _status: Any = field(repr=False)
    _empty: Any = field(repr=False)
    _reduction: ReductionTree | None = field(repr=False)
    _empty_partials: Any = field(repr=False)
    _recorded_tape: Any = field(default=None, repr=False)

    @property
    def dtype(self) -> Any:
        """Declared depth and depth-cotangent storage; field storage stays FP32."""
        return getattr(self.context.wp, self.spec.precision)

    @property
    def device(self) -> Any:
        return self.context.device

    @property
    def stream(self) -> Any:
        return self.context.stream

    @property
    def scratch_bytes(self) -> int:
        return 4 + (self._reduction.scratch_bytes if self._reduction is not None else 0)

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()

    def check_status(self) -> None:
        self.context.wp.synchronize_stream(self.stream)
        code = int(self._status.numpy()[0])
        if code & 1:
            raise ContractError("projection input violates finite non-negative field or rigid pose")
        if code:
            raise NumericalError("non-finite projection or gradient; discard these outputs")

    def discard_recording(self, tape: Any) -> None:
        """Release a forward abandoned by tape.reset(), after discarding its graph."""
        if self._recorded_tape is not None and self._recorded_tape is not tape:
            raise ContractError("this workspace belongs to another tape")
        if tape.launches:
            raise ContractError("reset the abandoned tape before releasing its input lifetime")
        self._recorded_tape = None


def prepare_projection(
    grid: GridSpec,
    geometry: DetectorGeometry,
    spec: ProjectionSpec,
    *,
    device: str = "cuda:0",
    stream: Any = None,
) -> ProjectionWorkspace:
    """Prepare fixed metadata and device scratch; never allocate image outputs."""
    context = prepare_context(device=device, stream=stream)
    wp = context.wp
    kernels = load_kernels("dpt.kernels.projection")
    geometry_kernels = load_kernels("dpt.kernels.geometry")
    config = kernels.Configuration()
    config.shape = wp.vec3i(grid.shape[2], grid.shape[1], grid.shape[0])
    # diag(1/h) G^T: spacing follows grid axes, not object/world coordinates.
    inverse_grid = tuple(
        grid.orientation[3 * j + i] / grid.spacing_mm[i] for i in range(3) for j in range(3)
    )
    if not all(math.isfinite(value) for value in inverse_grid):
        raise ContractError("grid spacing overflows the FP64 coordinate map")
    for row in (0, geometry.shape[0] - 1):
        for column in (0, geometry.shape[1] - 1):
            endpoint = geometry.pixel_centre(row, column)
            if not all(math.isfinite(value) for value in endpoint) or not math.isfinite(
                math.dist(endpoint, geometry.source_mm)
            ):
                raise ContractError("detector extent exceeds the FP64 finite-ray range")
    config.object_to_grid = wp.mat33d(*inverse_grid)
    config.grid_origin = wp.vec3d(*grid.origin_mm)
    config.source = wp.vec3d(*geometry.source_mm)
    config.detector_origin = wp.vec3d(*geometry.origin_mm)
    config.column_step = wp.vec3d(*(x * geometry.spacing_mm[0] for x in geometry.u))
    config.row_step = wp.vec3d(*(x * geometry.spacing_mm[1] for x in geometry.v))
    config.width = geometry.shape[1]
    config.pixels = geometry.pixels
    config.samples = spec.samples_per_ray
    reduction = None
    with context.scope():
        status = wp.zeros(1, dtype=wp.int32, device=context.device)
        empty = wp.empty(0, dtype=wp.float32, device=context.device)
        empty_partials = wp.empty(0, dtype=wp.float64, device=context.device)
        if spec.active_pose:
            reduction = prepare_reduction(context, geometry.pixels, tile=_BLOCK, components=12)
    return ProjectionWorkspace(
        grid,
        geometry,
        spec,
        context,
        kernels,
        geometry_kernels,
        config,
        status,
        empty,
        reduction,
        empty_partials,
    )


def _inputs(mu: Any, pose: Any, workspace: ProjectionWorkspace, stream: Any) -> None:
    ctx = workspace.context
    ctx.assert_stream(stream)
    ctx.array(mu, "mu", dtype=ctx.wp.float32, shape=(workspace.grid.voxels,))
    ctx.array(pose, "pose", dtype=ctx.wp.float64, shape=(12,))


def _validate(mu: Any, pose: Any, workspace: ProjectionWorkspace, seed: Any = None) -> None:
    wp = workspace.context.wp
    workspace.clear_status()
    wp.launch(
        workspace._kernels.validate_nonnegative,
        dim=workspace.grid.voxels,
        inputs=[mu],
        outputs=[workspace._status],
        stream=workspace.stream,
        record_tape=False,
    )
    wp.launch(
        workspace._geometry_kernels.validate_pose,
        dim=1,
        inputs=[pose],
        outputs=[workspace._status],
        stream=workspace.stream,
        record_tape=False,
    )
    if seed is not None:
        wp.launch(
            workspace._kernels.get_validate_finite(workspace.spec.precision == "float64"),
            dim=workspace.geometry.pixels,
            inputs=[seed],
            outputs=[workspace._status],
            stream=workspace.stream,
            record_tape=False,
        )
    workspace.check_status()


# region book:projection-public-operator
def project_optical_depth(
    mu: Any,
    pose: Any,
    *,
    workspace: ProjectionWorkspace,
    out_L: Any,
    stream: Any = None,
    tape: Any = None,
    validate: bool = True,
) -> None:
    """Overwrite dimensionless optical depth for every finite detector ray.

    mu is a flat non-negative FP32 field in inverse mm. pose is twelve FP64
    values (R_WO row-major, t_WO in mm). out_L is flat row-major detector
    storage in the precision declared by ProjectionSpec. A checked call
    synchronises domain diagnostics before writing and
    range diagnostics afterwards. Unchecked calls promise valid current inputs
    and require check_status() at the next acceptance checkpoint.
    """
    ensure_tape(tape)
    _inputs(mu, pose, workspace, stream)
    if workspace._recorded_tape is not None:
        raise ContractError("finish or discard the outstanding projection tape before reuse")
    ctx = workspace.context
    ctx.array(out_L, "out_L", dtype=workspace.dtype, shape=(workspace.geometry.pixels,))
    ctx.disjoint([("mu", mu), ("pose", pose)], [("out_L", out_L)])
    if tape is not None:
        _check_tape_arrays(mu, pose, out_L, workspace)
    if validate:
        _validate(mu, pose, workspace)
    ctx.wp.launch(
        workspace._kernels.get_forward(
            workspace.spec.precision == "float64", workspace.spec.integration == "cell_gauss"
        ),
        dim=workspace.geometry.pixels,
        inputs=[mu, pose, workspace._configuration],
        outputs=[out_L, workspace._status],
        stream=workspace.stream,
        block_dim=_BLOCK,
        record_tape=False,
    )
    if validate:
        workspace.check_status()
    if tape is not None:
        _record(tape, mu, pose, out_L, workspace)


# endregion book:projection-public-operator


def _check_tape_arrays(
    mu: Any, pose: Any, output: Any, workspace: ProjectionWorkspace
) -> list[Any]:
    arrays = [output]
    if workspace.spec.active_pose:
        arrays.append(pose)
    if workspace.spec.active_volume:
        arrays.append(mu)
    if len(arrays) == 1:
        raise ContractError("recording requires at least one active input")
    if any(array.grad is None for array in arrays):
        raise ContractError("active tape arrays require requires_grad=True at allocation")
    workspace.context.disjoint(
        [("mu", mu), ("pose", pose), ("output", output)],
        [("gradient", array.grad) for array in arrays],
    )
    return arrays


def _record(tape: Any, mu: Any, pose: Any, output: Any, workspace: ProjectionWorkspace) -> None:
    arrays = _check_tape_arrays(mu, pose, output, workspace)
    workspace._recorded_tape = tape
    if workspace.context.wp.config.verify_autograd_array_access:
        mu.mark_read()
        pose.mark_read()
        output.mark_write()
        # Register fixed inputs too, so Tape.reset() clears debug read markers.
        tape.record_launch(
            workspace._kernels.get_dependency_marker(workspace.spec.precision == "float64"),
            dim=0,
            max_blocks=0,
            inputs=[mu, pose],
            outputs=[output],
            device=workspace.device,
            block_dim=_BLOCK,
        )

    def backward() -> None:
        # Array storage has twelve coordinates. Its cotangent is NOT the
        # six-coordinate right tangent accepted by the standalone optimiser.
        projection_vjp(
            mu,
            pose,
            adj_L=output.grad,
            workspace=workspace,
            out_pose_matrix=pose.grad if workspace.spec.active_pose else None,
            out_mu=mu.grad if workspace.spec.active_volume else None,
            accumulate=True,
            validate=False,
            _from_tape=True,
        )
        with workspace.context.scope():
            if not output.retain_grad:
                output.grad.zero_()
        workspace._recorded_tape = None

    tape.record_func(backward, arrays)


def projection_vjp(
    mu: Any,
    pose: Any,
    *,
    adj_L: Any,
    workspace: ProjectionWorkspace,
    out_pose: Any = None,
    out_pose_matrix: Any = None,
    out_mu: Any = None,
    stream: Any = None,
    validate: bool = True,
    accumulate: bool = False,
    _from_tape: bool = False,
) -> None:
    """First-order discrete VJP, with explicit tangent or storage coordinates.

    out_pose is six FP64 values for a *local right* SE(3) increment in mm/rad.
    out_pose_matrix is twelve FP64 unconstrained storage partials, for a tape
    chain rule. They are mutually exclusive. Convert local right gradients
    using geometry.chart_gradient for a fixed-anchor optimisation chart.
    out_mu is FP32 and uses non-deterministic-order scatter atomics. No grid,
    acquisition, higher-order or non-smooth boundary derivative is promised.
    """
    require_no_tape()
    _inputs(mu, pose, workspace, stream)
    if workspace._recorded_tape is not None and not _from_tape:
        raise ContractError("the outstanding tape owns the projection inputs")
    if out_pose is not None and out_pose_matrix is not None:
        raise ContractError("request either a right-tangent or a storage pose gradient")
    target = out_pose if out_pose is not None else out_pose_matrix
    if target is None and out_mu is None:
        raise ContractError("request at least one projection gradient")
    if target is not None and not workspace.spec.active_pose:
        raise ContractError("pose is fixed in this workspace")
    if out_mu is not None and not workspace.spec.active_volume:
        raise ContractError("volume is fixed in this workspace")
    ctx, wp = workspace.context, workspace.context.wp
    ctx.array(adj_L, "adj_L", dtype=workspace.dtype, shape=(workspace.geometry.pixels,))
    components = 12 if out_pose_matrix is not None else 6
    writes: list[tuple[str, Any]] = []
    if target is not None:
        ctx.array(target, "pose gradient", dtype=wp.float64, shape=(components,))
        writes.append(("pose gradient", target))
    if out_mu is not None:
        ctx.array(out_mu, "out_mu", dtype=wp.float32, shape=(workspace.grid.voxels,))
        writes.append(("out_mu", out_mu))
    ctx.disjoint([("mu", mu), ("pose", pose), ("adj_L", adj_L)], writes)
    if validate:
        _validate(mu, pose, workspace, adj_L)
    if out_mu is not None and not accumulate:
        with ctx.scope():
            out_mu.zero_()
    count = (workspace.geometry.pixels + _BLOCK - 1) // _BLOCK
    wp.launch_tiled(
        workspace._kernels.get_vjp(
            out_mu is not None,
            components == 12,
            target is not None,
            double_seed=workspace.spec.precision == "float64",
            cell_gauss=workspace.spec.integration == "cell_gauss",
        ),
        dim=count,
        inputs=[mu, pose, workspace._configuration, adj_L],
        outputs=[
            out_mu if out_mu is not None else workspace._empty,
            workspace._reduction.partials[0]
            if workspace._reduction is not None
            else workspace._empty_partials,
            workspace._status,
        ],
        stream=workspace.stream,
        device=workspace.device,
        block_dim=_BLOCK,
        record_tape=False,
    )
    if target is not None:
        assert workspace._reduction is not None
        workspace._reduction.finish(
            count,
            target,
            workspace._status,
            components=components,
            accumulate=accumulate,
        )
    if out_mu is not None:
        # Atomic accumulation may overflow after individually finite increments.
        wp.launch(
            workspace._kernels.validate_finite,
            dim=workspace.grid.voxels,
            inputs=[out_mu],
            outputs=[workspace._status],
            stream=workspace.stream,
            record_tape=False,
        )
    if validate:
        workspace.check_status()


# region book:projection-pose-sensitivities
def projection_pose_sensitivities(
    mu: Any,
    pose: Any,
    depth_seeds: Any,
    *,
    out_jacobian: Any,
    workspace: ProjectionWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Export seeded per-ray local pose derivatives for bounded offline diagnostics.

    ``out_jacobian[6*p + axis]`` is ``depth_seeds[p] * dL[p]/d(delta[axis])``
    for ``pose @ exp(delta^)`` at zero, ordered tx, ty, tz, rx, ry, rz in mm/rad.
    All arrays are caller-owned, contiguous and on the workspace's CUDA device:
    seeds follow the declared depth precision, shape ``(pixels,)``; output is
    FP64 ``(6*pixels,)``. Ones export
    the optical-depth Jacobian; canonical transmission cotangents export the
    corresponding signal sensitivities. Missed rays and zero seeds give zero.

    The existing per-ray VJP calculation writes before reduction in O(P S)
    work, where S is samples per ray. This call allocates no device buffers and
    performs no image transfers. Checked calls synchronise for input/output
    status; unchecked calls require valid current inputs and a later
    ``workspace.check_status()``. The caller retains the O(6 P) diagnostic,
    unlike production optimisation. Ambient tapes and outstanding projection
    passes are rejected. Branch derivatives at knots/ties/grazing rays do not
    assert smoothness. Only the current local right chart is differentiated.
    """
    require_no_tape()
    _inputs(mu, pose, workspace, stream)
    if workspace._recorded_tape is not None:
        raise ContractError("the outstanding tape owns the projection inputs")
    if not workspace.spec.active_pose:
        raise ContractError("pose is fixed in this workspace")
    ctx, wp = workspace.context, workspace.context.wp
    pixels = workspace.geometry.pixels
    ctx.array(depth_seeds, "depth_seeds", dtype=workspace.dtype, shape=(pixels,))
    ctx.array(out_jacobian, "out_jacobian", dtype=wp.float64, shape=(6 * pixels,))
    ctx.disjoint(
        [("mu", mu), ("pose", pose), ("depth_seeds", depth_seeds)],
        [("out_jacobian", out_jacobian)],
    )
    if validate:
        _validate(mu, pose, workspace, depth_seeds)
    wp.launch_tiled(
        workspace._kernels.get_vjp(
            False,
            False,
            True,
            per_ray=True,
            double_seed=workspace.spec.precision == "float64",
            cell_gauss=workspace.spec.integration == "cell_gauss",
        ),
        dim=(pixels + _BLOCK - 1) // _BLOCK,
        inputs=[mu, pose, workspace._configuration, depth_seeds],
        outputs=[workspace._empty, out_jacobian, workspace._status],
        stream=workspace.stream,
        device=workspace.device,
        block_dim=_BLOCK,
        record_tape=False,
    )
    if validate:
        workspace.check_status()


# endregion book:projection-pose-sensitivities
