"""Fixed-density material-fraction paths using the canonical sampled-field projector.

Fields are material-major, flat FP32 arrays with M*V values; paths have M*P
values in millimetres, using the supplied ProjectionSpec precision for both
paths and their cotangents. Field-gradient storage remains FP32.
Each basis field is a volume fraction. Fractions are
non-negative and sum to at most one at every sample; the remainder is vacuum.
The FP64 sum admits 2^-24 excess for independently rounded binary32 fractions.
The explicit nonnegative domain instead accepts dimensionless equivalent-basis
fields without a sum cap. It uses the same integration and derivatives; basis
scales belong in the supplied attenuation coefficients. There is no hidden
normalisation or inferred CT-to-material conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from dpt._runtime import DeviceContext, load_kernels, require_no_tape
from dpt.contracts import ContractError, integer
from dpt.geometry import DetectorGeometry
from dpt.projection import ProjectionSpec, ProjectionWorkspace, prepare_projection
from dpt.projection import project_optical_depth as _project
from dpt.projection import projection_vjp as _vjp
from dpt.volumes import GridSpec


@dataclass(slots=True)
class MaterialProjectionWorkspace:
    """Prepared views retain their caller-owned parent buffers for the whole solve.

    M serial launches reuse a single projection workspace, avoiding duplicated
    ray geometry and quadrature code. This is an execution candidate, not a
    measured throughput claim. No per-evaluation views or device arrays are made.
    """

    projection: ProjectionWorkspace
    materials: int
    fields: Any
    paths: Any
    adj_paths: Any
    grad_fields: Any
    field_views: tuple[Any, ...]
    path_views: tuple[Any, ...]
    seed_views: tuple[Any, ...]
    gradient_views: tuple[Any, ...]
    kernels: Any = field(repr=False)
    status: Any = field(repr=False)
    field_domain: Literal["fractions", "nonnegative"] = "fractions"

    @property
    def context(self) -> DeviceContext:
        return self.projection.context

    @property
    def scratch_bytes(self) -> int:
        return self.projection.scratch_bytes + 4

    def clear_status(self) -> None:
        self.projection.clear_status()
        with self.context.scope():
            self.status.zero_()

    def check_status(self) -> None:
        self.projection.check_status()
        if int(self.status.numpy()[0]):
            name = "fractions" if self.field_domain == "fractions" else "nonnegative fields"
            raise ContractError(f"invalid material {name} or path cotangents")

    def validate_inputs(self, *, seeds: bool = False, stream: Any = None) -> None:
        """Check current bound fractions and optional cotangents without projecting.

        Fraction-domain roundoff permits a total up to 1 + 2^-24. Nonnegative
        fields have no sum cap. Values are never changed. This checks no pose
        and does not certify later field mutations.
        """
        require_no_tape()
        self.context.assert_stream(stream)
        if seeds and self.adj_paths is None:
            raise ContractError("material validation requires a bound path cotangent")
        _validate(self, seeds=seeds)


def prepare_material_projection(
    grid: GridSpec,
    geometry: DetectorGeometry,
    spec: ProjectionSpec,
    *,
    materials: int,
    fields: Any,
    out_paths: Any,
    adj_paths: Any = None,
    out_fields: Any = None,
    device: str = "cuda:0",
    stream: Any = None,
    field_domain: Literal["fractions", "nonnegative"] = "fractions",
) -> MaterialProjectionWorkspace:
    """Bind fields; nonnegative equivalent-basis fields have no fraction sum cap.

    Both domains use dimensionless fields, paths in mm and supplied basis
    attenuation in mm^-1. The caller records any mass-density normalisation.
    ``spec.precision`` selects FP32 or FP64 paths and path cotangents; fields
    and their accumulated volume gradients remain FP32.
    """
    if field_domain not in ("fractions", "nonnegative"):
        raise ContractError("field_domain must be fractions or nonnegative")
    integer(materials, "materials", minimum=1, maximum=32)
    integer(materials * grid.voxels, "material field length", minimum=1)
    integer(materials * geometry.pixels, "material path length", minimum=1)
    projection = prepare_projection(grid, geometry, spec, device=device, stream=stream)
    ctx, wp = projection.context, projection.context.wp
    ctx.array(fields, "fields", dtype=wp.float32, shape=(materials * grid.voxels,))
    path_dtype = wp.float64 if spec.precision == "float64" else wp.float32
    ctx.array(out_paths, "out_paths", dtype=path_dtype, shape=(materials * geometry.pixels,))
    reads = [("fields", fields)]
    writes = [("out_paths", out_paths)]
    if adj_paths is not None:
        ctx.array(adj_paths, "adj_paths", dtype=path_dtype, shape=(materials * geometry.pixels,))
        reads.append(("adj_paths", adj_paths))
    if out_fields is not None:
        if not spec.active_volume:
            raise ContractError("material fields are fixed in this projection specification")
        if adj_paths is None:
            raise ContractError("material field gradients require a bound path cotangent")
        ctx.array(out_fields, "out_fields", dtype=wp.float32, shape=(materials * grid.voxels,))
        writes.append(("out_fields", out_fields))
    ctx.disjoint(reads, writes)
    with ctx.scope():
        status = wp.zeros(1, dtype=wp.int32, device=ctx.device)
    field_views = tuple(fields[m * grid.voxels : (m + 1) * grid.voxels] for m in range(materials))
    path_views = tuple(
        out_paths[m * geometry.pixels : (m + 1) * geometry.pixels] for m in range(materials)
    )
    seeds = (
        tuple(adj_paths[m * geometry.pixels : (m + 1) * geometry.pixels] for m in range(materials))
        if adj_paths is not None
        else ()
    )
    gradients = (
        tuple(out_fields[m * grid.voxels : (m + 1) * grid.voxels] for m in range(materials))
        if out_fields is not None
        else ()
    )
    return MaterialProjectionWorkspace(
        projection,
        materials,
        fields,
        out_paths,
        adj_paths,
        out_fields,
        field_views,
        path_views,
        seeds,
        gradients,
        load_kernels("dpt.kernels.material_projection"),
        status,
        field_domain,
    )


def _validate(workspace: MaterialProjectionWorkspace, *, seeds: bool) -> None:
    ctx, projection = workspace.context, workspace.projection
    workspace.clear_status()
    ctx.wp.launch(
        workspace.kernels.validate_fractions
        if workspace.field_domain == "fractions"
        else workspace.kernels.validate_nonnegative,
        dim=projection.grid.voxels,
        inputs=[workspace.fields, workspace.materials, projection.grid.voxels],
        outputs=[workspace.status],
        stream=ctx.stream,
        record_tape=False,
    )
    if seeds:
        ctx.wp.launch(
            workspace.kernels.validate_seeds(projection.spec.precision),
            dim=workspace.materials * projection.geometry.pixels,
            inputs=[workspace.adj_paths],
            outputs=[workspace.status],
            stream=ctx.stream,
            record_tape=False,
        )
    workspace.check_status()


# region book:material-path-composition
def project_material_paths(
    pose: Any,
    *,
    workspace: MaterialProjectionWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Overwrite material-major path lengths in mm, using one canonical projector.

    All material data stay on CUDA. The host loop schedules M operations and
    never visits a voxel, pixel or quadrature sample. Explicit VJPs compose
    with spectral_signal; ambient tape recording is rejected.
    """
    require_no_tape()
    ctx, projection = workspace.context, workspace.projection
    ctx.assert_stream(stream)
    ctx.array(pose, "pose", dtype=ctx.wp.float64, shape=(12,))
    ctx.disjoint([("pose", pose)], [("paths", workspace.paths)])
    if validate:
        workspace.validate_inputs(stream=ctx.stream)
    for material in range(workspace.materials):
        _project(
            workspace.field_views[material],
            pose,
            workspace=projection,
            out_L=workspace.path_views[material],
            stream=ctx.stream,
            validate=validate and material == 0,
        )
    if validate:
        workspace.check_status()


# endregion book:material-path-composition


def material_projection_vjp(
    pose: Any,
    *,
    workspace: MaterialProjectionWorkspace,
    out_pose: Any = None,
    stream: Any = None,
    validate: bool = True,
    accumulate: bool = False,
) -> None:
    """Accumulate material cotangents into one local-right six-coordinate pose VJP.

    The optional bound field gradient is an ambient derivative. Feasible
    fraction perturbations must still respect non-negativity and the simplex;
    an optimiser must provide its own declared constrained parameterisation.
    Field scatter order follows the canonical projector's atomic contract.
    """
    require_no_tape()
    ctx, projection = workspace.context, workspace.projection
    ctx.assert_stream(stream)
    ctx.array(pose, "pose", dtype=ctx.wp.float64, shape=(12,))
    if not workspace.seed_views:
        raise ContractError("prepare a path cotangent before requesting material VJPs")
    writes: list[tuple[str, Any]] = []
    if out_pose is not None:
        if not projection.spec.active_pose:
            raise ContractError("pose is fixed in this projection specification")
        ctx.array(out_pose, "out_pose", dtype=ctx.wp.float64, shape=(6,))
        writes.append(("out_pose", out_pose))
    if workspace.grad_fields is not None:
        writes.append(("out_fields", workspace.grad_fields))
    if not writes:
        raise ContractError("request a pose or material field gradient")
    ctx.disjoint(
        [
            ("fields", workspace.fields),
            ("paths", workspace.paths),
            ("adj_paths", workspace.adj_paths),
            ("pose", pose),
        ],
        writes,
    )
    if validate:
        workspace.validate_inputs(seeds=True, stream=ctx.stream)
    for material in range(workspace.materials):
        # Pose contributions share one destination; distinct field slices must
        # each honour the caller's overwrite/accumulate request independently.
        field_gradient = workspace.gradient_views[material] if workspace.gradient_views else None
        if out_pose is not None and field_gradient is not None and material > 0 and not accumulate:
            with ctx.scope():
                field_gradient.zero_()
        _vjp(
            workspace.field_views[material],
            pose,
            adj_L=workspace.seed_views[material],
            workspace=projection,
            out_pose=out_pose,
            out_mu=field_gradient,
            stream=ctx.stream,
            validate=validate and material == 0,
            accumulate=accumulate or (out_pose is not None and material > 0),
        )
    if validate:
        workspace.check_status()
