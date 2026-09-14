"""Streaming finite-ray quadrature and an explicit first-order discrete VJP.

One lane traces one ray. Values and pose derivatives use FP64 registers with
FP32 field storage. Reverse recomputes each sample; no ray-by-sample tensor or
image Jacobian is retained. Pose partials use a fixed tree; active-volume
cotangents use FP32 scatter atomics and are not bitwise deterministic.
"""

# Warp annotations are executable DSL expressions; host interfaces remain strict.
# The optional GPU import is resolved only when an operator is prepared.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

from functools import cache

import warp as wp

from dpt.kernels.geometry import unpack_rotation, unpack_translation
from dpt.kernels.projection_checks import (
    dependency_marker as dependency_marker,
)
from dpt.kernels.projection_checks import (
    validate_finite as validate_finite,
)
from dpt.kernels.projection_checks import (
    validate_nonnegative as validate_nonnegative,
)

OPTIONS = {"fast_math": False, "fuse_fp": True, "enable_backward": False}
wp.set_module_options(OPTIONS)
BLOCK = 128
Vec12d = wp.types.vector(length=12, dtype=wp.float64)


@wp.struct
class Configuration:
    shape: wp.vec3i
    object_to_grid: wp.mat33d
    grid_origin: wp.vec3d
    source: wp.vec3d
    detector_origin: wp.vec3d
    column_step: wp.vec3d
    row_step: wp.vec3d
    width: int
    pixels: int
    samples: int


@wp.struct
class Ray:
    origin: wp.vec3d
    direction: wp.vec3d
    object_source: wp.vec3d
    object_direction: wp.vec3d
    world_source_relative: wp.vec3d
    world_direction: wp.vec3d
    length: wp.float64


@wp.func
def ray_for_pixel(config: Configuration, pose: wp.array(dtype=wp.float64), pixel: int) -> Ray:
    ray = Ray()
    row, column = pixel // config.width, pixel % config.width
    end = (
        config.detector_origin
        + wp.float64(column) * config.column_step
        + wp.float64(row) * config.row_step
    )
    inverse = wp.transpose(unpack_rotation(pose))
    ray.world_source_relative = config.source - unpack_translation(pose)
    ray.world_direction = end - config.source
    ray.object_source = inverse * ray.world_source_relative
    ray.object_direction = inverse * ray.world_direction
    ray.origin = config.object_to_grid * (ray.object_source - config.grid_origin)
    ray.direction = config.object_to_grid * ray.object_direction
    ray.length = wp.length(ray.world_direction)
    return ray


@wp.func
def finite_point(point: wp.vec3d) -> bool:
    return wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2])


@wp.func
def valid_ray(ray: Ray) -> bool:
    return (
        finite_point(ray.origin)
        and finite_point(ray.direction)
        and finite_point(ray.object_source)
        and finite_point(ray.object_direction)
        and finite_point(ray.world_source_relative)
        and finite_point(ray.world_direction)
        and wp.isfinite(ray.length)
        and ray.length > wp.float64(0.0)
    )


@wp.func
def inside_support(point: wp.vec3d, shape: wp.vec3i) -> bool:
    return (
        finite_point(point)
        and point[0] >= wp.float64(-0.5)
        and point[1] >= wp.float64(-0.5)
        and point[2] >= wp.float64(-0.5)
        and point[0] <= wp.float64(shape[0]) - wp.float64(0.5)
        and point[1] <= wp.float64(shape[1]) - wp.float64(0.5)
        and point[2] <= wp.float64(shape[2]) - wp.float64(0.5)
    )


@wp.struct
class Interval:
    lower: wp.float64
    upper: wp.float64
    lower_origin_gradient: wp.vec3d
    lower_direction_gradient: wp.vec3d
    upper_origin_gradient: wp.vec3d
    upper_direction_gradient: wp.vec3d


# region book:projection-active-intersection
@wp.func
def finite_interval(origin: wp.vec3d, direction: wp.vec3d, shape: wp.vec3i) -> Interval:
    """Clip a finite segment and retain the active face derivatives.

    Exact parallelism is handled separately; arbitrary epsilon thresholds would
    change the physical interval. At tied faces a derivative is not unique:
    strict comparisons choose the first active axis, a documented branch value.
    """
    interval = Interval()
    interval.lower = wp.float64(0.0)
    interval.upper = wp.float64(1.0)
    for axis in range(3):
        low = wp.float64(-0.5)
        high = wp.float64(shape[axis]) - wp.float64(0.5)
        velocity = direction[axis]
        if velocity == wp.float64(0.0):
            if origin[axis] < low or origin[axis] > high:
                interval.upper = wp.float64(-1.0)
        else:
            first = (low - origin[axis]) / velocity
            last = (high - origin[axis]) / velocity
            if first > last:
                temporary = first
                first = last
                last = temporary
            if first > interval.lower:
                interval.lower = first
                interval.lower_origin_gradient = wp.vec3d(0.0)
                interval.lower_direction_gradient = wp.vec3d(0.0)
                interval.lower_origin_gradient[axis] = -wp.float64(1.0) / velocity
                interval.lower_direction_gradient[axis] = -first / velocity
            if last < interval.upper:
                interval.upper = last
                interval.upper_origin_gradient = wp.vec3d(0.0)
                interval.upper_direction_gradient = wp.vec3d(0.0)
                interval.upper_origin_gradient[axis] = -wp.float64(1.0) / velocity
                interval.upper_direction_gradient[axis] = -last / velocity
    return interval


# endregion book:projection-active-intersection


@wp.struct
class Sample:
    value: wp.float64
    gradient: wp.vec3d


# region book:projection-clamped-trilinear
@wp.func
def sample_field(field: wp.array(dtype=wp.float32), point: wp.vec3d, shape: wp.vec3i) -> Sample:
    """Value and grid-coordinate slope of the half-cell-extended sampled field."""
    result = Sample()
    if not inside_support(point, shape):
        return result
    lower = wp.vec3i()
    fraction = wp.vec3d()
    slope = wp.vec3d()
    for axis in range(3):
        coordinate = wp.clamp(point[axis], wp.float64(0.0), wp.float64(shape[axis] - 1))
        lower[axis] = wp.min(int(wp.floor(coordinate)), wp.max(shape[axis] - 2, 0))
        fraction[axis] = coordinate - wp.float64(lower[axis])
        if point[axis] >= wp.float64(0.0) and point[axis] < wp.float64(shape[axis] - 1):
            slope[axis] = wp.float64(1.0)
    # Separable interpolation reuses the same eight loads for value and slope.
    # Form differences in FP64: FP32 subtraction would lose small field slopes.
    # The compiler removes slope work from the forward-only caller.
    x0, y0, z0 = lower[0], lower[1], lower[2]
    x1 = wp.min(x0 + 1, shape[0] - 1)
    y1 = wp.min(y0 + 1, shape[1] - 1)
    z1 = wp.min(z0 + 1, shape[2] - 1)
    row00 = (z0 * shape[1] + y0) * shape[0]
    row10 = (z0 * shape[1] + y1) * shape[0]
    row01 = (z1 * shape[1] + y0) * shape[0]
    row11 = (z1 * shape[1] + y1) * shape[0]
    v000 = wp.float64(field[row00 + x0])
    v100 = wp.float64(field[row00 + x1])
    v010 = wp.float64(field[row10 + x0])
    v110 = wp.float64(field[row10 + x1])
    v001 = wp.float64(field[row01 + x0])
    v101 = wp.float64(field[row01 + x1])
    v011 = wp.float64(field[row11 + x0])
    v111 = wp.float64(field[row11 + x1])
    dx00, dx10 = v100 - v000, v110 - v010
    dx01, dx11 = v101 - v001, v111 - v011
    fx, fy, fz = fraction[0], fraction[1], fraction[2]
    # Weighted lerp preserves a small endpoint next to a huge neighbour.
    # a+t*(b-a) would erase b at t=1 when the difference rounds to -a.
    ax, ay, az = wp.float64(1.0) - fx, wp.float64(1.0) - fy, wp.float64(1.0) - fz
    x00, x10 = ax * v000 + fx * v100, ax * v010 + fx * v110
    x01, x11 = ax * v001 + fx * v101, ax * v011 + fx * v111
    xy0, xy1 = ay * x00 + fy * x10, ay * x01 + fy * x11
    result.value = az * xy0 + fz * xy1
    # Differentiate corner values before interpolation. Subtracting two
    # already interpolated values can erase a small slope beside a huge
    # orthogonal background, even when its cotangent is representable.
    dx0, dx1 = ay * dx00 + fy * dx10, ay * dx01 + fy * dx11
    dy0 = ax * (v010 - v000) + fx * (v110 - v100)
    dy1 = ax * (v011 - v001) + fx * (v111 - v101)
    dz0 = ax * (v001 - v000) + fx * (v101 - v100)
    dz1 = ax * (v011 - v010) + fx * (v111 - v110)
    result.gradient = wp.vec3d(
        slope[0] * (az * dx0 + fz * dx1),
        slope[1] * (az * dy0 + fz * dy1),
        slope[2] * (ay * dz0 + fy * dz1),
    )
    return result


# endregion book:projection-clamped-trilinear


@wp.func
def scatter_field(
    gradient: wp.array(dtype=wp.float32),
    point: wp.vec3d,
    shape: wp.vec3i,
    seed: wp.float64,
    status: wp.array(dtype=wp.int32),
):
    if not finite_point(point):
        wp.atomic_or(status, 0, 2)
    if not inside_support(point, shape):
        return
    lower = wp.vec3i()
    fraction = wp.vec3d()
    for axis in range(3):
        coordinate = wp.clamp(point[axis], wp.float64(0.0), wp.float64(shape[axis] - 1))
        lower[axis] = wp.min(int(wp.floor(coordinate)), wp.max(shape[axis] - 2, 0))
        fraction[axis] = coordinate - wp.float64(lower[axis])
    for corner in range(8):
        bit = wp.vec3i(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1)
        index = wp.vec3i()
        contribution = seed
        for axis in range(3):
            index[axis] = wp.min(lower[axis] + bit[axis], shape[axis] - 1)
            if bit[axis] == 0:
                contribution *= wp.float64(1.0) - fraction[axis]
            else:
                contribution *= fraction[axis]
        rounded = wp.float32(contribution)
        if not wp.isfinite(rounded):
            wp.atomic_or(status, 0, 2)
        wp.atomic_add(gradient, (index[2] * shape[1] + index[1]) * shape[0] + index[0], rounded)


# region book:projection-streamed-quadrature
@wp.kernel
def forward(
    field: wp.array(dtype=wp.float32),
    pose: wp.array(dtype=wp.float64),
    config: Configuration,
    output: wp.array(dtype=wp.float32),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    ray = ray_for_pixel(config, pose, pixel)
    if not valid_ray(ray):
        wp.atomic_or(status, 0, 2)
        output[pixel] = wp.float32(0.0)
        return
    interval = finite_interval(ray.origin, ray.direction, config.shape)
    integral = wp.float64(0.0)
    if interval.upper > interval.lower:
        step = (interval.upper - interval.lower) / wp.float64(config.samples)
        for sample in range(config.samples):
            position = interval.lower + (wp.float64(sample) + wp.float64(0.5)) * step
            point = ray.origin + position * ray.direction
            if not finite_point(point):
                wp.atomic_or(status, 0, 2)
            integral += sample_field(field, point, config.shape).value
        integral *= ray.length * step
    output[pixel] = wp.float32(integral)
    if not wp.isfinite(output[pixel]):
        wp.atomic_or(status, 0, 2)


# endregion book:projection-streamed-quadrature


# region book:projection-discrete-vjp
@cache
def get_vjp(
    active_volume: bool, matrix_gradient: bool, active_pose: bool, *, per_ray: bool = False
):
    if per_ray and (active_volume or matrix_gradient or not active_pose):
        raise ValueError("per-ray diagnostics require only the six local pose coordinates")
    components = 12 if matrix_gradient else 6

    @wp.kernel(module="unique", module_options=OPTIONS)
    def vjp(
        field: wp.array(dtype=wp.float32),
        pose: wp.array(dtype=wp.float64),
        config: Configuration,
        seeds: wp.array(dtype=wp.float32),
        volume_gradient: wp.array(dtype=wp.float32),
        partials: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        block, lane = wp.tid()
        pixel = block * BLOCK + lane
        gradient = Vec12d()
        if pixel < config.pixels:
            ray = ray_for_pixel(config, pose, pixel)
            if not valid_ray(ray):
                wp.atomic_or(status, 0, 2)
            else:
                interval = finite_interval(ray.origin, ray.direction, config.shape)
                span = interval.upper - interval.lower
                if span > wp.float64(0.0):
                    mean = wp.float64(0.0)
                    direct_origin = wp.vec3d()
                    direct_direction = wp.vec3d()
                    shift_lower = wp.float64(0.0)
                    shift_upper = wp.float64(0.0)
                    for sample in range(config.samples):
                        alpha = (wp.float64(sample) + wp.float64(0.5)) / wp.float64(config.samples)
                        position = interval.lower + alpha * span
                        point = ray.origin + position * ray.direction
                        if not finite_point(point):
                            wp.atomic_or(status, 0, 2)
                        if wp.static(active_pose):
                            value = sample_field(field, point, config.shape)
                            mean += value.value
                            direct_origin += value.gradient
                            direct_direction += position * value.gradient
                            along = wp.dot(value.gradient, ray.direction)
                            shift_lower += (wp.float64(1.0) - alpha) * along
                            shift_upper += alpha * along
                        if wp.static(active_volume):
                            scatter_field(
                                volume_gradient,
                                point,
                                config.shape,
                                wp.float64(seeds[pixel])
                                * ray.length
                                * span
                                / wp.float64(config.samples),
                                status,
                            )
                    if wp.static(active_pose):
                        lower_seed = -mean + span * shift_lower
                        upper_seed = mean + span * shift_upper
                        origin_gradient = (
                            span * direct_origin
                            + lower_seed * interval.lower_origin_gradient
                            + upper_seed * interval.upper_origin_gradient
                        )
                        direction_gradient = (
                            span * direct_direction
                            + lower_seed * interval.lower_direction_gradient
                            + upper_seed * interval.upper_direction_gradient
                        )
                        factor = wp.float64(seeds[pixel]) * ray.length / wp.float64(config.samples)
                        source_adjoint = factor * (
                            wp.transpose(config.object_to_grid) * origin_gradient
                        )
                        direction_adjoint = factor * (
                            wp.transpose(config.object_to_grid) * direction_gradient
                        )
                        if wp.static(matrix_gradient):
                            for row in range(3):
                                for column in range(3):
                                    gradient[3 * row + column] = (
                                        ray.world_source_relative[row] * source_adjoint[column]
                                        + ray.world_direction[row] * direction_adjoint[column]
                                    )
                            translation_adjoint = -(unpack_rotation(pose) * source_adjoint)
                            for axis in range(3):
                                gradient[9 + axis] = translation_adjoint[axis]
                        else:
                            rotation_adjoint = wp.cross(
                                source_adjoint, ray.object_source
                            ) + wp.cross(direction_adjoint, ray.object_direction)
                            for axis in range(3):
                                gradient[axis] = -source_adjoint[axis]
                                gradient[axis + 3] = rotation_adjoint[axis]
        # The diagnostic writes the same ray derivative before any reduction.
        # Its caller owns O(6 P) storage; the production specialisation retains
        # only reduction partials and performs exactly its existing reduction.
        if wp.static(per_ray):
            if pixel < config.pixels:
                for component in range(components):
                    component_value = gradient[component]
                    partials[wp.int64(pixel) * wp.int64(components) + wp.int64(component)] = (
                        component_value
                    )
                    if not wp.isfinite(component_value):
                        wp.atomic_or(status, 0, 2)
        # All lanes, including padded rays, participate in the fixed reduction.
        elif wp.static(active_pose):
            for component in range(components):
                values = wp.tile(gradient[component])
                total = wp.tile_sum(values)
                wp.tile_store(partials, total, offset=block * components + component)

    return vjp


# endregion book:projection-discrete-vjp
