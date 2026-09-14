"""Constrained vector updates and physical-space penalties; no imaging physics."""

from functools import cache

import warp as wp

OPTIONS = {"fast_math": False, "fuse_fp": True, "enable_backward": False}
wp.set_module_options(OPTIONS)
METRIC_ROUNDOFF = wp.constant(wp.float64(16.0 * 2.0**-52))


@wp.kernel
def add_paths(source: wp.array(dtype=wp.float32), total: wp.array(dtype=wp.float64)):
    index = wp.tid()
    total[index] += wp.float64(source[index])


@wp.kernel
def narrow_paths(
    source: wp.array(dtype=wp.float64),
    destination: wp.array(dtype=wp.float32),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    value = wp.float32(source[index])
    destination[index] = value
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, 2)


@cache
def projected_step(materials: int, nonnegative: bool = False):
    vector = wp.types.vector(length=materials, dtype=wp.float64)

    @wp.kernel(module="unique", module_options=OPTIONS)
    def project(
        fields: wp.array(dtype=wp.float32),
        gradient: wp.array(dtype=wp.float32),
        voxels: int,
        step: wp.float64,
        trial: wp.array(dtype=wp.float32),
        slope_components: wp.array(dtype=wp.float64),
        mappings: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        voxel = wp.tid()
        values = vector()
        ordered = vector()
        total = wp.float64(0.0)
        maximum = wp.float64(-1.0e308)
        for material in range(materials):
            index = material * voxels + voxel
            value = wp.float64(fields[index]) - step * wp.float64(gradient[index])
            values[material] = value
            total += wp.max(value, wp.float64(0.0))
            maximum = wp.max(maximum, value)
            if not wp.isfinite(value):
                wp.atomic_or(status, 0, 2)
        threshold = wp.float64(0.0)
        if wp.static(not nonnegative) and total > wp.float64(1.0):
            # Translation invariance avoids subtracting two very large values
            # when a large first trial step lands far outside the simplex.
            for material in range(materials):
                values[material] = values[material] - maximum
                ordered[material] = values[material]
            # Small fixed M: insertion sort stays in registers, with no scratch
            # allocation and no global sort. The production envelope is M<=8.
            for material in range(1, materials):
                value = ordered[material]
                slot = material
                while slot > 0:
                    if ordered[slot - 1] >= value:
                        break
                    ordered[slot] = ordered[slot - 1]
                    slot -= 1
                ordered[slot] = value
            partial = wp.float64(0.0)
            for material in range(materials):
                partial += ordered[material]
                candidate = (partial - wp.float64(1.0)) / wp.float64(material + 1)
                if ordered[material] > candidate:
                    threshold = candidate
        mapping = wp.float64(0.0)
        for material in range(materials):
            index = material * voxels + voxel
            projected = wp.max(values[material] - threshold, wp.float64(0.0))
            original = wp.float64(fields[index])
            # Mapping uses the FP64 projected point; line search uses the
            # actual rounded FP32 update that the forward operator will see.
            mapping = wp.max(mapping, wp.abs((projected - original) / step))
            rounded = wp.float32(projected)
            if not wp.isfinite(rounded):
                wp.atomic_or(status, 0, 2)
            trial[index] = rounded
            slope_components[index] = wp.float64(gradient[index]) * (wp.float64(rounded) - original)
        mappings[voxel] = mapping

    return project


@wp.func
def metric_boundary_kkt(
    point: wp.vec2d,
    fields: wp.vec2d,
    gradient: wp.vec2d,
    step: wp.float64,
    metric: wp.vec3d,
    face: int,
) -> bool:
    displacement = point - fields
    q = step * gradient
    v0 = metric[0] * displacement[0] + metric[1] * displacement[1] + q[0]
    v1 = metric[1] * displacement[0] + metric[2] * displacement[1] + q[1]
    # Form the cap-tangent derivative without subtracting two huge common
    # normal components. Its error allowance must also exclude that common
    # component, or nearly tied vertices could incorrectly replace the cap.
    difference = (metric[0] - metric[1]) * displacement[0]
    difference += (metric[1] - metric[2]) * displacement[1]
    difference += step * (gradient[0] - gradient[1])
    epsilon = METRIC_ROUNDOFF
    # A small displacement still inherits rounding from constructing point
    # near a nonzero field and then subtracting the field. Bound those terms
    # separately; a displacement-only allowance vanishes as step shrinks and
    # can reject a correctly solved active edge indefinitely.
    magnitude0 = wp.abs(point[0]) + wp.abs(fields[0])
    magnitude1 = wp.abs(point[1]) + wp.abs(fields[1])
    tolerance0 = epsilon * (
        wp.abs(metric[0]) * magnitude0 + wp.abs(metric[1]) * magnitude1 + wp.abs(q[0])
    )
    tolerance1 = epsilon * (
        wp.abs(metric[1]) * magnitude0 + wp.abs(metric[2]) * magnitude1 + wp.abs(q[1])
    )
    tangent_tolerance = epsilon * (
        wp.abs(metric[0] - metric[1]) * magnitude0
        + wp.abs(metric[1] - metric[2]) * magnitude1
        + wp.abs(step * (gradient[0] - gradient[1]))
    )
    if face == 0:  # origin
        return v0 >= -tolerance0 and v1 >= -tolerance1
    if face == 1:  # (1,0)
        return v0 <= tolerance0 and difference <= tangent_tolerance
    if face == 2:  # (0,1)
        return v1 <= tolerance1 and difference >= -tangent_tolerance
    if face == 3:  # open horizontal edge
        return wp.abs(v0) <= tolerance0 and v1 >= -tolerance1
    if face == 4:  # open vertical edge
        return wp.abs(v1) <= tolerance1 and v0 >= -tolerance0
    # open sum-one edge
    return wp.abs(difference) <= tangent_tolerance and v0 <= tolerance0


@wp.kernel
def projected_metric_step(
    fields: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    voxels: int,
    step: wp.float64,
    metric: wp.vec3d,
    inverse: wp.vec3d,
    trial: wp.array(dtype=wp.float32),
    slope_components: wp.array(dtype=wp.float64),
    mappings: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    voxel = wp.tid()
    field = wp.vec2d(wp.float64(fields[voxel]), wp.float64(fields[voxels + voxel]))
    grad = wp.vec2d(wp.float64(gradient[voxel]), wp.float64(gradient[voxels + voxel]))
    linear = step * grad
    unconstrained = field - wp.vec2d(
        inverse[0] * linear[0] + inverse[1] * linear[1],
        inverse[1] * linear[0] + inverse[2] * linear[1],
    )
    point = unconstrained
    if (
        not wp.isfinite(unconstrained[0])
        or not wp.isfinite(unconstrained[1])
        or unconstrained[0] < wp.float64(0.0)
        or unconstrained[1] < wp.float64(0.0)
        or unconstrained[0] + unconstrained[1] > wp.float64(1.0)
    ):
        # Enumerate vertices and open-edge minima, then use KKT conditions.
        # Absolute objective scores lose their ordering for a large common
        # gradient; e.g. q=(-1e20,-1e20) still has a unique cap interior point.
        denominator = (metric[0] - metric[1]) + (metric[2] - metric[1])
        cap_numerator = (metric[0] - metric[1]) * field[0]
        cap_numerator += (metric[1] - metric[2]) * field[1]
        cap_numerator -= step * (grad[0] - grad[1])
        cap_numerator += metric[2] - metric[1]
        found = bool(False)  # noqa: UP018 - Warp requires a mutable loop variable.
        for face in range(6):
            candidate = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
            open_edge = bool(True)  # noqa: UP018 - Explicit Warp runtime Boolean.
            if face == 1:
                candidate[0] = wp.float64(1.0)
            elif face == 2:
                candidate[1] = wp.float64(1.0)
            elif face == 3:
                candidate[0] = field[0] + (metric[1] * field[1] - linear[0]) / metric[0]
                open_edge = candidate[0] > wp.float64(0.0) and candidate[0] < wp.float64(1.0)
            elif face == 4:
                candidate[1] = field[1] + (metric[1] * field[0] - linear[1]) / metric[2]
                open_edge = candidate[1] > wp.float64(0.0) and candidate[1] < wp.float64(1.0)
            elif face == 5:
                candidate[0] = cap_numerator / denominator
                candidate[1] = wp.float64(1.0) - candidate[0]
                open_edge = candidate[0] > wp.float64(0.0) and candidate[0] < wp.float64(1.0)
            if open_edge and metric_boundary_kkt(candidate, field, grad, step, metric, face):
                point = candidate
                found = True
                break
        if not found:
            wp.atomic_or(status, 0, 2)
    mapping = wp.float64(0.0)
    for material in range(2):
        index = material * voxels + voxel
        rounded = wp.float32(point[material])
        trial[index] = rounded
        slope_components[index] = grad[material] * (wp.float64(rounded) - field[material])
        mapping = wp.max(mapping, wp.abs((point[material] - field[material]) / step))
        if not wp.isfinite(rounded) or not wp.isfinite(linear[material]):
            wp.atomic_or(status, 0, 2)
    mappings[voxel] = mapping


@wp.kernel
def projected_metric_nonnegative_step(
    fields: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    voxels: int,
    step: wp.float64,
    metric: wp.vec3d,
    inverse: wp.vec3d,
    trial: wp.array(dtype=wp.float32),
    slope_components: wp.array(dtype=wp.float64),
    mappings: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    voxel = wp.tid()
    field = wp.vec2d(wp.float64(fields[voxel]), wp.float64(fields[voxels + voxel]))
    grad = wp.vec2d(wp.float64(gradient[voxel]), wp.float64(gradient[voxels + voxel]))
    linear = step * grad
    point = field - wp.vec2d(
        inverse[0] * linear[0] + inverse[1] * linear[1],
        inverse[1] * linear[0] + inverse[2] * linear[1],
    )
    if (
        not wp.isfinite(point[0])
        or not wp.isfinite(point[1])
        or point[0] < wp.float64(0.0)
        or point[1] < wp.float64(0.0)
    ):
        found = bool(False)  # noqa: UP018 - Warp requires a mutable loop variable.
        for face_index in range(3):
            candidate = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
            face = int(0)  # noqa: UP018, RUF046 - Mutable Warp runtime integer.
            feasible = bool(True)  # noqa: UP018 - Explicit Warp runtime Boolean.
            if face_index == 1:
                face = 3
                candidate[0] = field[0] + (metric[1] * field[1] - linear[0]) / metric[0]
                feasible = candidate[0] > wp.float64(0.0)
            elif face_index == 2:
                face = 4
                candidate[1] = field[1] + (metric[1] * field[0] - linear[1]) / metric[2]
                feasible = candidate[1] > wp.float64(0.0)
            if feasible and metric_boundary_kkt(candidate, field, grad, step, metric, face):
                point = candidate
                found = True
                break
        if not found:
            wp.atomic_or(status, 0, 2)
    mapping = wp.float64(0.0)
    for material in range(2):
        index = material * voxels + voxel
        rounded = wp.float32(point[material])
        trial[index] = rounded
        slope_components[index] = grad[material] * (wp.float64(rounded) - field[material])
        mapping = wp.max(mapping, wp.abs((point[material] - field[material]) / step))
        if not wp.isfinite(rounded) or not wp.isfinite(linear[material]):
            wp.atomic_or(status, 0, 2)
    mappings[voxel] = mapping


@wp.kernel
def max_partials(
    source: wp.array(dtype=wp.float64),
    count: int,
    destination: wp.array(dtype=wp.float64),
):
    block, lane = wp.tid()
    index = block * 256 + lane
    value = wp.float64(0.0)
    if index < count:
        value = source[index]
    result = wp.tile_max(wp.tile(value))
    wp.tile_store(destination, result, offset=block)


@wp.kernel
def regularise(
    fields: wp.array(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    coefficient_x: wp.float64,
    coefficient_y: wp.float64,
    coefficient_z: wp.float64,
    differentiate: bool,
    gradient: wp.array(dtype=wp.float32),
    components: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    voxel = index % (nx * ny * nz)
    x = voxel % nx
    y = (voxel // nx) % ny
    z = voxel // (nx * ny)
    value = wp.float64(fields[index])
    derivative = wp.float64(0.0)
    penalty = wp.float64(0.0)
    if x > 0:
        derivative += coefficient_x * (value - wp.float64(fields[index - 1]))
    if x + 1 < nx:
        difference = value - wp.float64(fields[index + 1])
        derivative += coefficient_x * difference
        penalty += wp.float64(0.5) * coefficient_x * difference * difference
    if y > 0:
        derivative += coefficient_y * (value - wp.float64(fields[index - nx]))
    if y + 1 < ny:
        difference = value - wp.float64(fields[index + nx])
        derivative += coefficient_y * difference
        penalty += wp.float64(0.5) * coefficient_y * difference * difference
    if z > 0:
        derivative += coefficient_z * (value - wp.float64(fields[index - nx * ny]))
    if z + 1 < nz:
        difference = value - wp.float64(fields[index + nx * ny])
        derivative += coefficient_z * difference
        penalty += wp.float64(0.5) * coefficient_z * difference * difference
    components[index] = penalty
    if differentiate:
        result = wp.float32(wp.float64(gradient[index]) + derivative)
        gradient[index] = result
        if not wp.isfinite(result):
            wp.atomic_or(status, 0, 2)


@wp.kernel
def scalar_sum(
    left: wp.array(dtype=wp.float64),
    right: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64),
):
    out[0] = left[0] + right[0]
