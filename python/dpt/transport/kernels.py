"""Canonical complete-history Warp kernels, including the likelihood-ratio replay.

No path-by-event storage, queue allocation, roulette or hidden truncation. One
thread owns one original history and writes one sparse detector contribution.
Derivative replay runs the *same* flight function with a selected active material.
This initial scheduling candidate is unprofiled: divergence, register pressure,
FP64 cost and tally contention require actual-device acceptance before any claim.
"""

# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
# Integer constructors declare mutable Warp loop variables.
# ruff: noqa: UP018, RUF046
import warp as wp

from dpt.kernels.random import random4

from ._constants import LOG_SOURCE_AMPLITUDE, RANDOM_NAMESPACE_BIT, SOURCE_AMPLITUDE

wp.set_module_options({"fast_math": False, "fuse_fp": True, "enable_backward": False})

MAX_DISTANCE = wp.constant(wp.float64(1.7976931348623157e308))
TWO_PI = wp.constant(wp.float64(6.2831853071795864769))
ELECTRON_REST_ENERGY_KEV = wp.constant(wp.float64(510.99895069))
MISS_PIXEL = wp.constant(-1)
INVALID_PIXEL = wp.constant(-2)


@wp.func_native("""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    // Every non-exited lane reaches this ballot, including misses and invalid
    // histories. Membership is established before the participating branch.
    const unsigned members = __ballot_sync(0xffffffffu, participating != 0);
    if (!participating) return wp::vec_t<2, double>(0.0, 0.0);
    const unsigned peers = __match_any_sync(members, target);
    const int leader = __ffs(peers) - 1;
    const int lane = threadIdx.x & 31;
    const int count = __popc(peers);
    double total = value;
    for (int delta = 1; delta < count; delta <<= 1) {
        // __fns selects the delta-th participating lane after this lane, even
        // for interleaved detector addresses or inactive/missing histories.
        const unsigned source = __fns(peers, lane, delta + 1);
        const double next = __shfl_sync(peers, total, source < 32 ? source : lane);
        if (source < 32)
            total = maximum ? fmax(total, next) : total + next;
    }
    return wp::vec_t<2, double>(total, lane == leader ? double(count) : 0.0);
#else
    return wp::vec_t<2, double>(participating ? value : 0.0, participating ? 1.0 : 0.0);
#endif
""")
def grouped_tally(value: wp.float64, target: int, maximum: int, participating: int) -> wp.vec2d:
    """Combine equal detector addresses selected by a converged warp ballot.

    Call unconditionally from every live kernel lane, after computing its input
    and participation flag. Every lane named by a peer mask then executes the
    same shuffles. Misses and the final partial warp contribute nothing. Only the
    peer leader updates global storage; unrelated detector addresses stay separate.
    Inputs are nonnegative magnitudes or bounded normalised scores/deviations.
    Summation order remains unspecified, as it was for per-history atomics.
    CUDA targets below SM 70 retain individual atomics; no unsupported match
    intrinsic is emitted there. The measured optimisation targets SM 121.
    """
    ...


@wp.func_native("""
if (a == 0.0 || b == 0.0 || c == 0.0) return 0.0;
int ea, eb, ec;
double ma = frexp(a, &ea);
double mb = frexp(b, &eb);
double mc = frexp(c, &ec);
return ldexp((ma * mb) * mc, ea + eb + ec);
""")
def product3(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64:
    """Form a finite-factor product without intermediate exponent overflow/underflow."""
    ...


@wp.func_native("""
if (a == 0.0 || b == 0.0 || c == 0.0 || d == 0.0) return 0.0;
int ea, eb, ec, ed;
double ma = frexp(a, &ea);
double mb = frexp(b, &eb);
double mc = frexp(c, &ec);
double md = frexp(d, &ed);
return ldexp(((ma * mb) * mc) * md, ea + eb + ec + ed);
""")
def product4(a: wp.float64, b: wp.float64, c: wp.float64, d: wp.float64) -> wp.float64:
    """A derivative product has its own range, independent of a rounded primal score."""
    ...


@wp.func_native("""
if (a == 0.0 || b == 0.0 || c == 0.0 || d == 0.0) return 0.0;
int ea, eb, ec, ed;
const double ma = frexp(a, &ea);
const double mb = frexp(b, &eb);
const double mc = frexp(c, &ec);
const double md = frexp(d, &ed);
const double mantissa = ((ma * mb) * mc) * md;
// Four finite binary64 factors cannot rescue attenuation above this bound.
// Check before converting the optical depth to an integer exponent.
if (tau > 8192.0) return copysign(0.0, mantissa);
const int attenuation_exponent = static_cast<int>(floor(tau * 1.4426950408889634074));
// Split ln(2) and one FMA retain the residual when tau is near 800 or larger.
const double residual = fma(double(attenuation_exponent), 0.69314718055994530942, -tau)
    + double(attenuation_exponent) * 2.3190468138462995584e-17;
return ldexp(mantissa * exp(residual), ea + eb + ec + ed - attenuation_exponent);
""")
def attenuated_product4(
    tau: wp.float64, a: wp.float64, b: wp.float64, c: wp.float64, d: wp.float64
) -> wp.float64:
    """Multiply original factors by exp(-tau) before the final binary64 rounding.

    Finite factors and nonnegative finite optical depth are checked by callers.
    An underflowed attenuation or forward score never determines derivative range.
    """
    ...


@wp.struct
class Parameters:
    origin: wp.vec3d
    spacing: wp.vec3d
    shape: wp.vec3i
    detector_lower: wp.vec2d
    detector_spacing: wp.vec2d
    detector_shape: wp.vec2i
    detector_z: wp.float64
    source_energy: wp.float64
    energy_bins: int
    max_events: int
    max_crossings: int
    max_angle_trials: int
    compton: int
    energy_score: int
    continuous_absorption: int


@wp.struct
class Trace:
    pixel: int
    events: int
    status: int
    energy: wp.float64
    score: wp.float64
    density_score: wp.float64
    absorption_depth: wp.float64


@wp.struct
class GridEntry:
    position: wp.vec3d
    cell: wp.vec3i
    alive: int
    status: int


@wp.struct
class FaceCrossing:
    position: wp.vec3d
    cell: wp.vec3i
    inside: int


@wp.func
def detector_pixel(position: wp.vec3d, direction: wp.vec3d, p: Parameters) -> int:
    pixel = int(MISS_PIXEL)
    if direction[2] > wp.float64(0.0):
        distance = (p.detector_z - position[2]) / direction[2]
        if not wp.isfinite(distance):
            return INVALID_PIXEL
        if distance >= wp.float64(0.0):
            x = position[0] + distance * direction[0] - p.detector_lower[0]
            y = position[1] + distance * direction[1] - p.detector_lower[1]
            if not wp.isfinite(x) or not wp.isfinite(y):
                return INVALID_PIXEL
            # Test support before converting potentially huge floating coordinates.
            if (
                x >= wp.float64(0.0)
                and y >= wp.float64(0.0)
                and x < wp.float64(p.detector_shape[0]) * p.detector_spacing[0]
                and y < wp.float64(p.detector_shape[1]) * p.detector_spacing[1]
            ):
                # Physical support has already been tested; quotient rounding
                # at its upper edge must not alias a neighbouring row.
                column = wp.min(int(wp.floor(x / p.detector_spacing[0])), p.detector_shape[0] - 1)
                row = wp.min(int(wp.floor(y / p.detector_spacing[1])), p.detector_shape[1] - 1)
                pixel = row * p.detector_shape[0] + column
    return pixel


@wp.func
def cell_index(position: wp.vec3d, direction: wp.vec3d, p: Parameters) -> wp.vec3i:
    cell = wp.vec3i(0)
    for axis in range(3):
        coordinate = (position[axis] - p.origin[axis]) / p.spacing[axis]
        index = int(wp.floor(coordinate))
        # A point on a face belongs to the downstream cell. No epsilon displaces
        # a physical path or silently changes a thin voxel's optical thickness.
        if direction[axis] < wp.float64(0.0) and coordinate == wp.float64(index):
            index -= 1
        cell[axis] = wp.clamp(index, 0, p.shape[axis] - 1)
    return cell


@wp.func
def coefficients(
    material: int,
    energy: wp.float64,
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    p: Parameters,
) -> wp.vec2d:
    offset = material * p.energy_bins
    if p.energy_bins == 1:
        return wp.vec2d(absorption[offset], scattering[offset])
    lower = int(0)
    upper = p.energy_bins - 1
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if energies[middle] <= energy:
            lower = middle
        else:
            upper = middle
    fraction = (energy - energies[lower]) / (energies[upper] - energies[lower])
    return wp.vec2d(
        (wp.float64(1.0) - fraction) * absorption[offset + lower]
        + fraction * absorption[offset + upper],
        (wp.float64(1.0) - fraction) * scattering[offset + lower]
        + fraction * scattering[offset + upper],
    )


@wp.func
def scatter_direction(direction: wp.vec3d, cosine: wp.float64, azimuth: wp.float64) -> wp.vec3d:
    # Choose the auxiliary axis away from collinearity; polar singularities do
    # not justify dividing by sin(theta) of the incoming direction.
    auxiliary = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    if wp.abs(direction[2]) > wp.float64(0.9):
        auxiliary = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
    tangent = wp.normalize(wp.cross(auxiliary, direction))
    bitangent = wp.cross(direction, tangent)
    sine = wp.sqrt(wp.max(wp.float64(0.0), wp.float64(1.0) - cosine * cosine))
    return wp.normalize(
        cosine * direction + sine * (wp.cos(azimuth) * tangent + wp.sin(azimuth) * bitangent)
    )


# region book:transport-compton-law
@wp.func
def compton_scatter(
    energy: wp.float64,
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    max_trials: int,
) -> wp.vec4d:
    """Cosine, azimuth, outgoing energy and status from the conditional free-electron law."""
    result = wp.vec4d(wp.float64(0.0), wp.float64(0.0), energy, wp.float64(6.0))
    for trial in range(max_trials):
        angular = random4(seed, history, event, wp.uint32(RANDOM_NAMESPACE_BIT) + wp.uint32(trial))
        cosine = wp.float64(2.0) * angular[0] - wp.float64(1.0)
        ratio = wp.float64(1.0) / (
            wp.float64(1.0) + energy / ELECTRON_REST_ENERGY_KEV * (wp.float64(1.0) - cosine)
        )
        # Klein-Nishina density divided by its envelope 2. This conditional
        # scattering law is independent of the active material-density scale.
        acceptance = wp.float64(0.5) * (
            ratio * ratio * ratio + ratio - ratio * ratio * (wp.float64(1.0) - cosine * cosine)
        )
        if angular[1] < acceptance:
            result = wp.vec4d(
                cosine,
                TWO_PI * angular[2],
                energy * ratio,
                wp.float64(0.0),
            )
            break
    return result


# endregion book:transport-compton-law


# region book:transport-grid-traversal
@wp.func
def enter_grid(position: wp.vec3d, direction: wp.vec3d, p: Parameters) -> GridEntry:
    """Intersect a forward ray and assign its downstream cell without an epsilon."""
    result = GridEntry()
    result.status = 0
    result.cell = wp.vec3i(0)
    entry = wp.float64(0.0)
    exit_distance = wp.float64(MAX_DISTANCE)
    intersects = int(1)
    for axis in range(3):
        lower = p.origin[axis]
        upper = lower + wp.float64(p.shape[axis]) * p.spacing[axis]
        if direction[axis] == wp.float64(0.0):
            if position[axis] < lower or position[axis] >= upper:
                intersects = 0
        else:
            first = (lower - position[axis]) / direction[axis]
            second = (upper - position[axis]) / direction[axis]
            if not wp.isfinite(first) or not wp.isfinite(second):
                result.status = 4
                intersects = 0
                break
            entry = wp.max(entry, wp.min(first, second))
            exit_distance = wp.min(exit_distance, wp.max(first, second))
    if exit_distance <= entry:
        intersects = 0
    result.alive = intersects
    if result.alive != 0:
        position += entry * direction
        # Only round the intersection coordinate to a face, not the travelled
        # distance. The entry came from these same faces and cannot be outside.
        for axis in range(3):
            position[axis] = wp.clamp(
                position[axis],
                p.origin[axis],
                p.origin[axis] + wp.float64(p.shape[axis]) * p.spacing[axis],
            )
    if result.alive != 0:
        result.cell = cell_index(position, direction, p)
    result.position = position
    return result


@wp.func
def distances_to_faces(
    position: wp.vec3d, direction: wp.vec3d, cell: wp.vec3i, p: Parameters
) -> wp.vec3d:
    """Distances to the next cell face on each axis, retaining exact ties."""
    distances = wp.vec3d(MAX_DISTANCE)
    for axis in range(3):
        if direction[axis] != wp.float64(0.0):
            face_index = cell[axis]
            if direction[axis] > wp.float64(0.0):
                face_index += 1
            face = p.origin[axis] + wp.float64(face_index) * p.spacing[axis]
            distances[axis] = (face - position[axis]) / direction[axis]
    return distances


@wp.func
def cross_faces(
    position: wp.vec3d,
    direction: wp.vec3d,
    cell: wp.vec3i,
    face_distances: wp.vec3d,
    distance: wp.float64,
    p: Parameters,
) -> FaceCrossing:
    """Cross every tied face and snap only its coordinate to the known cell face."""
    result = FaceCrossing()
    result.position = position
    result.cell = cell
    result.inside = 1
    for axis in range(3):
        if face_distances[axis] == distance:
            if direction[axis] > wp.float64(0.0):
                result.cell[axis] += 1
                result.position[axis] = (
                    p.origin[axis] + wp.float64(result.cell[axis]) * p.spacing[axis]
                )
            else:
                result.position[axis] = (
                    p.origin[axis] + wp.float64(result.cell[axis]) * p.spacing[axis]
                )
                result.cell[axis] -= 1
        if result.cell[axis] < 0 or result.cell[axis] >= p.shape[axis]:
            result.inside = 0
    return result


# endregion book:transport-grid-traversal


# region book:transport-residual-flight
@wp.func
def walk(
    initial_position: wp.vec3d,
    initial_direction: wp.vec3d,
    source_weight: wp.float64,
    source_amplitude: wp.float64,
    seed: wp.uint64,
    history: wp.uint64,
    active_material: int,
    material_ids: wp.array(dtype=wp.int32),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    p: Parameters,
) -> Trace:
    result = Trace()
    result.pixel = MISS_PIXEL
    result.events = 0
    result.status = 0
    result.energy = p.source_energy
    result.score = wp.float64(0.0)
    result.density_score = wp.float64(0.0)
    result.absorption_depth = wp.float64(0.0)
    position = initial_position
    direction = initial_direction
    entry = enter_grid(position, direction, p)
    position = entry.position
    cell = entry.cell
    alive = entry.alive
    result.status = entry.status
    crossings = int(0)
    while alive != 0:
        if result.events >= p.max_events:
            result.status = 2
            break
        if result.energy < energies[0] or result.energy > energies[p.energy_bins - 1]:
            result.status = 5
            break
        draws = random4(seed, history, wp.uint32(result.events), wp.uint32(0))
        residual = -wp.log(draws[0])
        collision = int(0)
        material = int(0)
        interaction = wp.vec2d(wp.float64(0.0))
        while alive != 0 and collision == 0:
            flat = (cell[2] * p.shape[1] + cell[1]) * p.shape[0] + cell[0]
            material = material_ids[flat]
            interaction = coefficients(material, result.energy, energies, absorption, scattering, p)
            extinction = density[material] * (interaction[0] + interaction[1])
            absorption_rate = wp.float64(0.0)
            if p.continuous_absorption != 0:
                extinction = density[material] * interaction[1]
                absorption_rate = density[material] * interaction[0]
            face_distance = distances_to_faces(position, direction, cell, p)
            distance = wp.min(face_distance[0], wp.min(face_distance[1], face_distance[2]))
            if distance < wp.float64(0.0) or not wp.isfinite(distance):
                result.status = 4
                alive = 0
                break
            optical_distance = extinction * distance
            if (
                not wp.isfinite(extinction)
                or not wp.isfinite(optical_distance)
                or not wp.isfinite(absorption_rate)
            ):
                result.status = 4
                alive = 0
                break
            absorption_distance = wp.float64(0.0)
            if p.continuous_absorption != 0:
                segment_distance = distance
                if extinction > wp.float64(0.0) and residual < optical_distance:
                    segment_distance = residual / extinction
                # Use the realised segment, not the entire distance to the face:
                # scattering can interrupt this segment before the face is reached.
                absorption_distance = absorption_rate * segment_distance
                result.absorption_depth += absorption_distance
                if not wp.isfinite(result.absorption_depth):
                    result.status = 4
                    alive = 0
                    break
            if extinction > wp.float64(0.0) and residual < optical_distance:
                distance = residual / extinction
                position += distance * direction
                if material == active_material:
                    result.density_score += wp.float64(1.0) - residual
                    if p.continuous_absorption != 0:
                        result.density_score -= absorption_distance
                collision = 1
            else:
                if material == active_material:
                    result.density_score -= optical_distance
                    if p.continuous_absorption != 0:
                        result.density_score -= absorption_distance
                residual -= optical_distance
                position += distance * direction
                crossing = cross_faces(position, direction, cell, face_distance, distance, p)
                position = crossing.position
                cell = crossing.cell
                alive = crossing.inside
                crossings += 1
                if alive != 0 and crossings >= p.max_crossings:
                    result.status = 3
                    alive = 0
        if collision != 0:
            result.events += 1
            absorption_probability = wp.float64(0.0)
            if p.continuous_absorption == 0:
                absorption_probability = interaction[0] / (interaction[0] + interaction[1])
            if p.continuous_absorption == 0 and draws[1] < absorption_probability:
                result.status = 1
                alive = 0
            else:
                cosine = wp.float64(2.0) * draws[2] - wp.float64(1.0)
                azimuth = TWO_PI * draws[3]
                if p.compton != 0:
                    angular = compton_scatter(
                        result.energy,
                        seed,
                        history,
                        wp.uint32(result.events - 1),
                        p.max_angle_trials,
                    )
                    cosine = angular[0]
                    azimuth = angular[1]
                    result.energy = angular[2]
                    if angular[3] != wp.float64(0.0):
                        result.status = 6
                        alive = 0
                if alive != 0:
                    direction = scatter_direction(direction, cosine, azimuth)
    if result.status == 0:
        result.pixel = detector_pixel(position, direction, p)
        if result.pixel == INVALID_PIXEL:
            result.pixel = MISS_PIXEL
            result.status = 4
        if result.pixel >= 0:
            score_energy = wp.float64(1.0)
            if p.energy_score != 0:
                score_energy = result.energy
            result.score = product3(source_weight, source_amplitude, score_energy)
            if p.continuous_absorption != 0:
                result.score = attenuated_product4(
                    result.absorption_depth,
                    source_weight,
                    source_amplitude,
                    score_energy,
                    wp.float64(1.0),
                )
    if not wp.isfinite(result.score) or not wp.isfinite(result.density_score):
        result.status = 4
    return result


# endregion book:transport-residual-flight


@wp.kernel
def validate_model(
    material_ids: wp.array(dtype=wp.int32),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    materials: int,
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    if index < material_ids.shape[0]:
        if material_ids[index] < 0 or material_ids[index] >= materials:
            wp.atomic_or(status, 0, 1)
    if index < absorption.shape[0]:
        a = absorption[index]
        s = scattering[index]
        if (
            not wp.isfinite(a)
            or not wp.isfinite(s)
            or a < wp.float64(0.0)
            or s < wp.float64(0.0)
            or not wp.isfinite(a + s)
        ):
            wp.atomic_or(status, 0, 1)


@wp.kernel
def validate_sources(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    detector_z: wp.float64,
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    if index < positions.shape[0]:
        position = positions[index]
        direction = directions[index]
        for axis in range(3):
            if not wp.isfinite(position[axis]) or not wp.isfinite(direction[axis]):
                wp.atomic_or(status, 0, 1)
        if (
            wp.abs(wp.dot(direction, direction) - wp.float64(1.0)) > wp.float64(1.0e-12)
            or not wp.isfinite(weights[index])
            or weights[index] < wp.float64(0.0)
            or position[2] >= detector_z
        ):
            wp.atomic_or(status, 0, 1)
    if index < density.shape[0]:
        if not wp.isfinite(density[index]) or density[index] <= wp.float64(0.0):
            wp.atomic_or(status, 0, 1)


@wp.kernel
def trace_histories(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: Parameters,
    seed: wp.uint64,
    first_history: wp.uint64,
    source_amplitude: wp.float64,
    out_pixel: wp.array(dtype=wp.int32),
    out_score: wp.array(dtype=wp.float64),
    out_energy: wp.array(dtype=wp.float64),
    out_events: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    result = walk(
        positions[index],
        directions[index],
        weights[index],
        source_amplitude,
        seed,
        first_history + wp.uint64(index),
        SOURCE_AMPLITUDE,
        material_ids,
        energies,
        absorption,
        scattering,
        density,
        parameters,
    )
    out_pixel[index] = result.pixel
    out_score[index] = result.score
    if not wp.isfinite(out_score[index]):
        wp.atomic_or(status, 0, 16)
    out_energy[index] = result.energy
    out_events[index] = result.events
    out_status[index] = result.status
    if result.status >= 2:
        wp.atomic_or(status, 0, 1 << result.status)


# region book:transport-density-score
@wp.kernel
def derivative_histories(
    positions: wp.array(dtype=wp.vec3d),
    directions: wp.array(dtype=wp.vec3d),
    weights: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    material_ids: wp.array(dtype=wp.int32),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    parameters: Parameters,
    seed: wp.uint64,
    first_history: wp.uint64,
    active_material: int,
    source_amplitude: wp.float64,
    out_pixel: wp.array(dtype=wp.int32),
    out_derivative: wp.array(dtype=wp.float64),
    out_status: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    amplitude = source_amplitude
    base_weight = weights[index]
    if active_material >= 0:
        # A density derivative is formed directly below. An unused primal
        # overflow/underflow must not determine its representability.
        base_weight = wp.float64(0.0)
    if active_material == SOURCE_AMPLITUDE:
        amplitude = wp.float64(1.0)
    result = walk(
        positions[index],
        directions[index],
        base_weight,
        amplitude,
        seed,
        first_history + wp.uint64(index),
        active_material,
        material_ids,
        energies,
        absorption,
        scattering,
        density,
        parameters,
    )
    out_pixel[index] = result.pixel
    derivative = wp.float64(0.0)
    if active_material >= 0 and result.pixel >= 0:
        score_energy = wp.float64(1.0)
        if parameters.energy_score != 0:
            score_energy = result.energy
        derivative = product4(weights[index], source_amplitude, score_energy, result.density_score)
        if parameters.continuous_absorption != 0:
            derivative = attenuated_product4(
                result.absorption_depth,
                weights[index],
                source_amplitude,
                score_energy,
                result.density_score,
            )
    if active_material == SOURCE_AMPLITUDE:
        # d(a * base_score)/da, including at a=0; never divide by amplitude.
        derivative = result.score
    if active_material == LOG_SOURCE_AMPLITUDE:
        derivative = result.score
    out_derivative[index] = derivative
    out_status[index] = result.status
    if result.status >= 2:
        wp.atomic_or(status, 0, 1 << result.status)
    if not wp.isfinite(derivative):
        wp.atomic_or(status, 0, 16)


# endregion book:transport-density-score


@wp.kernel
def centred_moments(
    pixel: wp.array(dtype=wp.int32),
    score: wp.array(dtype=wp.float64),
    mean: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    out_variance: wp.array(dtype=wp.float64),
    out_hits: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    history = wp.tid()
    target = pixel[history]
    participating = int(0)
    contribution = wp.float64(0.0)
    if target >= 0 and target < mean.shape[0]:
        residual = wp.float64(0.0)
        if scale[target] > wp.float64(0.0):
            value = score[history]
            average = mean[target]
            if (value >= wp.float64(0.0)) == (average >= wp.float64(0.0)):
                residual = (value - average) / scale[target]
            else:
                residual = value / scale[target] - average / scale[target]
        contribution = residual * residual
        if not wp.isfinite(contribution):
            wp.atomic_or(status, 0, 16)
        else:
            participating = 1
    aggregate = grouped_tally(contribution, target, 0, participating)
    if aggregate[1] > wp.float64(0.0):
        wp.atomic_add(out_variance, target, aggregate[0])
        wp.atomic_add(out_hits, target, int(aggregate[1]))


@wp.kernel
def finish_variance(
    mean: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    hits: wp.array(dtype=wp.int32),
    count: int,
    out_variance: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    n = wp.float64(count)
    missing = count - hits[pixel]
    centred = out_variance[pixel]
    if missing > 0 and scale[pixel] > wp.float64(0.0):
        residual = mean[pixel] / scale[pixel]
        centred += wp.float64(missing) * residual * residual
    # Accumulate normalised squares before restoring their exponent. Squaring
    # each tiny history first can turn a representable aggregate variance to zero.
    sigma = scale[pixel] * wp.sqrt(centred / n / (n - wp.float64(1.0)))
    variance = sigma * sigma
    out_variance[pixel] = variance
    if missing < 0 or not wp.isfinite(variance):
        wp.atomic_or(status, 0, 16)


@wp.kernel
def independent_product(
    mean_a: wp.array(dtype=wp.float64),
    other_b: wp.array(dtype=wp.float64),
    observed: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    loss_mode: int,
    out_components: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    a = mean_a[pixel]
    b = other_b[pixel]
    y = observed[pixel]
    w = weights[pixel]
    if (
        not wp.isfinite(a)
        or not wp.isfinite(b)
        or not wp.isfinite(y)
        or not wp.isfinite(w)
        or w < wp.float64(0.0)
    ):
        wp.atomic_or(status, 0, 1)
    value = w * (a - y) * b
    if loss_mode != 0:
        value = wp.float64(0.5) * w * (a - y) * (b - y)
    out_components[pixel] = value
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, 16)


@wp.kernel
def sample_parallel_source(
    lower: wp.vec3d,
    extent: wp.vec2d,
    direction: wp.vec3d,
    weight: wp.float64,
    seed: wp.uint64,
    first_history: wp.uint64,
    out_position: wp.array(dtype=wp.vec3d),
    out_direction: wp.array(dtype=wp.vec3d),
    out_weight: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    # The high event bit separates source draws from every supported collision.
    draw = random4(
        seed, first_history + wp.uint64(index), wp.uint32(RANDOM_NAMESPACE_BIT), wp.uint32(0)
    )
    out_position[index] = lower + wp.vec3d(
        extent[0] * draw[0], extent[1] * draw[1], wp.float64(0.0)
    )
    out_direction[index] = direction
    out_weight[index] = weight


@wp.kernel
def update_density_chart(
    parameters: wp.array(dtype=wp.float64),
    material_parameter: wp.array(dtype=wp.int32),
    base_density: wp.array(dtype=wp.float64),
    out_density: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    material = wp.tid()
    active = material_parameter[material]
    value = base_density[material]
    if active >= 0:
        value = wp.exp(parameters[active])
    out_density[material] = value
    if not wp.isfinite(value) or value <= wp.float64(0.0):
        wp.atomic_or(status, 0, 16)


@wp.kernel
def find_tally_scale(
    pixel: wp.array(dtype=wp.int32),
    score: wp.array(dtype=wp.float64),
    history_status: wp.array(dtype=wp.int32),
    pixels: int,
    out_scale: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    history = wp.tid()
    target = pixel[history]
    value = score[history]
    participating = int(0)
    terminal = history_status[history]
    if terminal >= 2 and terminal <= 6:
        wp.atomic_or(status, 0, 1 << terminal)
    elif (
        target < MISS_PIXEL
        or target >= pixels
        or not wp.isfinite(value)
        or terminal < 0
        or terminal > 6
        or (target == MISS_PIXEL and value != wp.float64(0.0))
        or (terminal == 1 and target != MISS_PIXEL)
    ):
        wp.atomic_or(status, 0, 1)
    elif target >= 0:
        participating = 1
    aggregate = grouped_tally(wp.abs(value), target, 1, participating)
    if aggregate[1] > wp.float64(0.0):
        wp.atomic_max(out_scale, target, aggregate[0])


@wp.kernel
def tally_sum(
    pixel: wp.array(dtype=wp.int32),
    score: wp.array(dtype=wp.float64),
    history_status: wp.array(dtype=wp.int32),
    pixels: int,
    scale: wp.array(dtype=wp.float64),
    out_sum: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    history = wp.tid()
    target = pixel[history]
    participating = int(0)
    contribution = wp.float64(0.0)
    if (
        history_status[history] <= 1
        and history_status[history] >= 0
        and target >= 0
        and target < pixels
        and scale[target] > wp.float64(0.0)
    ):
        contribution = score[history] / scale[target]
        if not wp.isfinite(contribution):
            wp.atomic_or(status, 0, 16)
        else:
            participating = 1
    aggregate = grouped_tally(contribution, target, 0, participating)
    if aggregate[1] > wp.float64(0.0):
        wp.atomic_add(out_sum, target, aggregate[0])


@wp.kernel
def finish_mean(
    sums: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    count: int,
    out_mean: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    # Scale after dividing the bounded sum: neither a huge raw sum nor a
    # subnormal score divided prematurely by the history count is required.
    value = scale[pixel] * (sums[pixel] / wp.float64(count))
    out_mean[pixel] = value
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, 16)


@wp.kernel
def validate_measurement(
    observation: wp.array(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    pixel = wp.tid()
    if (
        not wp.isfinite(observation[pixel])
        or not wp.isfinite(weights[pixel])
        or weights[pixel] < wp.float64(0.0)
    ):
        wp.atomic_or(status, 0, 1)
