"""Spectral primary signal and explicit first-order products.

Paths are material-major (M,P); coefficients are (M,K); weights/response are
shared (K) or energy-major (K,P). FP64 registers hold depths, products and sums.
There is no pixel-by-energy retained tensor. Global parameter gradients use
bounded split reductions; only diagnostic flags use atomics.
"""

# Warp annotations are executable DSL expressions; host interfaces remain strict.
# The optional GPU import is resolved only when an operator is prepared.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

from functools import cache

import warp as wp

STRICT = {"fast_math": False, "fuse_fp": True, "enable_backward": False}
wp.set_module_options(STRICT)
TILE = 256


@wp.func
def checked_store(value: wp.float64, status: wp.array(dtype=wp.int32)) -> wp.float32:
    rounded = wp.float32(value)
    if not wp.isfinite(rounded):
        wp.atomic_or(status, 0, 2)
    return rounded


@wp.kernel
def check_nonnegative(values: wp.array(dtype=wp.float32), status: wp.array(dtype=wp.int32)):
    index = wp.tid()
    if not wp.isfinite(values[index]) or values[index] < wp.float32(0.0):
        wp.atomic_or(status, 0, 1)


@wp.kernel
def check_finite(values: wp.array(dtype=wp.float32), status: wp.array(dtype=wp.int32)):
    index = wp.tid()
    if not wp.isfinite(values[index]):
        wp.atomic_or(status, 0, 1)


@wp.func
def optical_depth(
    paths: wp.array(dtype=wp.float32),
    coefficients: wp.array(dtype=wp.float32),
    pixel: int,
    energy: int,
    pixels: int,
    materials: int,
    energies: int,
) -> wp.float64:
    depth = wp.float64(0.0)
    for material in range(materials):
        depth = depth + wp.float64(paths[material * pixels + pixel]) * wp.float64(
            coefficients[material * energies + energy]
        )
    return depth


@wp.func
def attenuate(depth: wp.float64) -> wp.float64:
    # The canonical transmission law is exp(-L). Its monoenergetic device helper
    # accepts FP32 L; narrowing a summed spectral depth before the exponential
    # would change this operator. Preserve the FP64 sum through this exponential.
    return wp.exp(-depth)


# region book:spectral-primary-sum
@cache
def get_forward(materials: int, energies: int, shared_weights: bool, shared_response: bool):
    @wp.kernel(module="unique", module_options=STRICT)
    def forward(
        paths: wp.array(dtype=wp.float32),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        pixels: int,
        mean: wp.array(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        total = wp.float64(0.0)
        compensation = wp.float64(0.0)
        for energy in range(energies):
            wi = energy * pixels + pixel
            ri = wi
            if wp.static(shared_weights):
                wi = energy
            if wp.static(shared_response):
                ri = energy
            depth = optical_depth(paths, coefficients, pixel, energy, pixels, materials, energies)
            contribution = wp.float64(weights[wi]) * wp.float64(response[ri]) * attenuate(depth)
            # All terms are non-negative, but compensation retains small bins
            # when a broad response places many decades in the same sum.
            corrected = contribution - compensation
            updated = total + corrected
            compensation = (updated - total) - corrected
            total = updated
        mean[pixel] = checked_store(total, status)

    return forward


# endregion book:spectral-primary-sum


# region book:spectral-recomputed-adjoint
@cache
def get_pixel_vjp(
    materials: int,
    energies: int,
    shared_weights: bool,
    shared_response: bool,
    write_paths: bool,
    write_weights: bool,
    write_response: bool,
):
    gradient_vector = wp.types.vector(length=materials, dtype=wp.float64)

    @wp.kernel(module="unique", module_options=STRICT)
    def vjp(
        paths: wp.array(dtype=wp.float32),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=wp.float32),
        pixels: int,
        grad_paths: wp.array(dtype=wp.float32),
        grad_weights: wp.array(dtype=wp.float32),
        grad_response: wp.array(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        pixel = wp.tid()
        path_gradient = gradient_vector()
        path_compensation = gradient_vector()
        for energy in range(energies):
            wi = energy * pixels + pixel
            ri = wi
            if wp.static(shared_weights):
                wi = energy
            if wp.static(shared_response):
                ri = energy
            depth = optical_depth(paths, coefficients, pixel, energy, pixels, materials, energies)
            weighted_seed = wp.float64(seed[pixel]) * attenuate(depth)
            if wp.static(write_weights):
                grad_weights[wi] = checked_store(weighted_seed * wp.float64(response[ri]), status)
            if wp.static(write_response):
                grad_response[ri] = checked_store(weighted_seed * wp.float64(weights[wi]), status)
            if wp.static(write_paths):
                common = weighted_seed * wp.float64(weights[wi]) * wp.float64(response[ri])
                for material in range(materials):
                    term = -common * wp.float64(coefficients[material * energies + energy])
                    corrected = term - path_compensation[material]
                    updated = path_gradient[material] + corrected
                    path_compensation[material] = (updated - path_gradient[material]) - corrected
                    path_gradient[material] = updated
        if wp.static(write_paths):
            for material in range(materials):
                grad_paths[material * pixels + pixel] = checked_store(
                    path_gradient[material], status
                )

    return vjp


# endregion book:spectral-recomputed-adjoint


@cache
def get_shared_partials(
    materials: int,
    energies: int,
    shared_weights: bool,
    shared_response: bool,
    kind: int,
):
    """kind=0 weights or 1 response; coefficients have their own shared kernel."""
    if kind not in (0, 1):
        raise ValueError("shared partial kind must select weights or response")

    @wp.kernel(module="unique", module_options=STRICT)
    def partials(
        paths: wp.array(dtype=wp.float32),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=wp.float32),
        pixels: int,
        groups: int,
        output: wp.array(dtype=wp.float64),
    ):
        group, parameter, lane = wp.tid()
        energy = parameter
        total = wp.float64(0.0)
        compensation = wp.float64(0.0)
        pixel = group * TILE + lane
        while pixel < pixels:
            wi = energy * pixels + pixel
            ri = wi
            if wp.static(shared_weights):
                wi = energy
            if wp.static(shared_response):
                ri = energy
            depth = optical_depth(paths, coefficients, pixel, energy, pixels, materials, energies)
            term = wp.float64(seed[pixel]) * attenuate(depth)
            if wp.static(kind == 0):
                term = term * wp.float64(response[ri])
            else:
                term = term * wp.float64(weights[wi])
            corrected = term - compensation
            updated = total + corrected
            compensation = (updated - total) - corrected
            total = updated
            # The final stride may exceed int32 even when every input index fits.
            # Stop before adding it; pixels - pixel is non-negative and bounded.
            if pixels - pixel <= groups * TILE:
                break
            pixel = pixel + groups * TILE
        values = wp.tile(total)
        result = wp.tile_sum(values)
        wp.tile_store(output, result, offset=parameter * groups + group)

    return partials


@cache
def get_coefficient_partials(
    materials: int, energies: int, shared_weights: bool, shared_response: bool
):
    """Reuse each energy's depth across its material cotangents, without a P*K tensor."""
    accumulator = wp.types.vector(length=materials, dtype=wp.float64)

    @wp.kernel(module="unique", module_options=STRICT)
    def coefficient_partials(
        paths: wp.array(dtype=wp.float32),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=wp.float32),
        pixels: int,
        groups: int,
        output: wp.array(dtype=wp.float64),
    ):
        group, energy, lane = wp.tid()
        totals = accumulator()
        compensations = accumulator()
        pixel = group * TILE + lane
        while pixel < pixels:
            wi = energy * pixels + pixel
            ri = wi
            if wp.static(shared_weights):
                wi = energy
            if wp.static(shared_response):
                ri = energy
            depth = optical_depth(paths, coefficients, pixel, energy, pixels, materials, energies)
            # Keep the same product order as the scalar-parameter reduction.
            common = -wp.float64(seed[pixel]) * attenuate(depth)
            common = common * wp.float64(weights[wi]) * wp.float64(response[ri])
            for material in range(materials):
                term = common * wp.float64(paths[material * pixels + pixel])
                corrected = term - compensations[material]
                updated = totals[material] + corrected
                compensations[material] = (updated - totals[material]) - corrected
                totals[material] = updated
            # The final stride may exceed int32 even when every input index fits.
            # Stop before adding it; pixels - pixel is non-negative and bounded.
            if pixels - pixel <= groups * TILE:
                break
            pixel = pixel + groups * TILE
        # Each parameter retains the original lane tree and split-sum order.
        for material in range(materials):
            values = wp.tile(totals[material])
            result = wp.tile_sum(values)
            wp.tile_store(output, result, offset=(material * energies + energy) * groups + group)

    return coefficient_partials


@cache
def get_finish_shared(output64: bool):
    """One reduction implementation with explicitly chosen scalar destination precision."""
    output_dtype = wp.float64 if output64 else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def finish_shared(
        partials: wp.array(dtype=wp.float64),
        groups: int,
        output: wp.array(dtype=output_dtype),
        status: wp.array(dtype=wp.int32),
    ):
        parameter = wp.tid()
        total = wp.float64(0.0)
        compensation = wp.float64(0.0)
        for group in range(groups):
            corrected = partials[parameter * groups + group] - compensation
            updated = total + corrected
            compensation = (updated - total) - corrected
            total = updated
        if wp.static(output64):
            output[parameter] = total
            if not wp.isfinite(total):
                wp.atomic_or(status, 0, 2)
        else:
            output[parameter] = checked_store(total, status)

    return finish_shared


finish_shared = get_finish_shared(False)


@wp.kernel
def check_probability(values: wp.array(dtype=wp.float32), status: wp.array(dtype=wp.int32)):
    p = wp.tid()
    if not wp.isfinite(values[p]) or values[p] < wp.float32(0.0) or values[p] > wp.float32(1.0):
        wp.atomic_or(status, 0, 1)


@cache
def get_bin_counts(materials: int, energies: int, shared_weights: bool, shared_efficiency: bool):
    @wp.kernel(module="unique", module_options=STRICT)
    def counts(
        paths: wp.array(dtype=wp.float32),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        efficiency: wp.array(dtype=wp.float32),
        pixels: int,
        output: wp.array(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        energy, p = wp.tid()
        wi = energy * pixels + p
        ri = wi
        if wp.static(shared_weights):
            wi = energy
        if wp.static(shared_efficiency):
            ri = energy
        depth = optical_depth(paths, coefficients, p, energy, pixels, materials, energies)
        rate = wp.float64(weights[wi]) * wp.float64(efficiency[ri]) * attenuate(depth)
        output[energy * pixels + p] = checked_store(rate, status)

    return counts
