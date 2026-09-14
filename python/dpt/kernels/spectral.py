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


@cache
def get_value_check(wide: bool, nonnegative: bool):
    dtype = wp.float64 if wide else wp.float32

    @wp.kernel(module="unique", module_options=STRICT)
    def check(values: wp.array(dtype=dtype), status: wp.array(dtype=wp.int32)):
        index = wp.tid()
        invalid = not wp.isfinite(values[index])
        if wp.static(nonnegative):
            invalid = invalid or values[index] < dtype(0.0)
        if invalid:
            wp.atomic_or(status, 0, 1)

    return check


check_nonnegative = get_value_check(False, True)
check_finite = get_value_check(False, False)


@cache
def get_signal_store(wide: bool):
    dtype = wp.float64 if wide else wp.float32

    @wp.func
    def store(value: wp.float64, status: wp.array(dtype=wp.int32)) -> dtype:
        result = dtype(value)
        if not wp.isfinite(result):
            wp.atomic_or(status, 0, 2)
        return result

    return store


@cache
def get_optical_depth(wide: bool):
    dtype = wp.float64 if wide else wp.float32

    @wp.func
    def depth_at(
        paths: wp.array(dtype=dtype),
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

    return depth_at


@cache
def get_attenuation(wide: bool):
    @wp.func
    def attenuate(depth: wp.float64, status: wp.array(dtype=wp.int32)) -> wp.float64:
        factor = wp.exp(-depth)
        if wp.static(wide):
            # A zero exponential may hide a representable weighted tail. This
            # explicit range boundary avoids silently disagreeing with the VJP.
            if factor == wp.float64(0.0):
                wp.atomic_or(status, 0, 4)
        return factor

    return attenuate


@cache
def get_product(wide: bool):
    @wp.func
    def product(a: wp.float64, b: wp.float64, status: wp.array(dtype=wp.int32)) -> wp.float64:
        result = a * b
        if wp.static(wide):
            # A later large factor could restore a representable value. Reject
            # loss of an intermediate instead of silently returning a zero VJP.
            if a != wp.float64(0.0) and b != wp.float64(0.0) and result == wp.float64(0.0):
                wp.atomic_or(status, 0, 4)
        return result

    return product


# region book:spectral-primary-sum
@cache
def get_forward(
    materials: int, energies: int, shared_weights: bool, shared_response: bool, wide: bool = False
):
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide)
    attenuate = get_attenuation(wide)
    product = get_product(wide)
    store_signal = get_signal_store(wide)

    @wp.kernel(module="unique", module_options=STRICT)
    def forward(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        pixels: int,
        mean: wp.array(dtype=dtype),
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
            contribution = product(
                wp.float64(weights[wi]) * wp.float64(response[ri]), attenuate(depth, status), status
            )
            # All terms are non-negative, but compensation retains small bins
            # when a broad response places many decades in the same sum.
            corrected = contribution - compensation
            updated = total + corrected
            compensation = (updated - total) - corrected
            total = updated
        mean[pixel] = store_signal(total, status)

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
    wide: bool = False,
):
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide)
    attenuate = get_attenuation(wide)
    product = get_product(wide)
    store_signal = get_signal_store(wide)
    gradient_vector = wp.types.vector(length=materials, dtype=wp.float64)

    @wp.kernel(module="unique", module_options=STRICT)
    def vjp(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=dtype),
        pixels: int,
        grad_paths: wp.array(dtype=dtype),
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
            weighted_seed = product(wp.float64(seed[pixel]), attenuate(depth, status), status)
            if wp.static(write_weights):
                grad_weights[wi] = checked_store(
                    product(weighted_seed, wp.float64(response[ri]), status), status
                )
            if wp.static(write_response):
                grad_response[ri] = checked_store(
                    product(weighted_seed, wp.float64(weights[wi]), status), status
                )
            if wp.static(write_paths):
                common = product(
                    product(weighted_seed, wp.float64(weights[wi]), status),
                    wp.float64(response[ri]),
                    status,
                )
                for material in range(materials):
                    term = product(
                        -common, wp.float64(coefficients[material * energies + energy]), status
                    )
                    corrected = term - path_compensation[material]
                    updated = path_gradient[material] + corrected
                    path_compensation[material] = (updated - path_gradient[material]) - corrected
                    path_gradient[material] = updated
        if wp.static(write_paths):
            for material in range(materials):
                grad_paths[material * pixels + pixel] = store_signal(
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
    wide: bool = False,
):
    """kind=0 weights or 1 response; coefficients have their own shared kernel."""
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide)
    attenuate = get_attenuation(wide)
    product = get_product(wide)
    if kind not in (0, 1):
        raise ValueError("shared partial kind must select weights or response")

    @wp.kernel(module="unique", module_options=STRICT)
    def partials(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=dtype),
        pixels: int,
        groups: int,
        output: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
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
            term = product(wp.float64(seed[pixel]), attenuate(depth, status), status)
            if wp.static(kind == 0):
                term = product(term, wp.float64(response[ri]), status)
            else:
                term = product(term, wp.float64(weights[wi]), status)
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
    materials: int, energies: int, shared_weights: bool, shared_response: bool, wide: bool = False
):
    """Reuse each energy's depth across its material cotangents, without a P*K tensor."""
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide)
    attenuate = get_attenuation(wide)
    product = get_product(wide)
    accumulator = wp.types.vector(length=materials, dtype=wp.float64)

    @wp.kernel(module="unique", module_options=STRICT)
    def coefficient_partials(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        response: wp.array(dtype=wp.float32),
        seed: wp.array(dtype=dtype),
        pixels: int,
        groups: int,
        output: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
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
            common = product(-wp.float64(seed[pixel]), attenuate(depth, status), status)
            common = product(
                product(common, wp.float64(weights[wi]), status), wp.float64(response[ri]), status
            )
            for material in range(materials):
                term = product(common, wp.float64(paths[material * pixels + pixel]), status)
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
def get_bin_counts(
    materials: int, energies: int, shared_weights: bool, shared_efficiency: bool, wide: bool = False
):
    dtype = wp.float64 if wide else wp.float32
    optical_depth = get_optical_depth(wide)
    attenuate = get_attenuation(wide)
    product = get_product(wide)
    store_signal = get_signal_store(wide)

    @wp.kernel(module="unique", module_options=STRICT)
    def counts(
        paths: wp.array(dtype=dtype),
        coefficients: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        efficiency: wp.array(dtype=wp.float32),
        pixels: int,
        output: wp.array(dtype=dtype),
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
        rate = product(
            wp.float64(weights[wi]) * wp.float64(efficiency[ri]), attenuate(depth, status), status
        )
        output[energy * pixels + p] = store_signal(rate, status)

    return counts
