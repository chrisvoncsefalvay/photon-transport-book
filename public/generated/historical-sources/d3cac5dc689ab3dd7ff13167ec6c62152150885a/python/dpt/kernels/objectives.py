"""Fused objective/seed evaluation with a fixed FP64 reduction tree.

The image-space derivative is caller-owned. Only one scalar partial per tile is
retained: no per-pixel loss image or global floating-point atomic is needed.
"""

# Warp annotations are executable DSL expressions; host interfaces remain strict.
# The optional GPU import is resolved only when an operator is prepared.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

from functools import cache

import warp as wp

TILE = 256
OPTIONS = {"fast_math": False, "fuse_fp": True, "enable_backward": False}


@wp.func_native("return log1p(x);")
def log_one_plus(x: wp.float64) -> wp.float64: ...


@cache
def observation_validation_kernel(poisson: bool, weighted: bool, masked: bool = False):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def validate(
        observation: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        valid: wp.array(dtype=wp.uint8),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        active = bool(True)  # noqa: UP018 - Warp runtime Boolean.
        if wp.static(masked):
            if valid[p] > wp.uint8(1):
                wp.atomic_or(status, 0, 1)
            active = valid[p] == wp.uint8(1)
        if active:
            value = observation[p]
            invalid = not wp.isfinite(value)
            if wp.static(poisson):
                invalid = invalid or value < 0.0 or wp.floor(value) != value
            if wp.static(weighted):
                invalid = invalid or not wp.isfinite(weights[p]) or weights[p] < 0.0
            if invalid:
                wp.atomic_or(status, 0, 1)

    return validate


@cache
def validation_kernel(poisson: bool, weighted: bool, masked: bool = False):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def validate(
        prediction: wp.array(dtype=wp.float32),
        observation: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        valid: wp.array(dtype=wp.uint8),
        status: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        active = bool(True)  # noqa: UP018 - Warp runtime Boolean.
        if wp.static(masked):
            if valid[p] > wp.uint8(1):
                wp.atomic_or(status, 0, 1)
            active = valid[p] == wp.uint8(1)
        if active:
            a, b = prediction[p], observation[p]
            invalid = not wp.isfinite(a) or not wp.isfinite(b)
            if wp.static(poisson):
                invalid = invalid or a < 0.0 or b < 0.0 or wp.floor(b) != b
                invalid = invalid or (a == 0.0 and b > 0.0)
            if wp.static(weighted):
                invalid = invalid or not wp.isfinite(weights[p]) or weights[p] < 0.0
            if invalid:
                wp.atomic_or(status, 0, 1)

    return validate


# region book:objective-loss-and-seed
@cache
def evaluation_kernel(poisson: bool, weighted: bool, gradient: bool, masked: bool = False):
    @wp.kernel(module="unique", module_options=OPTIONS)
    def evaluate(
        prediction: wp.array(dtype=wp.float32),
        observation: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        valid: wp.array(dtype=wp.uint8),
        size: int,
        normalisation: wp.float64,
        seed: wp.array(dtype=wp.float32),
        partials: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
    ):
        block, lane = wp.tid()
        p = block * TILE + lane
        loss = wp.float64(0.0)
        if p < size:
            active = bool(True)  # noqa: UP018 - Warp runtime Boolean.
            if wp.static(masked):
                if valid[p] > wp.uint8(1):
                    wp.atomic_or(status, 0, 1)
                active = valid[p] == wp.uint8(1)
            if active:
                predicted = wp.float64(prediction[p])
                observed = wp.float64(observation[p])
                weight = normalisation
                if wp.static(weighted):
                    weight *= wp.float64(weights[p])
                difference = predicted - observed
                derivative = difference
                loss = wp.float64(0.5) * difference * difference
                if wp.static(poisson):
                    if observed == wp.float64(0.0):
                        loss = predicted
                        derivative = wp.float64(1.0)
                    else:
                        derivative = difference / predicted
                        # Half-deviance differs from the negative log likelihood
                        # only by observation constants. This form retains a small
                        # residual near equality without evaluating log(0) for y=0.
                        if wp.abs(difference) <= wp.float64(0.25) * observed:
                            ratio = difference / observed
                            if wp.abs(ratio) < wp.float64(0.001):
                                # r-log1p(r) also cancels. Through degree six the
                                # omitted relative term is < 3e-16 for |r|<1e-3.
                                series = wp.float64(0.5) + ratio * (
                                    (-wp.float64(1.0) / wp.float64(3.0))
                                    + ratio
                                    * (
                                        wp.float64(0.25)
                                        + ratio
                                        * (
                                            wp.float64(-0.2)
                                            + ratio * (wp.float64(1.0) / wp.float64(6.0))
                                        )
                                    )
                                )
                                loss = observed * ratio * ratio * series
                            else:
                                loss = observed * (ratio - log_one_plus(ratio))
                        else:
                            loss = difference + observed * (wp.log(observed) - wp.log(predicted))
                loss *= weight
                if not wp.isfinite(loss):
                    wp.atomic_or(status, 0, 2)
                if wp.static(gradient):
                    result = wp.float32(weight * derivative)
                    seed[p] = result
                    if not wp.isfinite(result):
                        wp.atomic_or(status, 0, 2)
            elif wp.static(gradient):
                seed[p] = wp.float32(0.0)
        total = wp.tile_sum(wp.tile(loss))
        wp.tile_store(partials, total, offset=block)

    return evaluate


# endregion book:objective-loss-and-seed
