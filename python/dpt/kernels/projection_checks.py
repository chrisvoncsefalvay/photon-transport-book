"""Small projection checks and stores, sharing the default CUDA block size.

Keep these separate from the 128-lane quadrature module: Warp specialises a
whole module for each block size, including kernels not used at that size.
"""

# Warp DSL annotations are executable type expressions.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false

from functools import cache

import warp as wp

wp.set_module_options({"fast_math": False, "fuse_fp": True, "enable_backward": False})


@wp.kernel
def validate_nonnegative(array: wp.array(dtype=wp.float32), status: wp.array(dtype=wp.int32)):
    index = wp.tid()
    if not wp.isfinite(array[index]) or array[index] < wp.float32(0.0):
        wp.atomic_or(status, 0, 1)


@wp.kernel
def validate_finite(array: wp.array(dtype=wp.float32), status: wp.array(dtype=wp.int32)):
    if not wp.isfinite(array[wp.tid()]):
        wp.atomic_or(status, 0, 2)


@wp.kernel(module="unique", module_options={"enable_backward": True})
def dependency_marker(
    field: wp.array(dtype=wp.float32),
    pose: wp.array(dtype=wp.float64),
    output: wp.array(dtype=wp.float32),
):
    """Zero-dimensional tape bookkeeping only; never launches a device thread."""
    pass


@cache
def get_validate_finite(double: bool = False):
    dtype = wp.float64 if double else wp.float32

    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def check(array: wp.array(dtype=dtype), status: wp.array(dtype=wp.int32)):
        if not wp.isfinite(array[wp.tid()]):
            wp.atomic_or(status, 0, 2)

    return check


@cache
def get_dependency_marker(double: bool = False):
    dtype = wp.float64 if double else wp.float32

    @wp.kernel(module="unique", module_options={"enable_backward": True})
    def marker(
        field: wp.array(dtype=wp.float32),
        pose: wp.array(dtype=wp.float64),
        output: wp.array(dtype=dtype),
    ):
        pass

    return marker
