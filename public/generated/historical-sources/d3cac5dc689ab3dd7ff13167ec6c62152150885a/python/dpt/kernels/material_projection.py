"""Joint material-fraction validation; projection physics stays in projection.py."""

# Warp annotations are executable DSL expressions; host interfaces remain strict.
# The optional GPU import is resolved only when an operator is prepared.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

import warp as wp

FRACTION_SUM_LIMIT = wp.constant(wp.float64(1.0 + 2.0**-24))


@wp.kernel(enable_backward=False)
def validate_fractions(
    fractions: wp.array(dtype=wp.float32),
    materials: int,
    voxels: int,
    status: wp.array(dtype=wp.int32),
):
    voxel = wp.tid()
    total = wp.float64(0.0)
    for material in range(materials):
        fraction = fractions[material * voxels + voxel]
        if not wp.isfinite(fraction) or fraction < wp.float32(0.0):
            wp.atomic_or(status, 0, 1)
        total += wp.float64(fraction)
    # Independently rounding a nonnegative partition to binary32 can increase
    # its sum by at most unit roundoff (2^-24). Accumulate in FP64 and allow
    # this representation error once, independent of the material count. Do
    # not renormalise fields or admit a whole binary32 ulp above unit mass.
    if total > FRACTION_SUM_LIMIT:
        wp.atomic_or(status, 0, 1)


@wp.kernel(enable_backward=False)
def validate_seeds(seeds: wp.array(dtype=wp.float32), status: wp.array(dtype=wp.int32)):
    if not wp.isfinite(seeds[wp.tid()]):
        wp.atomic_or(status, 0, 1)


@wp.kernel(enable_backward=False)
def validate_nonnegative(
    fields: wp.array(dtype=wp.float32),
    materials: int,
    voxels: int,
    status: wp.array(dtype=wp.int32),
):
    voxel = wp.tid()
    for material in range(materials):
        value = fields[material * voxels + voxel]
        if not wp.isfinite(value) or value < wp.float32(0.0):
            wp.atomic_or(status, 0, 1)
