"""Double-precision coordinate algebra shared by deterministic CUDA operators."""

# Warp annotations are executable DSL expressions; host interfaces remain strict.
# The optional GPU import is resolved only when an operator is prepared.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

import warp as wp


@wp.func
def unpack_rotation(pose: wp.array(dtype=wp.float64)) -> wp.mat33d:
    return wp.mat33d(
        pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], pose[7], pose[8]
    )


@wp.func
def unpack_translation(pose: wp.array(dtype=wp.float64)) -> wp.vec3d:
    return wp.vec3d(pose[9], pose[10], pose[11])


@wp.kernel(enable_backward=False)
def validate_pose(pose: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    rotation = unpack_rotation(pose)
    product = wp.transpose(rotation) * rotation
    for i in range(12):
        if not wp.isfinite(pose[i]):
            wp.atomic_or(status, 0, 1)
    for i in range(3):
        for j in range(3):
            expected = wp.float64(0.0)
            if i == j:
                expected = wp.float64(1.0)
            if wp.abs(product[i, j] - expected) > wp.float64(1e-10):
                wp.atomic_or(status, 0, 1)
    if wp.determinant(rotation) < wp.float64(0.0):
        wp.atomic_or(status, 0, 1)
