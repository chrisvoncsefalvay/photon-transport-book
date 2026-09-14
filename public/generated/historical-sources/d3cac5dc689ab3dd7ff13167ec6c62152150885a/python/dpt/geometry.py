"""Right-handed rigid geometry in millimetres, with explicit object-to-world maps.

These immutable host contracts import without Warp. Rotations are row-major
matrices acting on column vectors; an acquisition origin is its first pixel
centre. Small pose transfers belong at the optimisation control boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from dpt.contracts import ContractError, finite_tuple

type Vector3 = tuple[float, float, float]
type Matrix3 = tuple[float, float, float, float, float, float, float, float, float]
IDENTITY: Matrix3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def vector3(value: Vector3, name: str) -> Vector3:
    return cast(Vector3, finite_tuple(value, name, minimum=None, length=3))


def dot(a: Vector3, b: Vector3) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def cross(a: Vector3, b: Vector3) -> Vector3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def matvec(matrix: Matrix3, vector: Vector3) -> Vector3:
    return cast(
        Vector3, tuple(sum(matrix[3 * i + j] * vector[j] for j in range(3)) for i in range(3))
    )


def transpose(matrix: Matrix3) -> Matrix3:
    return cast(Matrix3, tuple(matrix[3 * j + i] for i in range(3) for j in range(3)))


def matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    return cast(
        Matrix3,
        tuple(
            sum(left[3 * i + k] * right[3 * k + j] for k in range(3))
            for i in range(3)
            for j in range(3)
        ),
    )


def rotation_matrix(value: Matrix3, name: str = "rotation") -> Matrix3:
    value = cast(Matrix3, finite_tuple(value, name, minimum=None, length=9))
    product = matmul(transpose(value), value)
    if max(abs(a - b) for a, b in zip(product, IDENTITY, strict=True)) > 1e-10:
        raise ContractError(f"{name} must be orthonormal to 1e-10")
    columns = [(value[i], value[i + 3], value[i + 6]) for i in range(3)]
    if dot(cross(columns[0], columns[1]), columns[2]) < 0.0:
        raise ContractError(f"{name} must be right-handed")
    return cast(Matrix3, tuple(float(x) for x in value))


# region book:geometry-rigid-transform
@dataclass(frozen=True, slots=True)
class RigidTransform:
    """T_WO maps object points into world coordinates; translation is in mm."""

    rotation: Matrix3 = IDENTITY
    translation_mm: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", rotation_matrix(self.rotation))
        object.__setattr__(self, "translation_mm", vector3(self.translation_mm, "translation"))

    def point(self, point_mm: Vector3) -> Vector3:
        rotated = matvec(self.rotation, point_mm)
        return cast(Vector3, tuple(rotated[i] + self.translation_mm[i] for i in range(3)))

    def direction(self, direction: Vector3) -> Vector3:
        return matvec(self.rotation, direction)

    def inverse(self) -> RigidTransform:
        inverse_rotation = transpose(self.rotation)
        translation = matvec(inverse_rotation, self.translation_mm)
        return RigidTransform(inverse_rotation, cast(Vector3, tuple(-x for x in translation)))

    def compose(self, right: RigidTransform) -> RigidTransform:
        """Return self @ right; the right-hand map is applied first."""
        return RigidTransform(
            matmul(self.rotation, right.rotation), self.point(right.translation_mm)
        )

    def packed(self) -> tuple[float, ...]:
        """Twelve float64 device values: row-major rotation, then translation."""
        return (*self.rotation, *self.translation_mm)


# endregion book:geometry-rigid-transform


@dataclass(frozen=True, slots=True)
class DetectorGeometry:
    """Finite source-to-pixel segments; u increases columns and v increases rows."""

    source_mm: Vector3
    origin_mm: Vector3
    u: Vector3
    v: Vector3
    spacing_mm: tuple[float, float]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        for name in ("source_mm", "origin_mm", "u", "v"):
            object.__setattr__(self, name, vector3(getattr(self, name), name))
        if abs(dot(self.u, self.u) - 1.0) > 1e-10 or abs(dot(self.v, self.v) - 1.0) > 1e-10:
            raise ContractError("detector basis vectors must have unit length")
        if abs(dot(self.u, self.v)) > 1e-10:
            raise ContractError("detector basis vectors must be perpendicular")
        spacing = finite_tuple(self.spacing_mm, "pixel spacing", positive=True, length=2)
        if len(self.shape) != 2 or any(type(x) is not int or x <= 0 for x in self.shape):
            raise ContractError("detector shape is (height, width), both positive integers")
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "spacing_mm", spacing)
        offset: Vector3 = cast(
            Vector3, tuple(self.source_mm[i] - self.origin_mm[i] for i in range(3))
        )
        if abs(dot(offset, cross(self.u, self.v))) <= 1e-12:
            raise ContractError("source must lie outside the detector plane")
        if self.pixels > 2**31 - 1:
            raise ContractError("detector exceeds signed 32-bit indexing")

    @property
    def pixels(self) -> int:
        return self.shape[0] * self.shape[1]

    def pixel_centre(self, row: int, column: int) -> Vector3:
        if type(row) is not int or type(column) is not int:
            raise ContractError("pixel indices must be integers")
        if not 0 <= row < self.shape[0] or not 0 <= column < self.shape[1]:
            raise ContractError("pixel index outside detector")
        return cast(
            Vector3,
            tuple(
                self.origin_mm[i]
                + column * self.spacing_mm[0] * self.u[i]
                + row * self.spacing_mm[1] * self.v[i]
                for i in range(3)
            ),
        )


def _skew(vector: Vector3) -> Matrix3:
    x, y, z = vector
    return (0.0, -z, y, z, 0.0, -x, -y, x, 0.0)


def _coefficients(theta2: float) -> tuple[float, float, float, float, float]:
    # Series avoid cancellation in both the exponential and its derivative.
    if not math.isfinite(theta2):
        raise ContractError("rotation norm exceeds the finite geometry range")
    if theta2 < 1e-2:
        a = 1 - theta2 / 6 + theta2**2 / 120 - theta2**3 / 5040
        b = 0.5 - theta2 / 24 + theta2**2 / 720 - theta2**3 / 40320
        c = 1 / 6 - theta2 / 120 + theta2**2 / 5040 - theta2**3 / 362880
        db = -1 / 12 + theta2 / 180 - theta2**2 / 6720 + theta2**3 / 453600
        dc = -1 / 60 + theta2 / 1260 - theta2**2 / 60480 + theta2**3 / 4989600
        return a, b, c, db, dc
    theta = math.sqrt(theta2)
    sin, cos = math.sin(theta), math.cos(theta)
    return (
        sin / theta,
        (1 - cos) / theta2,
        (theta - sin) / (theta2 * theta),
        (theta * sin - 2 * (1 - cos)) / theta2**2,
        (3 * sin - theta * (2 + cos)) / (theta2**2 * theta),
    )


# region book:geometry-right-se3-update
def compose_pose(anchor: RigidTransform, increment: tuple[float, ...]) -> RigidTransform:
    """Apply anchor @ exp(xi^); xi=(tx,ty,tz,rx,ry,rz), mm and radians.

    Translation is the Lie algebra coordinate, not a separately added world
    displacement. A solver keeps anchor fixed until it discards curvature history.
    """
    increment = finite_tuple(increment, "pose increment", minimum=None, length=6)
    translation: Vector3 = (increment[0], increment[1], increment[2])
    rotation: Vector3 = (increment[3], increment[4], increment[5])
    omega = _skew(rotation)
    omega2 = matmul(omega, omega)
    a, b, c, _, _ = _coefficients(dot(rotation, rotation))
    matrix: Matrix3 = cast(
        Matrix3, tuple(IDENTITY[i] + a * omega[i] + b * omega2[i] for i in range(9))
    )
    velocity: Matrix3 = cast(
        Matrix3, tuple(IDENTITY[i] + b * omega[i] + c * omega2[i] for i in range(9))
    )
    return anchor.compose(RigidTransform(matrix, matvec(velocity, translation)))


# endregion book:geometry-right-se3-update


def right_jacobian_se3(increment: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    """Map fixed-anchor chart perturbations to local right perturbations.

    exp(xi+d)^ ~= exp(xi^) exp((J_r(xi)d)^). This is an analytic
    derivative of Rodrigues' formula, including translation/rotation coupling.
    """
    transform = compose_pose(RigidTransform(), increment)
    rho: Vector3 = (increment[0], increment[1], increment[2])
    rotation: Vector3 = (increment[3], increment[4], increment[5])
    omega = _skew(rotation)
    omega2 = matmul(omega, omega)
    _, b, c, db, dc = _coefficients(dot(rotation, rotation))
    rt = transpose(transform.rotation)
    velocity: Matrix3 = cast(
        Matrix3, tuple(IDENTITY[i] + b * omega[i] + c * omega2[i] for i in range(9))
    )
    translation_block = matmul(rt, velocity)
    rotation_block = tuple(IDENTITY[i] - b * omega[i] + c * omega2[i] for i in range(9))
    coupling: list[Vector3] = []
    for axis in range(3):
        basis: Vector3 = cast(Vector3, tuple(float(j == axis) for j in range(3)))
        derivative = _skew(basis)
        left, right = matmul(derivative, omega), matmul(omega, derivative)
        dv: Matrix3 = cast(
            Matrix3,
            tuple(
                db * rotation[axis] * omega[i]
                + b * derivative[i]
                + dc * rotation[axis] * omega2[i]
                + c * (left[i] + right[i])
                for i in range(9)
            ),
        )
        coupling.append(matvec(rt, matvec(dv, rho)))
    return tuple(
        tuple(
            translation_block[3 * i + j]
            if i < 3 and j < 3
            else coupling[j - 3][i]
            if i < 3
            else rotation_block[3 * (i - 3) + (j - 3)]
            if j >= 3
            else 0.0
            for j in range(6)
        )
        for i in range(6)
    )


def chart_gradient(
    increment: tuple[float, ...], right_gradient: tuple[float, ...]
) -> tuple[float, ...]:
    right_gradient = finite_tuple(right_gradient, "right tangent gradient", minimum=None, length=6)
    jacobian = right_jacobian_se3(increment)
    return tuple(sum(jacobian[i][j] * right_gradient[i] for i in range(6)) for j in range(6))
