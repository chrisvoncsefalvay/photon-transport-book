"""Sampled scalar-field metadata; a volume is not a bag of tissue labels.

The sample buffer is contiguous, x-fastest (nz, ny, nx), in inverse mm for
attenuation or dimensionless for an explicitly declared material fraction.
No CT-to-attenuation conversion is inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from dpt.contracts import ContractError
from dpt.geometry import IDENTITY, Matrix3, Vector3, matvec, rotation_matrix, transpose, vector3


# region book:volume-sampling-contract
@dataclass(frozen=True, slots=True)
class GridSpec:
    """Oriented anisotropic grid, with origin at the first sample centre.

    Support extends half a sample spacing outside the outer centres. Within
    that support interpolation clamps to the outer sample; beyond it the field
    is zero. Nonzero boundary samples therefore produce a discontinuous field.
    """

    shape: tuple[int, int, int]
    spacing_mm: Vector3
    origin_mm: Vector3 = (0.0, 0.0, 0.0)
    orientation: Matrix3 = IDENTITY

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(type(x) is not int or x < 1 for x in self.shape):
            raise ContractError("grid shape is (nz, ny, nx), each a positive integer")
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "spacing_mm", vector3(self.spacing_mm, "spacing_mm"))
        object.__setattr__(self, "origin_mm", vector3(self.origin_mm, "origin_mm"))
        object.__setattr__(self, "orientation", rotation_matrix(self.orientation, "orientation"))
        if min(self.spacing_mm) <= 0:
            raise ContractError("grid spacing must be positive")
        if self.voxels > 2**31 - 1:
            raise ContractError("grid exceeds signed 32-bit indexing")

    @property
    def voxels(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def object_to_grid(self, point_mm: Vector3) -> Vector3:
        displacement: Vector3 = cast(
            Vector3, tuple(point_mm[i] - self.origin_mm[i] for i in range(3))
        )
        aligned = matvec(transpose(self.orientation), displacement)
        return cast(Vector3, tuple(aligned[i] / self.spacing_mm[i] for i in range(3)))

    def grid_to_object(self, index: Vector3) -> Vector3:
        scaled: Vector3 = cast(Vector3, tuple(index[i] * self.spacing_mm[i] for i in range(3)))
        aligned = matvec(self.orientation, scaled)
        return cast(Vector3, tuple(aligned[i] + self.origin_mm[i] for i in range(3)))

    @property
    def support(self) -> tuple[Vector3, Vector3]:
        return (-0.5, -0.5, -0.5), (self.shape[2] - 0.5, self.shape[1] - 0.5, self.shape[0] - 0.5)


# endregion book:volume-sampling-contract
