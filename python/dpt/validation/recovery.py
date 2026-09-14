"""Independent geometric recovery metrics; objective decrease is a separate fact.

These CPU reports consume explicit small landmark sets and rigid transforms.
They never estimate a reference pose from the image used by the optimiser.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from dpt.contracts import ContractError, NumericalError
from dpt.geometry import DetectorGeometry, RigidTransform, Vector3, vector3


@dataclass(frozen=True, slots=True)
class PoseError:
    translation_mm: float
    rotation_radians: float
    landmark_rms_mm: float
    landmark_max_mm: float
    landmarks: int


# region book:recovery-geometric-metrics
def pose_error(
    recovered: RigidTransform,
    reference: RigidTransform,
    landmarks_object_mm: Sequence[Vector3],
) -> PoseError:
    """Report origin displacement, geodesic rotation and held-out target errors.

    A translation norm depends on the chosen object origin. Landmark errors also
    expose rotation about that origin, so they are reported rather than hidden
    inside a single weighted pose norm with arbitrary mixed units.
    """
    if not landmarks_object_mm:
        raise ContractError("recovery metrics need at least one declared landmark")
    # Relative rotation R_reference.T @ R_recovered, independent scalar indexing.
    relative = tuple(
        math.fsum(reference.rotation[3 * k + i] * recovered.rotation[3 * k + j] for k in range(3))
        for i in range(3)
        for j in range(3)
    )
    cosine = max(-1.0, min(1.0, (relative[0] + relative[4] + relative[8] - 1.0) / 2.0))
    sine = 0.5 * math.hypot(
        relative[7] - relative[5], relative[2] - relative[6], relative[3] - relative[1]
    )
    distances: list[float] = []
    for landmark in landmarks_object_mm:
        point = vector3(landmark, "landmark")
        a, b = recovered.point(point), reference.point(point)
        distances.append(math.hypot(*(x - y for x, y in zip(a, b, strict=True))))
    translation = math.hypot(
        *(a - b for a, b in zip(recovered.translation_mm, reference.translation_mm, strict=True))
    )
    rms = math.hypot(*(value / math.sqrt(len(distances)) for value in distances))
    if not math.isfinite(translation) or not math.isfinite(rms):
        raise NumericalError("recovery metric exceeds the finite coordinate range")
    return PoseError(translation, math.atan2(sine, cosine), rms, max(distances), len(distances))


# endregion book:recovery-geometric-metrics


def detector_coordinate(geometry: DetectorGeometry, point_world_mm: Vector3) -> tuple[float, float]:
    """Continuous (column, row) of a point's perspective projection.

    Coordinates outside the detector remain meaningful residuals. A point on
    the source-parallel plane or behind the source is rejected; no clipped pixel
    is substituted for an undefined projection.
    """
    point = vector3(point_world_mm, "world landmark")
    u, v = geometry.u, geometry.v
    normal = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    offset = tuple(o - s for o, s in zip(geometry.origin_mm, geometry.source_mm, strict=True))
    ray = tuple(p - s for p, s in zip(point, geometry.source_mm, strict=True))
    plane_distance = math.fsum(a * b for a, b in zip(offset, normal, strict=True))
    direction = math.fsum(a * b for a, b in zip(ray, normal, strict=True))
    if direction == 0 or not math.isfinite(direction):
        raise ContractError("landmark projection is undefined on a source-parallel plane")
    scale = plane_distance / direction
    if not math.isfinite(scale) or scale <= 0:
        raise ContractError("landmark must project onto the forward detector plane")
    hit = tuple(scale * r - o for r, o in zip(ray, offset, strict=True))
    coordinate = (
        math.fsum(a * b for a, b in zip(hit, u, strict=True)) / geometry.spacing_mm[0],
        math.fsum(a * b for a, b in zip(hit, v, strict=True)) / geometry.spacing_mm[1],
    )
    if not all(map(math.isfinite, coordinate)):
        raise NumericalError("projected landmark exceeds the finite pixel range")
    return coordinate


def reprojection_rms_pixels(
    recovered: RigidTransform,
    reference: RigidTransform,
    geometry: DetectorGeometry,
    landmarks_object_mm: Sequence[Vector3],
) -> float:
    if not landmarks_object_mm:
        raise ContractError("reprojection metrics need at least one declared landmark")
    differences: list[float] = []
    for landmark in landmarks_object_mm:
        point = vector3(landmark, "landmark")
        a = detector_coordinate(geometry, recovered.point(point))
        b = detector_coordinate(geometry, reference.point(point))
        differences.extend(x - y for x, y in zip(a, b, strict=True))
    result = math.hypot(*(x / math.sqrt(len(landmarks_object_mm)) for x in differences))
    if not math.isfinite(result):
        raise NumericalError("reprojection residual exceeds the finite pixel range")
    return result
