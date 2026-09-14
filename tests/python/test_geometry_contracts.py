"""Geometry contracts and independent finite-step SE(3) checks; execution deferred."""

# Pytest 9.1 exposes approx() with incomplete annotations; production modules stay strict.
# pyright: reportUnknownMemberType=false

import math
from typing import Any, cast

import pytest

from dpt.contracts import ContractError
from dpt.geometry import (
    IDENTITY,
    DetectorGeometry,
    RigidTransform,
    chart_gradient,
    compose_pose,
    right_jacobian_se3,
)
from dpt.projection import ProjectionSpec
from dpt.volumes import GridSpec


def test_rigid_inverse_and_composition_direction() -> None:
    transform = RigidTransform((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (2.0, 3.0, 4.0))
    assert transform.point((1.0, 2.0, 3.0)) == (0.0, 4.0, 7.0)
    assert transform.inverse().point(transform.point((1.0, 2.0, 3.0))) == pytest.approx(
        (1.0, 2.0, 3.0)
    )
    assert transform.compose(transform.inverse()).packed() == pytest.approx(
        (*IDENTITY, 0.0, 0.0, 0.0)
    )


def test_reflection_and_nonorthogonal_pose_are_rejected() -> None:
    with pytest.raises(ValueError, match="right-handed"):
        RigidTransform((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="orthonormal"):
        RigidTransform((2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


def test_detector_origin_is_first_centre_and_not_array_corner() -> None:
    geometry = DetectorGeometry(
        (0.0, 0.0, -100.0), (10.0, 20.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.2, 0.7), (4, 6)
    )
    assert geometry.pixel_centre(0, 0) == (10.0, 20.0, 0.0)
    assert geometry.pixel_centre(3, 5) == pytest.approx((11.0, 22.1, 0.0))
    assert geometry.pixels == 24
    with pytest.raises(ValueError):
        geometry.pixel_centre(4, 0)


def test_detector_source_must_be_off_plane() -> None:
    with pytest.raises(ValueError, match="outside"):
        DetectorGeometry(
            (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0), (2, 2)
        )


def test_oriented_anisotropic_grid_and_half_cell_support() -> None:
    grid = GridSpec(
        (3, 4, 5),
        (2.0, 3.0, 4.0),
        (10.0, 20.0, 30.0),
        (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    assert grid.grid_to_object((1.0, 2.0, 0.5)) == (4.0, 22.0, 32.0)
    assert grid.object_to_grid((4.0, 22.0, 32.0)) == (1.0, 2.0, 0.5)
    assert grid.support == ((-0.5, -0.5, -0.5), (4.5, 3.5, 2.5))
    assert grid.voxels == 60


@pytest.mark.parametrize("spacing", [(0.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (math.nan, 1.0, 1.0)])
def test_grid_invalid_spacing(spacing: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        GridSpec((2, 2, 2), spacing)


def test_right_composition_keeps_translation_in_object_frame() -> None:
    anchor = RigidTransform((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    moved = compose_pose(anchor, (2.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert moved.translation_mm == pytest.approx((0.0, 2.0, 0.0))


def test_se3_translation_uses_left_jacobian_of_rotation() -> None:
    moved = compose_pose(RigidTransform(), (1.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2))
    assert moved.translation_mm == pytest.approx((2 / math.pi, 2 / math.pi, 0.0))


@pytest.mark.parametrize("rotation", [0.0, 1e-8, 0.001, 0.099, 0.101, 0.7])
def test_right_jacobian_matches_independent_pose_perturbation(rotation: float) -> None:
    chart = (0.7, -0.2, 0.3, rotation, -0.4 * rotation, 0.2 * rotation)
    current = compose_pose(RigidTransform(), chart)
    jacobian = right_jacobian_se3(chart)
    step = 1e-6
    for axis in range(6):
        plus = tuple(chart[i] + (step if i == axis else 0.0) for i in range(6))
        minus = tuple(chart[i] - (step if i == axis else 0.0) for i in range(6))
        forward = current.inverse().compose(compose_pose(RigidTransform(), plus))
        backward = current.inverse().compose(compose_pose(RigidTransform(), minus))
        linear = tuple(
            (forward.translation_mm[i] - backward.translation_mm[i]) / (2 * step) for i in range(3)
        )
        # vee((Rplus-Rminus)/(2h)) at identity; no canonical Jacobian helper.
        angular = tuple(
            (
                (forward.rotation[i] - forward.rotation[j])
                - (backward.rotation[i] - backward.rotation[j])
            )
            / (4 * step)
            for i, j in ((7, 5), (2, 6), (3, 1))
        )
        assert (*linear, *angular) == pytest.approx(
            tuple(jacobian[i][axis] for i in range(6)), rel=2e-6, abs=1e-9
        )


def test_chart_gradient_includes_translation_rotation_coupling() -> None:
    chart = (2.0, -1.0, 0.5, 0.2, -0.3, 0.1)
    right = (1.0, 2.0, 3.0, -0.5, 0.7, 1.0)
    jacobian = right_jacobian_se3(chart)
    mapped = chart_gradient(chart, right)
    direction = (0.1, -0.2, 0.3, 0.4, -0.1, 0.2)
    left = sum(mapped[j] * direction[j] for j in range(6))
    right_product = sum(
        right[i] * sum(jacobian[i][j] * direction[j] for j in range(6)) for i in range(6)
    )
    assert left == pytest.approx(right_product)
    assert mapped != right


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_quadrature_requires_positive_integer_count(count: object) -> None:
    with pytest.raises(ValueError):
        ProjectionSpec(cast(Any, count))


@pytest.mark.parametrize("value", [True, "1", complex(1, 0), float("inf"), float("nan")])
def test_geometry_rejects_invalid_coordinates_with_shared_contract_error(value: object) -> None:
    with pytest.raises(ContractError):
        RigidTransform(translation_mm=(cast(Any, value), 0.0, 0.0))
    with pytest.raises(ContractError):
        compose_pose(RigidTransform(), (cast(Any, value), 0.0, 0.0, 0.0, 0.0, 0.0))
