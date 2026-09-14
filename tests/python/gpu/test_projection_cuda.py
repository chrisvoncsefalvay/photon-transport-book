"""Real-device gates for the discrete projector and its first-order adjoints."""

# Pytest 9.1 exposes approx() with incomplete annotations; production modules stay strict.
# pyright: reportUnknownMemberType=false

import importlib
import math
import struct
from typing import Any

import pytest

from dpt.autodiff import FirstOrderPass
from dpt.contracts import ContractError
from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


def _field() -> tuple[GridSpec, list[float]]:
    angle = 0.2
    grid = GridSpec(
        (3, 4, 5),
        (1.2, 0.8, 1.1),
        (-2.0, -1.0, -1.0),
        (
            math.cos(angle),
            -math.sin(angle),
            0.0,
            math.sin(angle),
            math.cos(angle),
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )
    values = [
        struct.unpack(
            "f", struct.pack("f", 0.07 + 0.012 * x + 0.019 * y + 0.011 * z + 0.003 * x * y)
        )[0]
        for z in range(3)
        for y in range(4)
        for x in range(5)
    ]
    return grid, values


def _geometry(shape: tuple[int, int]) -> DetectorGeometry:
    return DetectorGeometry(
        (-12.0, 0.31, 0.43),
        (12.0, -0.6, -0.6),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.071, 0.11),
        shape,
    )


@pytest.mark.parametrize("shape", [(1, 1), (1, 129), (9, 17)])
def test_oriented_anisotropic_values_against_independent_midpoint(shape: tuple[int, int]) -> None:
    grid, values = _field()
    geometry = _geometry(shape)
    transform = compose_pose(RigidTransform(), (0.13, -0.08, 0.07, 0.02, -0.03, 0.01))
    workspace = prepare_projection(grid, geometry, ProjectionSpec(47))
    mu = wp.array(values, dtype=wp.float32, device="cuda:0")
    pose = wp.array(transform.packed(), dtype=wp.float64, device="cuda:0")
    output = wp.empty(geometry.pixels, dtype=wp.float32, device="cuda:0")
    project_optical_depth(mu, pose, workspace=workspace, out_L=output)
    expected = [
        integrate_sampled_field(values, grid, geometry, transform, row, column, midpoint_samples=47)
        for row in range(shape[0])
        for column in range(shape[1])
    ]
    assert output.numpy().tolist() == pytest.approx(expected, rel=2e-6, abs=1e-8)


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_pose_vjp_includes_active_box_faces_and_six_right_coordinates(integration: Any) -> None:
    grid, values = _field()
    geometry = _geometry((9, 17))
    transform = compose_pose(RigidTransform(), (0.13, -0.08, 0.07, 0.02, -0.03, 0.01))
    workspace = prepare_projection(
        grid, geometry, ProjectionSpec(47, precision="float64", integration=integration)
    )
    mu = wp.array(values, dtype=wp.float32, device="cuda:0")
    pose = wp.array(transform.packed(), dtype=wp.float64, device="cuda:0")
    seed_values = [(-1.0) ** p * (0.5 + (p % 7) / 8) for p in range(geometry.pixels)]
    seeds = wp.array(seed_values, dtype=workspace.dtype, device="cuda:0")
    gradient = wp.empty(6, dtype=wp.float64, device="cuda:0")
    projection_vjp(mu, pose, workspace=workspace, adj_L=seeds, out_pose=gradient)
    measured = gradient.numpy().tolist()

    def objective(current: RigidTransform) -> float:
        return math.fsum(
            seed_values[row * geometry.shape[1] + column]
            * integrate_sampled_field(
                values,
                grid,
                geometry,
                current,
                row,
                column,
                midpoint_samples=47 if integration == "midpoint" else None,
            )
            for row in range(geometry.shape[0])
            for column in range(geometry.shape[1])
        )

    for axis in range(6):
        errors: list[float] = []
        for step in (1e-4, 3e-5, 1e-5, 3e-6):
            delta = tuple(step if i == axis else 0.0 for i in range(6))
            plus = objective(compose_pose(transform, delta))
            minus = objective(compose_pose(transform, tuple(-x for x in delta)))
            errors.append(abs((plus - minus) / (2 * step) - measured[axis]))
        assert min(errors) < 2e-6 * max(1.0, abs(measured[axis])), (axis, errors, measured)
    # Fixed tree and immutable inputs should reproduce the same binary64 sum.
    projection_vjp(mu, pose, workspace=workspace, adj_L=seeds, out_pose=gradient)
    assert gradient.numpy().tolist() == measured


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_volume_adjoint_satisfies_linearity_inner_product(integration: Any) -> None:
    grid, values = _field()
    geometry = _geometry((3, 4))
    workspace = prepare_projection(
        grid, geometry, ProjectionSpec(31, active_volume=True, integration=integration)
    )
    mu = wp.array(values, dtype=wp.float32, device="cuda:0")
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    output = wp.empty(geometry.pixels, dtype=wp.float32, device="cuda:0")
    seed_values = [(-1.0) ** p * 0.5 for p in range(geometry.pixels)]
    seeds = wp.array(seed_values, dtype=wp.float32, device="cuda:0")
    gradient = wp.empty(grid.voxels, dtype=wp.float32, device="cuda:0")
    project_optical_depth(mu, pose, workspace=workspace, out_L=output)
    projection_vjp(mu, pose, workspace=workspace, adj_L=seeds, out_mu=gradient)
    lhs = math.fsum(float(x) * seed_values[i] for i, x in enumerate(output.numpy()))
    rhs = math.fsum(float(x) * values[i] for i, x in enumerate(gradient.numpy()))
    assert lhs == pytest.approx(rhs, rel=2e-5, abs=1e-7)


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_tape_storage_adjoint_maps_to_local_right_tangent(
    precision: Any,
    integration: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(wp.config, "verify_autograd_array_access", True)
    grid, values = _field()
    geometry = _geometry((2, 3))
    transform = compose_pose(RigidTransform(), (0.1, -0.2, 0.3, 0.2, -0.1, 0.05))
    workspace = prepare_projection(
        grid, geometry, ProjectionSpec(31, precision=precision, integration=integration)
    )
    mu = wp.array(values, dtype=wp.float32, device="cuda:0")
    pose = wp.array(transform.packed(), dtype=wp.float64, device="cuda:0", requires_grad=True)
    output = wp.empty(geometry.pixels, dtype=workspace.dtype, device="cuda:0", requires_grad=True)
    seed = wp.ones(geometry.pixels, dtype=workspace.dtype, device="cuda:0")
    with FirstOrderPass([workspace]) as reverse:
        project_optical_depth(mu, pose, workspace=workspace, out_L=output, tape=reverse.tape)
    with pytest.raises(ContractError, match="outstanding"):
        project_optical_depth(mu, pose, workspace=workspace, out_L=output)
    reverse.backward(output, seed)
    captured = capsys.readouterr()
    assert "may produce incorrect gradients" not in captured.out + captured.err
    storage = pose.grad.numpy().tolist()
    local = wp.empty(6, dtype=wp.float64, device="cuda:0")
    projection_vjp(mu, pose, workspace=workspace, adj_L=seed, out_pose=local)
    translation = tuple(
        sum(transform.rotation[3 * j + i] * storage[9 + j] for j in range(3)) for i in range(3)
    )
    product = [
        sum(transform.rotation[3 * k + i] * storage[3 * k + j] for k in range(3))
        for i in range(3)
        for j in range(3)
    ]
    mapped = (
        *translation,
        product[7] - product[5],
        product[2] - product[6],
        product[3] - product[1],
    )
    assert local.numpy().tolist() == pytest.approx(mapped, rel=2e-12, abs=1e-12)
    reverse.close()
    assert pose.grad.numpy().tolist() == [0.0] * 12


def test_invalid_field_rejected_before_output_writes() -> None:
    grid, values = _field()
    geometry = _geometry((2, 3))
    workspace = prepare_projection(grid, geometry, ProjectionSpec(31))
    values[4] = -1.0
    mu = wp.array(values, dtype=wp.float32, device="cuda:0")
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    output = wp.full(geometry.pixels, 123.0, dtype=wp.float32, device="cuda:0")
    with pytest.raises(ContractError, match="non-negative"):
        project_optical_depth(mu, pose, workspace=workspace, out_L=output)
    assert output.numpy().tolist() == [123.0] * geometry.pixels


def test_alias_stream_and_higher_order_contracts() -> None:
    grid, values = _field()
    geometry = _geometry((3, 20))
    workspace = prepare_projection(grid, geometry, ProjectionSpec(31))
    mu = wp.array(values, dtype=wp.float32, device="cuda:0")
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    with pytest.raises(ContractError, match="overlaps"):
        project_optical_depth(mu, pose, workspace=workspace, out_L=mu)
    output = wp.empty(geometry.pixels, dtype=wp.float32, device="cuda:0")
    with pytest.raises(ContractError, match="stream"):
        project_optical_depth(
            mu, pose, workspace=workspace, out_L=output, stream=wp.Stream("cuda:0")
        )
    with wp.Tape(), pytest.raises(ContractError, match="explicitly"):
        project_optical_depth(mu, pose, workspace=workspace, out_L=output)
    gradient = wp.empty(6, dtype=wp.float64, device="cuda:0")
    with wp.Tape(), pytest.raises(ContractError, match="ambient"):
        projection_vjp(mu, pose, workspace=workspace, adj_L=output, out_pose=gradient)


def test_derived_coordinate_overflow_is_rejected_before_sampling() -> None:
    from dpt.contracts import NumericalError

    grid, values = _field()
    geometry = _geometry((2, 3))
    with pytest.raises(ContractError, match="coordinate map"):
        prepare_projection(GridSpec((1, 1, 1), (1e-320, 1.0, 1.0)), geometry, ProjectionSpec(4))
    workspace = prepare_projection(grid, geometry, ProjectionSpec(4))
    field = wp.array(values, dtype=wp.float32, device="cuda:0")
    packed = (*RigidTransform().rotation, 1.7e308, -1.7e308, -1.7e308)
    pose = wp.array(packed, dtype=wp.float64, device="cuda:0")
    output = wp.empty(geometry.pixels, dtype=wp.float32, device="cuda:0")
    with pytest.raises(NumericalError, match="non-finite"):
        project_optical_depth(field, pose, workspace=workspace, out_L=output)


@pytest.mark.parametrize("small", [2.0**-100, 2.0**-140])
def test_low_attenuation_sample_centre_survives_high_contrast_neighbour(small: float) -> None:
    # At the right sample centre the left column has exactly zero weight.
    # Difference-form lerp a + t*(b-a) would lose b when t=1 and a/b is huge.
    grid = GridSpec((2, 2, 2), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    geometry = DetectorGeometry(
        (1.0, 0.5, -5.0), (1.0, 0.5, 5.0), (1, 0, 0), (0, 1, 0), (1.0, 1.0), (1, 1)
    )
    workspace = prepare_projection(grid, geometry, ProjectionSpec(16))
    mu = wp.array([2.0**100, small] * 4, dtype=wp.float32, device="cuda:0")
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    output = wp.empty(1, dtype=wp.float32, device="cuda:0")
    project_optical_depth(mu, pose, workspace=workspace, out_L=output)
    assert float(output.numpy()[0]) == 2.0 * small


@pytest.mark.parametrize("axis", [1, 2])
def test_small_transverse_slope_survives_large_orthogonal_background(axis: int) -> None:
    grid = GridSpec((2, 2, 2), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    if axis == 1:
        values = [2.0**100, 1.0, 2.0**100, 2.0] * 2
        source, centre, column, row = (0.5, 0.5, -1), (0.5, 0.5, 2), (1, 0, 0), (0, 1, 0)
    else:
        values = [2.0**100, 1.0] * 2 + [2.0**100, 2.0] * 2
        source, centre, column, row = (0.5, -1, 0.5), (0.5, 2, 0.5), (1, 0, 0), (0, 0, 1)
    geometry = DetectorGeometry(source, centre, column, row, (1.0, 1.0), (1, 1))
    workspace = prepare_projection(grid, geometry, ProjectionSpec(16))
    mu = wp.array(values, dtype=wp.float32, device="cuda:0")
    pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cuda:0")
    seed = wp.ones(1, dtype=wp.float32, device="cuda:0")
    gradient = wp.empty(6, dtype=wp.float64, device="cuda:0")
    projection_vjp(mu, pose, adj_L=seed, out_pose=gradient, workspace=workspace)
    # The transverse field slope is (1-fx)*(H-H) + fx*(2-1) = 1/2.
    # Translation moves the object against that slope for a two-mm segment.
    assert float(gradient.numpy()[axis]) == -1.0
