"""Independent CPU Fisher and regularisation references for CUDA voxel scaling."""

import importlib
import math
from typing import Any, Literal

import pytest

from dpt.contracts import ContractError
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_preconditioning import prepare_fisher_scaling
from dpt.material_reconstruction import (
    MaterialReconstruction,
    MaterialReconstructionSettings,
    MaterialReconstructionView,
)
from dpt.materials import Provenance
from dpt.spectral import SpectralSpec
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

np: Any = importlib.import_module("numpy")
pytestmark = pytest.mark.gpu


def _inputs(
    *,
    beta: float = 0.0,
    missed: bool = False,
    coefficients: bool = True,
    precision: Literal["float32", "float64"] = "float32",
) -> dict[str, Any]:
    grid = GridSpec((2, 2, 3), (1.0, 1.2, 0.8), (-1.0, -0.6, -0.4))
    provenance = Provenance(
        source="Independent small-volume Beer-Lambert/Fisher test",
        sha256="0" * 64,
        rights="Original test fixture, Apache-2.0",
        description="Numerical coefficient fixture; no anatomical or physical-material claim",
    )
    spectral = SpectralSpec(
        materials=2,
        energies=2,
        coefficients_provenance=(provenance, provenance),
        spectrum_provenance=provenance,
        response_provenance=provenance,
        output_unit="counts",
        input_description="Independent numerical count channels",
        precision=precision,
    )
    geometry = DetectorGeometry(
        (-20.0, 0.0, 0.0),
        (20.0, -0.8 if not missed else 20.0, -0.4),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.8, 0.8),
        (2, 3),
    )
    mu = np.array([[0.11, 0.03], [0.02, 0.13]], dtype=np.float32)
    if not coefficients:
        mu.fill(0)
    weights = np.array([[7000, 3000], [1000, 9000]], dtype=np.float32)
    response = np.ones((2, 2), dtype=np.float32)
    counts = np.full((2, 2, 3), 8100, dtype=np.float32)
    view = MaterialReconstructionView(geometry, RigidTransform(), counts, weights, response)
    fields = np.random.default_rng(512).uniform(0.08, 0.32, (2, *grid.shape)).astype(np.float32)
    return {
        "grid": grid,
        "views": (view,),
        "coefficients": mu,
        "spectral_spec": spectral,
        "initial_fractions": fields,
        "settings": MaterialReconstructionSettings(regularisation_mm_inverse=beta),
        "samples_per_ray": 256,
        "material_metric": (2.0, 0.3, 1.0),
    }


def _reference(inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
    grid, view = inputs["grid"], inputs["views"][0]
    matrix = np.array([[2.0, 0.3], [0.3, 1.0]])
    eigenvalues, vectors = np.linalg.eigh(matrix)
    inverse_root = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    mu = inputs["coefficients"].astype(np.float64)
    fields = inputs["initial_fractions"].reshape(2, -1).astype(np.float64)
    weights = view.weights.astype(np.float64) * view.response.astype(np.float64)
    projection_rows: list[list[float]] = []
    for row in range(view.geometry.shape[0]):
        for col in range(view.geometry.shape[1]):
            integrals: list[float] = []
            for voxel in range(grid.voxels):
                basis = np.zeros(grid.voxels, dtype=np.float32)
                basis[voxel] = 1
                integrals.append(
                    integrate_sampled_field(basis, grid, view.geometry, view.pose, row, col)
                )
            projection_rows.append(integrals)
    projection = np.array(projection_rows)
    blocks: list[float] = []
    fisher = np.zeros((2 * grid.voxels, 2 * grid.voxels))
    for ray in projection:
        paths = fields @ ray
        contributions = weights * np.exp(-(paths @ mu))[None, :]
        means = contributions.sum(axis=1)
        derivatives = -(contributions @ mu.T)
        block = derivatives.T @ (derivatives / means[:, None])
        blocks.append(float(np.linalg.eigvalsh(inverse_root @ block @ inverse_root)[-1]))
        for derivative, mean in zip(derivatives, means, strict=True):
            jacobian = np.kron(derivative, ray)
            fisher += np.outer(jacobian, jacobian) / mean
    raw = projection.T @ (np.asarray(blocks) * projection.sum(axis=1))
    laplacian = np.zeros((grid.voxels, grid.voxels))
    beta = inputs["settings"].regularisation_mm_inverse
    for z in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for x in range(grid.shape[2]):
                voxel = x + grid.shape[2] * (y + grid.shape[1] * z)
                for coordinate, limit, stride, spacing in zip(
                    (x, y, z),
                    grid.shape[::-1],
                    (1, grid.shape[2], grid.shape[2] * grid.shape[1]),
                    grid.spacing_mm,
                    strict=True,
                ):
                    if coordinate + 1 < limit:
                        coefficient = beta * math.prod(grid.spacing_mm) / spacing**2
                        neighbour = voxel + stride
                        laplacian[voxel, voxel] += coefficient
                        laplacian[neighbour, neighbour] += coefficient
                        laplacian[voxel, neighbour] -= coefficient
                        laplacian[neighbour, voxel] -= coefficient
    raw += 2 * np.diag(laplacian) / eigenvalues[0]
    fisher += np.kron(np.eye(2), laplacian)
    return raw, fisher, matrix


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("beta", [0.0, 0.07])
def test_fisher_scaling_matches_independent_dense_reference_and_preserves_state(
    beta: float,
    precision: Literal["float32", "float64"],
) -> None:
    inputs = _inputs(beta=beta, precision=precision)
    solver = MaterialReconstruction(**inputs)
    loss = solver.evaluate()
    fields, gradient = solver.fractions_numpy().copy(), solver.gradient.numpy().copy()
    expected, fisher, metric = _reference(inputs)
    result = prepare_fisher_scaling(solver)
    assert result.diagnostics["path_signal_precision"] == precision
    raw = np.maximum(expected, expected.max() * 1e-6)
    np.testing.assert_allclose(result.scale.numpy(), raw / raw.mean(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(result.diagnostics["normaliser"], raw.mean(), rtol=2e-5)
    # Independent full-matrix inequality, including all spatial cross-terms.
    majorant = np.kron(metric, np.diag(raw))
    assert np.linalg.eigvalsh(majorant - fisher).min() > -1e-9
    np.testing.assert_array_equal(solver.fractions_numpy(), fields)
    np.testing.assert_array_equal(solver.gradient.numpy(), gradient)
    assert solver.evaluate() == loss
    # The preparation must not depend on the measured counts.
    for channel in solver.views[0]["channels"]:
        channel["counts"].fill_(23.0)
    repeated = prepare_fisher_scaling(solver)
    np.testing.assert_allclose(repeated.scale.numpy(), result.scale.numpy(), rtol=3e-7)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_regulariser_supplies_positive_scaling_without_data_curvature(
    precision: Literal["float32", "float64"],
) -> None:
    inputs = _inputs(beta=0.07, coefficients=False, precision=precision)
    solver = MaterialReconstruction(**inputs)
    expected, _, _ = _reference(inputs)
    result = prepare_fisher_scaling(solver)
    np.testing.assert_allclose(result.scale.numpy(), expected / expected.mean(), rtol=2e-7)
    assert result.diagnostics["zero_data_curvature_voxels"] == solver.grid.voxels
    assert result.diagnostics["floored_voxels"] == 0


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_zero_total_curvature_is_rejected(precision: Literal["float32", "float64"]) -> None:
    solver = MaterialReconstruction(**_inputs(coefficients=False, precision=precision))
    with pytest.raises(ContractError, match="no positive Fisher"):
        prepare_fisher_scaling(solver)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_uncovered_voxels_use_the_declared_algorithmic_floor(
    precision: Literal["float32", "float64"],
) -> None:
    inputs = _inputs(precision=precision)
    original = inputs["views"][0]
    geometry = DetectorGeometry(
        (-20.0, -0.6, -0.4),
        (20.0, -0.6, -0.4),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0),
        (1, 1),
    )
    inputs["views"] = (
        MaterialReconstructionView(
            geometry,
            RigidTransform(),
            np.full((2, 1, 1), 8100, dtype=np.float32),
            original.weights,
            original.response,
        ),
    )
    solver = MaterialReconstruction(**inputs)
    result = prepare_fisher_scaling(solver, minimum_relative_curvature=1e-4)
    scale = result.scale.numpy()
    assert result.diagnostics["zero_data_curvature_voxels"] == 9
    assert result.diagnostics["floored_voxels"] == 9
    assert math.isclose(float(scale.max() / scale.min()), 1e4, rel_tol=2e-7)
    assert math.isclose(float(np.mean(scale, dtype=np.float64)), 1.0, rel_tol=2e-7)


@pytest.mark.parametrize("floor", [0.0, -1.0, float("nan"), 1.01])
def test_invalid_curvature_floor_is_rejected(floor: float) -> None:
    solver = MaterialReconstruction(**_inputs())
    with pytest.raises(ContractError, match="minimum_relative_curvature"):
        prepare_fisher_scaling(solver, minimum_relative_curvature=floor)


@pytest.mark.parametrize("double", [False, True])
def test_fisher_algebra_retains_declared_path_precision(double: bool) -> None:
    """Check the FP64 intermediate block and seed before the FP32 voxel adjoint."""
    wp: Any = importlib.import_module("warp")
    kernels: Any = importlib.import_module("dpt.kernels.material_preconditioning")
    dtype = wp.float64 if double else wp.float32
    host_dtype = np.float64 if double else np.float32
    means = np.array([3.1, 6.2], dtype=host_dtype)
    derivatives = np.array(
        [[0.131234567891, 0.253456789123], [0.392345678912, 0.471234567891]], dtype=host_dtype
    )
    lengths = np.array([1.123456789123, 2.234567891234], dtype=host_dtype)
    fisher = wp.zeros(2, dtype=wp.vec3d, device="cuda:0")
    seed = wp.empty(2, dtype=dtype, device="cuda:0")
    status = wp.zeros(1, dtype=wp.int32, device="cuda:0")
    mean_device = wp.array(means, dtype=dtype, device="cuda:0")
    derivative_device = wp.array(derivatives.ravel(), dtype=dtype, device="cuda:0")
    length_device = wp.array(lengths, dtype=dtype, device="cuda:0")
    wp.launch(
        kernels.get_add_fisher(double),
        dim=2,
        inputs=[
            mean_device,
            derivative_device,
            2,
            fisher,
            status,
        ],
    )
    wp.launch(
        kernels.get_fisher_row_seed(double),
        dim=2,
        inputs=[
            fisher,
            length_device,
            wp.vec3d(0.8, 0.13, 1.2),
            seed,
            status,
        ],
    )
    wp.synchronize_device("cuda:0")
    expected: list[list[float]] = []
    expected_seed: list[float] = []
    inverse_root = np.array([[0.8, 0.13], [0.13, 1.2]])
    for index in range(2):
        vector = derivatives[:, index].astype(np.float64)
        block = np.outer(vector, vector) / float(means[index])
        expected.append([block[0, 0], block[0, 1], block[1, 1]])
        expected_seed.append(
            np.linalg.eigvalsh(inverse_root @ block @ inverse_root)[-1] * float(lengths[index])
        )
    np.testing.assert_allclose(fisher.numpy(), expected, rtol=2e-14, atol=1e-16)
    np.testing.assert_allclose(seed.numpy(), expected_seed, rtol=2e-14 if double else 1e-6)
    assert int(status.numpy()[0]) == 0


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_extinguished_mean_fails_before_a_regulariser_can_mask_it(
    precision: Literal["float32", "float64"],
) -> None:
    from dpt.contracts import NumericalError

    inputs = _inputs(beta=0.07, precision=precision)
    inputs["coefficients"].fill(1e4)
    solver = MaterialReconstruction(**inputs)
    expected_error = (
        "spectral (exponential|intermediate) underflow"
        if precision == "float64"
        else "Fisher voxel scaling"
    )
    with pytest.raises(NumericalError, match=expected_error):
        prepare_fisher_scaling(solver)


@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize(
    "mean_value,derivative_value", [(0.0, 1.0), (float("inf"), 1.0), (1.0, float("nan"))]
)
def test_fisher_algebra_reports_unrepresentable_input_status(
    double: bool,
    mean_value: float,
    derivative_value: float,
) -> None:
    wp: Any = importlib.import_module("warp")
    kernels: Any = importlib.import_module("dpt.kernels.material_preconditioning")
    dtype = wp.float64 if double else wp.float32
    mean = wp.array([mean_value], dtype=dtype, device="cuda:0")
    derivative = wp.array([derivative_value, 1.0], dtype=dtype, device="cuda:0")
    fisher = wp.zeros(1, dtype=wp.vec3d, device="cuda:0")
    status = wp.zeros(1, dtype=wp.int32, device="cuda:0")
    wp.launch(kernels.get_add_fisher(double), dim=1, inputs=[mean, derivative, 1, fisher, status])
    wp.synchronize_device("cuda:0")
    assert int(status.numpy()[0]) & 2
