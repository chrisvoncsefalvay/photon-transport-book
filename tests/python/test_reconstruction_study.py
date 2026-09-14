"""Host checks for fixed post-log data preparation; no scientific GPU evidence."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

import importlib.util
import sys
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import pytest

from dpt.contracts import ContractError
from dpt.examples.reconstruction import SolverSettings
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.volumes import GridSpec

np: Any = pytest.importorskip("numpy", reason="requires the scientific environment")


def _baseline() -> Any:
    path = Path(__file__).resolve().parents[2] / "experiments/reconstruction-study/baseline.py"
    spec = importlib.util.spec_from_file_location("reconstruction_study_baseline_host", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_postlog_matches_independent_decimal_and_retains_negative_data() -> None:
    counts = np.array([[0, 1, 32, 120], [2**24, 99, 1, 2]], dtype=np.float32)
    beam = np.array([[80, 100, 64, 100], [1e-30, 100, 1e30, 2]], dtype=np.float32)
    before = (counts.tobytes(), beam.tobytes())
    result = _baseline().post_log_data(counts, beam)
    expected = np.zeros_like(counts)
    with localcontext() as context:
        context.prec = 70
        for index in np.ndindex(counts.shape):
            if counts[index] > 0:
                expected[index] = float(
                    (Decimal.from_float(float(beam[index])) / Decimal(int(counts[index]))).ln()
                )
    np.testing.assert_array_equal(result.optical_depth, expected)
    np.testing.assert_array_equal(result.weights, counts)
    np.testing.assert_array_equal(result.valid, (counts > 0).astype(np.uint8))
    assert result.excluded_zero_counts == 1
    assert result.negative_optical_depths == 2
    assert result.optical_depth[0, 0] == 0 and result.weights[0, 0] == 0
    assert (counts.tobytes(), beam.tobytes()) == before


@pytest.mark.parametrize("bad", [-1.0, 1.5, float("nan"), float("inf"), float(2**24 + 2)])
def test_postlog_rejects_invalid_counts(bad: float) -> None:
    counts = np.full((2, 3), 20, dtype=np.float32)
    counts[0, 0] = bad
    with pytest.raises(ContractError):
        _baseline().post_log_data(counts, np.full((2, 3), 100, dtype=np.float32))


def test_postlog_does_not_accept_nonpositive_beam_or_unmatched_images() -> None:
    counts = np.ones((2, 3), dtype=np.float32)
    beam = np.ones_like(counts)
    beam[0, 0] = 0
    with pytest.raises(ContractError, match="positive"):
        _baseline().post_log_data(counts, beam)
    with pytest.raises(ContractError, match="shape"):
        _baseline().post_log_data(counts, np.ones((3, 2), dtype=np.float32))


def _study() -> Any:
    path = Path(__file__).resolve().parents[2] / "experiments/reconstruction-study/_study.py"
    spec = importlib.util.spec_from_file_location("reconstruction_study_host_boundaries", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coarse_cells_match_explicit_blocks_and_oriented_physical_faces() -> None:
    grid = GridSpec((6, 8, 10), (0.5, 0.7, 1.2), (4, -2, 8), (0, -1, 0, 1, 0, 0, 0, 0, 1))
    fields = np.arange(2 * grid.voxels, dtype=np.float32).reshape(2, *grid.shape) / 1000
    coarse, result = _study().coarse_cells(fields, grid, (3, 2, 5))
    expected = np.empty((2, 3, 2, 5), dtype=np.float32)
    for material, z, y, x in np.ndindex(expected.shape):
        expected[material, z, y, x] = fields[
            material, 2 * z : 2 * z + 2, 4 * y : 4 * y + 4, 2 * x : 2 * x + 2
        ].mean(dtype=np.float64)
    np.testing.assert_array_equal(coarse, expected)
    np.testing.assert_allclose(result.spacing_mm, (1, 2.8, 2.4), rtol=0, atol=0)
    for native_corner, coarse_corner in zip(grid.support, result.support, strict=True):
        np.testing.assert_allclose(
            grid.grid_to_object(native_corner),
            result.grid_to_object(coarse_corner),
            atol=1e-14,
            rtol=0,
        )
    with pytest.raises(ValueError, match="divide"):
        _study().coarse_cells(fields, grid, (4, 2, 5))


def test_scalar_view_geometry_matches_independent_inverse_ray_endpoints() -> None:
    # Nonzero translation and a nontrivial rotation exercise more than the study's zero translation.
    pose = RigidTransform((0, -0.8, 0.6, 1, 0, 0, 0, 0.6, 0.8), (2, -3, 4))
    geometry = DetectorGeometry(
        (0, -1000, 0), (-30, 500, -24), (1, 0, 0), (0, 0, 1), (4, 3), (17, 16)
    )
    moved = _study().geometry_in_object(geometry, pose)
    rotation = np.array(pose.rotation).reshape(3, 3)
    np.testing.assert_allclose(
        moved.source_mm,
        rotation.T @ (np.array(geometry.source_mm) - pose.translation_mm),
        rtol=0,
        atol=1e-13,
    )
    for row, col in ((0, 0), (8, 7), (16, 15)):
        original = (
            np.array(geometry.origin_mm)
            + col * 4 * np.array(geometry.u)
            + row * 3 * np.array(geometry.v)
        )
        actual = (
            np.array(moved.origin_mm) + col * 4 * np.array(moved.u) + row * 3 * np.array(moved.v)
        )
        np.testing.assert_allclose(
            actual, rotation.T @ (original - pose.translation_mm), rtol=0, atol=2e-13
        )


def test_noise_keys_reproduce_pairing_without_model_or_view_collisions() -> None:
    config = {"noise_roles": {"scalar": 1, "spectral": 2}, "noise_root_seed": 732, "case": 2}
    means = np.full((3, 2, 4, 5), 100, dtype=np.float32)
    first, keys = _study().poisson_counts(means, config, "scalar")
    repeat, repeat_keys = _study().poisson_counts(means, config, "scalar")
    other, other_keys = _study().poisson_counts(means, config, "spectral")
    np.testing.assert_array_equal(first, repeat)
    assert keys == repeat_keys and len({tuple(k) for k in keys + other_keys}) == 12
    assert not np.array_equal(first, other)
    assert first.dtype == np.float32 and (first % 1 == 0).all()


def test_stationarity_gate_requires_both_constraints_and_handles_zero_start() -> None:
    study = _study()
    gate = study.stationarity_gate(1000, 0.01, 0.001, 0.005)
    assert gate["absolute_threshold"] == 0.5
    assert not study.stationarity_result(0.75, gate)["passed"]
    assert study.stationarity_result(0.5, gate)["passed"]
    zero = study.stationarity_gate(0, 0.01, 0.001, 0.005)
    assert study.stationarity_result(0, zero)["passed"]
    assert study.stationarity_result(1, zero)["relative_mapping"] is None


def _solver_settings() -> dict[str, Any]:
    return {
        "iterations": 2,
        "initial_step_mm_inverse_squared": 1e-4,
        "backtracking_factor": 0.5,
        "armijo": 1e-4,
        "maximum_backtracks": 40,
        "gradient_mapping_tolerance_mm": 0.0,
        "regularisation_mm": 0.4,
    }


def test_optional_scalar_mapping_step_retains_default_and_reads_explicit_value() -> None:
    settings = _solver_settings()
    assert SolverSettings.read(settings).mapping_step_mm_inverse_squared is None
    assert (
        SolverSettings.read(
            {**settings, "mapping_step_mm_inverse_squared": None}
        ).mapping_step_mm_inverse_squared
        is None
    )
    actual = SolverSettings.read({**settings, "mapping_step_mm_inverse_squared": 1e-8})
    assert actual.mapping_step_mm_inverse_squared == 1e-8
    assert actual.initial_step_mm_inverse_squared == 1e-4


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_optional_scalar_mapping_step_rejects_invalid_values(bad: float) -> None:
    with pytest.raises(ContractError):
        SolverSettings.read({**_solver_settings(), "mapping_step_mm_inverse_squared": bad})
