"""Independent CUDA checks for the scalar callback and post-log comparator."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false

import hashlib
import importlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from dpt.examples._common import load_case
from dpt.examples.reconstruction import Reconstruction
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

np: Any = importlib.import_module("numpy")
wp: Any = importlib.import_module("warp")
torch: Any = importlib.import_module("torch")
pytestmark = pytest.mark.gpu


def _baseline() -> Any:
    path = Path(__file__).resolve().parents[3] / "experiments/reconstruction-study/baseline.py"
    spec = importlib.util.spec_from_file_location("reconstruction_study_baseline_cuda", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pilot(monkeypatch: pytest.MonkeyPatch) -> Any:
    folder = Path(__file__).resolve().parents[3] / "experiments/reconstruction-study"
    monkeypatch.syspath_prepend(str(folder))
    for name in ("_study", "baseline", "probe_acquisition"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "reconstruction_pilot_cuda_check", folder / "pilot.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["RecordedPoisson", "RecordedPWLS"])
def test_pilot_trace_and_named_budget_preserve_the_actual_accepted_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    case, grid, _, geometries, _ = _case(tmp_path)
    module = _pilot(monkeypatch)
    engine = getattr(module, name)(case, device="cuda:0", torch=torch, wp=wp)
    engine.trace, engine.phase = [], "solve"
    accepted = []

    def stop(iteration, row):
        assert iteration == 1
        evaluations = [item for item in engine.trace if item["kind"] == "evaluate"]
        assert evaluations[-1]["objective"] == row["loss_after"]
        assert not evaluations[-1]["gradient"]
        accepted.append((row.copy(), engine.mu.cpu().numpy().copy()))
        raise module.PilotBudgetError("bounded CUDA callback test")

    with pytest.raises(module.PilotBudgetError):
        engine.solve(callback=stop)
    assert len(accepted) == 1
    np.testing.assert_array_equal(engine.mu.cpu().numpy(), accepted[0][1])
    final_loss = engine.evaluate(engine.mu, engine.mu_wp, gradient=True)
    assert final_loss == pytest.approx(accepted[0][0]["loss_after"], rel=2e-12, abs=1e-9)
    assert (engine.mu.cpu().numpy() >= 0).all()
    predictions = (
        engine.predictions_numpy()
        if name == "RecordedPWLS"
        else tuple(view["prediction"].numpy().reshape(view["shape"]) for view in engine.views)
    )
    for actual, geometry in zip(predictions, geometries, strict=True):
        expected = _mean(accepted[0][1], grid, geometry)
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=2e-5)
    assert any(row["kind"] == "proposal_or_mapping" for row in engine.trace)


def _case(folder: Path, *, all_counts_above_beam: bool = False):
    grid = GridSpec((3, 4, 5), (0.9, 1.2, 1.5), (-1.8, -1.8, -1.5))
    field = np.array(
        [
            0.025 + 0.001 * x + 0.002 * y + 0.003 * z
            for z in range(3)
            for y in range(4)
            for x in range(5)
        ],
        dtype=np.float32,
    ).reshape(grid.shape)
    geometries = [
        DetectorGeometry(
            (-20, 0.2, 0.3), (20, -1.2, -1.1), (0, 1, 0), (0, 0, 1), (0.5, 0.6), (3, 4)
        ),
        DetectorGeometry(
            (0.3, -20, -0.2), (1.1, 20, -1.0), (-1, 0, 0), (0, 0, 1), (0.4, 0.5), (3, 4)
        ),
    ]
    arrays = {}

    def store(name, value, units):
        path = folder / (name + ".npy")
        np.save(path, value, allow_pickle=False)
        arrays[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "units": units,
            "source": "independent software correctness fixture",
            "rights": "original fixture; Apache-2.0",
        }
        return name

    initial = store("initial", field, "mm^-1")
    views, observations = [], []
    for i, geometry in enumerate(geometries):
        counts = np.full(geometry.shape, 120, dtype=np.float32)
        if not all_counts_above_beam:
            counts[0, 0] = 0
            counts[0, 1] = 83 + i
        observations.append(counts)
        views.append(
            {
                "geometry": asdict(geometry),
                "counts": store(f"counts{i}", counts, "counts"),
                "open_beam": store(
                    f"beam{i}", np.full(geometry.shape, 100, dtype=np.float32), "counts"
                ),
            }
        )
    config = {
        "schema_version": 1,
        "arrays": arrays,
        "reconstruction": {
            "grid": asdict(grid),
            "fixed_pose": asdict(RigidTransform()),
            "initial_volume": initial,
            "samples_per_ray": 61,
            "views": views,
            "solver": {
                "iterations": 4,
                "initial_step_mm_inverse_squared": 0.001,
                "backtracking_factor": 0.5,
                "armijo": 1e-4,
                "maximum_backtracks": 40,
                "gradient_mapping_tolerance_mm": 0.0,
                "regularisation_mm": 0.4,
            },
        },
    }
    path = folder / "case.json"
    path.write_text(json.dumps(config))
    return load_case(path), grid, field, geometries, observations


def _mean(values, grid, geometry):
    return np.array(
        [
            100
            * math.exp(
                -integrate_sampled_field(
                    values.reshape(-1).tolist(),
                    grid,
                    geometry,
                    RigidTransform(),
                    *divmod(p, geometry.shape[1]),
                    midpoint_samples=61,
                    validate=False,
                )
            )
            for p in range(geometry.pixels)
        ]
    ).reshape(geometry.shape)


def test_scalar_default_callback_parity_and_actual_accepted_checkpoints(tmp_path: Path) -> None:
    case, grid, _, geometries, _ = _case(tmp_path)
    plain = Reconstruction(case, device="cuda:0", torch=torch, wp=wp)
    watched = Reconstruction(case, device="cuda:0", torch=torch, wp=wp)
    explicit = Reconstruction(case, device="cuda:0", torch=torch, wp=wp)
    explicit.policy = replace(explicit.policy, mapping_step_mm_inverse_squared=0.001)
    checkpoints = []

    def record(iteration, row):
        values = watched.mu.cpu().numpy().copy().reshape(grid.shape)
        assert iteration == row["iteration"] == len(checkpoints) + 1
        assert np.isfinite(values).all() and (values >= 0).all()
        assert row["loss_after"] < row["loss_before"]
        checkpoints.append((values, dict(row)))
        row["loss_after"] = float("nan")  # Detached caller record cannot mutate solver history.

    with torch.no_grad():
        expected = plain.solve()
        actual = watched.solve(callback=record)
        explicit_result = explicit.solve()
        assert explicit_result["termination"] == expected["termination"]
        assert explicit_result["accepted_steps"] == expected["accepted_steps"]
        np.testing.assert_allclose(
            explicit.mu.cpu().numpy(), plain.mu.cpu().numpy(), rtol=3e-6, atol=2e-8
        )
        assert explicit_result["final_gradient_mapping_mm"] == pytest.approx(
            expected["final_gradient_mapping_mm"], rel=3e-6, abs=2e-6
        )
        np.testing.assert_allclose(
            watched.mu.cpu().numpy(), plain.mu.cpu().numpy(), rtol=3e-6, atol=2e-8
        )
        assert actual["termination"] == expected["termination"]
        assert actual["accepted_steps"] == expected["accepted_steps"] == len(checkpoints) > 0
        assert actual["final_objective"] == pytest.approx(expected["final_objective"], rel=2e-7)
        assert all(math.isfinite(row["loss_after"]) for row in actual["history"])
        final = watched.mu.cpu().numpy().copy().reshape(grid.shape)
        np.testing.assert_array_equal(final, checkpoints[-1][0])
        for checkpoint, record_row in checkpoints:
            watched.trial.copy_(torch.as_tensor(checkpoint.reshape(-1), device="cuda:0"))
            loss = watched.evaluate(watched.trial, watched.trial_wp, gradient=False)
            assert loss == pytest.approx(record_row["loss_after"], rel=2e-7)
        watched.evaluate(watched.mu, watched.mu_wp, gradient=True)
        for geometry, view in zip(geometries, watched.views, strict=True):
            np.testing.assert_allclose(
                view["prediction"].numpy().reshape(geometry.shape),
                _mean(final, grid, geometry),
                rtol=3e-7,
                atol=1e-5,
            )


def test_scalar_distinct_mapping_step_controls_stopping_without_changing_trial(
    tmp_path: Path,
) -> None:
    case, _, initial, _, _ = _case(tmp_path, all_counts_above_beam=True)
    engine = Reconstruction(case, device="cuda:0", torch=torch, wp=wp)
    engine.policy = replace(
        engine.policy,
        iterations=2,
        initial_step_mm_inverse_squared=1e-7,
        mapping_step_mm_inverse_squared=1.0,
        gradient_mapping_tolerance_mm=0.06,
    )
    stopped = engine.solve()
    assert (
        stopped["termination"] == "projected_gradient_tolerance" and stopped["accepted_steps"] == 0
    )
    np.testing.assert_array_equal(engine.mu.cpu().numpy(), initial.ravel())
    gradient = engine.gradient.cpu().numpy().astype(np.float64)
    expected = np.max(np.abs(np.minimum(gradient, initial.ravel().astype(np.float64))))
    assert stopped["final_gradient_mapping_mm"] == expected
    assert np.max(np.abs(np.minimum(gradient, initial.ravel().astype(np.float64) / 1e-7))) > 0.06
    engine.policy = replace(engine.policy, gradient_mapping_tolerance_mm=0.0)
    advanced = engine.solve()
    assert advanced["accepted_steps"] == 2
    assert all(
        row["step_mm_inverse_squared"] == 1e-7 and row["backtracks"] == 0
        for row in advanced["history"]
    )
    final = engine.mu.cpu().numpy().astype(np.float64)
    gradient = engine.gradient.cpu().numpy().astype(np.float64)
    assert advanced["final_gradient_mapping_mm"] == np.max(np.abs(np.minimum(gradient, final)))


def test_pwls_composition_matches_independent_multiview_direction_and_exports(
    tmp_path: Path,
) -> None:
    case, grid, field, geometries, observations = _case(tmp_path)
    problem = _baseline().PostLogReconstruction(case, device="cuda:0", torch=torch, wp=wp)
    direction = np.cos(np.arange(field.size) * 0.61).reshape(grid.shape) * 0.01

    def reference(values):
        terms = []
        for geometry, counts in zip(geometries, observations, strict=True):
            for p in range(geometry.pixels):
                row, col = divmod(p, geometry.shape[1])
                k = float(counts[row, col])
                if k == 0:
                    continue
                depth = integrate_sampled_field(
                    values.reshape(-1).tolist(),
                    grid,
                    geometry,
                    RigidTransform(),
                    row,
                    col,
                    midpoint_samples=61,
                    validate=False,
                )
                terms.append(0.5 * k * (depth - math.log(100 / k)) ** 2)
        for axis, spacing in enumerate(reversed(grid.spacing_mm)):
            terms.append(
                0.5
                * 0.4
                * math.prod(grid.spacing_mm)
                / spacing**2
                * float(np.sum(np.diff(values, axis=axis) ** 2))
            )
        return math.fsum(terms)

    with torch.no_grad():
        loss = problem.evaluate(problem.mu, problem.mu_wp, gradient=True)
        gradient = problem.gradient.cpu().numpy().astype(np.float64).reshape(grid.shape)
        expected = reference(field.astype(np.float64))
        assert loss == pytest.approx(expected, rel=2e-6)
        finite_difference = (
            reference(field.astype(np.float64) + 1e-3 * direction)
            - reference(field.astype(np.float64) - 1e-3 * direction)
        ) / 2e-3
        assert float(np.sum(gradient * direction)) == pytest.approx(
            finite_difference, rel=2e-4, abs=2e-6
        )
        before = gradient.copy()
        problem.trial.copy_(problem.mu * 0.9)
        problem.evaluate(problem.trial, problem.trial_wp, gradient=False)
        np.testing.assert_array_equal(problem.gradient.cpu().numpy().reshape(grid.shape), before)
        result = problem.solve()
        final = problem.mu.cpu().numpy().copy().reshape(grid.shape)
        assert result["accepted_steps"] > 0 and result["final_objective"] < loss
        assert (final >= 0).all() and np.any(final == 0)
        for geometry, prediction in zip(geometries, problem.predictions_numpy(), strict=True):
            np.testing.assert_allclose(
                prediction, _mean(final, grid, geometry), rtol=3e-7, atol=1e-5
            )
        for metadata in problem.preprocessing:
            assert metadata["excluded_zero_counts"] == 1
            assert metadata["negative_optical_depths"] == 10
