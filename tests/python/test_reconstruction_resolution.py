"""Host contract tests; small arrays here are software checks, not recovery evidence."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

import copy
import importlib.util
import json
import sys
from collections.abc import Iterator
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import pytest

from dpt.geometry import DetectorGeometry
from dpt.volumes import GridSpec

np: Any = pytest.importorskip("numpy", reason="requires the optional scientific environment")
ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments/reconstruction-study"


@pytest.fixture(autouse=True)
def isolated_study_imports() -> Iterator[None]:
    """Each test gets this CLI directory's helpers and restores the prior cache."""
    names = {path.stem for path in STUDY.glob("*.py")}
    previous = {name: sys.modules[name] for name in names if name in sys.modules}
    for name in names:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


def helper() -> Any:
    path = STUDY / "_resolution.py"
    sys.path.insert(0, str(STUDY))
    try:
        spec = importlib.util.spec_from_file_location("resolution_contract_host", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(STUDY))


def config() -> dict[str, Any]:
    return json.loads((STUDY / "resolution-config-v1.json").read_text())


def test_exact_seven_outcomes_and_frozen_budgets() -> None:
    module, values = helper(), config()
    module.validate_schedule(values)
    assert len(module.required_outcomes()) == 7
    assert sum(r["maximum_accepted_updates"] for r in values["outcomes"]) == 140
    assert sum(r["soft_solve_seconds"] for r in values["outcomes"]) == 840
    for change in ("duplicate", "missing", "grid", "method", "cap", "step", "noise"):
        bad = copy.deepcopy(values)
        if change == "duplicate":
            bad["outcomes"][-1] = bad["outcomes"][0]
        elif change == "missing":
            bad["outcomes"].pop()
        elif change == "grid":
            bad["outcomes"][0]["shape_zyx"] = [64] * 3
        elif change == "method":
            bad["outcomes"][0]["method"] = "spectral_metric"
        elif change == "cap":
            bad["outcomes"][0]["soft_solve_seconds"] = 121
        elif change == "step":
            bad["scalar_initial_and_mapping_step_mm_inverse_squared"] = 3.125e-10
        else:
            bad["new_noise_phase_id"] = 0
        with pytest.raises(ValueError):
            module.validate_schedule(bad)


def test_exact_source_amendment_rejects_unrelated_or_partial_drift() -> None:
    module = helper()
    old = {"python/dpt/examples/reconstruction.py": "old", "python/dpt/projection.py": "same"}
    new = {**old, "python/dpt/examples/reconstruction.py": "new"}
    allowed = {"python/dpt/examples/reconstruction.py": {"observation": "old", "current": "new"}}
    assert module.check_source_amendment(old, new, allowed) == allowed
    assert module.check_source_amendment(new, new, allowed) == {}
    for candidate in (
        {**new, "python/dpt/projection.py": "bad"},
        {"python/dpt/examples/reconstruction.py": "new"},
        {**new, "python/dpt/examples/reconstruction.py": "different"},
    ):
        with pytest.raises(ValueError):
            module.check_source_amendment(old, candidate, allowed)


def test_fourfold_integrated_weights_preserve_coefficients_response_and_source() -> None:
    module = helper()
    incident = np.array([12500, 25000, 62500, 100000], dtype=np.float32)
    original = {
        "incident_weights": incident,
        "weights": np.tile(incident, (3, 1)),
        "mu_mm_inv": np.array(
            [[0.01, 0.02, 0.03, 0.04], [0.1, 0.09, 0.07, 0.05]], dtype=np.float32
        ),
        "response": np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
    }
    before = {k: v.tobytes() for k, v in original.items()}
    scaled = module.scaled_physics(original, 4.0)
    assert {k: v.tobytes() for k, v in original.items()} == before
    for name in ("mu_mm_inv", "response"):
        np.testing.assert_array_equal(scaled[name], original[name])
    assert scaled["incident_weights"].sum(dtype=np.float64) == 800000
    channel_total = (scaled["weights"].astype(np.float64) * scaled["response"]).sum()
    assert channel_total == 800000
    with pytest.raises(ValueError):
        module.scaled_physics(scaled, 4.0)
    with pytest.raises(ValueError):
        module.scaled_physics(original, 3.0)


def test_new_regime_streams_match_independent_explicit_seed_replay() -> None:
    module, values = helper(), config()
    means = np.arange(36 * 3 * 2 * 2, dtype=np.float32).reshape(36, 3, 2, 2) + 1
    before = means.tobytes()
    all_keys = []
    for regime, role in (("sparse", 1), ("limited", 2)):
        actual, keys = module.regime_counts(means, regime, values)
        assert actual.dtype == np.float32 and means.tobytes() == before
        for v in range(36):
            for c in range(3):
                key = [2026091251, 2, 2, role, 0, 2, v, c]
                assert keys[v * 3 + c] == key
                expected = np.random.Generator(
                    np.random.PCG64(np.random.SeedSequence(key))
                ).poisson(means[v, c].astype(np.float64))
                np.testing.assert_array_equal(actual[v, c], expected)
        all_keys.extend(map(tuple, keys))
    assert len(set(all_keys)) == 216


def test_regime_geometry_and_independent_corner_projection() -> None:
    module, values = helper(), config()
    grid = GridSpec((48, 48, 48), (3.1015625, 3.1015625, 4), (-72.88671875, -72.88671875, -94))
    geometry = DetectorGeometry(
        (0, -1000, 0), (-190, 500, -190), (1, 0, 0), (0, 0, 1), (4, 4), (96, 96)
    )
    for regime, number in (("dense", 144), ("sparse", 36), ("limited", 36)):
        rows = module.view_records(regime, values)
        assert rows == json.loads(json.dumps(rows))
        assert len(rows) == number and len({r["id"] for r in rows}) == number
        report = module.check_coverage(grid, geometry, rows)
        assert min(report["minimum_cell_clearance_uv"]) > 0
        # Rodrigues rotations computed independently from the fixed axes.
        first = rows[0]
        tilt, yaw = np.deg2rad([first["tilt_degrees"], first["yaw_degrees"]])

        def rotation(axis: Any, angle: float) -> Any:
            a = np.asarray(axis, dtype=np.float64)
            cross = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)

        expected = rotation((0, 0, 1), yaw) @ rotation((1, 0, 0), tilt)
        np.testing.assert_allclose(
            np.array(first["pose"]["rotation"]).reshape(3, 3), expected, atol=2e-16
        )
    limited = module.view_records("limited", values)
    assert limited[0]["yaw_degrees"] == -60 and limited[11]["yaw_degrees"] == 60
    truncated = DetectorGeometry((0, -1000, 0), (-2, 500, -2), (1, 0, 0), (0, 0, 1), (4, 4), (2, 2))
    with pytest.raises(ValueError, match="truncated"):
        module.check_coverage(grid, truncated, limited)


def test_recorded_paths_and_required_failure_rows(tmp_path: Path) -> None:
    module = helper()
    original = tmp_path / "input.bin"
    original.write_bytes(b"exact recorded bytes")
    hashes = {"input.bin": module.digest(original)}
    assert module.checked_file(tmp_path, "input.bin", hashes) == original
    original.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        module.checked_file(tmp_path, "input.bin", hashes)
    with pytest.raises(ValueError, match="unbounded"):
        module.child(tmp_path, "../outside.bin")
    rows = [
        {"id": name, "group": group, "method": method, "status": "failed"}
        for name, (group, method, _) in module.required_outcomes().items()
    ]
    module.require_coverage(rows)
    with pytest.raises(ValueError):
        module.require_coverage(rows[:-1])
    rows[-1] = rows[0]
    with pytest.raises(ValueError):
        module.require_coverage(rows)


def test_count_deviance_matches_independent_decimal_and_retains_signed_errors() -> None:
    module = helper()
    prediction = np.array([0.001, 80, 100, 1e6], dtype=np.float32)
    observed = np.array([0, 32, 120, 1000001], dtype=np.float32)
    mean = np.array([0.003, 100, 90, 1e6], dtype=np.float32)
    result = module.count_metrics(prediction, observed, mean)
    with localcontext() as context:
        context.prec = 70
        expected = Decimal(0)
        for p, k in zip(prediction, observed, strict=True):
            x, y = Decimal.from_float(float(p)), Decimal.from_float(float(k))
            expected += x - y + (y * (y / x).ln() if y else Decimal(0))
    assert result["observed_half_deviance"] == pytest.approx(float(expected), rel=2e-14)
    assert result["mean_signed_count_error"] < 0
    exact = module.count_metrics(mean, observed, mean)
    assert exact["mean_error_poisson_sd_rms"] == 0


def test_qualification_cartesian_guard_catches_duplicate_with_same_total() -> None:
    module = helper()
    reports = []
    for name, views, pair in (
        ("dense48", 144, (512, 1024)),
        ("dense64", 144, (512, 1024)),
        ("sparse-native", 36, (1024, 2048)),
        ("sparse48", 36, (512, 1024)),
        ("limited-native", 36, (1024, 2048)),
        ("limited48", 36, (512, 1024)),
    ):
        gates = []
        for view in range(views):
            for role, channels, samples in (
                ("independent_scalar", (0,), pair),
                ("independent_spectral", (0, 1, 2), pair),
                ("doubled_scalar", (0,), pair[:1]),
                ("doubled_spectral", (0, 1, 2), pair[:1]),
            ):
                for channel in channels:
                    for sample in samples:
                        gates.append(
                            dict(
                                role=role,
                                view=view,
                                channel=channel,
                                samples=sample,
                                passed=True,
                                p95_poisson_sd=0.01,
                                maximum_poisson_sd=0.1,
                            )
                        )
        reports.append(
            dict(
                comparison_arrays=f"qualification/{name}/",
                gates=gates,
                passed=True,
                gate_count=len(gates),
            )
        )
    assert module.qualification_coverage(reports) == 5184
    duplicate = copy.deepcopy(reports)
    duplicate[0]["gates"][0] = duplicate[0]["gates"][1]
    assert sum(r["gate_count"] for r in duplicate) == 5184
    with pytest.raises(ValueError, match="Cartesian"):
        module.qualification_coverage(duplicate)
    reports[-1]["gates"][-1]["maximum_poisson_sd"] = 0.2
    with pytest.raises(ValueError, match="numerical"):
        module.qualification_coverage(reports)
