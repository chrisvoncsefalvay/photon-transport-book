"""Independent host checks for the supplied application mathematics and recorder."""

# External NumPy/Pytest typing is not the production application boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from dpt.contracts import NumericalError
from dpt.examples.acquisition_design import target_jacobians, target_variance
from dpt.geometry import RigidTransform
from dpt.recovery import PoseChart
from dpt.registration import Evaluation, IterationRecord, RecoveryPolicy, recover_parameters

np: Any = pytest.importorskip(
    "numpy", reason="application checks require the scientific environment"
)


def _observer_class() -> Any:
    directory = Path(__file__).resolve().parents[2] / "experiments/application-study"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("application_study_run", directory / "run.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.AcceptedPoseObserver
    finally:
        sys.path.remove(str(directory))


def test_target_jacobian_matches_independent_axis_rotation_differences() -> None:
    angle = 0.31
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1.0]]
    )
    pose = RigidTransform(tuple(rotation.ravel()), (2.0, -1.0, 0.5))
    points = np.array([[2.0, 7.0, -3.0], [-4.0, 0.5, 11.0]])
    scales = np.array([1.2, 0.7, 2.0, 0.03, 0.02, 0.05])
    actual = target_jacobians(np, pose, points, scales)
    expected = np.empty_like(actual)
    h = 1e-4
    for axis in range(6):
        transformed = []
        for sign in (-1, 1):
            local_rotation, local_translation = np.eye(3), np.zeros(3)
            if axis < 3:
                local_translation[axis] = sign * h * scales[axis]
            else:
                k = axis - 3
                unit = np.eye(3)[k]
                skew = np.array(
                    [[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]]
                )
                theta = sign * h * scales[axis]
                local_rotation += math.sin(theta) * skew + (1 - math.cos(theta)) * skew @ skew
            transformed.append(
                (points @ local_rotation.T + local_translation) @ rotation.T
                + np.array(pose.translation_mm)
            )
        expected[:, :, axis] = (transformed[1] - transformed[0]) / (2 * h)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=2e-10)


def test_target_variance_matches_separable_closed_form_and_rejects_singular() -> None:
    derivatives = np.array(
        [
            [
                [1.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 3.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 4.0],
            ]
        ]
    )
    diagonal = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    expected = sum(
        float(derivatives[0, row, col]) ** 2 / float(diagonal[col])
        for row in range(3)
        for col in range(6)
    )
    score, covariance, condition = target_variance(np, np.diag(diagonal), derivatives, 10.0)
    assert score == pytest.approx(expected, rel=1e-14)
    np.testing.assert_allclose(covariance, np.diag(1 / diagonal), atol=1e-15)
    assert condition == 6.5
    with pytest.raises(NumericalError, match="positive definite"):
        target_variance(np, np.diag([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), derivatives, 1e10)
    with pytest.raises(NumericalError, match="positive definite"):
        target_variance(np, np.diag([1e-12, 1.0, 2.0, 3.0, 4.0, 5.0]), derivatives, 1e10)


def test_accepted_observer_excludes_rejected_trials_and_checks_callback() -> None:
    cls = _observer_class()
    chart = PoseChart()

    def objective(x: tuple[float, ...]) -> Evaluation:
        if abs(x[0]) > 0.6:
            raise NumericalError("outside software fixture domain")
        return Evaluation(sum((a - 0.1) ** 2 for a in x), tuple(2 * (a - 0.1) for a in x))

    observer = cls(objective, chart)
    result = recover_parameters(
        observer, (0.0,) * 6, policy=RecoveryPolicy(max_iterations=10), observe=observer.observe
    )
    assert result.stationary
    assert observer.trajectory[-1]["parameters"] == list(result.parameters)
    count = len(observer.trajectory)
    observer((0.3,) * 6)
    assert len(observer.trajectory) == count
    with pytest.raises(RuntimeError, match="do not match"):
        observer.observe(IterationRecord(999, -1.0, 0.0, 1.0, 1, 0))


def _study_module(name: str) -> Any:
    directory = Path(__file__).resolve().parents[2] / "experiments/application-study"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(f"study_{name}", directory / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(directory))


def _role_protocol() -> dict[str, Any]:
    return {
        "registration": {
            "noise_replicates_per_case_and_start": 2,
            "view_sets_degrees": {"orthogonal": [0, 90], "near_parallel": [0, 5]},
            "summed_open_beam_per_pixel": 20000,
            "initial_offsets_local_se3_mm_rad": [[float(i), 0, 0, 0, 0, 0] for i in range(4)],
            "stress": {
                "noise_replicates_per_case": 1,
                "views_degrees": [0],
                "open_beam_per_pixel": 200,
            },
        },
        "acquisition": {
            "replicates_per_case": 8,
            "base_view_degrees": 0,
            "candidate_angles_degrees": [-90, -60, -30, 30, 60, 90],
            "fixed_policy_angle_degrees": 90,
            "base_and_candidate_open_beam_per_pixel": 1000,
            "constrained": {
                "replicates_per_case": 4,
                "candidate_angles_degrees": [-15, 15, 30],
                "fixed_angle_degrees": 15,
            },
        },
        "randomness": {"random_policy_root_seed": 20260915},
    }


def test_final_schedule_preserves_all_136_outcomes_and_paired_observation_ids() -> None:
    module = _study_module("_final")
    protocol = _role_protocol()
    fitting = module.registration_jobs(protocol)
    assert len(fitting) == 20
    for case in (0, 1):
        designs = module.design_jobs(protocol, case)
        assert len(designs) == 12 and len(fitting) + 4 * len(designs) == 68
        assert designs == module.design_jobs(protocol, case)
        assert all(job["random_angle"] in job["angles"] for job in designs)
    for replicate in range(2):
        for start in range(4):
            paired = [
                j
                for j in fitting
                if j["role"] == "reg" and j["replicate"] == replicate and j["start"] == start
            ]
            assert len(paired) == 2
            assert paired[0]["view_ids"][0] == paired[1]["view_ids"][0]
            assert paired[0]["view_ids"][1] != paired[1]["view_ids"][1]
    for role, spec in module.role_specs(protocol).items():
        ids = {
            module.view_id(role, n, a) for n in range(spec["replicates"]) for a in spec["angles"]
        }
        heldout = {module.view_id(role, n, a) for n in range(spec["replicates"]) for a in (45, 135)}
        assert ids.isdisjoint(heldout)
    assert module.view_id("broad", 0, 0) != module.view_id("constrained", 0, 0)


def test_case_writer_includes_only_requested_fitting_arrays(tmp_path: Path) -> None:
    from dataclasses import asdict

    module = _study_module("_study")
    known = {
        "source_description": "explicit role-separation software fixture",
        "rights": "Apache-2.0",
        "attenuation": {"file": "mu.npy", "sha256": "a" * 64, "units": "mm^-1"},
        "grid": {"shape": [1, 1, 1], "spacing_mm": [1, 1, 1]},
    }
    module.write_json(tmp_path / "known.json", known)
    view = {
        "geometry": {"shape": [1, 1]},
        "open_beam_counts": 1.0,
        "observation": {"file": "fitting.npy", "sha256": "b" * 64, "units": "counts"},
        "mask": {"file": "mask.npy", "sha256": "c" * 64, "units": "dimensionless"},
    }
    public = {
        "known_sha256": module.digest(tmp_path / "known.json"),
        "views": {"fit": view, "future": {}},
    }
    protocol = {
        "numerics": {
            "chart_scales": [1] * 6,
            "chart_rotation_radius_radians": 1.0,
            "fitting_samples_per_ray": 8,
            "policy": {},
        }
    }
    path = module.registration_case(
        tmp_path, tmp_path, public, protocol, ["fit"], RigidTransform(), tmp_path / "case.json"
    )
    value = module.read_json(path)
    assert set(value["arrays"]) == {"attenuation", "fit-observation", "fit-mask"}
    assert "evaluation" not in value["registration"]
    assert value["registration"]["chart"]["anchor"]["rotation"] == list(
        asdict(RigidTransform())["rotation"]
    )


def test_final_coverage_rejects_duplicated_missing_mislabelled_and_unknown_outcomes() -> None:
    module = _study_module("_final")
    protocol = _role_protocol()
    rows = [dict(job, status="failed") for job in module.registration_jobs(protocol)]
    for job in module.design_jobs(protocol, 0):
        for policy in ("base", "selected", "fixed", "random"):
            rows.append(dict(job, name=job["name"] + "-" + policy, policy=policy, status="failed"))
    module.require_outcome_coverage(protocol, 0, rows)
    with pytest.raises(ValueError, match="missing, duplicate"):
        module.require_outcome_coverage(protocol, 0, [*rows[:-1], rows[0]])
    for changed, message in (
        ({"status": "forgotten"}, "status"),
        ({"role": "broad"}, "paired role"),
    ):
        with pytest.raises(ValueError, match=message):
            module.require_outcome_coverage(protocol, 0, [{**rows[0], **changed}, *rows[1:]])
