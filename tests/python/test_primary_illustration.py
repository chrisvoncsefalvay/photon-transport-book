"""CPU protocol and file-boundary checks, not reconstruction performance evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments/reconstruction-study"


@pytest.fixture(autouse=True)
def isolated_study_imports() -> Iterator[None]:
    """Bare CLI helper names from other experiment directories must not leak here."""
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
    path = STUDY / "_illustration.py"
    sys.path.insert(0, str(STUDY))
    try:
        spec = importlib.util.spec_from_file_location("primary_illustration_host", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(STUDY))


def config() -> dict[str, Any]:
    return json.loads((STUDY / "primary64-config-v1.json").read_text())


def test_source_bound_protocol_and_exact_dual_budget() -> None:
    module, values = helper(), config()
    module.validate_config(values)
    assert values["pilot_steps"] == 1000
    assert values["pilot_soft_seconds_per_fit"] == 7200
    assert values["stage_soft_seconds"] == 7560
    assert values["checkpoint_steps"] == [0, 1, 5, 10, 20, 50, *range(100, 1001, 100)]
    for key, value in (
        ("noise_root_seed", 1),
        ("pilot_steps", 1001),
        ("spectral_mapping_step", 1e-5),
    ):
        changed = copy.deepcopy(values)
        changed[key] = value
        with pytest.raises(ValueError, match="configuration"):
            module.validate_config(changed)


def test_execution_tree_guard_rejects_an_installed_sibling_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = helper()
    identities = module.loaded_source_identity(ROOT)
    assert identities["dpt"]["path"] == str(ROOT / "python/dpt/__init__.py")
    monkeypatch.setattr(sys.modules["dpt"], "__file__", str(tmp_path / "dpt/__init__.py"))
    with pytest.raises(ValueError, match="outside this execution tree"):
        module.loaded_source_identity(ROOT)


def test_all_fitting_and_withheld_poses_match_independent_rz_rx() -> None:
    module, values = helper(), config()
    fit, held = module.views(values, "fitting"), module.views(values, "withheld")
    assert len(fit) == 144 and len(held) == 24
    assert len({(r["tilt_degrees"], r["yaw_degrees"]) for r in fit + held}) == 168
    for row in fit + held:
        a, b = map(math.radians, (row["tilt_degrees"], row["yaw_degrees"]))
        # Independent expanded Rz(yaw) @ Rx(tilt), without canonical composition.
        expected = [
            [math.cos(b), -math.sin(b) * math.cos(a), math.sin(b) * math.sin(a)],
            [math.sin(b), math.cos(b) * math.cos(a), -math.cos(b) * math.sin(a)],
            [0, math.sin(a), math.cos(a)],
        ]
        np.testing.assert_allclose(
            np.asarray(row["pose"]["rotation"]).reshape(3, 3), expected, rtol=0, atol=4e-16
        )
        assert row["pose"]["translation_mm"] == [0, 0, 0]
    assert fit + held == json.loads(json.dumps(fit + held))
    grid = module.inverse_grid()
    faces = np.asarray([grid.grid_to_object(c) for c in grid.support])
    np.testing.assert_array_equal(faces, [[-74.4375, -74.4375, -96], [74.4375, 74.4375, 96]])


def test_illustration_keys_are_unique_across_roles_and_exclude_old_roots() -> None:
    module, values = helper(), config()
    groups = module.expected_noise_namespace(values)
    assert len(groups["fitting"]) == 432 and len(groups["withheld"]) == 72
    all_keys = groups["fitting"] + groups["withheld"]
    assert len({tuple(k) for k in all_keys}) == 504
    assert {k[0] for k in all_keys} == {2026091264}
    assert {k[3] for k in groups["fitting"]} == {1}
    assert {k[3] for k in groups["withheld"]} == {2}
    assert all(k[0] not in (2026091251, 120926, 20260913) for k in all_keys)
    with pytest.raises(ValueError):
        module.noise_keys(values, "unspecified")


def reports() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, n, pair in (
        ("fitting-native", 144, (1024, 2048)),
        ("fitting64", 144, (512, 1024)),
        ("withheld-native", 24, (1024, 2048)),
        ("withheld64", 24, (512, 1024)),
    ):
        gates: list[dict[str, Any]] = []
        for view in range(n):
            for model, channels in (("scalar", 1), ("spectral", 3)):
                for c in range(channels):
                    for label, samples in (("doubled", pair[:1]), ("independent", pair)):
                        for sample in samples:
                            gates.append(
                                {
                                    "role": label + "_" + model,
                                    "view": view,
                                    "channel": c,
                                    "samples": sample,
                                    "p95_poisson_sd": 0.01,
                                    "maximum_poisson_sd": 0.02,
                                    "passed": True,
                                }
                            )
        result[name] = {
            "gates": gates,
            "samples": list(pair),
            "passed": True,
            "gate_count": len(gates),
            "error_exposure_photons_per_ray": 800000.0,
        }
    return result


def test_full_qualification_requires_every_valid_cartesian_gate() -> None:
    module, original = helper(), reports()
    assert module.qualification_coverage(original) == 4032
    for mutation in (
        "missing_family",
        "missing_gate",
        "duplicate",
        "wrong_role",
        "nonfinite",
        "too_large",
        "false",
        "wrong_exposure",
    ):
        changed = copy.deepcopy(original)
        family = changed["withheld64"]
        if mutation == "missing_family":
            del changed["fitting-native"]
        elif mutation == "missing_gate":
            family["gates"].pop()
        elif mutation == "duplicate":
            family["gates"][-1] = family["gates"][0]
        elif mutation == "wrong_role":
            family["gates"][0]["role"] = "independent_other"
        elif mutation == "nonfinite":
            family["gates"][0]["maximum_poisson_sd"] = math.nan
        elif mutation == "too_large":
            family["gates"][0]["maximum_poisson_sd"] = 0.2
        elif mutation == "false":
            family["gates"][0]["passed"] = False
        else:
            family["error_exposure_photons_per_ray"] = 200000.0
        with pytest.raises(ValueError):
            module.qualification_coverage(changed)


def test_numeric_physical_identity_includes_dtype_shape_and_raw_bytes() -> None:
    module = helper()
    array = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    pins = {
        "shape": [2, 2],
        "dtype": array.dtype.str,
        "sha256_c_order_bytes": hashlib.sha256(array.tobytes()).hexdigest(),
    }
    values = {"physics_model_id": "software-fixture", "physics_array_pins": {"coefficients": pins}}
    metadata = {"model_id": "software-fixture"}
    module.verify_physics({"coefficients": array}, metadata, values)
    for changed in (array.astype(np.float64), array.reshape(4), array + np.float32(1)):
        with pytest.raises(ValueError):
            module.verify_physics({"coefficients": changed}, metadata, values)


def completion_fixture(tmp_path: Path) -> tuple[Any, Path, Path, dict[str, Any], dict[str, Any]]:
    """Small software records exercise admission; these are not scientific results."""
    module = helper()
    fitting, fit = tmp_path / "fitting", tmp_path / "fit"
    fitting.mkdir()
    (fit / "sources/python/dpt").mkdir(parents=True)
    src = fit / "sources/python/dpt/software-fixture.py"
    src.write_text("original source identity\n")

    def h(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = fitting / "public-observations.json"
    manifest.write_text("{}")
    base: dict[str, Any] = {
        "status": "complete",
        "sources_unchanged": True,
        "recorded_files_unchanged": True,
        "output_sha256": {},
        "source_sha256": {"python/dpt/software-fixture.py": h(src)},
    }
    (fitting / "run.json").write_text(json.dumps(base))
    public: dict[str, Any] = {
        "config": {
            "pilot_steps": 2,
            "checkpoint_steps": [0, 1, 2],
            "spectral_mapping_step": 1e-6,
            "relative_mapping_tolerance": 1e-4,
            "maximum_mapped_fraction_displacement": 1e-5,
            "detector_shape_hw": [2, 2],
        },
        "inverse_grid": {"shape": [2, 2, 2]},
        "views": [{}, {}],
    }
    gate = {
        "initial_mapping": 100.0,
        "mapping_step": 1e-6,
        "relative_limit": 1e-4,
        "physical_displacement_limit": 1e-5,
        "absolute_threshold": 0.01,
        "additional_relative_tolerance": 0.0,
    }
    row = {"iteration": 1, "accepted": True, "objective_before": 4.0, "objective": 2.0}
    values = {
        "solver-result.json": {
            "accepted_steps": 1,
            "termination": "wall_time_budget",
            "poisson_count_predictions_refreshed": True,
            "reference_or_generating_mean_access": False,
            "field_units": "dimensionless fractions",
            "initial_objective": 4.0,
            "final_objective": 2.0,
            "callback_history": [row],
            "final_stationarity": {
                **gate,
                "final_mapping": 1.0,
                "relative_mapping": 0.01,
                "mapped_physical_displacement": 1e-6,
                "relative_passed": False,
                "physical_passed": True,
                "passed": False,
                "diagnostic_metric": "euclidean",
            },
        },
        "stationarity-gate.json": gate,
        "evaluation-trace.json": [{"kind": "evaluate", "gradient": True, "objective": 2.0}],
        "history/accepted-0001.json": row,
        "checkpoints/state-0000.json": {"iteration": 0, "objective": 4.0, "mapping": 100.0},
        "checkpoints/state-0001.json": {**row, "accepted_trial_bytes_equal": True},
    }
    for name, value in values.items():
        path = fit / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    for name in (
        "fields.npy",
        "gradient.npy",
        "checkpoints/fields-0000.npy",
        "checkpoints/fields-0001.npy",
    ):
        np.save(fit / name, np.full((2, 2, 2, 2), 0.25, dtype=np.float32))
    np.save(fit / "predictions.npy", np.ones((2, 3, 2, 2), dtype=np.float32))
    output_names = [
        *values,
        "fields.npy",
        "gradient.npy",
        "predictions.npy",
        "checkpoints/fields-0000.npy",
        "checkpoints/fields-0001.npy",
    ]
    record: dict[str, Any] = {
        **base,
        "output_sha256": {name: h(fit / name) for name in output_names},
        "configuration": {
            "method": "spectral_metric",
            "config": public["config"],
            "inverse_grid": public["inverse_grid"],
            "observation_run_sha256": h(fitting / "run.json"),
            "observation_manifest_sha256": h(manifest),
            "reference_or_generating_mean_access": False,
        },
    }
    (fit / "run.json").write_text(json.dumps(record))
    return module, fitting, fit, public, record


def test_completed_fit_rejects_changed_source_or_input_before_evaluation(tmp_path: Path) -> None:
    module, fitting, fit, public, record = completion_fixture(tmp_path)
    assert module.verify_completed_fit(fit, fitting, public)["status"] == "complete"
    src = fit / "sources/python/dpt/software-fixture.py"
    src.write_text("different but internally hash-consistent source\n")
    changed = copy.deepcopy(record)
    changed["source_sha256"]["python/dpt/software-fixture.py"] = hashlib.sha256(
        src.read_bytes()
    ).hexdigest()
    (fit / "run.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fit source differs"):
        module.verify_completed_fit(fit, fitting, public)
    changed["status"] = "running"
    (fit / "run.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="incomplete"):
        module.verify_completed_fit(fit, fitting, public)


@pytest.mark.parametrize(
    "name",
    [
        "fields.npy",
        "gradient.npy",
        "predictions.npy",
        "stationarity-gate.json",
        "evaluation-trace.json",
        "history/accepted-0001.json",
        "checkpoints/fields-0000.npy",
        "checkpoints/state-0000.json",
        "checkpoints/fields-0001.npy",
        "checkpoints/state-0001.json",
    ],
)
def test_completion_rejects_omissions_even_if_manifest_rehashed(tmp_path: Path, name: str) -> None:
    module, fitting, fit, public, record = completion_fixture(tmp_path)
    del record["output_sha256"][name]
    (fit / name).unlink()
    (fit / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="missing required outputs"):
        module.verify_completed_fit(fit, fitting, public)


@pytest.mark.parametrize("corruption", ["shape", "nonfinite", "negative", "simplex"])
def test_completion_rejects_invalid_fitted_arrays_before_reference_access(
    tmp_path: Path, corruption: str
) -> None:
    module, fitting, fit, public, record = completion_fixture(tmp_path)
    fields = np.full((2, 2, 2, 2), 0.25, dtype=np.float32)
    if corruption == "shape":
        fields = fields.reshape(2, 2, 4)
    else:
        fields.flat[0] = {"nonfinite": float("nan"), "negative": -1.0, "simplex": 0.9}[corruption]
    np.save(fit / "fields.npy", fields)
    record["output_sha256"]["fields.npy"] = hashlib.sha256(
        (fit / "fields.npy").read_bytes()
    ).hexdigest()
    (fit / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match=r"primary array|primary fractions"):
        module.verify_completed_fit(fit, fitting, public)


@pytest.mark.parametrize(
    "field,value",
    [
        ("termination", "converged"),
        ("accepted_steps", 2),
        ("final_objective", float("nan")),
        ("callback_history", []),
        ("poisson_count_predictions_refreshed", 1),
    ],
)
def test_completion_rejects_false_completion_metadata(
    tmp_path: Path, field: str, value: Any
) -> None:
    module, fitting, fit, public, record = completion_fixture(tmp_path)
    path = fit / "solver-result.json"
    result = json.loads(path.read_text())
    result[field] = value
    path.write_text(json.dumps(result))
    record["output_sha256"]["solver-result.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (fit / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        module.verify_completed_fit(fit, fitting, public)


@pytest.mark.parametrize("corruption", ["initial_chain", "final_objective", "final_checkpoint"])
def test_completion_rejects_inconsistent_accepted_states_before_reference_access(
    tmp_path: Path, corruption: str
) -> None:
    module, fitting, fit, public, record = completion_fixture(tmp_path)
    result_path = fit / "solver-result.json"
    result = json.loads(result_path.read_text())
    if corruption == "initial_chain":
        row = result["callback_history"][0]
        row["objective_before"] = 10.0
        (fit / "history/accepted-0001.json").write_text(json.dumps(row))
        (fit / "checkpoints/state-0001.json").write_text(
            json.dumps({**row, "accepted_trial_bytes_equal": True})
        )
    elif corruption == "final_objective":
        result["final_objective"] = 3.0
    else:
        np.save(fit / "checkpoints/fields-0001.npy", np.full((2, 2, 2, 2), 0.1, dtype=np.float32))
    result_path.write_text(json.dumps(result))
    for name in record["output_sha256"]:
        record["output_sha256"][name] = hashlib.sha256((fit / name).read_bytes()).hexdigest()
    (fit / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match=r"accepted history|final objective|accepted checkpoint"):
        module.verify_completed_fit(fit, fitting, public)
