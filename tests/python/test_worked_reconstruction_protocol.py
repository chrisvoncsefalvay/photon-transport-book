"""CPU checks of the worked example's data split and immutable input boundary."""

import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")
pytest.importorskip("warp")
ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "experiments/worked-reconstruction/run.py"


def driver() -> Any:
    names = ("_study", "_resolution", "probe_acquisition")
    previous = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    search_path = sys.path.copy()
    try:
        spec = importlib.util.spec_from_file_location("worked_reconstruction_protocol", DRIVER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = search_path
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


def test_fitting_and_withheld_poses_are_disjoint() -> None:
    module = driver()
    config = json.loads(DRIVER.with_name("protocol-v1.json").read_text())
    rows = module.trajectories(config)
    keys = {
        role: {(row["tilt_degrees"], row["yaw_degrees"]) for row in values}
        for role, values in rows.items()
    }
    assert len(keys["fitting"]) == 48
    assert len(keys["withheld"]) == 12
    assert not keys["fitting"] & keys["withheld"]


def test_noise_namespaces_separate_phase_replicate_and_role() -> None:
    module = driver()
    config = json.loads(DRIVER.with_name("protocol-v1.json").read_text())
    means = np.full((2, 3, 4, 4), 100.0, dtype=np.float64)
    namespaces: set[tuple[int, ...]] = set()
    for phase in (config["development_noise_phase"], config["final_noise_phase"]):
        for replicate in config["replicates"]:
            for role in (0, 1):
                observed, keys = module.poisson(means, config, phase, replicate, role)
                repeated, repeated_keys = module.poisson(means, config, phase, replicate, role)
                np.testing.assert_array_equal(observed, repeated)
                assert keys == repeated_keys
                assert observed.dtype == np.float32
                for key in keys:
                    assert tuple(key) not in namespaces
                    namespaces.add(tuple(key))


def test_recorded_metadata_must_match_its_hash(tmp_path: Path) -> None:
    module = driver()
    path = tmp_path / "fitting.json"
    path.write_text('{"development": false}')
    record = {"output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}
    assert module.checked_json(tmp_path, path.name, record) == {"development": False}
    path.write_text('{"development": true}')
    with pytest.raises(ValueError, match="metadata changed"):
        module.checked_json(tmp_path, path.name, record)


def test_physical_provenance_cannot_change_behind_identical_array_bytes(tmp_path: Path) -> None:
    module = driver()
    arrays, metadata = tmp_path / "physics.npz", tmp_path / "metadata.json"
    arrays.write_bytes(b"fixed test array identity")
    metadata.write_text('{"model_id": "first"}')
    public = {
        "physics_sha256": module.digest(arrays),
        "physics_metadata_sha256": module.digest(metadata),
    }
    module.require_physics_identity(tmp_path, public)
    metadata.write_text('{"model_id": "second"}')
    with pytest.raises(ValueError, match="provenance changed"):
        module.require_physics_identity(tmp_path, public)


def test_portable_derivative_requires_pinned_metadata_and_array() -> None:
    module = driver()
    folder = ROOT / "public/generated/worked-examples/inputs/reconstruction/anatomy"
    config = json.loads(DRIVER.with_name("protocol-v1.json").read_text())
    with pytest.raises(ValueError, match="derivative identity"):
        module.acquisition_anatomy(folder, config)
    config.update(
        coarse_anatomy_metadata_sha256="d3d1903022a2dc804fd3c2b7e940e08b82b84b084ced5e29958253933c9fcd71",
        coarse_fractions_sha256="5c8bd82c378f28830d9daf7af6338f6c631dfb9b4dba2b4a2fbd6a0e3d323c90",
    )
    anatomy, fields, grid = module.acquisition_anatomy(folder, config)
    assert fields.shape == (2, 16, 16, 16)
    assert grid.shape == (16, 16, 16)
    assert anatomy["original_forward_fractions_sha256"] == config["forward_fractions_sha256"]
    config["source_sha256"] = {}
    with pytest.raises(ValueError, match="source or crop"):
        module.acquisition_anatomy(folder, config)


@pytest.mark.parametrize(
    "passed,development,exit_code",
    [([True, True], False, 0), ([True, False], False, 1), ([True, True], True, 0)],
)
def test_complete_wrapper_freezes_every_fit_before_evaluation_and_gates_final_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passed: list[bool],
    development: bool,
    exit_code: int,
) -> None:
    """Test orchestration only; stand-ins are not physics or recovery evidence."""
    module = driver()
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps({"replicates": [0, 1]}))
    source = tmp_path / "control-flow-fixture.txt"
    source.write_text("Temporary orchestration fixture; no scientific measurements.\n")
    output = tmp_path / "outputs"
    calls: list[tuple[str, int | None]] = []

    def fixture_sources(_config: Path, _extra: dict[str, Path]) -> dict[str, Path]:
        return {"fixture.txt": source}

    monkeypatch.setattr(module, "sources", fixture_sources)

    def record(args: Any, role: str, replicate: int | None) -> None:
        with module.RunRecorder(
            args.output,
            configuration={"role": role, "replicate": replicate},
            sources={"fixture.txt": source},
        ) as run:
            run.write_json("fixture.json", {"purpose": "orchestration test only"})

    def acquire(args: Any) -> None:
        calls.append(("acquire", None))
        record(args, "acquire", None)

    def fit(args: Any) -> None:
        calls.append(("fit", args.replicate))
        assert args.acquisition == output / "acquisition"
        assert (args.acquisition / "run.json").is_file()
        record(args, "fit", args.replicate)

    def evaluate(args: Any) -> dict[str, Any]:
        assert calls[:3] == [("acquire", None), ("fit", 0), ("fit", 1)]
        for replicate in (0, 1):
            frozen = module.complete_record(output / f"fit-rep{replicate}")
            assert frozen["status"] == "complete"
        fitted = module.complete_record(args.fit)
        replicate = fitted["configuration"]["replicate"]
        calls.append(("evaluate", replicate))
        record(args, "evaluate", replicate)
        return {
            "replicate": replicate,
            "development": development,
            "numerical_passed": passed[replicate],
            "accepted": passed[replicate] and not development,
        }

    monkeypatch.setattr(module, "acquire", acquire)
    monkeypatch.setattr(module, "fit", fit)
    monkeypatch.setattr(module, "evaluate", evaluate)
    arguments = [str(DRIVER), "all", "--config", str(config), "--output", str(output)]
    if development:
        arguments.append("--development")
    monkeypatch.setattr(sys, "argv", arguments)
    if exit_code:
        with pytest.raises(SystemExit) as error:
            module.main()
        assert error.value.code == exit_code
    else:
        module.main()
    assert calls == [("acquire", None), ("fit", 0), ("fit", 1), ("evaluate", 0), ("evaluate", 1)]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["numerical_passed"] is all(passed)
    assert summary["accepted"] is (all(passed) and not development)
    assert len(summary["child_records_sha256"]) == 5
    for name, expected in summary["child_records_sha256"].items():
        assert module.digest(output / name) == expected


@pytest.mark.parametrize("replicates", [[], [0], [0, 0], [0, 1, 2]])
def test_complete_wrapper_rejects_missing_or_changed_replicates_before_acquisition(
    tmp_path: Path, replicates: list[int]
) -> None:
    import argparse

    module = driver()
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps({"replicates": replicates}))
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="replicates"):
        module.run_worked_example(
            argparse.Namespace(config=config, output=output, development=False)
        )
    assert not output.exists()


class _StageClock:
    """Deterministic orchestration clock; no measured performance is represented."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


class _StageSolver:
    """Control-flow fixture, deliberately not a material solver or scientific evidence."""

    def __init__(
        self,
        settings: Any,
        clock: _StageClock,
        plans: list[tuple[str, int, float]],
        update_seconds: float = 0.0,
    ) -> None:
        self.settings = settings
        self.clock = clock
        self.plans = plans
        self.update_seconds = update_seconds
        self.accepted = 0
        self.calls: list[dict[str, Any]] = []

    def solve(self, callback: Callable[[int, dict[str, Any]], None]) -> dict[str, Any]:
        termination, steps, final_seconds = self.plans[len(self.calls)]
        self.calls.append(asdict(self.settings))
        assert steps <= self.settings.iterations
        history: list[dict[str, Any]] = []
        for iteration in range(1, steps + 1):
            self.accepted += 1
            self.clock.now += self.update_seconds
            row = {
                "iteration": iteration,
                "accepted": True,
                "objective": 100 - self.accepted,
                "gradient_mapping_before": 20,
            }
            history.append(row)
            callback(iteration, row)
        if termination == "line_search_failed":
            history.append({"iteration": steps + 1, "accepted": False, "failure": termination})
        self.clock.now += final_seconds
        return {"termination": termination, "accepted_steps": steps, "history": history}

    def fractions_numpy(self) -> Any:
        return np.full((2, 1, 1, 1), self.accepted, dtype=np.float32)


def _run_stage_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plans: list[tuple[str, int, float]],
    *,
    iterations: int = 7,
    refresh_steps: list[int] | None = None,
    seconds: float = 100.0,
    update_seconds: float = 0.0,
    refresh_seconds: float = 0.0,
) -> tuple[dict[str, Any], _StageSolver, list[tuple[str, int]], dict[str, int]]:
    """Exercise actual driver recording with explicitly synthetic control states."""
    module = driver()
    clock = _StageClock()
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=clock.perf_counter))
    settings = module.MaterialReconstructionSettings(
        iterations=iterations,
        gradient_mapping_tolerance=10.0,
        relative_gradient_mapping_tolerance=0.0,
        mapping_step=1e-5,
        step_selection="bb",
        acceleration="inertial",
    )
    solver = _StageSolver(settings, clock, plans, update_seconds)
    refreshes: list[tuple[str, int]] = []
    checkpoints: dict[str, int] = {}

    def prepare_metric(current: _StageSolver, _run: Any, prefix: str) -> None:
        assert current is solver
        refreshes.append((prefix, current.accepted))
        clock.now += refresh_seconds

    def write_array(_run: Any, name: str, values: Any) -> None:
        assert name not in checkpoints
        checkpoints[name] = int(values[0, 0, 0, 0])

    monkeypatch.setattr(module, "prepare_metric", prepare_metric)
    monkeypatch.setattr(module.LIB, "write_array", write_array)
    source = tmp_path / "control-flow-fixture.txt"
    source.write_text("Synthetic orchestration states; not material recovery evidence.\n")
    with module.RunRecorder(
        tmp_path / "record",
        configuration={"purpose": "control-flow test only"},
        sources={"fixture.txt": source},
    ) as run:
        result = module.solve_stages(
            solver,
            run,
            {"checkpoint_steps": [1, 2, 5, 7]},
            iterations,
            seconds,
            [2, 5] if refresh_steps is None else refresh_steps,
        )
    return result, solver, refreshes, checkpoints


def test_metric_stages_preserve_gate_global_updates_and_boundary_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, solver, refreshes, checkpoints = _run_stage_fixture(
        tmp_path,
        monkeypatch,
        [
            ("iteration_budget", 2, 0),
            ("iteration_budget", 3, 0),
            ("projected_gradient_tolerance", 2, 0),
        ],
    )
    assert result["termination"] == "projected_gradient_tolerance"
    assert result["accepted_steps"] == solver.accepted == 7
    assert [row["iteration"] for row in result["history"]] == list(range(1, 8))
    assert [row["stage_iteration"] for row in result["history"]] == [1, 2, 1, 2, 3, 1, 2]
    assert [row["stage"] for row in result["history"]] == [0, 0, 1, 1, 1, 2, 2]
    assert [stage["accepted_updates_before"] for stage in result["stages"]] == [0, 2, 5]
    assert [call["iterations"] for call in solver.calls] == [2, 3, 2]
    original = {key: value for key, value in solver.calls[0].items() if key != "iterations"}
    assert original["gradient_mapping_tolerance"] == 10.0
    assert original["relative_gradient_mapping_tolerance"] == 0.0
    assert all(
        {key: value for key, value in call.items() if key != "iterations"} == original
        for call in solver.calls
    )
    assert refreshes == [("stages/after-0002/", 2), ("stages/after-0005/", 5)]
    assert checkpoints == {f"checkpoints/fields-{step:04d}.npy": step for step in [1, 2, 5, 7]}
    recorded = sorted((tmp_path / "record/history").glob("accepted-*.json"))
    assert len(recorded) == 7
    assert [json.loads(path.read_text())["iteration"] for path in recorded] == list(range(1, 8))


@pytest.mark.parametrize("termination", ["projected_gradient_tolerance", "line_search_failed"])
def test_metric_stages_do_not_refresh_after_stationarity_or_failed_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, termination: str
) -> None:
    result, solver, refreshes, _ = _run_stage_fixture(tmp_path, monkeypatch, [(termination, 1, 0)])
    assert result["termination"] == termination
    assert result["accepted_steps"] == 1
    assert len(solver.calls) == 1 and not refreshes
    stage = json.loads((tmp_path / "record/stages/stage-00.json").read_text())
    if termination == "line_search_failed":
        assert stage["history"][-1]["accepted"] is False
        assert stage["history"][-1]["failure"] == termination
        assert all(row["accepted"] for row in result["history"])


def test_metric_stages_without_refresh_keep_one_original_solve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, solver, refreshes, _ = _run_stage_fixture(
        tmp_path, monkeypatch, [("iteration_budget", 7, 0)], refresh_steps=[]
    )
    assert result["accepted_steps"] == 7
    assert [call["iterations"] for call in solver.calls] == [7]
    assert not refreshes


def test_metric_stages_share_callback_budget_across_refreshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, solver, refreshes, _ = _run_stage_fixture(
        tmp_path,
        monkeypatch,
        [("iteration_budget", 2, 0), ("iteration_budget", 3, 0)],
        seconds=3.0,
        update_seconds=1.0,
    )
    assert result["termination"] == "wall_time_budget"
    assert result["accepted_steps"] == solver.accepted == 3
    assert len(solver.calls) == 2
    assert refreshes == [("stages/after-0002/", 2)]
    assert [row["wall_seconds"] for row in result["history"]] == [1.0, 2.0, 3.0]
    stage = json.loads((tmp_path / "record/stages/stage-01.json").read_text())
    assert stage["accepted_steps"] == 1 and stage["termination"] == "wall_time_budget"


@pytest.mark.parametrize("refresh_seconds,final_seconds", [(0.0, 3.0), (3.0, 0.0)])
def test_metric_stages_check_budget_before_and_after_metric_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refresh_seconds: float,
    final_seconds: float,
) -> None:
    result, solver, refreshes, _ = _run_stage_fixture(
        tmp_path,
        monkeypatch,
        [("iteration_budget", 2, final_seconds)],
        seconds=3.0,
        refresh_seconds=refresh_seconds,
    )
    assert result["termination"] == "wall_time_budget"
    assert result["accepted_steps"] == solver.accepted == 2
    assert len(solver.calls) == 1
    assert refreshes == ([] if final_seconds else [("stages/after-0002/", 2)])


def _observation_fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Small hash-bound arrays for admission tests, never recovery evidence."""
    module = driver()
    config = json.loads(DRIVER.with_name("protocol-v3.json").read_text())
    pins: dict[str, str] = {}
    for replicate in (0, 1):
        for folder, role in (("fitting", "fitting"), ("evaluation", "withheld")):
            name = f"{folder}/rep{replicate}-{role}-counts.npy"
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.full((1, 1, 1, 1), 20 + replicate, dtype=np.float32))
            pins[name] = module.digest(path)
    config["final_observation_sha256"] = pins
    return config, pins


@pytest.mark.parametrize("malformation", ["missing", "incomplete", "extra", "uppercase", "list"])
def test_precision_repair_requires_exact_four_original_observation_pins(
    tmp_path: Path, malformation: str
) -> None:
    module = driver()
    config, pins = _observation_fixture(tmp_path)
    if malformation == "missing":
        del config["final_observation_sha256"]
    elif malformation == "incomplete":
        pins.pop(next(iter(pins)))
    elif malformation == "extra":
        pins["another.npy"] = "a" * 64
    elif malformation == "uppercase":
        pins[next(iter(pins))] = "A" * 64
    else:
        config["final_observation_sha256"] = list(pins)
    with pytest.raises(ValueError, match="all four original final count arrays"):
        module.observation_pins(config)


@pytest.mark.parametrize("role", ["fitting", "evaluation"])
@pytest.mark.parametrize("replicate", [0, 1])
def test_final_observation_admission_checks_every_array_before_fitting(
    tmp_path: Path, role: str, replicate: int
) -> None:
    module = driver()
    config, pins = _observation_fixture(tmp_path)
    identity = module.verify_observation_identity(tmp_path, config, False)
    assert identity["count_files_sha256"] == pins
    assert identity["matches_preserved_final_observations"] is True
    suffix = "fitting" if role == "fitting" else "withheld"
    target = tmp_path / f"{role}/rep{replicate}-{suffix}-counts.npy"
    np.save(target, np.full((1, 1, 1, 1), 999, dtype=np.float32))
    with pytest.raises(ValueError, match="differ from the preserved final observations"):
        module.verify_observation_identity(tmp_path, config, False)
    development = module.verify_observation_identity(tmp_path, config, True)
    assert development["development"] is True
    assert development["matches_preserved_final_observations"] is False
    assert development["count_files_sha256"] != pins


def test_precision_repair_keeps_generation_seed_namespaces_gates_and_budget() -> None:
    module = driver()
    old_path = DRIVER.with_name("protocol-v2.json")
    old = json.loads(old_path.read_text())
    new = json.loads(DRIVER.with_name("protocol-v3.json").read_text())
    assert new["precision_repair"]["previous_protocol_sha256"] == module.digest(old_path)
    assert {key: value for key, value in new.items() if key in old and key != "id"} == {
        key: value for key, value in old.items() if key != "id"
    }
    assert set(new) - set(old) == {
        "generation_precision",
        "calculation_precision",
        "precision_repair",
        "final_observation_sha256",
    }
    assert new["generation_precision"] == "float32"
    assert new["calculation_precision"] == "float64"
    assert module.observation_pins(old) == {}
    with pytest.raises(ValueError, match="generation must retain"):
        module.observation_pins({**new, "generation_precision": "float64"})
    with pytest.raises(ValueError, match="calculation precision"):
        module.observation_pins({**new, "calculation_precision": "float16"})
    invalid_pin_values: tuple[object, ...] = ({}, None, [])
    for invalid_pins in invalid_pin_values:
        with pytest.raises(ValueError, match="all four original final count arrays"):
            module.observation_pins({**old, "final_observation_sha256": invalid_pins})


@pytest.mark.parametrize(
    "failure", [None, "counts", "identity", "inconsistent_identity", "development", "generation"]
)
def test_fit_checks_bound_receipt_and_own_counts_without_opening_withheld_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    """Real recording/admission; stop before physics loading or CUDA preparation."""
    module = driver()
    fixture_dir = tmp_path / "input-fixture"
    config, pins = _observation_fixture(fixture_dir)
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps(config))
    physics = tmp_path / "physics"
    physics.mkdir()
    (physics / "physics.npz").write_bytes(b"admission fixture, not a physics archive")
    (physics / "metadata.json").write_text('{"purpose":"admission fixture"}')
    public = {
        "protocol": config,
        "protocol_sha256": module.digest(config_path),
        "development": False,
        "physics_sha256": module.digest(physics / "physics.npz"),
        "physics_metadata_sha256": module.digest(physics / "metadata.json"),
    }
    identity = module.verify_observation_identity(fixture_dir, config, False)
    if failure == "inconsistent_identity":
        identity["matches_preserved_final_observations"] = False
    if failure == "development":
        identity["development"] = True
    if failure == "generation":
        identity["generation_precision"] = "float64"
    acquisition = tmp_path / "acquisition"
    name = "fitting/rep0-fitting-counts.npy"
    with module.RunRecorder(
        acquisition,
        configuration={"purpose": "admission fixture, not scientific output"},
        sources={"fixture-protocol.json": config_path},
    ) as run:
        run.write_json("fitting.json", public)
        run.write_json("qualification.json", {"passed": True})
        run.write_json("observation-identity.json", identity)
        run.write_bytes(name, (fixture_dir / name).read_bytes())
    assert not (acquisition / "evaluation").exists()
    if failure == "counts":
        np.save(acquisition / name, np.full((1, 1, 1, 1), 888, dtype=np.float32))
    if failure == "identity":
        (acquisition / "observation-identity.json").write_text("{}")
    assert module.digest(fixture_dir / name) == pins[name]
    reads: list[str] = []
    original = module.checked_array

    def checked(folder: Path, requested: str, record: dict[str, Any]) -> Any:
        reads.append(requested)
        assert requested == name
        return original(folder, requested, record)

    class ReachedPhysicsBoundaryError(Exception):
        pass

    def stop_before_physics(_folder: Path) -> None:
        raise ReachedPhysicsBoundaryError

    monkeypatch.setattr(module, "checked_array", checked)
    monkeypatch.setattr(module.LIB, "load_physics", stop_before_physics)
    args = SimpleNamespace(
        acquisition=acquisition, config=config_path, replicate=0, physics=physics
    )
    with pytest.raises(ReachedPhysicsBoundaryError if failure is None else ValueError):
        module.fit(args)
    assert reads == [name]
