"""Four bounded development fits with explicit accepted-checkpoint recording."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from _study import (
    PilotBudgetError,
    complete_record,
    digest,
    recorded_array,
    scalar_case,
    stationarity_gate,
    stationarity_result,
)
from baseline import PostLogReconstruction
from probe_acquisition import helpers

from dpt.examples._common import load_case
from dpt.examples.reconstruction import Reconstruction
from dpt.experiments import RunRecorder, experiment_sources, private_output, repository_root
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_reconstruction import (
    MaterialReconstruction,
    MaterialReconstructionSettings,
    MaterialReconstructionView,
)

np: Any = importlib.import_module("numpy")
wp: Any = importlib.import_module("warp")
torch: Any = importlib.import_module("torch")


class ScalarTrace:
    """Observe calls only; canonical solve controls every accepted update."""

    trace: list[dict[str, Any]]
    phase: str

    def evaluate(self, volume: Any, volume_wp: Any, *, gradient: bool) -> float:
        tick = time.perf_counter()
        row: dict[str, Any] = {"kind": "evaluate", "gradient": gradient, "phase": self.phase}
        try:
            value = cast(Any, super()).evaluate(volume, volume_wp, gradient=gradient)
            row["objective"] = value
            return value
        except Exception as error:
            row["error"] = repr(error)
            raise
        finally:
            row["seconds"] = time.perf_counter() - tick
            self.trace.append(row)

    def displacement(self, step: float) -> tuple[float, float]:
        tick = time.perf_counter()
        result = cast(Any, super()).displacement(step)
        self.trace.append(
            {
                "kind": "proposal_or_mapping",
                "phase": self.phase,
                "step": step,
                "mapping": result[0],
                "rounded_slope": result[1],
                "seconds": time.perf_counter() - tick,
            }
        )
        return result


class RecordedPoisson(ScalarTrace, Reconstruction):
    pass


class RecordedPWLS(ScalarTrace, PostLogReconstruction):
    pass


class RecordedMaterial(MaterialReconstruction):
    trace: list[dict[str, Any]]
    phase: str

    def evaluate(self, *, gradient: bool = True, trial: bool = False) -> float:
        tick = time.perf_counter()
        row: dict[str, Any] = {
            "kind": "evaluate",
            "gradient": gradient,
            "trial": trial,
            "phase": self.phase,
        }
        try:
            value = super().evaluate(gradient=gradient, trial=trial)
            row["objective"] = value
            return value
        except Exception as error:
            row["error"] = repr(error)
            raise
        finally:
            row["seconds"] = time.perf_counter() - tick
            self.trace.append(row)

    def projected_trial(self, step: float, *, use_metric: bool = True) -> tuple[float, float]:
        tick = time.perf_counter()
        row: dict[str, Any] = {
            "kind": "proposal" if use_metric else "mapping",
            "phase": self.phase,
            "step": step,
        }
        try:
            result = super().projected_trial(step, use_metric=use_metric)
            row.update(mapping=result[0], rounded_slope=result[1])
            return result
        except Exception as error:
            row["error"] = repr(error)
            raise
        finally:
            row["seconds"] = time.perf_counter() - tick
            self.trace.append(row)


def run_method(
    output: Path,
    method: str,
    acquisition: Path,
    public: dict[str, Any],
    source_inputs: dict[str, Path],
    case_path: Path,
    device: str,
    *,
    scope: str = "development cost screen; not final reconstruction evidence",
) -> dict[str, Any]:
    library = helpers()
    config = public["config"]
    spectral = method.startswith("spectral_")
    sources = experiment_sources(
        __file__,
        extra={
            **source_inputs,
            **{f"reconstruction-study/{p.name}": p for p in Path(__file__).parent.glob("*.py")},
            "helpers/spectral-run.py": Path(library.__file__),
        },
    )
    case: Any = None
    if not spectral:
        case = load_case(case_path)
        sources["scalar-case.json"] = case_path
        sources["initial-scalar.npy"] = case_path.parent / "initial.npy"
    configuration: dict[str, Any] = {
        "method": method,
        "config": config,
        "inverse_grid": public["inverse_grid"],
        "observation_manifest_sha256": digest(acquisition / "public-observations.json"),
        "observation_run_sha256": digest(acquisition / "run.json"),
        "reference_or_generating_mean_access": False,
        "scope": scope,
    }
    history: list[dict[str, Any]] = []
    checkpoint_costs: list[dict[str, Any]] = []
    solve_start = 0.0
    last_callback = 0.0
    longest_iteration = 0.0
    result: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    with RunRecorder(output, configuration=configuration, sources=sources) as run:
        started = time.perf_counter()
        if spectral:
            grid = library.grid_from(public["inverse_grid"])
            geometry = DetectorGeometry(**{k: tuple(v) for k, v in public["geometry"].items()})
            _, physics, spec = library.load_physics(acquisition / "physics")
            entries = tuple(
                MaterialReconstructionView(
                    geometry=geometry,
                    pose=RigidTransform(**{k: tuple(v) for k, v in v["pose"].items()}),
                    counts=np.load(acquisition / v["spectral_counts"], allow_pickle=False),
                    weights=physics["weights"],
                    response=physics["response"],
                )
                for v in public["views"]
            )
            initial = np.zeros((2, *grid.shape), dtype=np.float32)
            initial[0] = 1.0
            settings = MaterialReconstructionSettings(
                iterations=config["pilot_steps"],
                initial_step=config["spectral_initial_step"],
                maximum_step=config["spectral_maximum_step"],
                regularisation_mm_inverse=config["fraction_regularisation_mm_inverse"],
                gradient_mapping_tolerance=0.0,
                relative_gradient_mapping_tolerance=0.0,
                mapping_step=config["spectral_mapping_step"],
            )
            matrix = public["metric"]["matrix"]
            metric = (
                (matrix[0][0], matrix[0][1], matrix[1][1]) if method == "spectral_metric" else None
            )
            engine: Any = RecordedMaterial(
                grid=grid,
                views=entries,
                coefficients=physics["mu_mm_inv"],
                spectral_spec=spec,
                initial_fractions=initial,
                settings=settings,
                samples_per_ray=public["inverse_samples"],
                device=device,
                material_metric=metric,
            )
        else:
            kind = RecordedPoisson if method == "scalar_poisson" else RecordedPWLS
            engine = kind(case, device=device, torch=torch, wp=wp)
            grid = engine.grid
        trace: list[dict[str, Any]] = []
        engine.trace, engine.phase = trace, "setup"

        def fields_numpy() -> Any:
            return (
                engine.fractions_numpy()
                if spectral
                else engine.mu.detach().cpu().numpy().reshape(grid.shape).copy()
            )

        def evaluate_gradient() -> float:
            return (
                engine.evaluate(gradient=True)
                if spectral
                else engine.evaluate(engine.mu, engine.mu_wp, gradient=True)
            )

        step = (
            config["spectral_mapping_step"]
            if spectral
            else config["scalar_initial_and_mapping_step_mm_inverse_squared"]
        )

        def mapping() -> float:
            return (
                engine.projected_trial(step, use_metric=False)[0]
                if spectral
                else engine.displacement(step)[0]
            )

        initial_loss = evaluate_gradient()
        initial_mapping = mapping()
        physical_limit = (
            config["maximum_mapped_fraction_displacement"]
            if spectral
            else config["maximum_mapped_scalar_displacement_water_ratio"]
            * config["scalar_coefficients_80kev_mm_inverse"][0]
        )
        gate = stationarity_gate(
            initial_mapping, step, config["relative_mapping_tolerance"], physical_limit
        )
        if spectral:
            engine.settings = replace(
                engine.settings, gradient_mapping_tolerance=gate["absolute_threshold"]
            )
        else:
            engine.policy = replace(
                engine.policy, gradient_mapping_tolerance_mm=gate["absolute_threshold"]
            )
        run.write_json("stationarity-gate.json", gate)
        library.write_array(run, "checkpoints/fields-0000.npy", fields_numpy())
        run.write_json(
            "checkpoints/state-0000.json",
            {"iteration": 0, "objective": initial_loss, "mapping": initial_mapping},
        )
        run.set_metadata(
            gpu_name=wp.get_device(device).name,
            device=device,
            torch_version=torch.__version__,
            settings=asdict(engine.settings if spectral else engine.policy),
            device_memory_free_total_bytes=list(torch.cuda.mem_get_info(device)),
            torch_allocator_bytes={
                "allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device),
            },
            memory_scope="shared-device snapshot; Torch counters exclude Warp-owned buffers",
        )
        setup_seconds = time.perf_counter() - started

        # Paired real evaluations at the same immutable start; no backend changes.
        engine.phase = "profile"
        profile_start = time.perf_counter()
        for _ in range(2):
            evaluate_gradient()
            if spectral:
                engine.evaluate(gradient=False)
            else:
                engine.evaluate(engine.mu, engine.mu_wp, gradient=False)
        profile_seconds = time.perf_counter() - profile_start

        def accepted(iteration: int, row: dict[str, Any]) -> None:
            nonlocal last_callback, longest_iteration
            now = time.perf_counter()
            longest_iteration = max(longest_iteration, now - last_callback)
            last_callback = now
            if iteration != len(history) + 1:
                raise ValueError("accepted callback IDs are not contiguous")
            before = row["objective_before"] if spectral else row["loss_before"]
            after = row["objective"] if spectral else row["loss_after"]
            if not math.isfinite(after) or after >= before:
                raise ValueError("accepted callback is not a finite decrease")
            # Evaluate trace's final trial value is the exact value accepted by solve.
            evaluations = [r for r in engine.trace if r["kind"] == "evaluate"]
            if evaluations[-1].get("objective") != after or evaluations[-1]["gradient"]:
                raise ValueError("callback loss differs from the accepted trial evaluation")
            row = {**row, "charged_wall_seconds": now - solve_start}
            history.append(row)
            tick = time.perf_counter()
            run.write_json(f"history/accepted-{iteration:04d}.json", row)
            if iteration in config["checkpoint_steps"]:
                actual = fields_numpy()
                trial = (
                    engine.trial.numpy().reshape(actual.shape)
                    if spectral
                    else engine.trial.detach().cpu().numpy().reshape(actual.shape)
                )
                if not np.array_equal(actual, trial):
                    raise ValueError("accepted checkpoint bytes differ from the accepted trial")
                library.write_array(run, f"checkpoints/fields-{iteration:04d}.npy", actual)
                run.write_json(
                    f"checkpoints/state-{iteration:04d}.json",
                    {**row, "accepted_trial_bytes_equal": True},
                )
            checkpoint_costs.append({"iteration": iteration, "seconds": time.perf_counter() - tick})
            print(
                f"{method} {iteration}/{config['pilot_steps']}: objective {after:.8g}, "
                f"elapsed {time.perf_counter() - solve_start:.2f}s",
                flush=True,
            )
            if time.perf_counter() - solve_start >= config["pilot_soft_seconds_per_fit"]:
                raise PilotBudgetError("accepted callback reached the frozen soft time budget")

        engine.phase = "solve"
        solve_start = last_callback = time.perf_counter()
        budget_stop = False
        try:
            result = engine.solve(callback=accepted)
        except PilotBudgetError:
            budget_stop = True
            result = {
                "termination": "wall_time_budget",
                "accepted_steps": len(history),
                "history": history,
                "rejection_record": "all calls/errors retained in evaluation-trace.json",
            }
        solve_seconds = time.perf_counter() - solve_start
        engine.phase = "finalisation"
        final_start = time.perf_counter()
        # Ordinary solve already refreshes its accepted gradient; budget exits require it here.
        final_loss = evaluate_gradient() if budget_stop else result["final_objective"]
        final_mapping = mapping()
        fields = fields_numpy()
        if (
            not np.isfinite(fields).all()
            or (fields < 0).any()
            or (spectral and (fields.sum(axis=0, dtype=np.float64) > 1 + 2**-24).any())
        ):
            raise ValueError("final accepted fields violate the declared constraint")
        if spectral or method == "scalar_pwls":
            predictions = np.stack(engine.predictions_numpy())
        else:
            predictions = np.stack(
                [v["prediction"].numpy().reshape(v["shape"]) for v in engine.views]
            )
        library.write_array(run, "fields.npy", fields)
        library.write_array(run, "predictions.npy", predictions)
        library.write_array(
            run,
            "gradient.npy",
            engine.gradient.numpy().reshape(fields.shape)
            if spectral
            else engine.gradient.detach().cpu().numpy().reshape(fields.shape),
        )
        finalisation_seconds = time.perf_counter() - final_start
        diagnostic = stationarity_result(final_mapping, gate)
        result.update(
            final_objective=final_loss,
            final_stationarity=diagnostic,
            initial_objective=initial_loss,
            callback_history=history,
            setup_seconds=setup_seconds,
            profile_seconds=profile_seconds,
            solve_seconds_including_internal_final_refresh=solve_seconds,
            explicit_finalisation_seconds=finalisation_seconds,
            checkpoint_seconds=sum(r["seconds"] for r in checkpoint_costs),
            checkpoint_costs=checkpoint_costs,
            longest_accepted_iteration_seconds=longest_iteration,
            soft_budget_overshoot_seconds=max(
                0.0, solve_seconds - config["pilot_soft_seconds_per_fit"]
            ),
            field_units="dimensionless fractions" if spectral else "mm^-1",
            reference_or_generating_mean_access=False,
            poisson_count_predictions_refreshed=True,
        )
        if method == "scalar_pwls":
            result["post_log_preprocessing"] = engine.preprocessing
        run.write_json("evaluation-trace.json", engine.trace)
        run.write_json("solver-result.json", result)
        summary = {
            "method": method,
            "termination": result["termination"],
            "accepted_steps": result["accepted_steps"],
            "initial_objective": initial_loss,
            "final_objective": final_loss,
            "stationarity": diagnostic,
            "solve_seconds": solve_seconds,
        }
    engine = None
    gc.collect()
    torch.cuda.empty_cache()
    return {
        **summary,
        "run_sha256": digest(output / "run.json"),
        "result_sha256": digest(output / "solver-result.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    record = complete_record(args.acquisition)
    public_path = args.acquisition / "public-observations.json"
    if digest(public_path) != record["output_sha256"]["public-observations.json"]:
        raise ValueError("fitting manifest hash differs from acquisition")
    public = json.loads(public_path.read_text())
    if public["reference_access_permitted_for_fitting"] or public["inverse_grid"]["shape"] != [
        32,
        32,
        32,
    ]:
        raise ValueError("this driver only supports the prescribed 32-cubed development screen")
    source_inputs = {
        "acquisition/run.json": args.acquisition / "run.json",
        "acquisition/public-observations.json": public_path,
    }
    for view in public["views"]:
        for model in ("scalar", "spectral"):
            name = view[f"{model}_counts"]
            values = recorded_array(args.acquisition, name, record)
            if (
                digest(args.acquisition / name) != view["sha256"][model]
                or (values < 0).any()
                or (values % 1 != 0).any()
                or (values > 2**24).any()
            ):
                raise ValueError("invalid fitting count bytes")
            source_inputs[f"acquisition/{name}"] = args.acquisition / name
    for name, expected in {
        public["open_beam"]: public["open_beam_sha256"],
        **{f"physics/{k}": v for k, v in public["physics"].items()},
    }.items():
        if digest(args.acquisition / name) != expected or expected != record["output_sha256"][name]:
            raise ValueError(f"fixed physical input changed: {name}")
        source_inputs[f"acquisition/{name}"] = args.acquisition / name
    root = repository_root(__file__)
    for name, expected in record["source_sha256"].items():
        if name.startswith("python/dpt/") and digest(root / name) != expected:
            raise ValueError(f"canonical source differs from observation freeze: {name}")
    output = private_output(args.output, root)
    output.mkdir(parents=True, exist_ok=False)
    case_path = scalar_case(output / "scalar-input", args.acquisition, public, public["config"])
    wp.init()
    outcomes: list[dict[str, Any]] = []
    for method in public["config"]["pilot_methods"]:
        outcomes.append(
            run_method(
                output / method,
                method,
                args.acquisition,
                public,
                source_inputs,
                case_path,
                args.device,
            )
        )
    if {r["method"] for r in outcomes} != {
        "scalar_poisson",
        "scalar_pwls",
        "spectral_metric",
        "spectral_euclidean",
    } or len(outcomes) != 4:
        raise ValueError("the prescribed four-method outcome coverage is incomplete")
    (output / "pilot.json").write_text(
        json.dumps(
            {
                "status": "complete bounded development screen",
                "outcomes": outcomes,
                "reference_access": False,
                "larger_study_executed": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Development pilot complete: {output}", flush=True)


if __name__ == "__main__":
    main()
