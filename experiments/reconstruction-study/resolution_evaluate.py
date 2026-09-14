"""CPU development evaluation only after the complete seven-outcome output freeze."""

from __future__ import annotations

import argparse
import importlib
import math
import statistics
import time
from pathlib import Path
from typing import Any, cast

from _resolution import (
    checked_file,
    checked_json,
    count_metrics,
    load_fitting_inputs,
    require_coverage,
    verify_prepared,
)
from _study import complete_record, digest, recorded_array
from evaluate_fields import evaluate_material_fields, evaluate_scalar_field, reference_rois
from probe_acquisition import helpers

from dpt.experiments import RunRecorder, experiment_sources, private_output, repository_root
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.validation.projection import integrate_sampled_field

np: Any = importlib.import_module("numpy")


def independent_prediction_check(
    fields: Any, predictions: Any, public: dict[str, Any], physics: dict[str, Any], pixels: Any
) -> dict[str, Any]:
    """Exact CPU trilinear integral of the actual accepted fields, not a reference fit."""
    library = helpers()
    grid = library.grid_from(public["inverse_grid"])
    geometry = DetectorGeometry(**{k: tuple(v) for k, v in public["geometry"].items()})
    spectral = fields.ndim == 4
    flat = fields.astype(np.float64).reshape((-1, grid.voxels))
    mu = physics["mu_mm_inv"].astype(np.float64)
    weights = physics["weights"].astype(np.float64) * physics["response"]
    exact: list[Any] = []
    for view in public["views"]:
        pose = RigidTransform(**{k: tuple(v) for k, v in view["pose"].items()})
        rays: list[Any] = []
        for row, column in pixels:
            paths = np.asarray(
                [
                    integrate_sampled_field(f, grid, geometry, pose, row, column, validate=False)
                    for f in flat
                ]
            )
            rays.append(
                weights @ np.exp(-paths @ mu)
                if spectral
                else np.array(
                    [public["config"]["incident_photons_per_ray"] * math.exp(-float(paths[0]))]
                )
            )
        exact.append(np.stack(rays, axis=1))
    oracle = np.stack(exact)
    rr, cc = np.asarray(pixels).T
    actual = predictions[:, :, rr, cc] if spectral else predictions[:, None, rr, cc]
    scale = (
        public["config"]["sampling_gate_photons_per_ray"]
        / public["config"]["incident_photons_per_ray"]
    )
    error = np.sqrt(scale) * np.abs(actual.astype(np.float64) - oracle) / np.sqrt(oracle)
    summaries = [
        {"p95_sd": float(np.quantile(error[:, c], 0.95)), "maximum_sd": float(error[:, c].max())}
        for c in range(error.shape[1])
    ]
    return {
        "oracle": oracle,
        "actual": actual,
        "per_channel": summaries,
        "passed": all(s["p95_sd"] < 0.05 and s["maximum_sd"] < 0.2 for s in summaries),
        "role": (
            "Actual final accepted field versus exact CPU trilinear integral; "
            "fixed 24 rays per evaluated view"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "fits", "original-acquisition", "config", "freeze", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    root = repository_root(__file__)
    receipt = verify_prepared(args.prepared, args.config, args.freeze)
    batch_record = complete_record(args.fits)
    batch = checked_json(args.fits, "batch.json", batch_record)
    require_coverage(batch["outcomes"])
    if batch_record["configuration"]["prepared_sha256"] != digest(args.prepared / "prepared.json"):
        raise ValueError("evaluation inputs differ from the batch's prepared acquisition")
    public = {
        name: load_fitting_inputs(args.prepared / name, root)[0] for name in receipt["bundles"]
    }
    records: dict[str, Any] = {}
    for row in batch["outcomes"]:
        folder = args.fits / row["id"]
        if "run_sha256" in row and digest(folder / "run.json") != row["run_sha256"]:
            raise ValueError("prescribed outcome run changed")
        if row["status"] != "recorded":
            continue
        record = complete_record(folder)
        for name in record["output_sha256"]:
            checked_file(folder, name, record["output_sha256"])
        for name in record["source_sha256"]:
            checked_file(folder / "sources", name, record["source_sha256"])
        group = row["group"]
        if (
            record["configuration"]["observation_run_sha256"]
            != receipt["bundles"][group]["run_sha256"]
            or record["configuration"]["observation_manifest_sha256"]
            != receipt["bundles"][group]["manifest_sha256"]
            or digest(folder / "solver-result.json") != row["result_sha256"]
        ):
            raise ValueError("fit is not bound to these exact observation bytes")
        records[row["id"]] = record
    qualification = complete_record(args.prepared / "qualification")
    original = complete_record(args.original_acquisition)
    if (
        digest(args.original_acquisition / "run.json")
        != qualification["configuration"]["source_acquisition_run_sha256"]
    ):
        raise ValueError("native generating means differ from the original dense acquisition")
    sources = experiment_sources(
        __file__,
        args.config,
        extra={
            "canonical-freeze.json": args.freeze,
            "prepared.json": args.prepared / "prepared.json",
            "fits/run.json": args.fits / "run.json",
            "fits/batch.json": args.fits / "batch.json",
            "qualification/run.json": args.prepared / "qualification/run.json",
            "original-acquisition/run.json": args.original_acquisition / "run.json",
            **{f"reconstruction-study/{p.name}": p for p in Path(__file__).parent.glob("*.py")},
        },
    )
    output = private_output(args.output, root)
    library = helpers()
    with RunRecorder(
        output,
        configuration={
            "scope": "development training-view evaluation; no reserved generalisation claim",
            "prepared_sha256": digest(args.prepared / "prepared.json"),
            "batch_run_sha256": digest(args.fits / "run.json"),
        },
        sources=sources,
    ) as run:
        # This record is written before the first reference/generating-mean NumPy load.
        run.write_json(
            "compared-output-freeze.json",
            {
                "outcomes": batch["outcomes"],
                "recorded_outputs": {name: rec["output_sha256"] for name, rec in records.items()},
                "reference_arrays_opened": False,
            },
        )
        # Only the evaluator reads all qualification payload bytes, after output freeze.
        for name in qualification["output_sha256"]:
            checked_file(args.prepared / "qualification", name, qualification["output_sha256"])
        references: dict[int, dict[str, Any]] = {}
        for n in (48, 64):
            references[n] = {
                kind: recorded_array(
                    args.prepared / "qualification",
                    f"evaluation/reference-{kind}{n}.npy",
                    qualification,
                )
                for kind in ("scalar", "fractions")
            }
        results: list[dict[str, Any]] = []
        for row in batch["outcomes"]:
            if row["status"] != "recorded":
                results.append(row)
                continue
            name, group, method = row["id"], row["group"], row["method"]
            folder = args.fits / name
            record = records[name]
            fields = recorded_array(folder, "fields.npy", record)
            predictions = recorded_array(folder, "predictions.npy", record)
            result = checked_json(folder, "solver-result.json", record)
            trace = checked_json(folder, "evaluation-trace.json", record)
            spectral = method.startswith("spectral_")
            p = public[group]
            grid = library.grid_from(p["inverse_grid"])
            n = grid.shape[0]
            ref = references[n]
            initial = (
                np.stack(
                    [np.ones(grid.shape, dtype=np.float32), np.zeros(grid.shape, dtype=np.float32)]
                )
                if spectral
                else np.full(
                    grid.shape,
                    p["config"]["scalar_coefficients_80kev_mm_inverse"][0],
                    dtype=np.float32,
                )
            )
            if spectral:
                coefficients = cast(
                    tuple[float, float], tuple(p["config"]["scalar_coefficients_80kev_mm_inverse"])
                )
                error_metrics = evaluate_material_fields(
                    fields, ref["fractions"], grid, mu80_mm_inverse=coefficients
                )
                initial_metrics = evaluate_material_fields(
                    initial, ref["fractions"], grid, mu80_mm_inverse=coefficients
                )
            else:
                rois = reference_rois(ref["fractions"], grid)
                error_metrics = evaluate_scalar_field(fields, ref["scalar"], grid, rois=rois)
                initial_metrics = evaluate_scalar_field(initial, ref["scalar"], grid, rois=rois)
            model = "spectral" if spectral else "scalar"
            counts = np.stack(
                [
                    np.load(args.prepared / group / view[f"{model}_counts"], allow_pickle=False)
                    for view in p["views"]
                ]
            )
            if group.startswith("dense"):
                means = recorded_array(
                    args.original_acquisition, f"evaluation/native-{model}-means.npy", original
                )
            else:
                regime = group.removesuffix("48")
                means = recorded_array(
                    args.prepared / "qualification",
                    f"qualification/{regime}-native/spectral-1024.npy",
                    qualification,
                )
            floor = recorded_array(
                args.prepared / "qualification",
                f"qualification/{group}/{model}-512.npy",
                qualification,
            )
            q = checked_json(
                args.prepared / "qualification", f"qualification/{group}.json", qualification
            )
            _, physics, _ = library.load_physics(args.prepared / group / "physics")
            independent = independent_prediction_check(
                fields, predictions, p, physics, q["independent_pixels_row_column"]
            )
            library.write_array(run, f"independent/{name}-oracle.npy", independent.pop("oracle"))
            library.write_array(run, f"independent/{name}-actual.npy", independent.pop("actual"))
            channels = range(predictions.shape[1]) if spectral else range(1)
            projection_metrics: list[dict[str, Any]] = []
            for c in channels:
                select = (slice(None), c) if spectral else (slice(None),)
                projection_metrics.append(
                    {
                        "channel": c,
                        "recovered": count_metrics(
                            predictions[select], counts[select], means[select]
                        ),
                        "inverse_reference_model_floor": count_metrics(
                            floor[select], counts[select], means[select]
                        ),
                    }
                )
            history = result["callback_history"]
            before_name, after_name = (
                ("objective_before", "objective") if spectral else ("loss_before", "loss_after")
            )
            previous = result["initial_objective"]
            history_consistent = len(history) == result["accepted_steps"]
            for index, accepted in enumerate(history, 1):
                before, after = accepted[before_name], accepted[after_name]
                history_consistent = history_consistent and (
                    accepted["iteration"] == index
                    and math.isfinite(before)
                    and math.isfinite(after)
                    and math.isclose(before, previous, rel_tol=1e-10, abs_tol=1e-7)
                    and after < before
                )
                previous = after
            history_consistent = history_consistent and (
                math.isfinite(result["final_objective"])
                and math.isclose(result["final_objective"], previous, rel_tol=1e-10, abs_tol=1e-7)
            )
            domain_passed = bool(
                np.isfinite(fields).all()
                and (fields >= 0).all()
                and (not spectral or (fields.sum(axis=0, dtype=np.float64) <= 1 + 2**-24).all())
            )
            gradient_times = [
                t["seconds"]
                for t in trace
                if t["kind"] == "evaluate" and t["phase"] == "profile" and t["gradient"]
            ]
            value_times = [
                t["seconds"]
                for t in trace
                if t["kind"] == "evaluate" and t["phase"] == "profile" and not t["gradient"]
            ]
            summary = {
                **row,
                "field_metrics": error_metrics,
                "initial_field_metrics": initial_metrics,
                "projection_metrics_per_channel": projection_metrics,
                "actual_field_prediction_check": independent,
                "field_domain": {
                    "passed": domain_passed,
                    "minimum": float(fields.min()),
                    "maximum": float(fields.max()),
                    "simplex_roundoff_allowance": 2**-24 if spectral else None,
                },
                "accepted_history_and_final_objective_consistent": history_consistent,
                "objective_recheck_tolerances": {"relative": 1e-10, "absolute": 1e-7},
                "hard_correctness_passed": bool(
                    domain_passed and history_consistent and independent["passed"]
                ),
                "accepted_backtracks": [r["backtracks"] for r in history],
                "median_warm_gradient_seconds": statistics.median(gradient_times),
                "median_warm_value_seconds": statistics.median(value_times),
                "trial_errors": [t for t in trace if "error" in t],
                "timing": {
                    k: result[k]
                    for k in (
                        "setup_seconds",
                        "profile_seconds",
                        "solve_seconds_including_internal_final_refresh",
                        "explicit_finalisation_seconds",
                        "checkpoint_seconds",
                        "longest_accepted_iteration_seconds",
                        "soft_budget_overshoot_seconds",
                    )
                },
                "reference_role": (
                    "Assigned CT-mask phantom cell averages; not patient composition truth"
                ),
                "no_reserved_views": True,
            }
            run.write_json(f"outcomes/{name}.json", summary)
            results.append(summary)
            print(f"Evaluated {name}: independent predictions {independent['passed']}", flush=True)
        run.write_json(
            "evaluation.json",
            {
                "status": "complete development evaluation",
                "outcomes": results,
                "all_recorded_actual_predictions_pass": all(
                    r.get("actual_field_prediction_check", {}).get("passed", False) for r in results
                ),
                "all_seven_hard_correctness_gates_pass": all(
                    r.get("hard_correctness_passed", False) for r in results
                ),
                "cpu_evaluation_seconds": time.perf_counter() - started,
                "elapsed_stage_seconds": time.perf_counter()
                - receipt["stage_start_monotonic_seconds"],
                "limits": [
                    "No final or withheld-view result",
                    "Short budgets do not establish convergence",
                    "Scalar/spectral losses are different observations",
                    "Sequential shared-device timings are workload-specific",
                ],
            },
        )


if __name__ == "__main__":
    main()
