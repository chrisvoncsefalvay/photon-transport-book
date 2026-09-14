"""Evaluate all frozen final comparisons against reserved poses and whole views."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from _final import require_outcome_coverage, view_id
from _study import (
    checked_array,
    digest,
    keyed_rng,
    read_json,
    record_sources,
    registration_case,
    rigid,
)

from dpt.examples._common import load_case, write_array
from dpt.examples.registration import SharedPoseObjective, prepare_views
from dpt.experiments import RunRecorder
from dpt.recovery import PoseChart
from dpt.validation.recovery import pose_error


def verified_record(folder: Path) -> dict:
    record = read_json(folder / "run.json")
    if (
        record["status"] != "complete"
        or not record["sources_unchanged"]
        or not record["recorded_files_unchanged"]
    ):
        raise ValueError(f"incomplete or changed producer: {folder}")
    recorded = {
        **record["output_sha256"],
        **{f"sources/{name}": value for name, value in record["source_sha256"].items()},
    }
    for name, expected in recorded.items():
        if digest(folder / name) != expected:
            raise ValueError(f"recorded output changed: {folder / name}")
    return record


def half_deviance(mean, observed):
    mean = np.asarray(mean, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    positive = observed > 0
    term = mean.copy()
    relative = (mean[positive] - observed[positive]) / observed[positive]
    term[positive] = observed[positive] * (relative - np.log1p(relative))
    return float(np.sum(term, dtype=np.float64))


def evaluate(known_roots, observation_roots, batches, output, protocol_path, device):
    protocol = read_json(protocol_path)
    started = time.perf_counter()
    if len(known_roots) != len(observation_roots) or len(batches) != len(known_roots):
        raise ValueError("supply corresponding known, observation and batch paths")
    # Complete/hash-check every compared fit in BOTH phantoms before opening any reference.
    inputs = []
    for known_root, obs_root, batch in zip(known_roots, observation_roots, batches, strict=True):
        batch_record = verified_record(batch)
        summary = read_json(batch / "batch-summary.json")
        if summary["recorded_outcome_count"] != summary["expected_fit_count"]:
            raise ValueError("a prespecified final outcome is missing")
        if batch_record["configuration"]["protocol"] != protocol:
            raise ValueError("final fitting protocol mismatch")
        known = read_json(known_root / "known.json")
        require_outcome_coverage(protocol, known["case"], summary["rows"])
        if digest(known_root / "known.json") != batch_record["configuration"]["known_sha256"]:
            raise ValueError("known field identity differs from fitting")
        if (
            digest(obs_root / "run.json") != batch_record["source_sha256"]["generation"]
            or digest(obs_root / "public-observations.json")
            != batch_record["configuration"]["observation_manifest_sha256"]
        ):
            raise ValueError("evaluation generator differs from the observations used for fitting")
        for row in summary["rows"]:
            if row["status"] == "complete":
                child = batch / row["output"]
                verified_record(child)
                if (
                    digest(child / "optimisation.json") != row["result_sha256"]
                    or digest(child / "run.json") != row["run_sha256"]
                ):
                    raise ValueError("fixed fit identity mismatch")
        inputs.append((known_root, obs_root, batch, known, summary))
    if sorted(x[3]["case"] for x in inputs) != sorted(protocol["data"]["final_cases"]):
        raise ValueError("all prespecified final cases must be fixed before any reference access")
    sources = record_sources(__file__, protocol_path)
    for _, obs_root, batch, known, _ in inputs:
        prefix = f"case{known['case']}"
        for name, path in [
            ("batch", batch / "run.json"),
            ("summary", batch / "batch-summary.json"),
            ("generation", obs_root / "run.json"),
            ("reference", obs_root / "generator-reference.json"),
            ("reserved", obs_root / "reserved-observations.json"),
        ]:
            sources[prefix + "/" + name + ".json"] = path
    results = []
    with RunRecorder(
        output,
        configuration={
            "scope": "independent evaluation only; no further fitting or selection",
            "protocol": protocol,
            "both_batch_hashes_fixed_before_reference_access": True,
        },
        sources=sources,
    ) as recorder:
        recorder.write_json(
            "fitting-freeze.json",
            {
                "batches": [
                    {
                        "case": known["case"],
                        "run_sha256": digest(batch / "run.json"),
                        "summary_sha256": digest(batch / "batch-summary.json"),
                    }
                    for _, _, batch, known, _ in inputs
                ]
            },
        )
        for known_root, obs_root, batch, known, summary in inputs:
            verified_record(obs_root)
            reference = read_json(obs_root / "generator-reference.json")
            reserved = read_json(obs_root / "reserved-observations.json")
            if reserved[
                "role"
            ] != "reserved observations; evaluation after batch freeze only" or reserved[
                "known_sha256"
            ] != digest(known_root / "known.json"):
                raise ValueError("reserved manifest role/known identity mismatch")
            target = rigid(reference["pose_object_to_world"])
            probes = tuple(tuple(v) for v in reference["geometric_probes_object_mm"])
            for row in summary["rows"]:
                record = {**row, "case": known["case"]}
                if row["status"] != "complete":
                    record["geometric_success"] = False
                    results.append(record)
                    continue
                fixed = read_json(batch / row["output"] / "optimisation.json")
                recovered = rigid(fixed["pose_object_to_world"])
                error = pose_error(recovered, target, probes)
                initial_error = pose_error(rigid(fixed["chart"]["anchor"]), target, probes)
                record.update(errors=asdict(error), initial_errors=asdict(initial_error))
                gate = protocol["registration"]["geometric_success"]
                record["geometric_success"] = (
                    error.landmark_rms_mm <= gate["rms_probe_error_mm_max"]
                    and error.landmark_max_mm <= gate["maximum_probe_error_mm_max"]
                    and math.degrees(error.rotation_radians) <= gate["rotation_error_degrees_max"]
                )
                record["stationary"] = fixed["stationary"]
                ids = [
                    view_id(row["role"], row["replicate"], a)
                    for a in protocol["evaluation"]["reserved_view_angles_degrees"]
                ]
                name = f"case{known['case']}/{row['name']}"
                case_path = output / "cases" / f"{name}.json"
                case_path.parent.mkdir(parents=True, exist_ok=True)
                registration_case(
                    known_root,
                    obs_root,
                    reserved,
                    protocol,
                    ids,
                    recovered,
                    case_path,
                    samples=reserved["effective_fitting_samples"],
                )
                case = load_case(case_path)
                chart_values = dict(case.config["registration"]["chart"])
                chart_values["anchor"] = recovered
                chart = PoseChart(**chart_values)
                views = prepare_views(case, case.config["registration"], chart, device)
                value = SharedPoseObjective(views, chart)((0.0,) * 6)
                residuals = []
                for identifier, view in zip(ids, views, strict=True):
                    supplied = reserved["views"][identifier]
                    observed = checked_array(obs_root, supplied["observation"]).astype(np.float64)
                    expected = checked_array(obs_root, supplied["generating_mean"])
                    prediction = view.evaluator.prediction.numpy().reshape(observed.shape)
                    write_array(recorder, f"predictions/{name}/{identifier}.npy", prediction)
                    residuals.append(
                        {
                            "view": identifier,
                            "pixels": observed.size,
                            "expected_error_poisson_sd_rms": float(
                                np.sqrt(
                                    np.mean(
                                        (prediction.astype(np.float64) - expected) ** 2 / expected
                                    )
                                )
                            ),
                            "expected_image_relative_l2": float(
                                np.linalg.norm(prediction - expected) / np.linalg.norm(expected)
                            ),
                            "observed_half_deviance": half_deviance(prediction, observed),
                            "generating_expectation_half_deviance": half_deviance(
                                expected, observed
                            ),
                            "reference_mean_sha256": supplied["generating_mean"]["sha256"],
                            "observed_sha256": supplied["observation"]["sha256"],
                        }
                    )
                record["reserved"] = residuals
                record["reserved_half_deviance"] = value.loss
                record["reserved_expectation_standardised_rms"] = float(
                    np.sqrt(np.mean([r["expected_error_poisson_sd_rms"] ** 2 for r in residuals]))
                )
                results.append(record)
                print(
                    f"evaluated case{known['case']} {row['name']} "
                    f"RMS{error.landmark_rms_mm:.5f}mm {row['termination']}",
                    flush=True,
                )
        paired = []
        for case in protocol["data"]["final_cases"]:
            for role in ("broad", "constrained"):
                for comparison in ("fixed", "random"):
                    for metric in ("geometric_rms_mm", "reserved_standardised_rms"):
                        differences = []
                        replicate_rows = []
                        count = (
                            protocol["acquisition"]["replicates_per_case"]
                            if role == "broad"
                            else protocol["acquisition"]["constrained"]["replicates_per_case"]
                        )
                        for replicate in range(count):
                            matches = {
                                r.get("policy"): r
                                for r in results
                                if r["case"] == case
                                and r["role"] == role
                                and r["replicate"] == replicate
                            }
                            a, b = matches.get("selected"), matches.get(comparison)
                            if (
                                not a
                                or not b
                                or a["status"] != "complete"
                                or b["status"] != "complete"
                            ):
                                replicate_rows.append(
                                    {"replicate": replicate, "status": "failed pair retained"}
                                )
                                continue
                            av = (
                                a["errors"]["landmark_rms_mm"]
                                if metric == "geometric_rms_mm"
                                else a["reserved_expectation_standardised_rms"]
                            )
                            bv = (
                                b["errors"]["landmark_rms_mm"]
                                if metric == "geometric_rms_mm"
                                else b["reserved_expectation_standardised_rms"]
                            )
                            differences.append(av - bv)
                            replicate_rows.append(
                                {
                                    "replicate": replicate,
                                    "selected": av,
                                    "comparison": bv,
                                    "difference": av - bv,
                                }
                            )
                        interval = None
                        if differences:
                            rng = keyed_rng(
                                protocol["evaluation"]["bootstrap_seed"],
                                case,
                                1 if role == "broad" else 2,
                                1 if comparison == "fixed" else 2,
                                1 if metric == "geometric_rms_mm" else 2,
                            )
                            draws = rng.choice(
                                np.array(differences),
                                size=(
                                    protocol["evaluation"]["bootstrap_resamples"],
                                    len(differences),
                                ),
                                replace=True,
                            ).mean(axis=1)
                            interval = np.quantile(draws, [0.025, 0.975]).tolist()
                        paired.append(
                            {
                                "case": case,
                                "role": role,
                                "comparison": comparison,
                                "metric": metric,
                                "direction": "selected minus comparison; negative favours selected",
                                "successful_pairs": len(differences),
                                "prespecified_pairs": count,
                                "mean_paired_difference": float(np.mean(differences))
                                if differences
                                else None,
                                "descriptive_percentile95_interval": interval,
                                "replicates": replicate_rows,
                                "scope": (
                                    "within-phantom paired noise only; failed pairs explicit; "
                                    "no patient-population inference"
                                ),
                            }
                        )
        recorder.write_json(
            "evaluation.json",
            {
                "results": results,
                "paired_acquisition": paired,
                "evaluated_rows": len(results),
                "wall_seconds": time.perf_counter() - started,
                "all_failed_rows_retained": True,
                "geometry_and_stationarity_separate": True,
                "no_fitting_after_reference_access": True,
            },
        )
    print(f"Final evaluation complete: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("known", "observations", "batch"):
        parser.add_argument("--" + name, type=Path, action="append", required=True)
    for name in ("output", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate(args.known, args.observations, args.batch, args.output, args.protocol, args.device)


if __name__ == "__main__":
    main()
