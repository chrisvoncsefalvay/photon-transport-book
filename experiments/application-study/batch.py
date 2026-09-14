"""Execute every prescribed final fit and finite-candidate policy before evaluation."""

from __future__ import annotations

import argparse
import importlib
import itertools
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from _final import design_jobs, registration_jobs, require_outcome_coverage, view_id
from _study import (
    array_input,
    checked_array,
    digest,
    geometry,
    grid_spec,
    read_json,
    record_sources,
    registration_case,
    rigid,
    save_array,
    write_json,
)
from run import fit_case

from dpt._runtime import prepare_context
from dpt.examples.acquisition_design import candidate_information
from dpt.experiments import RunRecorder
from dpt.geometry import RigidTransform, compose_pose


def rank_design(known_root, observations_root, public, protocol, job, base_pose, output, device):
    """Known volume and accepted base pose only; no candidate counts are opened."""
    output.mkdir(exist_ok=False)
    known = read_json(known_root / "known.json")
    field = checked_array(known_root, known["attenuation"])
    torch = importlib.import_module("torch")
    wp = importlib.import_module("warp")
    wp.init()
    stream = torch.cuda.current_stream(torch.device(device))
    context = prepare_context(device=device, stream=wp.stream_from_torch(stream))
    beam_value = protocol["acquisition"]["base_and_candidate_open_beam_per_pixel"]
    beam = np.full(protocol["geometry"]["detector_shape_hw"], beam_value, dtype=np.float32)
    scales = np.array(protocol["numerics"]["chart_scales"], dtype=np.float64)
    with torch.cuda.stream(stream), context.scope():
        information, info = candidate_information(
            np=np,
            torch=torch,
            ctx=context,
            grid=grid_spec(known["grid"]),
            geometry=geometry(protocol, protocol["acquisition"]["base_view_degrees"]),
            attenuation=wp.array(field.ravel(), dtype=wp.float32, device=device),
            pose=wp.array(base_pose.packed(), dtype=wp.float64, device=device),
            beam_host=beam,
            scales=torch.as_tensor(scales, dtype=torch.float64, device=device),
            samples_per_ray=public["effective_fitting_samples"],
            precision=protocol["numerics"].get("precision", "float32"),
            integration=protocol["numerics"].get("integration", "midpoint"),
        )
    arrays = {"attenuation": array_input(known_root, known["attenuation"], known)}
    for name, array, unit in [
        ("precision", information, "dimensionless"),
        ("beam", beam, "photons/pixel"),
        (
            "targets",
            np.array(list(itertools.product((-40.0, 40.0), repeat=3)), dtype=np.float64),
            "mm",
        ),
    ]:
        stored = save_array(
            output,
            name + ".npy",
            array,
            units=unit,
            role="known design input; no future observations",
        )
        arrays[name] = array_input(output, stored, known)
    settings = {
        "grid": known["grid"],
        "pose": asdict(base_pose),
        "attenuation": "attenuation",
        "parameter_scales": scales.tolist(),
        "prior_precision": "precision",
        "targets_object_mm": "targets",
        "samples_per_ray": public["effective_fitting_samples"],
        "precision": protocol["numerics"].get("precision", "float32"),
        "integration": protocol["numerics"].get("integration", "midpoint"),
        "max_candidate_pixels": beam.size,
        "maximum_cost": float(beam.sum(dtype=np.float64)),
        "cost_unit": "expected_detector_photons",
        "maximum_precision_condition": protocol["acquisition"]["maximum_precision_condition"],
        "rank_relative_tolerance": protocol["acquisition"]["rank_relative_tolerance"],
        "current_information_description": (
            "base-view likelihood Fisher at accepted base pose, once; no ridge prior"
        ),
        "candidates": [
            {
                "name": f"a{round(angle) + 180:03d}",
                "geometry": asdict(geometry(protocol, angle)),
                "open_beam": "beam",
                "cost": float(beam.sum(dtype=np.float64)),
                "feasibility_description": "prespecified synthetic angular availability",
            }
            for angle in job["angles"]
        ],
    }
    case = output / "case.json"
    write_json(case, {"schema_version": 1, "arrays": arrays, "acquisition_design": settings})
    write_json(output / "base-fisher-diagnostics.json", info)
    started = time.perf_counter()
    with (output / "command.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "dpt.examples.acquisition_design",
                "--case",
                str(case),
                "--output",
                str(output / "run"),
                "--device",
                device,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
        )
    design = read_json(output / "run/design.json")
    record = read_json(output / "run/run.json")
    if (
        record["status"] != "complete"
        or digest(output / "run/design.json") != record["output_sha256"]["design.json"]
    ):
        raise ValueError("incomplete or changed candidate selection")
    return {
        "selected_angle": int(design["selected_candidate"][1:]) - 180,
        "design_run_sha256": digest(output / "run/run.json"),
        "design_result_sha256": digest(output / "run/design.json"),
        "wall_seconds": time.perf_counter() - started,
        "future_counts_access": False,
    }


def execute(known_root, observations_root, output, protocol_path, device):
    protocol = read_json(protocol_path)
    known = read_json(known_root / "known.json")
    public = read_json(observations_root / "public-observations.json")
    generation = read_json(observations_root / "run.json")
    if (
        generation["status"] != "complete"
        or digest(observations_root / "public-observations.json")
        != generation["output_sha256"]["public-observations.json"]
    ):
        raise ValueError("final observation generation is incomplete or changed")
    if (
        public["case"] not in protocol["data"]["final_cases"]
        or public["role"] != "final fitting observations only"
        or public["protocol_sha256"] != digest(protocol_path)
    ):
        raise ValueError("wrong final case, role or protocol identity")
    if public["known_sha256"] != digest(known_root / "known.json"):
        raise ValueError("known field manifest changed")
    started = time.perf_counter()
    rows = []
    selections = []
    tasks = registration_jobs(protocol)
    designs = design_jobs(protocol, known["case"])
    with RunRecorder(
        output,
        configuration={
            "scope": "final fitting and finite-view selection; reference access forbidden",
            "protocol": protocol,
            "known_sha256": digest(known_root / "known.json"),
            "observation_manifest_sha256": digest(observations_root / "public-observations.json"),
            "case": known["case"],
        },
        sources=record_sources(
            __file__,
            protocol_path,
            known=known_root / "known.json",
            observations=observations_root / "public-observations.json",
            generation=observations_root / "run.json",
        ),
    ) as recorder:
        (output / "cases").mkdir()
        (output / "fits").mkdir()
        (output / "designs").mkdir()
        recorder.write_json(
            "prespecified-jobs.json",
            {
                "registration": tasks,
                "acquisition": designs,
                "random_selection_before_count_access": True,
                "expected_fit_count": len(tasks) + 4 * len(designs),
            },
        )

        def fit(job, ids, anchor):
            began = time.perf_counter()
            name = job["name"]
            row = {**job, "fit_view_ids": ids, "output": f"fits/{name}"}
            try:
                path = registration_case(
                    known_root,
                    observations_root,
                    public,
                    protocol,
                    ids,
                    anchor,
                    output / f"cases/{name}.json",
                    samples=public["effective_fitting_samples"],
                )
                result = fit_case(
                    path,
                    output / f"fits/{name}",
                    device=device,
                    deadline=time.perf_counter() + protocol["numerics"]["solve_soft_seconds"],
                    profile=not rows,
                )
                row.update(
                    status="complete",
                    result_sha256=digest(output / f"fits/{name}/optimisation.json"),
                    run_sha256=digest(output / f"fits/{name}/run.json"),
                    termination=result["result"]["reason"],
                    stationary=result["stationary"],
                    final_objective=result["result"]["evaluation"]["loss"],
                    solve_seconds=result["solve_wall_seconds"],
                )
            except Exception as error:
                row.update(status="failed", error_type=type(error).__name__, error=str(error))
                result = None
            row["wall_seconds"] = time.perf_counter() - began
            rows.append(row)
            recorder.write_json(f"progress/{len(rows):03d}.json", row)
            print(
                f"case{known['case']} fit{len(rows):02d}/68 {name} {row['status']} "
                f"{row.get('termination', '')} {row['wall_seconds']:.2f}s "
                f"total{time.perf_counter() - started:.1f}s",
                flush=True,
            )
            return result

        for job in tasks:
            anchor = compose_pose(RigidTransform(), tuple(job["offset"]))
            fit(job, job["view_ids"], anchor)
        for job in designs:
            basejob = {**job, "name": job["name"] + "-base", "policy": "base"}
            base = fit(basejob, [job["base_id"]], RigidTransform())
            if base is None:
                for policy in ("selected", "fixed", "random"):
                    rows.append(
                        {
                            **job,
                            "name": job["name"] + "-" + policy,
                            "policy": policy,
                            "status": "blocked_by_base_failure",
                        }
                    )
                continue
            accepted = rigid(base["pose_object_to_world"])
            try:
                selection = rank_design(
                    known_root,
                    observations_root,
                    public,
                    protocol,
                    job,
                    accepted,
                    output / "designs" / job["name"],
                    device,
                )
                selection["status"] = "complete"
            except Exception as error:
                selection = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            # This receipt precedes opening the chosen future count array.
            recorder.write_json(
                f"selections/{job['name']}.json",
                {
                    **selection,
                    "base_result_sha256": digest(
                        output / f"fits/{basejob['name']}/optimisation.json"
                    ),
                    "job": job,
                },
            )
            selections.append({"job": job["name"], **selection})
            for policy in ("selected", "fixed", "random"):
                if policy == "selected" and selection["status"] != "complete":
                    rows.append(
                        {
                            **job,
                            "name": job["name"] + "-" + policy,
                            "policy": policy,
                            "status": "blocked_by_design_failure",
                        }
                    )
                    continue
                angle = (
                    selection["selected_angle"] if policy == "selected" else job[policy + "_angle"]
                )
                fit(
                    {
                        **job,
                        "name": job["name"] + "-" + policy,
                        "policy": policy,
                        "selected_angle": angle,
                    },
                    [job["base_id"], view_id(job["role"], job["replicate"], angle)],
                    accepted,
                )
        require_outcome_coverage(protocol, known["case"], rows)
        recorder.write_json(
            "batch-summary.json",
            {
                "status": "all prespecified outcomes retained",
                "case": known["case"],
                "rows": rows,
                "selections": selections,
                "expected_fit_count": len(tasks) + 4 * len(designs),
                "recorded_outcome_count": len(rows),
                "wall_seconds": time.perf_counter() - started,
                "reference_or_reserved_access": False,
                "failures_retained": True,
            },
        )
    print(f"Completed case{known['case']}: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("known", "observations", "output", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    execute(args.known, args.observations, args.output, args.protocol, args.device)


if __name__ == "__main__":
    main()
