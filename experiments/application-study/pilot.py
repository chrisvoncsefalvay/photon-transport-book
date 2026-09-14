"""Bounded development fit, measured finite-candidate ranking and downstream refits."""

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
from evaluate import geometric_errors
from run import fit_case

from dpt._runtime import prepare_context
from dpt.examples.acquisition_design import candidate_information
from dpt.experiments import RunRecorder
from dpt.geometry import RigidTransform, compose_pose


def run_pilot(
    prepared: Path, observations: Path, output: Path, protocol_path: Path, device: str
) -> None:
    protocol = read_json(protocol_path)
    known = read_json(prepared / "known.json")
    public = read_json(observations / "public-observations.json")
    if (
        public["role"] != "development only"
        or known["case"] != protocol["data"]["development_case"]
    ):
        raise ValueError("pilot inputs must be development only")
    sampling = read_json(observations / "sampling.json")
    if sampling["status"] != "passed":
        raise ValueError("development sampling checks did not pass")
    protocol["numerics"]["fitting_samples_per_ray"] = public["effective_fitting_samples"]
    start = time.perf_counter()
    deadline = start + protocol["pilot"]["max_solve_seconds"]
    wp = importlib.import_module("warp")
    torch = importlib.import_module("torch")
    memory_before = torch.cuda.mem_get_info(device)
    with RunRecorder(
        output,
        configuration={
            "scope": "development pilot; no final evaluation",
            "protocol": protocol,
            "total_soft_seconds": protocol["pilot"]["max_solve_seconds"],
        },
        sources=record_sources(
            __file__,
            protocol_path,
            known=prepared / "known.json",
            observations=observations / "public-observations.json",
            sampling=observations / "sampling.json",
        ),
    ) as run:
        cases = output / "cases"
        cases.mkdir()
        initial = compose_pose(
            RigidTransform(), tuple(protocol["registration"]["initial_offsets_local_se3_mm_rad"][0])
        )
        results = {}
        for label, ids, anchor, profile in (
            ("orthogonal", ["reg-a180", "reg-a270"], initial, True),
            ("base", ["design-a180"], RigidTransform(), False),
        ):
            if time.perf_counter() >= deadline:
                break
            case = registration_case(
                prepared, observations, public, protocol, ids, anchor, cases / f"{label}.json"
            )
            results[label] = fit_case(
                case, output / label, device=device, deadline=deadline, profile=profile
            )
            print(
                label,
                results[label]["result"]["reason"],
                results[label]["result"]["evaluation"]["loss"],
                flush=True,
            )
        design_seconds = None
        if "base" in results and time.perf_counter() < deadline:
            accepted = rigid(results["base"]["pose_object_to_world"])
            field = checked_array(prepared, known["attenuation"])
            stream = torch.cuda.current_stream(torch.device(device))
            ctx = prepare_context(device=device, stream=wp.stream_from_torch(stream))
            samples = public["effective_fitting_samples"]
            beam = np.full(protocol["geometry"]["detector_shape_hw"], 1000.0, dtype=np.float32)
            scales = np.asarray(protocol["numerics"]["chart_scales"], dtype=np.float64)
            with torch.cuda.stream(stream), ctx.scope():
                devfield = wp.array(field.ravel(), dtype=wp.float32, device=device)
                devpose = wp.array(accepted.packed(), dtype=wp.float64, device=device)
                prior, _ = candidate_information(
                    np=np,
                    torch=torch,
                    ctx=ctx,
                    grid=grid_spec(known["grid"]),
                    geometry=geometry(protocol, 0.0),
                    attenuation=devfield,
                    pose=devpose,
                    beam_host=beam,
                    scales=torch.as_tensor(scales, dtype=torch.float64, device=device),
                    samples_per_ray=samples,
                )
            arrays = {"attenuation": array_input(prepared, known["attenuation"], known)}
            for name, value, units in (
                ("precision", prior, "dimensionless"),
                ("beam", beam, "photons/pixel"),
                ("task-points", np.array(list(itertools.product((-40.0, 40.0), repeat=3))), "mm"),
            ):
                r = save_array(cases, f"{name}.npy", value, units=units, role="known design input")
                arrays[name] = array_input(cases, r, known)
            config = {
                "schema_version": 1,
                "arrays": arrays,
                "acquisition_design": {
                    "grid": known["grid"],
                    "pose": asdict(accepted),
                    "attenuation": "attenuation",
                    "parameter_scales": scales.tolist(),
                    "prior_precision": "precision",
                    "targets_object_mm": "task-points",
                    "samples_per_ray": samples,
                    "max_candidate_pixels": beam.size,
                    "maximum_cost": float(beam.sum(dtype=np.float64)),
                    "cost_unit": "expected_detector_photons",
                    "maximum_precision_condition": protocol["acquisition"][
                        "maximum_precision_condition"
                    ],
                    "current_information_description": (
                        "base-view likelihood Fisher at accepted current pose, once; no ridge prior"
                    ),
                    "candidates": [
                        {
                            "name": f"a{round(a) + 180:03d}",
                            "geometry": asdict(geometry(protocol, a)),
                            "open_beam": "beam",
                            "cost": float(beam.sum(dtype=np.float64)),
                            "feasibility_description": "supplied synthetic angular availability",
                        }
                        for a in protocol["acquisition"]["candidate_angles_degrees"]
                    ],
                },
            }
            design_case = cases / "design.json"
            write_json(design_case, config)
            began = time.perf_counter()
            command = [
                sys.executable,
                "-m",
                "dpt.examples.acquisition_design",
                "--case",
                str(design_case),
                "--output",
                str(output / "design"),
                "--device",
                device,
            ]
            with (output / "design-command.log").open("x") as log:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=max(0.1, deadline - time.perf_counter()),
                )
            design_seconds = time.perf_counter() - began
            choice = read_json(output / "design/design.json")["selected_candidate"]
            # The selection is now hash-bound. Only then form cases for future observations.
            run.write_json(
                "selection-before-new-count-access.json",
                {
                    "selected": choice,
                    "design_run_sha256": digest(output / "design/run.json"),
                    "design_result_sha256": digest(output / "design/design.json"),
                },
            )
            for label, view in (("selected", choice), ("fixed", "a270")):
                if time.perf_counter() >= deadline:
                    break
                case = registration_case(
                    prepared,
                    observations,
                    public,
                    protocol,
                    ["design-a180", f"design-{view}"],
                    accepted,
                    cases / f"{label}.json",
                )
                results[label] = fit_case(case, output / label, device=device, deadline=deadline)
                print(
                    label,
                    results[label]["result"]["reason"],
                    results[label]["result"]["evaluation"]["loss"],
                    flush=True,
                )
        # Development reference opens only after every executed fit is fixed.
        evaluation = {
            label: geometric_errors(output / label, observations / "generator-reference.json")
            for label in results
        }
        run.write_json(
            "pilot-report.json",
            {
                "status": "completed development budget",
                "wall_seconds": time.perf_counter() - start,
                "soft_budget_seconds": protocol["pilot"]["max_solve_seconds"],
                "design_driver_wall_seconds": design_seconds,
                "memory_free_total_before": list(memory_before),
                "memory_free_total_after": list(torch.cuda.mem_get_info(device)),
                "memory_scope": (
                    "shared device before/after observations; not peak or exclusive allocation"
                ),
                "sampling": sampling,
                "evaluation": evaluation,
                "fit_summary": {
                    k: {
                        "termination": v["result"]["reason"],
                        "evaluations": v["result"]["evaluations"],
                        "solve_wall_seconds": v["solve_wall_seconds"],
                        "initial_objective": v["initial_objective"],
                        "final_objective": v["result"]["evaluation"]["loss"],
                    }
                    for k, v in results.items()
                },
            },
        )
    print(f"Pilot complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("protocol-v1.json")
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_pilot(args.prepared, args.observations, args.output, args.protocol, args.device)


if __name__ == "__main__":
    main()
