"""Complete a CT-derived registration and then select, observe and fit a new view.

All projection, derivatives and optimisation use the canonical dpt CUDA code.
The known case is deliberately a teaching example, not a held-out clinical study.
Future noisy observations are generated only after the view decision is saved.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dpt.experiments import RunRecorder, experiment_sources, private_output, repository_root
from dpt.geometry import RigidTransform, compose_pose
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth
from dpt.validation.projection import integrate_sampled_field
from dpt.validation.recovery import detector_coordinate, pose_error

# Reuse the supplied-input experiment orchestration; there is no second solver.
APPLICATION = Path(__file__).resolve().parents[1] / "application-study"
sys.path.insert(0, str(APPLICATION))
from _study import (  # noqa: E402
    checked_array,
    digest,
    geometry,
    grid_spec,
    keyed_rng,
    read_json,
    registration_case,
    rigid,
    save_array,
)
from batch import rank_design  # noqa: E402


def _load_fit():
    # This entrypoint is itself called run.py: load its sibling driver by path.
    spec = importlib.util.spec_from_file_location("application_fit_driver", APPLICATION / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fit_case


def validate_protocol(protocol):
    """Reject changes that would invalidate this experiment's shared contracts."""
    acquisition = protocol["acquisition"]
    angles = acquisition["candidate_angles_degrees"]
    if protocol["task_coordinates_mm"] != [-40.0, 40.0]:
        raise ValueError("the reused design driver requires the eight ±40-mm task corners")
    if acquisition["base_view_degrees"] != 0.0:
        raise ValueError("this worked sequence starts from the declared zero-degree base view")
    if (
        not angles
        or any(not math.isfinite(a) or a != round(a) for a in angles)
        or len(set(angles)) != len(angles)
        or acquisition["fixed_view_degrees"] not in angles
    ):
        raise ValueError(
            "candidate angles must be distinct integral degrees including the baseline"
        )
    if protocol["randomness"]["replicates"] != 8:
        raise ValueError("the paired summary is declared for eight independent replicates")
    if (
        protocol["numerics"]["policy"]["gradient_tolerance"]
        != protocol["acceptance"]["gradient_tolerance"]
    ):
        raise ValueError("the solver and acceptance gradient thresholds must agree")
    if set(protocol["held_out_views_degrees"]) & {
        0.0,
        *angles,
        *protocol["registration"]["views_degrees"],
    }:
        raise ValueError("reserved views must be disjoint from every fitting candidate")


def _project(wp, field_device, grid, protocol, angle, pose, samples, device, *, generating=False):
    geom = geometry(protocol, angle)
    numerics = protocol["numerics"]
    precision = numerics["generation_precision" if generating else "precision"]
    integration = numerics["generation_integration" if generating else "integration"]
    workspace = prepare_projection(
        grid,
        geom,
        ProjectionSpec(samples, precision=precision, integration=integration),
        device=device,
    )
    output = wp.empty(geom.pixels, dtype=workspace.dtype, device=device)
    packed = wp.array(pose.packed(), dtype=wp.float64, device=device)
    project_optical_depth(field_device, packed, workspace=workspace, out_L=output)
    return output.numpy().reshape(geom.shape).astype(np.float64)


def _sampling(wp, field, field_device, grid, protocol, target, angles, device):
    """Qualify every prospective angle before producing any noisy data."""
    shape = protocol["geometry"]["detector_shape_hw"]
    rows = tuple(min(shape[0] - 1, round(shape[0] * x)) for x in (0.125, 0.375, 0.625, 0.875))
    cols = tuple(min(shape[1] - 1, round(shape[1] * x)) for x in (0, 0.25, 0.4375, 0.5, 0.75, 1))
    pixels = list(itertools.product(rows, cols))
    values = field.ravel().tolist()
    means = {}
    diagnostics = []
    cfg = protocol["numerics"]
    beam = cfg["sampling_gate_open_beam_per_pixel"]
    for angle in angles:
        geom = geometry(protocol, angle)
        corners = [
            grid.grid_to_object(tuple(index))
            for index in itertools.product(
                (-0.5, grid.shape[2] - 0.5),
                (-0.5, grid.shape[1] - 0.5),
                (-0.5, grid.shape[0] - 0.5),
            )
        ]
        coverage = [detector_coordinate(geom, target.point(corner)) for corner in corners]
        if any(
            not (-0.5 <= c <= geom.shape[1] - 0.5 and -0.5 <= r <= geom.shape[0] - 0.5)
            for c, r in coverage
        ):
            raise ValueError("the detector does not cover the declared transformed field")
        generated = _project(
            wp,
            field_device,
            grid,
            protocol,
            angle,
            target,
            cfg["generation_samples_per_ray"],
            device,
            generating=True,
        )
        fitted = _project(
            wp,
            field_device,
            grid,
            protocol,
            angle,
            target,
            cfg["fitting_samples_per_ray"],
            device,
        )
        exact = np.array(
            [
                integrate_sampled_field(
                    values, grid, geometry(protocol, angle), target, r, c, validate=False
                )
                for r, c in pixels
            ]
        )
        exact_mean = beam * np.exp(-exact)
        independent_error = np.abs(
            beam * np.exp(-np.array([generated[r, c] for r, c in pixels])) - exact_mean
        ) / np.sqrt(exact_mean)
        fitting_independent_error = np.abs(
            beam * np.exp(-np.array([fitted[r, c] for r, c in pixels])) - exact_mean
        ) / np.sqrt(exact_mean)
        reference_mean = beam * np.exp(-generated)
        full_error = np.abs(beam * np.exp(-fitted) - reference_mean) / np.sqrt(reference_mean)
        row = {
            "angle_degrees": angle,
            "projected_support_corners_column_row": coverage,
            "pixels": pixels,
            "independent_exact_depths": exact.tolist(),
            "generation_independent_p95_sd": float(np.quantile(independent_error, 0.95)),
            "generation_independent_max_sd": float(independent_error.max()),
            "generation_integration": cfg["generation_integration"],
            "fitting_integration": cfg["integration"],
            "fitting_independent_p95_sd": float(np.quantile(fitting_independent_error, 0.95)),
            "fitting_independent_max_sd": float(fitting_independent_error.max()),
            "fitting_full_image_p95_sd": float(np.quantile(full_error, 0.95)),
            "fitting_full_image_max_sd": float(full_error.max()),
        }
        row["passed"] = (
            max(
                row["generation_independent_p95_sd"],
                row["fitting_independent_p95_sd"],
                row["fitting_full_image_p95_sd"],
            )
            <= cfg["sampling_error_poisson_sd_p95_max"]
            and max(
                row["generation_independent_max_sd"],
                row["fitting_independent_max_sd"],
                row["fitting_full_image_max_sd"],
            )
            <= cfg["sampling_error_poisson_sd_max"]
        )
        diagnostics.append(row)
        means[angle] = np.exp(-generated)
        print(f"sampling angle={angle:g} passed={row['passed']}", flush=True)
    return means, diagnostics


def _observation(record, public, means, protocol, angle, beam, role, replicate, seed):
    code = round(angle) + 180
    identifier = f"r{role}-n{replicate:02d}-a{code:03d}"
    if identifier in public["views"]:
        return identifier
    counts64 = keyed_rng(seed, role, replicate, code).poisson(beam * means[angle])
    if counts64.max() > 2**24:
        raise ValueError("generated counts exceed exact binary32 integers")
    counts = counts64.astype(np.float32)
    mask = np.ones(counts.shape, dtype=np.float32)
    observation = save_array(
        record.output,
        f"observations/{identifier}.npy",
        counts,
        units="counts",
        role="simulated fitting observation",
        recorder=record,
    )
    mask_record = save_array(
        record.output,
        f"masks/{identifier}.npy",
        mask,
        units="dimensionless",
        role="fixed binary fitting mask",
        recorder=record,
    )
    public["views"][identifier] = {
        "angle_degrees": angle,
        "geometry": asdict(geometry(protocol, angle)),
        "open_beam_counts": beam,
        "observation": observation,
        "mask": mask_record,
    }
    return identifier


def _fit(known, output, public, protocol, name, identifiers, anchor, device):
    case_path = output / "cases" / f"{name}.json"
    registration_case(known, output, public, protocol, identifiers, anchor, case_path)
    fit_case = _load_fit()
    result = fit_case(
        case_path,
        output / "fits" / name,
        device=device,
        deadline=time.perf_counter() + protocol["numerics"]["solve_soft_seconds"],
    )
    print(
        f"fit {name}: {result['result']['reason']} "
        f"gradient={max(map(abs, result['result']['evaluation']['gradient'])):.6g}",
        flush=True,
    )
    return result


def _assess(wp, field_device, grid, protocol, means, target, outcome, points, device):
    recovered = rigid(outcome["pose_object_to_world"])
    errors = asdict(pose_error(recovered, target, tuple(map(tuple, points))))
    initial = asdict(
        pose_error(rigid(outcome["chart"]["anchor"]), target, tuple(map(tuple, points)))
    )
    reserved = []
    beam = protocol["acquisition"]["base_and_candidate_open_beam_per_pixel"]
    for angle in protocol["held_out_views_degrees"]:
        depth = _project(
            wp,
            field_device,
            grid,
            protocol,
            angle,
            recovered,
            protocol["numerics"]["fitting_samples_per_ray"],
            device,
        )
        expected = beam * means[angle]
        discrepancy = (beam * np.exp(-depth) - expected) / np.sqrt(expected)
        reserved.append(
            {
                "angle_degrees": angle,
                "expected_error_poisson_sd_rms": float(np.sqrt(np.mean(discrepancy**2))),
            }
        )
    gate = protocol["acceptance"]
    # pose_error exposes radians; the protocol intentionally uses reader-facing degrees.
    geometric = (
        errors["landmark_rms_mm"] <= gate["rms_probe_error_mm_max"]
        and errors["landmark_max_mm"] <= gate["maximum_probe_error_mm_max"]
        and math.degrees(errors["rotation_radians"]) <= gate["rotation_error_degrees_max"]
    )
    return {
        "stationary": outcome["stationary"],
        "termination": outcome["result"]["reason"],
        "errors": errors,
        "initial_errors": initial,
        "held_out": reserved,
        "geometric_pass": geometric,
        "held_out_pass": all(
            x["expected_error_poisson_sd_rms"] <= gate["held_out_expected_error_poisson_sd_rms_max"]
            for x in reserved
        ),
    }


def execute(known_root, output, config, device, development=False):
    import warp as wp

    wp.init()
    known_root = known_root.resolve()
    protocol = read_json(config)
    validate_protocol(protocol)
    known = read_json(known_root / "known.json")
    if (
        known["case"] != protocol["case"]
        or known["attenuation"]["sha256"] != protocol["known_attenuation_sha256"]
    ):
        raise ValueError("this worked example requires its declared CT-derived field")
    field = checked_array(known_root, known["attenuation"])
    grid = grid_spec(known["grid"])
    target = compose_pose(RigidTransform(), tuple(protocol["generating_local_se3_mm_rad"]))
    points = list(itertools.product(protocol["task_coordinates_mm"], repeat=3))
    angles = sorted(
        set(
            [
                *protocol["registration"]["views_degrees"],
                protocol["acquisition"]["base_view_degrees"],
                *protocol["acquisition"]["candidate_angles_degrees"],
                *protocol["held_out_views_degrees"],
            ]
        )
    )
    replicates = 1 if development else protocol["randomness"]["replicates"]
    seed = protocol["randomness"]["development_root_seed" if development else "root_seed"]
    sources = experiment_sources(
        __file__,
        config,
        extra={
            "known.json": known_root / "known.json",
            "known-attenuation.npy": known_root / known["attenuation"]["file"],
            **{f"application-study/{p.name}": p for p in APPLICATION.glob("*.py")},
        },
    )
    output = private_output(output, repository_root(__file__))
    with RunRecorder(
        output,
        configuration={
            "protocol": protocol,
            "development": development,
            "replicates": replicates,
            "root_seed": seed,
            "scope": "CT-derived simulated teaching example; not clinical validation",
        },
        sources=sources,
    ) as record:
        record.set_metadata(device=str(wp.get_device(device)), warp=wp.__version__)
        with wp.ScopedDevice(device):
            field_device = wp.array(field.ravel(), dtype=wp.float32, device=device)
            means, sampling = _sampling(
                wp, field, field_device, grid, protocol, target, angles, device
            )
            record.write_json("sampling.json", sampling)
            if not all(x["passed"] for x in sampling):
                raise ValueError("prospective quadrature checks failed; no observations generated")
            record.write_json(
                "evaluation-reference.json",
                {
                    "pose_object_to_world": asdict(target),
                    "task_points_object_mm": points,
                    "role": "generation/evaluation only; not supplied to fit or selector",
                },
            )
            public = {
                "known_sha256": digest(known_root / "known.json"),
                "views": {},
                "effective_fitting_samples": protocol["numerics"]["fitting_samples_per_ray"],
            }
            (output / "cases").mkdir()
            (output / "fits").mkdir()
            reg_ids = [
                _observation(
                    record,
                    public,
                    means,
                    protocol,
                    a,
                    protocol["registration"]["open_beam_per_view"],
                    1,
                    0,
                    seed,
                )
                for a in protocol["registration"]["views_degrees"]
            ]
            # region book:worked-registration
            registration = _fit(
                known_root,
                output,
                public,
                protocol,
                "registration",
                reg_ids,
                RigidTransform(),
                device,
            )
            # endregion book:worked-registration
            results = []
            # region book:worked-application-sequence
            for replicate in range(replicates):
                beam = protocol["acquisition"]["base_and_candidate_open_beam_per_pixel"]
                base_id = _observation(
                    record, public, means, protocol, 0.0, beam, 2, replicate, seed
                )
                base = _fit(
                    known_root,
                    output,
                    public,
                    protocol,
                    f"base-{replicate:02d}",
                    [base_id],
                    RigidTransform(),
                    device,
                )
                accepted = rigid(base["pose_object_to_world"])
                decision_path = output / f"decision-{replicate:02d}"
                selection = rank_design(
                    known_root,
                    output,
                    public,
                    protocol,
                    {
                        "angles": protocol["acquisition"]["candidate_angles_degrees"],
                    },
                    accepted,
                    decision_path,
                    device,
                )
                # Selection is on disk before either future count image exists.
                record.write_json(f"decision-{replicate:02d}-binding.json", selection)
                pair = {"replicate": replicate, "base": base, "selection": selection}
                for policy, angle in (
                    ("selected", selection["selected_angle"]),
                    ("near_parallel", protocol["acquisition"]["fixed_view_degrees"]),
                ):
                    future_id = _observation(
                        record, public, means, protocol, angle, beam, 3, replicate, seed
                    )
                    pair[policy] = _fit(
                        known_root,
                        output,
                        public,
                        protocol,
                        f"{policy}-{replicate:02d}",
                        [base_id, future_id],
                        accepted,
                        device,
                    )
                results.append(pair)
            # endregion book:worked-application-sequence
            record.write_json("fitting-observations.json", public)
            evaluations = {
                "registration": _assess(
                    wp, field_device, grid, protocol, means, target, registration, points, device
                ),
                "pairs": [],
            }
            for pair in results:
                evaluations["pairs"].append(
                    {
                        "replicate": pair["replicate"],
                        "selection": pair["selection"],
                        **{
                            policy: _assess(
                                wp,
                                field_device,
                                grid,
                                protocol,
                                means,
                                target,
                                pair[policy],
                                points,
                                device,
                            )
                            for policy in ("base", "selected", "near_parallel")
                        },
                    }
                )
            selected = np.array(
                [p["selected"]["errors"]["landmark_rms_mm"] ** 2 for p in evaluations["pairs"]]
            )
            fixed = np.array(
                [p["near_parallel"]["errors"]["landmark_rms_mm"] ** 2 for p in evaluations["pairs"]]
            )
            differences = selected - fixed
            ratio = float(selected.mean() / fixed.mean())
            upper = (
                None
                if development
                else float(
                    differences.mean()
                    + protocol["acceptance"]["paired_t_critical_7df"]
                    * differences.std(ddof=1)
                    / np.sqrt(len(differences))
                )
            )
            all_outcomes = [
                evaluations["registration"],
                *[
                    p[k]
                    for p in evaluations["pairs"]
                    for k in ("base", "selected", "near_parallel")
                ],
            ]
            completed = all(
                x["stationary"] and x["geometric_pass"] and x["held_out_pass"] for x in all_outcomes
            )
            benefit = (
                ratio
                <= protocol["acceptance"]["selected_vs_fixed_mean_squared_task_error_ratio_max"]
                and upper is not None
                and upper < protocol["acceptance"]["paired_error_difference_t95_upper_max"]
            )
            record.write_json("evaluation.json", evaluations)
            summary = {
                "development": development,
                "replicates": replicates,
                "all_solves_stationary_and_accurate": completed,
                "selected_mean_squared_task_error_mm2": float(selected.mean()),
                "near_parallel_mean_squared_task_error_mm2": float(fixed.mean()),
                "selected_to_near_parallel_ratio": ratio,
                "paired_mean_difference_mm2": float(differences.mean()),
                "paired_descriptive_t95_upper_mm2": upper,
                "benefit_pass": bool(benefit),
                "passed": bool(not development and completed and benefit),
                "uncertainty_scope": (
                    "Eight independent prescribed Poisson pairs on one teaching anatomy; "
                    "descriptive t interval, not a population or clinical claim."
                ),
            }
            record.write_json("summary.json", summary)
            children = {
                str(p.relative_to(output)): digest(p)
                for p in sorted(output.rglob("run.json"))
                if p != output / "run.json" and "sources" not in p.parts
            }
            record.write_json("child-records.json", children)
            print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--development",
        action="store_true",
        help="Separate prescribed development seed and one pair; never a final passing result.",
    )
    args = parser.parse_args()
    result = execute(args.known, args.output, args.config.resolve(), args.device, args.development)
    if not args.development and not result["passed"]:
        raise SystemExit("worked-example acceptance failed; retained all outcomes")


if __name__ == "__main__":
    main()
