"""Generate the approved final CT-derived observations, keeping reserved data separate."""

from __future__ import annotations

import argparse
import importlib
import itertools
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from _final import ROLE_KEYS, role_specs, view_id
from _study import (
    checked_array,
    digest,
    geometry,
    grid_spec,
    keyed_rng,
    read_json,
    record_sources,
    save_array,
)

from dpt.experiments import RunRecorder
from dpt.geometry import RigidTransform, compose_pose
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth
from dpt.validation.projection import integrate_sampled_field
from dpt.validation.recovery import detector_coordinate


def generate(prepared: Path, output: Path, protocol_path: Path, device: str) -> None:
    protocol, known = read_json(protocol_path), read_json(prepared / "known.json")
    case = known["case"]
    if case not in protocol["data"]["final_cases"]:
        raise ValueError("final generator requires a prespecified final known case")
    if known["protocol_sha256"] != digest(protocol_path):
        raise ValueError("known field was not prepared under this exact protocol")
    field = checked_array(prepared, known["attenuation"])
    grid = grid_spec(known["grid"])
    values = field.ravel().tolist()
    cfg = protocol["numerics"]
    if cfg["generation_samples_per_ray"] != 4096:
        raise ValueError("the accepted final generator uses4096samples")
    rng = keyed_rng(protocol["randomness"]["final_hidden_pose_root_seed"], case)
    hidden = protocol["hidden_pose_distribution"]
    pose = compose_pose(
        RigidTransform(),
        tuple(
            np.r_[
                rng.uniform(*hidden["translation_uniform_mm"], 3),
                rng.uniform(*hidden["rotation_vector_uniform_radians"], 3),
            ]
        ),
    )
    roles = role_specs(protocol)
    reserved_angles = protocol["evaluation"]["reserved_view_angles_degrees"]
    fitting_angles = {a for role in roles.values() for a in role["angles"]}
    if fitting_angles.intersection(reserved_angles):
        raise ValueError("a reserved angle appears among fitting/candidate views")
    angles = sorted(fitting_angles.union(reserved_angles))
    wp = importlib.import_module("warp")
    devfield = wp.array(field.ravel(), dtype=wp.float32, device=device)
    devpose = wp.array(pose.packed(), dtype=wp.float64, device=device)
    started = time.perf_counter()
    with RunRecorder(
        output,
        configuration={
            "role": "final generator; independent reference access only",
            "protocol": protocol,
            "known_sha256": digest(prepared / "known.json"),
        },
        sources=record_sources(
            __file__,
            protocol_path,
            known=prepared / "known.json",
            attenuation=prepared / known["attenuation"]["file"],
        ),
    ) as run:

        def save(name, array, *, units, role):
            return save_array(output, name, array, units=units, role=role, recorder=run)

        depths, checks, coverage = {}, {}, {}
        pixels = list(itertools.product((16, 48, 80, 112), (0, 32, 56, 64, 96, 127)))
        beam = cfg["sampling_gate_open_beam_per_pixel"]
        candidates = sorted(set([4096, *cfg["fitting_sampling_candidates"]]))
        for angle in angles:
            geom = geometry(protocol, angle)
            corners = [
                tuple(np.array(grid.origin_mm) + np.array(index) * np.array(grid.spacing_mm))
                for index in itertools.product(
                    (-0.5, grid.shape[2] - 0.5),
                    (-0.5, grid.shape[1] - 0.5),
                    (-0.5, grid.shape[0] - 0.5),
                )
            ]
            coverage[str(angle)] = [
                detector_coordinate(geom, pose.point(corner)) for corner in corners
            ]
            if any(
                not (-0.5 <= col <= geom.shape[1] - 0.5 and -0.5 <= row <= geom.shape[0] - 0.5)
                for col, row in coverage[str(angle)]
            ):
                raise ValueError("prespecified object support projects beyond detector")
            exact = np.array(
                [
                    integrate_sampled_field(values, grid, geom, pose, row, col, validate=False)
                    for row, col in pixels
                ]
            )
            reference_mean = beam * np.exp(-exact)
            sampled = {}
            for count in candidates:
                workspace = prepare_projection(grid, geom, ProjectionSpec(count), device=device)
                buf = wp.empty(geom.pixels, dtype=wp.float32, device=device)
                project_optical_depth(devfield, devpose, workspace=workspace, out_L=buf)
                sampled[count] = buf.numpy().reshape(geom.shape)
            depths[angle] = sampled[4096]
            full_reference = beam * np.exp(-sampled[4096].astype(np.float64))
            checks[str(angle)] = {}
            for count, depth in sampled.items():
                selected = np.array([depth[row, col] for row, col in pixels], dtype=np.float64)
                error = np.abs(beam * np.exp(-selected) - reference_mean) / np.sqrt(reference_mean)
                full = np.abs(beam * np.exp(-depth.astype(np.float64)) - full_reference) / np.sqrt(
                    full_reference
                )
                checks[str(angle)][str(count)] = {
                    "p95": float(np.quantile(error, 0.95)),
                    "maximum": float(error.max()),
                    "full_image_vs_4096_p95": float(np.quantile(full, 0.95)),
                    "full_image_vs_4096_maximum": float(full.max()),
                    "independent_depths": exact.tolist(),
                    "independent_pixels": pixels,
                }
            print(f"case{case} checked angle{angle:g}", flush=True)

        def passes(count):
            return all(
                max(rows[str(count)]["p95"], rows[str(count)]["full_image_vs_4096_p95"])
                <= cfg["sampling_error_poisson_sd_p95_max"]
                and max(rows[str(count)]["maximum"], rows[str(count)]["full_image_vs_4096_maximum"])
                <= cfg["sampling_error_poisson_sd_max"]
                for rows in checks.values()
            )

        fitting = next((n for n in cfg["fitting_sampling_candidates"] if passes(n)), None)
        sampling = {
            "status": "passed" if passes(4096) and fitting else "failed",
            "effective_fitting_samples": fitting,
            "generation_samples": 4096,
            "checks": checks,
            "projected_box_corners_column_row": coverage,
        }
        run.write_json("sampling.json", sampling)
        if sampling["status"] != "passed":
            raise ValueError("prespecified sampling gate failed; no final counts generated")
        full_mask = np.ones(tuple(protocol["geometry"]["detector_shape_hw"]), dtype=np.float32)
        strip = np.zeros_like(full_mask)
        centre = strip.shape[1] // 2
        strip[:, centre - 4 : centre + 4] = 1
        masks = {
            "full": save(
                "masks/full.npy",
                full_mask,
                units="dimensionless",
                role="fixed fitting/evaluation mask",
            ),
            "stress": save(
                "masks/stress.npy",
                strip,
                units="dimensionless",
                role="prespecified central eight columns",
            ),
        }
        common = {
            "status": "complete",
            "known_sha256": digest(prepared / "known.json"),
            "case": case,
            "effective_fitting_samples": fitting,
            "generation_samples": 4096,
            "source_description": known["source_description"],
            "rights": known["rights"],
            "protocol_sha256": digest(protocol_path),
        }
        public = {**common, "role": "final fitting observations only", "views": {}}
        reserved = {
            **common,
            "role": "reserved observations; evaluation after batch freeze only",
            "views": {},
        }
        means = {}
        for role, info in roles.items():
            for angle in sorted(set(info["angles"]).union(reserved_angles)):
                mean = info["beam"] * np.exp(-depths[angle].astype(np.float64))
                means[(role, angle)] = save(
                    f"generation-only/{role}-a{round(angle) + 180:03d}-mean.npy",
                    mean,
                    units="counts",
                    role="generator expectation; forbidden fitting/selection input",
                )
            for replicate in range(info["replicates"]):
                for is_reserved, selected_angles in (
                    (False, info["angles"]),
                    (True, reserved_angles),
                ):
                    for angle in selected_angles:
                        identifier = view_id(role, replicate, angle)
                        mean = info["beam"] * np.exp(-depths[angle].astype(np.float64))
                        counts = keyed_rng(
                            protocol["randomness"]["final_observation_root_seed"],
                            case,
                            ROLE_KEYS[role],
                            replicate,
                            round(angle) + 180,
                            int(is_reserved),
                        ).poisson(mean)
                        if counts.max() > 2**24:
                            raise ValueError("Poisson draw exceeds exactFP32integer")
                        counts = counts.astype(np.float32)
                        mask_name = "stress" if role == "stress" and not is_reserved else "full"
                        if mask_name == "stress":
                            save(
                                f"generation-only/{identifier}-unmasked.npy",
                                counts.copy(),
                                units="counts",
                                role="original simulated draws outside fixed fitting support",
                            )
                            counts[strip == 0] = 0
                        record = save(
                            f"{'reserved' if is_reserved else 'observations'}/{identifier}.npy",
                            counts,
                            units="counts",
                            role="reserved count observation"
                            if is_reserved
                            else "fitting count observation",
                        )
                        destination = reserved if is_reserved else public
                        destination["views"][identifier] = {
                            "angle_degrees": angle,
                            "geometry": asdict(geometry(protocol, angle)),
                            "open_beam_counts": info["beam"],
                            "observation": record,
                            "mask": masks[mask_name],
                            "role": role,
                            "replicate": replicate,
                        }
                        if is_reserved:
                            destination["views"][identifier]["generating_mean"] = means[
                                (role, angle)
                            ]
        run.write_json("public-observations.json", public)
        run.write_json("reserved-observations.json", reserved)
        run.write_json(
            "generator-reference.json",
            {
                "pose_object_to_world": asdict(pose),
                "geometric_probes_object_mm": [
                    list(p) for p in itertools.product((-60.0, -30.0, 0.0, 30.0, 60.0), repeat=3)
                ],
                "role": "final evaluation only; never fitting or design",
                "case": case,
            },
        )
        run.write_json(
            "generation-summary.json",
            {
                "wall_seconds": time.perf_counter() - started,
                "fitting_view_count": len(public["views"]),
                "reserved_view_count": len(reserved["views"]),
                "fitting_samples": fitting,
                "counts_reuse": (
                    "same case/role/replicate/view observations reused across "
                    "compared starts/policies"
                ),
            },
        )
    print(f"Final generated case{case}: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "output", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    generate(args.prepared, args.output, args.protocol, args.device)


if __name__ == "__main__":
    main()
