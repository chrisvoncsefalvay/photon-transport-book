"""Generate independently sampled development observations; no fitting occurs here."""

from __future__ import annotations

import argparse
import importlib
import itertools
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
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


def development(prepared: Path, output: Path, protocol_path: Path, device: str) -> None:
    protocol = read_json(protocol_path)
    known = read_json(prepared / "known.json")
    if known["case"] != protocol["data"]["development_case"]:
        raise ValueError("this development generator cannot open a final case")
    field = checked_array(prepared, known["attenuation"])
    grid = grid_spec(known["grid"])
    reference_values = field.ravel().tolist()
    rng = keyed_rng(protocol["randomness"]["development_root_seed"], 0)
    hidden = np.r_[rng.uniform(-4, 4, 3), rng.uniform(-0.035, 0.035, 3)]
    pose = compose_pose(RigidTransform(), tuple(hidden))
    angles = sorted(
        set([0.0, 5.0, 90.0, 45.0, 135.0, *protocol["acquisition"]["candidate_angles_degrees"]])
    )
    wp = importlib.import_module("warp")
    field_device = wp.array(field.ravel(), dtype=wp.float32, device=device)
    pose_device = wp.array(pose.packed(), dtype=wp.float64, device=device)
    start = time.perf_counter()
    with RunRecorder(
        output,
        configuration={
            "role": "development only",
            "protocol": protocol,
            "known_manifest_sha256": digest(prepared / "known.json"),
        },
        sources=record_sources(
            __file__,
            protocol_path,
            known_manifest=prepared / "known.json",
            attenuation=prepared / known["attenuation"]["file"],
        ),
    ) as run:

        def save_generated(name, array, *, units, role):
            return save_array(output, name, array, units=units, role=role, recorder=run)

        depths = {}
        coverage = {}
        for angle in angles:
            geom = geometry(protocol, angle)
            box_corners = [
                tuple(np.array(grid.origin_mm) + np.array(index) * np.array(grid.spacing_mm))
                for index in itertools.product(
                    (-0.5, grid.shape[2] - 0.5),
                    (-0.5, grid.shape[1] - 0.5),
                    (-0.5, grid.shape[0] - 0.5),
                )
            ]
            projected = [detector_coordinate(geom, pose.point(corner)) for corner in box_corners]
            if any(
                not (-0.5 <= column <= geom.shape[1] - 0.5 and -0.5 <= row <= geom.shape[0] - 0.5)
                for column, row in projected
            ):
                raise ValueError("the declared generating pose does not fit the detector")
            coverage[str(angle)] = projected
            # Generation owns a separate high-sample projector. It never invokes a pose fit.
            workspace = prepare_projection(grid, geom, ProjectionSpec(4096), device=device)
            buffer = wp.empty(geom.pixels, dtype=wp.float32, device=device)
            project_optical_depth(field_device, pose_device, workspace=workspace, out_L=buffer)
            depths[angle] = buffer.numpy().reshape(geom.shape)
        # Fixed independent rays on the base view; no pixel is selected by its residual.
        geom = geometry(protocol, 0.0)
        pixels = list(itertools.product((16, 48, 80, 112), (0, 32, 56, 64, 96, 127)))
        exact = np.array(
            [
                integrate_sampled_field(
                    reference_values, grid, geom, pose, row, col, validate=False
                )
                for row, col in pixels
            ]
        )
        means_exact = 20000 * np.exp(-exact)
        checks = {}
        for count in sorted(
            set(
                protocol["numerics"]["fitting_sampling_candidates"]
                + protocol["numerics"]["generation_sampling_candidates"]
            )
        ):
            workspace = prepare_projection(grid, geom, ProjectionSpec(count), device=device)
            buffer = wp.empty(geom.pixels, dtype=wp.float32, device=device)
            project_optical_depth(field_device, pose_device, workspace=workspace, out_L=buffer)
            actual = buffer.numpy().reshape(geom.shape)
            chosen = np.array([actual[row, col] for row, col in pixels])
            error = np.abs(20000 * np.exp(-chosen.astype(np.float64)) - means_exact) / np.sqrt(
                means_exact
            )
            full_error = np.abs(
                20000 * np.exp(-actual.astype(np.float64))
                - 20000 * np.exp(-depths[0.0].astype(np.float64))
            ) / np.sqrt(20000 * np.exp(-depths[0.0]))
            checks[str(count)] = {
                "independent_rays": pixels,
                "exact_depths": exact.tolist(),
                "error_in_poisson_sd": error.tolist(),
                "p95": float(np.quantile(error, 0.95)),
                "maximum": float(error.max()),
                "full_image_vs_4096_p95": float(np.quantile(full_error, 0.95)),
                "full_image_vs_4096_maximum": float(full_error.max()),
            }
        cfg = protocol["numerics"]

        def passes(n):
            c = checks[str(n)]
            return (
                max(c["p95"], c["full_image_vs_4096_p95"])
                <= cfg["sampling_error_poisson_sd_p95_max"]
                and max(c["maximum"], c["full_image_vs_4096_maximum"])
                <= cfg["sampling_error_poisson_sd_max"]
            )

        fitting = next((n for n in cfg["fitting_sampling_candidates"] if passes(n)), None)
        if not passes(4096) or fitting is None:
            run.write_json("sampling.json", {"status": "failed", "checks": checks})
            raise ValueError("development sampling gate failed; do not generate final observations")
        # All views use 4096 already checked against the independent continuous-field oracle.
        public = {
            "status": "complete",
            "role": "development only",
            "known_sha256": digest(prepared / "known.json"),
            "case": known["case"],
            "effective_fitting_samples": fitting,
            "generation_samples": 4096,
            "views": {},
            "source_description": known["source_description"],
            "rights": known["rights"],
        }
        for angle in angles:
            code = round(angle) + 180
            for study, beam in (("reg", 10000.0), ("design", 1000.0), ("stress", 200.0)):
                if study == "stress" and angle != 0.0:
                    continue
                name = f"{study}-a{code:03d}"
                mean = beam * np.exp(-depths[angle].astype(np.float64))
                role = {"reg": 1, "design": 2, "stress": 3}[study]
                counts = keyed_rng(
                    protocol["randomness"]["development_root_seed"], 1, role, code
                ).poisson(mean)
                if counts.max() > 2**24:
                    raise ValueError("counts exceed exact FP32 integers")
                counts = counts.astype(np.float32)
                mask = np.ones(counts.shape, dtype=np.float32)
                if study == "stress":
                    mask[:, :] = 0
                    centre = mask.shape[1] // 2
                    mask[:, centre - 4 : centre + 4] = 1
                raw = counts.copy()
                counts[mask == 0] = 0
                obs = save_generated(
                    f"observations/{name}.npy",
                    counts,
                    units="counts",
                    role="fitting observation",
                )
                mask_record = save_generated(
                    f"masks/{name}.npy",
                    mask,
                    units="dimensionless",
                    role="fixed binary mask",
                )
                public["views"][name] = {
                    "angle_degrees": angle,
                    "geometry": asdict(geometry(protocol, angle)),
                    "open_beam_counts": beam,
                    "observation": obs,
                    "mask": mask_record,
                }
                save_generated(
                    f"generation-only/{name}-expected.npy",
                    mean,
                    units="counts",
                    role="generator expectation; not fitting",
                )
                if study == "stress":
                    save_generated(
                        f"generation-only/{name}-raw.npy",
                        raw,
                        units="counts",
                        role="original unmasked count array",
                    )
        run.write_json("public-observations.json", public)
        run.write_json(
            "generator-reference.json",
            {
                "pose_object_to_world": asdict(pose),
                "geometric_probes_object_mm": [
                    list(p) for p in itertools.product((-60.0, -30.0, 0.0, 30.0, 60.0), repeat=3)
                ],
                "role": "development generator/evaluation only; never supplied to fit/design",
            },
        )
        run.write_json(
            "sampling.json",
            {
                "status": "passed",
                "effective_fitting_samples": fitting,
                "generation_samples": 4096,
                "checks": checks,
                "projected_box_corners_column_row": coverage,
                "generation_wall_seconds": time.perf_counter() - start,
            },
        )
    print(f"Development observations: {output}; fit samples={fitting}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("protocol-v1.json")
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    development(args.prepared, args.output, args.protocol, args.device)


if __name__ == "__main__":
    main()
