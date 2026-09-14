"""Acquire, fit and evaluate one source-bound open-model primary material illustration."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from _illustration import (
    FAMILIES,
    counts_from_means,
    expected_noise_namespace,
    fitting_inputs,
    inverse_grid,
    loaded_source_identity,
    qualification_coverage,
    source_files,
    validate_config,
    verify_completed_fit,
    verify_physics,
    views,
)
from _resolution import check_coverage, checked_file, checked_json, count_metrics
from _study import coarse_cells, complete_record, digest, recorded_array
from acquire import check_family
from evaluate_fields import evaluate_material_fields
from probe_acquisition import detector, fixed_metric, helpers
from resolution_evaluate import independent_prediction_check

from dpt.experiments import RunRecorder, private_output, repository_root
from dpt.geometry import RigidTransform

np: Any = importlib.import_module("numpy")
wp: Any = importlib.import_module("warp")


def acquire(args: argparse.Namespace) -> None:
    root, library = repository_root(__file__), helpers()
    config = json.loads(args.config.read_text())
    validate_config(config)
    output = private_output(args.output, root)
    started = time.perf_counter()
    anatomy = json.loads((args.anatomy / "anatomy.json").read_text())
    metadata, physics, spec = library.load_physics(args.physics)
    verify_physics(physics, metadata, config)
    preparation = library.physics_preparation_sources(args.physics, metadata)
    for name in ("prepare_open_physics.py", "provision_inputs.py"):
        if metadata["source_sha256"][name] != digest(
            root / "experiments/spectral-reconstruction" / name
        ):
            raise ValueError("physical inputs were not prepared from this reviewed source tree")
    for name, expected, path in (
        (
            "prepare_vertebrae.py",
            anatomy["implementation"]["script_sha256"],
            args.anatomy / "prepare_vertebrae.py",
        ),
        (
            "prepare_anatomy.py",
            anatomy["implementation"]["helper_sha256"],
            root / "experiments/spectral-reconstruction/prepare_anatomy.py",
        ),
    ):
        if (
            digest(path) != expected
            or digest(root / "experiments/spectral-reconstruction" / name) != expected
        ):
            raise ValueError("anatomical inputs were not prepared from this reviewed source tree")
        preparation[f"preparation/{name}"] = path
    licence = args.anatomy / "source-license-manifest.json"
    if digest(licence) != anatomy["implementation"]["source_license_manifest_sha256"]:
        raise ValueError("anatomical source licence receipt changed")
    preparation["inputs/anatomy-source-license-manifest.json"] = licence
    if (
        {name: row["sha256"] for name, row in anatomy["source_files"].items()}
        != config["source_sha256"]
        or anatomy["native_crop_xyz"] != config["native_crop_xyz"]
        or digest(args.anatomy / "forward-fractions.npy")
        != config["expected_forward_fractions_sha256"]
    ):
        raise ValueError("primary anatomy differs from the exact openly prepared assignment")
    native = library.checked_array(args.anatomy, "forward-fractions.npy", anatomy["outputs"])
    native_grid = library.grid_from(anatomy["forward_grid"])
    if (
        list(native_grid.shape) != config["native_shape_zyx"]
        or list(native_grid.spacing_mm) != config["native_spacing_mm_xyz"]
    ):
        raise ValueError("native grid differs from its fixed primary protocol")
    fields, grid = coarse_cells(native, native_grid, (64, 64, 64))
    if grid != inverse_grid():
        raise ValueError("coarsening changed the prescribed primary physical grid")
    geometry = detector(config)
    trajectories = {role: views(config, role) for role in ("fitting", "withheld")}
    coverage = {
        role: check_coverage(native_grid, geometry, rows) for role, rows in trajectories.items()
    }
    namespace = expected_noise_namespace(config)
    coefficients80 = np.asarray(config["scalar_coefficients_80kev_mm_inverse"], dtype=np.float64)
    scalar_native = np.asarray(
        np.einsum("m,mzyx->zyx", coefficients80, native.astype(np.float64)), dtype=np.float32
    )
    scalar_inverse, _ = coarse_cells(scalar_native, native_grid, (64, 64, 64))
    metric = fixed_metric(physics)
    if not np.allclose(metric["matrix"], config["material_metric_matrix"], rtol=1e-13, atol=1e-13):
        raise ValueError("reproduced physical Fisher differs from the frozen metric")
    metric["matrix"] = config["material_metric_matrix"]
    sources = source_files(
        Path(__file__),
        {
            **preparation,
            "inputs/anatomy.json": args.anatomy / "anatomy.json",
            "inputs/forward-fractions.npy": args.anatomy / "forward-fractions.npy",
            "inputs/physics.npz": args.physics / "physics.npz",
            "inputs/physics-metadata.json": args.physics / "metadata.json",
        },
    )
    with RunRecorder(
        output,
        configuration={
            "scope": "primary acquisition; no inverse solve",
            "config": config,
            "loaded_source_identity": loaded_source_identity(root),
        },
        sources=sources,
    ) as acquisition_run:
        reports: dict[str, dict[str, Any]] = {}
        qualification_folder = output / "qualification"
        with RunRecorder(
            qualification_folder,
            configuration={
                "scope": "primary known-field qualification before all observation draws",
                "config": config,
                "physical_coverage": coverage,
                "noise_keys_fixed_before_projection": namespace,
            },
            sources=sources,
        ) as run:
            wp.init()
            for name, (_, sample_pair) in FAMILIES.items():
                role = "fitting" if name.startswith("fitting") else "withheld"
                fine = name.endswith("native")
                _, _, reports[name] = check_family(
                    run,
                    name=name,
                    fields=native if fine else fields,
                    scalar=scalar_native if fine else scalar_inverse,
                    grid=native_grid if fine else grid,
                    geometry=geometry,
                    poses=[
                        RigidTransform(**{k: tuple(v) for k, v in row["pose"].items()})
                        for row in trajectories[role]
                    ],
                    physics=physics,
                    spec=spec,
                    samples=sample_pair,
                    config=config,
                    device=args.device,
                )
            gate_count = qualification_coverage(reports)
            library.write_array(run, "evaluation/reference-fractions.npy", fields)
            run.write_json(
                "qualification.json",
                {
                    "passed": True,
                    "gate_count": gate_count,
                    "counts_drawn": False,
                    "fit_executed": False,
                },
            )
        # Qualification must complete and become immutable before even the first Poisson draw.
        qualification = complete_record(qualification_folder)
        bundle_sources = source_files(
            Path(__file__),
            {
                "qualification/run.json": qualification_folder / "run.json",
                "qualification/qualification.json": qualification_folder / "qualification.json",
                **{
                    f"qualification/{name}.json": qualification_folder
                    / "qualification"
                    / f"{name}.json"
                    for name in FAMILIES
                },
            },
        )
        bundles: dict[str, Any] = {}
        for role in ("fitting", "withheld"):
            mean_path = checked_file(
                qualification_folder,
                f"qualification/{role}-native/spectral-1024.npy",
                qualification["output_sha256"],
            )
            means = np.load(mean_path, allow_pickle=False)
            counts = counts_from_means(means, config, role)
            folder = output / role
            with RunRecorder(
                folder,
                configuration={
                    "role": role,
                    "config": config,
                    "qualification_run_sha256": digest(qualification_folder / "run.json"),
                    "gates_passed_before_counts": True,
                },
                sources=bundle_sources,
            ) as run:
                rows: list[dict[str, Any]] = []
                for index, view in enumerate(trajectories[role]):
                    name = f"observations/spectral-view{index:03d}.npy"
                    library.write_array(run, name, counts[index])
                    rows.append(
                        {
                            **view,
                            "spectral_counts": name,
                            "sha256": {"spectral": digest(folder / name)},
                        }
                    )
                library.write_array(
                    run, "open-beam.npy", np.full(geometry.shape, 200000.0, dtype=np.float32)
                )
                for name in ("physics.npz", "metadata.json"):
                    run.write_bytes(f"physics/{name}", (args.physics / name).read_bytes())
                run.write_json(
                    "public-observations.json",
                    {
                        "schema_version": 1,
                        "role": "primary fitting observations and admissible support only"
                        if role == "fitting"
                        else "primary withheld observations; evaluation only",
                        "group": "primary64",
                        "geometry": asdict(geometry),
                        "inverse_grid": asdict(grid),
                        "inverse_samples": 512,
                        "views": rows,
                        "open_beam": "open-beam.npy",
                        "open_beam_sha256": digest(folder / "open-beam.npy"),
                        "physics": {
                            name: digest(folder / "physics" / name)
                            for name in ("physics.npz", "metadata.json")
                        },
                        "metric": metric,
                        "config": config,
                        "reference_access_permitted_for_fitting": role != "fitting",
                        "spectral_total_incident_per_pixel_including_all_channels": len(rows)
                        * float(physics["incident_weights"].astype(np.float64).sum()),
                    },
                )
                run.write_json(
                    "noise.json",
                    {
                        "role": role,
                        "keys": namespace[role],
                        "root_is_distinct_from_prior_development": True,
                    },
                )
            bundles[role] = {
                "run_sha256": digest(folder / "run.json"),
                "manifest_sha256": digest(folder / "public-observations.json"),
            }
        acquisition_run.write_json(
            "acquisition.json",
            {
                "status": "complete",
                "bundles": bundles,
                "qualification_run_sha256": digest(qualification_folder / "run.json"),
                "gate_count": 4032,
                "all_sampling_gates_passed": True,
                "preparation_seconds": time.perf_counter() - started,
                "fitting_executed": False,
                "evaluation_executed": False,
            },
        )


def fit(args: argparse.Namespace) -> None:
    from pilot import run_method

    root = repository_root(__file__)
    public, inputs = fitting_inputs(args.observations, root)
    result = run_method(
        private_output(args.output, root),
        "spectral_metric",
        args.observations,
        public,
        inputs,
        Path("unused-scalar-case"),
        args.device,
        scope=public["config"]["run_scope"],
    )
    print(json.dumps(result), flush=True)


def evaluate(args: argparse.Namespace) -> None:
    root, library = repository_root(__file__), helpers()
    started = time.perf_counter()
    data = complete_record(args.acquisition)
    acquisition = checked_json(args.acquisition, "acquisition.json", data)
    fitting = args.acquisition / "fitting"
    public, inputs = fitting_inputs(fitting, root)
    record = verify_completed_fit(args.fit, fitting, public)
    for role in ("fitting", "withheld"):
        folder = args.acquisition / role
        for name, key in (
            ("run.json", "run_sha256"),
            ("public-observations.json", "manifest_sha256"),
        ):
            if digest(folder / name) != acquisition["bundles"][role][key]:
                raise ValueError("acquisition bundle identity changed")
    qualification_folder = args.acquisition / "qualification"
    if digest(qualification_folder / "run.json") != acquisition["qualification_run_sha256"]:
        raise ValueError("evaluation qualification identity changed")
    if record["source_sha256"]["qualification/run.json"] != acquisition["qualification_run_sha256"]:
        raise ValueError("fit and evaluation came from different qualified acquisitions")
    output = private_output(args.output, root)
    with RunRecorder(
        output,
        configuration={
            "scope": "post-freeze primary field and whole-view evaluation",
            "fit_run_sha256": digest(args.fit / "run.json"),
            "loaded_source_identity": loaded_source_identity(root),
        },
        sources=source_files(
            Path(__file__),
            {
                **inputs,
                "fit/run.json": args.fit / "run.json",
                "acquisition/run.json": args.acquisition / "run.json",
            },
        ),
    ) as run:
        # This durable record precedes all reference/withheld-array payload access.
        run.write_json(
            "compared-output-freeze.json",
            {
                "fit_run_sha256": digest(args.fit / "run.json"),
                "outputs": record["output_sha256"],
                "reference_arrays_opened": False,
                "withheld_arrays_opened": False,
            },
        )
        qualification = complete_record(qualification_folder)
        for name in qualification["output_sha256"]:
            checked_file(qualification_folder, name, qualification["output_sha256"])
        reference = recorded_array(
            qualification_folder, "evaluation/reference-fractions.npy", qualification
        )
        fields = recorded_array(args.fit, "fields.npy", record)
        predictions = recorded_array(args.fit, "predictions.npy", record)
        result = checked_json(args.fit, "solver-result.json", record)
        grid = library.grid_from(public["inverse_grid"])
        if fields.shape != (2, *grid.shape):
            raise ValueError("completed primary field shape changed")
        initial = np.zeros_like(fields)
        initial[0] = 1.0
        coefficients = cast(
            tuple[float, float], tuple(public["config"]["scalar_coefficients_80kev_mm_inverse"])
        )
        metrics = evaluate_material_fields(fields, reference, grid, mu80_mm_inverse=coefficients)
        initial_metrics = evaluate_material_fields(
            initial, reference, grid, mu80_mm_inverse=coefficients
        )
        _, physics, spec = library.load_physics(fitting / "physics")
        verify_physics(
            physics, json.loads((fitting / "physics/metadata.json").read_text()), public["config"]
        )
        groups: dict[str, Any] = {}
        wp.init()
        forward = library.Forward(
            grid, detector(public["config"]), fields, physics, spec, 512, args.device
        )
        for role in ("fitting", "withheld"):
            folder = args.acquisition / role
            group_record = complete_record(folder)
            p = checked_json(folder, "public-observations.json", group_record)
            expected_views = views(public["config"], role)
            if [
                {k: v[k] for k in ("id", "tilt_degrees", "yaw_degrees", "pose")} for v in p["views"]
            ] != expected_views:
                raise ValueError("evaluation view geometry differs from the prescribed role")
            counts = np.stack(
                [recorded_array(folder, v["spectral_counts"], group_record) for v in p["views"]]
            )
            actual = np.stack(
                [
                    forward.project(RigidTransform(**{k: tuple(x) for k, x in v["pose"].items()}))
                    for v in p["views"]
                ]
            )
            if role == "fitting" and not np.array_equal(actual, predictions):
                raise ValueError("final accepted fitting predictions do not reproduce exactly")
            library.write_array(run, f"{role}/predictions.npy", actual)
            mean = recorded_array(
                qualification_folder,
                f"qualification/{role}-native/spectral-1024.npy",
                qualification,
            )
            floor = recorded_array(
                qualification_folder, f"qualification/{role}64/spectral-512.npy", qualification
            )
            report = checked_json(
                qualification_folder, f"qualification/{role}64.json", qualification
            )
            oracle = independent_prediction_check(
                fields, actual, p, physics, report["independent_pixels_row_column"]
            )
            library.write_array(run, f"{role}/oracle.npy", oracle.pop("oracle"))
            library.write_array(run, f"{role}/oracle-actual.npy", oracle.pop("actual"))
            groups[role] = {
                "whole_views": len(p["views"]),
                "actual_field_prediction_check": oracle,
                "per_channel": [
                    {
                        "channel": c,
                        "recovered": count_metrics(actual[:, c], counts[:, c], mean[:, c]),
                        "inverse_reference_residual": count_metrics(
                            floor[:, c], counts[:, c], mean[:, c]
                        ),
                    }
                    for c in range(3)
                ],
            }
        previous = result["initial_objective"]
        history_passed = len(result["callback_history"]) == result["accepted_steps"]
        for index, row in enumerate(result["callback_history"], 1):
            history_passed = (
                history_passed
                and row["iteration"] == index
                and math.isfinite(row["objective"])
                and row["objective"] < row["objective_before"]
                and math.isclose(row["objective_before"], previous, rel_tol=1e-10, abs_tol=1e-7)
            )
            previous = row["objective"]
        history_passed = (
            history_passed
            and math.isfinite(result["final_objective"])
            and math.isclose(result["final_objective"], previous, rel_tol=1e-10, abs_tol=1e-7)
        )
        domain_passed = bool(
            np.isfinite(fields).all()
            and (fields >= 0).all()
            and (fields.sum(axis=0, dtype=np.float64) <= 1 + 2**-24).all()
        )
        run.write_json(
            "evaluation.json",
            {
                "status": "complete primary evaluation; scientific gates remain explicit",
                "field_metrics": metrics,
                "initial_field_metrics": initial_metrics,
                "projection_groups": groups,
                "termination": result["termination"],
                "accepted_steps": result["accepted_steps"],
                "stationarity": result["final_stationarity"],
                "accepted_history_consistent": history_passed,
                "field_domain_passed": domain_passed,
                "hard_correctness_passed": domain_passed
                and history_passed
                and all(g["actual_field_prediction_check"]["passed"] for g in groups.values()),
                "evaluation_seconds": time.perf_counter() - started,
                "limitations": [
                    "Assigned CT-derived composition and ideal primary-only spectrum/response; "
                    "no acquired material truth",
                    "The inverse-reference residual is not a proven attainable minimum",
                    "An illustration/development case, not the 40-fit final comparison",
                    "Budget termination is not convergence; "
                    "stationarity does not prove a global optimum",
                ],
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, required in (
        ("acquire", ("anatomy", "physics", "config", "output")),
        ("fit", ("observations", "output")),
        ("evaluate", ("acquisition", "fit", "output")),
    ):
        sub = commands.add_parser(command)
        for name in required:
            sub.add_argument("--" + name, type=Path, required=True)
        sub.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    {"acquire": acquire, "fit": fit, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
