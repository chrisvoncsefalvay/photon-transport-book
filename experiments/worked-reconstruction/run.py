"""A fixed CT-derived material-reconstruction example with independent acceptance.

The three commands separate acquisition, fitting and post-freeze evaluation.
The matched coarse generating/inverse basis is intentional and declared; this
is an instructional correctness study, not patient-material validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from dpt.experiments import RunRecorder, experiment_sources, private_output, repository_root
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_reconstruction import (
    MaterialReconstruction,
    MaterialReconstructionSettings,
    MaterialReconstructionView,
)
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

STUDY = Path(__file__).resolve().parents[1] / "reconstruction-study"
sys.path.insert(0, str(STUDY))
from _resolution import check_coverage  # noqa: E402
from _study import (  # noqa: E402
    coarse_cells,
    complete_record,
    stationarity_gate,
    stationarity_result,
)
from probe_acquisition import detector, fixed_metric, helpers, pose_at  # noqa: E402

LIB = helpers()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid_from(record: dict[str, Any]) -> GridSpec:
    return GridSpec(**{key: tuple(value) for key, value in record.items()})


def checked_array(folder: Path, name: str, record: dict[str, Any]) -> np.ndarray:
    path = folder / name
    if digest(path) != record["output_sha256"][name]:
        raise ValueError(f"recorded input changed: {name}")
    return np.load(path, allow_pickle=False)


def checked_json(folder: Path, name: str, record: dict[str, Any]) -> dict[str, Any]:
    path = folder / name
    if digest(path) != record["output_sha256"][name]:
        raise ValueError(f"recorded metadata changed: {name}")
    return json.loads(path.read_text())


def require_physics_identity(folder: Path, public: dict[str, Any]) -> None:
    if (
        digest(folder / "physics.npz") != public["physics_sha256"]
        or digest(folder / "metadata.json") != public["physics_metadata_sha256"]
    ):
        raise ValueError("physical arrays or provenance changed after acquisition")


def sources(config: Path, extra: dict[str, Path]) -> dict[str, Path]:
    result = experiment_sources(__file__, config, extra=extra)
    for name in ("_study.py", "_resolution.py", "probe_acquisition.py"):
        result[f"helpers/{name}"] = STUDY / name
    result["helpers/spectral-run.py"] = Path(LIB.__file__)
    return result


def trajectories(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = {}
    for role, count in (
        ("fitting", config["fitting_views_per_ring"]),
        ("withheld", config["withheld_views_per_ring"]),
    ):
        offset = config["withheld_yaw_offset_degrees"] if role == "withheld" else 0.0
        rows[role] = [
            {
                "tilt_degrees": tilt,
                "yaw_degrees": offset + j * 360.0 / count,
                "pose": asdict(pose_at(tilt, offset + j * 360.0 / count)),
            }
            for tilt in config["tilts_degrees"]
            for j in range(count)
        ]
    return rows


def pose_from(row: dict[str, Any]) -> RigidTransform:
    return RigidTransform(**{key: tuple(value) for key, value in row["pose"].items()})


def poisson(
    means: np.ndarray, config: dict[str, Any], phase: int, replicate: int, role: int
) -> tuple[np.ndarray, list[list[int]]]:
    values = np.empty_like(means, dtype=np.float32)
    keys = []
    for view in range(means.shape[0]):
        for channel in range(means.shape[1]):
            key = [config["noise_root_seed"], phase, replicate, role, view, channel]
            sample = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key))).poisson(
                means[view, channel].astype(np.float64)
            )
            if np.any(sample > 2**24):
                raise ValueError("Poisson count exceeds exact FP32 integer representation")
            values[view, channel] = sample
            keys.append(key)
    return values, keys


def mean_error(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float64)
    error = (actual.astype(np.float64) - reference) / np.sqrt(reference)
    if not np.isfinite(error).all():
        raise ValueError("nonfinite Poisson-scaled mean error")
    return {
        "rms_poisson_sd": float(np.sqrt(np.mean(error**2))),
        "p95_poisson_sd": float(np.quantile(np.abs(error), 0.95)),
        "maximum_poisson_sd": float(np.max(np.abs(error))),
    }


def independent_means(
    fields: np.ndarray,
    grid: GridSpec,
    geometry: DetectorGeometry,
    pose: RigidTransform,
    physics: dict[str, Any],
    pixels: list[list[int]],
) -> np.ndarray:
    flat = fields.astype(np.float64).reshape(2, -1)
    paths = np.array(
        [
            [
                integrate_sampled_field(flat[m], grid, geometry, pose, row, col, validate=False)
                for row, col in pixels
            ]
            for m in range(2)
        ]
    )
    spectrum = physics["weights"].astype(np.float64) * physics["response"].astype(np.float64)
    return spectrum @ np.exp(-physics["mu_mm_inv"].astype(np.float64).T @ paths)


def require_sampling(error: dict[str, float], config: dict[str, Any]) -> None:
    if (
        error["p95_poisson_sd"] > config["quadrature_p95_poisson_sd"]
        or error["maximum_poisson_sd"] > config["quadrature_maximum_poisson_sd"]
    ):
        raise ValueError(f"prescribed quadrature failed: {error}")


def independent_mapping(fields: np.ndarray, gradient: np.ndarray, step: float) -> float:
    """CPU projection onto the two-material nonnegative unit simplex."""
    original = fields.astype(np.float64).reshape(2, -1)
    values = original - step * gradient.astype(np.float64).reshape(2, -1)
    projected = np.maximum(values, 0.0)
    cap = projected.sum(axis=0) > 1.0
    projected[0, cap] = np.clip(0.5 * (1.0 + values[0, cap] - values[1, cap]), 0.0, 1.0)
    projected[1, cap] = 1.0 - projected[0, cap]
    return float(np.max(np.abs((projected - original) / step)))


def acquisition_anatomy(
    folder: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, GridSpec]:
    """Admit either the original field or its hash-pinned portable coarse derivative."""
    anatomy = json.loads((folder / "anatomy.json").read_text())
    native = LIB.checked_array(folder, "forward-fractions.npy", anatomy["outputs"])
    if anatomy.get("format") == "worked-material-basis-v1":
        if (
            digest(folder / "anatomy.json") != config.get("coarse_anatomy_metadata_sha256")
            or digest(folder / "forward-fractions.npy") != config.get("coarse_fractions_sha256")
            or anatomy["original_forward_fractions_sha256"] != config["forward_fractions_sha256"]
        ):
            raise ValueError("portable anatomy differs from its admitted derivative identity")
        fields, grid = native, grid_from(anatomy["forward_grid"])
        if tuple(grid.shape) != tuple(config["shape_zyx"]):
            raise ValueError("portable material basis has the wrong shape")
    else:
        if digest(folder / "forward-fractions.npy") != config["forward_fractions_sha256"]:
            raise ValueError("native anatomy differs from the prospective input identity")
        fields, grid = coarse_cells(
            native, grid_from(anatomy["forward_grid"]), tuple(config["shape_zyx"])
        )
    if {name: row["sha256"] for name, row in anatomy["source_files"].items()} != config[
        "source_sha256"
    ] or anatomy["native_crop_xyz"] != config["native_crop_xyz"]:
        raise ValueError("anatomical source or crop differs from the prospective identity")
    return anatomy, fields, grid


def observation_pins(config: dict[str, Any]) -> dict[str, str]:
    """Admit the exact preserved observations required by the precision repair."""
    if config.get("generation_precision", "float32") != "float32":
        raise ValueError("observation generation must retain the original float32 precision")
    precision = config.get("calculation_precision", "float32")
    if precision not in ("float32", "float64"):
        raise ValueError("calculation precision must be float32 or float64")
    pins = config.get("final_observation_sha256", {})
    names = {
        f"{folder}/rep{replicate}-{role}-counts.npy"
        for replicate in (0, 1)
        for folder, role in (("fitting", "fitting"), ("evaluation", "withheld"))
    }
    if (precision == "float64" or "final_observation_sha256" in config) and (
        not isinstance(pins, dict)
        or set(pins) != names
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in pins.values()
        )
    ):
        raise ValueError("the precision repair must pin all four original final count arrays")
    return pins


def verify_observation_identity(
    folder: Path, config: dict[str, Any], development: bool
) -> dict[str, Any]:
    """Check generation output before either fit; no evaluation values enter fitting."""
    pins = observation_pins(config)
    actual = {name: digest(folder / name) for name in pins}
    if pins and not development and actual != pins:
        raise ValueError("generated counts differ from the preserved final observations")
    return {
        "generation_precision": "float32",
        "count_files_sha256": actual,
        "matches_preserved_final_observations": bool(pins) and not development,
        "development": development,
    }


def acquire(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    observation_pins(config)
    anatomy, fields, grid = acquisition_anatomy(args.anatomy, config)
    metadata, physics, spec = LIB.load_physics(args.physics)
    spec = replace(spec, precision="float32")
    if metadata.get("model_id") != config["physics_model_id"]:
        raise ValueError("physical inputs do not match the prospective protocol")
    if digest(args.physics / "physics.npz") != config["physics_npz_sha256"]:
        raise ValueError("physics differs from the prospective input identity")
    geometry, rows = detector(config), trajectories(config)
    coverage = {role: check_coverage(grid, geometry, records) for role, records in rows.items()}
    phase = config["development_noise_phase"] if args.development else config["final_noise_phase"]
    public = {
        "protocol": config,
        "protocol_sha256": digest(args.config),
        "grid": asdict(grid),
        "geometry": asdict(geometry),
        "trajectories": rows,
        "metric": fixed_metric(physics),
        "noise_phase": phase,
        "development": args.development,
        "physics_sha256": digest(args.physics / "physics.npz"),
        "physics_metadata_sha256": digest(args.physics / "metadata.json"),
        "coverage": coverage,
        "anatomy_source_files": anatomy["source_files"],
        "assignment": config["representation"],
    }
    extra = {
        "inputs/anatomy.json": args.anatomy / "anatomy.json",
        "inputs/forward-fractions.npy": args.anatomy / "forward-fractions.npy",
        "rights/anatomy.json": args.anatomy / "source-license-manifest.json",
        "inputs/physics.npz": args.physics / "physics.npz",
        "inputs/physics-metadata.json": args.physics / "metadata.json",
        **LIB.physics_preparation_sources(args.physics, metadata),
    }
    if (
        digest(args.anatomy / "source-license-manifest.json")
        != anatomy["implementation"]["source_license_manifest_sha256"]
    ):
        raise ValueError("anatomy rights record changed")
    with RunRecorder(
        private_output(args.output, repository_root(__file__)),
        configuration=public,
        sources=sources(args.config, extra),
    ) as run:
        LIB.write_array(run, "evaluation/reference-fractions.npy", fields)
        run.write_json("fitting.json", public)
        forwards = [
            LIB.Forward(grid, geometry, fields, physics, spec, samples, args.device)
            for samples in (config["fitting_samples"], config["generating_samples"])
        ]
        means_by_role = {}
        checks = []
        pixels = config["independent_pixels_rc"]
        for role, records in rows.items():
            means = []
            for index, row in enumerate(records):
                pose = pose_from(row)
                fitting, generating = [forward.project(pose) for forward in forwards]
                error = mean_error(fitting, generating)
                require_sampling(error, config)
                cpu = independent_means(fields, grid, geometry, pose, physics, pixels)
                selected = np.array(
                    [[generating[c, r, k] for r, k in pixels] for c in range(generating.shape[0])]
                )
                cpu_error = mean_error(selected, cpu)
                require_sampling(cpu_error, config)
                checks.append(
                    {"role": role, "view": index, "quadrature": error, "independent_cpu": cpu_error}
                )
                means.append(generating)
                print(f"qualified {role} view {index + 1}/{len(records)}", flush=True)
            means_by_role[role] = np.stack(means)
            LIB.write_array(run, f"evaluation/{role}-means.npy", means_by_role[role])
        run.write_json(
            "qualification.json", {"passed": True, "checks": checks, "before_any_count_draw": True}
        )
        for rep in config["replicates"]:
            for role_id, (role, means) in enumerate(means_by_role.items()):
                counts, keys = poisson(means, config, phase, rep, role_id)
                prefix = "fitting" if role == "fitting" else "evaluation"
                LIB.write_array(run, f"{prefix}/rep{rep}-{role}-counts.npy", counts)
                run.write_json(f"{prefix}/rep{rep}-{role}-noise.json", {"keys": keys})
        run.write_json(
            "observation-identity.json",
            verify_observation_identity(args.output, config, args.development),
        )


class SolveBudgetError(Exception):
    """An accepted state exhausted the prospectively declared soft solve budget."""


def make_solver(
    public: dict[str, Any],
    counts: np.ndarray,
    physics: dict[str, Any],
    spec: Any,
    settings: MaterialReconstructionSettings,
    device: str,
    initial: np.ndarray | None = None,
) -> MaterialReconstruction:
    grid, config = grid_from(public["grid"]), public["protocol"]
    if initial is None:
        initial = np.empty((2, *grid.shape), dtype=np.float32)
        initial[0].fill(config["initial_water"])
        initial[1].fill(config["initial_bone"])
    geometry = DetectorGeometry(**{key: tuple(value) for key, value in public["geometry"].items()})
    views = tuple(
        MaterialReconstructionView(
            geometry, pose_from(row), counts[i], physics["weights"], physics["response"]
        )
        for i, row in enumerate(public["trajectories"]["fitting"])
    )
    h = public["metric"]["matrix"]
    return MaterialReconstruction(
        grid=grid,
        views=views,
        coefficients=physics["mu_mm_inv"],
        spectral_spec=spec,
        initial_fractions=initial,
        settings=settings,
        samples_per_ray=config["fitting_samples"],
        device=device,
        material_metric=(h[0][0], h[0][1], h[1][1]),
    )


def derivative_check(solver: MaterialReconstruction, config: dict[str, Any]) -> dict[str, Any]:
    """Feasible central differences check the complete fitting objective/VJP."""
    original = solver.fractions_numpy().copy()
    interior = np.empty_like(original)
    interior[0].fill(0.65)
    interior[1].fill(0.2)
    solver.set_fractions(interior)
    solver.evaluate(gradient=True)
    gradient = solver.gradient.numpy().astype(np.float64)
    checks = []
    for seed in (7051, 7052, 7053):
        direction = (
            np.random.default_rng(seed).uniform(-0.1, 0.1, interior.shape).astype(np.float32)
        )
        analytic = float(gradient @ direction.ravel().astype(np.float64))
        differences = []
        for step in (0.02, 0.01, 0.005):
            high = np.ascontiguousarray(interior + step * direction, dtype=np.float32)
            low = np.ascontiguousarray(interior - step * direction, dtype=np.float32)
            solver.set_fractions(high)
            upper = solver.evaluate(gradient=False)
            solver.set_fractions(low)
            lower = solver.evaluate(gradient=False)
            differences.append({"step": step, "finite_difference": (upper - lower) / (2 * step)})
        error = min(abs(x["finite_difference"] - analytic) for x in differences)
        limit = max(
            config["directional_derivative_absolute_tolerance"],
            abs(analytic) * config["directional_derivative_relative_tolerance"],
        )
        checks.append(
            {
                "seed": seed,
                "analytic": analytic,
                "differences": differences,
                "minimum_error": error,
                "limit": limit,
                "passed": error <= limit,
            }
        )
    solver.set_fractions(original)
    if not all(row["passed"] for row in checks):
        raise ValueError(f"full-objective directional derivative check failed: {checks}")
    return {"passed": True, "checks": checks}


def independent_derivative_check(
    public: dict[str, Any], counts: np.ndarray, physics: dict[str, Any], spec: Any, device: str
) -> dict[str, Any]:
    """Compare the CUDA VJP with exact CPU path integrals on fixed actual rays.

    The artificial interior field and directions are derivative test inputs,
    never example anatomy or optimisation initialisation. Counts come only
    from the fitting bundle. No withheld/reference data enter this check.
    """
    config, grid = public["protocol"], grid_from(public["grid"])
    geometry = DetectorGeometry(**{key: tuple(value) for key, value in public["geometry"].items()})
    fields = np.empty((2, *grid.shape), dtype=np.float32)
    fields[0].fill(0.65)
    fields[1].fill(0.2)
    selection = [
        (view, row, col) for view in (0, 17, 34) for row, col in ((16, 16), (24, 24), (32, 32))
    ]
    views = []
    for view, row, col in selection:
        pixel_origin = tuple(
            geometry.origin_mm[i]
            + col * geometry.spacing_mm[0] * geometry.u[i]
            + row * geometry.spacing_mm[1] * geometry.v[i]
            for i in range(3)
        )
        pixel_geometry = DetectorGeometry(
            geometry.source_mm, pixel_origin, geometry.u, geometry.v, geometry.spacing_mm, (1, 1)
        )
        observed = np.ascontiguousarray(counts[view, :, row : row + 1, col : col + 1])
        views.append(
            MaterialReconstructionView(
                pixel_geometry,
                pose_from(public["trajectories"]["fitting"][view]),
                observed,
                physics["weights"],
                physics["response"],
            )
        )
    solver = MaterialReconstruction(
        grid=grid,
        views=tuple(views),
        coefficients=physics["mu_mm_inv"],
        spectral_spec=spec,
        initial_fractions=fields,
        settings=MaterialReconstructionSettings(iterations=1),
        samples_per_ray=config["fitting_samples"],
        device=device,
    )
    solver.evaluate(gradient=True)
    gradient = solver.gradient.numpy().astype(np.float64)
    coefficients = physics["mu_mm_inv"].astype(np.float64)
    spectral_weights = physics["weights"].astype(np.float64) * physics["response"].astype(
        np.float64
    )
    checks = []
    for seed in (8041, 8042, 8043):
        direction = np.random.default_rng(seed).uniform(-0.1, 0.1, fields.shape).astype(np.float32)
        cpu = 0.0
        for view, row, col in selection:
            pose = pose_from(public["trajectories"]["fitting"][view])
            paths = np.array(
                [
                    integrate_sampled_field(
                        fields[m].ravel(), grid, geometry, pose, row, col, validate=False
                    )
                    for m in (0, 1)
                ]
            )
            direction_paths = np.array(
                [
                    integrate_sampled_field(
                        direction[m].ravel(), grid, geometry, pose, row, col, validate=False
                    )
                    for m in (0, 1)
                ]
            )
            contributions = spectral_weights * np.exp(-(paths @ coefficients))
            mean = contributions.sum(axis=1)
            derivative = -(contributions @ (direction_paths @ coefficients))
            cpu += float(np.sum((1.0 - counts[view, :, row, col] / mean) * derivative))
        cuda = float(gradient @ direction.ravel().astype(np.float64))
        limit = max(
            config["directional_derivative_absolute_tolerance"],
            abs(cpu) * config["directional_derivative_relative_tolerance"],
        )
        checks.append(
            {
                "seed": seed,
                "cpu_exact_integral_derivative": cpu,
                "cuda_sampled_derivative": cuda,
                "absolute_error": abs(cuda - cpu),
                "limit": limit,
                "passed": abs(cuda - cpu) <= limit,
            }
        )
    if not all(row["passed"] for row in checks):
        raise ValueError(f"independent CPU derivative comparison failed: {checks}")
    return {
        "passed": True,
        "selected_fitting_rays": selection,
        "checks": checks,
        "scope": "exact CPU trilinear integration plus analytic spectral Poisson derivative",
    }


def prepare_metric(solver: MaterialReconstruction, run: RunRecorder, prefix: str) -> None:
    """Prepare an owned metric from this accepted field, outside a solve."""
    from dpt.material_preconditioning import prepare_fisher_scaling

    scaling = prepare_fisher_scaling(solver)
    solver.set_voxel_metric_scale(scaling.scale)
    run.write_json(f"{prefix}preconditioning.json", scaling.diagnostics)
    LIB.write_array(
        run,
        f"{prefix}voxel-metric-scale.npy",
        solver.voxel_metric_scale.numpy().reshape(solver.grid.shape),
    )


def solve_stages(
    solver: MaterialReconstruction,
    run: RunRecorder,
    config: dict[str, Any],
    iterations: int,
    seconds: float,
    refresh_steps: list[int],
) -> dict[str, Any]:
    """Keep one accepted trajectory/gate across prescribed metric refreshes.

    Each canonical solve starts fresh BB/inertial histories. Refreshes only
    occur after a completed update block; no field or tolerance is changed.
    All stages share one soft time budget and global checkpoint numbering.
    """
    history: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    started = time.perf_counter()
    boundaries = [step for step in refresh_steps if step < iterations] + [iterations]
    result: dict[str, Any] = {}
    for stage, boundary in enumerate(boundaries):
        offset = len(history)
        if stage:
            if time.perf_counter() - started >= seconds:
                result["termination"] = "wall_time_budget"
                break
            prepare_metric(solver, run, f"stages/after-{offset:04d}/")
            if time.perf_counter() - started >= seconds:
                result["termination"] = "wall_time_budget"
                break
        solver.settings = replace(solver.settings, iterations=boundary - offset)

        def callback(
            iteration: int, row: dict[str, Any], *, offset: int = offset, stage: int = stage
        ) -> None:
            global_iteration = offset + iteration
            record = {
                **row,
                "iteration": global_iteration,
                "stage": stage,
                "stage_iteration": iteration,
                "wall_seconds": time.perf_counter() - started,
            }
            history.append(record)
            run.write_json(f"history/accepted-{global_iteration:04d}.json", record)
            if global_iteration in config["checkpoint_steps"]:
                LIB.write_array(
                    run,
                    f"checkpoints/fields-{global_iteration:04d}.npy",
                    solver.fractions_numpy(),
                )
            if global_iteration % 25 == 0:
                print(
                    f"accepted {global_iteration}: objective={row['objective']:.8g}, "
                    f"mapping_before={row['gradient_mapping_before']:.8g}",
                    flush=True,
                )
            if time.perf_counter() - started >= seconds:
                raise SolveBudgetError

        try:
            result = solver.solve(callback)
        except SolveBudgetError:
            result = {
                "termination": "wall_time_budget",
                "accepted_steps": len(history) - offset,
                "history": history[offset:],
            }
        run.write_json(f"stages/stage-{stage:02d}.json", result)
        stages.append(
            {
                "stage": stage,
                "accepted_updates_before": offset,
                "accepted_steps": result["accepted_steps"],
                "termination": result["termination"],
                "metric_refreshed": stage > 0,
                "secant_and_inertial_histories": "fresh",
            }
        )
        if result["termination"] != "iteration_budget":
            break
    # Failed trials remain in the corresponding stage record. The global
    # trajectory contains accepted updates only, with no duplicate boundary.
    result.update(
        accepted_steps=len(history),
        history=history,
        stages=stages,
        solve_seconds=time.perf_counter() - started,
    )
    return result


def fit(args: argparse.Namespace) -> None:
    record = complete_record(args.acquisition)
    public = checked_json(args.acquisition, "fitting.json", record)
    config = public["protocol"]
    pins = observation_pins(config)
    if (
        digest(args.config) != public["protocol_sha256"]
        or args.replicate not in config["replicates"]
    ):
        raise ValueError("protocol or replicate differs from frozen acquisition")
    require_physics_identity(args.physics, public)
    qualification = checked_json(args.acquisition, "qualification.json", record)
    if not qualification["passed"]:
        raise ValueError("acquisition qualification did not pass")
    name = f"fitting/rep{args.replicate}-fitting-counts.npy"
    counts = checked_array(args.acquisition, name, record)
    if pins and not public["development"]:
        identity = checked_json(args.acquisition, "observation-identity.json", record)
        if (
            identity["count_files_sha256"] != pins
            or identity["matches_preserved_final_observations"] is not True
            or identity["development"] is not False
            or identity["generation_precision"] != "float32"
            or record["output_sha256"][name] != pins[name]
        ):
            raise ValueError("fitting requires the exact preserved final observations")
    _, physics, spec = LIB.load_physics(args.physics)
    spec = replace(spec, precision=config.get("calculation_precision", "float32"))
    development = public["development"]
    iterations = (
        config["development_accepted_updates"]
        if development
        else config["maximum_accepted_updates"]
    )
    seconds = (
        config["development_soft_solve_seconds"] if development else config["soft_solve_seconds"]
    )
    if args.development_seconds is not None:
        if not development or not 0 < args.development_seconds <= config["soft_solve_seconds"]:
            raise ValueError("a development time override must stay within the final budget")
        seconds = args.development_seconds
    if args.development_iterations is not None:
        if (
            not development
            or not 0 < args.development_iterations <= config["maximum_accepted_updates"]
        ):
            raise ValueError("a development update override must stay within the final budget")
        iterations = args.development_iterations
    step_selection = config.get("step_selection", "geometric")
    acceleration = config.get("acceleration", "none")
    preconditioner = config.get("preconditioner", "none")
    if args.development_step_selection is not None:
        if not development:
            raise ValueError("a step-selection override is allowed only for development")
        step_selection = args.development_step_selection
    if args.development_acceleration is not None:
        if not development:
            raise ValueError("an acceleration override is allowed only for development")
        acceleration = args.development_acceleration
    if args.development_preconditioner is not None:
        if not development:
            raise ValueError("a preconditioner override is allowed only for development")
        preconditioner = args.development_preconditioner
    if preconditioner not in ("none", "fisher"):
        raise ValueError("preconditioner must be none or fisher")
    refresh_steps = config.get("preconditioner_refresh_after", [])
    if (
        not isinstance(refresh_steps, list)
        or any(type(step) is not int or step <= 0 for step in refresh_steps)
        or refresh_steps != sorted(set(refresh_steps))
        or (refresh_steps and preconditioner != "fisher")
    ):
        raise ValueError(
            "metric refreshes require ordered unique positive steps and Fisher scaling"
        )
    settings = MaterialReconstructionSettings(
        iterations=iterations,
        initial_step=config["initial_step"],
        mapping_step=config["mapping_step"],
        regularisation_mm_inverse=config["regularisation_mm_inverse"],
        step_selection=step_selection,
        acceleration=acceleration,
    )
    extra = {
        "inputs/acquisition-run.json": args.acquisition / "run.json",
        "inputs/fitting.json": args.acquisition / "fitting.json",
        "inputs/counts.npy": args.acquisition / name,
        "inputs/qualification.json": args.acquisition / "qualification.json",
        "inputs/physics.npz": args.physics / "physics.npz",
        "inputs/physics-metadata.json": args.physics / "metadata.json",
    }
    if pins:
        extra["inputs/observation-identity.json"] = args.acquisition / "observation-identity.json"
    resumed = None
    if args.resume_development_fit is not None:
        if not development:
            raise ValueError("checkpoint continuation is allowed only for development")
        previous = complete_record(args.resume_development_fit)
        if (
            not previous["configuration"]["development"]
            or previous["configuration"]["replicate"] != args.replicate
            or previous["source_sha256"]["inputs/acquisition-run.json"]
            != digest(args.acquisition / "run.json")
        ):
            raise ValueError("continuation must use the same development acquisition and replicate")
        resumed = {
            "fields": checked_array(args.resume_development_fit, "fields.npy", previous),
            "gate": checked_json(args.resume_development_fit, "stationarity-gate.json", previous),
            "result": checked_json(args.resume_development_fit, "solver-result.json", previous),
        }
        for name in ("run.json", "fields.npy", "stationarity-gate.json", "solver-result.json"):
            extra[f"continuation/{name}"] = args.resume_development_fit / name
    with RunRecorder(
        private_output(args.output, repository_root(__file__)),
        configuration={
            "protocol": config,
            "development": development,
            "replicate": args.replicate,
            "maximum_updates": iterations,
            "soft_seconds": seconds,
            "step_selection": step_selection,
            "acceleration": acceleration,
            "preconditioner": preconditioner,
            "preconditioner_refresh_after": refresh_steps,
            "fitting_reference_access": False,
            "continuation": str(args.resume_development_fit) if resumed is not None else None,
            "secant_history_restart": resumed is not None,
        },
        sources=sources(args.config, extra),
    ) as run:
        run.write_json(
            "independent-derivatives.json",
            independent_derivative_check(public, counts, physics, spec, args.device),
        )
        solver = make_solver(public, counts, physics, spec, settings, args.device)
        run.set_metadata(device=str(solver.device), device_name=solver.device.name)
        run.write_json("derivatives.json", derivative_check(solver, config))
        initial_objective = solver.evaluate(gradient=True)
        initial_mapping, _ = solver.projected_trial(config["mapping_step"], use_metric=False)
        gate = stationarity_gate(
            initial_mapping,
            config["mapping_step"],
            config["relative_mapping_tolerance"],
            config["maximum_mapped_fraction_displacement"],
        )
        continuation = None
        if resumed is not None:
            # Round-off in a repeated GPU reduction cannot alter the admitted
            # original-start threshold. Keep the actual recorded gate exactly.
            if not np.isclose(
                gate["initial_mapping"], resumed["gate"]["initial_mapping"], rtol=1e-6
            ):
                raise ValueError("continued solver does not reproduce the original start mapping")
            gate = resumed["gate"]
            solver.set_fractions(resumed["fields"])
            continuation = {
                "previous_accepted_steps": resumed["result"]["accepted_steps"],
                "start_objective": solver.evaluate(gradient=True),
                "secant_history": "fresh restart at the recorded accepted field",
                "original_gate_retained": True,
            }
        solver.settings = replace(
            settings,
            gradient_mapping_tolerance=gate["absolute_threshold"],
            relative_gradient_mapping_tolerance=0.0,
        )
        run.write_json("stationarity-gate.json", gate)
        if preconditioner == "fisher":
            prepare_metric(solver, run, "")
        LIB.write_array(run, "checkpoints/fields-0000.npy", solver.fractions_numpy())
        started = time.perf_counter()
        result = solve_stages(solver, run, config, iterations, seconds, refresh_steps)
        final_objective = solver.evaluate(gradient=True)
        mapping, _ = solver.projected_trial(config["mapping_step"], use_metric=False)
        result.update(
            initial_objective=initial_objective,
            initial_gradient_mapping=initial_mapping,
            final_objective=final_objective,
            stationarity=stationarity_result(mapping, gate),
            solve_seconds=time.perf_counter() - started,
            reference_or_withheld_data_access=False,
            continuation=continuation,
        )
        LIB.write_array(run, "fields.npy", solver.fractions_numpy())
        LIB.write_array(run, "gradient.npy", solver.gradient.numpy())
        LIB.write_array(run, "fitting-predictions.npy", np.stack(solver.predictions_numpy()))
        run.write_json("solver-result.json", result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("termination", "accepted_steps", "stationarity", "solve_seconds")
                }
            ),
            flush=True,
        )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    acquisition = complete_record(args.acquisition)
    fitting = complete_record(args.fit)
    public = checked_json(args.acquisition, "fitting.json", acquisition)
    config = public["protocol"]
    if digest(args.config) != public["protocol_sha256"]:
        raise ValueError("evaluation protocol changed")
    # Freeze and verify actual accepted output before opening the assigned reference.
    fields = checked_array(args.fit, "fields.npy", fitting)
    result = checked_json(args.fit, "solver-result.json", fitting)
    if fitting["source_sha256"]["inputs/acquisition-run.json"] != digest(
        args.acquisition / "run.json"
    ):
        raise ValueError("fit is not bound to this acquisition")
    frozen = {
        name: digest(args.fit / name)
        for name in ("run.json", "fields.npy", "gradient.npy", "solver-result.json")
    }
    gradient = checked_array(args.fit, "gradient.npy", fitting)
    cpu_mapping = independent_mapping(fields, gradient, config["mapping_step"])
    if not np.isclose(cpu_mapping, result["stationarity"]["final_mapping"], rtol=1e-10, atol=1e-7):
        raise ValueError("independent CPU projected mapping disagrees with the final CUDA state")
    reference = checked_array(args.acquisition, "evaluation/reference-fractions.npy", acquisition)
    grid = grid_from(public["grid"])
    geometry = DetectorGeometry(**{key: tuple(value) for key, value in public["geometry"].items()})
    require_physics_identity(args.physics, public)
    _, physics, spec = LIB.load_physics(args.physics)
    extra = {
        "inputs/acquisition-run.json": args.acquisition / "run.json",
        "inputs/fit-run.json": args.fit / "run.json",
        "inputs/fields.npy": args.fit / "fields.npy",
        "inputs/gradient.npy": args.fit / "gradient.npy",
        "inputs/solver-result.json": args.fit / "solver-result.json",
        "inputs/reference.npy": args.acquisition / "evaluation/reference-fractions.npy",
        "inputs/physics.npz": args.physics / "physics.npz",
        "inputs/physics-metadata.json": args.physics / "metadata.json",
    }
    with RunRecorder(
        private_output(args.output, repository_root(__file__)),
        configuration={"protocol": config, "accepted_output_freeze": frozen},
        sources=sources(args.config, extra),
    ) as run:
        forward = LIB.Forward(
            grid, geometry, fields, physics, spec, config["generating_samples"], args.device
        )
        errors = fields.astype(np.float64) - reference.astype(np.float64)
        rmse = np.sqrt(np.mean(errors**2, axis=(1, 2, 3)))
        metrics = {
            "material_rmse_water_bone": rmse.tolist(),
            "stationarity": result["stationarity"],
            "termination": result["termination"],
            "accepted_updates": result["accepted_steps"],
            "replicate": fitting["configuration"]["replicate"],
            "development": public["development"],
            "independent_cpu_mapping": cpu_mapping,
        }
        for role, rows in public["trajectories"].items():
            predictions = np.stack([forward.project(pose_from(row)) for row in rows])
            means = checked_array(args.acquisition, f"evaluation/{role}-means.npy", acquisition)
            metrics[role] = mean_error(predictions, means)
            LIB.write_array(run, f"{role}-predictions.npy", predictions)
            checks = []
            for index, row in enumerate(rows):
                pixels = config["independent_pixels_rc"]
                cpu = independent_means(fields, grid, geometry, pose_from(row), physics, pixels)
                selected = np.array(
                    [
                        [predictions[index, c, r, k] for r, k in pixels]
                        for c in range(predictions.shape[1])
                    ]
                )
                error = mean_error(selected, cpu)
                require_sampling(error, config)
                checks.append({"view": index, **error})
            run.write_json(
                f"{role}-independent-final-field.json", {"passed": True, "checks": checks}
            )
        domain_passed = bool(
            np.isfinite(fields).all()
            and np.all(fields >= 0)
            and np.all(fields.astype(np.float64).sum(axis=0) <= 1 + 2**-24)
        )
        metrics["domain_passed"] = domain_passed
        metrics["numerical_passed"] = bool(
            domain_passed
            and result["stationarity"]["passed"]
            and np.all(rmse <= config["maximum_material_rmse"])
            and metrics["withheld"]["rms_poisson_sd"]
            <= config["maximum_withheld_generating_mean_error_rms_poisson_sd"]
        )
        metrics["accepted"] = metrics["numerical_passed"] and not public["development"]
        metrics["scope"] = (
            "Matched coarse CT-derived assigned composition; ideal primary-only physics, "
            "not patient material validation"
        )
        run.write_json("evaluation.json", metrics)
        LIB.write_array(run, "signed-material-error.npy", errors)
        print(json.dumps(metrics), flush=True)
    return metrics


# region book:worked-reconstruction-workflow
def run_worked_example(args: argparse.Namespace) -> dict[str, Any]:
    """Generate once, freeze both fits, then assess every prescribed replicate."""
    config = json.loads(args.config.read_text())
    if config["replicates"] != [0, 1]:
        raise ValueError("the complete worked example requires both prescribed replicates [0, 1]")
    with RunRecorder(
        private_output(args.output, repository_root(__file__)),
        configuration={"protocol": config, "development": args.development},
        sources=sources(args.config, {}),
    ) as run:
        acquisition = args.output / "acquisition"
        acquisition_args = argparse.Namespace(**vars(args))
        acquisition_args.output = acquisition
        acquire(acquisition_args)
        fits = {}
        for replicate in config["replicates"]:
            fit_args = argparse.Namespace(**vars(args))
            fit_args.acquisition = acquisition
            fit_args.replicate = replicate
            fit_args.output = args.output / f"fit-rep{replicate}"
            fit(fit_args)
            fits[replicate] = fit_args.output
        # No reference or withheld assessment is opened until both fits exist.
        outcomes = []
        for replicate in config["replicates"]:
            evaluation_args = argparse.Namespace(**vars(args))
            evaluation_args.acquisition = acquisition
            evaluation_args.fit = fits[replicate]
            evaluation_args.output = args.output / f"evaluation-rep{replicate}"
            outcomes.append(evaluate(evaluation_args))
        child_records = {
            str(path.relative_to(args.output)): digest(path)
            for path in sorted(args.output.glob("*/run.json"))
        }
        summary = {
            "development": args.development,
            "replicates": outcomes,
            "numerical_passed": all(row["numerical_passed"] for row in outcomes),
            "accepted": all(row["accepted"] for row in outcomes),
            "all_fits_frozen_before_reference_evaluation": True,
            "child_records_sha256": child_records,
        }
        run.write_json("summary.json", summary)
    return summary


# endregion book:worked-reconstruction-workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("all", "acquire", "fit", "evaluate"), nargs="?", default="all"
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("protocol-v3.json"))
    bundle = (
        Path(__file__).resolve().parents[2]
        / "public/generated/worked-examples/inputs/reconstruction"
    )
    parser.add_argument("--anatomy", type=Path, default=bundle / "anatomy")
    parser.add_argument("--physics", type=Path, default=bundle / "physics")
    parser.add_argument("--acquisition", type=Path)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--development-seconds", type=float)
    parser.add_argument("--development-iterations", type=int)
    parser.add_argument("--development-step-selection", choices=("geometric", "bb"))
    parser.add_argument("--development-acceleration", choices=("none", "inertial"))
    parser.add_argument("--development-preconditioner", choices=("none", "fisher"))
    parser.add_argument("--resume-development-fit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    required = {
        "all": (),
        "acquire": ("anatomy",),
        "fit": ("acquisition",),
        "evaluate": ("acquisition", "fit"),
    }
    for name in required[args.command]:
        if getattr(args, name) is None:
            parser.error(f"{args.command} requires --{name}")
    result = {"all": run_worked_example, "acquire": acquire, "fit": fit, "evaluate": evaluate}[
        args.command
    ](args)
    if args.command in ("all", "evaluate") and not result["accepted"] and not result["development"]:
        parser.exit(
            1,
            "The retained final outcomes did not pass scientific acceptance; "
            "inspect the recorded report.\n",
        )


if __name__ == "__main__":
    main()
