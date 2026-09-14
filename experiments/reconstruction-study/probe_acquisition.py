"""Measure three native anatomical projections and their quadrature error.

No counts are drawn and no inverse problem is solved. A complete probe is not
permission to generate the full acquisition or change the fixed configuration.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from dpt.experiments import RunRecorder, experiment_sources, private_output, repository_root
from dpt.geometry import DetectorGeometry, Matrix3, RigidTransform
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth
from dpt.transmission import TransmissionSpec, prepare_transmission, transmit
from dpt.validation.projection import integrate_sampled_field

wp: Any = importlib.import_module("warp")


def helpers() -> Any:
    path = Path(__file__).resolve().parent.parent / "spectral-reconstruction/run.py"
    spec = importlib.util.spec_from_file_location("reconstruction_spectral_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("the portable spectral helper source is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pose_at(tilt: float, yaw: float) -> RigidTransform:
    a, b = math.radians(tilt), math.radians(yaw)
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    rz = np.array([[math.cos(b), -math.sin(b), 0], [math.sin(b), math.cos(b), 0], [0, 0, 1]])
    return RigidTransform(rotation=cast(Matrix3, tuple(float(v) for v in (rz @ rx).ravel())))


def detector(config: dict[str, Any]) -> DetectorGeometry:
    height, width = config["detector_shape_hw"]
    pitch = config["detector_pitch_mm"]
    sid, sdd = config["source_isocentre_mm"], config["source_detector_mm"]
    return DetectorGeometry(
        source_mm=(0.0, -sid, 0.0),
        origin_mm=(-(width - 1) * pitch / 2, sdd - sid, -(height - 1) * pitch / 2),
        u=(1, 0, 0),
        v=(0, 0, 1),
        spacing_mm=(pitch, pitch),
        shape=(height, width),
    )


def coverage(grid: Any, config: dict[str, Any]) -> dict[str, Any]:
    corners = [
        grid.grid_to_object(tuple(p)) for p in itertools.product(*zip(*grid.support, strict=True))
    ]
    sid, sdd = config["source_isocentre_mm"], config["source_detector_mm"]
    half = np.array(config["detector_shape_hw"][::-1]) * config["detector_pitch_mm"] / 2
    maximum = np.zeros(2)
    for tilt in config["tilts_degrees"]:
        for index in range(config["dense_views_per_ring"]):
            transform = pose_at(tilt, index * 360 / config["dense_views_per_ring"])
            for corner in corners:
                x, y, z = transform.point(corner)
                if y + sid <= 0:
                    raise ValueError("support crosses the source plane")
                maximum = np.maximum(maximum, np.abs(np.array([x, z]) * sdd / (y + sid)))
    if np.any(maximum >= half):
        raise ValueError("fixed detector truncates a prescribed dense view")
    return {
        "all_dense_views": 3 * config["dense_views_per_ring"],
        "minimum_clearance_mm_uv": (half - maximum).tolist(),
    }


def errors(prediction: Any, reference: Any, scale: float) -> dict[str, Any]:
    prediction, reference = (
        np.asarray(prediction, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )
    if (
        not np.isfinite(prediction).all()
        or not np.isfinite(reference).all()
        or (reference < 0).any()
    ):
        raise ValueError("invalid projection comparison")
    valid = reference > 0
    if np.any(prediction[~valid] != 0):
        raise ValueError("nonzero prediction for an exact zero expectation")
    error = (
        math.sqrt(scale) * np.abs(prediction[valid] - reference[valid]) / np.sqrt(reference[valid])
    )
    return {
        "positive_reference_pixels": int(valid.sum()),
        "zero_reference_pixels": int((~valid).sum()),
        "p95_poisson_sd": float(np.quantile(error, 0.95)) if error.size else 0.0,
        "maximum_poisson_sd": float(error.max()) if error.size else 0.0,
    }


def fixed_metric(physics: dict[str, Any]) -> dict[str, Any]:
    """New determinant-one path Fisher template, using physical inputs only."""
    paths = np.array([200.0, 10.0], dtype=np.float64)
    coefficients = physics["mu_mm_inv"].astype(np.float64)
    photons = physics["weights"].astype(np.float64) * physics["response"]
    photons *= np.exp(-paths @ coefficients)
    means = photons.sum(axis=1)
    if (means <= 0).any():
        raise ValueError("nonpositive model means cannot define the pilot metric")
    jacobian = -(photons @ coefficients.T)
    fisher = jacobian.T @ (jacobian / means[:, None])
    fisher = 0.5 * (fisher + fisher.T)
    eigenvalues = np.linalg.eigvalsh(fisher)
    if eigenvalues[0] <= 0 or eigenvalues[-1] / eigenvalues[0] > 1e6:
        raise ValueError("physical metric is outside the canonical SPD envelope")
    normalisation = float(np.sqrt(np.linalg.det(fisher)))
    metric = fisher / normalisation
    return {
        "id": "open-icrp-spekpy-path-fisher-det1-development-v1",
        "paths_water_bone_mm": paths.tolist(),
        "fisher_per_mm_squared": fisher.tolist(),
        "normalisation_per_mm_squared": normalisation,
        "matrix": metric.tolist(),
        "eigenvalues": np.linalg.eigvalsh(metric).tolist(),
        "condition": float(eigenvalues[-1] / eigenvalues[0]),
        "scope": "physical-input coupling template; no anatomy or counts; not a voxel Hessian",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("anatomy", "physics", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    library = helpers()
    config = json.loads(args.config.read_text())
    anatomy = json.loads((args.anatomy / "anatomy.json").read_text())
    if (
        config["case"] != 2
        or {k: v["sha256"] for k, v in anatomy["source_files"].items()} != config["source_sha256"]
        or anatomy["native_crop_xyz"] != config["native_crop_xyz"]
        or anatomy["forward_grid"]["spacing_mm"] != config["native_spacing_mm_xyz"]
    ):
        raise ValueError("anatomy does not match the fixed case-2 crop and source identity")
    metadata, physics, spec = library.load_physics(args.physics)
    if metadata.get("model_id") != config["physics_model_id"]:
        raise ValueError("wrong physical model for this new study")
    if anatomy["outputs"]["forward-fractions.npy"]["role"] != "observation generation only":
        raise ValueError("the forward field has an unexpected scientific role")
    fields = library.checked_array(args.anatomy, "forward-fractions.npy", anatomy["outputs"])
    grid = library.grid_from(anatomy["forward_grid"])
    if list(grid.shape) != config["native_shape_zyx"] or fields.shape != (2, *grid.shape):
        raise ValueError("wrong native anatomical grid")
    np.testing.assert_allclose(
        metadata["validation"]["material_checks"]["reference_80kev_mm_inv"],
        config["scalar_coefficients_80kev_mm_inverse"],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        physics["incident_weights"].astype(np.float64).sum(),
        config["incident_photons_per_ray"],
        rtol=2e-7,
    )
    fractions64 = fields.astype(np.float64)
    scalar = np.asarray(
        np.einsum("m,mzyx->zyx", config["scalar_coefficients_80kev_mm_inverse"], fractions64),
        dtype=np.float32,
    )
    geometry = detector(config)
    geometry_check = coverage(grid, config)
    metric = fixed_metric(physics)
    poses = [pose_at(v["tilt_degrees"], v["yaw_degrees"]) for v in config["probe_views"]]
    sources = experiment_sources(
        __file__,
        args.config,
        extra={
            "inputs/anatomy.json": args.anatomy / "anatomy.json",
            "inputs/forward-fractions.npy": args.anatomy / "forward-fractions.npy",
            "inputs/physics.npz": args.physics / "physics.npz",
            "inputs/physics-metadata.json": args.physics / "metadata.json",
            "helpers/spectral-run.py": Path(library.__file__),
            **library.physics_preparation_sources(args.physics, metadata),
        },
    )
    for path in Path(__file__).parent.glob("*.py"):
        sources[f"reconstruction-study/{path.name}"] = path
    for filename, key in (
        ("prepare_vertebrae.py", "script_sha256"),
        ("prepare_anatomy.py", "helper_sha256"),
        ("source-license-manifest.json", "source_license_manifest_sha256"),
    ):
        path = (
            Path(library.__file__).with_name(filename)
            if filename == "prepare_anatomy.py"
            else args.anatomy / filename
        )
        if library.digest(path) != anatomy["implementation"][key]:
            raise ValueError("anatomy preparation source/rights changed")
        sources[f"preparation/{filename}"] = path
    output = private_output(args.output, repository_root(__file__))
    with RunRecorder(
        output,
        configuration={
            **config,
            "geometry": asdict(geometry),
            "poses": [asdict(p) for p in poses],
            "native_grid": asdict(grid),
            "coverage": geometry_check,
            "pilot_metric": metric,
        },
        sources=sources,
    ) as run:
        wp.init()
        scalar_device = wp.array(scalar.ravel(), dtype=wp.float32, device=args.device)
        scalar_pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=args.device)
        depth = wp.empty(geometry.pixels, dtype=wp.float32, device=args.device)
        mean = wp.empty_like(depth)
        transmission = prepare_transmission(
            TransmissionSpec(beam="scalar"), max_pixels=geometry.pixels, device=args.device
        )
        setup: list[dict[str, Any]] = []
        measurements: list[dict[str, Any]] = []
        predictions: dict[tuple[int, int], tuple[Any, Any]] = {}
        for samples in config["probe_samples"]:
            begin = time.perf_counter()
            forward = library.Forward(grid, geometry, fields, physics, spec, samples, args.device)
            projection = prepare_projection(
                grid,
                geometry,
                ProjectionSpec(samples, active_pose=False, active_volume=False),
                device=args.device,
            )
            setup.append({"samples": samples, "seconds": time.perf_counter() - begin})
            for index, pose in enumerate(poses):
                mono: Any = None
                spectral: Any = None
                mono_seconds, spectral_seconds = 0.0, 0.0
                for repeat in range(2):
                    scalar_pose.assign(np.asarray(pose.packed(), dtype=np.float64))
                    begin = time.perf_counter()
                    project_optical_depth(
                        scalar_device, scalar_pose, workspace=projection, out_L=depth
                    )
                    transmit(
                        depth,
                        config["incident_photons_per_ray"],
                        workspace=transmission,
                        out_counts=mean,
                    )
                    mono = mean.numpy().reshape(geometry.shape)
                    mono_seconds = time.perf_counter() - begin
                    begin = time.perf_counter()
                    spectral = forward.project(pose)
                    spectral_seconds = time.perf_counter() - begin
                    measurements.append(
                        {
                            "samples": samples,
                            "view": index,
                            "repeat": repeat,
                            "mono_seconds": mono_seconds,
                            "spectral_seconds": spectral_seconds,
                        }
                    )
                if mono is None or spectral is None:
                    raise ValueError("the prescribed projection repetitions did not execute")
                predictions[samples, index] = (mono, spectral)
                for name, values in (("mono", mono), ("spectral", spectral)):
                    library.write_array(run, f"means/{name}-{samples}-view{index}.npy", values)
                print(
                    f"native samples{samples} view{index}: mono{mono_seconds:.4f}s "
                    f"spectral{spectral_seconds:.4f}s",
                    flush=True,
                )
        reference_started = time.perf_counter()
        pixels = list(
            itertools.product(config["independent_probe_rows"], config["independent_probe_columns"])
        )
        pixels[pixels.index((32, 38))] = (48, 48)
        coefficient64 = physics["mu_mm_inv"].astype(np.float64)
        incident_response = physics["weights"].astype(np.float64) * physics["response"]
        flat_materials: Any = fractions64.reshape(2, -1)
        scalar64: Any = scalar.astype(np.float64).ravel()
        independent: list[dict[str, Any]] = []
        scale = config["sampling_gate_photons_per_ray"] / config["incident_photons_per_ray"]
        for index, pose in enumerate(poses):
            mono_reference: list[float] = []
            spectral_reference: list[Any] = []
            for row, col in pixels:
                paths = np.array(
                    [
                        integrate_sampled_field(v, grid, geometry, pose, row, col, validate=False)
                        for v in flat_materials
                    ]
                )
                optical_depth = integrate_sampled_field(
                    scalar64, grid, geometry, pose, row, col, validate=False
                )
                mono_reference.append(config["incident_photons_per_ray"] * math.exp(-optical_depth))
                spectral_reference.append(incident_response @ np.exp(-paths @ coefficient64))
            rr, cc = np.array(pixels).T
            reference_values = (np.array(mono_reference), np.array(spectral_reference).T)
            for samples in config["probe_samples"]:
                mono, spectral = predictions[samples, index]
                independent.append(
                    {
                        "samples": samples,
                        "view": index,
                        "mono": errors(mono[rr, cc], reference_values[0], scale),
                        "spectral_per_channel": [
                            errors(spectral[c, rr, cc], reference_values[1][c], scale)
                            for c in range(3)
                        ],
                    }
                )
            run.write_json(
                f"independent/view{index}.json",
                {
                    "pixels_row_column": pixels,
                    "mono_mean": mono_reference,
                    "spectral_mean": np.array(spectral_reference).T.tolist(),
                },
            )
        differences: list[dict[str, Any]] = []
        low, high = config["probe_samples"]
        for index in range(len(poses)):
            a, b = predictions[low, index], predictions[high, index]
            differences.append(
                {
                    "view": index,
                    "mono": errors(a[0], b[0], scale),
                    "spectral_per_channel": [errors(a[1][c], b[1][c], scale) for c in range(3)],
                }
            )
        warm = [r for r in measurements if r["samples"] == low and r["repeat"] == 1]
        times = [r["mono_seconds"] + r["spectral_seconds"] for r in warm]
        gates: list[bool] = []
        for family in (independent, differences):
            for row in family:
                for channel in (row["mono"], *row["spectral_per_channel"]):
                    channel["passed"] = (
                        channel["p95_poisson_sd"] < config["sampling_p95_poisson_sd_max"]
                        and channel["maximum_poisson_sd"] < config["sampling_max_poisson_sd_max"]
                    )
                    gates.append(channel["passed"])
        run.write_json(
            "probe.json",
            {
                "status": "three-view development probe complete; no full acquisition or solve",
                "measured_projections": measurements,
                "workspace_setup": setup,
                "independent_checks": independent,
                "full_image_doubled_sampling": differences,
                "all_probe_sampling_gates_passed": all(gates),
                "sampling_gate_count": len(gates),
                "independent_reference_seconds": time.perf_counter() - reference_started,
                "extrapolated_full144_forward_seconds_min_max": [
                    144 * min(times),
                    144 * max(times),
                ],
                "forecast_scope": (
                    "forward plus synchronous mean export only; excludes all-view independent "
                    "checks, count sampling, disk recording and inverse solve"
                ),
                "total_seconds": time.perf_counter() - started,
                "inverse_sampling_check": (
                    "deferred; no inverse observations or fitting authorised by this probe"
                ),
                "counts_generated": False,
                "reconstruction_executed": False,
            },
        )
    print(f"Probe complete: {output}", flush=True)


if __name__ == "__main__":
    main()
