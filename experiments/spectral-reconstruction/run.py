"""Record CT-derived spectral observations and reconstruct material fractions.

Acquisition and reconstruction have separate immutable records. Evaluation-only
material fields are loaded after fitting; no anatomy-dependent initialisation is
accepted. This finite-view simulation uses an explicitly ideal detector model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import warp as wp

from dpt.detector import ObservationIdentity, prepare_detector, sample_poisson_counts
from dpt.experiments import RunRecorder, experiment_sources, private_output, repository_root
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_projection import prepare_material_projection, project_material_paths
from dpt.materials import Provenance
from dpt.projection import ProjectionSpec
from dpt.spectral import SpectralSpec, prepare_spectral, spectral_signal
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_array(run: RunRecorder, name: str, array: np.ndarray) -> None:
    payload = io.BytesIO()
    np.save(payload, np.ascontiguousarray(array), allow_pickle=False)
    run.write_bytes(name, payload.getvalue())


def checked_array(folder: Path, name: str, entries: dict) -> np.ndarray:
    path = folder / name
    if digest(path) != entries[name]["sha256"]:
        raise ValueError(f"recorded input changed: {path}")
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.float32 or not values.flags.c_contiguous:
        raise ValueError(f"expected native contiguous float32 input: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite input: {path}")
    return values


def load_physics(folder: Path) -> tuple[dict, dict, SpectralSpec]:
    metadata = json.loads((folder / "metadata.json").read_text())
    if digest(folder / "physics.npz") != metadata["physics_npz_sha256"]:
        raise ValueError("physics archive differs from its frozen provenance")
    with np.load(folder / "physics.npz", allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    for name, array in arrays.items():
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"physics array {name} must be finite float32")
    coefficients = arrays["mu_mm_inv"]
    response = arrays["response"]
    if np.any(response < 0) or np.any(response > 1) or np.any(response.sum(axis=0) > 1):
        raise ValueError("this acquisition needs disjoint photon-count response probabilities")
    spec = SpectralSpec(
        materials=coefficients.shape[0],
        energies=coefficients.shape[1],
        coefficients_provenance=tuple(Provenance(**p) for p in metadata["coefficients_provenance"]),
        spectrum_provenance=Provenance(**metadata["spectrum_provenance"]),
        response_provenance=Provenance(**metadata["response_provenance"]),
        output_unit="counts",
        input_description=(
            "Fixed reference-density water/bone fractions; supplied open-beam integrated "
            "photon weights per pixel; ideal nonoverlapping energy channels, unit quantum "
            "efficiency, no scatter, blur, pileup, read noise or inferred vendor calibration."
        ),
    )
    return metadata, arrays, spec


def physics_preparation_sources(folder: Path, metadata: dict) -> dict[str, Path]:
    """Retain approved executed preparation sources; never import metadata paths.

    Known filenames and output labels are selected here. Hashes bind their
    retained bytes to the supplied physical model, including its rights record.
    Historical NIST preparations keep their own generator and identity.
    """
    folder = folder.resolve()
    sources = {}

    def checked(relative: str, expected: str, label: str) -> None:
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"physical preparation identity changed: {relative}")
        sources[label] = path

    model = metadata.get("model_id")
    if model == "xraylib-icrp-spekpy120-al25-v1":
        for name in ("prepare_open_physics.py", "provision_inputs.py"):
            checked(name, metadata["source_sha256"][name], f"preparation/{name}")
        for name, key in (
            ("source-license-manifest.json", "source_license_manifest_sha256"),
            ("installed-distributions.json", "installed_distributions_sha256"),
        ):
            checked(name, metadata[key], f"inputs/physics/{name}")
        notices = {
            "sources/xraylib-license_all.txt",
            "sources/xraylib-license_tom.txt",
            "sources/xraylib-license_teemu.txt",
            "sources/xraylib-xraylib-nist-compounds-internal.h",
            "sources/spekpy-licence.txt",
        }
        records = metadata["notices"]
        if len(records) != len(notices) or {r["file"] for r in records} != notices:
            raise ValueError("unexpected open-physics notice inventory")
        for record in records:
            name = record["file"]
            checked(name, record["sha256"], f"inputs/physics/{name}")
        checked(
            "source-spectrum.csv",
            metadata["spectrum_provenance"]["sha256"],
            "inputs/physics/source-spectrum.csv",
        )
    elif model is None and metadata.get("kind") == (
        "simulated_spectral_acquisition_with_reference_material_coefficients"
    ):
        checked(
            "prepare_physics.py",
            metadata["implementation"]["sha256"],
            "preparation/physics.py",
        )
        checked(
            "spekpy-native-spectrum.csv",
            metadata["spectrum_provenance"]["sha256"],
            "inputs/physics/spekpy-native-spectrum.csv",
        )
    else:
        raise ValueError(f"unrecognised physical preparation model: {model!r}")
    checked(
        "acquisition-definition.json",
        metadata["response_provenance"]["sha256"],
        "inputs/physics/acquisition-definition.json",
    )
    return sources


def grid_from(record: dict) -> GridSpec:
    return GridSpec(**{key: tuple(value) for key, value in record.items()})


def material_metric(physics: dict) -> tuple[tuple[float, float, float], dict]:
    """Fixed path-space Fisher template; depends on calibration, not anatomy."""
    paths = np.array([200.0, 10.0], dtype=np.float64)
    mu = physics["mu_mm_inv"].astype(np.float64)
    terms = physics["weights"].astype(np.float64) * physics["response"]
    terms *= np.exp(-paths @ mu)
    means = terms.sum(axis=1)
    jacobian = -(terms @ mu.T)
    fisher = jacobian.T @ (jacobian / means[:, None])
    metric = 2 * fisher / np.trace(fisher)
    return (float(metric[0, 0]), float(metric[0, 1]), float(metric[1, 1])), {
        "kind": "fixed two-material path Fisher template, normalised to trace 2",
        "reference_paths_mm": paths.tolist(),
        "fisher_per_mm_squared": fisher.tolist(),
        "matrix": metric.tolist(),
        "condition": float(np.linalg.cond(metric)),
        "source": "frozen physical calibration only; no reference CT or withheld projections",
        "scope": "material coupling template, not the full spatial voxel Hessian",
    }


def orbit(
    grid: GridSpec, views_per_ring: int, height: int, width: int
) -> tuple[DetectorGeometry, list[RigidTransform], dict]:
    """Three tilted great-circle orbits, expressed as equivalent object poses."""
    poses = []
    angles = []
    for tilt in (-15.0, 0.0, 15.0):
        a = math.radians(tilt)
        rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        for index in range(views_per_ring):
            yaw = 2 * math.pi * index / views_per_ring
            rz = np.array(
                [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
            )
            poses.append(RigidTransform(rotation=tuple((rz @ rx).ravel())))
            angles.append({"tilt_degrees": tilt, "yaw_degrees": math.degrees(yaw)})
    corners = [
        grid.grid_to_object(tuple(p)) for p in itertools.product(*zip(*grid.support, strict=True))
    ]
    detector_points = []
    for pose in poses:
        for corner in corners:
            x, y, z = pose.point(corner)
            scale = 1500 / (y + 1000)
            detector_points.append((x * scale, z * scale))
    required_half = np.max(np.abs(detector_points), axis=0)
    # Fixed corner-based coverage uses only the public grid box, not tissue masks.
    pitch = np.ceil(2 * required_half * 1.03 / [width, height] * 10) / 10
    geometry = DetectorGeometry(
        source_mm=(0, -1000, 0),
        origin_mm=(-(width - 1) * pitch[0] / 2, 500, -(height - 1) * pitch[1] / 2),
        u=(1, 0, 0),
        v=(0, 0, 1),
        spacing_mm=tuple(pitch),
        shape=(height, width),
    )
    clearance = np.array([width, height]) * pitch / 2 - required_half
    if np.any(clearance <= 0):
        raise ValueError("detector truncates the phantom support box")
    return (
        geometry,
        poses,
        {
            "description": "three tilted great-circle source/detector orbits; pose Rz(yaw)Rx(tilt)",
            "source_isocentre_mm": 1000,
            "source_detector_mm": 1500,
            "angles": angles,
            "minimum_box_corner_clearance_mm": clearance.tolist(),
            "completeness": (
                "finite-view regularised problem; no continuous cone-beam completeness claim"
            ),
        },
    )


class Forward:
    """One persistent canonical projection/spectral workspace for offline export."""

    def __init__(self, grid, geometry, fields, physics, spec, samples, device):
        self.device = device
        self.geometry = geometry
        self.channels = physics["response"].shape[0]
        self.fields = wp.array(fields.ravel(), dtype=wp.float32, device=device)
        self.pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
        self.paths = wp.empty(spec.materials * geometry.pixels, dtype=wp.float32, device=device)
        self.mean = wp.empty(geometry.pixels, dtype=wp.float32, device=device)
        self.coefficients = wp.array(physics["mu_mm_inv"].ravel(), dtype=wp.float32, device=device)
        self.weights = [wp.array(v, dtype=wp.float32, device=device) for v in physics["weights"]]
        self.response = [wp.array(v, dtype=wp.float32, device=device) for v in physics["response"]]
        self.projector = prepare_material_projection(
            grid,
            geometry,
            ProjectionSpec(samples, active_pose=False),
            materials=spec.materials,
            fields=self.fields,
            out_paths=self.paths,
            device=device,
        )
        self.spectral = prepare_spectral(spec, max_pixels=geometry.pixels, device=device)
        for w, r in zip(self.weights, self.response, strict=True):
            self.spectral.validate_inputs(
                self.coefficients, w, r, pixels=geometry.pixels, probability_response=True
            )

    def project(self, pose: RigidTransform) -> np.ndarray:
        self.pose.assign(np.asarray(pose.packed(), dtype=np.float64))
        project_material_paths(self.pose, workspace=self.projector)
        means = []
        for weights, response in zip(self.weights, self.response, strict=True):
            spectral_signal(
                self.paths,
                self.coefficients,
                weights,
                response,
                out_mean=self.mean,
                workspace=self.spectral,
            )
            means.append(self.mean.numpy().reshape(self.geometry.shape))
        return np.stack(means)


def acquire(args) -> None:
    anatomy = json.loads((args.anatomy / "anatomy.json").read_text())
    physics_metadata, physics, spec = load_physics(args.physics)
    preparation_sources = physics_preparation_sources(args.physics, physics_metadata)
    fields = checked_array(args.anatomy, "forward-fractions.npy", anatomy["outputs"])
    grid = grid_from(anatomy["forward_grid"])
    geometry, poses, coverage = orbit(grid, args.views_per_ring, args.height, args.width)
    heldout = [i for i in range(len(poses)) if i % args.views_per_ring % 8 == 4]
    fitted = [i for i in range(len(poses)) if i not in heldout]
    if not heldout:
        raise ValueError("the requested orbit must contain held-out views")
    configuration = {
        "anatomy": str(args.anatomy),
        "physics": str(args.physics),
        "physics_npz_sha256": digest(args.physics / "physics.npz"),
        "physics_metadata_sha256": digest(args.physics / "metadata.json"),
        "physics_model_id": physics_metadata.get("model_id", "historical-nist-icru44"),
        "geometry": asdict(geometry),
        "poses": [asdict(p) for p in poses],
        "forward_grid": anatomy["forward_grid"],
        "inverse_grid": anatomy["inverse_grid"],
        "forward_samples_per_ray": args.samples,
        "seed": args.seed,
        "fit_views": fitted,
        "heldout_views": heldout,
        "coverage": coverage,
        "observation_identity": "seed fixed, observation_id=view, energy_offset=channel",
        "model_role": "assigned anatomical phantom with simulated spectral photon counts",
    }
    sources = experiment_sources(
        __file__,
        extra={
            "inputs/anatomy.json": args.anatomy / "anatomy.json",
            "inputs/forward-fractions.npy": args.anatomy / "forward-fractions.npy",
            "inputs/physics.npz": args.physics / "physics.npz",
            "inputs/physics-metadata.json": args.physics / "metadata.json",
            "preparation/anatomy.py": Path(__file__).with_name("prepare_anatomy.py"),
            "preparation/vertebrae.py": Path(__file__).with_name("prepare_vertebrae.py"),
            **preparation_sources,
        },
    )
    destination = private_output(args.output, repository_root(__file__))
    forward = Forward(grid, geometry, fields, physics, spec, args.samples, args.device)
    sampler = prepare_detector(max_pixels=geometry.pixels, device=args.device)
    mean_device = wp.empty(geometry.pixels, dtype=wp.float32, device=args.device)
    draws = wp.empty(geometry.pixels, dtype=wp.uint64, device=args.device)
    with RunRecorder(destination, configuration=configuration, sources=sources) as run:
        means, counts = [], []
        started = time.perf_counter()
        for index, pose in enumerate(poses):
            mean = forward.project(pose)
            observed = []
            for channel in range(spec_input_channels := physics["response"].shape[0]):
                mean_device.assign(mean[channel].ravel())
                sample_poisson_counts(
                    mean_device,
                    out_counts=draws,
                    identity=ObservationIdentity(args.seed, index, energy_offset=channel),
                    workspace=sampler,
                )
                host = draws.numpy()
                if np.max(host) > 2**24:
                    raise ValueError("Poisson observation exceeds exact float32 integer envelope")
                observed.append(host.astype(np.float32).reshape(geometry.shape))
            means.append(mean)
            counts.append(np.stack(observed))
            if index % 6 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "acquire",
                            "views": index + 1,
                            "total": len(poses),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        write_array(run, "expected-counts.npy", np.stack(means))
        write_array(run, "observed-counts.npy", np.stack(counts))
        # Independent FP64 piecewise-Gauss integration of the trilinear field.
        references = []
        convergence = []
        doubled = Forward(grid, geometry, fields, physics, spec, 2 * args.samples, args.device)
        for view in (0, len(poses) // 2):
            expected = forward.project(poses[view])
            gpu_paths = forward.paths.numpy().reshape(spec.materials, *geometry.shape)
            finer = doubled.project(poses[view])
            whitened = np.abs(expected.astype(np.float64) - finer) / np.sqrt(finer)
            maximum = float(whitened.max())
            convergence.append(
                {
                    "view": view,
                    "samples": [args.samples, 2 * args.samples],
                    "max_difference_in_poisson_sd": maximum,
                    "limit": 0.1,
                }
            )
            if maximum > 0.1:
                raise ValueError(
                    "forward quadrature change exceeds 0.1 Poisson SD; increase samples"
                )
            bone_pixel = np.unravel_index(np.argmax(gpu_paths[1]), geometry.shape)
            water_pixel = np.unravel_index(np.argmax(gpu_paths[0]), geometry.shape)
            edge_pixel = np.unravel_index(
                np.argmax(np.abs(np.gradient(gpu_paths[0], axis=1))), geometry.shape
            )
            for role, (row, column) in [
                ("vacuum", (0, 0)),
                ("bone", bone_pixel),
                ("water", water_pixel),
                ("boundary", edge_pixel),
            ]:
                oracle = [
                    integrate_sampled_field(
                        v.ravel(),
                        grid,
                        geometry,
                        poses[view],
                        int(row),
                        int(column),
                        validate=False,
                    )
                    for v in fields
                ]
                transmission = np.exp(-np.asarray(oracle, dtype=np.float64) @ physics["mu_mm_inv"])
                cpu_mean = np.sum(physics["weights"] * physics["response"] * transmission, axis=1)
                scaled_error = np.abs(expected[:, row, column] - cpu_mean) / np.sqrt(cpu_mean)
                if not np.isfinite(scaled_error).all() or np.max(scaled_error) > 0.1:
                    raise ValueError("independent CPU radiograph check exceeds 0.1 Poisson SD")
                references.append(
                    {
                        "view": view,
                        "role": role,
                        "row": int(row),
                        "column": int(column),
                        "cpu_paths_mm": oracle,
                        "gpu_paths_mm": gpu_paths[:, row, column].tolist(),
                        "cpu_expected_counts": cpu_mean.tolist(),
                        "gpu_expected_counts": expected[:, row, column].tolist(),
                        "max_count_error_in_poisson_sd": float(scaled_error.max()),
                        "limit": 0.1,
                        "passed": True,
                    }
                )
        run.write_json("independent-ray-checks.json", references)
        run.write_json("quadrature-convergence.json", convergence)
        run.set_metadata(
            device=str(wp.get_device(args.device)),
            channels=spec_input_channels,
            elapsed_seconds=time.perf_counter() - started,
        )


def solve(args) -> None:
    # Import only for this explicitly requested new solver execution.
    from dpt.material_reconstruction import (
        MaterialReconstruction,
        MaterialReconstructionSettings,
        MaterialReconstructionView,
    )

    acquisition = json.loads((args.acquisition / "run.json").read_text())
    if acquisition["status"] != "complete":
        raise ValueError("observations must have a complete immutable acquisition record")
    config = acquisition["configuration"]
    for name, sha in acquisition["output_sha256"].items():
        if digest(args.acquisition / name) != sha:
            raise ValueError(f"acquisition output changed: {name}")
    physics_metadata, physics, spec = load_physics(args.physics)
    preparation_sources = physics_preparation_sources(args.physics, physics_metadata)
    if digest(args.physics / "physics.npz") != config["physics_npz_sha256"]:
        raise ValueError("reconstruction physics differs from the frozen acquisition")
    if digest(args.physics / "metadata.json") != config["physics_metadata_sha256"]:
        raise ValueError("physics provenance differs from the frozen acquisition")
    observed = np.load(args.acquisition / "observed-counts.npy", allow_pickle=False)
    geometry = DetectorGeometry(**{k: tuple(v) for k, v in config["geometry"].items()})
    poses = [RigidTransform(**{k: tuple(v) for k, v in p.items()}) for p in config["poses"]]
    grid = grid_from(config["inverse_grid"])
    metric, metric_metadata = (
        material_metric(physics) if args.metric == "spectral" else (None, {"kind": "Euclidean"})
    )
    settings = MaterialReconstructionSettings(
        iterations=args.iterations,
        initial_step=args.initial_step,
        regularisation_mm_inverse=args.regularisation,
        mapping_step=args.mapping_step,
    )
    initial = np.empty((spec.materials, *grid.shape), dtype=np.float32)
    initial[0].fill(args.initial_water)
    initial[1].fill(args.initial_bone)
    views = tuple(
        MaterialReconstructionView(
            geometry=geometry,
            pose=poses[i],
            counts=observed[i],
            weights=physics["weights"],
            response=physics["response"],
        )
        for i in config["fit_views"]
    )
    solver = MaterialReconstruction(
        grid=grid,
        views=views,
        coefficients=physics["mu_mm_inv"],
        spectral_spec=spec,
        initial_fractions=initial,
        settings=settings,
        samples_per_ray=args.samples,
        device=args.device,
        material_metric=metric,
    )
    configuration = {
        "acquisition": str(args.acquisition),
        "acquisition_sha256": digest(args.acquisition / "run.json"),
        "physics_model_id": physics_metadata.get("model_id", "historical-nist-icru44"),
        "settings": asdict(settings),
        "samples_per_ray": args.samples,
        "initialisation": (
            f"uniform water={args.initial_water} and bone={args.initial_bone} "
            "throughout the known grid box; no internal anatomy used"
        ),
        "reference_access": "reference volumes read only after solver completion for evaluation",
        "fit_views": config["fit_views"],
        "heldout_views": config["heldout_views"],
        "material_metric": metric_metadata,
    }
    sources = experiment_sources(
        __file__,
        extra={
            "inputs/acquisition-run.json": args.acquisition / "run.json",
            "inputs/observed-counts.npy": args.acquisition / "observed-counts.npy",
            "inputs/physics.npz": args.physics / "physics.npz",
            "inputs/physics-metadata.json": args.physics / "metadata.json",
            **preparation_sources,
        },
    )
    with RunRecorder(
        private_output(args.output, repository_root(__file__)),
        configuration=configuration,
        sources=sources,
    ) as run:
        write_array(run, "initial-fractions.npy", initial)

        def checkpoint(iteration, record):
            print(json.dumps({"stage": "solve", "iteration": iteration, **record}), flush=True)
            if iteration in (1, 10, 25, 50, 100, 200, 500, 1000):
                write_array(
                    run, f"checkpoints/fractions-{iteration:04d}.npy", solver.fractions_numpy()
                )

        report = solver.solve(callback=checkpoint)
        fractions = solver.fractions_numpy()
        write_array(run, "fractions.npy", fractions)
        write_array(run, "fit-predictions.npy", solver.predictions_numpy())
        run.write_json("solver-report.json", report)
        # Fitting has finished. Only now read withheld reference fields/means.
        anatomy_path = Path(config["anatomy"])
        frozen_anatomy = args.acquisition / "sources/inputs/anatomy.json"
        if digest(frozen_anatomy) != acquisition["source_sha256"]["inputs/anatomy.json"]:
            raise ValueError("frozen acquisition anatomy identity changed")
        anatomy = json.loads(frozen_anatomy.read_text())
        if digest(anatomy_path / "anatomy.json") != digest(frozen_anatomy):
            raise ValueError("external anatomy metadata differs from the acquisition snapshot")
        reference = checked_array(anatomy_path, "evaluation-fractions.npy", anatomy["outputs"])
        run.write_json(
            "evaluation-provenance.json",
            {
                "anatomy_json_sha256": digest(anatomy_path / "anatomy.json"),
                "reference": anatomy["outputs"]["evaluation-fractions.npy"],
                "access": "after solve returned; not used for initialisation, priors or stopping",
            },
        )
        metrics = {
            "material_names": ["water", "reference cortical bone"],
            "material_rmse": np.sqrt(
                np.mean((fractions - reference).astype(np.float64) ** 2, axis=(1, 2, 3))
            ).tolist(),
            "initial_material_rmse": np.sqrt(
                np.mean((initial - reference).astype(np.float64) ** 2, axis=(1, 2, 3))
            ).tolist(),
            "minimum_fraction": float(fractions.min()),
            "maximum_sum": float(fractions.sum(axis=0).max()),
            "grid": config["inverse_grid"],
            "scope": "coarse assigned-material phantom; no clinical composition truth",
        }
        difference = (fractions - reference).astype(np.float64)
        metrics["material_mean_bias"] = difference.mean(axis=(1, 2, 3)).tolist()
        voxel_volume = math.prod(grid.spacing_mm)
        metrics["reference_material_volume_mm3"] = (
            reference.sum(axis=(1, 2, 3), dtype=np.float64) * voxel_volume
        ).tolist()
        metrics["recovered_material_volume_mm3"] = (
            fractions.sum(axis=(1, 2, 3), dtype=np.float64) * voxel_volume
        ).tolist()
        metrics["evaluation_regions"] = {}
        for name, region in [
            ("foreground_reference_sum_gt_0.1", reference.sum(axis=0) > 0.1),
            ("bone_reference_fraction_gt_0.05", reference[1] > 0.05),
        ]:
            metrics["evaluation_regions"][name] = {
                "voxels": int(region.sum()),
                "material_rmse": np.sqrt(np.mean(difference[:, region] ** 2, axis=1)).tolist(),
                "material_bias": np.mean(difference[:, region], axis=1).tolist(),
            }
        predicted = Forward(grid, geometry, fractions, physics, spec, args.samples, args.device)
        baseline = Forward(grid, geometry, initial, physics, spec, args.samples, args.device)
        finer = Forward(grid, geometry, fractions, physics, spec, 2 * args.samples, args.device)
        inverse_checks = []
        for view in (config["fit_views"][0], config["fit_views"][len(views) // 2]):
            coarse_mean = predicted.project(poses[view])
            fine_mean = finer.project(poses[view])
            delta = np.abs(coarse_mean.astype(np.float64) - fine_mean) / np.sqrt(fine_mean)
            inverse_checks.append(
                {
                    "view": view,
                    "samples": [args.samples, 2 * args.samples],
                    "max_difference_in_poisson_sd": float(delta.max()),
                    "limit": 0.1,
                    "passed": bool(delta.max() <= 0.1),
                }
            )
        run.write_json("inverse-quadrature-checks.json", inverse_checks)
        heldout_predictions, heldout_metrics = [], []
        for i in config["heldout_views"]:
            fit_mean = predicted.project(poses[i])
            initial_mean = baseline.project(poses[i])
            y = observed[i].astype(np.float64)
            heldout_predictions.append(fit_mean)
            row = {"view": i}
            for label, mean in [("reconstructed", fit_mean), ("initial", initial_mean)]:
                lam = mean.astype(np.float64)
                if np.any(lam <= 0):
                    raise ValueError("nonpositive prediction in held-out evaluation")
                term = lam - y
                positive = y > 0
                term[positive] += y[positive] * np.log(y[positive] / lam[positive])
                row[f"{label}_deviance_per_pixel_by_channel"] = (
                    2 * term.mean(axis=(1, 2))
                ).tolist()
            heldout_metrics.append(row)
        metrics["heldout"] = heldout_metrics
        write_array(run, "heldout-predictions.npy", np.stack(heldout_predictions))
        run.write_json("evaluation.json", metrics)
        run.set_metadata(
            device=str(wp.get_device(args.device)),
            scientific_acceptance="see independent tests and held-out diagnostics",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("acquire", "solve"))
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anatomy", type=Path)
    parser.add_argument("--acquisition", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--views-per-ring", type=int, default=24)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026091201)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--initial-step", type=float, default=1e-5)
    parser.add_argument("--initial-water", type=float, default=0.1)
    parser.add_argument("--initial-bone", type=float, default=0.02)
    parser.add_argument("--mapping-step", type=float, default=1.0)
    parser.add_argument("--regularisation", type=float, default=0.1)
    parser.add_argument("--metric", choices=("euclidean", "spectral"), default="euclidean")
    args = parser.parse_args()
    initial_values = (args.initial_water, args.initial_bone)
    if not all(math.isfinite(value) and value >= 0 for value in initial_values):
        parser.error("initial material fractions must be finite and nonnegative")
    if sum(initial_values) > 1:
        parser.error("initial material fractions must sum to at most one")
    if args.stage == "acquire" and args.anatomy is None:
        parser.error("acquire requires --anatomy")
    if args.stage == "solve" and args.acquisition is None:
        parser.error("solve requires --acquisition")
    wp.init()
    {"acquire": acquire, "solve": solve}[args.stage](args)


if __name__ == "__main__":
    main()
