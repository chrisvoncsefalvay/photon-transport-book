"""Generate one gated development acquisition; never fit a reference field."""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _study import coarse_cells, complete_record, digest, poisson_counts
from probe_acquisition import coverage, detector, errors, fixed_metric, helpers, pose_at

from dpt.experiments import RunRecorder, experiment_sources, private_output, repository_root
from dpt.geometry import RigidTransform
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth
from dpt.transmission import TransmissionSpec, prepare_transmission, transmit
from dpt.validation.projection import integrate_sampled_field

np: Any = importlib.import_module("numpy")
wp: Any = importlib.import_module("warp")


class ScalarForward:
    """Persistent canonical scalar projection and mean-count export."""

    def __init__(
        self, grid: Any, geometry: Any, field: Any, samples: int, beam: float, device: str
    ):
        self.geometry, self.beam = geometry, beam
        self.field = wp.array(field.ravel(), dtype=wp.float32, device=device)
        self.pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
        self.depth = wp.empty(geometry.pixels, dtype=wp.float32, device=device)
        self.mean = wp.empty_like(self.depth)
        self.projection = prepare_projection(
            grid,
            geometry,
            ProjectionSpec(samples, active_pose=False, active_volume=False),
            device=device,
        )
        self.transmission = prepare_transmission(
            TransmissionSpec(beam="scalar"), max_pixels=geometry.pixels, device=device
        )

    def project(self, pose: RigidTransform) -> Any:
        self.pose.assign(np.asarray(pose.packed(), dtype=np.float64))
        project_optical_depth(self.field, self.pose, workspace=self.projection, out_L=self.depth)
        transmit(self.depth, self.beam, workspace=self.transmission, out_counts=self.mean)
        return self.mean.numpy().reshape(self.geometry.shape)


def check_family(
    run: RunRecorder,
    *,
    name: str,
    fields: Any,
    scalar: Any,
    grid: Any,
    geometry: Any,
    poses: list[RigidTransform],
    physics: dict[str, Any],
    spec: Any,
    samples: tuple[int, int],
    config: dict[str, Any],
    device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    library = helpers()
    results: dict[int, tuple[Any, Any]] = {}
    timings: list[dict[str, Any]] = []
    start = time.perf_counter()
    for count in samples:
        mono_forward = ScalarForward(
            grid, geometry, scalar, count, config["incident_photons_per_ray"], device
        )
        material_forward = library.Forward(grid, geometry, fields, physics, spec, count, device)
        mono = np.empty((len(poses), *geometry.shape), dtype=np.float32)
        spectral = np.empty(
            (len(poses), physics["response"].shape[0], *geometry.shape), dtype=np.float32
        )
        for index, pose in enumerate(poses):
            tick = time.perf_counter()
            mono[index] = mono_forward.project(pose)
            mono_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            spectral[index] = material_forward.project(pose)
            timings.append(
                {
                    "samples": count,
                    "view": index,
                    "mono_seconds": mono_seconds,
                    "spectral_seconds": time.perf_counter() - tick,
                }
            )
            if (index + 1) % 24 == 0:
                print(f"{name} {count} samples: {index + 1}/{len(poses)} views", flush=True)
        results[count] = mono, spectral
        del mono_forward, material_forward
    forward_seconds = time.perf_counter() - start
    recording_started = time.perf_counter()
    for count, (mono, spectral) in results.items():
        library.write_array(run, f"qualification/{name}/scalar-{count}.npy", mono)
        library.write_array(run, f"qualification/{name}/spectral-{count}.npy", spectral)
    mean_recording_seconds = time.perf_counter() - recording_started
    pixels = list(
        itertools.product(config["independent_probe_rows"], config["independent_probe_columns"])
    )
    pixels[pixels.index((32, 38))] = (48, 48)
    rr, cc = np.array(pixels).T
    fractions64 = fields.astype(np.float64).reshape(fields.shape[0], -1)
    scalar64 = scalar.astype(np.float64).ravel()
    mu = physics["mu_mm_inv"].astype(np.float64)
    incident = physics["weights"].astype(np.float64) * physics["response"]
    scale = config["sampling_gate_photons_per_ray"] / config["incident_photons_per_ray"]
    gates: list[dict[str, Any]] = []
    oracle_mono: list[Any] = []
    oracle_spectral: list[Any] = []

    def compare(
        prediction: Any, reference: Any, role: str, view: int, channel: int, count: int
    ) -> None:
        entry = errors(prediction, reference, scale)
        entry.update(role=role, view=view, channel=channel, samples=count)
        entry["passed"] = (
            entry["p95_poisson_sd"] < config["sampling_p95_poisson_sd_max"]
            and entry["maximum_poisson_sd"] < config["sampling_max_poisson_sd_max"]
        )
        gates.append(entry)

    tick = time.perf_counter()
    for index, pose in enumerate(poses):
        exact_mono: Any = []
        exact_spectral: Any = []
        for row, column in pixels:
            path = np.array(
                [
                    integrate_sampled_field(f, grid, geometry, pose, row, column, validate=False)
                    for f in fractions64
                ]
            )
            depth = integrate_sampled_field(
                scalar64, grid, geometry, pose, row, column, validate=False
            )
            exact_mono.append(config["incident_photons_per_ray"] * math.exp(-depth))
            exact_spectral.append(incident @ np.exp(-path @ mu))
        exact_mono, exact_spectral = np.array(exact_mono), np.array(exact_spectral).T
        oracle_mono.append(exact_mono)
        oracle_spectral.append(exact_spectral)
        for count in samples:
            mono, spectral = results[count]
            compare(mono[index, rr, cc], exact_mono, "independent_scalar", index, 0, count)
            for channel in range(spectral.shape[1]):
                compare(
                    spectral[index, channel, rr, cc],
                    exact_spectral[channel],
                    "independent_spectral",
                    index,
                    channel,
                    count,
                )
        low, high = results[samples[0]], results[samples[1]]
        compare(low[0][index], high[0][index], "doubled_scalar", index, 0, samples[0])
        for channel in range(low[1].shape[1]):
            compare(
                low[1][index, channel],
                high[1][index, channel],
                "doubled_spectral",
                index,
                channel,
                samples[0],
            )
    oracle_seconds = time.perf_counter() - tick
    library.write_array(run, f"qualification/{name}/oracle-scalar.npy", np.stack(oracle_mono))
    library.write_array(run, f"qualification/{name}/oracle-spectral.npy", np.stack(oracle_spectral))
    report = {
        "role": "known-field quadrature qualification before observation draws",
        "grid": asdict(grid),
        "samples": samples,
        "gates": gates,
        "passed": all(g["passed"] for g in gates),
        "gate_count": len(gates),
        "independent_pixels_row_column": pixels,
        "oracle_seconds": oracle_seconds,
        "forward_export_seconds": forward_seconds,
        "mean_recording_seconds": mean_recording_seconds,
        "comparison_arrays": f"qualification/{name}/",
        "timings": timings,
        "error_exposure_photons_per_ray": config["sampling_gate_photons_per_ray"],
    }
    run.write_json(f"qualification/{name}.json", report)
    # Retain failed numerical qualification too; no count draw is allowed after failure.
    if not report["passed"]:
        raise ValueError(f"{name} failed the frozen sampling gates; no counts generated")
    return *results[samples[0]], report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "anatomy", "physics", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    library = helpers()
    probe = complete_record(args.probe)
    probe_result = json.loads((args.probe / "probe.json").read_text())
    if (
        digest(args.probe / "probe.json") != probe["output_sha256"]["probe.json"]
        or not probe_result["all_probe_sampling_gates_passed"]
    ):
        raise ValueError("the required three-view probe did not pass")
    config = json.loads(args.config.read_text())
    if any(probe["configuration"][key] != value for key, value in config.items()):
        raise ValueError("development configuration differs from the reviewed probe")
    for key, path in {
        "inputs/anatomy.json": args.anatomy / "anatomy.json",
        "inputs/forward-fractions.npy": args.anatomy / "forward-fractions.npy",
        "inputs/physics.npz": args.physics / "physics.npz",
        "inputs/physics-metadata.json": args.physics / "metadata.json",
    }.items():
        if digest(path) != probe["source_sha256"][key]:
            raise ValueError(f"input differs from accepted probe: {key}")
    anatomy = json.loads((args.anatomy / "anatomy.json").read_text())
    metadata, physics, spectral_spec = library.load_physics(args.physics)
    native = library.checked_array(args.anatomy, "forward-fractions.npy", anatomy["outputs"])
    grid = library.grid_from(anatomy["forward_grid"])
    scalar = np.asarray(
        np.einsum(
            "m,mzyx->zyx", config["scalar_coefficients_80kev_mm_inverse"], native.astype(np.float64)
        ),
        dtype=np.float32,
    )
    shape = tuple(config["pilot_inverse_shape_zyx"])
    coarse, inverse_grid = coarse_cells(native, grid, shape)
    coarse_scalar, _ = coarse_cells(scalar, grid, shape)
    geometry = detector(config)
    views = [
        {
            "id": index * config["dense_views_per_ring"] + yaw,
            "tilt_degrees": tilt,
            "yaw_degrees": yaw * 360 / config["dense_views_per_ring"],
            "pose": asdict(pose_at(tilt, yaw * 360 / config["dense_views_per_ring"])),
        }
        for index, tilt in enumerate(config["tilts_degrees"])
        for yaw in range(config["dense_views_per_ring"])
    ]
    poses = [RigidTransform(**{k: tuple(v) for k, v in view["pose"].items()}) for view in views]
    sources = experiment_sources(
        __file__,
        args.config,
        extra={
            "accepted-probe/run.json": args.probe / "run.json",
            "accepted-probe/probe.json": args.probe / "probe.json",
            **{
                f"accepted-probe/sources/{name}": args.probe / "sources" / name
                for name in probe["source_sha256"]
            },
            **{
                f"reconstruction-study/{path.name}": path
                for path in Path(__file__).parent.glob("*.py")
            },
            "helpers/spectral-run.py": Path(library.__file__),
            **library.physics_preparation_sources(args.physics, metadata),
        },
    )
    # Match canonical scientific bytes to the already accepted probe.
    for name, path in sources.items():
        if name.startswith("python/dpt/") and digest(path) != probe["source_sha256"][name]:
            raise ValueError(f"canonical source changed since probe: {name}")
    output = private_output(args.output, repository_root(__file__))
    configuration: dict[str, Any] = {
        **config,
        "execution_scope": "144 development views and qualification only",
        "coverage": coverage(grid, config),
        "native_grid": asdict(grid),
        "inverse_grid": asdict(inverse_grid),
        "geometry": asdict(geometry),
        "views": views,
        "metric": fixed_metric(physics),
        "native_samples": 1024,
        "inverse_samples": 512,
    }
    with RunRecorder(output, configuration=configuration, sources=sources) as run:
        wp.init()
        run.set_metadata(
            device=str(wp.get_device(args.device)),
            gpu_name=wp.get_device(args.device).name,
            physical_role=anatomy["scientific_role"],
        )
        started = time.perf_counter()
        mono, spectral, native_report = check_family(
            run,
            name="native",
            fields=native,
            scalar=scalar,
            grid=grid,
            geometry=geometry,
            poses=poses,
            physics=physics,
            spec=spectral_spec,
            samples=(1024, 2048),
            config=config,
            device=args.device,
        )
        _, _, inverse_report = check_family(
            run,
            name="inverse32",
            fields=coarse,
            scalar=coarse_scalar,
            grid=inverse_grid,
            geometry=geometry,
            poses=poses,
            physics=physics,
            spec=spectral_spec,
            samples=(512, 1024),
            config=config,
            device=args.device,
        )
        qualified_seconds = time.perf_counter() - started
        tick = time.perf_counter()
        scalar_counts, scalar_keys = poisson_counts(mono[:, None], config, "scalar")
        spectral_counts, spectral_keys = poisson_counts(spectral, config, "spectral")
        if len({tuple(k) for k in scalar_keys + spectral_keys}) != len(scalar_keys + spectral_keys):
            raise ValueError("noise namespaces collided")
        sampling_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        public_views: list[dict[str, Any]] = []
        for index, view in enumerate(views):
            names = {
                model: f"observations/{model}-view{index:03d}.npy"
                for model in ("scalar", "spectral")
            }
            library.write_array(run, names["scalar"], scalar_counts[index, 0])
            library.write_array(run, names["spectral"], spectral_counts[index])
            public_views.append(
                {
                    **view,
                    "scalar_counts": names["scalar"],
                    "spectral_counts": names["spectral"],
                    "sha256": {model: digest(output / name) for model, name in names.items()},
                }
            )
        library.write_array(
            run,
            "open-beam.npy",
            np.full(geometry.shape, config["incident_photons_per_ray"], dtype=np.float32),
        )
        run.write_bytes("physics/physics.npz", (args.physics / "physics.npz").read_bytes())
        run.write_bytes("physics/metadata.json", (args.physics / "metadata.json").read_bytes())
        # Evaluation-only outputs are never opened by the pilot fitting driver.
        for name, values in (
            ("native-scalar-means", mono),
            ("native-spectral-means", spectral),
            ("reference-scalar", coarse_scalar),
            ("reference-fractions", coarse),
        ):
            library.write_array(run, f"evaluation/{name}.npy", values)
        run.write_json(
            "noise.json",
            {
                "generator": "NumPy PCG64/SeedSequence",
                "scalar_keys": scalar_keys,
                "spectral_keys": spectral_keys,
            },
        )
        public: dict[str, Any] = {
            "schema_version": 1,
            "role": "development fitting observations and admissible support only",
            "geometry": asdict(geometry),
            "inverse_grid": asdict(inverse_grid),
            "inverse_samples": 512,
            "views": public_views,
            "open_beam": "open-beam.npy",
            "open_beam_sha256": digest(output / "open-beam.npy"),
            "physics": {
                name: digest(output / "physics" / name) for name in ("physics.npz", "metadata.json")
            },
            "metric": fixed_metric(physics),
            "config": config,
            "scalar_total_incident_per_pixel": len(views) * config["incident_photons_per_ray"],
            "spectral_total_incident_per_pixel_including_all_channels": len(views)
            * float(physics["incident_weights"].astype(np.float64).sum()),
            "reference_access_permitted_for_fitting": False,
        }
        run.write_json("public-observations.json", public)
        run.write_json(
            "acquisition.json",
            {
                "status": "complete development acquisition",
                "native_gates_passed": native_report["passed"],
                "inverse_gates_passed": inverse_report["passed"],
                "qualification_seconds": qualified_seconds,
                "count_sampling_seconds": sampling_seconds,
                "recording_seconds": time.perf_counter() - tick,
                "counts_generated": True,
                "reconstruction_executed": False,
                "all_view_gate_count": native_report["gate_count"] + inverse_report["gate_count"],
            },
        )
    print(f"Development acquisition complete: {output}", flush=True)


if __name__ == "__main__":
    main()
