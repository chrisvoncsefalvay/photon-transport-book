"""Qualify actual inverse grids and prepare separate development fitting inputs."""

from __future__ import annotations

import argparse
import copy
import io
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _resolution import (
    check_coverage,
    check_source_amendment,
    checked_file,
    checked_json,
    current_sources,
    qualification_coverage,
    regime_counts,
    scaled_physics,
    validate_schedule,
    view_records,
)
from _study import coarse_cells, complete_record, digest
from acquire import check_family
from probe_acquisition import detector, fixed_metric, helpers, np, wp

from dpt.experiments import RunRecorder, experiment_sources, private_output, repository_root
from dpt.geometry import RigidTransform


def source_files(config: Path, freeze: Path, extra: dict[str, Path]) -> dict[str, Path]:
    return experiment_sources(
        __file__,
        config,
        extra={
            "canonical-freeze.json": freeze,
            **{f"reconstruction-study/{p.name}": p for p in Path(__file__).parent.glob("*.py")},
            "helpers/spectral-run.py": Path(helpers().__file__),
            **extra,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "anatomy", "physics", "config", "freeze", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    stage_start = time.perf_counter()
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    root = repository_root(__file__)
    config = json.loads(args.config.read_text())
    validate_schedule(config)
    freeze = json.loads(args.freeze.read_text())
    current_sources(root, freeze)
    old = complete_record(args.acquisition)
    public = checked_json(args.acquisition, "public-observations.json", old)
    amendment = check_source_amendment(
        old["source_sha256"], freeze["files"], config["source_amendments"]
    )
    for name in old["source_sha256"]:
        checked_file(args.acquisition / "sources", name, old["source_sha256"])
    for key in (
        "case",
        "physics_model_id",
        "source_sha256",
        "native_crop_xyz",
        "native_shape_zyx",
        "native_spacing_mm_xyz",
        "detector_shape_hw",
        "detector_pitch_mm",
        "source_isocentre_mm",
        "source_detector_mm",
        "tilts_degrees",
        "scalar_coefficients_80kev_mm_inverse",
        "fraction_regularisation_mm_inverse",
        "scalar_initial_and_mapping_step_mm_inverse_squared",
        "spectral_mapping_step",
        "spectral_initial_step",
        "spectral_maximum_step",
        "relative_mapping_tolerance",
    ):
        if config[key] != public["config"][key]:
            raise ValueError(f"frozen physical/numerical input changed: {key}")
    views = {name: view_records(name, config) for name in ("dense", "sparse", "limited")}
    if views["dense"] != [
        {k: v[k] for k in ("id", "tilt_degrees", "yaw_degrees", "pose")} for v in public["views"]
    ]:
        raise ValueError("dense reused view geometry differs from original observations")
    library = helpers()
    if digest(Path(library.__file__)) != old["source_sha256"]["helpers/spectral-run.py"]:
        raise ValueError("physical forward helper changed since acquisition")
    inputs = {
        "source-acquisition/run.json": args.acquisition / "run.json",
        "source-acquisition/public-observations.json": args.acquisition
        / "public-observations.json",
        "source-acquisition/noise.json": checked_file(
            args.acquisition, "noise.json", old["output_sha256"]
        ),
        "anatomy.json": args.anatomy / "anatomy.json",
        "forward-fractions.npy": args.anatomy / "forward-fractions.npy",
        "physics.npz": args.physics / "physics.npz",
        "physics-metadata.json": args.physics / "metadata.json",
    }
    for name in ("anatomy.json", "forward-fractions.npy", "physics.npz", "physics-metadata.json"):
        if digest(inputs[name]) != old["source_sha256"][f"accepted-probe/sources/inputs/{name}"]:
            raise ValueError(f"generation input differs from original study: {name}")
    anatomy = json.loads(inputs["anatomy.json"].read_text())
    metadata, physics, spec = library.load_physics(args.physics)
    native = library.checked_array(args.anatomy, "forward-fractions.npy", anatomy["outputs"])
    native_grid = library.grid_from(anatomy["forward_grid"])
    geometry = detector(config)
    scalar = np.asarray(
        np.einsum(
            "m,mzyx->zyx", config["scalar_coefficients_80kev_mm_inverse"], native.astype(np.float64)
        ),
        dtype=np.float32,
    )
    grids: dict[int, Any] = {}
    fields: dict[int, Any] = {}
    scalars: dict[int, Any] = {}
    for n in (48, 64):
        fields[n], grids[n] = coarse_cells(native, native_grid, (n, n, n))
        scalars[n], _ = coarse_cells(scalar, native_grid, (n, n, n))
    scaled = scaled_physics(physics, 4.0)
    np.testing.assert_allclose(
        fixed_metric(scaled)["matrix"], public["metric"]["matrix"], rtol=1e-12, atol=1e-12
    )
    output = private_output(args.output, root)
    output.mkdir(parents=True, exist_ok=False)
    sources = source_files(
        args.config,
        args.freeze,
        {
            **inputs,
            **library.physics_preparation_sources(args.physics, metadata),
        },
    )
    qualifications: list[dict[str, Any]] = []
    with RunRecorder(
        output / "qualification",
        configuration={
            "config": config,
            "source_amendment": amendment,
            "source_acquisition_run_sha256": digest(inputs["source-acquisition/run.json"]),
            "native_grid": asdict(native_grid),
            "coverage": {
                name: check_coverage(native_grid, geometry, rows) for name, rows in views.items()
            },
            "stage_start_monotonic_seconds": stage_start,
            "clock_boot_id": boot_id,
        },
        sources=sources,
    ) as run:
        wp.init()
        for n in (48, 64):
            _, _, report = check_family(
                run,
                name=f"dense{n}",
                fields=fields[n],
                scalar=scalars[n],
                grid=grids[n],
                geometry=geometry,
                poses=[
                    RigidTransform(**{k: tuple(x) for k, x in v["pose"].items()})
                    for v in views["dense"]
                ],
                physics=physics,
                spec=spec,
                samples=(512, 1024),
                config=config,
                device=args.device,
            )
            qualifications.append(report)
        for regime in ("sparse", "limited"):
            local = {**config, "incident_photons_per_ray": 800000.0}
            poses = [
                RigidTransform(**{k: tuple(x) for k, x in v["pose"].items()}) for v in views[regime]
            ]
            for name, f, mu, grid, pair in (
                (f"{regime}-native", native, scalar, native_grid, (1024, 2048)),
                (f"{regime}48", fields[48], scalars[48], grids[48], (512, 1024)),
            ):
                _, _, report = check_family(
                    run,
                    name=name,
                    fields=f,
                    scalar=mu,
                    grid=grid,
                    geometry=geometry,
                    poses=poses,
                    physics=scaled,
                    spec=spec,
                    samples=pair,
                    config=local,
                    device=args.device,
                )
                qualifications.append(report)
        qualification_coverage(qualifications)
        for n in (48, 64):
            library.write_array(run, f"evaluation/reference-fractions{n}.npy", fields[n])
            library.write_array(run, f"evaluation/reference-scalar{n}.npy", scalars[n])
        run.write_json(
            "qualification.json",
            {
                "passed": True,
                "gate_count": 5184,
                "families": [q["comparison_arrays"] for q in qualifications],
                "seconds_since_stage_start": time.perf_counter() - stage_start,
                "counts_drawn": False,
                "fits_executed": False,
            },
        )
    # No draw occurs until every prescribed sampling family is complete and immutable.
    qualification = complete_record(output / "qualification")
    old_noise = checked_json(args.acquisition, "noise.json", old)
    old_keys = {tuple(k) for model in ("scalar", "spectral") for k in old_noise[f"{model}_keys"]}
    new_keys: set[tuple[int, ...]] = set()
    bundles: dict[str, Any] = {}
    for group in ("dense48", "dense64", "sparse48", "limited48"):
        regime = group.removesuffix("48").removesuffix("64")
        n = 64 if group == "dense64" else 48
        dense = regime == "dense"
        bundle_sources = source_files(
            args.config,
            args.freeze,
            {
                "qualification/run.json": output / "qualification/run.json",
                "qualification/qualification.json": output / "qualification/qualification.json",
                "original/run.json": args.acquisition / "run.json",
                "original/public-observations.json": args.acquisition / "public-observations.json",
                "original/physics.npz": args.physics / "physics.npz",
                "original/physics-metadata.json": args.physics / "metadata.json",
            },
        )
        counts: Any = None
        keys: list[list[int]] = []
        if not dense:
            mean_name = f"qualification/{regime}-native/spectral-1024.npy"
            means = np.load(
                checked_file(output / "qualification", mean_name, qualification["output_sha256"]),
                allow_pickle=False,
            )
            counts, keys = regime_counts(means, regime, config)
            for key in keys:
                if tuple(key) in old_keys or tuple(key) in new_keys:
                    raise ValueError("a noise role was reused")
                new_keys.add(tuple(key))
        with RunRecorder(
            output / group,
            configuration={
                "group": group,
                "regime": regime,
                "inverse_grid": asdict(grids[n]),
                "config": config,
                "source_amendment": amendment,
                "qualification_run_sha256": digest(output / "qualification/run.json"),
                "original_acquisition_run_sha256": digest(args.acquisition / "run.json"),
                "original_observation_manifest_sha256": digest(
                    args.acquisition / "public-observations.json"
                ),
                "stage_start_monotonic_seconds": stage_start,
                "clock_boot_id": boot_id,
                "known_field_qualification_role_separate_from_fitting": True,
            },
            sources=bundle_sources,
        ) as run:
            rows: list[dict[str, Any]] = []
            for index, view in enumerate(views[regime]):
                row: dict[str, Any] = {**view, "sha256": {}}
                for model in ("scalar", "spectral") if dense else ("spectral",):
                    name = f"observations/{model}-view{index:03d}.npy"
                    if dense:
                        old_name = public["views"][index][f"{model}_counts"]
                        path = checked_file(args.acquisition, old_name, old["output_sha256"])
                        if digest(path) != public["views"][index]["sha256"][model]:
                            raise ValueError("dense count manifest differs from recorded bytes")
                        run.write_bytes(name, path.read_bytes())
                    else:
                        library.write_array(run, name, counts[index])
                    row[f"{model}_counts"] = name
                    row["sha256"][model] = digest(output / group / name)
                rows.append(row)
            library.write_array(
                run,
                "open-beam.npy",
                np.full(geometry.shape, 200000.0 if dense else 800000.0, dtype=np.float32),
            )
            if dense:
                for name in ("physics.npz", "metadata.json"):
                    run.write_bytes(f"physics/{name}", (args.physics / name).read_bytes())
            else:
                payload = io.BytesIO()
                np.savez(payload, **scaled)
                run.write_bytes("physics/physics.npz", payload.getvalue())
                derived = {
                    k: copy.deepcopy(metadata[k])
                    for k in (
                        "schema_version",
                        "model_id",
                        "coefficients_provenance",
                        "spectrum_provenance",
                        "response_provenance",
                        "limitations",
                    )
                }
                derived.update(
                    kind="fourfold integrated-incident-weight acquisition derivative",
                    parent_metadata_sha256=digest(args.physics / "metadata.json"),
                    parent_physics_sha256=digest(args.physics / "physics.npz"),
                    physics_npz_sha256=digest(output / group / "physics/physics.npz"),
                    exposure_scaling={
                        "integrated_weight_multiplier": 4.0,
                        "incident_photons_per_ray": 800000.0,
                        "response_coefficients_unchanged": True,
                        "additional_exposure_or_gain": 1.0,
                    },
                )
                derived["spectrum_provenance"]["description"] += (
                    " This derivative multiplies integrated weights by four exactly once: "
                    "800000 photons per pixel/exposure."
                )
                run.write_json("physics/metadata.json", derived)
            local_config = {
                **config,
                "pilot_inverse_shape_zyx": [n] * 3,
                "incident_photons_per_ray": 200000.0 if dense else 800000.0,
            }
            run.write_json(
                "public-observations.json",
                {
                    "schema_version": 1,
                    "role": "development fitting observations and admissible support only",
                    "group": group,
                    "geometry": asdict(geometry),
                    "inverse_grid": asdict(grids[n]),
                    "inverse_samples": 512,
                    "views": rows,
                    "open_beam": "open-beam.npy",
                    "open_beam_sha256": digest(output / group / "open-beam.npy"),
                    "physics": {
                        name: digest(output / group / "physics" / name)
                        for name in ("physics.npz", "metadata.json")
                    },
                    "metric": public["metric"],
                    "config": local_config,
                    "reference_access_permitted_for_fitting": False,
                    "spectral_total_incident_per_pixel_including_all_channels": len(rows)
                    * float(
                        (physics if dense else scaled)["incident_weights"].astype(np.float64).sum()
                    ),
                    "dense_observations_reused_exactly": dense,
                },
            )
            run.write_json(
                "noise.json",
                {
                    "role": "reused exact original counts"
                    if dense
                    else "new development phase2 independent regime counts",
                    "keys": keys,
                },
            )
        bundles[group] = {
            "run_sha256": digest(output / group / "run.json"),
            "manifest_sha256": digest(output / group / "public-observations.json"),
        }
    if len(new_keys) != 216:
        raise ValueError("new noise role inventory is incomplete")
    current_sources(root, freeze)
    receipt = {
        "status": "complete",
        "qualification_run_sha256": digest(output / "qualification/run.json"),
        "bundles": bundles,
        "all_5184_sampling_gates_passed": True,
        "new_noise_keys": 216,
        "stage_start_monotonic_seconds": stage_start,
        "clock_boot_id": boot_id,
        "preparation_seconds": time.perf_counter() - stage_start,
        "config_sha256": digest(args.config),
        "canonical_freeze_sha256": digest(args.freeze),
        "fitting_executed": False,
    }
    (output / "prepared.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Resolution inputs complete: {output}", flush=True)


if __name__ == "__main__":
    main()
