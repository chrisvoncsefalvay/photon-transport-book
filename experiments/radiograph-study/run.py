"""Record native-grid MAISI slices, canonical radiographs, yaw and Poisson texture."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import warp as wp
from PIL import Image

from dpt.detector import ObservationIdentity, prepare_detector, sample_poisson_counts
from dpt.experiments import RunRecorder, experiment_parser, experiment_sources, private_output
from dpt.geometry import RigidTransform, compose_pose
from dpt.validation.projection import integrate_sampled_field

ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = ROOT / "experiments/pose-sensitivity/common.py"
spec = importlib.util.spec_from_file_location("radiograph_canonical_scene", SCENE_PATH)
assert spec is not None and spec.loader is not None
scene_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene_module)
Scene = scene_module.Scene


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png(values: np.ndarray, window: list[float], *, inverse: bool = False) -> bytes:
    pixels = np.clip((values.astype(float) - window[0]) / (window[1] - window[0]), 0, 1)
    if inverse:
        pixels = 1 - pixels
    buffer = io.BytesIO()
    Image.fromarray(np.rint(255 * pixels).astype(np.uint8)).save(
        buffer, format="PNG", optimize=True
    )
    return buffer.getvalue()


def matrix(pose: RigidTransform) -> list[float]:
    result = np.eye(4)
    result[:3, :3] = np.asarray(pose.rotation).reshape(3, 3)
    result[:3, 3] = pose.translation_mm
    return result.ravel().tolist()


def prepare(image_path: Path, entry: dict, config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    if digest(image_path) != entry["image_sha256"]:
        raise ValueError(f"source image identity changed: {entry['id']}")
    image = nib.load(image_path)
    hu = image.get_fdata(dtype=np.float32)
    affine = np.asarray(image.affine, dtype=np.float64)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    orientation = affine[:3, :3] / spacing
    if (
        image.shape != (256, 256, 256)
        or not np.isfinite(hu).all()
        or not np.isfinite(affine).all()
        or not np.allclose(orientation, np.eye(3), rtol=0, atol=1e-12)
        or float(hu.min()) != -1000
        or float(hu.max()) != 1000
    ):
        raise ValueError("native archived RAS CT contract failed")
    pivot = nib.affines.apply_affine(affine, (np.asarray(image.shape) - 1) / 2)
    values = np.ascontiguousarray(
        (config["mu_water_mm_inv"] * np.maximum(0, 1 + hu.astype(float) / 1000)).transpose(2, 1, 0),
        dtype=np.float32,
    )
    metadata = {
        **{key: entry[key] for key in ["id", "label", "coverage"]},
        "source_image": entry["image"],
        "source_image_sha256": entry["image_sha256"],
        "native_affine": affine.tolist(),
        "shape_xyz": list(image.shape),
        "spacing_mm": spacing.tolist(),
        "pivot_ras_mm": pivot.tolist(),
        "pivot_definition": "Physical centre of the native voxel-support box",
        "grid": {
            "shape": list(values.shape),
            "spacing_mm": spacing.tolist(),
            "origin_mm": (affine[:3, 3] - pivot).tolist(),
            "orientation": orientation.ravel().tolist(),
        },
        "mu_range_mm_inv": [float(values.min()), float(values.max())],
        "mu_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "resampled": False,
        "attenuation_masked": False,
    }
    return hu, values, metadata


def slices(run: RunRecorder, hu: np.ndarray, volume: dict, config: dict) -> list[dict]:
    nx, ny, nz = hu.shape
    sx, sy, sz = volume["spacing_mm"]
    planes = [
        (
            "axial",
            nz // 2,
            np.flipud(hu[:, :, nz // 2].T),
            nx * sx,
            ny * sy,
            {"left": "L", "right": "R", "top": "A", "bottom": "P"},
        ),
        (
            "coronal",
            ny // 2,
            np.flipud(hu[:, ny // 2, :].T),
            nx * sx,
            nz * sz,
            {"left": "L", "right": "R", "top": "S", "bottom": "I"},
        ),
        (
            "sagittal",
            nx // 2,
            np.flipud(hu[nx // 2, :, :].T),
            ny * sy,
            nz * sz,
            {"left": "P", "right": "A", "top": "S", "bottom": "I"},
        ),
    ]
    rows = []
    for name, index, values, width_mm, height_mm, orientation in planes:
        filename = f"{volume['id']}/{name}.png"
        run.write_bytes(filename, png(values, config["ct_hu_window"]))
        rows.append(
            {
                "id": name,
                "index": index,
                "image": filename,
                "width": values.shape[1],
                "height": values.shape[0],
                "width_mm": width_mm,
                "height_mm": height_mm,
                "orientation": orientation,
            }
        )
    return rows


def project_view(
    scene: Scene, values: np.ndarray, config: dict, angle: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    increment = (0.0, 0.0, 0.0, 0.0, 0.0, math.radians(angle))
    pose = compose_pose(RigidTransform(), increment)
    scene.set_samples(config["sampling_sweep"][0])
    coarse = scene.forward(increment)
    scene.set_samples(config["samples_per_ray"])
    counts = scene.forward(increment)
    depth = scene.depth.numpy().reshape(scene.geometry.shape).copy()
    delta = np.abs(counts.astype(float) - coarse)
    refinement = {
        "coarse_samples": config["sampling_sweep"][0],
        "fine_samples": config["samples_per_ray"],
        "count_max": float(delta.max()),
        "count_p99": float(np.quantile(delta, 0.99)),
    }
    if (
        refinement["count_max"] > config["count_max_budget"]
        or refinement["count_p99"] > config["count_p99_budget"]
    ):
        raise ValueError(f"quadrature refinement failed: {refinement}")
    references = []
    for row, column in config["reference_pixels"]:
        reference_depth = integrate_sampled_field(
            values.ravel(), scene.grid, scene.geometry, pose, row, column, validate=False
        )
        reference_count = config["n0"] * math.exp(-reference_depth)
        error = abs(float(counts[row, column]) - reference_count)
        references.append(
            {
                "row": row,
                "column": column,
                "reference_depth": reference_depth,
                "reference_count": reference_count,
                "recorded_count": float(counts[row, column]),
                "absolute_count_error": error,
            }
        )
        if error > config["reference_count_budget"]:
            raise ValueError(f"independent ray failed: {references[-1]}")
    if not np.isfinite(counts).all() or not np.isfinite(depth).all() or np.min(depth) < 0:
        raise ValueError("invalid canonical projection output")
    source_body = np.asarray(pose.rotation).reshape(3, 3).T @ np.asarray(config["source_mm"])
    return (
        counts,
        depth,
        {
            "yaw_degrees": angle,
            "matrix": matrix(pose),
            "source_body_mm": source_body.tolist(),
            "refinement": refinement,
            "independent_rays": references,
        },
    )


def export_view(
    run: RunRecorder,
    prefix: str,
    label: str,
    counts: np.ndarray,
    depth: np.ndarray,
    row: dict,
    config: dict,
) -> dict:
    run.write_bytes(f"{prefix}.png", png(depth, config["radiograph_depth_window"]))
    run.write_bytes(f"{prefix}.f32", counts.astype("<f4").tobytes())
    run.write_bytes(f"{prefix}-depth.f32", depth.astype("<f4").tobytes())
    return {
        "id": prefix.split("/")[-1],
        "label": label,
        "image": f"{prefix}.png",
        "counts_f32": f"{prefix}.f32",
        "depth_f32": f"{prefix}-depth.f32",
        **row,
    }


def noise_strip(run: RunRecorder, reference: np.ndarray, config: dict, device: str) -> dict:
    settings = config["noise"]
    size, stride, border = settings["roi_size"], settings["roi_stride"], settings["roi_border"]
    height, width = reference.shape
    candidates = []
    for y in range(int(height * 0.2), int(height * 0.8) - size + 1, stride):
        for x in range(int(width * 0.2), int(width * 0.8) - size + 1, stride):
            patch = reference[y : y + size, x : x + size].astype(float)
            mean = float(patch.mean())
            if (
                settings["roi_transmission_range"][0]
                <= mean / config["n0"]
                <= settings["roi_transmission_range"][1]
            ):
                candidates.append((float(patch.std() / mean), y, x))
    if not candidates:
        raise ValueError("no eligible low-contrast projection region")
    cv, y, x = min(candidates)
    if cv > settings["roi_max_cv"]:
        raise ValueError(f"lowest available patch CV {cv} exceeds frozen limit")
    patch = reference[y : y + size, x : x + size].astype(float)
    background = np.ones(patch.shape, dtype=bool)
    background[border:-border, border:-border] = False
    background_mean = float(patch[background].mean())
    normalised = patch / background_mean
    run.write_bytes(
        "noise/ideal.png", png(normalised, settings["normalised_display_window"], inverse=True)
    )
    means_gpu = wp.empty(size * size, dtype=wp.float32, device=device)
    counts_gpu = wp.empty(size * size, dtype=wp.uint64, device=device)
    workspace = prepare_detector(max_pixels=size * size, device=device)
    rows = []
    for index, level in enumerate(settings["background_counts"]):
        means = np.ascontiguousarray(level * normalised, dtype=np.float32)
        wp.copy(means_gpu, wp.array(means.ravel(), dtype=wp.float32, device=device))
        identity = ObservationIdentity(
            seed=settings["seed"], observation_id=settings["first_observation_id"] + index
        )
        sample_poisson_counts(
            means_gpu, out_counts=counts_gpu, identity=identity, workspace=workspace
        )
        counts = counts_gpu.numpy().reshape(size, size).copy()
        sample_poisson_counts(
            means_gpu, out_counts=counts_gpu, identity=identity, workspace=workspace
        )
        if not np.array_equal(counts, counts_gpu.numpy().reshape(size, size)):
            raise ValueError("Poisson identity replay failed")
        expected_total = float(means.sum(dtype=float))
        total_z = (float(counts.sum()) - expected_total) / math.sqrt(expected_total)
        pearson = float(np.sum((counts.astype(float) - means) ** 2 / means))
        pearson_z = (pearson - means.size) / math.sqrt(float(np.sum(2 + 1 / means.astype(float))))
        if max(abs(total_z), abs(pearson_z)) > settings["statistic_sigma_budget"]:
            raise ValueError("Poisson statistics exceed the predeclared budget")
        prefix = f"noise/counts-{level}"
        run.write_bytes(
            f"{prefix}.png",
            png(counts.astype(float) / level, settings["normalised_display_window"], inverse=True),
        )
        run.write_bytes(f"{prefix}-expected.f32", means.astype("<f4").tobytes())
        run.write_bytes(f"{prefix}-observed.u64", counts.astype("<u8").tobytes())
        rows.append(
            {
                "background_counts": level,
                "image": f"{prefix}.png",
                "expected_f32": f"{prefix}-expected.f32",
                "observed_u64": f"{prefix}-observed.u64",
                "seed": settings["seed"],
                "observation_id": identity.observation_id,
                "statistics": {
                    "expected_total": expected_total,
                    "observed_total": int(counts.sum()),
                    "total_z": total_z,
                    "pearson_z": pearson_z,
                    "expected_background_mean": float(means[background].mean(dtype=float)),
                    "observed_background_mean": float(counts[background].mean()),
                    "normalised_rmse": float(
                        np.sqrt(np.mean((counts.astype(float) / level - normalised) ** 2))
                    ),
                },
                "replay_identical": True,
            }
        )
    return {
        "source": "pelvis/ap",
        "width": size,
        "height": size,
        "roi": {
            "x": x,
            "y": y,
            "width": size,
            "height": size,
            "border_pixels": border,
            "coefficient_of_variation": cv,
            "selection": (
                "Lowest CV among eligible fixed-stride windows in the central 60% of the detector"
            ),
        },
        "normalisation": {
            "background_reference_counts": background_mean,
            "reference_open_beam_counts": config["n0"],
            "definition": "Mean of the four-pixel border of the same synthetic pelvic AP region",
        },
        "display_window": settings["normalised_display_window"],
        "display_mapping": "Inverse linear normalised counts; common fixed window",
        "ideal_image": "noise/ideal.png",
        "levels": rows,
    }


def main() -> None:
    parser = experiment_parser(__file__, __doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = private_output(args.output, ROOT)
    sources = experiment_sources(
        __file__,
        args.config,
        extra={
            "experiments/pose-sensitivity/common.py": SCENE_PATH,
            "experiments/radiograph-study/README.md": Path(__file__).with_name("README.md"),
        },
    )
    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise ValueError("actual CUDA is required")
    with RunRecorder(output, configuration=config, sources=sources) as run:
        run.set_metadata(
            device=str(device),
            device_name=device.name,
            warp=wp.__version__,
            nibabel=nib.__version__,
            numpy=np.__version__,
            precision="FP32 field/counts/depth; FP64 geometry and independent scalar references",
            model="Uncalibrated 80 keV water-equivalent primary transmission of synthetic MAISI CT",
            display_only_clipping=True,
        )
        study = {
            "schema_version": 1,
            "id": "maisi-radiograph-study",
            "model": {
                "synthetic": True,
                "mu_formula": "0.01837 mm^-1 * max(0, 1 + HU/1000)",
                "reference_energy_kev": config["energy_kev"],
                "source": "Archived NVIDIA MAISI/NV-Generate-CT outputs",
                "limits": (
                    "Uncalibrated water-equivalent primary transmission; no spectrum, scatter "
                    "or clinical validation. Native cropped fields are preserved."
                ),
            },
            "detector": {
                "width": config["width"],
                "height": config["height"],
                "width_mm": config["detector_width_mm"],
                "height_mm": config["detector_height_mm"],
                "pixel_spacing_mm": [
                    config["detector_width_mm"] / config["width"],
                    config["detector_height_mm"] / config["height"],
                ],
                "source_mm": config["source_mm"],
                "centre_mm": config["detector_centre_mm"],
                "u_axis": [1, 0, 0],
                "v_axis": [0, 0, -1],
                "orientation": (
                    "Acquisition coordinates; columns +x, rows -z; no clinical left-right mirror"
                ),
            },
            "display": {
                "radiograph_depth_window": config["radiograph_depth_window"],
                "radiograph_mapping": "Linear optical depth; higher attenuation is brighter",
                "ct_hu_window": config["ct_hu_window"],
            },
            "volumes": [],
            "animation": {
                "volume": "pelvis",
                "axis": "rz",
                "frames": [],
                "suggested_frames_per_second": 15,
                "playback": "Recorded frames in forward/reverse order; no interpolated projections",
            },
        }
        for entry in config["volumes"]:
            print("preparing", entry["id"], flush=True)
            hu, values, volume = prepare(args.input_root / entry["image"], entry, config)
            volume["slices"] = slices(run, hu, volume, config)
            buffer = io.BytesIO()
            np.save(buffer, values, allow_pickle=False)
            run.write_bytes(f"{entry['id']}/mu.npy", buffer.getvalue())
            scene = Scene(values, volume, config, str(device))
            volume["views"] = []
            ap_counts = None
            ap_result = None
            for name, label, angle in [
                ("ap", "AP", 0),
                ("pa", "PA", 180),
                ("lateral", "Right-to-left lateral", 90),
            ]:
                counts, depth, row = project_view(scene, values, config, angle)
                result = export_view(
                    run, f"{entry['id']}/{name}", label, counts, depth, row, config
                )
                volume["views"].append(result)
                if name == "ap":
                    ap_counts, ap_result = counts, result
                print("projected", entry["id"], name, row["refinement"], flush=True)
            if entry["id"] == "pelvis":
                assert ap_counts is not None and ap_result is not None
                for index, angle in enumerate(config["yaw_degrees"]):
                    if angle == 0:
                        result = {**ap_result, "index": index}
                    else:
                        counts, depth, row = project_view(scene, values, config, angle)
                        result = export_view(
                            run,
                            f"pelvis/yaw-{index:03d}",
                            f"Yaw {angle:+g}°",
                            counts,
                            depth,
                            row,
                            config,
                        )
                        result["index"] = index
                    study["animation"]["frames"].append(result)
                    print("yaw", index + 1, len(config["yaw_degrees"]), angle, flush=True)
                study["noise"] = noise_strip(run, ap_counts, config, str(device))
            study["volumes"].append(volume)
            del scene, values, hu
        if any(
            digest(args.input_root / entry["image"]) != entry["image_sha256"]
            for entry in config["volumes"]
        ):
            raise ValueError("an input image changed during recording")
        study["checks"] = {
            "input_hashes": True,
            "native_geometry": True,
            "finite_nonnegative": True,
            "all_view_refinements": True,
            "independent_reference_rays": True,
            "poisson_statistics": True,
            "poisson_replay": True,
        }
        study["status"] = "passed"
        run.write_json("study.json", study)
        print(
            "accepted",
            len(study["volumes"]),
            "volumes;",
            len(study["animation"]["frames"]),
            "yaw frames;",
            len(study["noise"]["levels"]),
            "Poisson exposures",
            flush=True,
        )


if __name__ == "__main__":
    main()
