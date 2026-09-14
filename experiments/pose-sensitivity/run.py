"""Record and validate the actual pelvic projection and its six pose sensitivities.

Only canonical dpt operators execute on CUDA. CPU work prepares inputs, evaluates
independent reference rays and renders recorded numbers to unfiltered images.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
from common import AXES, Scene
from PIL import Image, ImageDraw, ImageFont

from dpt.experiments import (
    RunRecorder,
    experiment_parser,
    experiment_sources,
    private_output,
    repository_root,
)
from dpt.geometry import RigidTransform, compose_pose
from dpt.projection import projection_vjp
from dpt.validation.projection import integrate_sampled_field

ROOT = repository_root(__file__)


def _read_record(directory: Path, name: str) -> bytes:
    record = json.loads((directory / "run.json").read_text())
    if (
        record["status"] != "complete"
        or not record["sources_unchanged"]
        or not record["recorded_files_unchanged"]
    ):
        raise ValueError("input preparation record is incomplete")
    value = (directory / name).read_bytes()
    if hashlib.sha256(value).hexdigest() != record["output_sha256"][name]:
        raise ValueError(f"input preparation hash mismatch: {name}")
    return value


def _png(array: np.ndarray, limit: float | None = None) -> bytes:
    if limit is None:
        pixels = np.rint(np.log1p(np.clip(array, 0, 1000)) / np.log1p(1000) * 255).astype(np.uint8)
    else:
        t = (np.arcsinh(array / (limit * 0.02)) / np.arcsinh(50))[..., None]
        negative = np.array([0x35, 0x5F, 0x78], dtype=float)
        zero = np.array([0xF7, 0xF4, 0xEE], dtype=float)
        positive = np.array([0xA9, 0x57, 0x32], dtype=float)
        pixels = np.rint(zero + abs(t) * np.where(t < 0, negative - zero, positive - zero))
        pixels = pixels.astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _matrix(increment: tuple[float, ...]) -> list[float]:
    pose = compose_pose(RigidTransform(), increment)
    rotation = np.asarray(pose.rotation).reshape(3, 3)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pose.translation_mm
    return matrix.ravel().tolist()


def _relative_l2(delta: np.ndarray, reference: np.ndarray) -> list[float]:
    return (
        np.linalg.norm(delta.reshape(-1, 6), axis=0)
        / np.maximum(np.linalg.norm(reference.reshape(-1, 6), axis=0), 1e-30)
    ).tolist()


def _display_sweep(scene: Scene, config: dict) -> tuple[list, list]:
    """Check image refinement at every retained physical pose, then keep fine counts."""
    records = [("reference", 0.0, (0.0,) * 6)]
    for axis, name in enumerate(AXES):
        for step in config["display_translation_mm" if axis < 3 else "display_rotation_rad"]:
            if step != 0:
                increment = [0.0] * 6
                increment[axis] = step
                records.append((name, step, tuple(increment)))
    coarse_samples = config["sampling_sweep"][-2]
    fine_samples = config["samples_per_ray"]
    scene.set_samples(coarse_samples)
    coarse = [scene.forward(increment) for _, _, increment in records]
    scene.set_samples(fine_samples)
    fine, checks = [], []
    for (name, step, increment), previous in zip(records, coarse, strict=True):
        counts = scene.forward(increment)
        delta = abs(counts.astype(float) - previous)
        checks.append(
            {
                "parameter": name,
                "value": step,
                "coarse_samples": coarse_samples,
                "fine_samples": fine_samples,
                "count_max": float(delta.max()),
                "count_p99": float(np.quantile(delta, 0.99)),
            }
        )
        fine.append(counts)
        print("display refinement", name, step, json.dumps(checks[-1]), flush=True)
    return fine, checks


def main() -> None:
    parser = experiment_parser(__file__, __doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--meshed", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    volume_bytes = _read_record(args.prepared, "volume.json")
    volume = json.loads(volume_bytes)
    values = np.load(io.BytesIO(_read_record(args.prepared, "mu.npy")), allow_pickle=False)
    meshes_bytes = _read_record(args.meshed, "meshes.json")
    meshes = json.loads(meshes_bytes)
    if (
        volume["image_sha256"] != config["image_sha256"]
        or volume["label_sha256"] != config["label_sha256"]
        or meshes["attribution"]["label_sha256"] != config["label_sha256"]
        or values.dtype != np.float32
        or not np.isfinite(values).all()
        or values.min() < 0
        or not values.flags.c_contiguous
    ):
        raise ValueError("prepared input contract changed")
    extra = {
        f"experiments/pose-sensitivity/{name}": Path(__file__).with_name(name)
        for name in ["config.json", "common.py", "prepare.py", "mesh.py", "requirements-volume.txt"]
    }
    sources = experiment_sources(__file__, args.config, extra=extra)
    budgets = config["acceptance"]
    with RunRecorder(
        private_output(args.output, ROOT), configuration=config, sources=sources
    ) as run:
        scene = Scene(values, volume, config, args.device)
        run.set_metadata(
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            device=str(scene.wp_device),
            device_name=scene.wp_device.name,
            warp=wp.__version__,
            cuda_driver=scene.wp_device.runtime.driver_version,
            precision="FP32 field/counts/seeds; FP64 geometry/quadrature/pose derivatives",
            input_image_sha256=config["image_sha256"],
            input_label_sha256=config["label_sha256"],
            mu_sha256=hashlib.sha256(_read_record(args.prepared, "mu.npy")).hexdigest(),
            derivative="local right SE3 at fixed sacral pivot; tx ty tz rx ry rz; mm/rad",
            field_model=volume["approximation"],
            mesh_attribution=meshes["attribution"],
            private_input_records={"preparation": str(args.prepared), "meshes": str(args.meshed)},
        )
        run.write_json("volume.json", volume)
        previous = None
        refinements = []
        for samples in config["sampling_sweep"]:
            scene.set_samples(samples)
            counts = scene.forward()
            jacobian = scene.derivatives()
            if previous is not None:
                delta = abs(counts.astype(float) - previous[0])
                derivative_delta = jacobian - previous[1]
                refinements.append(
                    {
                        "samples": samples,
                        "count_max": float(delta.max()),
                        "count_p99": float(np.quantile(delta, 0.99)),
                        "jacobian_relative_l2": _relative_l2(derivative_delta, jacobian),
                        "jacobian_normalised_max": (
                            np.max(abs(derivative_delta), axis=(0, 1))
                            / np.max(abs(jacobian), axis=(0, 1))
                        ).tolist(),
                    }
                )
                print("sampling", json.dumps(refinements[-1]), flush=True)
            previous = (counts, jacobian)
        if config["sampling_sweep"][-1] != config["samples_per_ray"]:
            raise ValueError("final quadrature must be the last refinement")
        baseline, jacobian = previous
        run.write_bytes("baseline.f32", baseline.astype("<f4").tobytes())
        run.write_bytes("jacobian.f64", jacobian.astype("<f8").tobytes())
        reference_seeds = scene.seeds.numpy().copy()
        best_pixel_error = np.full_like(jacobian, np.inf)
        fd_rows = []
        for axis, name in enumerate(AXES):
            steps = config["translation_steps_mm" if axis < 3 else "rotation_steps_rad"]
            rows = []
            for step in steps:
                increment = np.zeros(6)
                increment[axis] = step
                plus = scene.forward(tuple(increment)).astype(float)
                minus = scene.forward(tuple(-increment)).astype(float)
                fd = (plus - minus) / (2 * step)
                error = abs(fd - jacobian[..., axis])
                best_pixel_error[..., axis] = np.minimum(best_pixel_error[..., axis], error)
                relative = float(np.linalg.norm(error) / np.linalg.norm(jacobian[..., axis]))
                rows.append(
                    {"step": step, "relative_l2": relative, "max_absolute": float(error.max())}
                )
            fd_rows.append(
                {
                    "axis": name,
                    "steps": rows,
                    "best_relative_l2": min(row["relative_l2"] for row in rows),
                }
            )
            print("finite differences", name, fd_rows[-1]["best_relative_l2"], flush=True)
        # A directional Taylor remainder supplements each coordinate's map check.
        mixed = np.array([1, -1, 1, 0.01, -0.01, 0.01])
        tangent = np.einsum("hwk,k->hw", jacobian, mixed)
        taylor = []
        for step in config["translation_steps_mm"]:
            for sign in [-1, 1]:
                delta = sign * step * mixed
                actual = scene.forward(tuple(delta)).astype(float)
                remainder = np.linalg.norm(actual - baseline - sign * step * tangent)
                taylor.append(
                    {
                        "step": sign * step,
                        "remainder_l2": float(remainder),
                        "remainder_per_step": float(remainder / step),
                    }
                )
        # Return to exactly the reference pose before checking arbitrary contractions.
        scene.forward()
        weights = np.sin(np.arange(scene.geometry.pixels, dtype=float) * 0.017).astype(np.float32)
        seed = wp.array(reference_seeds * weights, dtype=wp.float32, device=args.device)
        gradient = wp.empty(6, dtype=wp.float64, device=args.device)
        projection_vjp(
            scene.field, scene.pose, adj_L=seed, out_pose=gradient, workspace=scene.workspace
        )
        measured = gradient.numpy()
        # Multiplication into the FP32 depth seed rounds; reproduce just that seed ratio.
        effective_weights = np.divide(
            (reference_seeds * weights).astype(float),
            reference_seeds,
            out=np.zeros_like(weights, dtype=float),
            where=reference_seeds != 0,
        )
        contracted = np.sum(jacobian.reshape(-1, 6) * effective_weights[:, None], axis=0)
        contraction_relative = np.abs(measured - contracted) / np.maximum(1, np.abs(measured))
        # Fixed rays plus each coordinate's strongest pixel; every selection is retained.
        rays = {(0, 0), (255, 479), (128, 240), (40, 100), (40, 380), (180, 100), (180, 380)}
        for axis in range(6):
            rays.add(
                tuple(
                    int(v)
                    for v in np.unravel_index(
                        np.argmax(abs(jacobian[..., axis])), scene.geometry.shape
                    )
                )
            )
        stored_values = values.ravel().tolist()  # immutable oracle input, already validated once
        references = []
        for row, column in sorted(rays):

            def count_reference(
                increment: tuple[float, ...],
                samples: int | None,
                row: int = row,
                column: int = column,
            ) -> float:
                depth = integrate_sampled_field(
                    stored_values,
                    scene.grid,
                    scene.geometry,
                    compose_pose(RigidTransform(), increment),
                    row,
                    column,
                    midpoint_samples=samples,
                    validate=False,
                )
                return config["n0"] * math.exp(-depth)

            zero = (0.0,) * 6
            exact = count_reference(zero, None)
            discrete = count_reference(zero, config["samples_per_ray"])
            derivatives = []
            for axis in range(6):
                errors = []
                for h in [1e-4, 1e-5, 1e-6] if axis < 3 else [1e-6, 1e-7, 1e-8]:
                    increment = np.zeros(6)
                    increment[axis] = h
                    plus = count_reference(tuple(increment), config["samples_per_ray"])
                    minus = count_reference(tuple(-increment), config["samples_per_ray"])
                    fd = (plus - minus) / (2 * h)
                    expected = float(jacobian[row, column, axis])
                    scale = max(abs(expected), np.max(abs(jacobian[..., axis])) * 1e-6)
                    errors.append(
                        {
                            "h": h,
                            "finite_difference": fd,
                            "normalised_error": abs(fd - expected) / scale,
                        }
                    )
                derivatives.append(
                    {
                        "axis": AXES[axis],
                        "value": float(jacobian[row, column, axis]),
                        "steps": errors,
                        "best_normalised_error": min(e["normalised_error"] for e in errors),
                    }
                )
            references.append(
                {
                    "row": row,
                    "column": column,
                    "piecewise_gauss_counts": exact,
                    "discrete_fp64_counts": discrete,
                    "cuda_counts": float(baseline[row, column]),
                    "count_discrete_error": abs(discrete - float(baseline[row, column])),
                    "count_quadrature_error": abs(exact - float(baseline[row, column])),
                    "derivatives": derivatives,
                }
            )
            print("independent reference", row, column, flush=True)
        display_counts, display_quadrature = _display_sweep(scene, config)
        if not np.array_equal(display_counts[0], baseline):
            raise ValueError("display sweep reference differs from the validated baseline")
        finite = bool(
            np.isfinite(baseline).all()
            and np.isfinite(jacobian).all()
            and all(np.isfinite(counts).all() for counts in display_counts)
        )
        checks = {
            "input_hashes": True,
            "finite": finite,
            "quadrature_counts": all(
                r["count_max"] <= budgets["count_max"] and r["count_p99"] <= budgets["count_p99"]
                for r in refinements[-2:]
            ),
            "quadrature_derivatives": all(
                max(r["jacobian_relative_l2"]) <= budgets["jacobian_relative_l2"]
                and max(r["jacobian_normalised_max"]) <= budgets["jacobian_normalised_max"]
                for r in refinements[-2:]
            ),
            "discrete_fd": all(
                r["best_relative_l2"] <= budgets["discrete_fd_relative_l2"] for r in fd_rows
            ),
            "independent_reference": all(
                r["count_discrete_error"] <= budgets["reference_count_absolute"]
                and r["count_quadrature_error"] <= budgets["count_max"]
                and max(d["best_normalised_error"] for d in r["derivatives"])
                <= budgets["reference_gradient_relative"]
                for r in references
            ),
            "contractions": bool(max(contraction_relative) <= budgets["contraction_relative"]),
            "display_quadrature": all(
                r["count_max"] <= budgets["count_max"] and r["count_p99"] <= budgets["count_p99"]
                for r in display_quadrature
            ),
        }
        # An unresolved FD pixel is diagnostic, not evidence of an invalid branch derivative.
        rms = np.sqrt(np.mean(jacobian**2, axis=(0, 1)))
        unresolved = best_pixel_error > (0.05 * abs(jacobian) + 0.002 * rms)
        validation = {
            "schema_version": 1,
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "budgets": budgets,
            "samples_per_ray": config["samples_per_ray"],
            "quadrature_refinements": refinements,
            "display_quadrature": display_quadrature,
            "finite_differences": fd_rows,
            "independent_rays": references,
            "contraction_relative_error": contraction_relative.tolist(),
            "mixed_taylor": taylor,
            "fd_unresolved_pixels_per_axis": unresolved.sum(axis=(0, 1)).tolist(),
            "fd_pixel_tolerance": {"relative": 0.05, "axis_rms_fraction": 0.002},
            "limitations": (
                "Local derivatives of the fixed discrete quadrature. Trilinear knots and "
                "active-boundary ties do not assert global differentiability. Clinical validity "
                "and material-specific attenuation are not established."
            ),
        }
        run.write_json("validation.json", validation)
        run.write_bytes("fd-unresolved.u8", unresolved.astype(np.uint8).tobytes())
        print("validation", checks, flush=True)
        if not all(checks.values()):
            raise ValueError("pelvic figure numerical acceptance failed; preserve this record")
        _export(
            run, scene, baseline, jacobian, meshes, validation, config, unresolved, display_counts
        )


def _export(run, scene, baseline, jacobian, meshes, validation, config, unresolved, display_counts):
    h, w = scene.geometry.shape
    run.write_bytes("figure/fd-status.u8", unresolved.astype(np.uint8).tobytes())
    limits = [float(np.max(abs(jacobian[..., :3]))), float(np.max(abs(jacobian[..., 3:])))]
    channels = np.concatenate([baseline[None], jacobian.transpose(2, 0, 1)])
    run.write_bytes("figure/reference.f32", channels.astype("<f4").tobytes())
    run.write_json("figure/meshes.json", meshes)
    run.write_json("figure/validation.json", validation)
    projection_png = _png(baseline)
    run.write_bytes("figure/projection.png", projection_png)
    run.write_bytes("figure/projection.f32", baseline.astype("<f4").tobytes())
    sensitivities = []
    images = [Image.open(io.BytesIO(projection_png)).convert("RGB")]
    for axis, name in enumerate(AXES):
        png = _png(jacobian[..., axis], limits[axis // 3])
        run.write_bytes(f"figure/{name}.png", png)
        images.append(Image.open(io.BytesIO(png)))
        sensitivities.append(
            {
                "id": name,
                "label": name,
                "unit": "counts/mm" if axis < 3 else "counts/rad",
                "image": f"{name}.png",
            }
        )
    poses = [
        {
            "id": "reference",
            "label": "Reference pose",
            "parameter": "reference",
            "value": 0,
            "matrix": _matrix((0.0,) * 6),
            "projection": "projection.png",
            "projection_f32": "projection.f32",
        }
    ]
    # New filenames prevent an old open page from pairing a stale matrix with a
    # newly overwritten per-index image after the configured range changes.
    version = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    recorded_counts = iter(display_counts[1:])
    for axis, name in enumerate(AXES):
        steps = config["display_translation_mm" if axis < 3 else "display_rotation_rad"]
        for index, step in enumerate(steps):
            if step == 0:
                continue
            increment = np.zeros(6)
            increment[axis] = step
            counts = next(recorded_counts)
            identifier = f"{name}-{index}-{version}"
            run.write_bytes(f"figure/{identifier}.png", _png(counts))
            run.write_bytes(f"figure/{identifier}.f32", counts.astype("<f4").tobytes())
            poses.append(
                {
                    "id": identifier,
                    "label": f"{name} {step:+g} {'mm' if axis < 3 else 'rad'}",
                    "parameter": name,
                    "value": step,
                    "matrix": _matrix(tuple(increment)),
                    "projection": f"{identifier}.png",
                    "projection_f32": f"{identifier}.f32",
                }
            )
    data = {
        "schema_version": 1,
        "id": "introduction-pose-sensitivity",
        "title": "One projection, six ways to move the pelvis",
        "anatomy_label": "Synthetic pelvic CT",
        "signal_label": "Expected primary counts",
        "signal_unit": "counts",
        "width": w,
        "height": h,
        "detector": {
            "source_mm": config["source_mm"],
            "centre_mm": config["detector_centre_mm"],
            "u_axis": [1, 0, 0],
            "v_axis": [0, 0, -1],
            "pixel_spacing_mm": list(scene.geometry.spacing_mm),
        },
        "pivot_ras_mm": config["pivot_ras_mm"],
        "reference": {
            "projection": "projection.png",
            "channels_f32": "reference.f32",
            "fd_status_u8": "fd-status.u8",
            "display": {
                "projection": "log1p",
                "sensitivities": "asinh",
                "asinh_softening_fraction": 0.02,
                "projection_max": 1000,
            },
            "translation_limit": limits[0],
            "rotation_limit": limits[1],
        },
        "sensitivities": sensitivities,
        "poses": poses,
        "provenance": {
            "label": "Recorded CUDA calculation and validation",
            "url": "artifact.manifest.json",
        },
    }
    run.write_json("figure/data.json", data)
    # Full-resolution static sheet, laid out without smoothing or medical display inversion.
    panel = Image.new("RGB", (3 * w + 64, 3 * (h + 38) + 58), "#f7f4ee")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=15)
    titles = ["Expected primary counts (log display, 0 to 1000)"] + [
        f"dF/d{name} ({'counts/mm' if i < 3 else 'counts/rad'})" for i, name in enumerate(AXES)
    ]
    positions = [(w + 32, 28)] + [
        (16 + (i % 3) * (w + 16), 28 + (1 + i // 3) * (h + 38)) for i in range(6)
    ]
    for image, label, (x, y) in zip(images, titles, positions, strict=True):
        draw.text((x, y), label, fill="#24343b", font=font)
        panel.paste(image, (x, y + 20))
    draw.text(
        (16, panel.height - 22),
        f"Translations: +/-{limits[0]:.3g} counts/mm; rotations: +/-{limits[1]:.3g} counts/rad. "
        "Synthetic CT; reference-pose derivatives; asinh colour scales.",
        font=font,
        fill="#24343b",
    )
    buffer = io.BytesIO()
    panel.save(buffer, format="PNG", optimize=True)
    run.write_bytes("figure/panel.png", buffer.getvalue())
    print("wrote", len(poses), "recorded poses and six sensitivity maps", flush=True)


if __name__ == "__main__":
    main()
