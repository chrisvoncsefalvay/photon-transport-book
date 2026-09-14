"""Record isolated-pelvis primary projections and Poisson count frames on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import warp as wp
from PIL import Image

from dpt.detector import ObservationIdentity, prepare_detector, sample_poisson_counts
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth
from dpt.transmission import TransmissionSpec, prepare_transmission, transmit
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = Path(__file__).with_name("config.json")
    config = json.loads(config_path.read_text())
    source_paths = [Path(__file__), config_path, *sorted((ROOT / "python/dpt").rglob("*.py"))]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    record = json.loads((args.prepared / "run.json").read_text())
    if record["status"] != "complete" or not record["sources_unchanged"]:
        raise ValueError("prepared record is incomplete")
    for filename in ["mu.npy", "volume.json"]:
        if digest(args.prepared / filename) != record["output_sha256"][filename]:
            raise ValueError("prepared file hash mismatch")
    if digest(args.labels) != config["label_sha256"]:
        raise ValueError("conditioning mask hash mismatch")
    volume = json.loads((args.prepared / "volume.json").read_text())
    if volume["image_sha256"] != config["image_sha256"]:
        raise ValueError("synthetic source identity changed")
    field = np.load(args.prepared / "mu.npy", allow_pickle=False)
    labels = nib.load(args.labels)
    if not np.array_equal(labels.affine, np.array(volume["native_affine"])):
        raise ValueError("mask and attenuation grids differ")
    mask = np.asanyarray(labels.dataobj).transpose(2, 1, 0)
    if mask.shape != field.shape:
        raise ValueError("mask and attenuation shapes differ")
    isolated = np.where(np.isin(mask, config["labels"]), field, 0).astype(np.float32)
    if not np.isfinite(isolated).all() or isolated.min() < 0:
        raise ValueError("invalid isolated attenuation")
    g = volume["grid"]
    grid = GridSpec(
        tuple(g["shape"]), tuple(g["spacing_mm"]), tuple(g["origin_mm"]), tuple(g["orientation"])
    )
    w, h = config["width"], config["height"]
    du, dv = config["detector_width_mm"] / w, config["detector_height_mm"] / h
    centre = config["detector_centre_mm"]
    geometry = DetectorGeometry(
        tuple(config["source_mm"]),
        (centre[0] - (w - 1) * du / 2, centre[1], centre[2] + (h - 1) * dv / 2),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (du, dv),
        (h, w),
    )
    wp.init()
    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise ValueError("recording requires actual CUDA")
    field_gpu = wp.array(isolated.ravel(), dtype=wp.float32, device=device)
    pose_gpu = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
    depth_gpu = wp.empty(w * h, dtype=wp.float32, device=device)
    means_gpu = wp.empty(w * h, dtype=wp.float32, device=device)
    counts_gpu = wp.empty(w * h, dtype=wp.uint64, device=device)
    transmission = prepare_transmission(
        TransmissionSpec(beam="scalar"), max_pixels=w * h, device=str(device)
    )
    projections = []
    for samples in config["samples_per_ray"]:
        workspace = prepare_projection(
            grid, geometry, ProjectionSpec(samples, active_pose=False), device=str(device)
        )
        project_optical_depth(field_gpu, pose_gpu, out_L=depth_gpu, workspace=workspace)
        transmit(depth_gpu, 1000.0, out_counts=means_gpu, workspace=transmission)
        wp.synchronize_device(device)
        projections.append(means_gpu.numpy().reshape(h, w))
        print("projected", samples, flush=True)
    delta = np.abs(projections[-1].astype(float) - projections[-2])
    convergence = {"count_max": float(delta.max()), "count_p99": float(np.quantile(delta, 0.99))}
    if convergence["count_max"] > 0.25 or convergence["count_p99"] > 0.05:
        raise ValueError(f"projection refinement failed: {convergence}")
    transmit(
        depth_gpu,
        config["open_beam_counts_per_frame"],
        out_counts=means_gpu,
        workspace=transmission,
    )
    wp.synchronize_device(device)
    means = means_gpu.numpy().reshape(h, w)
    reference_checks = []
    for row, column in [(80, 96), (55, 70), (55, 122), (106, 71), (106, 124), (5, 5), (145, 170)]:
        depth = integrate_sampled_field(
            isolated.ravel(), grid, geometry, RigidTransform(), row, column, validate=False
        )
        expected = 1000 * np.exp(-depth)
        error = abs(float(projections[-1][row, column]) - expected)
        reference_checks.append({"row": row, "column": column, "absolute_count_error": error})
        if error > 0.05:
            raise ValueError("independent piecewise-polynomial ray reference failed")
    detector = prepare_detector(max_pixels=w * h, device=str(device))
    frames = []
    for frame in range(config["frame_count"]):
        sample_poisson_counts(
            means_gpu,
            out_counts=counts_gpu,
            identity=ObservationIdentity(seed=config["seed"], observation_id=frame),
            workspace=detector,
        )
        frames.append(counts_gpu.numpy().reshape(h, w).copy())
    counts = np.stack(frames)
    if int(counts.max()) > 255:
        raise ValueError("atlas uint8 capacity exceeded; never clip observations")
    # Re-run an observation to verify identity-preserving replay on this device.
    sample_poisson_counts(
        means_gpu,
        out_counts=counts_gpu,
        identity=ObservationIdentity(seed=config["seed"], observation_id=0),
        workspace=detector,
    )
    if not np.array_equal(counts_gpu.numpy().reshape(h, w), counts[0]):
        raise ValueError("observation identity did not reproduce")
    expected_total = float(means.sum(dtype=np.float64)) * config["frame_count"]
    total = int(counts.sum())
    total_z = (total - expected_total) / np.sqrt(expected_total)
    active = means > 0
    empirical_mean = counts.mean(axis=0)
    empirical_var = counts.var(axis=0, ddof=1)
    fano = float(empirical_var[active].sum() / empirical_mean[active].sum())
    if abs(total_z) > 5 or not 0.95 < fano < 1.05:
        raise ValueError("observation statistics outside declared broad checks")
    cols = config["atlas_columns"]
    rows = config["frame_count"] // cols
    atlas = (
        counts.astype(np.uint8)
        .reshape(rows, cols, h, w)
        .transpose(0, 2, 1, 3)
        .reshape(rows * h, cols * w)
    )
    Image.fromarray(atlas).save(args.output / "impact-atlas.png", optimize=True)
    # Sparse flight endpoints are selected only from actual recorded nonzero cells.
    rng = np.random.default_rng(config["seed"])
    flights = []
    yy, xx = np.mgrid[:h, :w]
    visible = ((xx + 0.5) / w * 2 - 1) ** 2 + ((yy + 0.5) / h * 2 - 1) ** 2 < 0.64**2
    for frame in counts:
        cells = np.flatnonzero((frame > 0) & visible)
        chosen = rng.choice(
            cells,
            size=min(5, len(cells)),
            replace=False,
            p=frame.ravel()[cells] / frame.ravel()[cells].sum(),
        )
        flights.append([int(p) for p in chosen])
    np.save(args.output / "expected-counts.npy", projections[-1], allow_pickle=False)
    np.save(args.output / "count-frames.npy", counts, allow_pickle=False)
    Image.fromarray(np.rint(projections[-1] / 1000 * 255).astype(np.uint8)).save(
        args.output / "expected-counts.png"
    )
    if any(digest(ROOT / path) != value for path, value in source_hashes.items()):
        raise ValueError("source files changed during recording")
    metadata = {
        "schemaVersion": 1,
        "status": "complete",
        "sourcesUnchanged": True,
        "width": w,
        "height": h,
        "frameRate": config["frame_rate"],
        "frameCount": config["frame_count"],
        "atlasColumns": cols,
        "atlas": "impact-atlas.png",
        "tauSeconds": config["tau_seconds"],
        "flashTauSeconds": config["flash_tau_seconds"],
        "openBeamCountsPerFrame": config["open_beam_counts_per_frame"],
        "flights": flights,
        "source": config["source_mm"],
        "detectorCentre": centre,
        "detectorWidth": config["detector_width_mm"],
        "detectorHeight": config["detector_height_mm"],
        "observation": {
            "seed": config["seed"],
            "totalPhotons": total,
            "expectedTotal": expected_total,
            "totalZ": float(total_z),
            "fano": fano,
            "maxCount": int(counts.max()),
            "firstFrameReproduced": True,
        },
        "projectionConvergence": convergence,
        "independentReferenceRays": reference_checks,
        "provenance": {
            "model": config["model"],
            "imageSha256": config["image_sha256"],
            "labelSha256": config["label_sha256"],
            "preparedMuSha256": digest(args.prepared / "mu.npy"),
            "isolatedMuSha256": hashlib.sha256(isolated.tobytes()).hexdigest(),
            "sourceSha256": source_hashes,
            "atlasSha256": digest(args.output / "impact-atlas.png"),
            "recordedFramesSha256": digest(args.output / "count-frames.npy"),
            "expectedCountsSha256": digest(args.output / "expected-counts.npy"),
            "warpVersion": wp.__version__,
            "device": device.name,
            "limits": (
                "Synthetic isolated conditioning-mask phantom; retained water-equivalent primary "
                "attenuation, no scatter or clinical detector calibration. Native cropped anatomy "
                "retained. Central pencil-ray means without detector-area quadrature; "
                "pixel-centre/time-bin-endpoint observations; finite recording loops. "
                "Exposure, spatial spread and persistence are authored display parameters."
            ),
        },
    }
    (args.output / "impact-record.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "convergence": convergence,
                "observations": metadata["observation"],
                "atlasBytes": (args.output / "impact-atlas.png").stat().st_size,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
