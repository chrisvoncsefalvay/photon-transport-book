"""Prepare the archived synthetic pelvic CT without resampling its physical grid.

This is offline ingestion, never a renderer. The label volume establishes the
sacral pivot and input consistency; every CT voxel contributes to attenuation.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from dpt.experiments import RunRecorder, experiment_parser, private_output, repository_root

ROOT = repository_root(__file__)


def prepare_arrays(image: nib.Nifti1Image, labels: nib.Nifti1Image, config: dict) -> tuple:
    """Return x-fast attenuation and a lossless native-grid description."""
    if image.shape != (256, 256, 256) or labels.shape != image.shape:
        raise ValueError("selected image/label must share the archived 256 cubed grid")
    if not np.array_equal(image.affine, labels.affine):
        raise ValueError("image and label physical grids differ")
    affine = np.asarray(image.affine, dtype=np.float64)
    if not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError("invalid affine")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    orientation = affine[:3, :3] / spacing
    if not np.allclose(orientation, np.eye(3), rtol=0, atol=1e-12):
        raise ValueError("the selected candidate must retain its native RAS orientation")
    hu = image.get_fdata(dtype=np.float32)
    label = np.asanyarray(labels.dataobj)
    if not np.isfinite(hu).all() or float(hu.min()) != -1000 or float(hu.max()) != 1000:
        raise ValueError("archived synthetic HU range changed")
    if set(np.unique(label).tolist()) != {0, 1, 2, 3, 4}:
        raise ValueError("archived label IDs changed")
    sacrum = np.argwhere(label == 2).mean(axis=0)
    pivot = affine[:3, :3] @ sacrum + affine[:3, 3]
    if not np.allclose(pivot, config["pivot_ras_mm"], rtol=0, atol=1e-10):
        raise ValueError("sacral pivot does not match the planned physical point")
    mu_xyz = config["mu_water_mm_inv"] * np.maximum(0, 1 + hu.astype(np.float64) / 1000)
    mu = np.ascontiguousarray(mu_xyz.transpose(2, 1, 0), dtype=np.float32)
    assert mu.flags.c_contiguous and np.isfinite(mu).all() and float(mu.min()) >= 0
    # Check asymmetric indices so an axis permutation cannot hide behind a cube.
    for xyz in [(32, 89, 121), (193, 133, 41), (97, 202, 153)]:
        if mu[xyz[2], xyz[1], xyz[0]] != np.float32(mu_xyz[xyz]):
            raise ValueError("XYZ to x-fast ZYX mapping failed")
    metadata = {
        "schema_version": 1,
        "synthetic": True,
        "image_sha256": config["image_sha256"],
        "label_sha256": config["label_sha256"],
        "native_affine": affine.tolist(),
        "grid": {
            "shape": list(mu.shape),
            "spacing_mm": spacing.tolist(),
            "origin_mm": (affine[:3, 3] - pivot).tolist(),
            "orientation": orientation.ravel().tolist(),
        },
        "pivot_ras_mm": pivot.tolist(),
        "hu_range": [float(hu.min()), float(hu.max())],
        "mu_range_mm_inv": [float(mu.min()), float(mu.max())],
        "mu_formula": "0.01837 mm^-1 * max(0, 1 + HU/1000)",
        "attenuation_reference": (
            "https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html"
        ),
        "approximation": (
            "80 keV water-equivalent monoenergetic primary transmission; uncalibrated "
            "synthetic CT, not material-specific spectral attenuation"
        ),
        "coverage": (
            "Central/superior pelvis; left hip reaches the inferior CT face. Detector "
            "excludes the inferior 20 mm slab at the checked poses."
        ),
        "resampled": False,
        "attenuation_masked": False,
        "checks": {
            "finite": True,
            "paired_grid": True,
            "native_RAS": True,
            "pivot": True,
            "axis_permutation": True,
        },
    }
    return mu, metadata


def main() -> None:
    parser = experiment_parser(__file__, __doc__, cuda=False)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for path, key in [(args.image, "image_sha256"), (args.labels, "label_sha256")]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != config[key]:
            raise ValueError(f"archived {key} mismatch")
    mu, metadata = prepare_arrays(nib.load(args.image), nib.load(args.labels), config)
    sources = {
        "experiments/pose-sensitivity/prepare.py": Path(__file__),
        "experiments/pose-sensitivity/config.json": args.config,
        "experiments/pose-sensitivity/requirements-volume.txt": Path(__file__).with_name(
            "requirements-volume.txt"
        ),
        "python/dpt/experiments.py": ROOT / "python/dpt/experiments.py",
    }
    with RunRecorder(
        private_output(args.output, ROOT),
        configuration=config,
        sources=sources,
        metadata={"nibabel": nib.__version__, "numpy": np.__version__},
    ) as record:
        data = io.BytesIO()
        np.save(data, mu, allow_pickle=False)
        record.write_bytes("mu.npy", data.getvalue())
        record.write_json("volume.json", metadata)
        print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
