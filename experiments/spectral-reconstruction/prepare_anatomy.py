"""Prepare an explicitly assigned water/bone phantom from the archived TCIA CT.

This is offline input preparation, not patient material decomposition. Original
CT and label bytes stay unchanged. Reference fields never initialise the solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

SOURCE_DOI = "https://doi.org/10.7937/tcia.2019.tt7f4v7o"
RIGHTS = "CC BY 3.0; Rister et al., CT-ORG, TCIA"
HASHES = {
    "volume-2.nii.gz": "ca702e58f0fb736ff4bc407aac0c4b698b8689f3f7e0281da32a546a87ae8a40",
    "labels-2.nii.gz": "c5951c1065f517c9e8e794aec0a921ae0ebaa2198d4b38f0ed492bc7aca74e8f",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def block_mean(values: np.ndarray, factors_xyz: tuple[int, int, int]) -> np.ndarray:
    """Conservative average of aligned native voxel cells; return x-fastest ZYX."""
    x, y, z = values.shape
    a, b, c = factors_xyz
    if x % a or y % b or z % c:
        raise ValueError("crop dimensions must be divisible by the averaging factors")
    reduced = values.reshape(x // a, a, y // b, b, z // c, c).mean(axis=(1, 3, 5), dtype=np.float64)
    return np.ascontiguousarray(reduced.transpose(2, 1, 0), dtype=np.float32)


def grid(shape_zyx: tuple[int, ...], extent_xyz: np.ndarray) -> dict:
    spacing = extent_xyz / np.array(shape_zyx[::-1])
    return {
        "shape": list(shape_zyx),
        "spacing_mm": spacing.tolist(),
        "origin_mm": (-extent_xyz / 2 + spacing / 2).tolist(),
        "orientation": np.eye(3).ravel().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new output directory; prepared inputs are immutable")
    for name, expected in HASHES.items():
        if digest(args.source / name) != expected:
            raise ValueError(f"archived TCIA input identity changed: {name}")
    image = nib.load(args.source / "volume-2.nii.gz")
    mask = nib.load(args.source / "labels-2.nii.gz")
    if image.shape != (512, 512, 517) or not np.array_equal(image.affine, mask.affine):
        raise ValueError("unexpected shape or mismatched CT/label affine")
    if nib.aff2axcodes(image.affine) != ("L", "A", "S"):
        raise ValueError("this frozen case expects the verified LAS source header")
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    if not np.allclose(image.affine[:3, :3], np.diag(spacing * [-1, 1, 1])):
        raise ValueError("unexpected oblique source affine")
    # Fixed inferior 320 mm includes the pelvic ring and lumbar spine. Reversing
    # X is a lossless header-led reorientation; it cannot certify source laterality.
    hu = np.asarray(image.dataobj[:, :, :320], dtype=np.float32)[::-1].copy()
    label_values = np.asarray(mask.dataobj[:, :, :320], dtype=np.float32)[::-1]
    if not np.isfinite(hu).all() or not np.isfinite(label_values).all():
        raise ValueError("non-finite source samples")
    rounded = np.rint(label_values)
    if np.max(np.abs(label_values - rounded)) > 1e-6:
        raise ValueError("labels do not recover the documented integer IDs")
    bone = rounded == 5
    del rounded, label_values
    # Segmentation background means unlabelled, not air. The largest connected
    # CT foreground removes the disconnected table; axial hole filling preserves
    # internal low-density anatomy for the explicitly assigned water fraction.
    components, count = ndimage.label(hu > -500)
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    sizes[0] = 0
    body = components == sizes.argmax()
    del components
    for z in range(body.shape[2]):
        body[:, :, z] = ndimage.binary_fill_holes(body[:, :, z])
    bone &= body
    water = np.clip(1 + hu / 1000, 0, 1)
    water *= body
    water[bone] = 0.35
    bone_fraction = bone.astype(np.float32) * np.float32(0.65)
    extent = np.array(hu.shape) * spacing
    source_centre = nib.affines.apply_affine(image.affine, [255.5, 255.5, 159.5])
    forward = np.stack([block_mean(v, (4, 4, 2)) for v in (water, bone_fraction)])
    reference = np.stack([block_mean(v, (8, 8, 4)) for v in (water, bone_fraction)])
    hu_coarse = block_mean(hu, (8, 8, 4))
    if (
        min(forward.min(), reference.min()) < 0
        or max(forward.sum(axis=0).max(), reference.sum(axis=0).max()) > 1 + 2**-24
    ):
        raise ValueError("prepared fractions violate the voxel simplex")
    args.output.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "scientific_role": "CT-derived assigned material phantom; not patient composition truth",
        "source": SOURCE_DOI,
        "rights": RIGHTS,
        "source_files": {
            name: {"path": str(args.source / name), "sha256": sha} for name, sha in HASHES.items()
        },
        "source_affine": image.affine.tolist(),
        "native_crop_xyz": [[0, 512], [0, 512], [0, 320]],
        "reorientation": "X reversed from header LAS to RAS, no resampling",
        "laterality": "header-derived; source laterality not independently established",
        "object_centre_source_ras_mm": source_centre.tolist(),
        "body_rule": "HU > -500; largest 3D 6-connected component; fill holes in each axial plane",
        "material_assignment": {
            "water": "inside body: clip(1 + HU/1000, 0, 1); 0.35 on body-intersected label 5",
            "bone": "0.65 on body-intersected label 5, otherwise zero",
            "vacuum": "1 - water - bone",
            "interpretation": (
                "fixed synthetic fractions at NIST reference densities; no CT calibration"
            ),
            "unknown_high_hu": (
                "non-bone high-HU content capped at water fraction 1; no inferred iodine"
            ),
        },
        "forward_grid": grid(forward.shape[1:], extent),
        "inverse_grid": grid(reference.shape[1:], extent),
        "resampling": (
            "aligned native-cell volume averages; forward factors XYZ 4,4,2; inverse 8,8,4"
        ),
        "boundary": "finite cropped phantom, vacuum outside box; not a complete patient scan",
        "solver_access": (
            "inverse grid and projections only; no CT, labels or reference fields "
            "in initialisation/regulariser"
        ),
        "body_native_voxels": int(body.sum()),
        "bone_native_voxels": int(bone.sum()),
        "native_assigned_material_volume_mm3": [
            float(v.sum(dtype=np.float64) * spacing.prod()) for v in (water, bone_fraction)
        ],
        "outputs": {},
    }
    for name, values, role in [
        ("forward-fractions.npy", forward, "observation generation only"),
        ("evaluation-fractions.npy", reference, "withheld evaluation only"),
        ("evaluation-ct-hu.npy", hu_coarse, "display only"),
    ]:
        path = args.output / name
        np.save(path, values, allow_pickle=False)
        metadata["outputs"][name] = {
            "sha256": digest(path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "role": role,
        }
    (args.output / "anatomy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "forward_shape": list(forward.shape),
                "inverse_shape": list(reference.shape),
                "status": "prepared",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
