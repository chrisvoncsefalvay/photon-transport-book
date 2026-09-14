"""Prepare a real CT-ORG bone-mask crop inside an explicitly assigned water box.

This controlled material phantom is not patient tissue decomposition. Reference
fractions and the separately retained CT crop are forbidden solver inputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def helpers():
    """Reuse the existing preparation boundary without duplicating its numerics."""
    path = Path(__file__).with_name("prepare_anatomy.py")
    spec = importlib.util.spec_from_file_location("dpt_anatomy_preparation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical anatomy preparation helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("medical derivatives must remain outside the checkout")
    if output.exists() and any(
        p.name not in {"source-license-manifest.json", "preflight"} for p in output.iterdir()
    ):
        raise FileExistsError("refusing to overwrite an existing prepared input")
    helper = helpers()
    for name, expected in helper.HASHES.items():
        if helper.digest(source / name) != expected:
            raise ValueError(f"archived input identity changed: {name}")
    output.mkdir(parents=True, exist_ok=True)
    intent = output / "source-license-manifest.json"
    if not intent.exists():
        intent.write_text(
            json.dumps(
                {
                    "created_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "source": helper.SOURCE_DOI,
                    "rights": helper.RIGHTS,
                    "source_files": helper.HASHES,
                    "derivative_intent": (
                        "Native bone-mask crop in assigned water box; no publication"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
    ct, labels = (nib.load(source / name) for name in ("volume-2.nii.gz", "labels-2.nii.gz"))
    if ct.shape != (512, 512, 517) or labels.shape != ct.shape:
        raise ValueError("unexpected archived source shape")
    if not np.array_equal(ct.affine, labels.affine):
        raise ValueError("CT and mask geometry differ")
    spacing = np.linalg.norm(ct.affine[:3, :3], axis=0)
    if nib.aff2axcodes(ct.affine) != ("L", "A", "S") or not np.array_equal(
        ct.affine[:3, :3], np.diag(spacing * [-1, 1, 1])
    ):
        raise ValueError("this frozen source requires its verified axis-aligned LAS header")
    # Private inspection of the proposed z208:400 crop showed inferior iliac
    # bone. Move superiorly; preserve native cells and the requested 192^3 size.
    starts = np.array([160, 48, 256])
    stops = starts + 192
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True))
    hu = np.asarray(ct.dataobj[slices], dtype=np.float32)[::-1].copy()
    raw_labels = np.asarray(labels.dataobj[slices], dtype=np.float32)[::-1].copy()
    if not np.isfinite(hu).all() or not np.isfinite(raw_labels).all():
        raise ValueError("non-finite source samples")
    rounded = np.rint(raw_labels)
    if np.max(np.abs(raw_labels - rounded)) > 1e-6:
        raise ValueError("source label values do not recover integer IDs")
    bone_mask = rounded == 5
    bone = bone_mask.astype(np.float32) * np.float32(0.65)
    water = np.float32(1) - bone
    extent = np.asarray(hu.shape) * spacing
    forward = np.stack([helper.block_mean(v, (1, 1, 1)) for v in (water, bone)])
    reference = np.stack([helper.block_mean(v, (3, 3, 3)) for v in (water, bone)])
    ct_coarse = helper.block_mean(hu, (3, 3, 3))
    ct_native = np.ascontiguousarray(hu.transpose(2, 1, 0))
    forward_grid = helper.grid(forward.shape[1:], extent)
    inverse_grid = helper.grid(reference.shape[1:], extent)
    centre_index = (starts + stops - 1) / 2
    centre = nib.affines.apply_affine(ct.affine, centre_index)
    first_source_index = np.array([stops[0] - 1, starts[1], starts[2]])
    first_source_point = nib.affines.apply_affine(ct.affine, first_source_index)
    np.testing.assert_allclose(first_source_point - centre, forward_grid["origin_mm"], atol=1e-10)
    faces = []
    volumes = []
    for values, description in ((forward, forward_grid), (reference, inverse_grid)):
        total = values.sum(axis=0, dtype=np.float64)
        if values.min() < 0 or total.max() > 1 + 2**-24:
            raise ValueError("material fractions violate the represented simplex")
        np.testing.assert_allclose(total, 1, atol=2**-24, rtol=0)
        h = np.asarray(description["spacing_mm"])
        low = np.asarray(description["origin_mm"]) - h / 2
        high = low + h * np.asarray(description["shape"][::-1])
        np.testing.assert_allclose(low, -extent / 2, atol=1e-12)
        np.testing.assert_allclose(high, extent / 2, atol=1e-12)
        faces.append([low.tolist(), high.tolist()])
        volumes.append(values.sum(axis=(1, 2, 3), dtype=np.float64) * h.prod())
    np.testing.assert_allclose(volumes[0], volumes[1], rtol=1e-7, atol=1e-6)
    # Exact source label selection, axis reversal and voxel conservation.
    np.testing.assert_array_equal(forward[1].transpose(2, 1, 0), bone)
    counts_by_face = {
        axis: [int(np.take(bone_mask, side, axis=i).sum()) for side in (0, -1)]
        for i, axis in enumerate(("x", "y", "z"))
    }
    if any(counts_by_face["y"]):
        raise ValueError("anterior/posterior crop unexpectedly intersects bone")
    components, number = ndimage.label(bone_mask)
    sizes = np.bincount(components.ravel(), minlength=number + 1)[1:]
    native_centroid = np.mean(np.argwhere(bone_mask), axis=0)
    display_xyz = np.clip(np.floor(native_centroid / 3).astype(int), 0, 63)
    metadata = {
        "schema_version": 1,
        "title": "CT-derived vertebral-region bone in water",
        "scientific_role": (
            "Controlled CT-mask-derived water-box phantom; not patient composition truth"
        ),
        "source": helper.SOURCE_DOI,
        "rights": helper.RIGHTS,
        "source_files": {
            name: {"path": str(source / name), "sha256": sha} for name, sha in helper.HASHES.items()
        },
        "source_affine": ct.affine.tolist(),
        "native_crop_xyz": [[int(a), int(b)] for a, b in zip(starts, stops, strict=True)],
        "crop_selection": (
            "Private native-mask inspection moved proposed z208:400 to z256:448 "
            "to remove inferior iliac bone. X160:352 and Y48:240 unchanged."
        ),
        "reorientation": "Lossless X reversal from source LAS header to RAS; no native resampling",
        "laterality": "header-derived; source laterality not independently established",
        "object_centre_source_ras_mm": centre.tolist(),
        "body_rule": "Entire known rectangular box is filled; no HU threshold or patient body mask",
        "material_assignment": {
            "water": "1 - assigned bone fraction everywhere inside known box",
            "bone": "0.65 on archived CT-ORG label5, otherwise zero; no mask smoothing or editing",
            "vacuum": "Zero inside assigned reference box; vacuum outside finite box",
            "interpretation": (
                "Declared controlled phantom at fixed reference densities; "
                "not CT-calibrated patient tissue"
            ),
        },
        "roi": {
            "anatomy": (
                "Vertebral region with adjacent cropped rib segments; label5 identifies all bone"
            ),
            "isolated_vertebrae": False,
            "bone_face_intersections_xyz": counts_by_face,
            "boundary_interpretation": (
                "Lateral ribs and superior/inferior bone may be cut at the finite phantom faces"
            ),
            "components_6_connected": int(number),
            "largest_component_native_voxels": int(sizes.max()),
            "bone_centroid_reoriented_native_xyz": native_centroid.tolist(),
        },
        "forward_grid": forward_grid,
        "inverse_grid": inverse_grid,
        "resampling": (
            "Forward native192^3 cells; inverse64^3 aligned3x3x3 native-cell volume averages"
        ),
        "boundary": (
            "Known finite water-filled rectangular phantom; vacuum outside; "
            "not a complete patient scan"
        ),
        "solver_access": (
            "Grid and fitting projections only; both fractions remain free; "
            "no reference fields, labels, CT or bone-derived initialisation"
        ),
        "display_indices_zyx": display_xyz[::-1].tolist(),
        "display_indices": {
            "axial": int(display_xyz[2]),
            "coronal": int(display_xyz[1]),
            "sagittal": int(display_xyz[0]),
        },
        "body_native_voxels": int(bone_mask.size),
        "bone_native_voxels": int(bone_mask.sum()),
        "native_assigned_material_volume_mm3": volumes[0].tolist(),
        "verification": {
            "forward_inverse_box_faces_mm": faces,
            "inverse_material_volume_mm3": volumes[1].tolist(),
            "relative_volume_errors": ((volumes[1] - volumes[0]) / volumes[0]).tolist(),
            "source_label5_native_voxels_preserved": True,
            "simplex_and_filled_box": "passed to explicit binary32 rounding tolerance",
            "geometry_and_volume_conservation": "passed",
        },
        "implementation": {
            "script_sha256": helper.digest(Path(__file__)),
            "helper_sha256": helper.digest(Path(__file__).with_name("prepare_anatomy.py")),
            "source_license_manifest_sha256": helper.digest(intent),
        },
        "outputs": {},
    }
    for name, values, role in (
        ("forward-fractions.npy", forward, "observation generation only"),
        ("evaluation-fractions.npy", reference, "withheld evaluation only"),
        ("evaluation-ct-hu.npy", ct_coarse, "display context only"),
        (
            "evaluation-native-ct-hu.npy",
            ct_native,
            "unaltered native CT crop, lossless reorientation; display context only",
        ),
        (
            "evaluation-native-bone-mask.npy",
            np.ascontiguousarray(bone_mask.transpose(2, 1, 0)),
            "source label5 crop provenance only; forbidden solver input",
        ),
    ):
        path = output / name
        np.save(path, values, allow_pickle=False)
        metadata["outputs"][name] = {
            "sha256": helper.digest(path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "role": role,
        }
    (output / "anatomy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "prepare_vertebrae.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "output": str(output),
                "native_crop_xyz": metadata["native_crop_xyz"],
                "forward_shape": list(forward.shape),
                "inverse_shape": list(reference.shape),
                "extent_mm": extent.tolist(),
                "bone_voxels": int(bone_mask.sum()),
                "display_indices_zyx": metadata["display_indices_zyx"],
                "relative_volume_errors": metadata["verification"]["relative_volume_errors"],
                "bone_face_intersections_xyz": counts_by_face,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
