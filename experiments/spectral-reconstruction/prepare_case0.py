"""Prepare the fixed CT-ORG case-0 lumbar/partial-pelvic water-box phantom.

Original HU and scaled label-5 bytes define only private reference/display data.
The assigned fractions are not patient composition. No observations or fit are
read or generated. A failed preparation retains its .partial directory.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import nibabel as nib
import numpy as np
import prepare_anatomy as helper
import provision_inputs as provision

CROP_XYZ = ((160, 352), (99, 291), (21, 53))
SOURCE_FACE_COUNTS = {"x": [1003, 1148], "y": [41, 0], "z": [3297, 7010]}
BLOCKS_XYZ = (4, 4, 1)
SELECTION_RULE = (
    "X: centred 192 native samples; Z: centred 32 native samples; "
    "Y: floor(median label-5 Y) minus 96, clamped to 0:320, using all label-5 "
    "voxels within central 64 X samples and the selected Z slab; half-open crops. "
    "This fixed rule was accepted before case-0 observations or fitting."
)


def select_crop(scaled_labels: np.ndarray) -> tuple[tuple[slice, ...], np.ndarray, dict]:
    """Recover source labels before applying the fixed header/label-only rule."""
    if scaled_labels.shape != (512, 512, 75) or not np.isfinite(scaled_labels).all():
        raise ValueError("unexpected case-0 label shape or non-finite scaled values")
    rounded = np.rint(scaled_labels)
    rounding_error = float(np.max(np.abs(scaled_labels - rounded)))
    if rounding_error > 1e-6 or not np.array_equal(np.unique(rounded), np.arange(6)):
        raise ValueError("scaled labels do not recover the documented integer IDs 0 through 5")
    bone = rounded == 5
    nx, ny, nz = bone.shape
    x0, z0 = (nx - 192) // 2, (nz - 32) // 2
    central_x = (nx // 2 - 32, nx // 2 + 32)
    population = np.argwhere(bone[central_x[0] : central_x[1], :, z0 : z0 + 32])
    if not len(population):
        raise ValueError("fixed selection region contains no source label 5")
    median_y = float(np.median(population[:, 1]))
    y0 = int(np.clip(np.floor(median_y) - 96, 0, ny - 192))
    bounds = ((x0, x0 + 192), (y0, y0 + 192), (z0, z0 + 32))
    if bounds != CROP_XYZ or median_y != 195 or len(population) != 161307:
        raise ValueError("header/label selection differs from the accepted case-0 crop")
    slices = tuple(slice(a, b) for a, b in bounds)
    cropped = bone[slices]
    faces = face_counts(cropped)
    if faces != SOURCE_FACE_COUNTS or int(cropped.sum()) != 302304:
        raise ValueError("case-0 source bone population or accepted crop face counts changed")
    return (
        slices,
        cropped,
        {
            "rule": SELECTION_RULE,
            "central_x_for_y_rule": list(central_x),
            "median_y": median_y,
            "selection_population": len(population),
            "source_label_rounding_max": rounding_error,
            "bone_face_intersections_source_xyz": faces,
        },
    )


def face_counts(mask: np.ndarray) -> dict[str, list[int]]:
    return {
        axis: [int(np.take(mask, side, axis=i).sum()) for side in (0, -1)]
        for i, axis in enumerate("xyz")
    }


def prepare(source: Path, output: Path) -> Path:
    source = source.expanduser().resolve()
    proposed = output.expanduser().resolve()
    if proposed.is_relative_to(source):
        raise ValueError("prepared output must not be inside the original source directory")
    output, stage = provision.external_stage(proposed)
    intent = provision.source_manifest(0)
    intent["derivative_intent"] = (
        "Private fixed native label-5 water-box phantom and aligned 4x4x1 averages; "
        "no projection, fitting, patient-composition claim or publication"
    )
    provision.write_json(stage / "source-license-manifest.json", intent)
    originals = provision.case_spec(0)["files"]
    for entry in originals:
        provision.verify_file(source / entry["filename"], entry)
    geometry = provision.validate_geometry(source, 0)
    ct = nib.load(source / "volume-0.nii.gz")
    labels = nib.load(source / "labels-0.nii.gz")
    # Float64 preserves the source slope/intercept rounding error for the audit.
    slices, source_mask, selection = select_crop(np.asarray(labels.dataobj, dtype=np.float64))
    hu = np.asarray(ct.dataobj[slices], dtype=np.float32)[::-1].copy()
    bone_mask = source_mask[::-1].copy()
    if not np.isfinite(hu).all():
        raise ValueError("non-finite source HU samples")
    bone = bone_mask.astype(np.float32) * np.float32(0.65)
    water = np.float32(1) - bone
    spacing = np.asarray(geometry["spacing_mm_xyz"])
    extent = np.asarray(bone_mask.shape) * spacing
    forward = np.stack([helper.block_mean(v, (1, 1, 1)) for v in (water, bone)])
    reference = np.stack([helper.block_mean(v, BLOCKS_XYZ) for v in (water, bone)])
    forward_grid = helper.grid(forward.shape[1:], extent)
    inverse_grid = helper.grid(reference.shape[1:], extent)
    starts = np.array([v[0] for v in CROP_XYZ])
    stops = np.array([v[1] for v in CROP_XYZ])
    centre = nib.affines.apply_affine(ct.affine, (starts + stops - 1) / 2)
    first_index = np.array([stops[0] - 1, starts[1], starts[2]])
    np.testing.assert_array_equal(
        nib.affines.apply_affine(ct.affine, first_index) - centre,
        forward_grid["origin_mm"],
    )
    if forward.shape != (2, 32, 192, 192) or reference.shape != (2, 32, 48, 48):
        raise ValueError("unexpected native or inverse array layout")
    faces, volumes, simplex_errors = [], [], []
    for values, grid in ((forward, forward_grid), (reference, inverse_grid)):
        total = values.sum(axis=0, dtype=np.float64)
        simplex_error = float(np.max(np.abs(total - 1)))
        if not np.isfinite(values).all() or values.min() < 0 or simplex_error > 2**-24:
            raise ValueError("fractions exceed their declared binary32 simplex tolerance")
        step = np.asarray(grid["spacing_mm"])
        low = np.asarray(grid["origin_mm"]) - step / 2
        high = low + step * np.asarray(grid["shape"][::-1])
        np.testing.assert_array_equal(low, -extent / 2)
        np.testing.assert_array_equal(high, extent / 2)
        faces.append([low.tolist(), high.tolist()])
        volumes.append(values.sum(axis=(1, 2, 3), dtype=np.float64) * step.prod())
        simplex_errors.append(simplex_error)
    # Each stored mean is rounded once to binary32. For values in [0,1],
    # half an ULP at 1 bounds per-cell absolute error; integrate over the box.
    box_volume = float(extent.prod())
    mean_rounding_bound = float(np.spacing(np.float32(1))) / 2 * box_volume
    summation_margin = 8 * np.finfo(np.float64).eps * box_volume
    inverse_error = volumes[1] - volumes[0]
    if np.max(np.abs(inverse_error)) > mean_rounding_bound + summation_margin:
        raise ValueError("block-average material-volume error exceeds its FP32 rounding bound")
    bone_support_volume = float(bone_mask.sum() * spacing.prod())
    nominal_bone = 0.65 * bone_support_volume
    nominal_volumes = np.array([box_volume - nominal_bone, nominal_bone])
    np.testing.assert_array_equal(forward[1].transpose(2, 1, 0), bone)
    np.testing.assert_array_equal(forward[0].transpose(2, 1, 0), water)
    output_arrays = (
        ("forward-fractions.npy", forward, "observation generation only"),
        ("evaluation-fractions.npy", reference, "withheld evaluation only"),
        ("evaluation-ct-hu.npy", helper.block_mean(hu, BLOCKS_XYZ), "display context only"),
        (
            "evaluation-native-ct-hu.npy",
            np.ascontiguousarray(hu.transpose(2, 1, 0)),
            "unaltered native CT crop, lossless reorientation; display context only",
        ),
        (
            "evaluation-native-bone-mask.npy",
            np.ascontiguousarray(bone_mask.transpose(2, 1, 0)),
            "source label5 crop provenance only; forbidden solver input",
        ),
    )
    display_xyz = (np.asarray(reference.shape[1:][::-1]) - 1) // 2
    metadata = {
        "schema_version": 1,
        "case": 0,
        "title": "CT-derived lumbar and partial pelvic bone in water",
        "scientific_role": "Assigned CT-mask water-box phantom; not patient composition truth",
        "source": provision.DOI,
        "rights": provision.RIGHTS,
        "source_files": {
            v["filename"]: {"path": str(source / v["filename"]), "sha256": v["sha256"]}
            for v in originals
        },
        "source_affine": ct.affine.tolist(),
        "native_crop_xyz": [list(v) for v in CROP_XYZ],
        "crop_selection": selection,
        "reorientation": "Lossless X reversal from source LAS to RAS; XYZ-to-ZYX storage",
        "laterality": "Header-derived; source laterality not independently certified",
        "object_centre_source_ras_mm": centre.tolist(),
        "body_rule": "Entire known rectangular box filled; no HU threshold or patient body mask",
        "material_assignment": {
            "water": "1 - assigned bone in box; 1 off label 5, nominally 0.35 on label 5",
            "bone": "binary32(0.65) on original scaled/rounded label5, zero otherwise",
            "native_bone_fraction_binary32": float(np.float32(0.65)),
            "native_water_on_bone_binary32": float(np.float32(1) - np.float32(0.65)),
            "interpretation": (
                "Assigned fractions, not CT-calibrated tissue or measured composition"
            ),
        },
        "roi": {
            "anatomy": "Lumbar vertebral structures and partial inferior sacral/iliac bone",
            "isolated_vertebrae": False,
            "bone_face_intersections_xyz": face_counts(bone_mask),
            "face_coordinate_convention": "Reoriented RAS axes; X sides reverse source LAS indices",
            "bone_face_intersections_source_xyz": SOURCE_FACE_COUNTS,
            "boundary_interpretation": "Preserve all cuts, including 41 source posterior-Y samples",
            "native_axial_spacing_mm": 5.0,
        },
        "forward_grid": forward_grid,
        "inverse_grid": inverse_grid,
        "resampling": "Native 192x192x32 XYZ cells; inverse 48x48x32 XYZ aligned 4x4x1 averages",
        "boundary": (
            "Finite assigned water-filled box; vacuum outside is phantom, not patient anatomy"
        ),
        "solver_access": (
            "Grid/support and fitting projections only; no labels, CT or reference fields"
        ),
        "display_indices_zyx": display_xyz[::-1].tolist(),
        "display_indices": dict(
            zip(("sagittal", "coronal", "axial"), display_xyz.tolist(), strict=True)
        ),
        "body_native_voxels": int(bone_mask.size),
        "bone_native_voxels": int(bone_mask.sum()),
        "native_assigned_material_volume_mm3": volumes[0].tolist(),
        "verification": {
            "source_geometry": geometry,
            "forward_inverse_box_faces_mm": faces,
            "nominal_decimal_material_volume_mm3": nominal_volumes.tolist(),
            "native_binary32_minus_nominal_volume_mm3": (volumes[0] - nominal_volumes).tolist(),
            "inverse_material_volume_mm3": volumes[1].tolist(),
            "inverse_minus_native_volume_mm3": inverse_error.tolist(),
            "relative_volume_errors": (inverse_error / volumes[0]).tolist(),
            "per_material_absolute_rounding_bound_mm3": mean_rounding_bound,
            "fp64_summation_margin_mm3": summation_margin,
            "bound_formula": (
                "0.5 * spacing_float32(1.0) * box_volume_mm3; FP64 summation margin separate"
            ),
            "simplex_max_abs_sum_error": simplex_errors,
            "simplex_absolute_tolerance": 2**-24,
            "source_label5_native_voxels_preserved": True,
        },
        "implementation": {
            "script_sha256": helper.digest(Path(__file__)),
            "helper_sha256": helper.digest(Path(helper.__file__)),
            "provisioner_sha256": helper.digest(Path(provision.__file__)),
            "source_license_manifest_sha256": helper.digest(stage / "source-license-manifest.json"),
        },
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "nibabel", "scipy")
        },
        "outputs": {},
    }
    for name, values, role in output_arrays:
        target = stage / name
        np.save(target, values, allow_pickle=False)
        metadata["outputs"][name] = {
            "sha256": helper.digest(target),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "role": role,
        }
    for script in (Path(__file__), Path(helper.__file__), Path(provision.__file__)):
        (stage / script.name).write_bytes(script.read_bytes())
    provision.write_json(stage / "anatomy.json", metadata)
    for entry in originals:
        provision.verify_file(source / entry["filename"], entry)
    for key, path in (
        ("script_sha256", Path(__file__)),
        ("helper_sha256", Path(helper.__file__)),
        ("provisioner_sha256", Path(provision.__file__)),
    ):
        if helper.digest(path) != metadata["implementation"][key]:
            raise ValueError("preparation source changed during execution")
    stage.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(f"Prepared case0: {prepare(args.source, args.output)}", flush=True)


if __name__ == "__main__":
    main()
