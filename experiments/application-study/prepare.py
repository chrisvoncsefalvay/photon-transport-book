"""Prepare a known, explicitly assigned CT-derived monochromatic phantom on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import shutil
import zipfile
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
from _study import digest, new_output, read_json, save_array, write_json
from scipy import ndimage


def prepare(
    source: Path, output: Path, protocol_path: Path, case: int, xraylib_package: Path
) -> None:
    protocol = read_json(protocol_path)
    expected = protocol["data"]["source_sha256"]
    source_paths = [source / f"{kind}-{case}.nii.gz" for kind in ("volume", "labels")]
    for path in source_paths:
        if digest(path) != expected[path.name]:
            raise ValueError(f"source CT/label hash mismatch: {path.name}")
    xraylib = importlib.import_module("xraylib")
    if xraylib.__version__ != "4.3.0":
        raise ValueError("this protocol requires xraylib 4.3.0")
    if digest(xraylib_package) != protocol["data"]["xraylib_package_sha256"]:
        raise ValueError("retained xraylib package identity mismatch")
    runtime_hashes = {}
    with zipfile.ZipFile(xraylib_package) as wheel:
        for module_name in ("xraylib", "_xraylib"):
            path = Path(importlib.import_module(module_name).__file__)
            expected_runtime = hashlib.sha256(wheel.read(path.name)).hexdigest()
            if digest(path) != expected_runtime:
                raise ValueError("loaded xraylib runtime differs from the retained wheel")
            runtime_hashes[path.name] = expected_runtime
    coefficients = []
    for name in protocol["data"]["xraylib_compounds"]:
        compound = xraylib.GetCompoundDataNISTByName(name)
        mass = xraylib.CS_Total_CP(name, protocol["data"]["material_energy_kev"])
        elemental = sum(
            weight * xraylib.CS_Total(element, protocol["data"]["material_energy_kev"])
            for element, weight in zip(compound["Elements"], compound["massFractions"], strict=True)
        )
        if abs(elemental - mass) > 1e-14:
            raise ValueError("compound coefficient differs from its explicit elemental sum")
        coefficients.append(
            {
                "compound": compound,
                "mass_attenuation_cm2_g": mass,
                "mu_mm_inverse": mass * compound["density"] / 10,
                "elemental_sum_cm2_g": elemental,
            }
        )
    image, labels = (nib.load(path) for path in source_paths)
    if image.shape != labels.shape or not np.array_equal(image.affine, labels.affine):
        raise ValueError("CT and label grids differ")
    if nib.aff2axcodes(image.affine) != ("L", "A", "S"):
        raise ValueError("unexpected source axes")
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    if not np.allclose(image.affine[:3, :3], np.diag(spacing * [-1, 1, 1]), atol=1e-8):
        raise ValueError("unexpected oblique source grid")
    nz_float = protocol["data"]["crop_inferior_mm"] / spacing[2]
    nz = round(nz_float)
    if abs(nz - nz_float) > 1e-9 or nz > image.shape[2] or nz % 64:
        raise ValueError("exact requested physical crop cannot be established")
    hu = np.asarray(image.dataobj[:, :, :nz], dtype=np.float32)[::-1].copy()
    values = np.asarray(labels.dataobj[:, :, :nz], dtype=np.float32)[::-1]
    rounded = np.rint(values)
    if not np.isfinite(hu).all() or np.max(np.abs(values - rounded)) > 1e-6:
        raise ValueError("invalid HU/label values")
    bone = rounded == 5
    del values, rounded
    components, _ = ndimage.label(hu > -500)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    body = components == sizes.argmax()
    del components
    for z in range(nz):
        body[:, :, z] = ndimage.binary_fill_holes(body[:, :, z])
    original_bone_count = int(bone.sum())
    bone &= body
    water = np.clip(1 + hu / 1000, 0, 1) * body
    water[bone] = 0.35
    mu = (
        water.astype(np.float64) * coefficients[0]["mu_mm_inverse"]
        + bone * 0.65 * coefficients[1]["mu_mm_inverse"]
    )
    factors = (4, 4, nz // 64)
    blocks = mu.reshape(128, 4, 128, 4, 64, factors[2])
    field = np.ascontiguousarray(blocks.mean(axis=(1, 3, 5)).transpose(2, 1, 0), dtype=np.float32)
    extent = np.array(hu.shape) * spacing
    new_spacing = extent / np.array(field.shape[::-1])
    grid = {
        "shape": list(field.shape),
        "spacing_mm": new_spacing.tolist(),
        "origin_mm": (-extent / 2 + new_spacing / 2).tolist(),
        "orientation": np.eye(3).ravel().tolist(),
    }
    out = new_output(output)
    record = save_array(
        out,
        "known-attenuation.npy",
        field,
        units="mm^-1",
        role="known CT-derived assigned attenuation; admissible fitting/design input",
    )
    source_hashes = {path.name: digest(path) for path in source_paths}
    if any(value != expected[name] for name, value in source_hashes.items()):
        raise ValueError("source changed during preparation")
    write_json(
        out / "known.json",
        {
            "status": "complete",
            "case": case,
            "protocol_sha256": digest(protocol_path),
            "source_description": "CT-ORG assigned monochromatic phantom; "
            + protocol["data"]["source_doi"],
            "rights": protocol["data"]["rights"],
            "source_sha256": source_hashes,
            "source_affine": image.affine.tolist(),
            "native_crop_xyz": [[0, 512], [0, 512], [0, nz]],
            "source_spacing_mm": spacing.tolist(),
            "grid": grid,
            "attenuation": record,
            "xraylib_version": xraylib.__version__,
            "xraylib_package_sha256": digest(xraylib_package),
            "xraylib_loaded_runtime_sha256": runtime_hashes,
            "coefficients": coefficients,
            "energy_kev": protocol["data"]["material_energy_kev"],
            "assignment": protocol["data"]["assignment"],
            "resampling": "aligned native-cell volume means",
            "source_bone_voxels": original_bone_count,
            "retained_bone_voxels": int(bone.sum()),
            "body_voxels": int(body.sum()),
            "source_laterality": "header only; not independently certified",
            "patient_composition_truth": False,
            "preparation_sha256": digest(Path(__file__)),
            "versions": {name: version(name) for name in ("numpy", "nibabel", "scipy")},
        },
    )
    shutil.copyfile(protocol_path, out / "protocol.json")
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, out / f"source-{path.name}")
    print(f"Prepared case {case}: {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("protocol-v1.json")
    )
    parser.add_argument("--case", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--xraylib-package", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output, args.protocol, args.case, args.xraylib_package)


if __name__ == "__main__":
    main()
