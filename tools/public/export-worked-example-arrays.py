"""Export exact accepted arrays for figures drawn in the page.

Only NumPy and the standard library are used. The destination must be a new
directory outside the repository and the scientific records. This command does
not run a model, draw a figure, admit public assets or alter a recorded result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = "tools/public/export-worked-example-arrays.py"
IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
FloatArray = npt.NDArray[np.float32 | np.float64]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return cast(dict[str, Any], value)


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


@dataclass
class Evidence:
    root: Path
    records: dict[str, str]
    files: dict[Path, str]
    verified_hash_count: int

    def assert_unchanged(self) -> None:
        for path, expected in self.files.items():
            if digest(path) != expected:
                raise ValueError(f"Changed execution evidence: {path.relative_to(self.root)}")

    def bound_file(self, relative: str) -> Path:
        path = self.root / relative
        if path not in self.files:
            raise ValueError(f"Input is not bound by an execution record: {relative}")
        if digest(path) != self.files[path]:
            raise ValueError(f"Changed input: {relative}")
        return path

    def array(self, relative: str) -> FloatArray:
        value = np.load(self.bound_file(relative), allow_pickle=False)
        if value.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"Expected a recorded FP32 or FP64 array: {relative}")
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite input: {relative}")
        return value


def verified_evidence(root: Path, expected_records: int) -> Evidence:
    records: dict[str, str] = {}
    files: dict[Path, str] = {}
    count = 0
    for path in sorted(root.rglob("run.json")):
        if "sources" in path.relative_to(root).parts:
            continue
        record = read_json(path)
        if (
            record.get("status") != "complete"
            or record.get("sources_unchanged") is not True
            or record.get("recorded_files_unchanged") is not True
        ):
            raise ValueError(f"Incomplete or changed execution: {path.relative_to(root)}")
        records[str(path.relative_to(root))] = digest(path)
        files[path] = digest(path)
        for key, base in (
            ("source_sha256", path.parent / "sources"),
            ("output_sha256", path.parent),
        ):
            for relative, expected in record[key].items():
                candidate = (base / relative).resolve()
                if not candidate.is_relative_to(base) or digest(candidate) != expected:
                    raise ValueError(f"Changed or unsafe execution member: {relative}")
                if candidate in files and files[candidate] != expected:
                    raise ValueError(f"Conflicting recorded hashes: {relative}")
                files[candidate] = expected
                count += 1
    if len(records) != expected_records:
        raise ValueError(f"Expected {expected_records} completed records; found {len(records)}")
    return Evidence(root, records, files, count)


def packed_array(value: FloatArray, shape: tuple[int, ...]) -> dict[str, Any]:
    if value.shape != shape:
        raise ValueError(f"Recorded shape {value.shape} differs from required shape {shape}")
    packed = {"shape": list(shape), "dtype": value.dtype.name, "values": value.ravel().tolist()}
    restored = np.asarray(json.loads(json_bytes(packed))["values"], dtype=value.dtype).reshape(
        shape
    )
    if restored.tobytes(order="C") != value.tobytes(order="C"):
        raise ValueError("JSON conversion changed recorded numerical bits")
    return packed


def registration(evidence: Evidence) -> tuple[dict[str, Any], dict[str, str]]:
    summary = read_json(evidence.bound_file("summary.json"))
    if summary.get("development") is not False or summary.get("passed") is not True:
        raise ValueError("Registration requires the accepted final application study")
    children = read_json(evidence.bound_file("child-records.json"))
    if children != {key: value for key, value in evidence.records.items() if key != "run.json"}:
        raise ValueError("Application child-record bindings differ")
    protocol = read_json(evidence.root / "run.json")["configuration"]["protocol"]
    geometry = protocol["geometry"]
    shape = geometry["detector_shape_hw"]
    spacing = geometry["detector_spacing_mm_uv"]
    if shape != [128, 128] or spacing != [8.0, 8.0]:
        raise ValueError("Expected the accepted complete 128-square detector")
    if protocol["registration"]["views_degrees"] != [0.0, 90.0]:
        raise ValueError("Expected the prescribed two orthogonal registration views")
    width, height = shape[1] * spacing[0], shape[0] * spacing[1]
    inputs: dict[str, str] = {}
    arrays: dict[str, Any] = {}
    for suffix in ("a180", "a270"):
        view = f"r1-n00-{suffix}"
        for stage, relative in (
            ("initial", f"fits/registration/initial-{view}.npy"),
            ("observed", f"observations/{view}.npy"),
            ("final", f"fits/registration/prediction-{view}.npy"),
        ):
            arrays[f"{stage}_{suffix}"] = packed_array(evidence.array(relative), (128, 128))
            inputs[relative] = digest(evidence.bound_file(relative))
    value = {
        "schema_version": 1,
        "kind": "registration",
        "axis_order": ["detector_row", "detector_column"],
        "array_order": "C",
        "units": "counts",
        "grid": {"shape": shape, "spacing_mm": spacing},
        "extent_mm": [-width / 2, width / 2, -height / 2, height / 2],
        "first_row": "negative physical detector y",
        "views": [{"id": "a180", "angle_degrees": 0}, {"id": "a270", "angle_degrees": 90}],
        "stages": [
            {"id": "initial", "label": "Initial prediction"},
            {"id": "observed", "label": "Observed counts"},
            {"id": "final", "label": "Recovered prediction"},
        ],
        "display": {
            "transform": "log10(1 + counts)",
            "window_counts": [0, 10000],
            "pixel_interpolation": "nearest",
        },
        "arrays": arrays,
    }
    return value, inputs


def reconstruction(
    evidence: Evidence, fields: Path
) -> tuple[dict[str, Any], dict[str, str], dict[Path, str]]:
    summary = read_json(evidence.bound_file("summary.json"))
    if summary.get("development") is not False or summary.get("accepted") is not True:
        raise ValueError("Reconstruction requires both accepted final replicates")
    if summary["child_records_sha256"] != {
        key: value for key, value in evidence.records.items() if key != "run.json"
    }:
        raise ValueError("Reconstruction child-record bindings differ")
    fitting = read_json(evidence.bound_file("acquisition/fitting.json"))
    grid = fitting["grid"]
    if grid["shape"] != [16, 16, 16] or grid["orientation"] != IDENTITY:
        raise ValueError("Expected the accepted axis-aligned 16-cubed grid")
    if [row["replicate"] for row in summary["replicates"]] != [0, 1]:
        raise ValueError("Both prescribed replicates must remain present in order")
    for replicate in (0, 1):
        assessment = read_json(evidence.bound_file(f"evaluation-rep{replicate}/evaluation.json"))
        result = read_json(evidence.bound_file(f"fit-rep{replicate}/solver-result.json"))
        if (
            assessment != summary["replicates"][replicate]
            or assessment.get("accepted") is not True
            or result["stationarity"].get("passed") is not True
        ):
            raise ValueError(f"Replicate {replicate} does not satisfy final acceptance")
    curator = fields.with_name("provenance.json")
    curator_record = read_json(curator)
    if (
        digest(fields) != curator_record["output_sha256"]["fields.npz"]
        or curator_record["execution_record_sha256"] != evidence.records["run.json"]
        or curator_record["child_records_sha256"] != summary["child_records_sha256"]
    ):
        raise ValueError("The curated field archive is not bound to the accepted run")
    mapping = {"reference": "acquisition/evaluation/reference-fractions.npy"}
    for replicate in (0, 1):
        mapping.update(
            {
                f"initial_rep{replicate}": f"fit-rep{replicate}/checkpoints/fields-0000.npy",
                f"final_rep{replicate}": f"fit-rep{replicate}/fields.npy",
                f"update100_rep{replicate}": f"fit-rep{replicate}/checkpoints/fields-0100.npy",
            }
        )
    arrays: dict[str, Any] = {}
    inputs: dict[str, str] = {}
    with np.load(fields, allow_pickle=False) as archive:
        if set(archive.files) != set(mapping):
            raise ValueError("The complete seven-array field archive is required")
        for name, relative in mapping.items():
            raw = evidence.array(relative)
            archived = archive[name]
            if archived.dtype != raw.dtype or archived.shape != raw.shape:
                raise ValueError(f"Archived field metadata differs: {name}")
            if archived.tobytes(order="C") != raw.tobytes(order="C"):
                raise ValueError(f"Archived field differs from the accepted raw field: {name}")
            arrays[name] = packed_array(archived, (2, 16, 16, 16))
            inputs[relative] = digest(evidence.bound_file(relative))
    spacing, origin = np.asarray(grid["spacing_mm"]), np.asarray(grid["origin_mm"])
    lower = origin - spacing / 2
    upper = lower + np.asarray(grid["shape"])[::-1] * spacing
    if origin[2] + 7.5 * spacing[2] != 0:
        raise ValueError("The recorded zero plane is not halfway between centres 7 and 8")
    value = {
        "schema_version": 1,
        "kind": "reconstruction",
        "axis_order": ["material", "z", "y", "x"],
        "array_order": "C",
        "units": "dimensionless material fraction",
        "materials": ["Water", "Bone"],
        "grid": grid,
        "extent_mm": [float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])],
        "box_extent_xyz_mm": [[float(lo), float(hi)] for lo, hi in zip(lower, upper, strict=True)],
        "first_row": "negative physical object y",
        "stages": [
            {"id": "reference", "label": "Assigned reference"},
            {"id": "initial", "label": "Uniform start"},
            {"id": "update100", "label": "Update 100", "accepted_update": 100},
            {"id": "final", "label": "Stationary field"},
            {"id": "error", "label": "Signed final error", "expression": "final - reference"},
        ],
        "display": {
            "fraction_window": [0, 1],
            "signed_error_window": [-0.1, 0.1],
            "pixel_interpolation": "nearest",
            "default_plane": {
                "axis": "z",
                "coordinate_mm": 0,
                "sample_indices": [7, 8],
                "weights": [0.5, 0.5],
                "interpolation": "linear between adjacent sample centres in physical z",
            },
        },
        "arrays": arrays,
    }
    return value, inputs, {fields: digest(fields), curator: digest(curator)}


def export(applications: Path, materials: Path, fields: Path, output: Path) -> None:
    if output.exists() or any(
        output.is_relative_to(base) for base in (ROOT, applications, materials)
    ):
        raise ValueError("Output must be a new directory outside the repository and input records")
    exporter_sha256 = digest(Path(__file__))
    application_evidence = verified_evidence(applications, 34)
    material_evidence = verified_evidence(materials, 6)
    registration_data, registration_inputs = registration(application_evidence)
    reconstruction_data, reconstruction_inputs, curated_inputs = reconstruction(
        material_evidence, fields
    )
    payloads: list[tuple[str, bytes, bytes]] = []
    for evidence, data, inputs in (
        (application_evidence, registration_data, registration_inputs),
        (material_evidence, reconstruction_data, reconstruction_inputs),
    ):
        arrays = json_bytes(data)
        known = (
            read_json(evidence.bound_file("sources/known.json"))
            if data["kind"] == "registration"
            else read_json(evidence.bound_file("acquisition/fitting.json"))["protocol"]
        )
        provenance = {
            "schema_version": 1,
            "kind": data["kind"],
            "source": "https://doi.org/10.7937/tcia.2019.tt7f4v7o",
            "rights": "CC BY 3.0; Rister et al., CT-ORG, TCIA; assigned-material derivatives",
            "source_file_sha256": known["source_sha256"],
            "execution_records_sha256": evidence.records,
            "verified_record_count": len(evidence.records),
            "verified_source_and_output_hash_count": evidence.verified_hash_count,
            "input_sha256": inputs,
            "curated_input_sha256": {
                path.name: expected for path, expected in curated_inputs.items()
            }
            if data["kind"] == "reconstruction"
            else {},
            "exporter": {"file": EXPORTER, "sha256": exporter_sha256},
            "output_sha256": {"arrays.json": hashlib.sha256(arrays).hexdigest()},
            "verification": {
                "array_count": len(data["arrays"]),
                "value_count": sum(len(row["values"]) for row in data["arrays"].values()),
                "json_roundtrip": "bit-for-bit equality at each recorded dtype",
                "processing": "C-order flattening only; all recorded numerical values retained",
            },
        }
        payloads.append((data["kind"], arrays, json_bytes(provenance)))
    application_evidence.assert_unchanged()
    material_evidence.assert_unchanged()
    for path, expected in curated_inputs.items():
        if digest(path) != expected:
            raise ValueError(f"Changed curated input: {path.name}")
    if digest(Path(__file__)) != exporter_sha256:
        raise ValueError("Exporter source changed during extraction")
    output.mkdir(parents=True, exist_ok=False)
    for kind, arrays, provenance in payloads:
        folder = output / kind
        folder.mkdir()
        (folder / "arrays.json").write_bytes(arrays)
        (folder / "arrays-provenance.json").write_bytes(provenance)
    print(f"Exported 98,304 registration and 57,344 material values to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--applications-source", type=Path, required=True)
    parser.add_argument("--reconstruction-source", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(
        args.applications_source.resolve(),
        args.reconstruction_source.resolve(),
        args.fields.resolve(),
        args.output.resolve(),
    )
