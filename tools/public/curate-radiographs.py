"""Render display-only radiographs from hash-verified recorded detector counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "public/generated/radiographs"
SOURCE = ROOT / "public/generated/introduction-pose-sensitivity"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError(f"Invalid recorded path: {relative}")
    return path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--directional", type=Path)
    args = parser.parse_args()
    manifest = json.loads((SOURCE / "artifact.manifest.json").read_text())
    for name, expected in manifest["output_digests"].items():
        if digest(checked(ROOT, name)) != expected:
            raise ValueError(f"Pose input digest mismatch: {name}")
    data = json.loads((SOURCE / "data.json").read_text())
    shape = (data["height"], data["width"])

    def counts(name: str) -> np.ndarray:
        a = np.fromfile(checked(SOURCE, name), dtype="<f4").astype(np.float64)
        if (
            a.size != shape[0] * shape[1]
            or not np.isfinite(a).all()
            or (a <= 0).any()
            or (a > 1000).any()
        ):
            raise ValueError("Invalid primary counts")
        return a.reshape(shape)

    reference = counts("projection.f32")
    arrays = {p["id"]: counts(p["projection_f32"]) for p in data["poses"]}
    if len(arrays) != 55:
        raise ValueError("Expected all 55 accepted local poses")
    TARGET.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    def png(name: str, unit: np.ndarray) -> str:
        if not np.isfinite(unit).all():
            raise ValueError("Non-finite display value")
        path = TARGET / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.rint(np.clip(unit, 0, 1) * 255).astype(np.uint8)).save(path)
        outputs.append(path)
        return name

    # Fixed across every pose; no image-by-image histogram equalisation.
    window = [2.0, 5.3]
    difference_limit = max(float(np.max(np.abs(a - reference))) for a in arrays.values())
    softening = 0.01 * difference_limit
    rendered = {}
    for pose in data["poses"]:
        a = arrays[pose["id"]]
        depth = -np.log(a / 1000)
        rendered[pose["id"]] = {
            "id": pose["id"],
            "value": pose["value"],
            "matrix": pose["matrix"],
            "image": png(f"local/{pose['id']}.png", (depth - window[0]) / (window[1] - window[0])),
            "difference": png(
                f"local/{pose['id']}-difference.png",
                np.arcsinh(np.abs(a - reference) / softening)
                / np.arcsinh(difference_limit / softening),
            ),
            "source_counts": "/generated/introduction-pose-sensitivity/" + pose["projection_f32"],
            "max_absolute_count_change": float(np.max(np.abs(a - reference))),
        }
    groups = []
    for parameter in ["tx", "ty", "tz", "rx", "ry", "rz"]:
        poses = sorted(
            [p for p in data["poses"] if p["parameter"] in [parameter, "reference"]],
            key=lambda p: p["value"],
        )
        frames = [rendered[p["id"]] for p in poses]
        groups.append(
            {
                "id": parameter,
                "label": {
                    "tx": "Translation x · right",
                    "ty": "Translation y · anterior",
                    "tz": "Translation z · superior",
                    "rx": "Rotation about x",
                    "ry": "Rotation about y",
                    "rz": "Rotation about z",
                }[parameter],
                "unit": "mm" if parameter[0] == "t" else "rad",
                "initial": next(i for i, f in enumerate(frames) if f["id"] == "reference"),
                "frames": frames,
            }
        )
    local = {
        "schema_version": 1,
        "width": shape[1],
        "height": shape[0],
        "aspect": 600 / 320,
        "groups": groups,
        "display": {
            "optical_depth_window": window,
            "difference_limit_counts": difference_limit,
            "difference_asinh_softening_counts": softening,
        },
        "formation": {
            "counts": png("formation/counts.png", reference / 1000),
            "depth": rendered["reference"]["image"],
            "transmission_min": float(reference.min() / 1000),
            "depth_max": float((-np.log(reference / 1000)).max()),
        },
        "model": (
            "Full synthetic MAISI CT; illustrative water-equivalent attenuation at 80 keV; "
            "expected primary counts, no scatter or detector noise."
        ),
        "provenance": "provenance.json",
    }
    write_json(TARGET / "local-poses.json", local)
    outputs.append(TARGET / "local-poses.json")
    provenance = {
        "schema_version": 1,
        "local_pose_input_manifest_sha256": digest(SOURCE / "artifact.manifest.json"),
        "local_pose_output_digests": manifest["output_digests"],
        "local_pose_producer_sources": manifest.get("source_digests", {}),
        "display_only": True,
        "display_conventions": local["display"],
        "interpolation": (
            "None between poses: each displayed frame is separately recorded. "
            "PNG intensities quantise the stated fixed display mapping to 8 bits."
        ),
        "model": local["model"],
        "rights": (
            "See ATTRIBUTIONS.md: MAISI synthetic CT outputs; "
            "no raw CT or conditioning mask is included."
        ),
    }
    if args.study:
        study_root = args.study.resolve()
        run = json.loads(checked(study_root, "run.json").read_text())
        if run["status"] != "complete":
            raise ValueError("Study run incomplete")
        for name, expected in run["output_sha256"].items():
            if digest(checked(study_root, name)) != expected:
                raise ValueError(f"Study digest mismatch: {name}")
        study = json.loads(checked(study_root, "study.json").read_text())

        # Preserve scientific metadata, replace private binary links by their digests.
        def curate(value):
            if isinstance(value, list):
                return [curate(v) for v in value]
            if isinstance(value, dict):
                result = {}
                for key, v in value.items():
                    if key in ["counts_f32", "depth_f32", "expected_f32", "observed_u64"]:
                        result[key + "_sha256"] = digest(checked(study_root, v))
                    elif (
                        key in ["image", "ideal_image"]
                        and isinstance(v, str)
                        and v.endswith(".png")
                    ):
                        source = checked(study_root, v)
                        dest = TARGET / "study" / v
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(source.read_bytes())
                        outputs.append(dest)
                        result[key] = "study/" + v
                    else:
                        result[key] = curate(v)
                return result
            if isinstance(value, str) and ("/home/" in value or "/tmp/" in value):
                raise ValueError("Private path in study metadata")
            return value

        clean = curate(study)
        write_json(TARGET / "study.json", clean)
        outputs.append(TARGET / "study.json")
        provenance["study"] = {
            "run_sha256": digest(study_root / "run.json"),
            "configuration_sha256": run["configuration_sha256"],
            "source_sha256": run["source_sha256"],
            "output_sha256": run["output_sha256"],
            "checks": study["checks"],
        }
    if args.directional:
        run_root = args.directional.resolve()
        record = json.loads(checked(run_root, "run.json").read_text())
        if record["status"] != "complete":
            raise ValueError("Directional run incomplete")
        for name, expected in record["output_sha256"].items():
            if digest(checked(run_root, name)) != expected:
                raise ValueError(f"Directional digest mismatch: {name}")
        payload = json.loads(checked(run_root, "directional-checks.json").read_text())
        if payload["checks"]["passed"] is not True:
            raise ValueError("Directional acceptance failed")
        write_json(TARGET / "directional-checks.json", payload)
        outputs.append(TARGET / "directional-checks.json")
        provenance["directional"] = {
            "run_sha256": digest(run_root / "run.json"),
            "configuration_sha256": record["configuration_sha256"],
            "source_sha256": record["source_sha256"],
            "output_sha256": record["output_sha256"],
        }
    write_json(TARGET / "provenance.json", provenance)
    outputs.append(TARGET / "provenance.json")

    def relative(p: Path) -> str:
        return p.relative_to(ROOT).as_posix()

    outputs = sorted(set(outputs))
    generator = Path(__file__).resolve()
    write_json(
        TARGET / "artifact.manifest.json",
        {
            "schema_version": 1,
            "id": "maisi-radiograph-displays",
            "chapter": "multiple",
            "experiment": "maisi-radiographs",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "generator": relative(generator),
            "mode": "deterministic-precomputed-sweep",
            "parameters": {"display_only": True, "local_pose_count": 55},
            "outputs": [relative(p) for p in outputs],
            "stochastic": False,
            "created_at": datetime.now(UTC).isoformat(),
            "notes": (
                "Display curation of archived deterministic projections and separately labelled "
                "recorded observations. Producer hashes and stochastic sampling details are in "
                "provenance.json; the curation commit does not assert that archived inputs "
                "used current scientific sources."
            ),
            "source_digests": {relative(generator): digest(generator)},
            "output_digests": {relative(p): digest(p) for p in outputs},
        },
    )
    print(json.dumps({"outputs": len(outputs), "local_poses": 55, "study": bool(args.study)}))


if __name__ == "__main__":
    main()
