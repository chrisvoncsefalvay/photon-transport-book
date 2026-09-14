"""Verify and provision the two original CT-ORG case 0 or 2 NIfTI files externally.

TCIA distributes this collection through its public Aspera package. Download
only the selected case's volume-N.nii.gz and labels-N.nii.gz using the official link,
then provide that directory with --cache. This script does not invent an HTTP
mirror, install a proprietary transfer client or retain browser credentials.
Use --instructions to record rights and exact file pins before that transfer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

COLLECTION = "https://www.cancerimagingarchive.net/collection/ct-org/"
DOI = "https://doi.org/10.7937/tcia.2019.tt7f4v7o"
RIGHTS = "Creative Commons Attribution 3.0 Unported (CC BY 3.0)"
FILES = (
    {
        "filename": "volume-2.nii.gz",
        "bytes": 160079756,
        "sha256": "ca702e58f0fb736ff4bc407aac0c4b698b8689f3f7e0281da32a546a87ae8a40",
    },
    {
        "filename": "labels-2.nii.gz",
        "bytes": 2922485,
        "sha256": "c5951c1065f517c9e8e794aec0a921ae0ebaa2198d4b38f0ed492bc7aca74e8f",
    },
)


# FILES remains the original case-2 compatibility constant.
CASES = {
    0: {
        "files": (
            {
                "filename": "volume-0.nii.gz",
                "bytes": 17553992,
                "sha256": "dcd2faf240a668bcc5834b52cdac68436bd06cd138b8d7381238aef2bb4a53d7",
            },
            {
                "filename": "labels-0.nii.gz",
                "bytes": 440112,
                "sha256": "4102df9468ea355733e4916787cae92cec4ad128a248cf07bf23b886d1f6b6f5",
            },
        ),
        "shape": (512, 512, 75),
        "spacing": (0.703125, 0.703125, 5.0),
        "preparation": "prepare_case0.py",
    },
    2: {
        "files": FILES,
        "shape": (512, 512, 517),
        "spacing": (0.775390625, 0.775390625, 1.0),
        "preparation": "prepare_vertebrae.py",
    },
}


def case_spec(case: int) -> dict[str, Any]:
    if case not in CASES:
        raise ValueError("CT-ORG case must be 0 or 2")
    return CASES[case]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def external_stage(output: Path) -> tuple[Path, Path]:
    output = output.expanduser().resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("inputs and derivatives must remain outside the checkout")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True, exist_ok=False)
    return output, stage


def verify_file(path: Path, record: dict[str, Any]) -> None:
    if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError(f"size/SHA-256 mismatch: {path.name}")


def obtain(record: dict[str, Any], destination: Path, caches: list[Path]) -> dict[str, Any]:
    """Copy verified cached bytes, or fetch an explicitly pinned HTTPS source.

    Called only after the caller records rights. A failed transfer stays under
    its .partial name; neither existing files nor partials are overwritten.
    """
    if destination.exists():
        raise FileExistsError(destination)
    candidates = [root.expanduser().resolve() / record["filename"] for root in caches]
    candidates = [path for path in candidates if path.is_file()]
    temporary = destination.with_name(destination.name + ".partial")
    with temporary.open("xb") as target:
        if candidates:
            source = candidates[0]
            verify_file(source, record)
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, target, length=1024 * 1024)
            mode = "verified local cache"
        else:
            url = record.get("download_url")
            if not isinstance(url, str) or not url.startswith("https://"):
                raise FileNotFoundError(
                    f"Download {record['filename']} through {COLLECTION}, "
                    "then supply its directory with --cache; no direct HTTPS URL is asserted."
                )
            request = urllib.request.Request(url, headers={"User-Agent": "DPT-input-preparation/1"})
            with urllib.request.urlopen(request, timeout=120) as stream:
                if not stream.geturl().startswith("https://"):
                    raise ValueError("source redirected outside HTTPS")
                count = 0
                while block := stream.read(1024 * 1024):
                    count += len(block)
                    if count > record["bytes"]:
                        raise ValueError("source exceeds the pinned byte length")
                    target.write(block)
            mode = "pinned HTTPS download"
    verify_file(temporary, record)
    temporary.rename(destination)
    return {**record, "intake": mode}


def source_manifest(case: int = 2) -> dict[str, Any]:
    selected = case_spec(case)
    return {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "collection": "CT-ORG",
        "collection_version": 1,
        "case": case,
        "source": DOI,
        "collection_url": COLLECTION,
        "official_readme": "https://www.cancerimagingarchive.net/wp-content/uploads/ctorg_README.txt",
        "rights": RIGHTS,
        "licence_url": "https://creativecommons.org/licenses/by/3.0/",
        "citation": (
            "Rister, B., Shivakumar, K., Nobashi, T., & Rubin, D. L. (2019). "
            "CT-ORG: A Dataset of CT Volumes With Multiple Organ Segmentations "
            "(Version 1). The Cancer Imaging Archive. DOI: 10.7937/tcia.2019.tt7f4v7o."
        ),
        "hash_origin": (
            "SHA-256/size pins from the authors' previously verified official TCIA "
            "Aspera intake. These are reproducibility pins, not upstream-published checksums."
        ),
        "transfer": {
            "provider": "TCIA public Aspera package",
            "package_id_at_pin": 592,
            "folder": "/CT-ORG/OrganSegmentations/",
            "access": (
                "Open the collection page, follow Download in the NIfTI data section, "
                "select only the two named files, and use TCIA's supported Aspera client. "
                "Client installation/licensing is separate; no account/session keys are bundled."
            ),
        },
        "files": list(selected["files"]),
        "derivative_intent": (
            "Original bytes retained; separate explicit assigned-phantom preparation."
        ),
    }


def validate_geometry(source: Path, case: int = 2) -> dict[str, Any]:
    import nibabel as nib
    import numpy as np

    selected = case_spec(case)
    # Reading to gzip EOF validates CRC without replacing the compressed originals.
    for record in selected["files"]:
        with gzip.open(source / record["filename"], "rb") as stream:
            while stream.read(1024 * 1024):
                pass
    ct, label = (nib.load(source / record["filename"]) for record in selected["files"])
    if ct.shape != selected["shape"] or label.shape != ct.shape:
        raise ValueError(f"unexpected CT-ORG case {case} shape")
    if not np.array_equal(ct.affine, label.affine) or not np.isfinite(ct.affine).all():
        raise ValueError("CT and label geometries differ or are non-finite")
    expected_spacing = np.array(selected["spacing"])
    np.testing.assert_array_equal(ct.affine[:3, :3], np.diag(expected_spacing * [-1, 1, 1]))
    if nib.aff2axcodes(ct.affine) != ("L", "A", "S"):
        raise ValueError("unexpected header axis codes")
    return {
        "shape_xyz": list(ct.shape),
        "affine": ct.affine.tolist(),
        "spacing_mm_xyz": expected_spacing.tolist(),
        "header_axis_codes": list(nib.aff2axcodes(ct.affine)),
        "source_dtypes": [str(image.get_data_dtype()) for image in (ct, label)],
        "gzip_crc": "passed for both original files",
        "laterality": "Header-led orientation; independent clinical laterality is not certified.",
        "label_interpretation": (
            "Apply NIfTI scaling, round within tolerance 1e-6 and use label 5 for bone; "
            f"{selected['preparation']} verifies all crop samples and material-volume conservation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new external directory")
    parser.add_argument("--cache", type=Path, action="append", default=[])
    parser.add_argument("--case", type=int, choices=(0, 2), default=2)
    parser.add_argument(
        "--instructions", action="store_true", help="write rights/pins before manual download"
    )
    args = parser.parse_args()
    output, stage = external_stage(args.output)
    manifest = source_manifest(args.case)
    write_json(stage / "source-license-manifest.json", manifest)
    if args.instructions:
        stage.rename(output)
        print(json.dumps(manifest, indent=2))
        return
    records = [
        obtain(record, stage / record["filename"], args.cache)
        for record in case_spec(args.case)["files"]
    ]
    geometry = validate_geometry(stage, args.case)
    write_json(stage / "verification.json", {"files": records, "geometry": geometry})
    (stage / "provision_inputs.py").write_bytes(Path(__file__).read_bytes())
    stage.rename(output)
    print(json.dumps({"output": str(output), "status": "verified", "files": records}, indent=2))


if __name__ == "__main__":
    main()
