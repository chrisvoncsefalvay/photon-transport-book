"""Extract attributed display-only pelvic surfaces from the paired conditioning mask."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy
import skimage
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from dpt.experiments import RunRecorder, experiment_parser, private_output, repository_root

ROOT = repository_root(__file__)


def main() -> None:
    parser = experiment_parser(__file__, __doc__, cuda=False)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if hashlib.sha256(args.labels.read_bytes()).hexdigest() != config["label_sha256"]:
        raise ValueError("conditioning mask SHA256 mismatch")
    nifti = nib.load(args.labels)
    labels = np.asanyarray(nifti.dataobj)
    pivot = np.asarray(config["pivot_ras_mm"])
    meshes = []
    for label, name, title in [
        (2, "sacrum", "Sacrum"),
        (3, "left-hip", "Left hip"),
        (4, "right-hip", "Right hip"),
    ]:
        surface = gaussian_filter((labels == label).astype(np.float32), sigma=0.8)
        vertices, faces, _, _ = marching_cubes(
            surface, level=0.5, step_size=2, allow_degenerate=False
        )
        vertices = vertices @ nifti.affine[:3, :3].T + nifti.affine[:3, 3] - pivot
        tri = vertices[faces]
        signed_volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()
        if signed_volume < 0:
            faces = faces[:, ::-1]
        meshes.append(
            {
                "id": name,
                "label": title,
                "positions": vertices.round(4).ravel().tolist(),
                "indices": faces.ravel().tolist(),
            }
        )
        print(title, len(vertices), "vertices", len(faces), "triangles", flush=True)
    payload = {
        "schema_version": 1,
        "coordinate_system": "RAS-mm-pivot-centred",
        "meshes": meshes,
        "attribution": {
            "title": "TotalSegmentator dataset v2",
            "creator": "Jakob Wasserthal and University Hospital Basel",
            "source": "https://doi.org/10.5281/zenodo.10047292",
            "licence": "https://creativecommons.org/licenses/by/4.0/",
            "changes": (
                "NVIDIA preprocessing, resampling and augmentation; labels 2/3/4 "
                "extracted, Gaussian-smoothed sigma 0.8 voxels, level 0.5 stride-2 "
                "marching cubes, centred in RAS mm; vertices rounded to 0.0001 mm. Display "
                "geometry only, not a fresh CT segmentation."
            ),
            "label_sha256": config["label_sha256"],
        },
    }
    sources = {
        "experiments/pose-sensitivity/mesh.py": Path(__file__),
        "experiments/pose-sensitivity/config.json": args.config,
        "experiments/pose-sensitivity/requirements-volume.txt": Path(__file__).with_name(
            "requirements-volume.txt"
        ),
        "ATTRIBUTIONS.md": ROOT / "ATTRIBUTIONS.md",
    }
    with RunRecorder(
        private_output(args.output, ROOT),
        configuration=config,
        sources=sources,
        metadata={
            "nibabel": nib.__version__,
            "skimage": skimage.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
        },
    ) as record:
        record.write_json("meshes.json", payload)


if __name__ == "__main__":
    main()
