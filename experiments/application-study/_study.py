"""Recorded input and geometry helpers; scientific operators live in dpt."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from dpt.experiments import experiment_sources, private_output, repository_root
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.volumes import GridSpec


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def grid_spec(raw: dict) -> GridSpec:
    return GridSpec(**{key: tuple(value) for key, value in raw.items()})


def rigid(raw: dict) -> RigidTransform:
    return RigidTransform(**{key: tuple(value) for key, value in raw.items()})


def new_output(path: Path) -> Path:
    result = private_output(path, repository_root(__file__))
    result.mkdir(parents=True, exist_ok=False)
    return result


def record_sources(entrypoint: str, protocol: Path, **extra: Path) -> dict[str, Path]:
    sources = experiment_sources(entrypoint, protocol, extra=extra)
    for path in Path(__file__).parent.glob("*.py"):
        sources[f"application-study/{path.name}"] = path
    return sources


def checked_array(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = root / record["file"]
    if path.is_symlink() or digest(path) != record["sha256"]:
        raise ValueError(f"array identity mismatch: {path}")
    result = np.load(path, allow_pickle=False)
    if result.dtype != np.dtype(record["dtype"]) or list(result.shape) != record["shape"]:
        raise ValueError("recorded dtype/shape mismatch")
    if not result.flags.c_contiguous or not np.isfinite(result).all():
        raise ValueError("arrays must be finite and C-contiguous")
    return result


def save_array(
    root: Path, name: str, array: np.ndarray, *, units: str, role: str, recorder: Any = None
) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if recorder is None:
        with path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
    else:
        payload = io.BytesIO()
        np.save(payload, array, allow_pickle=False)
        recorder.write_bytes(name, payload.getvalue())
    return {
        "file": name,
        "sha256": digest(path),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "units": units,
        "role": role,
    }


def geometry(protocol: dict, angle_degrees: float) -> DetectorGeometry:
    """Rotate source, detector centre and detector axes together about world Z."""
    cfg = protocol["geometry"]
    angle = math.radians(angle_degrees)
    normal = np.array([math.cos(angle), math.sin(angle), 0.0])
    u, v = np.array([-math.sin(angle), math.cos(angle), 0.0]), np.array([0.0, 0.0, 1.0])
    height, width = cfg["detector_shape_hw"]
    hu, hv = cfg["detector_spacing_mm_uv"]
    sid, sdd = cfg["source_isocentre_mm"], cfg["source_detector_mm"]
    source = -sid * normal
    origin = (sdd - sid) * normal - (width - 1) * hu / 2 * u - (height - 1) * hv / 2 * v
    return DetectorGeometry(
        tuple(source), tuple(origin), tuple(u), tuple(v), (hu, hv), (height, width)
    )


def array_input(root: Path, record: dict, known: dict) -> dict:
    return {
        "path": str((root / record["file"]).resolve()),
        "sha256": record["sha256"],
        "units": record["units"],
        "source": known["source_description"],
        "rights": known["rights"],
    }


def registration_case(
    known_root: Path,
    observations_root: Path,
    public: dict,
    protocol: dict,
    ids: list[str],
    anchor: Any,
    output: Path,
    *,
    samples: int | None = None,
) -> Path:
    """Only named fitting observations are opened; reference files are not inputs."""
    known = read_json(known_root / "known.json")
    if digest(known_root / "known.json") != public["known_sha256"]:
        raise ValueError("known-volume manifest changed")
    arrays = {"attenuation": array_input(known_root, known["attenuation"], known)}
    views = []
    for identifier in ids:
        view = public["views"][identifier]
        for role in ("observation", "mask"):
            arrays[f"{identifier}-{role}"] = array_input(observations_root, view[role], known)
        views.append(
            {
                "id": identifier,
                "geometry": view["geometry"],
                "observation": f"{identifier}-observation",
                "mask": f"{identifier}-mask",
                "open_beam_counts": view["open_beam_counts"],
            }
        )
    config = {
        "schema_version": 1,
        "arrays": arrays,
        "registration": {
            "grid": known["grid"],
            "attenuation": "attenuation",
            "chart": {
                "anchor": asdict(anchor),
                "scales": protocol["numerics"]["chart_scales"],
                "rotation_radius_radians": protocol["numerics"]["chart_rotation_radius_radians"],
            },
            "samples_per_ray": samples or protocol["numerics"]["fitting_samples_per_ray"],
            "precision": protocol["numerics"].get("precision", "float64"),
            "integration": protocol["numerics"].get("integration", "midpoint"),
            "policy": protocol["numerics"]["policy"],
            "views": views,
        },
    }
    write_json(output, config)
    return output


def keyed_rng(seed: int, *keys: int) -> np.random.Generator:
    """Stable integer namespace; never use Python's process-randomised hash()."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, *keys])))
