"""Portable host boundaries for the bounded reconstruction development study."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from dpt.geometry import DetectorGeometry, RigidTransform, Vector3
from dpt.volumes import GridSpec

np: Any = importlib.import_module("numpy")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete_record(folder: Path) -> dict[str, Any]:
    record = json.loads((folder / "run.json").read_text())
    if (
        record.get("status") != "complete"
        or not record.get("sources_unchanged")
        or not record.get("recorded_files_unchanged")
    ):
        raise ValueError("input run is incomplete or failed its identity gates")
    return record


def recorded_array(folder: Path, name: str, record: dict[str, Any]) -> Any:
    path = folder / name
    if digest(path) != record["output_sha256"][name]:
        raise ValueError(f"recorded array changed: {name}")
    values = np.load(path, allow_pickle=False)
    if (
        values.dtype != np.float32
        or not values.dtype.isnative
        or not values.flags.c_contiguous
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"expected finite native contiguous FP32: {name}")
    return values


def coarse_cells(values: Any, grid: GridSpec, shape: tuple[int, int, int]) -> tuple[Any, GridSpec]:
    """Average exact groups of native cells, preserving every outer cell face."""
    factors = tuple(n // m for n, m in zip(grid.shape, shape, strict=True))
    if any(n != m * f for n, m, f in zip(grid.shape, shape, factors, strict=True)):
        raise ValueError("coarse shape must divide the native grid exactly")
    if tuple(values.shape[-3:]) != grid.shape or values.ndim not in (3, 4):
        raise ValueError("field shape does not match the source grid")
    prefix = values.shape[:-3]
    reshaped = values.reshape(
        *prefix, shape[0], factors[0], shape[1], factors[1], shape[2], factors[2]
    )
    offset = len(prefix)
    averaged = reshaped.mean(axis=(offset + 1, offset + 3, offset + 5), dtype=np.float64)
    factors_xyz = factors[::-1]
    result = GridSpec(
        shape,
        cast(Vector3, tuple(h * f for h, f in zip(grid.spacing_mm, factors_xyz, strict=True))),
        grid.grid_to_object(cast(Vector3, tuple(0.5 * (f - 1) for f in factors_xyz))),
        grid.orientation,
    )
    np.testing.assert_allclose(
        [grid.grid_to_object(c) for c in grid.support],
        [result.grid_to_object(c) for c in result.support],
        rtol=0,
        atol=1e-12,
    )
    return np.ascontiguousarray(averaged, dtype=np.float32), result


def geometry_in_object(geometry: DetectorGeometry, pose: RigidTransform) -> DetectorGeometry:
    """Represent identical rays with identity object pose for the scalar driver."""
    inverse = pose.inverse()
    return DetectorGeometry(
        inverse.point(geometry.source_mm),
        inverse.point(geometry.origin_mm),
        inverse.direction(geometry.u),
        inverse.direction(geometry.v),
        geometry.spacing_mm,
        geometry.shape,
    )


def poisson_counts(means: Any, config: dict[str, Any], model: str) -> tuple[Any, list[list[int]]]:
    if model not in config["noise_roles"] or means.ndim != 4:
        raise ValueError("explicit model role and view/channel/image means required")
    if not np.isfinite(means).all() or (means < 0).any():
        raise ValueError("Poisson expectations must be finite and nonnegative")
    result = np.empty(means.shape, dtype=np.float32)
    keys: list[list[int]] = []
    for view in range(means.shape[0]):
        for channel in range(means.shape[1]):
            # 0 identifies development, distinct from forthcoming final namespaces.
            key = [
                config["noise_root_seed"],
                0,
                config["case"],
                config["noise_roles"][model],
                view,
                channel,
            ]
            sample = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key))).poisson(
                means[view, channel].astype(np.float64)
            )
            if (sample > 2**24).any():
                raise ValueError("an observed count is not exactly representable in FP32")
            result[view, channel] = sample
            keys.append(key)
    return result, keys


def stationarity_gate(
    initial_mapping: float, step: float, relative: float, displacement: float
) -> dict[str, float]:
    if (
        not all(math.isfinite(v) and v >= 0 for v in (initial_mapping, relative, displacement))
        or not math.isfinite(step)
        or step <= 0
    ):
        raise ValueError("invalid stationarity gate inputs")
    return {
        "initial_mapping": initial_mapping,
        "mapping_step": step,
        "relative_limit": relative,
        "physical_displacement_limit": displacement,
        "absolute_threshold": min(relative * initial_mapping, displacement / step),
        "additional_relative_tolerance": 0.0,
    }


def stationarity_result(mapping: float, gate: dict[str, float]) -> dict[str, Any]:
    if not math.isfinite(mapping) or mapping < 0:
        raise ValueError("final mapping is invalid")
    initial = gate["initial_mapping"]
    relative = mapping / initial if initial > 0 else (0.0 if mapping == 0 else None)
    physical = gate["mapping_step"] * mapping
    return {
        **gate,
        "final_mapping": mapping,
        "relative_mapping": relative,
        "mapped_physical_displacement": physical,
        "relative_passed": relative is not None and relative <= gate["relative_limit"],
        "physical_passed": physical <= gate["physical_displacement_limit"],
        "passed": mapping <= gate["absolute_threshold"],
        "diagnostic_metric": "euclidean",
    }


class PilotBudgetError(RuntimeError):
    """Only an accepted callback may raise this dedicated soft-budget stop."""


def scalar_case(
    folder: Path, acquisition: Path, public: dict[str, Any], config: dict[str, Any]
) -> Path:
    """Write a supplied-driver case using observations and declared support only."""
    grid = GridSpec(**{k: tuple(v) for k, v in public["inverse_grid"].items()})
    initial = np.full(
        grid.shape, config["scalar_coefficients_80kev_mm_inverse"][0], dtype=np.float32
    )
    folder.mkdir(parents=True, exist_ok=False)
    initial_path = folder / "initial.npy"
    # This helper writes ordinary NPY bytes; all inputs are later source-snapshotted.
    np.save(initial_path, initial, allow_pickle=False)
    arrays = {}

    def add(name: str, path: Path, units: str) -> str:
        arrays[name] = {
            "path": str(path.resolve()),
            "sha256": digest(path),
            "units": units,
            "source": "development counts or declared homogeneous water support",
            "rights": "original derivative; source attribution retained in acquisition record",
        }
        return name

    initial_name = add("initial", initial_path, "mm^-1")
    views: list[dict[str, Any]] = []
    geometry = DetectorGeometry(**{k: tuple(v) for k, v in public["geometry"].items()})
    for index, item in enumerate(public["views"]):
        pose = RigidTransform(**{k: tuple(v) for k, v in item["pose"].items()})
        views.append(
            {
                "geometry": asdict(geometry_in_object(geometry, pose)),
                "counts": add(f"view{index}", acquisition / item["scalar_counts"], "counts"),
                "open_beam": "beam",
            }
        )
    add("beam", acquisition / public["open_beam"], "counts")
    result: dict[str, Any] = {
        "schema_version": 1,
        "arrays": arrays,
        "reconstruction": {
            "grid": asdict(grid),
            "fixed_pose": asdict(RigidTransform()),
            "initial_volume": initial_name,
            "samples_per_ray": public["inverse_samples"],
            "views": views,
            "solver": {
                "iterations": config["pilot_steps"],
                "initial_step_mm_inverse_squared": config.get(
                    "scalar_initial_trial_step_mm_inverse_squared",
                    config["scalar_initial_and_mapping_step_mm_inverse_squared"],
                ),
                "backtracking_factor": 0.5,
                "armijo": 1e-4,
                "maximum_backtracks": 40,
                "gradient_mapping_tolerance_mm": 0.0,
                "regularisation_mm": config["fraction_regularisation_mm_inverse"]
                / config["scalar_coefficients_80kev_mm_inverse"][0] ** 2,
            },
        },
    }
    if "scalar_initial_trial_step_mm_inverse_squared" in config:
        result["reconstruction"]["solver"]["mapping_step_mm_inverse_squared"] = config[
            "scalar_initial_and_mapping_step_mm_inverse_squared"
        ]
    path = folder / "case.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path
