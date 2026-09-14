"""Pure CPU field summaries for the new reconstruction comparison.

Inputs are explicit finite arrays and GridSpec only. This module does not read
files, discover references, choose fits or certify material-domain constraints.
Values are promoted to FP64 without clipping. Role-gated orchestration and
scientific provenance belong to the caller. Validation examples are tests only.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from math import prod
from typing import Any, cast

import numpy as np

from dpt.volumes import GridSpec

# SciPy supplies no complete EDT typing; retain its installed function unchanged.
distance_transform_edt: Any = import_module("scipy.ndimage").distance_transform_edt

RULE_ID = "reconstruction-field-rois-v1"
WATER_TOLERANCE = 1e-7
BONE_THRESHOLD = 0.325
INTERFACE_UPPER = float(np.float32(0.65)) - 1e-7
CORE_DISTANCE_MM = 5.0


def _field(values: np.ndarray, grid: GridSpec, *, materials: bool = False) -> np.ndarray:
    if not isinstance(cast(object, grid), GridSpec):
        raise TypeError("grid must be an explicit GridSpec")
    shape = (2, *grid.shape) if materials else grid.shape
    if (
        not isinstance(cast(object, values), np.ndarray)
        or values.dtype.kind not in "fiu"
        or values.shape != shape
    ):
        raise ValueError(f"field must be a real numeric NumPy array with shape {shape}")
    converted = values.astype(np.float64, copy=False)
    if not np.isfinite(converted).all():
        raise ValueError("every field value must be finite, including outside any named ROI")
    return converted


def _masks(rois: Mapping[str, np.ndarray] | None, grid: GridSpec) -> dict[str, np.ndarray]:
    result = {"whole_box": np.ones(grid.shape, dtype=bool)}
    if rois is not None:
        for name, mask in rois.items():
            if (
                not isinstance(cast(object, name), str)
                or not name
                or not isinstance(cast(object, mask), np.ndarray)
                or mask.dtype != np.bool_
                or mask.shape != grid.shape
            ):
                raise ValueError("each named ROI must be a Boolean NumPy array of grid shape")
            if name == "whole_box" and not mask.all():
                raise ValueError("whole_box cannot exclude samples")
            result[name] = mask
    return result


def reference_rois(reference: np.ndarray, grid: GridSpec) -> dict[str, np.ndarray]:
    """Overlapping reference ROIs; strict discrete-centre EDT cores in millimetres.

    One excluded sample is padded on every face. Distances refer to excluded
    voxel centres, not half-voxel surfaces. Proper grid rotations preserve them.
    Comparisons occur after FP64 promotion, including decimal threshold 0.325.
    """
    water, bone = _field(reference, grid, materials=True)
    pure_water = (bone <= WATER_TOLERANCE) & (np.abs(water - 1) <= WATER_TOLERANCE)
    bone_rich = bone >= BONE_THRESHOLD
    masks = {
        "whole_box": np.ones(grid.shape, dtype=bool),
        "pure_water": pure_water,
        "bone_rich": bone_rich,
        "partial_volume_interface": (bone > WATER_TOLERANCE) & (bone < INTERFACE_UPPER),
    }
    for name, mask in (("pure_water", pure_water), ("bone_rich", bone_rich)):
        distances = distance_transform_edt(
            np.pad(mask, 1, constant_values=False), sampling=grid.spacing_mm[::-1]
        )[1:-1, 1:-1, 1:-1]
        masks[f"{name}_core"] = mask & (distances > CORE_DISTANCE_MM)
    return masks


def _errors(recovered: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {"status": "empty_roi", "rmse": None, "mae": None, "signed_bias": None}
    error = recovered[mask] - reference[mask]
    if not np.isfinite(error).all():
        raise ValueError("field differences exceed finite FP64 representation")
    scale = float(np.max(np.abs(error)))
    # Scale before squaring so a representable large RMSE does not overflow.
    rmse = scale * float(np.sqrt(np.mean((error / scale) ** 2))) if scale else 0.0
    return {
        "status": "defined",
        "rmse": rmse,
        "mae": float(np.sum(np.abs(error) / count)),
        "signed_bias": float(np.sum(error / count)),
    }


def _dice(recovered: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    target = (reference >= BONE_THRESHOLD) & mask
    estimate = (recovered >= BONE_THRESHOLD) & mask
    n_target, n_estimate = int(target.sum()), int(estimate.sum())
    denominator = n_target + n_estimate
    empty_roi = not mask.any()
    return {
        "threshold": BONE_THRESHOLD,
        "threshold_comparison": "FP64-promoted stored values >= decimal 0.325",
        "reference_positive_voxels": n_target,
        "recovered_positive_voxels": n_estimate,
        "intersection_voxels": int((target & estimate).sum()),
        "status": "empty_roi"
        if empty_roi
        else "both_masks_empty"
        if not denominator
        else "defined",
        "value": 2 * int((target & estimate).sum()) / denominator if denominator else None,
    }


def _coupling(errors: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {"status": "empty_roi", "population_covariance": None, "correlation": None}
    selected = errors[:, mask]
    # Shift by an actual sample first: constant errors then have exactly zero
    # variance even when averaging their absolute value would round differently.
    shifted = selected - selected[:, :1]
    centred = shifted - shifted.mean(axis=1, keepdims=True)
    covariance = centred @ centred.T / count
    if not np.isfinite(covariance).all():
        raise ValueError("error covariance exceeds finite FP64 representation")
    correlation: list[list[float | None]] = []
    for i in range(2):
        row: list[float | None] = []
        for j in range(2):
            if covariance[i, i] == 0 or covariance[j, j] == 0:
                row.append(None)
            else:
                # Separate square roots avoid overflowing a variance product.
                value = covariance[i, j] / np.sqrt(covariance[i, i]) / np.sqrt(covariance[j, j])
                row.append(float(np.clip(value, -1, 1)))
        correlation.append(row)
    return {
        "status": "defined",
        "population_covariance": covariance.tolist(),
        "covariance_units": "fraction^2",
        "correlation": correlation,
        "undefined_correlation": "None whenever either error variance is zero",
        "interpretation": "Descriptive centred error coupling; not a material-response matrix",
    }


def centre_axis_profiles(values: np.ndarray, grid: GridSpec) -> dict[str, Any]:
    """Transverse linear interpolation at grid centre; no longitudinal interpolation."""
    is_material = isinstance(cast(object, values), np.ndarray) and values.ndim == 4
    field = _field(values, grid, materials=is_material)
    centre = (np.array(grid.shape[::-1], dtype=np.float64) - 1) / 2
    profiles: dict[str, Any] = {}
    offset = int(is_material)
    for xyz_axis, name in enumerate("xyz"):
        varying_zyx_axis = 2 - xyz_axis
        line = field
        for zyx_axis in sorted(set(range(3)) - {varying_zyx_axis}, reverse=True):
            coordinate = (grid.shape[zyx_axis] - 1) / 2
            lo, hi = int(np.floor(coordinate)), int(np.ceil(coordinate))
            first = np.take(line, lo, axis=offset + zyx_axis)
            line = (
                first if lo == hi else 0.5 * first + 0.5 * np.take(line, hi, axis=offset + zyx_axis)
            )
        coordinates = np.repeat(centre[None, :], grid.shape[varying_zyx_axis], axis=0)
        coordinates[:, xyz_axis] = np.arange(grid.shape[varying_zyx_axis])
        profiles[name] = {
            "grid_coordinates_xyz": coordinates.tolist(),
            "object_coordinates_xyz_mm": [
                grid.grid_to_object((float(v[0]), float(v[1]), float(v[2]))) for v in coordinates
            ],
            "local_axis_distance_from_centre_mm": (
                (coordinates[:, xyz_axis] - centre[xyz_axis]) * grid.spacing_mm[xyz_axis]
            ).tolist(),
            "values": line.tolist(),
        }
    return profiles


def evaluate_scalar_field(
    recovered: np.ndarray,
    reference: np.ndarray,
    grid: GridSpec,
    *,
    rois: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Scalar attenuation summaries in mm^-1; supplied masks may come from reference_rois."""
    recovered = _field(recovered, grid)
    reference = _field(reference, grid)
    masks = _masks(rois, grid)
    voxel_volume = prod(grid.spacing_mm)
    return {
        "units": "mm^-1",
        "error_sign": "recovered-reference",
        "voxel_volume_mm3": voxel_volume,
        "rois": {
            name: {
                "voxel_count": int(mask.sum()),
                "roi_volume_mm3": int(mask.sum()) * voxel_volume,
                **_errors(recovered, reference, mask),
            }
            for name, mask in masks.items()
        },
        "centre_profiles": {
            "reference": centre_axis_profiles(reference, grid),
            "recovered": centre_axis_profiles(recovered, grid),
            "error": centre_axis_profiles(recovered - reference, grid),
        },
    }


def evaluate_material_fields(
    recovered: np.ndarray,
    reference: np.ndarray,
    grid: GridSpec,
    *,
    mu80_mm_inverse: tuple[float, float],
) -> dict[str, Any]:
    """Water/bone fraction metrics and a separately labelled derived 80keV attenuation.

    Material constraints and the provenance/units of the supplied coefficients
    are caller responsibilities. Empty ROI statistics and component volumes are
    None; a both-empty Dice is None, while one nonempty mask gives defined zero.
    """
    recovered = _field(recovered, grid, materials=True)
    reference = _field(reference, grid, materials=True)
    coefficients = np.asarray(mu80_mm_inverse, dtype=np.float64)
    if (
        coefficients.shape != (2,)
        or not np.isfinite(coefficients).all()
        or (coefficients <= 0).any()
    ):
        raise ValueError("mu80_mm_inverse must contain two finite positive water/bone coefficients")
    masks = reference_rois(reference, grid)
    voxel_volume = prod(grid.spacing_mm)
    errors = recovered - reference
    reports = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        components = {}
        for i, material in enumerate(("water", "bone")):
            components[material] = {
                **_errors(recovered[i], reference[i], mask),
                "reference_component_volume_mm3": (
                    float(reference[i][mask].sum() * voxel_volume) if count else None
                ),
                "recovered_component_volume_mm3": (
                    float(recovered[i][mask].sum() * voxel_volume) if count else None
                ),
            }
        reports[name] = {
            "voxel_count": count,
            "roi_volume_mm3": count * voxel_volume,
            "components": components,
            "bone_dice": _dice(recovered[1], reference[1], mask),
            "error_coupling": _coupling(errors, mask),
        }
    false_bone = {}
    for name in ("pure_water", "pure_water_core"):
        mask = masks[name]
        count = int(mask.sum())
        false_bone[name] = {
            "status": "defined" if count else "empty_roi",
            "mean_recovered_bone_fraction": float(recovered[1][mask].mean()) if count else None,
            "recovered_bone_component_volume_mm3": reports[name]["components"]["bone"][
                "recovered_component_volume_mm3"
            ],
            "threshold_positive_voxels": int(((recovered[1] >= BONE_THRESHOLD) & mask).sum()),
            "threshold_positive_fraction_of_roi": (
                float((recovered[1][mask] >= BONE_THRESHOLD).mean()) if count else None
            ),
            "bone_error": reports[name]["components"]["bone"],
        }
    return {
        "roi_rule_id": RULE_ID,
        "material_order": ["water", "bone"],
        "units": "fraction",
        "error_sign": "recovered-reference",
        "roi_overlap": "Not a partition; bone-rich/interface overlap and cores are subsets",
        "core_distance": (
            "Strictly >5 mm to nearest excluded voxel centre, including padded exterior"
        ),
        "voxel_volume_mm3": voxel_volume,
        "rois": reports,
        "false_bone_in_pure_water": false_bone,
        "water_bias_in_bone_rich": {
            name: reports[name]["components"]["water"]["signed_bias"]
            for name in ("bone_rich", "bone_rich_core")
        },
        "derived_mu80": {
            "interpretation": (
                "Derived attenuation under supplied fixed coefficients; separate errors"
            ),
            "coefficients_mm_inverse": coefficients.tolist(),
            **evaluate_scalar_field(
                np.einsum("m,mzyx->zyx", coefficients, recovered),
                np.einsum("m,mzyx->zyx", coefficients, reference),
                grid,
                rois=masks,
            ),
        },
        "centre_profiles": {
            "reference": centre_axis_profiles(reference, grid),
            "recovered": centre_axis_profiles(recovered, grid),
            "error": centre_axis_profiles(recovered - reference, grid),
        },
    }
