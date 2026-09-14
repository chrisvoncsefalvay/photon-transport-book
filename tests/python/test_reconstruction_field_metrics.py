"""Independent CPU algebra/geometry tests; these arrays are not scientific results."""

from __future__ import annotations

# Pytest approx has partial external annotations; numerical assertions remain explicit.
# pyright: reportUnknownMemberType=false
import importlib.util
import itertools
import json
import math
from pathlib import Path
from typing import Any

import pytest

from dpt.volumes import GridSpec

np: Any = pytest.importorskip(
    "numpy", reason="field metrics require the scientific CPU environment"
)
pytest.importorskip("scipy", reason="physical EDT metrics require SciPy")


@pytest.fixture(scope="module")
def metrics() -> Any:
    source = (
        Path(__file__).resolve().parents[2] / "experiments/reconstruction-study/evaluate_fields.py"
    )
    spec = importlib.util.spec_from_file_location("reconstruction_field_metrics_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scalar_algebra_roi_denominator_units_and_preservation(metrics: Any) -> None:
    grid = GridSpec((2, 2, 3), (2, 3, 5), (4, -7, 2))
    reference = np.arange(12, dtype=np.float64).reshape(grid.shape) / 16
    error = (
        np.array([-4, 2, 1, 0, 3, -1, 4, -2, 2, 1, -3, 1], dtype=np.float64).reshape(grid.shape)
        / 32
    )
    recovered = reference + error
    selected = np.zeros(grid.shape, dtype=bool)
    selected.ravel()[[0, 2, 4, 7]] = True
    empty = np.zeros_like(selected)
    before = (reference.tobytes(), recovered.tobytes(), selected.tobytes())
    result = metrics.evaluate_scalar_field(
        recovered, reference, grid, rois={"selected": selected, "empty": empty}
    )
    assert result["units"] == "mm^-1" and result["voxel_volume_mm3"] == 30
    for name, mask in (("whole_box", np.ones_like(selected)), ("selected", selected)):
        values = [float(error[index]) for index in np.ndindex(grid.shape) if mask[index]]
        r = result["rois"][name]
        assert r["voxel_count"] == len(values)
        assert r["roi_volume_mm3"] == len(values) * 30
        assert r["rmse"] == pytest.approx(math.sqrt(math.fsum(v * v for v in values) / len(values)))
        assert r["mae"] == pytest.approx(math.fsum(abs(v) for v in values) / len(values))
        assert r["signed_bias"] == pytest.approx(math.fsum(values) / len(values))
    assert result["rois"]["empty"]["rmse"] is None
    assert result["rois"]["empty"]["signed_bias"] is None
    assert before == (reference.tobytes(), recovered.tobytes(), selected.tobytes())
    json.dumps(result, allow_nan=False)


def test_exact_threshold_neighbours_and_roi_overlap(metrics: Any) -> None:
    upper = float(np.float32(0.65)) - 1e-7
    bone = np.array(
        [
            0.0,
            1e-7,
            np.nextafter(1e-7, np.inf),
            float(np.float32(0.325)),
            np.nextafter(0.325, -np.inf),
            0.325,
            np.nextafter(0.325, np.inf),
            np.nextafter(upper, -np.inf),
            upper,
            np.nextafter(upper, np.inf),
            float(np.float32(0.65)),
            0.4,
        ]
    ).reshape(1, 3, 4)
    reference = np.stack([1 - bone, bone])
    rois = metrics.reference_rois(reference, GridSpec(bone.shape, (1, 1, 1)))
    assert np.flatnonzero(rois["pure_water"]).tolist() == [0, 1]
    assert np.flatnonzero(rois["bone_rich"]).tolist() == [5, 6, 7, 8, 9, 10, 11]
    assert np.flatnonzero(rois["partial_volume_interface"]).tolist() == [2, 3, 4, 5, 6, 7, 11]
    assert int((rois["bone_rich"] & rois["partial_volume_interface"]).sum()) == 4
    assert not rois["pure_water_core"].any() and not rois["bone_rich_core"].any()
    # The nominal midpoint stored as FP32 is below decimal0.325 after FP64 promotion.
    stored = np.array(
        [
            np.nextafter(np.float32(0.325), np.float32(-np.inf)),
            np.float32(0.325),
            np.nextafter(np.float32(0.325), np.float32(np.inf)),
        ],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    r = metrics.evaluate_material_fields(
        np.stack([1 - stored, stored]),
        np.stack([1 - stored, stored]),
        GridSpec(stored.shape, (1, 1, 1)),
        mu80_mm_inverse=(0.02, 0.04),
    )
    assert r["rois"]["whole_box"]["bone_dice"]["reference_positive_voxels"] == 1
    assert r["rois"]["whole_box"]["bone_dice"]["value"] == 1


def _brute_core(mask: Any, grid: GridSpec) -> Any:
    # Enumerate excluded centres in the single padded layer and all internal holes.
    excluded: list[list[float]] = []
    rotation = np.array(grid.orientation).reshape(3, 3)
    spacing = np.array(grid.spacing_mm)
    origin = np.array(grid.origin_mm)
    for z, y, x in itertools.product(*(range(-1, n + 1) for n in grid.shape)):
        inside = 0 <= z < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= x < grid.shape[2]
        if not inside or not mask[z, y, x]:
            excluded.append((rotation @ (np.array([x, y, z]) * spacing) + origin).tolist())
    result = np.zeros_like(mask)
    for z, y, x in np.ndindex(grid.shape):
        if mask[z, y, x]:
            point = rotation @ (np.array([x, y, z]) * spacing) + origin
            result[z, y, x] = min(math.dist(point, other) for other in excluded) > 5
    return result


@pytest.mark.parametrize("bone_fraction", [0.0, 0.65])
def test_padded_edt_anisotropic_rotated_grid_strict_boundary(
    metrics: Any, bone_fraction: float
) -> None:
    # A 90-degree proper rotation is exact; brute distances test physical XYZ independently.
    grid = GridSpec((3, 5, 7), (2, 3, 5), (13, -9, 4), (0, -1, 0, 1, 0, 0, 0, 0, 1))
    bone = np.full(grid.shape, bone_fraction)
    reference = np.stack([1 - bone, bone])
    result = metrics.reference_rois(reference, grid)
    parent = "pure_water" if bone_fraction == 0 else "bone_rich"
    np.testing.assert_array_equal(result[parent + "_core"], _brute_core(result[parent], grid))
    assert result[parent + "_core"][1, 2, 3]
    # First Z plane is exactly5mm from padded excluded centres, hence not in the core.
    assert not result[parent + "_core"][0].any()
    # An interior excluded voxel adds a physical centre boundary, even away from box faces.
    reference[:, 1, 2, 3] = [0, 0]
    revised = metrics.reference_rois(reference, grid)
    np.testing.assert_array_equal(revised[parent + "_core"], _brute_core(revised[parent], grid))
    assert not revised[parent + "_core"][1, 2, 3]


def test_material_volumes_bias_false_bone_derived_mu_and_covariance(metrics: Any) -> None:
    grid = GridSpec((2, 2, 3), (2, 3, 5))
    bone = np.array(
        [0, 0, 0.25, 0.5, 0.65, 0.65, 0, 0.1, 0.4, 0.5, 0.6, 0], dtype=np.float64
    ).reshape(grid.shape)
    reference = np.stack([1 - bone, bone])
    e = np.arange(12, dtype=np.float64).reshape(grid.shape) / 128
    recovered = reference + np.stack([e, -2 * e + 0.125])
    before = (reference.tobytes(), recovered.tobytes())
    result = metrics.evaluate_material_fields(
        recovered, reference, grid, mu80_mm_inverse=(0.02, 0.05)
    )
    masks = metrics.reference_rois(reference, grid)
    for name, mask in masks.items():
        selected = [index for index in np.ndindex(grid.shape) if mask[index]]
        report = result["rois"][name]
        for component, material in enumerate(["water", "bone"]):
            r = report["components"][material]
            if not selected:
                assert r["rmse"] is None and r["reference_component_volume_mm3"] is None
                continue
            actual = [
                float(recovered[(component, *i)] - reference[(component, *i)]) for i in selected
            ]
            assert r["rmse"] == pytest.approx(
                math.sqrt(math.fsum(v * v for v in actual) / len(selected))
            )
            assert r["mae"] == pytest.approx(math.fsum(abs(v) for v in actual) / len(selected))
            assert r["signed_bias"] == pytest.approx(math.fsum(actual) / len(selected))
            assert r["reference_component_volume_mm3"] == pytest.approx(
                math.fsum(float(reference[(component, *i)]) for i in selected) * 30
            )
            assert r["recovered_component_volume_mm3"] == pytest.approx(
                math.fsum(float(recovered[(component, *i)]) for i in selected) * 30
            )
        if not selected:
            assert report["error_coupling"]["population_covariance"] is None
            assert report["bone_dice"]["value"] is None
            continue
        errors = [
            [float(recovered[(c, *i)] - reference[(c, *i)]) for i in selected] for c in range(2)
        ]
        means = [math.fsum(v) / len(v) for v in errors]
        covariance = [
            [
                math.fsum(
                    (a - means[c]) * (b - means[d])
                    for a, b in zip(errors[c], errors[d], strict=True)
                )
                / len(selected)
                for d in range(2)
            ]
            for c in range(2)
        ]
        np.testing.assert_allclose(
            report["error_coupling"]["population_covariance"], covariance, rtol=1e-14, atol=1e-17
        )
        if covariance[0][0] > 0:
            assert report["error_coupling"]["correlation"][0][1] == pytest.approx(-1)
    pure = masks["pure_water"]
    rich = masks["bone_rich"]
    assert result["false_bone_in_pure_water"]["pure_water"][
        "mean_recovered_bone_fraction"
    ] == pytest.approx(math.fsum(recovered[1][pure]) / int(pure.sum()))
    assert result["water_bias_in_bone_rich"]["bone_rich"] == pytest.approx(
        math.fsum(e[rich]) / int(rich.sum())
    )
    attenuation_error = 0.02 * (recovered[0] - reference[0]) + 0.05 * (recovered[1] - reference[1])
    assert result["derived_mu80"]["units"] == "mm^-1"
    assert result["derived_mu80"]["rois"]["whole_box"]["rmse"] == pytest.approx(
        math.sqrt(math.fsum(v * v for v in attenuation_error.ravel()) / grid.voxels)
    )
    assert (
        result["derived_mu80"]["rois"]["whole_box"]["rmse"]
        != result["rois"]["whole_box"]["components"]["bone"]["rmse"]
    )
    assert before == (reference.tobytes(), recovered.tobytes())
    json.dumps(result, allow_nan=False)


def test_empty_dice_one_sided_dice_and_zero_variance_are_explicit(metrics: Any) -> None:
    grid = GridSpec((1, 2, 2), (2, 3, 5))
    reference = np.stack([np.ones(grid.shape), np.zeros(grid.shape)])
    zero = metrics.evaluate_material_fields(
        reference, reference, grid, mu80_mm_inverse=(0.02, 0.04)
    )
    whole = zero["rois"]["whole_box"]
    assert (
        whole["bone_dice"]["value"] is None and whole["bone_dice"]["status"] == "both_masks_empty"
    )
    assert whole["error_coupling"]["population_covariance"] == [[0, 0], [0, 0]]
    assert whole["error_coupling"]["correlation"] == [[None, None], [None, None]]
    assert zero["rois"]["bone_rich"]["components"]["water"]["rmse"] is None
    assert zero["water_bias_in_bone_rich"]["bone_rich"] is None
    recovered = reference.copy()
    recovered[1, 0, 0, 0] = 0.5
    one = metrics.evaluate_material_fields(recovered, reference, grid, mu80_mm_inverse=(0.02, 0.04))
    assert one["rois"]["whole_box"]["bone_dice"]["value"] == 0
    assert one["rois"]["whole_box"]["bone_dice"]["status"] == "defined"
    assert one["rois"]["whole_box"]["error_coupling"]["correlation"][0] == [None, None]
    assert one["rois"]["whole_box"]["error_coupling"]["correlation"][1][1] == pytest.approx(1)
    assert one["false_bone_in_pure_water"]["pure_water"]["threshold_positive_voxels"] == 1
    assert (
        one["false_bone_in_pure_water"]["pure_water"]["threshold_positive_fraction_of_roi"] == 0.25
    )


def test_centre_profiles_interpolate_even_axes_and_report_rotated_object_coordinates(
    metrics: Any,
) -> None:
    grid = GridSpec((3, 4, 6), (2, 3, 5), (17, -11, 6), (0, -0.8, 0.6, 1, 0, 0, 0, 0.6, 0.8))

    def formula(x: float, y: float, z: float) -> float:
        return 2 * x + 3 * y - 4 * z + 0.5 * x * y + 0.125 * x * z

    values = np.empty(grid.shape)
    for z, y, x in np.ndindex(grid.shape):
        values[z, y, x] = formula(x, y, z)
    centre = [2.5, 1.5, 1.0]
    for data in (values, np.stack([values, 2 * values + 1])):
        profiles = metrics.centre_axis_profiles(data, grid)
        for axis, name in enumerate("xyz"):
            coordinates = profiles[name]["grid_coordinates_xyz"]
            for i, index in enumerate(coordinates):
                expected_index = centre.copy()
                expected_index[axis] = float(i)
                assert index == expected_index
                x, y, z = index
                # Independent explicit matrix multiplication in the object frame.
                expected_object = [
                    17 - 0.8 * (y * 3) + 0.6 * (z * 5),
                    -11 + x * 2,
                    6 + 0.6 * (y * 3) + 0.8 * (z * 5),
                ]
                np.testing.assert_allclose(
                    profiles[name]["object_coordinates_xyz_mm"][i],
                    expected_object,
                    atol=1e-14,
                    rtol=0,
                )
                expected_value = formula(x, y, z)
                if data.ndim == 3:
                    assert profiles[name]["values"][i] == expected_value
                else:
                    assert profiles[name]["values"][0][i] == expected_value
                    assert profiles[name]["values"][1][i] == 2 * expected_value + 1
                assert (
                    profiles[name]["local_axis_distance_from_centre_mm"][i]
                    == (i - centre[axis]) * grid.spacing_mm[axis]
                )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_rejected_even_outside_custom_roi(metrics: Any, invalid: float) -> None:
    grid = GridSpec((1, 2, 2), (1, 1, 1))
    ref = np.zeros(grid.shape)
    rec = ref.copy()
    rec[0, 0, 0] = invalid
    mask = np.ones(grid.shape, dtype=bool)
    mask[0, 0, 0] = False
    with pytest.raises(ValueError, match="finite"):
        metrics.evaluate_scalar_field(rec, ref, grid, rois={"selected": mask})


def test_shape_mask_type_and_coefficients_are_checked(metrics: Any) -> None:
    grid = GridSpec((1, 2, 2), (1, 1, 1))
    ref = np.zeros(grid.shape)
    for bad in (
        np.zeros((2, 2, 2)),
        np.zeros(grid.shape, dtype=complex),
        np.zeros(grid.shape, dtype=bool),
    ):
        with pytest.raises(ValueError):
            metrics.evaluate_scalar_field(bad, ref, grid)
    for mask in (np.ones(grid.shape, dtype=np.uint8), np.ones((2, 2, 2), dtype=bool)):
        with pytest.raises(ValueError):
            metrics.evaluate_scalar_field(ref, ref, grid, rois={"bad": mask})
    with pytest.raises(ValueError):
        metrics.evaluate_scalar_field(
            ref, ref, grid, rois={"whole_box": np.zeros(grid.shape, dtype=bool)}
        )
    material = np.stack([ref, ref])
    for coefficients in ((0.02,), (0.02, 0), (0.02, -1), (0.02, np.inf)):
        with pytest.raises(ValueError):
            metrics.evaluate_material_fields(material, material, grid, mu80_mm_inverse=coefficients)


def test_singleton_profiles_and_large_representable_rmse(metrics: Any) -> None:
    grid = GridSpec((1, 1, 1), (2, 3, 5), (4, 6, 8))
    result = metrics.evaluate_scalar_field(np.full(grid.shape, 1e200), np.zeros(grid.shape), grid)
    assert result["rois"]["whole_box"]["rmse"] == 1e200
    for axis in "xyz":
        assert result["centre_profiles"]["recovered"][axis]["values"] == [1e200]
        assert result["centre_profiles"]["recovered"][axis]["object_coordinates_xyz_mm"] == [
            (4.0, 6.0, 8.0)
        ]


def test_nonbinary_constant_errors_have_exact_zero_covariance(metrics: Any) -> None:
    grid = GridSpec((1, 1, 3), (2, 3, 5))
    reference = np.stack([np.ones(grid.shape), np.zeros(grid.shape)])
    recovered = reference + 0.1
    result = metrics.evaluate_material_fields(
        recovered, reference, grid, mu80_mm_inverse=(0.02, 0.04)
    )
    coupling = result["rois"]["whole_box"]["error_coupling"]
    assert coupling["population_covariance"] == [[0, 0], [0, 0]]
    assert coupling["correlation"] == [[None, None], [None, None]]
    for axis in "xyz":
        np.testing.assert_array_equal(
            result["centre_profiles"]["error"][axis]["values"],
            metrics.centre_axis_profiles(recovered - reference, grid)[axis]["values"],
        )


def test_transverse_average_of_extreme_finite_values_remains_finite(metrics: Any) -> None:
    grid = GridSpec((1, 2, 2), (2, 3, 5))
    source = np.full(grid.shape, 1e308)
    profiles = metrics.centre_axis_profiles(source, grid)
    for axis in "xyz":
        assert all(value == 1e308 for value in profiles[axis]["values"])
