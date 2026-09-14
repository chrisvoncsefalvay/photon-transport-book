"""Explicit supplied-data ingestion shared by spectral experiment drivers.

JSON records frames, units, array roles and provenance; NPZ stores only numeric
arrays. Object arrays/pickles, implicit precision changes and omitted physical
metadata are rejected. This preparation boundary uploads once and synchronises
before releasing the source host arrays. No physical data are bundled here.
"""

from __future__ import annotations

# JSON input still requires runtime type checks despite annotated helper calls.
# pyright: reportUnnecessaryIsInstance=false
import importlib
from pathlib import Path
from typing import Any

from dpt._runtime import prepare_context
from dpt.contracts import ContractError, integer
from dpt.detector import BlurSpec
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.materials import Provenance, validate_energy_grid
from dpt.objectives import ObjectiveSpec
from dpt.recovery import PoseChart
from dpt.spectral import SpectralSpec
from dpt.spectral_recovery import CalibrationBlock, SpectralPoseProblem, SpectralView
from dpt.volumes import GridSpec


def _keys(value: dict[str, Any], allowed: set[str], required: set[str], where: str) -> None:
    unknown, missing = set(value) - allowed, required - set(value)
    if unknown or missing:
        raise ContractError(
            f"{where}: unknown keys {sorted(unknown)}; missing keys {sorted(missing)}"
        )


def _provenance(value: dict[str, Any]) -> Provenance:
    return Provenance(**value)


def read_spectral_inputs(
    archive: Path,
    metadata: dict[str, Any],
    *,
    samples_per_ray: int,
    device: str = "cuda:0",
    stream: Any = None,
) -> tuple[SpectralPoseProblem, PoseChart]:
    """Load explicitly described material fractions and actual observation arrays.

    The returned problem owns CUDA arrays. Original NPZ bytes and their digest
    remain the experiment runner's provenance responsibility. ``metadata`` is
    versioned and recorded verbatim by that runner; see the spectral-projection
    experiment README for its complete schema and supported observation domains.
    """
    _keys(
        metadata,
        {
            "schema_version",
            "grid",
            "materials",
            "fields",
            "fields_provenance",
            "views",
            "calibration",
            "pose_chart",
        },
        {
            "schema_version",
            "grid",
            "materials",
            "fields",
            "fields_provenance",
            "views",
            "calibration",
        },
        "model metadata",
    )
    if metadata["schema_version"] != 1:
        raise ContractError("spectral input metadata requires schema_version=1")
    raw_grid = dict(metadata["grid"])
    for name in ("shape", "spacing_mm", "origin_mm", "orientation"):
        if name in raw_grid:
            raw_grid[name] = tuple(raw_grid[name])
    grid = GridSpec(**raw_grid)
    materials = integer(metadata["materials"], "materials", minimum=1, maximum=32)
    blocks = tuple(CalibrationBlock(**value) for value in metadata["calibration"])
    chart_values = dict(metadata.get("pose_chart", {}))
    if "anchor" in chart_values:
        anchor = dict(chart_values["anchor"])
        for name in ("rotation", "translation_mm"):
            if name in anchor:
                anchor[name] = tuple(anchor[name])
        chart_values["anchor"] = RigidTransform(**anchor)
    if "scales" in chart_values:
        chart_values["scales"] = tuple(chart_values["scales"])
    chart = PoseChart(**chart_values)
    ctx = prepare_context(device=device, stream=stream)
    np: Any = importlib.import_module("numpy")
    uploads: dict[str, tuple[tuple[int, ...], Any]] = {}
    source_arrays: list[Any] = []
    with np.load(archive, allow_pickle=False) as arrays, ctx.scope():

        def upload(name: str, shapes: tuple[tuple[int, ...], ...]) -> Any:
            if not isinstance(name, str) or not name:
                raise ContractError("array roles must name a non-empty NPZ member")
            if name in uploads:
                actual_shape, result = uploads[name]
                if actual_shape not in shapes:
                    raise ContractError(
                        f"NPZ array {name} was bound with an incompatible role shape"
                    )
                return result
            if name not in arrays:
                raise ContractError(f"NPZ archive lacks required array {name}")
            value = arrays[name]
            if value.dtype != np.dtype("float32") or not value.flags.c_contiguous:
                raise ContractError(
                    f"{name} must already be contiguous native binary32; convert explicitly"
                )
            if tuple(value.shape) not in shapes:
                raise ContractError(f"{name} has shape {value.shape}; expected one of {shapes}")
            source_arrays.append(value)
            result = ctx.wp.array(value.reshape(-1), dtype=ctx.wp.float32, device=ctx.device)
            uploads[name] = (tuple(value.shape), result)
            return result

        try:
            fields = upload(
                metadata["fields"],
                ((materials, *grid.shape), (materials, grid.voxels), (materials * grid.voxels,)),
            )
            views: list[SpectralView] = []
            for raw in metadata["views"]:
                _keys(
                    raw,
                    {
                        "name",
                        "geometry",
                        "energies_kev",
                        "coefficients",
                        "coefficients_unit",
                        "coefficients_provenance",
                        "weights",
                        "spectrum_provenance",
                        "response",
                        "response_provenance",
                        "shared_weights",
                        "shared_response",
                        "output_unit",
                        "input_description",
                        "observation",
                        "observation_provenance",
                        "calibration_group",
                        "objective",
                        "objective_weights",
                        "objective_weight",
                        "spatial_response",
                    },
                    {
                        "name",
                        "geometry",
                        "energies_kev",
                        "coefficients",
                        "coefficients_unit",
                        "coefficients_provenance",
                        "weights",
                        "spectrum_provenance",
                        "response",
                        "response_provenance",
                        "output_unit",
                        "input_description",
                        "observation",
                        "observation_provenance",
                        "calibration_group",
                        "objective",
                    },
                    f"view {raw.get('name', '')}",
                )
                if raw["coefficients_unit"] != "mm^-1":
                    raise ContractError(
                        "NPZ coefficients must already be total attenuation in mm^-1"
                    )
                raw_geometry = {name: tuple(value) for name, value in raw["geometry"].items()}
                geometry = DetectorGeometry(**raw_geometry)
                energies = tuple(raw["energies_kev"])
                validate_energy_grid(energies)
                spec = SpectralSpec(
                    materials,
                    len(energies),
                    tuple(_provenance(p) for p in raw["coefficients_provenance"]),
                    _provenance(raw["spectrum_provenance"]),
                    _provenance(raw["response_provenance"]),
                    raw["output_unit"],
                    raw["input_description"],
                    shared_weights=raw.get("shared_weights", True),
                    shared_response=raw.get("shared_response", True),
                )
                count, channels = geometry.pixels, spec.energies
                field_shapes = ((channels, *geometry.shape), (channels, count), (channels * count,))
                coefficients = upload(
                    raw["coefficients"], ((materials, channels), (materials * channels,))
                )
                weights = upload(
                    raw["weights"], ((channels,),) if spec.shared_weights else field_shapes
                )
                response = upload(
                    raw["response"], ((channels,),) if spec.shared_response else field_shapes
                )
                observation = upload(raw["observation"], (geometry.shape, (count,)))
                objective = ObjectiveSpec(**raw["objective"])
                objective_weights = (
                    upload(raw["objective_weights"], (geometry.shape, (count,)))
                    if raw.get("objective_weights") is not None
                    else None
                )
                spatial = None
                if raw.get("spatial_response") is not None:
                    values = dict(raw["spatial_response"])
                    values["weights"] = tuple(values["weights"])
                    values["provenance"] = _provenance(values["provenance"])
                    spatial = BlurSpec(geometry.shape[0], geometry.shape[1], **values)
                views.append(
                    SpectralView(
                        raw["name"],
                        geometry,
                        spec,
                        coefficients,
                        weights,
                        response,
                        observation,
                        raw["calibration_group"],
                        _provenance(raw["observation_provenance"]),
                        objective,
                        objective_weights,
                        raw.get("objective_weight", 1.0),
                        spatial,
                    )
                )
            return SpectralPoseProblem(
                grid,
                materials,
                fields,
                _provenance(metadata["fields_provenance"]),
                tuple(views),
                blocks,
                samples_per_ray,
            ), chart
        finally:
            # Source NumPy allocations must outlive every asynchronous upload.
            ctx.wp.synchronize_stream(ctx.stream)
