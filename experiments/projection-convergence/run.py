"""Record separate grid and quadrature sweeps from the canonical CUDA projector.

This driver creates a labelled analytic field, never anatomy. It writes actual
executions and independent reference discrepancies; no plots or pass claims are
manufactured from configured tolerances.
"""

from __future__ import annotations

import importlib
import json
import math
from typing import Any, cast

from dpt.contracts import integer
from dpt.experiments import (
    RunRecorder,
    experiment_parser,
    experiment_sources,
    private_output,
    repository_root,
)
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth
from dpt.validation.projection import (
    integrate_quadratic_field,
    integrate_sampled_field,
    quadratic_sample_values,
)
from dpt.volumes import GridSpec

ROOT = repository_root(__file__)


def _counts(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    values = cast(list[object], value)
    return tuple(integer(item, name, minimum=1, maximum=4096) for item in values)


def main() -> None:
    parser = experiment_parser(__file__, __doc__)
    arguments = parser.parse_args()
    config = cast(dict[str, object], json.loads(arguments.config.read_text()))
    if (
        set(config)
        != {
            "schema_version",
            "grid_resolutions",
            "quadrature_samples",
            "quadrature_grid",
            "grid_quadrature",
        }
        or config["schema_version"] != 1
    ):
        raise ValueError("unsupported convergence configuration")
    grids = _counts(config["grid_resolutions"], "grid_resolutions")
    samples = _counts(config["quadrature_samples"], "quadrature_samples")
    fixed_grid = integer(config["quadrature_grid"], "quadrature_grid", minimum=1, maximum=1024)
    fixed_quadrature = integer(config["grid_quadrature"], "grid_quadrature", minimum=1)
    target = private_output(arguments.output, ROOT)
    sources = experiment_sources(__file__, arguments.config)
    with RunRecorder(target, configuration=config, sources=sources) as record:
        wp: Any = importlib.import_module("warp")
        device = wp.get_device(arguments.device)
        if not device.is_cuda:
            raise ValueError("this experiment requires CUDA")
        record.set_metadata(
            warp=wp.__version__,
            device=str(device),
            device_name=device.name,
            architecture=device.arch,
            cuda_driver=device.runtime.driver_version,
            precision="FP32 storage; FP64 coordinates and accumulation",
            fixture="analytic quadratic attenuation; 20 mm cubic support",
        )
        geometry = DetectorGeometry(
            (-80.0, 1.0, 2.0),
            (80.0, -12.0, -10.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (4.0, 4.0),
            (7, 7),
        )
        identity = RigidTransform()
        pose = wp.array(identity.packed(), dtype=wp.float64, device=device)
        output = wp.empty(geometry.pixels, dtype=wp.float32, device=device)
        rows: list[dict[str, object]] = []
        cases = [("quadrature", fixed_grid, count) for count in samples]
        cases.extend(("grid", resolution, fixed_quadrature) for resolution in grids)
        for sweep, resolution, sample_count in cases:
            spacing = 20.0 / resolution
            grid = GridSpec(
                (resolution, resolution, resolution),
                (spacing, spacing, spacing),
                (-10.0 + spacing / 2, -10.0 + spacing / 2, -10.0 + spacing / 2),
            )
            values = quadratic_sample_values(
                grid, intercept=0.01, curvature=(0.00003, 0.00005, 0.00002)
            )
            field = wp.array(values, dtype=wp.float32, device=device)
            # Exact-input reference sees the FP32 field actually uploaded.
            stored_values = [float(value) for value in field.numpy()]
            workspace = prepare_projection(
                grid, geometry, ProjectionSpec(sample_count), device=arguments.device
            )
            project_optical_depth(field, pose, workspace=workspace, out_L=output)
            actual = [float(value) for value in output.numpy()]
            sampled = [
                integrate_sampled_field(stored_values, grid, geometry, identity, row, column)
                for row in range(geometry.shape[0])
                for column in range(geometry.shape[1])
            ]
            continuous = [
                integrate_quadratic_field(
                    grid,
                    geometry,
                    identity,
                    row,
                    column,
                    intercept=0.01,
                    curvature=(0.00003, 0.00005, 0.00002),
                )
                for row in range(geometry.shape[0])
                for column in range(geometry.shape[1])
            ]
            rows.append(
                {
                    "sweep": sweep,
                    "resolution": resolution,
                    "samples_per_ray": sample_count,
                    "optical_depth": actual,
                    "exact_sampled_integral": sampled,
                    "continuous_integral": continuous,
                    "max_quadrature_absolute_error": max(
                        abs(a - b) for a, b in zip(actual, sampled, strict=True)
                    ),
                    "max_grid_absolute_error": max(
                        abs(a - b) for a, b in zip(sampled, continuous, strict=True)
                    ),
                    "rms_total_error": math.sqrt(
                        math.fsum((a - b) ** 2 for a, b in zip(actual, continuous, strict=True))
                        / geometry.pixels
                    ),
                }
            )
            record.write_json(f"cases/{len(rows) - 1:04d}.json", rows[-1])
        record.write_json("convergence.json", rows)


if __name__ == "__main__":
    main()
