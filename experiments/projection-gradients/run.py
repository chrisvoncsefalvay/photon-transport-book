"""Record six-coordinate CUDA pose VJPs against independent directional sweeps.

Both the GPU-forward finite differences and the independent FP64 discrete
reference are reported. Their different roundoff floors must not be confused
with proof that every geometry configuration is smooth or well conditioned.
"""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import asdict
from typing import Any, cast

from dpt.contracts import finite_scalar, integer
from dpt.experiments import (
    RunRecorder,
    experiment_parser,
    experiment_sources,
    private_output,
    repository_root,
)
from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.validation.autodiff import directional_sweep
from dpt.validation.projection import integrate_sampled_field, quadratic_sample_values
from dpt.volumes import GridSpec

ROOT = repository_root(__file__)


def _positive(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    values = cast(list[object], value)
    result = tuple(finite_scalar(item, name, minimum=0.0) for item in values)
    if any(item == 0 for item in result):
        raise ValueError(f"{name} must be strictly positive")
    return result


def main() -> None:
    parser = experiment_parser(__file__, __doc__)
    arguments = parser.parse_args()
    config = cast(dict[str, object], json.loads(arguments.config.read_text()))
    if (
        set(config)
        != {"schema_version", "grid_resolution", "samples_per_ray", "steps", "coordinate_scales"}
        or config["schema_version"] != 1
    ):
        raise ValueError("unsupported gradient configuration")
    resolution = integer(config["grid_resolution"], "grid_resolution", minimum=1, maximum=1024)
    samples = integer(config["samples_per_ray"], "samples_per_ray", minimum=1)
    steps = _positive(config["steps"], "steps")
    scales = _positive(config["coordinate_scales"], "coordinate_scales")
    if len(scales) != 6:
        raise ValueError("coordinate_scales must contain three mm and three radian values")
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
            precision="FP32 field/output; FP64 pose VJP and CPU reference",
            derivative="local right SE3, tx ty tz rx ry rz, mm/rad",
            fixture="analytic quadratic attenuation; 20 mm cubic support",
        )
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
        stored_values = [float(value) for value in field.numpy()]
        geometry = DetectorGeometry(
            (-80.0, 1.0, 2.0),
            (80.0, -12.0, -10.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (4.0, 4.0),
            (7, 7),
        )
        anchor = compose_pose(RigidTransform(), (0.23, -0.14, 0.17, 0.03, -0.02, 0.01))
        pose = wp.array(anchor.packed(), dtype=wp.float64, device=device)
        output = wp.empty(geometry.pixels, dtype=wp.float32, device=device)
        seed_values = [
            (-1.0) ** pixel * (0.5 + (pixel % 7) / 8) for pixel in range(geometry.pixels)
        ]
        seed = wp.array(seed_values, dtype=wp.float32, device=device)
        gradient = wp.empty(6, dtype=wp.float64, device=device)
        workspace = prepare_projection(
            grid, geometry, ProjectionSpec(samples), device=arguments.device
        )
        projection_vjp(field, pose, adj_L=seed, out_pose=gradient, workspace=workspace)
        measured = tuple(float(value) for value in gradient.numpy())
        host_pose = wp.empty(12, dtype=wp.float64, device="cpu", pinned=True)
        host_view = host_pose.numpy()

        def gpu_value(increment: tuple[float, ...]) -> float:
            host_view[:] = compose_pose(anchor, increment).packed()
            wp.copy(pose, host_pose, stream=workspace.stream)
            project_optical_depth(field, pose, out_L=output, workspace=workspace)
            return math.fsum(
                float(value) * seed_values[index] for index, value in enumerate(output.numpy())
            )

        def reference_value(increment: tuple[float, ...]) -> float:
            transform = compose_pose(anchor, increment)
            return math.fsum(
                seed_values[row * geometry.shape[1] + column]
                * integrate_sampled_field(
                    stored_values, grid, geometry, transform, row, column, midpoint_samples=samples
                )
                for row in range(geometry.shape[0])
                for column in range(geometry.shape[1])
            )

        rows: list[dict[str, object]] = []
        for axis, name in enumerate(("tx_mm", "ty_mm", "tz_mm", "rx_rad", "ry_rad", "rz_rad")):
            direction = tuple(float(index == axis) for index in range(6))
            independent = directional_sweep(
                reference_value, (0.0,) * 6, measured, direction, steps, scales=scales
            )
            actual = directional_sweep(
                gpu_value, (0.0,) * 6, measured, direction, steps, scales=scales
            )
            rows.append(
                {
                    "coordinate": name,
                    "cuda_vjp": measured[axis],
                    "independent_discrete_reference": [asdict(item) for item in independent],
                    "cuda_forward_differences": [asdict(item) for item in actual],
                }
            )
            record.write_json(f"coordinates/{name}.json", rows[-1])
        record.write_json("gradients.json", rows)


if __name__ == "__main__":
    main()
