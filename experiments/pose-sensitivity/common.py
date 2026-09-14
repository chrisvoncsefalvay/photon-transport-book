"""One prepared canonical CUDA composition used by the recorded figure driver."""

from __future__ import annotations

import numpy as np
import warp as wp

from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
from dpt.projection import (
    ProjectionSpec,
    prepare_projection,
    project_optical_depth,
    projection_pose_sensitivities,
)
from dpt.transmission import (
    TransmissionSpec,
    prepare_transmission,
    transmission_vjp,
    transmit,
)
from dpt.volumes import GridSpec

AXES = ("tx", "ty", "tz", "rx", "ry", "rz")


class Scene:
    """Keep the field, pose and work buffers resident; explicitly download records."""

    def __init__(self, values: np.ndarray, volume: dict, config: dict, device: str):
        self.wp_device = wp.get_device(device)
        if not self.wp_device.is_cuda:
            raise ValueError("figure data must be generated on CUDA")
        grid = volume["grid"]
        self.grid = GridSpec(
            tuple(grid["shape"]),
            tuple(grid["spacing_mm"]),
            tuple(grid["origin_mm"]),
            tuple(grid["orientation"]),
        )
        h, w = config["height"], config["width"]
        du, dv = config["detector_width_mm"] / w, config["detector_height_mm"] / h
        centre = config["detector_centre_mm"]
        self.geometry = DetectorGeometry(
            tuple(config["source_mm"]),
            (centre[0] - (w - 1) * du / 2, centre[1], centre[2] + (h - 1) * dv / 2),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (du, dv),
            (h, w),
        )
        self.n0 = float(config["n0"])
        self.stream = wp.get_stream(self.wp_device)
        self.field = wp.array(values.ravel(), dtype=wp.float32, device=device)
        self.pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
        self.host_pose = wp.empty(12, dtype=wp.float64, device="cpu", pinned=True)
        self.host_pose_view = self.host_pose.numpy()
        p = self.geometry.pixels
        self.depth = wp.empty(p, dtype=wp.float32, device=device)
        self.counts = wp.empty(p, dtype=wp.float32, device=device)
        self.seeds = wp.empty(p, dtype=wp.float32, device=device)
        self.ones = wp.ones(p, dtype=wp.float32, device=device)
        self.jacobian = wp.empty(6 * p, dtype=wp.float64, device=device)
        self.transmission = prepare_transmission(
            TransmissionSpec(beam="scalar"), device=device, max_pixels=p, stream=self.stream
        )
        self.workspace = None
        self.set_samples(config["samples_per_ray"])

    def set_samples(self, samples: int) -> None:
        self.workspace = prepare_projection(
            self.grid,
            self.geometry,
            ProjectionSpec(samples),
            device=str(self.wp_device),
            stream=self.stream,
        )

    def set_pose(self, increment: tuple[float, ...]) -> RigidTransform:
        pose = compose_pose(RigidTransform(), increment)
        self.host_pose_view[:] = pose.packed()
        wp.copy(self.pose, self.host_pose, stream=self.stream)
        return pose

    def forward(self, increment: tuple[float, ...] = (0.0,) * 6) -> np.ndarray:
        self.set_pose(increment)
        project_optical_depth(self.field, self.pose, out_L=self.depth, workspace=self.workspace)
        transmit(self.depth, self.n0, out_counts=self.counts, workspace=self.transmission)
        wp.synchronize_stream(self.stream)
        return self.counts.numpy().reshape(self.geometry.shape).copy()

    def derivatives(self) -> np.ndarray:
        transmission_vjp(
            self.depth,
            self.n0,
            seed_counts=self.ones,
            out_grad_L=self.seeds,
            workspace=self.transmission,
        )
        projection_pose_sensitivities(
            self.field, self.pose, self.seeds, out_jacobian=self.jacobian, workspace=self.workspace
        )
        wp.synchronize_stream(self.stream)
        return self.jacobian.numpy().reshape((*self.geometry.shape, 6)).copy()
