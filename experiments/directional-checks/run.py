"""Record canonical directional checks on a classified trilinear interior knot."""

from __future__ import annotations

import json

import numpy as np
import warp as wp

from dpt.experiments import (
    RunRecorder,
    experiment_parser,
    experiment_sources,
    private_output,
    repository_root,
)
from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.transmission import TransmissionSpec, prepare_transmission, transmission_vjp, transmit
from dpt.validation.projection import integrate_sampled_field
from dpt.volumes import GridSpec


def main():
    args = experiment_parser(__file__, __doc__).parse_args()
    config = json.loads(args.config.read_text())
    root = repository_root(__file__)
    target = private_output(args.output, root)
    with RunRecorder(
        target, configuration=config, sources=experiment_sources(__file__, args.config)
    ) as run:
        device = wp.get_device(args.device)
        if not device.is_cuda:
            raise ValueError("CUDA required")
        run.set_metadata(
            device_name=device.name,
            architecture=device.arch,
            warp=wp.__version__,
            precision="FP32 field/counts; FP64 pose VJP and independent reference",
            fixture="Prescribed trilinear tent, not anatomy",
        )
        grid = GridSpec((3, 3, 3), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0))
        geometry = DetectorGeometry(
            (0.0, 0.0, -5.0), (0.0, 0.0, 5.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0), (1, 1)
        )
        values = np.tile(np.array([1.0, 2.0, 1.0], dtype=np.float32) * config["mu_mm_inv"], 9)
        field = wp.array(values, dtype=wp.float32, device=device)
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device=device)
        host = wp.empty(12, dtype=wp.float64, device="cpu", pinned=True)
        host_values = host.numpy()
        depth = wp.empty(1, dtype=wp.float32, device=device)
        counts = wp.empty(1, dtype=wp.float32, device=device)
        seed = wp.ones(1, dtype=wp.float32, device=device)
        adj_depth = wp.empty(1, dtype=wp.float32, device=device)
        gradient = wp.empty(6, dtype=wp.float64, device=device)
        workspace = prepare_projection(
            grid, geometry, ProjectionSpec(config["samples_per_ray"]), device=args.device
        )
        transmission = prepare_transmission(
            TransmissionSpec(beam="scalar"),
            device=args.device,
            max_pixels=1,
            stream=workspace.stream,
        )

        def forward(t):
            host_values[:] = compose_pose(RigidTransform(), (t, 0.0, 0.0, 0.0, 0.0, 0.0)).packed()
            wp.copy(pose, host, stream=workspace.stream)
            project_optical_depth(field, pose, out_L=depth, workspace=workspace)
            transmit(depth, config["n0"], out_counts=counts, workspace=transmission)
            wp.synchronize_stream(workspace.stream)
            return float(counts.numpy()[0])

        cases = []
        for label, anchor in [("smooth", 0.25), ("knot", 0.0)]:
            baseline = forward(anchor)
            transmission_vjp(
                depth, config["n0"], seed_counts=seed, out_grad_L=adj_depth, workspace=transmission
            )
            projection_vjp(field, pose, adj_L=adj_depth, out_pose=gradient, workspace=workspace)
            wp.synchronize_stream(workspace.stream)
            vjp = float(gradient.numpy()[0])
            independent_depth = integrate_sampled_field(
                values.tolist(),
                grid,
                geometry,
                compose_pose(RigidTransform(), (anchor, 0.0, 0.0, 0.0, 0.0, 0.0)),
                0,
                0,
                midpoint_samples=config["samples_per_ray"],
            )
            reference = float(config["n0"] * np.exp(-independent_depth))
            if abs(reference - baseline) > 0.001:
                raise ValueError("Independent forward reference failed")
            rows = []
            for h in config["steps_mm"]:
                plus = forward(anchor + h)
                minus = forward(anchor - h)
                central = (plus - minus) / (2 * h)
                rows.append(
                    {
                        "step_mm": h,
                        "plus_counts": plus,
                        "minus_counts": minus,
                        "central_counts_per_mm": central,
                        "forward_counts_per_mm": (plus - baseline) / h,
                        "backward_counts_per_mm": (baseline - minus) / h,
                        "absolute_disagreement_counts_per_mm": abs(central - vjp),
                        "crosses_knot": h >= abs(anchor),
                    }
                )
            cases.append(
                {
                    "id": label,
                    "translation_mm": anchor,
                    "counts": baseline,
                    "vjp_counts_per_mm": vjp,
                    "independent_counts": reference,
                    "rows": rows,
                }
            )
        mu = float(values[0])
        factor = 3.0 * mu
        expected_smooth = config["n0"] * np.exp(-factor * 1.75) * factor
        if abs(cases[0]["vjp_counts_per_mm"] - expected_smooth) > 0.001:
            run.write_json(
                "failed-check.json", {"cases": cases, "expected_smooth": float(expected_smooth)}
            )
            raise ValueError(
                "Independent analytic smooth derivative failed: "
                f"{cases[0]['vjp_counts_per_mm']} vs {expected_smooth}"
            )
        if max(abs(r["central_counts_per_mm"]) for r in cases[1]["rows"]) > 0.01:
            raise ValueError("Symmetric knot check failed")
        result = {
            "schema_version": 1,
            "cases": cases,
            "fixture": {
                "shape": [3, 3, 3],
                "spacing_mm": [1, 1, 1],
                "origin_mm": [-1, -1, -1],
                "x_weights": [1, 2, 1],
                "mu_mm_inv_stored": mu,
                "ray_source_mm": [0, 0, -5],
                "detector_mm": [0, 0, 5],
            },
            "checks": {
                "independent_forward_absolute_tolerance_counts": 0.001,
                "analytic_smooth_derivative_counts_per_mm": float(expected_smooth),
                "smooth_derivative_absolute_error": abs(
                    cases[0]["vjp_counts_per_mm"] - expected_smooth
                ),
                "passed": True,
            },
            "convention": (
                "Interior x=0 interpolation knot. Selected floor-cell branch is recorded; "
                "an ordinary derivative at the knot does not exist. "
                "Smooth rows with h>=0.25 cross that knot."
            ),
            "memory": (
                "Persistent field, pose and projection/transmission workspaces on CUDA. "
                "Host downloads record scalar results; no performance claim."
            ),
        }
        run.write_json("directional-checks.json", result)
        print(json.dumps({"cases": 2, "checks": result["checks"]}), flush=True)


if __name__ == "__main__":
    main()
