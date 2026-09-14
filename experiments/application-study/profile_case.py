"""Capture a bounded representative supplied-case evaluation; no optimisation."""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

from _study import record_sources, rigid

from dpt.examples._common import load_case
from dpt.examples.registration import SharedPoseObjective, prepare_views
from dpt.experiments import RunRecorder
from dpt.recovery import PoseChart


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    case = load_case(args.case)
    settings = case.config["registration"]
    if "evaluation" in settings:
        raise ValueError("profile accepts fitting inputs only")
    chart_values = dict(settings["chart"])
    chart_values["anchor"] = rigid(chart_values["anchor"])
    chart = PoseChart(**chart_values)
    torch = importlib.import_module("torch")
    wp = importlib.import_module("warp")
    with RunRecorder(
        args.output,
        configuration={
            "scope": (
                "two warmed full-gradient evaluations at fixed initial training pose; "
                "no fitting or reference access"
            )
        },
        sources=record_sources(__file__, args.case),
    ) as run:
        views = prepare_views(case, settings, chart, args.device)
        objective = SharedPoseObjective(views, chart)
        expected = objective((0.0,) * 6)
        retained_names = (
            "pose_device",
            "optical_depth",
            "prediction",
            "image_seed",
            "depth_seed",
            "pose_seed",
            "loss",
        )
        identities = {
            view.identifier: {
                name: int(getattr(view.evaluator, name).ptr or 0) for name in retained_names
            }
            for view in views
        }
        torch.cuda.nvtx.range_push("dpt-application-warmed-evaluations")
        times = []
        for _ in range(2):
            start = time.perf_counter()
            result = objective((0.0,) * 6)
            times.append(time.perf_counter() - start)
            if result != expected:
                raise ValueError("same-point evaluation changed during profile")
            for view in views:
                current = {
                    name: int(getattr(view.evaluator, name).ptr or 0) for name in retained_names
                }
                if current != identities[view.identifier]:
                    raise ValueError("prepared device buffer was replaced during warm evaluation")
        torch.cuda.nvtx.range_pop()
        wp.synchronize_device(args.device)
        run.write_json(
            "profile.json",
            {
                "view_count": len(views),
                "pixels_per_view": [v.evaluator.problem.geometry.pixels for v in views],
                "samples_per_ray": settings["samples_per_ray"],
                "precision": views[0].evaluator.problem.precision,
                "integration": views[0].evaluator.problem.integration,
                "retained_device_buffer_pointers_unchanged": True,
                "retained_primary_image_bytes_per_view": [
                    sum(
                        getattr(view.evaluator, name).size
                        * (8 if getattr(view.evaluator, name).dtype == wp.float64 else 4)
                        for name in ("optical_depth", "prediction", "image_seed", "depth_seed")
                    )
                    for view in views
                ],
                "seconds": times,
                "gpu_name": torch.cuda.get_device_name(args.device),
                "free_total_bytes": torch.cuda.mem_get_info(args.device),
                "timing_scope": (
                    "shared GPU, profiler instrumentation if present; "
                    "not an exclusive or peak allocation benchmark"
                ),
            },
        )


if __name__ == "__main__":
    main()
