"""Run recorded supplied-data registration with an accepted-pose observer."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _study import record_sources, rigid

from dpt.examples._common import load_case, write_array
from dpt.examples.registration import SharedPoseObjective, prepare_views
from dpt.recovery import PoseChart
from dpt.registration import Evaluation, IterationRecord, RecoveryPolicy, recover_parameters


class AcceptedPoseObserver:
    """Capture accepted states without modifying the canonical optimiser.

    The line search returns its last successful evaluation on acceptance. The
    observer checks the recorded loss and gradient norm against that evaluation
    before copying the six-vector; rejected evaluations alone are never poses.
    """

    def __init__(self, objective: Any, chart: PoseChart) -> None:
        self.objective, self.chart = objective, chart
        self.last_parameters: tuple[float, ...] | None = None
        self.last_value: Evaluation | None = None
        self.trajectory: list[dict] = []
        self.trials: list[dict] = []
        self.trace_host_seconds = 0.0
        self.observer_host_seconds = 0.0

    def __call__(self, parameters: tuple[float, ...]) -> Evaluation:
        try:
            value = self.objective(parameters)
        except Exception as error:
            self.trials.append(
                {
                    "call": len(self.trials) + 1,
                    "parameters": list(parameters),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise
        started = time.perf_counter()
        self.last_parameters, self.last_value = tuple(parameters), value
        self.trials.append(
            {"call": len(self.trials) + 1, "parameters": list(parameters), **asdict(value)}
        )
        self.trace_host_seconds += time.perf_counter() - started
        return value

    def observe(self, record: IterationRecord) -> None:
        started = time.perf_counter()
        value, parameters = self.last_value, self.last_parameters
        if value is None or parameters is None:
            raise RuntimeError("accepted callback arrived without an evaluated state")
        if record.loss != value.loss or record.gradient_infinity_norm != max(
            map(abs, value.gradient)
        ):
            raise RuntimeError(
                "accepted callback loss/gradient do not match the last evaluated pose"
            )
        self.trajectory.append(
            {
                **asdict(record),
                "parameters": list(parameters),
                "pose_object_to_world": asdict(self.chart.pose(parameters)),
            }
        )
        self.observer_host_seconds += time.perf_counter() - started


def fit_case(
    case_path: Path,
    output: Path,
    *,
    device: str = "cuda:0",
    deadline: float | None = None,
    profile: bool = False,
) -> dict:
    """Use the supplied driver preparation/objective and canonical solver unchanged."""
    case = load_case(case_path)
    cfg = case.config["registration"]
    if "evaluation" in cfg:
        raise ValueError("study fitting cases must not contain evaluation references")
    chart_cfg = dict(cfg["chart"])
    chart_cfg["anchor"] = rigid(chart_cfg["anchor"])
    chart = PoseChart(**chart_cfg)
    recorder = case.record(
        output,
        entrypoint=__file__,
        metadata={
            "application": "CT-derived supplied registration",
            "reference_access": False,
            "trajectory": "accepted observer with exact loss/gradient-state assertions",
        },
    )
    recorder.sources.update(record_sources(__file__, case_path))
    with recorder as record:
        setup_started = time.perf_counter()
        views = prepare_views(case, cfg, chart, device)
        setup_seconds = time.perf_counter() - setup_started
        objective = SharedPoseObjective(views, chart)
        zero = (0.0,) * 6
        first = objective(zero)
        for view in views:
            write_array(
                record,
                f"initial-{view.identifier}.npy",
                view.evaluator.prediction.numpy().reshape(view.evaluator.problem.geometry.shape),
            )
        timings: dict[str, Any] = {"setup_seconds": setup_seconds, "gradient_evaluations": {}}
        if profile:
            for label, selected in (("one_view", views[:1]), ("all_views", views)):
                function = SharedPoseObjective(selected, chart)
                function(zero)
                measurements = []
                for _ in range(4):
                    start = time.perf_counter()
                    function(zero)
                    measurements.append(time.perf_counter() - start)
                timings["gradient_evaluations"][label] = measurements
            timings["memory_scope"] = (
                "persistent Warp buffers; no peak/allocation claim without timeline"
            )
        tracker = AcceptedPoseObserver(objective, chart)
        started = time.perf_counter()
        result = recover_parameters(
            tracker,
            zero,
            policy=RecoveryPolicy(**cfg["policy"]),
            observe=tracker.observe,
            cancelled=lambda: deadline is not None and time.perf_counter() >= deadline,
        )
        elapsed = time.perf_counter() - started
        if tracker.trajectory and tracker.trajectory[-1]["parameters"] != list(result.parameters):
            raise RuntimeError("last accepted trajectory pose differs from solver return")
        export_started = time.perf_counter()
        final = objective(result.parameters)
        if final != result.evaluation:
            raise RuntimeError("accepted final objective/gradient failed exact refresh")
        losses = {}
        for view in views:
            value = view.evaluator(result.parameters)
            losses[view.identifier] = value.loss
            write_array(
                record,
                f"prediction-{view.identifier}.npy",
                view.evaluator.prediction.numpy().reshape(view.evaluator.problem.geometry.shape),
            )
        if math.fsum(losses.values()) != final.loss:
            raise RuntimeError("final per-view losses do not reproduce accepted shared loss")
        report = {
            "pose_object_to_world": asdict(chart.pose(result.parameters)),
            "chart": asdict(chart),
            "policy": asdict(RecoveryPolicy(**cfg["policy"])),
            "result": asdict(result),
            "stationary": result.stationary,
            "initial_objective": first.loss,
            "solve_wall_seconds": elapsed,
            "soft_time_budget_reached": result.reason == "cancelled",
            "final_losses_by_view": losses,
            "reference_access": False,
            "trace_host_seconds": tracker.trace_host_seconds,
            "observer_host_seconds": tracker.observer_host_seconds,
            "final_refresh_and_array_export_seconds": time.perf_counter() - export_started,
        }
        record.write_json("optimisation.json", report)
        record.write_json("trajectory.json", tracker.trajectory)
        record.write_json("trials.json", tracker.trials)
        record.write_json("profile.json", timings)
        record.set_metadata(
            device=str(views[0].evaluator.context.device),
            warp=views[0].evaluator.context.wp.__version__,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        raise ValueError("time limit must be finite and positive")
    report = fit_case(
        args.case,
        args.output,
        device=args.device,
        deadline=time.perf_counter() + args.time_limit,
        profile=args.profile,
    )
    print(report["result"]["reason"], report["result"]["evaluation"]["loss"], flush=True)


if __name__ == "__main__":
    main()
