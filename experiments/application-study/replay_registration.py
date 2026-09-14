"""Diagnose or re-evaluate a recorded registration case without retuning its gate."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from _study import read_json, record_sources, rigid
from diagnose_precision import _depth_loss, _independent_subset

from dpt.examples._common import load_case
from dpt.examples.registration import SharedPoseObjective, prepare_views
from dpt.experiments import RunRecorder
from dpt.recovery import PoseChart
from dpt.registration import RecoveryPolicy, recover_parameters


def replay(
    case_path: Path,
    previous: Path,
    output: Path,
    device: str,
    independent: bool,
    stationarity_only: bool = False,
    precision: str = "float64",
    integration: str = "midpoint",
) -> None:
    case = load_case(case_path)
    cfg = dict(case.config["registration"])
    cfg["precision"] = precision
    cfg["integration"] = integration
    if "evaluation" in cfg:
        raise ValueError("diagnostic fitting case must not contain evaluation references")
    chart_cfg = dict(cfg["chart"])
    chart_cfg["anchor"] = rigid(chart_cfg["anchor"])
    chart = PoseChart(**chart_cfg)
    old = read_json(previous)["result"]
    policy = dict(cfg["policy"])
    if stationarity_only:
        policy.update(relative_loss_tolerance=0.0, step_tolerance=0.0)
    sources = record_sources(__file__, case_path)
    sources["input/previous-optimisation.json"] = previous
    with RunRecorder(
        output,
        configuration={
            "role": "prescribed final fit replay; unchanged observations and gradient gate",
            "case": str(case_path),
            "previous": str(previous),
            "reference_access": False,
            "original_policy": cfg["policy"],
            "policy": policy,
            "stationarity_only_diagnostic": stationarity_only,
            "precision": precision,
            "integration": integration,
            "independent_subset": independent,
        },
        sources=sources,
    ) as recorder:
        views = prepare_views(case, cfg, chart, device)
        objective = SharedPoseObjective(views, chart)
        trials = []

        def traced(point):
            value = objective(point)
            trials.append({"parameters": list(point), **asdict(value)})
            return value

        result = recover_parameters(traced, (0.0,) * 6, policy=RecoveryPolicy(**policy))
        endpoints = {}
        for label, parameters in (("previous", old["parameters"]), ("current", result.parameters)):
            point = np.asarray(parameters)
            value = objective(tuple(point))
            gradient = np.asarray(value.gradient)
            direction = -gradient / np.linalg.norm(gradient)
            rows = []
            for step in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
                sides, depth_sides = [], []
                for sign in (-1, 1):
                    sides.append(objective(tuple(point + sign * step * direction)).loss)
                    depth_sides.append(_depth_loss(views))
                rows.append(
                    {
                        "step": step,
                        "objective_difference": (sides[1] - sides[0]) / (2 * step),
                        "fp64_from_stored_depth_difference": (depth_sides[1] - depth_sides[0])
                        / (2 * step),
                    }
                )
            endpoints[label] = {
                "parameters": list(parameters),
                "evaluation": asdict(value),
                "direction": direction.tolist(),
                "directional_gradient": math.fsum(
                    a * b for a, b in zip(gradient, direction, strict=True)
                ),
                "finite_differences": rows,
            }
        data = {
            "result": asdict(result),
            "exact_previous_replay": json.loads(json.dumps(asdict(result))) == old,
            "endpoints": endpoints,
            "trials": trials,
            "independent_subset": _independent_subset(views, chart, np.asarray(result.parameters))
            if independent
            else None,
        }
        recorder.write_json("replay.json", data)
        print(
            json.dumps(
                {
                    "reason": result.reason,
                    "gradient_norm": max(map(abs, result.evaluation.gradient)),
                    "evaluations": result.evaluations,
                    "exact_previous_replay": data["exact_previous_replay"],
                    "output": str(output),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--independent", action="store_true")
    parser.add_argument("--stationarity-only", action="store_true")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--integration", choices=("midpoint", "cell_gauss"), default="midpoint")
    args = parser.parse_args()
    replay(
        args.case,
        args.previous,
        args.output,
        args.device,
        args.independent,
        args.stationarity_only,
        args.precision,
        args.integration,
    )
