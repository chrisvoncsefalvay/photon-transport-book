"""Replay development fits and measure the local loss precision without tuning."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from _study import digest, read_json, record_sources, rigid

from dpt.examples._common import load_case
from dpt.examples.registration import PreparedView, SharedPoseObjective, prepare_views
from dpt.experiments import RunRecorder
from dpt.recovery import PoseChart, PrimaryPoseEvaluator
from dpt.registration import RecoveryPolicy, recover_parameters
from dpt.validation.projection import integrate_sampled_field


def _independent_subset(views, chart, point):
    """Check a predetermined 24-ray/view CPU midpoint oracle on the actual known CT."""
    import itertools

    prepared, references = [], []
    field = views[0].evaluator.problem.attenuation.numpy().astype(np.float64).tolist()
    for view in views:
        evaluator = view.evaluator
        problem, wp = evaluator.problem, evaluator.context.wp
        rows = list(itertools.product((16, 48, 80, 112), (0, 32, 56, 64, 96, 127)))
        mask = np.zeros(problem.geometry.pixels, dtype=np.float32)
        pixels = [row * problem.geometry.shape[1] + col for row, col in rows]
        mask[pixels] = 1.0
        mask *= problem.weights.numpy()
        weights = wp.array(mask, dtype=wp.float32, device=evaluator.context.device)
        selected = PrimaryPoseEvaluator(
            replace(problem, weights=weights), chart, device=str(evaluator.context.device)
        )
        prepared.append(PreparedView(view.identifier, selected, int(mask.sum())))
        references.append((problem, problem.observation.numpy(), rows, mask))
    value = SharedPoseObjective(tuple(prepared), chart)(tuple(point))
    direction = np.array([1.0, -2.0, 0.5, 0.7, -0.3, 1.1])
    direction /= np.linalg.norm(direction)

    def independent(coords):
        terms = []
        pose = chart.pose(tuple(coords))
        for problem, observed, pixels, mask in references:
            for row, col in pixels:
                weight = float(mask[row * problem.geometry.shape[1] + col])
                if weight == 0.0:
                    continue
                depth = integrate_sampled_field(
                    field,
                    problem.grid,
                    problem.geometry,
                    pose,
                    row,
                    col,
                    midpoint_samples=problem.samples_per_ray
                    if problem.integration == "midpoint"
                    else None,
                    validate=False,
                )
                mean = problem.open_beam * math.exp(-depth)
                target = float(observed[row * problem.geometry.shape[1] + col])
                relative = (mean - target) / target if target else 0.0
                term = target * (relative - math.log1p(relative)) if target else mean
                terms.append(weight * term)
        return math.fsum(terms)

    predicted = float(np.array(value.gradient) @ direction)
    rows = []
    for h in (1e-3, 1e-4, 1e-5):
        finite_difference = (
            independent(point + h * direction) - independent(point - h * direction)
        ) / (2 * h)
        rows.append(
            {
                "step": h,
                "finite_difference": finite_difference,
                "minus_vjp": finite_difference - predicted,
            }
        )
    return {
        "scope": (
            "fixed 24 rays/view, independent FP64 CPU declared ray integration "
            "of the actual known field"
        ),
        "canonical_loss": value.loss,
        "independent_loss": independent(point),
        "direction": direction.tolist(),
        "vjp": predicted,
        "differences": rows,
    }


def _depth_loss(views) -> float:
    """FP64 pointwise arithmetic on unchanged canonical FP32 optical depths."""
    terms = []
    for view in views:
        evaluator = view.evaluator
        depth = evaluator.optical_depth.numpy().astype(np.float64)
        observed = evaluator.problem.observation.numpy().astype(np.float64)
        valid = evaluator.problem.weights.numpy() > 0
        mean = evaluator.problem.open_beam * np.exp(-depth[valid])
        target = observed[valid]
        positive = target > 0
        term = mean.copy()
        relative = (mean[positive] - target[positive]) / target[positive]
        term[positive] = target[positive] * (relative - np.log1p(relative))
        terms.extend(term.tolist())
    return math.fsum(terms)


def diagnose(pilot: Path, output: Path, device: str) -> None:
    parent = read_json(pilot / "run.json")
    if parent["status"] != "complete" or parent["configuration"]["scope"] != (
        "development pilot; no final evaluation"
    ):
        raise ValueError("diagnostic accepts completed development pilot only")
    sources = record_sources(__file__, pilot / "run.json")
    for label in ("orthogonal", "base", "selected", "fixed"):
        sources[f"input/{label}-case.json"] = pilot / f"cases/{label}.json"
        sources[f"input/{label}-optimisation.json"] = pilot / label / "optimisation.json"
        sources[f"input/{label}-run.json"] = pilot / label / "run.json"
    with RunRecorder(
        output,
        configuration={
            "role": "development-only arithmetic diagnosis; no gate or solver changes",
            "reference_access": False,
            "steps": [10.0**-k for k in range(2, 8)],
        },
        sources=sources,
    ) as recorder:
        for label in ("orthogonal", "base", "selected", "fixed"):
            source_run = read_json(pilot / label / "run.json")
            source_result = pilot / label / "optimisation.json"
            if digest(source_result) != source_run["output_sha256"]["optimisation.json"]:
                raise ValueError("frozen accepted result identity mismatch")
            original = read_json(source_result)
            case = load_case(pilot / f"cases/{label}.json")
            cfg = case.config["registration"]
            if "evaluation" in cfg:
                raise ValueError("diagnostic fitting input includes a forbidden reference")
            chart_cfg = dict(cfg["chart"])
            chart_cfg["anchor"] = rigid(chart_cfg["anchor"])
            chart = PoseChart(**chart_cfg)
            views = prepare_views(case, cfg, chart, device)
            objective = SharedPoseObjective(views, chart)
            trials = []

            def traced(point, objective=objective, trials=trials):
                value = objective(point)
                trials.append({"call": len(trials) + 1, "parameters": list(point), **asdict(value)})
                return value

            result = recover_parameters(traced, (0.0,) * 6, policy=RecoveryPolicy(**cfg["policy"]))
            # JSON turns immutable vectors into lists; compare its normalised representation.
            import json

            if json.loads(json.dumps(asdict(result))) != original["result"]:
                raise ValueError("development replay changed a frozen accepted state or history")
            x = np.array(result.parameters)
            gradient = np.array(result.evaluation.gradient)
            final = objective(tuple(x))
            repeated = [asdict(objective(tuple(x))) for _ in range(3)]
            if any(value != asdict(final) for value in repeated):
                raise ValueError("identical-point objective/gradient is nondeterministic")
            last_accepted_call = result.history[-1].evaluations
            for trial in trials[last_accepted_call:]:
                displacement = np.array(trial["parameters"]) - x
                slope = float(gradient @ displacement)
                trial["loss_change"] = trial["loss"] - final.loss
                trial["linear_predicted_change"] = slope
                trial["armijo_acceptance_from_final"] = (
                    trial["loss"] <= final.loss + cfg["policy"]["armijo"] * slope
                )
                trial["chart_displacement_norm"] = float(np.linalg.norm(displacement))
            directions = {
                "negative_gradient_unit": -gradient / np.linalg.norm(gradient),
                "fixed_mixed_unit": np.array([1.0, -2.0, 0.5, 0.7, -0.3, 1.1]),
            }
            differences = {}
            for name, direction in directions.items():
                direction = direction / np.linalg.norm(direction)
                predicted = float(gradient @ direction)
                rows = []
                for exponent in range(2, 8):
                    step = 10.0**-exponent
                    sides = []
                    for sign in (-1, 1):
                        value = objective(tuple(x + sign * step * direction))
                        sides.append(
                            {
                                "canonical_loss": value.loss,
                                "depth_fp64_loss": _depth_loss(views),
                                "directional_gradient": float(np.array(value.gradient) @ direction),
                            }
                        )
                    fd = (sides[1]["canonical_loss"] - sides[0]["canonical_loss"]) / (2 * step)
                    depth_fd = (sides[1]["depth_fp64_loss"] - sides[0]["depth_fp64_loss"]) / (
                        2 * step
                    )
                    rows.append(
                        {
                            "step": step,
                            "canonical_directional_difference": fd,
                            "fp64_depth_directional_difference": depth_fd,
                            "canonical_minus_vjp": fd - predicted,
                            "depth_fp64_minus_vjp": depth_fd - predicted,
                            "sides": sides,
                        }
                    )
                differences[name] = {
                    "direction": direction.tolist(),
                    "vjp": predicted,
                    "rows": rows,
                }
            recorder.write_json(
                f"{label}.json",
                {
                    "accepted_replay_exact": True,
                    "same_point_repeated_exact": True,
                    "original_optimisation_sha256": digest(source_result),
                    "final_parameters": result.parameters,
                    "final_evaluation": asdict(final),
                    "final_gradient_infinity_norm": max(map(abs, gradient)),
                    "last_accepted_history": [asdict(v) for v in result.history[-2:]],
                    "last_accepted_call": last_accepted_call,
                    "last_rejected_trial": trials[-1],
                    "trials": trials,
                    "directional_precision": differences,
                    "independent_actual_field_subset": (
                        _independent_subset(views, chart, x) if label == "orthogonal" else None
                    ),
                    "limits": (
                        "FP64 loss uses canonical FP32 depths; it isolates pointwise count/loss "
                        "arithmetic but is not an independent forward projector or a solver change"
                    ),
                },
            )
            print(label, "exact replay and directional precision complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    diagnose(args.pilot, args.output, args.device)


if __name__ == "__main__":
    main()
