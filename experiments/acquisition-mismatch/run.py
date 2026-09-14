"""Compare fixed calibration with joint pose/calibration fitting on supplied split data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from dpt._runtime import prepare_context
from dpt.experiments import (
    RunRecorder,
    experiment_parser,
    experiment_sources,
    private_output,
    repository_root,
)
from dpt.recovery import PoseChart
from dpt.registration import RecoveryPolicy
from dpt.spectral_inputs import read_spectral_inputs
from dpt.spectral_recovery import (
    CalibrationBlock,
    SpectralPoseEvaluator,
    SpectralPoseProblem,
    recover_spectral_pose,
)

ROOT = repository_root(__file__)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def subset(problem: SpectralPoseProblem, names: tuple[str, ...]) -> SpectralPoseProblem:
    views = tuple(view for view in problem.views if view.name in names)
    if len(views) != len(names) or len(set(names)) != len(names) or not views:
        raise ValueError("a data split must contain distinct existing view names")
    groups = {view.calibration_group for view in views}
    return replace(
        problem,
        views=views,
        calibration=tuple(block for block in problem.calibration if block.name in groups),
    )


def fixed(block: CalibrationBlock) -> CalibrationBlock:
    return replace(
        block,
        fit_scale="none",
        fit_offset=False,
        scale_prior_precision=0.0,
        offset_prior_precision=0.0,
    )


def main() -> None:
    parser = experiment_parser(__file__, __doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fit-views", nargs="+", required=True)
    parser.add_argument("--heldout-views", nargs="+", required=True)
    arguments = parser.parse_args()
    output = private_output(arguments.output, ROOT)
    fit_names, heldout_names = tuple(arguments.fit_views), tuple(arguments.heldout_views)
    if set(fit_names) & set(heldout_names):
        raise ValueError("fitting and held-out observations must be disjoint")
    configuration: dict[str, Any] = json.loads(arguments.config.read_text())
    if (
        set(configuration) != {"schema_version", "samples_per_ray", "optimisation"}
        or configuration["schema_version"] != 1
    ):
        raise ValueError("unsupported acquisition-mismatch configuration")
    policy = RecoveryPolicy(**configuration["optimisation"])
    input_hash = digest(arguments.inputs)
    sources = experiment_sources(
        __file__, arguments.config, extra={"input/model.json": arguments.model}
    )
    with RunRecorder(
        output,
        configuration={**configuration, "fit_views": fit_names, "heldout_views": heldout_names},
        sources=sources,
        metadata={
            "input_npz_sha256": input_hash,
            "input_role": "caller-supplied independent fit and held-out observations",
        },
    ) as run:
        metadata: dict[str, Any] = json.loads(
            (output / "sources" / "input" / "model.json").read_text()
        )
        problem, chart = read_spectral_inputs(
            arguments.inputs,
            metadata,
            samples_per_ray=configuration["samples_per_ray"],
            device=arguments.device,
        )
        fit_problem, heldout_problem = subset(problem, fit_names), subset(problem, heldout_names)
        context = prepare_context(device=arguments.device)
        context.disjoint(
            [(view.name, view.observation) for view in fit_problem.views],
            [(view.name, view.observation) for view in heldout_problem.views],
        )
        if not any(block.dimension for block in fit_problem.calibration):
            raise ValueError(
                "joint comparison requires at least one declared active nuisance parameter"
            )
        if not {block.name for block in heldout_problem.calibration} <= {
            block.name for block in fit_problem.calibration
        }:
            raise ValueError("held-out calibration groups must have corresponding fitting views")
        reports: list[dict[str, Any]] = []
        for label, blocks in (
            ("fixed-calibration", tuple(fixed(block) for block in fit_problem.calibration)),
            ("joint-calibration", fit_problem.calibration),
        ):
            fitted = replace(fit_problem, calibration=blocks)
            evaluator = SpectralPoseEvaluator(fitted, chart, device=arguments.device)
            result = recover_spectral_pose(evaluator, policy=policy)
            estimates = {value.name: value for value in result.calibration}
            evaluation_blocks = tuple(
                replace(
                    fixed(block),
                    gain=estimates[block.name].gain,
                    exposure=estimates[block.name].exposure,
                    offset=estimates[block.name].offset,
                )
                for block in heldout_problem.calibration
            )
            evaluation_problem = replace(heldout_problem, calibration=evaluation_blocks)
            # The held-out evaluator only evaluates the accepted fitted pose.
            # It never supplies observations or gradients to either fitting solve.
            heldout = SpectralPoseEvaluator(
                evaluation_problem,
                PoseChart(anchor=result.pose, scales=chart.scales),
                device=arguments.device,
            )
            score = heldout((0.0,) * heldout.dimension)
            report = {
                "model": label,
                "recovery": asdict(result),
                "heldout_loss": score.loss,
                "heldout_view_losses": heldout.last_view_losses,
            }
            reports.append(report)
            run.write_json(f"{label}.json", report)
            run.set_metadata(
                device=str(evaluator.context.device), device_name=evaluator.context.device.name
            )
        run.write_json(
            "comparison.json",
            {
                "models": reports,
                "interpretation": (
                    "Held-out residual change is not proof of pose recovery "
                    "or physical calibration."
                ),
            },
        )
        if digest(arguments.inputs) != input_hash:
            raise RuntimeError("input NPZ changed during execution")


if __name__ == "__main__":
    main()
