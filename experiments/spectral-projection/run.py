"""Execute supplied spectral data through the canonical library; no figures or invented inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dpt.experiments import (
    RunRecorder,
    experiment_parser,
    experiment_sources,
    private_output,
    repository_root,
)
from dpt.registration import RecoveryPolicy
from dpt.spectral_inputs import read_spectral_inputs
from dpt.spectral_recovery import SpectralPoseEvaluator, recover_spectral_pose

ROOT = repository_root(__file__)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = experiment_parser(__file__, __doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="supplied numeric NPZ archive")
    parser.add_argument("--model", type=Path, required=True, help="explicit physical metadata JSON")
    parser.add_argument("--export-predictions", action="store_true")
    arguments = parser.parse_args()
    output = private_output(arguments.output, ROOT)
    configuration: dict[str, Any] = json.loads(arguments.config.read_text())
    if set(configuration) != {"schema_version", "samples_per_ray", "mode", "optimisation"}:
        raise ValueError("unexpected or missing spectral experiment configuration keys")
    if configuration["schema_version"] != 1 or configuration["mode"] not in ("evaluate", "recover"):
        raise ValueError("configuration requires schema_version=1 and mode=evaluate or recover")
    policy = RecoveryPolicy(**configuration["optimisation"])
    input_hash = digest(arguments.inputs)
    sources = experiment_sources(
        __file__, arguments.config, extra={"input/model.json": arguments.model}
    )
    with RunRecorder(
        output,
        configuration=configuration,
        sources=sources,
        metadata={
            "input_npz_sha256": input_hash,
            "input_npz_bytes": arguments.inputs.stat().st_size,
            "input_role": "caller-supplied material fields and observations",
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
        evaluator = SpectralPoseEvaluator(problem, chart, device=arguments.device)
        if configuration["mode"] == "recover":
            result = recover_spectral_pose(evaluator, policy=policy)
            parameters = result.optimisation.parameters
            run.write_json("recovery.json", asdict(result))
        else:
            parameters = (0.0,) * evaluator.dimension
        # Scratch can belong to a rejected line-search trial. This explicit
        # accepted-state evaluation is necessary before recording predictions.
        evaluation = evaluator(parameters)
        run.write_json(
            "evaluation.json",
            {
                "parameters": parameters,
                "loss": evaluation.loss,
                "gradient": evaluation.gradient,
                "view_losses": evaluator.last_view_losses,
                "calibration": [
                    asdict(value) for value in evaluator.calibration_estimates(parameters)
                ],
            },
        )
        run.set_metadata(
            device=str(evaluator.context.device),
            device_name=evaluator.context.device.name,
            image_shapes=[view.geometry.shape for view in problem.views],
            coefficient_provenance=[
                [asdict(p) for p in view.spectral.coefficients_provenance] for view in problem.views
            ],
            storage="binary32 images; binary64 coordinates, spectral sums and pose gradients",
            acceptance=(
                "execution record; independent numerical and recovery acceptance remain separate"
            ),
        )
        if arguments.export_predictions:
            # This opt-in record boundary is outside the optimiser. It is the
            # only place the driver downloads complete detector images.
            for index, (view, prediction) in enumerate(
                zip(problem.views, evaluator.predicted_images, strict=True)
            ):
                run.write_json(
                    f"predictions/view-{index}.json",
                    {
                        "view": view.name,
                        "shape": view.geometry.shape,
                        "unit": view.spectral.output_unit,
                        "values": prediction.numpy().tolist(),
                    },
                )
        if digest(arguments.inputs) != input_hash:
            raise RuntimeError("input NPZ changed during execution")


if __name__ == "__main__":
    main()
