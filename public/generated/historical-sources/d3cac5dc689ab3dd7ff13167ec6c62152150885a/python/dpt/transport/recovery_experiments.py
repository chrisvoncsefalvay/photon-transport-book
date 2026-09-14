"""Private, preregistered recovery comparisons using the production CUDA oracle.

Independent layer integrals supply observations and assess answers, never runtime
proposals. The sampled reference for scattering uses a separate seed pool; it is
not fed back into model construction or candidate acceptance.
"""

# This private experiment driver inspects prepared workspaces for execution evidence.
# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from dpt.experiments import RunRecorder
from dpt.statistics import mean_standard_error
from dpt.stochastic_recovery import StochasticPolicy, recover_expected_signal

from .derivatives import TransportParameter
from .experiments import AbsorptionExperiment, _device_metadata, _recovery
from .forward import prepare_transport
from .inverse import prepare_transport_inverse
from .model import MaterialGrid, PlanarDetector, TransportSpec
from .rng import HistoryBatch
from .source import ParallelBeam


def _scattering_run(seed: int, recorder: RunRecorder, *, reference: dict[str, Any] | None) -> None:
    from dpt._runtime import prepare_context

    context = prepare_context()
    wp = context.wp
    # Fixed mathematical cube, not a surrogate tissue or measured acquisition.
    spec = TransportSpec(
        MaterialGrid((-0.5, -0.5, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1),
        PlanarDetector((-2.0, -2.0), (4.0, 4.0), (1, 1), 2.0),
        80.0,
        "declared isotropic scattering cube for recovery validation",
        estimator="continuous-absorption",
    )
    workspace = prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=context.device),
        absorption=wp.array([0.3], dtype=wp.float64, device=context.device),
        scattering=wp.array([0.1], dtype=wp.float64, device=context.device),
        max_histories=65536,
        device=str(context.device),
        stream=context.stream,
    )
    # Observation is declared directly. The independent reference locates its
    # optimum afterwards; no analytic/reference signal enters the oracle.
    observation = 0.08
    oracle = prepare_transport_inverse(
        workspace,
        source=ParallelBeam((0.0, 0.0, -1.0), (0.0, 0.0)),
        parameters=(TransportParameter("log-source-amplitude"),),
        observation=wp.array([observation], dtype=wp.float64, device=context.device),
        pixel_weights=wp.ones(1, dtype=wp.float64, device=context.device),
        base_density=(1.0,),
        local_model=True,
    )
    if reference is None:
        means: list[float] = []
        for replicate in range(256):
            oracle._chart((0.0,))
            oracle._mean(
                HistoryBatch(seed, replicate * 65536, 65536), 1.0, oracle._arrays["mean_a"]
            )
            workspace.check_status()
            means.append(float(oracle._arrays["mean_a"].numpy()[0]))
        mean, standard_error = mean_standard_error(tuple(means))
        recorder.set_metadata(**_device_metadata(workspace))
        recorder.write_json(
            "reference.json",
            {
                "mean": mean,
                "standard_error": standard_error,
                "absolute_uncertainty": 7 * standard_error,
                "uncertainty_method": (
                    "seven replicate standard errors; heuristic, not a finite-sample bound"
                ),
                "seed": seed,
                "batch": 65536,
                "replicates": means,
                "histories_traced": oracle.histories_traced,
                "purpose": "independent held-out forward mean; never optimiser input",
            },
        )
        return
    policy = StochasticPolicy(
        proposal="quadratic",
        replicates=8,
        initial_batch=4096,
        maximum_batch=65536,
        final_validation_batch=65536,
        unique_history_budget=16_000_000,
        numerical_gradient_allowance=1e-12,
    )
    start = perf_counter()
    try:
        result = recover_expected_signal(oracle, (math.log(0.08),), seed=seed, policy=policy)
    except Exception as error:
        snapshot = getattr(error, "recovery_diagnostics", None)
        recorder.write_json(
            "failure-diagnostics.json",
            {
                "error": repr(error),
                "controller": asdict(snapshot) if snapshot is not None else None,
                "histories_submitted_including_replay": oracle.histories_traced,
                "parameter_upload_bytes": oracle.parameter_upload_bytes,
                "scalar_download_bytes": oracle.scalar_download_bytes,
            },
        )
        raise
    elapsed = perf_counter() - start
    amplitude = math.exp(result.parameters[0])
    mu = reference["mean"]
    error = reference["absolute_uncertainty"]
    # Monotone in this positive-signal neighbourhood; evaluate both interval ends
    # and its possible quadratic vertex to enclose the reference gradient.
    means = [mu - error, mu + error]
    vertex = observation / (2 * amplitude)
    if means[0] < vertex < means[1]:
        means.append(vertex)
    gradients = [(amplitude * value - observation) * amplitude * value for value in means]
    recorder.set_metadata(**_device_metadata(workspace))
    recorder.write_json(
        "recovery.json",
        {
            "seed": seed,
            "policy": asdict(policy),
            "result": asdict(result),
            "amplitude": amplitude,
            "independent_optimum_interval": [
                observation / (mu + error),
                observation / (mu - error),
            ],
            "reference_gradient_interval": [min(gradients), max(gradients)],
            "reference_stationarity_pass": max(abs(value) for value in gradients)
            <= policy.gradient_tolerance,
            "histories_traced_including_replay": oracle.histories_traced,
            "allocated_device_bytes": oracle.allocated_bytes,
            "scalar_download_bytes": oracle.scalar_download_bytes,
            "parameter_upload_bytes": oracle.parameter_upload_bytes,
            "elapsed_seconds_contended": elapsed,
            "deterministic_sampling": oracle.deterministic_sampling,
        },
    )


def main(script: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("original", "ablation", "family", "reference", "stochastic"),
        required=True,
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--seed-base", type=int, default=9031001)
    args = parser.parse_args()
    if not 0 <= args.seed_base <= 2**64 - 1 - 31 * 104729:
        parser.error("the complete 32-run seed sequence must fit unsigned 64 bits")
    root = Path(script).resolve().parents[2]
    output = args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError("raw execution records must remain outside the authoring tree")
    sources = {str(path.relative_to(root)): path for path in (root / "python/dpt").rglob("*.py")}
    sources.update({str(Path(script).resolve().relative_to(root)): Path(script).resolve()})
    sources.update({name: root / name for name in ("pyproject.toml", "uv.lock")})
    cases: list[tuple[str, AbsorptionExperiment]] = []
    if args.mode == "original":
        cases = [
            (
                "original",
                AbsorptionExperiment(
                    seed=23971,
                    histories=256,
                    recovery_maximum_batch=256,
                    recovery_history_budget=1_000_000,
                    estimator="continuous-absorption",
                    proposal="quadratic",
                    numerical_gradient_allowance=1e-12,
                ),
            )
        ]
    if args.mode == "ablation":
        for estimator in ("analogue", "continuous-absorption"):
            for proposal in ("linear", "quadratic"):
                cases.append(
                    (
                        f"{estimator}-{proposal}",
                        AbsorptionExperiment(
                            seed=23971,
                            histories=256,
                            estimator=estimator,
                            proposal=proposal,
                            numerical_gradient_allowance=1e-12,
                            final_validation_batch=262144,
                        ),
                    )
                )
    if args.mode == "family":
        for depth in (0.1, 1.0, 3.0, 6.0):
            for target in (0.5, 1.25, 2.0):
                for initial in (0.6, 1.0, 1.8):
                    cases.append(
                        (
                            f"depth-{depth}-target-{target}-initial-{initial}",
                            AbsorptionExperiment(
                                layer_optical_depths=(depth,),
                                target_density=target,
                                initial_density=initial,
                                histories=16,
                                replicates=2,
                                recovery_maximum_batch=16,
                                recovery_history_budget=1_000_000,
                                estimator="continuous-absorption",
                                proposal="quadratic",
                                numerical_gradient_allowance=1e-12,
                            ),
                        )
                    )
    for name, config in cases:
        with RunRecorder(output / name, configuration=asdict(config), sources=sources) as recorder:
            _recovery(config, recorder)
        record = json.loads((output / name / "recovery.json").read_text())
        print(
            name,
            record["result"]["reason"],
            record["absolute_density_error"],
            record["independent_true_gradient"],
            flush=True,
        )
    if args.mode == "reference":
        with RunRecorder(
            output / "scattering-reference", configuration={"seed": 419003}, sources=sources
        ) as recorder:
            _scattering_run(419003, recorder, reference=None)
    if args.mode == "stochastic":
        if args.reference is None:
            raise ValueError("stochastic evaluation requires a frozen independent reference")
        reference: dict[str, Any] = json.loads(args.reference.read_text())
        sources["evaluation-reference.json"] = args.reference
        for i in range(32):
            seed = args.seed_base + 104729 * i
            evaluation_config: dict[str, Any] = {
                "seed": seed,
                "evaluation_index": i,
                "repetitions": 32,
                "reference": reference,
            }
            with RunRecorder(
                output / f"replicate-{i:02d}", configuration=evaluation_config, sources=sources
            ) as recorder:
                _scattering_run(seed, recorder, reference=reference)
            print(f"completed independent evaluation {i + 1}/32", flush=True)
