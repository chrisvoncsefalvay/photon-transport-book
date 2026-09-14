"""Warmed canonical CUDA workloads; analytic fixtures are timing inputs, not measurements."""

from __future__ import annotations

import hashlib
import importlib
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from dpt.experiments import (
    RunRecorder,
    experiment_parser,
    experiment_sources,
    private_output,
    repository_root,
)

ROOT = repository_root(__file__)


@dataclass
class Workload:
    operations: dict[str, Callable[[bool], None]]
    check: Callable[[], None]
    scratch_bytes: int
    description: str
    diagnostics: Callable[[], dict[str, object]] | None = None


def projection(args: Any, wp: Any, np: Any, stream: Any) -> Workload:
    from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
    from dpt.projection import (
        ProjectionSpec,
        prepare_projection,
        project_optical_depth,
        projection_vjp,
    )
    from dpt.volumes import GridSpec

    n, side = args.grid, args.side
    spacing = 20.0 / n
    grid = GridSpec((n, n, n), (spacing,) * 3, (-10 + spacing / 2,) * 3)
    geometry = DetectorGeometry(
        (-80, 1, 2), (80, -20, -20), (0, 1, 0), (0, 0, 1), (40 / side,) * 2, (side, side)
    )
    axis = np.arange(n, dtype=np.float64) * spacing - 10 + spacing / 2
    values = 0.01 + 0.00002 * (
        axis[:, None, None] ** 2 + 2 * axis[None, :, None] ** 2 + 3 * axis[None, None, :] ** 2
    )
    field = wp.array(values.astype(np.float32).ravel(), dtype=wp.float32, device=args.device)
    pose = wp.array(
        compose_pose(RigidTransform(), (0.23, -0.14, 0.17, 0.03, -0.02, 0.01)).packed(),
        dtype=wp.float64,
        device=args.device,
    )
    output = wp.empty(side * side, dtype=wp.float32, device=args.device)
    seed = wp.ones(side * side, dtype=wp.float32, device=args.device)
    gradient = wp.empty(6, dtype=wp.float64, device=args.device)
    ws = prepare_projection(
        grid, geometry, ProjectionSpec(args.samples), device=args.device, stream=stream
    )

    def forward(validate: bool) -> None:
        project_optical_depth(
            field, pose, out_L=output, workspace=ws, stream=stream, validate=validate
        )

    def reverse(validate: bool) -> None:
        projection_vjp(
            field,
            pose,
            adj_L=seed,
            out_pose=gradient,
            workspace=ws,
            stream=stream,
            validate=validate,
        )

    return Workload(
        {"projection_forward": forward, "projection_pose_vjp": reverse},
        ws.check_status,
        ws.scratch_bytes,
        "Analytic quadratic attenuation on a 20 mm cube; fixed oblique pose.",
    )


def spectral(args: Any, wp: Any, np: Any, stream: Any) -> Workload:
    from dpt.detector import CalibrationSpec, calibrate, calibration_vjp, prepare_detector
    from dpt.materials import Provenance
    from dpt.spectral import SpectralSpec, prepare_spectral, spectral_signal, spectral_vjp

    pixels, materials, energies = args.side**2, args.materials, args.energies
    formula = b"timing fixture: paths=10; mu[m,k]=0.01*(m+1)/(1+k/K); weights=1000/K; response=1"
    provenance = Provenance(
        formula.decode(),
        hashlib.sha256(formula).hexdigest(),
        "CC0 analytic formula",
        "Synthetic timing fixture; not a material database or measured spectrum",
    )
    spec = SpectralSpec(
        materials,
        energies,
        (provenance,) * materials,
        provenance,
        provenance,
        "counts",
        "Analytic timing inputs; shared uniform bin populations",
    )
    ws = prepare_spectral(spec, max_pixels=pixels, device=args.device, stream=stream)
    detector = prepare_detector(max_pixels=pixels, device=args.device, stream=stream)
    calibration = CalibrationSpec("counts", active_gain=True, active_offset=True)

    def full(size: int, value: float, dtype: Any = None) -> Any:
        return wp.full(size, value, dtype=dtype or wp.float32, device=args.device)

    paths = full(materials * pixels, 10)
    coefficients = wp.array(
        (0.01 * (np.arange(materials)[:, None] + 1) / (1 + np.arange(energies)[None, :] / energies))
        .astype(np.float32)
        .ravel(),
        dtype=wp.float32,
        device=args.device,
    )
    weights, response = full(energies, 1000 / energies), full(energies, 1)
    mean, signal, seed, mean_seed = [full(pixels, 1) for _ in range(4)]
    path_gradient = full(materials * pixels, 0)
    gain, exposure, offset = full(1, 1.2), full(1, 0.8), full(1, 0.1)
    gain_gradient, offset_gradient = full(1, 0, wp.float64), full(1, 0, wp.float64)

    def forward(validate: bool) -> None:
        spectral_signal(
            paths,
            coefficients,
            weights,
            response,
            out_mean=mean,
            workspace=ws,
            stream=stream,
            validate=validate,
        )

    def detector_forward(validate: bool) -> None:
        calibrate(
            mean,
            gain,
            exposure,
            offset,
            out_signal=signal,
            spec=calibration,
            workspace=detector,
            stream=stream,
            validate=validate,
        )

    def detector_reverse(validate: bool) -> None:
        calibration_vjp(
            mean,
            gain,
            exposure,
            offset,
            seed=seed,
            out_grad_mean=mean_seed,
            out_grad_gain=gain_gradient,
            out_grad_offset=offset_gradient,
            spec=calibration,
            workspace=detector,
            stream=stream,
            validate=validate,
        )

    def reverse(validate: bool) -> None:
        spectral_vjp(
            paths,
            coefficients,
            weights,
            response,
            seed=mean_seed,
            out_grad_paths=path_gradient,
            workspace=ws,
            stream=stream,
            validate=validate,
        )

    def check() -> None:
        ws.check_status()
        detector.check_status()

    return Workload(
        {
            "spectral_forward": forward,
            "detector_forward": detector_forward,
            "detector_vjp": detector_reverse,
            "spectral_paths_vjp": reverse,
        },
        check,
        ws.scratch_bytes + detector.scratch_bytes,
        formula.decode(),
    )


def transport(args: Any, wp: Any, np: Any, stream: Any) -> Workload:
    from dpt.transport.derivatives import TransportParameter, derivative_histories
    from dpt.transport.estimators import history_moments, prepare_estimators
    from dpt.transport.forward import prepare_transport, trace_histories
    from dpt.transport.model import MaterialGrid, PlanarDetector, TransportSpec
    from dpt.transport.rng import HistoryBatch

    n, count = args.grid, args.histories
    grid = MaterialGrid((-10, -10, 0), (20 / n,) * 3, (n,) * 3, 1)
    detector = PlanarDetector((-20, -20), (40 / args.side,) * 2, (args.side,) * 2, 30)
    spec = TransportSpec(
        grid,
        detector,
        80,
        "Analytic homogeneous timing fixture: absorption=0.01/mm, "
        "isotropic elastic scattering=0.02/mm",
        estimator=args.estimator,
    )

    def full(size: int, value: Any, dtype: Any) -> Any:
        return wp.full(size, value, dtype=dtype, device=args.device)

    ws = prepare_transport(
        spec,
        material_ids=full(n**3, 0, wp.int32),
        absorption=full(1, 0.01, wp.float64),
        scattering=full(1, 0.02, wp.float64),
        max_histories=count,
        device=args.device,
        stream=stream,
    )
    moments = prepare_estimators(ws)
    positions = full(count, wp.vec3d(0, 0, -1), wp.vec3d)
    directions = full(count, wp.vec3d(0, 0, 1), wp.vec3d)
    weights, density = full(count, 1, wp.float64), full(1, 1, wp.float64)
    pixel, events, status, derivative_pixel, derivative_status = [
        full(count, 0, wp.int32) for _ in range(5)
    ]
    score, energy, derivative = [full(count, 0, wp.float64) for _ in range(3)]
    mean, variance = [full(args.side**2, 0, wp.float64) for _ in range(2)]
    batch = HistoryBatch(args.seed, 0, count)
    parameter = TransportParameter("log-material-density", 0)

    def forward(validate: bool) -> None:
        trace_histories(
            positions,
            directions,
            weights,
            density,
            batch=batch,
            workspace=ws,
            out_pixel=pixel,
            out_score=score,
            out_energy=energy,
            out_events=events,
            out_status=status,
            stream=stream,
            validate=validate,
        )

    def reverse(validate: bool) -> None:
        derivative_histories(
            positions,
            directions,
            weights,
            density,
            parameter=parameter,
            batch=batch,
            workspace=ws,
            out_pixel=derivative_pixel,
            out_derivative=derivative,
            out_status=derivative_status,
            stream=stream,
            validate=validate,
        )

    def reduce(validate: bool) -> None:
        history_moments(
            pixel,
            score,
            status,
            batch=batch,
            workspace=moments,
            out_mean=mean,
            out_variance_of_mean=variance,
            stream=stream,
            validate=validate,
        )

    def diagnostics() -> dict[str, object]:
        # All storage belongs to the prepared workload. These explicit host
        # downloads happen after timing and cuda_profiler_stop, not in a launch
        # or the measured CUDA region. Reverse/reduction operations have their
        # own destinations, so these remain the complete forward batch outputs.
        sampled_events = events.numpy()
        sampled_scores = score.numpy()
        sampled_pixels = pixel.numpy()
        sampled_status = status.numpy()
        sample_variance = float(np.var(sampled_scores, ddof=1)) if count > 1 else None
        return {
            "estimator": spec.estimator,
            "original_histories": count,
            "seed": batch.seed,
            "first_history": batch.first_history,
            "sampling_unit": (
                "One complete forward batch; repeated timing launches replay these same "
                "identities and provide no additional independent samples."
            ),
            "events": {
                "meaning": "scattering collisions"
                if spec.estimator == "continuous-absorption"
                else "scattering and terminal absorption collisions",
                "total": int(np.sum(sampled_events, dtype=np.int64)),
                "mean_per_history": float(np.mean(sampled_events)),
                "median_per_history": float(np.median(sampled_events)),
                "p95_per_history": float(np.quantile(sampled_events, 0.95)),
                "p99_per_history": float(np.quantile(sampled_events, 0.99)),
                "maximum_per_history": int(np.max(sampled_events)),
                "zero_event_histories": int(np.count_nonzero(sampled_events == 0)),
            },
            "terminal_status_counts": {
                str(code): int(np.count_nonzero(sampled_status == code)) for code in range(7)
            },
            "detector_hit_histories": int(np.count_nonzero(sampled_pixels >= 0)),
            "nonzero_score_histories": int(np.count_nonzero(sampled_scores)),
            "total_detector_score": float(np.sum(sampled_scores, dtype=np.float64)),
            "mean_detector_score_per_history": float(np.mean(sampled_scores)),
            "sample_variance_per_history": sample_variance,
            "variance_of_mean": None if sample_variance is None else sample_variance / count,
            "score_scope": (
                "Total detector score over all pixels, with misses and absorbed histories "
                "included as zero; this is not the sum of marginal pixel variances."
            ),
            "diagnostic_device_to_host_bytes": count * (8 + 4 + 4 + 4),
            "diagnostic_transfer_scope": "After all timings and outside the CUDA profiler region.",
        }

    return Workload(
        {
            "transport_forward": forward,
            "transport_density_replay": reverse,
            "transport_moments": reduce,
        },
        ws.check_status,
        ws.scratch_bytes + moments.scratch_bytes,
        "Repeated deterministic central ray, IID transport streams; "
        "fixed batch replay for matched timings. Concentrated direct hits "
        "intentionally stress tally contention.",
        diagnostics,
    )


def contention() -> dict[str, object]:
    try:
        process = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False
        )
        return {
            "time_ns": time.time_ns(),
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"time_ns": time.time_ns(), "unavailable": str(error)}


def summary(values: list[float]) -> dict[str, object]:
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "median_ms": statistics.median(values),
        "iqr_ms": quartiles[2] - quartiles[0],
        "samples_ms": values,
    }


def main() -> None:
    parser = experiment_parser(__file__, __doc__, configuration=False)
    parser.add_argument("--case", choices=("projection", "spectral", "transport"), required=True)
    parser.add_argument("--operation", help="time only this named operation; warm all dependencies")
    parser.add_argument(
        "--estimator",
        choices=("analogue", "continuous-absorption"),
        default="analogue",
        help="transport estimator; other workloads do not use this selection",
    )
    for name, default in (
        ("side", 256),
        ("grid", 64),
        ("samples", 128),
        ("histories", 65536),
        ("materials", 2),
        ("energies", 32),
        ("iterations", 10),
        ("repeats", 7),
        ("warmup", 3),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--seed", type=int, default=193)
    args = parser.parse_args()
    if (
        any(
            getattr(args, name) <= 0
            for name in (
                "side",
                "grid",
                "samples",
                "histories",
                "materials",
                "energies",
                "iterations",
                "warmup",
            )
        )
        or args.repeats < 2
    ):
        parser.error("sizes, iterations and warmup must be positive; repeats must be at least two")
    output = private_output(args.output, ROOT)
    config = dict(vars(args))
    config["output"] = str(output)
    sources = experiment_sources(__file__, None)
    with RunRecorder(output, configuration=config, sources=sources) as run:
        wp: Any = importlib.import_module("warp")
        np: Any = importlib.import_module("numpy")
        nvtx: Any = importlib.import_module("nvtx")
        wp.init()
        device = wp.get_device(args.device)
        if not device.is_cuda:
            raise ValueError("profiling requires an actual CUDA device")
        run.write_json("contention-before.json", contention())
        stream = wp.Stream(device)
        with wp.ScopedStream(stream):
            workload = {"projection": projection, "spectral": spectral, "transport": transport}[
                args.case
            ](args, wp, np, stream)
            for operation in workload.operations.values():
                operation(True)
            for _ in range(args.warmup):
                for operation in workload.operations.values():
                    operation(False)
            workload.check()
            if args.operation is not None and args.operation not in workload.operations:
                raise ValueError(f"unknown operation; choose one of {tuple(workload.operations)}")
            start, stop = wp.Event(device, enable_timing=True), wp.Event(device, enable_timing=True)
            rows: dict[str, object] = {}
            wp.cuda_profiler_start(device)
            try:
                for name, operation in workload.operations.items():
                    if args.operation is not None and name != args.operation:
                        continue
                    elapsed: list[float] = []
                    wall: list[float] = []
                    for _ in range(args.repeats):
                        wp.synchronize_stream(stream)
                        with nvtx.annotate(name, domain="dpt-library"):
                            begin = time.perf_counter_ns()
                            stream.record_event(start)
                            for _ in range(args.iterations):
                                operation(False)
                            stream.record_event(stop)
                            wp.synchronize_event(stop)
                            wall.append((time.perf_counter_ns() - begin) / 1e6 / args.iterations)
                            elapsed.append(
                                float(wp.get_event_elapsed_time(start, stop)) / args.iterations
                            )
                    rows[name] = {"cuda_event": summary(elapsed), "host_wall": summary(wall)}
            finally:
                wp.synchronize_stream(stream)
                wp.cuda_profiler_stop(device)
            workload.check()
            if workload.diagnostics is not None:
                diagnostics = workload.diagnostics()
                variance_of_mean = diagnostics.get("variance_of_mean")
                forward_timing = rows.get("transport_forward")
                if isinstance(variance_of_mean, float) and isinstance(forward_timing, dict):
                    forward_cuda = cast(float, forward_timing["cuda_event"]["median_ms"])
                    diagnostics["variance_of_mean_times_forward_median_ms"] = (
                        variance_of_mean * forward_cuda
                    )
                    diagnostics["variance_cost_scope"] = (
                        "Estimated total-score mean variance times warmed forward CUDA time "
                        "for this fixed batch; shared-device contention limits comparisons. "
                        "History throughput alone does not measure useful precision."
                    )
                run.write_json("forward-history-diagnostics.json", diagnostics)
        run.write_json("timings.json", rows)
        run.write_json("contention-after.json", contention())
        run.set_metadata(
            device=str(device),
            device_name=device.name,
            architecture=device.arch,
            warp=wp.__version__,
            cuda_driver=device.runtime.driver_version,
            scratch_bytes=workload.scratch_bytes,
            fixture=workload.description,
            timing=(
                "Warmed unchecked calls; per-call CUDA event interval and host wall time. "
                "Event intervals can include host submission gaps. "
                "No setup, compilation, status or history-diagnostic transfers inside timing. "
                "Shared-device timings may be contended; inspect the recorded device snapshots."
            ),
            acceptance=(
                "Timing execution with status checks; "
                "independent correctness and sanitizer acceptance are separate."
            ),
        )


if __name__ == "__main__":
    main()
