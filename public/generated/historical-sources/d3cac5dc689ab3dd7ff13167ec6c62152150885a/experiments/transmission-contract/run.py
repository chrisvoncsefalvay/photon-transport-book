"""Execute the canonical CUDA operator, validate it, and record actual artefacts.

No physical expression is reimplemented here: this runner chooses mathematical
stress inputs, invokes the library and its independent oracle, and plots records.
Generated records are private until the separate artefact/release checks accept
them. The warmed benchmark uses CUDA events and excludes setup and compilation.
"""

from __future__ import annotations

import importlib
import json
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from dpt.experiments import (
    RunRecorder,
    experiment_parser,
    experiment_sources,
    private_output,
    repository_root,
)
from dpt.transmission import (
    TransmissionSpec,
    optical_depth_from_removed,
    prepare_transmission,
    transmission_vjp,
    transmit,
)
from dpt.validation.transmission import (
    MIN_SUBNORMAL,
    U32,
    U64,
    analytic_cases,
    exact_input,
    forward_reference,
    inverse_reference,
    value_error,
    vjp_reference,
)

ROOT = repository_root(__file__)
OUTPUT_NAMES = ("T", "counts", "log_T", "removed")
OUTPUT_ARGUMENTS = ("out_T", "out_counts", "out_log_T", "out_removed")


@dataclass(frozen=True)
class Config:
    schema_version: int
    sweep_points: int
    benchmark_sizes: list[int]
    block_dims: list[int]
    regimes: list[str]
    batches: int
    iterations: int
    warmup_iterations: int

    def validate(self) -> None:
        if self.schema_version != 1 or not 3 <= self.sweep_points <= 4097:
            raise ValueError("schema_version must be 1; sweep_points must be 3..4097")
        if not self.benchmark_sizes or any(
            type(n) is not int or not 1 <= n <= 2**31 - 1 for n in self.benchmark_sizes
        ):
            raise ValueError("benchmark sizes must be positive int32 lengths")
        if not self.block_dims or any(n not in (128, 256) for n in self.block_dims):
            raise ValueError("block_dims must contain only 128 and/or 256")
        if not self.regimes or any(r not in ("ordinary", "tail", "mixed") for r in self.regimes):
            raise ValueError("unknown benchmark numerical regime")
        if self.batches < 5 or self.iterations < 1 or self.warmup_iterations < 1:
            raise ValueError("use at least five batches and positive iteration counts")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def command_output(arguments: list[str]) -> str:
    try:
        result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError as error:
        return f"unavailable: {error}"
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def metadata(wp: Any, device: Any, config: Config) -> dict[str, Any]:
    runtime = device.runtime
    return {
        "command": sys.argv,
        "source_commit": command_output(["git", "rev-parse", "HEAD"]),
        "source_status": command_output(["git", "status", "--short"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "warp": wp.__version__,
        "numpy": importlib.import_module("numpy").__version__,
        "matplotlib": importlib.import_module("matplotlib").__version__,
        "device": str(device),
        "device_name": device.name,
        "compute_capability": device.arch,
        "cuda_toolkit": runtime.toolkit_version,
        "cuda_driver_api": runtime.driver_version,
        "nvcc": command_output(["nvcc", "--version"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "numerics": {
            "storage": "binary32",
            "intermediates": "binary64 products, sums and exponential above optical depth 64",
            "fast_math": False,
            "fuse_fp": True,
            "oracle_decimal_digits": 100,
            "reduction_tile_size": 256,
        },
        "input_provenance": "deterministic mathematical stress inputs; no anatomical data",
    }


# region book:transmission-experiment
def evaluate_sweep(wp: Any, np: Any, depths: Any, beam: float, device: Any) -> dict[str, Any]:
    """Execute all four canonical outputs, then check every stored value."""
    workspace = prepare_transmission(
        TransmissionSpec(beam="scalar"), device=str(device), max_pixels=len(depths)
    )
    optical_depth = wp.array(depths, dtype=wp.float32, device=device)
    outputs = [wp.empty(len(depths), dtype=wp.float32, device=device) for _ in OUTPUT_NAMES]
    transmit(
        optical_depth,
        beam,
        workspace=workspace,
        **dict(zip(OUTPUT_ARGUMENTS, outputs, strict=True)),
    )
    wp.synchronize_stream(workspace.stream)
    measured = [output.numpy() for output in outputs]
    cases: list[dict[str, Any]] = []
    for index, depth in enumerate(depths):
        reference = forward_reference(float(depth), beam)
        checks = {}
        for name, values in zip(OUTPUT_NAMES, measured, strict=True):
            result = value_error(
                float(values[index]), getattr(reference, name), counts=name == "counts"
            )
            exact_log = name != "log_T" or struct.pack("!f", float(values[index])) == struct.pack(
                "!f", -float(depth)
            )
            checks[name] = {
                "actual": float(values[index]),
                "reference_decimal": str(getattr(reference, name)),
                "ulps": result.ulps,
                "allowed_ulps": 0 if name == "log_T" else result.allowed_ulps,
                "passed": result.passed and exact_log,
            }
            if not result.passed or not exact_log:
                raise AssertionError(f"{name} failed at optical depth {depth}: {checks[name]}")
        cases.append({"optical_depth": float(depth), "checks": checks})
    return {
        "beam": float(np.float32(beam)),
        "cases": cases,
        "outputs": {
            name: values.tolist() for name, values in zip(OUTPUT_NAMES, measured, strict=True)
        },
        "optical_depths": depths.tolist(),
    }


# endregion book:transmission-experiment


def validation_run(wp: Any, np: Any, device: Any, config: Config) -> dict[str, Any]:
    points = config.sweep_points
    result: dict[str, Any] = {
        "schema_version": 1,
        "sweep": evaluate_sweep(wp, np, np.linspace(0, 20, points, dtype=np.float32), 1e6, device),
        "weak": evaluate_sweep(
            wp, np, np.geomspace(1e-12, 0.1, points, dtype=np.float32), 1.0, device
        ),
        "tail": evaluate_sweep(
            wp, np, np.linspace(80, 200, points, dtype=np.float32), 1e30, device
        ),
    }
    analytic = analytic_cases()
    result["analytic"] = evaluate_sweep(
        wp,
        np,
        np.array([float(case.optical_depth) for case in analytic], dtype=np.float32),
        1.0,
        device,
    )
    result["analytic"]["definitions"] = [asdict(case) for case in analytic]
    for definition in result["analytic"]["definitions"]:
        definition["optical_depth"] = str(definition["optical_depth"])
    decrements = np.geomspace(1e-12, 0.9, points, dtype=np.float32)
    workspace = prepare_transmission(device=str(device), max_pixels=points)
    delta = wp.array(decrements, dtype=wp.float32, device=device)
    restored = wp.empty(points, dtype=wp.float32, device=device)
    optical_depth_from_removed(delta, out_L=restored, workspace=workspace)
    measured = restored.numpy()
    inverse_cases: list[dict[str, Any]] = []
    for decrement, actual in zip(decrements, measured, strict=True):
        expected = inverse_reference(float(decrement))
        check = value_error(float(actual), expected)
        if not check.passed:
            raise AssertionError(f"inverse decrement failed for {decrement}")
        inverse_cases.append(
            {"decrement": float(decrement), "actual": float(actual), "ulps": check.ulps}
        )
    result["inverse"] = inverse_cases
    # Diagnostic postprocessing of actual stored library output, deliberately
    # showing cancellation. This is never used as a transport implementation.
    stored_transmission = np.asarray(result["weak"]["outputs"]["T"], dtype=np.float32)
    result["weak"]["diagnostic_one_minus_stored_T"] = (
        np.float32(1.0) - stored_transmission
    ).tolist()
    result["status"] = "passed"
    return result


def render_figures(result: dict[str, Any], destination: Path) -> list[str]:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    np = importlib.import_module("numpy")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "dpt-transmission-v1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
        }
    )
    files: list[str] = []

    def save(figure: Any, name: str) -> None:
        figure.savefig(
            destination / name, format="svg", metadata={"Date": None}, bbox_inches="tight"
        )
        figure.savefig(destination / Path(name).with_suffix(".png"), dpi=150, bbox_inches="tight")
        plt.close(figure)
        files.append(name)

    sweep = result["sweep"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), layout="constrained")
    for ax, key, title in zip(
        axes,
        ("T", "counts", "log_T"),
        ("Transmission", "Expected counts", "Log transmission"),
        strict=True,
    ):
        ax.plot(sweep["optical_depths"], sweep["outputs"][key], color="#0072B2")
        ax.set(xlabel="Optical depth L", ylabel=title, title=title)
        if key != "log_T":
            ax.set_yscale("log")
    fig.suptitle("Canonical CUDA execution · open-beam expectation = 10⁶")
    save(fig, "transmission-sweep.svg")

    weak = result["weak"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    axes[0].loglog(
        weak["optical_depths"],
        weak["outputs"]["removed"],
        color="#0072B2",
        label="CUDA removed-primary fraction",
    )
    oracle = [float(case["checks"]["removed"]["reference_decimal"]) for case in weak["cases"]]
    axes[0].loglog(
        weak["optical_depths"][::8],
        oracle[::8],
        "o",
        color="#D55E00",
        markersize=4,
        fillstyle="none",
        label="100-digit reference",
    )
    axes[0].set(xlabel="Optical depth L", ylabel="Removed-primary fraction")
    naive = np.asarray(weak["diagnostic_one_minus_stored_T"])
    nonzero = naive > 0
    axes[0].loglog(
        np.asarray(weak["optical_depths"])[nonzero],
        naive[nonzero],
        linestyle="--",
        color="#444444",
        label="1 - stored T (FP32 diagnostic)",
    )
    zero_count = int(np.count_nonzero(~nonzero))
    axes[0].text(
        0.97,
        0.05,
        f"Subtraction gave zero at {zero_count} sampled depths\n(zeros omitted on log axes)",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    axes[0].legend(loc="upper left", fontsize=9)
    inverse = result["inverse"]
    axes[1].semilogx(
        [row["decrement"] for row in inverse],
        [row["ulps"] for row in inverse],
        color="#009E73",
        marker=".",
        markersize=3,
    )
    axes[1].set(xlabel="Supplied decrement δ", ylabel="Inverse error (FP32 ULPs)")
    maximum_ulp = max(row["ulps"] for row in inverse)
    axes[1].set_ylim(-0.1, max(1, maximum_ulp + 0.5))
    axes[1].set_yticks(range(maximum_ulp + 2))
    fig.suptitle("Weak attenuation · direct decrement and independently checked inverse")
    save(fig, "transmission-weak-attenuation.svg")

    tail = result["tail"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for ax, key, title in zip(
        axes, ("T", "counts"), ("Stored transmission", "Stored expected counts"), strict=True
    ):
        values = np.asarray(tail["outputs"][key])
        positive = values > 0
        depths = np.asarray(tail["optical_depths"])
        ax.semilogy(depths[positive], values[positive], color="#0072B2")
        zero_depths = depths[~positive]
        if zero_depths.size:
            ax.axvline(float(zero_depths[0]), color="#D55E00", linestyle="--")
            ax.text(
                0.97,
                0.96,
                f"First stored zero in sweep: L = {zero_depths[0]:g}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )
        ax.set(xlabel="Optical depth L", ylabel=title, xlim=(80, 200))
    fig.suptitle("Tail range · open-beam expectation = 10³⁰ · zero values omitted from log axes")
    save(fig, "transmission-tail-rescue.svg")
    return files


def timed_batches(
    wp: Any, stream: Any, operation: Callable[[], None], config: Config
) -> dict[str, Any]:
    for _ in range(config.warmup_iterations):
        operation()
    wp.synchronize_stream(stream)
    start = wp.Event(device=stream.device, enable_timing=True)
    stop = wp.Event(device=stream.device, enable_timing=True)
    samples: list[float] = []
    for _ in range(config.batches):
        stream.record_event(start)
        for _ in range(config.iterations):
            operation()
        stream.record_event(stop)
        samples.append(float(wp.get_event_elapsed_time(start, stop)) / config.iterations)
    return {
        "timing_kind": "CUDA event",
        "cuda_event_ms_per_call": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "batches": config.batches,
        "iterations_per_batch": config.iterations,
        "includes": "device work plus any host submission gaps between consecutive launches",
        "excludes": "allocation, H2D upload, compilation, warmup, content validation",
    }


def wall_batches(
    wp: Any, stream: Any, operation: Callable[[], None], config: Config
) -> dict[str, Any]:
    for _ in range(config.warmup_iterations):
        operation()
    wp.synchronize_stream(stream)
    samples: list[float] = []
    for _ in range(config.batches):
        start = time.perf_counter_ns()
        for _ in range(config.iterations):
            operation()
            wp.synchronize_stream(stream)
        samples.append((time.perf_counter_ns() - start) / 1e6 / config.iterations)
    return {
        "timing_kind": "checked wall clock",
        "wall_ms_per_call": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "batches": config.batches,
        "iterations_per_batch": config.iterations,
        "includes": "host validation, device content scans, status readback and completed output",
        "excludes": "allocation, input upload, compilation and warmup",
    }


def benchmark_case(
    wp: Any, np: Any, device: Any, size: int, block: int, regime: str, config: Config
) -> list[dict[str, Any]]:
    # The same 256-value pattern supports a bounded independent correctness check
    # before timing any size; this is a declared synthetic numerical workload.
    ordinary = np.linspace(0, 20, 256, dtype=np.float32)
    tail = np.linspace(64, 200, 256, dtype=np.float32)
    pattern = ordinary if regime == "ordinary" else tail.copy()
    if regime == "mixed":
        pattern[::2] = ordinary[::2]
    host_depth = np.resize(pattern, size)
    depth = wp.array(host_depth, dtype=wp.float32, device=device)
    outputs = [wp.empty(size, dtype=wp.float32, device=device) for _ in range(4)]
    seed = wp.ones(size, dtype=wp.float32, device=device)
    gradient = wp.empty(size, dtype=wp.float32, device=device)
    rows: list[dict[str, Any]] = []
    prechecks: dict[str, Any] = {}
    references = [forward_reference(float(v), 1000.0) for v in pattern[: min(size, 256)]]

    def record(
        label: str,
        operation: Callable[[], None],
        workspace: Any,
        logical_bytes: int,
        *,
        checked: bool = False,
    ) -> None:
        timer = wall_batches if checked else timed_batches
        timing = timer(wp, workspace.stream, operation, config)
        workspace.check_status()
        rows.append(
            {
                "pixels": size,
                "block_dim": block,
                "regime": regime,
                "operation": label,
                "scratch_bytes": workspace.scratch_bytes,
                "logical_array_bytes_estimate": logical_bytes,
                "bytes_note": "minimum accesses; excludes caches/status/FP64 reduction scratch",
                "precheck": prechecks.copy(),
                **timing,
            }
        )

    for mode in ("scalar", "per-pixel", "device-scalar"):
        prechecks.clear()
        workspace = prepare_transmission(
            TransmissionSpec(beam=mode, active_beam=mode != "scalar", block_dim=block),
            device=str(device),
            max_pixels=size,
        )
        beam = (
            1000.0
            if mode == "scalar"
            else wp.full(
                1 if mode == "device-scalar" else size, 1000.0, dtype=wp.float32, device=device
            )
        )
        beam_gradient = (
            None
            if mode == "scalar"
            else wp.empty(1 if mode == "device-scalar" else size, dtype=wp.float32, device=device)
        )
        kwargs = dict(zip(OUTPUT_ARGUMENTS, outputs, strict=True))
        transmit(depth, beam, workspace=workspace, **kwargs)
        for name, output in zip(OUTPUT_NAMES, outputs, strict=True):
            # A diagnostic read occurs before timing; nothing is downloaded inside it.
            values = output[: len(references)].numpy()
            for actual, reference, input_depth in zip(
                values, references, pattern[: len(references)], strict=True
            ):
                check = value_error(
                    float(actual), getattr(reference, name), counts=name == "counts"
                )
                exact_log = name != "log_T" or struct.pack("!f", float(actual)) == struct.pack(
                    "!f", -float(input_depth)
                )
                if not check.passed or not exact_log:
                    raise AssertionError(f"benchmark precheck failed: {name}, {mode}, {regime}")
        transmission_vjp(
            depth,
            beam,
            seed_counts=seed,
            out_grad_L=gradient,
            out_grad_n0=beam_gradient,
            workspace=workspace,
        )
        grad_values = gradient[: len(references)].numpy()
        for actual, value in zip(grad_values, pattern[: len(references)], strict=True):
            ref = vjp_reference(float(value), 1000.0, seed_counts=1.0)
            if abs(exact_input(float(actual)) - ref.grad_L) > ref.budget_L:
                raise AssertionError("benchmark VJP precheck failed")
        if mode == "per-pixel":
            assert beam_gradient is not None
            for actual, value in zip(
                beam_gradient[: len(references)].numpy(), pattern[: len(references)], strict=True
            ):
                ref = vjp_reference(float(value), 1000.0, seed_counts=1.0)
                if abs(exact_input(float(actual)) - ref.grad_n0) > ref.budget_n0:
                    raise AssertionError("per-pixel beam VJP precheck failed")
        elif mode == "device-scalar":
            assert beam_gradient is not None
            # This positive repeated-pattern sum uses the oracle's contributions;
            # its count-scaled bound avoids iterating over millions of host values.
            with localcontext() as context:
                context.prec = 100
                whole, remainder = divmod(size, len(pattern))
                contributions = [vjp_reference(float(v), seed_counts=1.0) for v in pattern]
                expected = Decimal(whole) * sum((v.grad_n0 for v in contributions), Decimal(0))
                expected += sum((v.grad_n0 for v in contributions[:remainder]), Decimal(0))
                element_budget = Decimal(whole) * sum(
                    (v.budget_n0 for v in contributions), Decimal(0)
                )
                element_budget += sum((v.budget_n0 for v in contributions[:remainder]), Decimal(0))
                count, levels = (size + 255) // 256, 1
                while count > 1:
                    count = (count + 255) // 256
                    levels += 1
                # Warp 1.17 tile_reduce.h combines each warp in five shuffle
                # steps, then thread zero adds eight warp totals serially.
                # The longest arithmetic path is therefore 5 + 7, not log2(256).
                addition_depth = 12 * levels
                product = addition_depth * U64
                gamma = product / (1 - product)
                # Seeds and factors here are positive, so magnitude == total.
                budget = element_budget + gamma * expected + U32 * expected + MIN_SUBNORMAL / 2
                if abs(exact_input(float(beam_gradient.numpy()[0])) - expected) > budget:
                    raise AssertionError("scalar beam VJP precheck failed")
                prechecks.update(
                    {
                        "scalar_gradient_reference": str(expected),
                        "scalar_gradient_absolute_budget": str(budget),
                        "scalar_reduction_addition_depth": addition_depth,
                    }
                )
        prechecks.update(
            {
                "forward_and_depth_samples": len(references),
                "log_transmission_check": "exact bits, including signed zero",
                "status": "passed",
            }
        )

        def forward(beam: Any = beam, workspace: Any = workspace, kwargs: Any = kwargs) -> None:
            transmit(depth, beam, workspace=workspace, validate=False, **kwargs)

        def backward(
            beam: Any = beam, workspace: Any = workspace, beam_gradient: Any = beam_gradient
        ) -> None:
            transmission_vjp(
                depth,
                beam,
                seed_counts=seed,
                out_grad_L=gradient,
                out_grad_n0=beam_gradient,
                workspace=workspace,
                validate=False,
            )

        def forward_backward(
            forward: Callable[[], None] = forward, backward: Callable[[], None] = backward
        ) -> None:
            forward()
            backward()

        def checked_forward(
            beam: Any = beam, workspace: Any = workspace, kwargs: Any = kwargs
        ) -> None:
            transmit(depth, beam, workspace=workspace, **kwargs)

        def checked_backward(
            beam: Any = beam, workspace: Any = workspace, beam_gradient: Any = beam_gradient
        ) -> None:
            transmission_vjp(
                depth,
                beam,
                seed_counts=seed,
                out_grad_L=gradient,
                out_grad_n0=beam_gradient,
                workspace=workspace,
            )

        record(
            f"forward-fused-{mode}", forward, workspace, size * (24 if mode == "per-pixel" else 20)
        )
        record(f"vjp-counts-{mode}", backward, workspace, size * (12 if mode == "scalar" else 20))
        record(
            f"forward-plus-vjp-{mode}",
            forward_backward,
            workspace,
            size * (32 if mode == "scalar" else (44 if mode == "per-pixel" else 40)),
        )
        with wp.ScopedCapture(stream=workspace.stream, force_module_load=False) as capture:
            forward_backward()

        def graph_replay(graph: Any = capture.graph, stream: Any = workspace.stream) -> None:
            wp.capture_launch(graph, stream=stream)

        record(
            f"graph-forward-plus-vjp-{mode}",
            graph_replay,
            workspace,
            size * (32 if mode == "scalar" else (44 if mode == "per-pixel" else 40)),
        )
        record(
            f"checked-forward-{mode}",
            checked_forward,
            workspace,
            size * (24 if mode == "per-pixel" else 20),
            checked=True,
        )
        record(
            f"checked-vjp-{mode}",
            checked_backward,
            workspace,
            size * (12 if mode == "scalar" else 20),
            checked=True,
        )
        if mode == "scalar":

            def matched_copy(stream: Any = workspace.stream) -> None:
                wp.copy(outputs[0], depth, stream=stream)

            record("matched-device-copy", matched_copy, workspace, size * 8)

            def separate(beam: Any = beam, workspace: Any = workspace) -> None:
                for argument, output in zip(OUTPUT_ARGUMENTS, outputs, strict=True):
                    transmit(depth, beam, workspace=workspace, validate=False, **{argument: output})

            record("forward-four-separate-launches", separate, workspace, size * 32)
            for argument, output in zip(OUTPUT_ARGUMENTS, outputs, strict=True):

                def selected(
                    argument: str = argument,
                    output: Any = output,
                    beam: Any = beam,
                    workspace: Any = workspace,
                ) -> None:
                    transmit(depth, beam, workspace=workspace, validate=False, **{argument: output})

                record(f"forward-{argument}", selected, workspace, size * 8)
    return rows


def main() -> int:
    parser = experiment_parser(__file__, __doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", help="override benchmark pixel counts")
    parser.add_argument("--iterations", type=int, help="override iterations per timing batch")
    arguments = parser.parse_args()
    config_data = json.loads(arguments.config.read_text(encoding="utf-8"))
    if arguments.sizes is not None:
        config_data["benchmark_sizes"] = arguments.sizes
    if arguments.iterations is not None:
        config_data["iterations"] = arguments.iterations
    config = Config(**config_data)
    config.validate()
    destination = private_output(arguments.output, ROOT)
    sources = experiment_sources(
        __file__,
        arguments.config,
        extra={
            name: ROOT / name
            for name in (
                "experiments/transmission-contract/profile_cuda.py",
                "experiments/transmission-contract/config.json",
            )
        },
    )
    started = time.monotonic()
    with RunRecorder(destination, configuration=asdict(config), sources=sources) as record:
        wp, np = importlib.import_module("warp"), importlib.import_module("numpy")
        wp.init()
        device = wp.get_device(arguments.device)
        if not device.is_cuda:
            raise ValueError("this experiment requires actual CUDA execution")
        record.set_metadata(**metadata(wp, device, config))
        result = validation_run(wp, np, device, config)
        record.write_json("validation.json", result)
        with tempfile.TemporaryDirectory(prefix="dpt-figures-") as temporary:
            figures = render_figures(result, Path(temporary))
            for name in figures:
                record.write_bytes(name, (Path(temporary) / name).read_bytes())
        record.set_metadata(figures=figures)
        print(f"Validated CUDA sweeps; figures and records: {destination}", flush=True)
        if arguments.benchmark:
            rows: list[dict[str, Any]] = []
            for size in config.benchmark_sizes:
                for block in config.block_dims:
                    for regime in config.regimes:
                        rows.extend(benchmark_case(wp, np, device, size, block, regime, config))
                        print(
                            f"Timed P={size}, block={block}, regime={regime}; {len(rows)} rows",
                            flush=True,
                        )
            record.write_json("benchmark.json", {"schema_version": 1, "rows": rows})
            record.set_metadata(benchmark_rows=len(rows))
        record.set_metadata(elapsed_seconds=time.monotonic() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
