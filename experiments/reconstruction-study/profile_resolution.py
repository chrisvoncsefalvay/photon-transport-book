"""Profile one gradient/value pair at a completed accepted reconstruction.

No solve, reference discovery or image formation outside the canonical engines.
Run under Nsight Systems with --capture-range=cudaProfilerApi; setup, warmup,
ordinary timings and output are outside the one explicitly captured pair.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import time
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, cast

from _resolution import checked_file, checked_json, load_fitting_inputs
from _study import complete_record, digest, geometry_in_object, recorded_array
from probe_acquisition import helpers

from dpt.examples._common import load_case
from dpt.examples.reconstruction import Reconstruction, SolverSettings
from dpt.experiments import RunRecorder, experiment_sources, private_output, repository_root
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.material_reconstruction import (
    MaterialReconstruction,
    MaterialReconstructionSettings,
    MaterialReconstructionView,
)

np: Any = importlib.import_module("numpy")


def verify_fit(
    fit: Path, observations: Path, root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    """Bind only completed fit outputs and the prepared fitting-input closure."""
    record = complete_record(fit)
    if record["configuration"]["method"] not in ("scalar_poisson", "spectral_metric"):
        raise ValueError("profile accepts only scalar_poisson or spectral_metric fits")
    public, inputs = load_fitting_inputs(observations, root)
    config = record["configuration"]
    if (
        config["observation_run_sha256"] != digest(observations / "run.json")
        or config["observation_manifest_sha256"]
        != digest(observations / "public-observations.json")
        or config["inverse_grid"] != public["inverse_grid"]
    ):
        raise ValueError("profile observations or grid differ from the completed fit")
    for name in record["output_sha256"]:
        checked_file(fit, name, record["output_sha256"])
    for name in record["source_sha256"]:
        checked_file(fit / "sources", name, record["source_sha256"])
    # Require exact current numerical code, including the full sibling closure.
    current = experiment_sources(__file__)
    current.update(
        {f"reconstruction-study/{p.name}": p for p in Path(__file__).parent.glob("*.py")}
    )
    current["helpers/spectral-run.py"] = Path(helpers().__file__)
    for name, path in current.items():
        key = name
        if name == Path(__file__).relative_to(root).as_posix():
            key = f"reconstruction-study/{Path(__file__).name}"
        if record["source_sha256"].get(key) != digest(path):
            raise ValueError(f"profile code differs from the completed fit: {key}")
    inputs.update(
        {
            "completed-fit/run.json": fit / "run.json",
            "completed-fit/fields.npy": fit / "fields.npy",
            "completed-fit/predictions.npy": fit / "predictions.npy",
            "completed-fit/solver-result.json": fit / "solver-result.json",
        }
    )
    return record, public, inputs


def scalar_profile_case(
    fit: Path, observations: Path, record: dict[str, Any], public: dict[str, Any]
) -> dict[str, Any]:
    """Relocate only checked scalar inputs; preserve the fitted mathematical case."""
    source = checked_file(fit / "sources", "scalar-case.json", record["source_sha256"])
    case = copy.deepcopy(json.loads(source.read_text()))
    settings = case["reconstruction"]
    if "evaluation" in case or "evaluation" in settings:
        raise ValueError("profiling forbids evaluation inputs")
    if settings["grid"] != public["inverse_grid"]:
        raise ValueError("scalar profile grid changed")
    geometry = DetectorGeometry(**{k: tuple(v) for k, v in public["geometry"].items()})
    replacements = {
        settings["initial_volume"]: checked_file(
            fit / "sources", "initial-scalar.npy", record["source_sha256"]
        )
    }
    for view, supplied in zip(settings["views"], public["views"], strict=True):
        pose = RigidTransform(**{k: tuple(v) for k, v in supplied["pose"].items()})
        expected = json.loads(json.dumps(asdict(geometry_in_object(geometry, pose))))
        if view["geometry"] != expected:
            raise ValueError("scalar profile rays differ from the recorded observations")
        replacements[view["counts"]] = observations / supplied["scalar_counts"]
        replacements[view["open_beam"]] = observations / public["open_beam"]
    if set(replacements) != set(case["arrays"]):
        raise ValueError("scalar case contains inputs outside the fitted observation closure")
    for name, path in replacements.items():
        if digest(path) != case["arrays"][name]["sha256"]:
            raise ValueError(f"scalar profile array changed: {name}")
        case["arrays"][name]["path"] = str(path.resolve())
    return case


def union_bytes(spans: list[tuple[int, int]]) -> int:
    """Count a union of byte intervals; shared views must not be counted twice."""
    total = 0
    right = 0
    for start, end in sorted(spans):
        if start < 0 or end < start:
            raise ValueError("invalid allocation span")
        total += max(0, end - max(start, right))
        right = max(right, end)
    return total


def buffer_accounting(engine: Any, torch: Any, wp: Any) -> dict[str, Any]:
    """Enumerate reachable DPT buffers; exclude framework runtime/global objects."""
    seen: set[int] = set()
    arrays: list[dict[str, Any]] = []
    spans: dict[str, list[tuple[int, int]]] = {}
    excluded = {"context", "_context", "wp", "np", "torch", "stream", "device", "kernels"}

    def visit(value: Any, label: str) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if torch.is_tensor(value):
            if not value.is_cuda:
                return
            storage = value.untyped_storage()
            pointer, size, device = storage.data_ptr(), storage.nbytes(), str(value.device)
            dtype = str(value.dtype)
        elif isinstance(value, wp.array):
            if not value.device.is_cuda:
                return
            pointer, size, device = (
                value.ptr,
                value.size * wp.types.type_size_in_bytes(value.dtype),
                str(value.device),
            )
            dtype = str(value.dtype)
        else:
            members: Iterable[tuple[Any, Any]]
            if isinstance(value, dict):
                members = cast(dict[Any, Any], value).items()
            elif isinstance(value, (tuple, list)):
                members = enumerate(cast(list[Any] | tuple[Any, ...], value))
            elif (
                type(value).__module__.startswith("dpt.")
                and is_dataclass(value)
                and not isinstance(value, type)
            ):
                members = (
                    (field.name, getattr(value, field.name)) for field in dataclass_fields(value)
                )
            elif type(value).__module__.startswith("dpt.") and hasattr(value, "__dict__"):
                members = cast(dict[str, Any], vars(value)).items()
            else:
                return
            for name, child in members:
                if name not in excluded and not str(name).endswith("kernels"):
                    visit(child, f"{label}.{name}")
            return
        arrays.append({"name": label, "bytes": int(size), "dtype": dtype, "device": device})
        rows = spans.setdefault(device, [])
        if size:
            if pointer is None or int(pointer) <= 0:
                raise ValueError("positive-size device array has no valid pointer")
            rows.append((int(pointer), int(pointer + size)))

    visit(engine, "engine")
    return {
        "arrays": arrays,
        "union_visible_device_bytes": {name: union_bytes(rows) for name, rows in spans.items()},
        "scope": (
            "Reachable DPT-owned tensor storage/Warp array spans, with aliases unioned. "
            "Excludes framework/context/JIT allocations and non-introspectable workspaces; "
            "not a full process allocation total, peak measurement or traffic estimate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "observations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = repository_root(__file__)
    output = private_output(args.output, root)
    record, public, inputs = verify_fit(args.fit, args.observations, root)
    fields = recorded_array(args.fit, "fields.npy", record)
    expected_predictions = recorded_array(args.fit, "predictions.npy", record)
    result = checked_json(args.fit, "solver-result.json", record)
    method = record["configuration"]["method"]
    spectral = method == "spectral_metric"
    scalar_case = (
        None if spectral else scalar_profile_case(args.fit, args.observations, record, public)
    )
    if not spectral:
        inputs.update(
            {
                "completed-fit/scalar-case.json": args.fit / "sources/scalar-case.json",
                "completed-fit/initial-scalar.npy": args.fit / "sources/initial-scalar.npy",
            }
        )
    library = helpers()
    sources = experiment_sources(
        __file__,
        extra={
            **inputs,
            **{f"reconstruction-study/{p.name}": p for p in Path(__file__).parent.glob("*.py")},
            "helpers/spectral-run.py": Path(library.__file__),
        },
    )
    configuration: dict[str, Any] = {
        "method": method,
        "completed_fit_sha256": digest(args.fit / "run.json"),
        "observation_run_sha256": digest(args.observations / "run.json"),
        "field_sha256": record["output_sha256"]["fields.npy"],
        "warm_pairs": 1,
        "ordinary_pairs": 1,
        "captured_pairs": 1,
        "objective_relative_tolerance": 1e-10,
        "objective_absolute_tolerance": 1e-7,
        "prediction_check": "exact stored FP32 equality",
        "reference_or_generating_mean_access": False,
        "solve_called": False,
    }
    torch: Any = importlib.import_module("torch")
    wp: Any = importlib.import_module("warp")
    if not str(args.device).startswith("cuda:"):
        raise ValueError("profiling requires an explicitly selected CUDA device")
    with RunRecorder(output, configuration=configuration, sources=sources) as run:
        with torch.cuda.device(args.device), torch.no_grad():
            started = time.perf_counter()
            if spectral:
                grid = library.grid_from(public["inverse_grid"])
                _, physics, spec = library.load_physics(args.observations / "physics")
                geometry = DetectorGeometry(**{k: tuple(v) for k, v in public["geometry"].items()})
                entries = tuple(
                    MaterialReconstructionView(
                        geometry,
                        RigidTransform(**{k: tuple(v) for k, v in view["pose"].items()}),
                        np.load(args.observations / view["spectral_counts"], allow_pickle=False),
                        physics["weights"],
                        physics["response"],
                    )
                    for view in public["views"]
                )
                matrix = public["metric"]["matrix"]
                engine: Any = MaterialReconstruction(
                    grid=grid,
                    views=entries,
                    coefficients=physics["mu_mm_inv"],
                    spectral_spec=spec,
                    initial_fractions=fields,
                    settings=MaterialReconstructionSettings(**record["metadata"]["settings"]),
                    samples_per_ray=public["inverse_samples"],
                    material_metric=(matrix[0][0], matrix[0][1], matrix[1][1]),
                    device=args.device,
                )
            else:
                run.write_json("profile-case.json", scalar_case)
                case = load_case(output / "profile-case.json")
                engine = Reconstruction(case, device=args.device, torch=torch, wp=wp)
                if fields.shape != engine.grid.shape or (fields < 0).any():
                    raise ValueError("completed scalar field violates its grid/domain")
                engine.policy = SolverSettings(**record["metadata"]["settings"])
                engine.mu.copy_(torch.as_tensor(fields.reshape(-1), device=args.device))
            wp.synchronize_stream(engine.stream)
            setup_seconds = time.perf_counter() - started
            run.write_json("visible-buffer-accounting.json", buffer_accounting(engine, torch, wp))
            timings: list[dict[str, Any]] = []

            def evaluate(gradient: bool, phase: str) -> None:
                wp.synchronize_stream(engine.stream)
                tick = time.perf_counter()
                value = (
                    engine.evaluate(gradient=gradient)
                    if spectral
                    else engine.evaluate(engine.mu, engine.mu_wp, gradient=gradient)
                )
                wp.synchronize_stream(engine.stream)
                timings.append(
                    {
                        "phase": phase,
                        "gradient": gradient,
                        "seconds": time.perf_counter() - tick,
                        "objective": value,
                    }
                )
                if not math.isclose(value, result["final_objective"], rel_tol=1e-10, abs_tol=1e-7):
                    raise ValueError("profile objective differs from the completed accepted state")

            for phase in ("warmup", "ordinary"):
                for gradient in (True, False):
                    evaluate(gradient, phase)
            wp.cuda_profiler_start(args.device)
            try:
                for gradient in (True, False):
                    torch.cuda.nvtx.range_push(
                        "dpt-resolution-gradient" if gradient else "dpt-resolution-value"
                    )
                    try:
                        evaluate(gradient, "captured")
                    finally:
                        torch.cuda.nvtx.range_pop()
            finally:
                wp.cuda_profiler_stop(args.device)
            actual_fields = (
                engine.fractions_numpy()
                if spectral
                else engine.mu.cpu().numpy().reshape(fields.shape)
            )
            predictions = (
                np.stack(engine.predictions_numpy())
                if spectral
                else np.stack(
                    [view["prediction"].numpy().reshape(view["shape"]) for view in engine.views]
                )
            )
            if not np.array_equal(actual_fields, fields) or not np.array_equal(
                predictions, expected_predictions
            ):
                raise ValueError("profiling changed accepted fields or their recorded predictions")
            run.write_json("timings.json", timings)
            run.write_json(
                "profile.json",
                {
                    "method": method,
                    "grid": public["inverse_grid"],
                    "views": len(public["views"]),
                    "detector_shape": public["geometry"]["shape"],
                    "samples_per_ray": public["inverse_samples"],
                    "setup_seconds": setup_seconds,
                    "fields_and_predictions_bitwise_equal_to_completed_fit": True,
                    "device": args.device,
                    "gpu_name": torch.cuda.get_device_name(args.device),
                    "free_total_device_bytes": list(torch.cuda.mem_get_info(args.device)),
                    "torch_allocated_reserved_bytes": [
                        torch.cuda.memory_allocated(args.device),
                        torch.cuda.memory_reserved(args.device),
                    ],
                    "timing_scope": (
                        "Synchronous host-wall pair evaluations on a shared device; "
                        "ordinary pair excludes capture; captured timing includes "
                        "instrumentation. No repeatability or general speed claim."
                    ),
                    "static_algorithm": {
                        "residency": (
                            "Persistent device fields/views/scratch; small objective/status "
                            "reads synchronise each evaluation. Final arrays export only "
                            "after capture."
                        ),
                        "precision": (
                            "FP32 fields/path/count/gradient storage; FP64 ray geometry, "
                            "quadrature and scalar reductions; spectral energy accumulation "
                            "uses FP64 before checked FP32 storage."
                        ),
                        "adjoint": (
                            "Explicit first-order volume VJP, recomputing samples; "
                            "no ray-by-sample tape or image Jacobian."
                        ),
                        "atomics": (
                            "FP32 volume-gradient scatter accumulation; low-order "
                            "non-bitwise variation permitted in gradients. Forward "
                            "predictions remain independently checked."
                        ),
                        "work_shape": (
                            "Serial views/channels, bounded sample loops with per-ray "
                            "support rejection. Divergence, occupancy, actual memory "
                            "traffic and launch cost require the accompanying actual "
                            "profiler report."
                        ),
                        "memory_limit": (
                            "Torch allocator and free-memory values are shared-device "
                            "snapshots and omit Warp-owned allocator accounting; "
                            "not exclusive process peak."
                        ),
                    },
                },
            )


if __name__ == "__main__":
    main()
