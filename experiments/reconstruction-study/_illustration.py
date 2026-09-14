"""Host contracts for the single, separately seeded primary material illustration."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from _resolution import checked_file, checked_json, load_fitting_inputs, view_records
from _study import complete_record, digest, stationarity_gate, stationarity_result

from dpt.experiments import experiment_sources
from dpt.geometry import RigidTransform
from dpt.volumes import GridSpec

np: Any = importlib.import_module("numpy")
CONFIG_PATH = Path(__file__).with_name("primary64-config-v1.json")
FAMILIES = {
    "fitting-native": (144, (1024, 2048)),
    "fitting64": (144, (512, 1024)),
    "withheld-native": (24, (1024, 2048)),
    "withheld64": (24, (512, 1024)),
}


def validate_config(config: dict[str, Any]) -> None:
    if config != json.loads(CONFIG_PATH.read_text()):
        raise ValueError("configuration differs from the source-bound primary protocol")


def views(config: dict[str, Any], role: str) -> list[dict[str, Any]]:
    """Reuse the dense trajectory; compose its eight spaced poses with the fixed offset."""
    fitting = view_records("dense", config)
    if role == "fitting":
        return fitting
    if role != "withheld":
        raise ValueError("unknown illustration observation role")
    angle = math.radians(config["withheld_offset_degrees"])
    c, s = math.cos(angle), math.sin(angle)
    offset = RigidTransform(rotation=(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0))
    result: list[dict[str, Any]] = []
    for ring in range(3):
        for k in range(8):
            base = fitting[48 * ring + 6 * k]
            pose = offset.compose(
                RigidTransform(**{name: tuple(value) for name, value in base["pose"].items()})
            )
            result.append(
                {
                    "id": len(result),
                    "tilt_degrees": base["tilt_degrees"],
                    "yaw_degrees": 3.75 + 45.0 * k,
                    "pose": {name: list(value) for name, value in asdict(pose).items()},
                }
            )
    return result


def inverse_grid() -> GridSpec:
    return GridSpec(
        (64, 64, 64), (2.326171875, 2.326171875, 3.0), (-73.2744140625, -73.2744140625, -94.5)
    )


def noise_keys(config: dict[str, Any], role: str) -> list[list[int]]:
    if role not in ("fitting", "withheld"):
        raise ValueError("an explicit fitting or withheld noise role is required")
    return [
        [
            config["noise_root_seed"],
            config["noise_namespace_version"],
            config["case"],
            config["noise_roles"][role],
            config["noise_replicate"],
            view,
            channel,
        ]
        for view in range(144 if role == "fitting" else 24)
        for channel in range(3)
    ]


def counts_from_means(means: Any, config: dict[str, Any], role: str) -> Any:
    expected = (144 if role == "fitting" else 24, 3, *config["detector_shape_hw"])
    if means.shape != expected or not np.isfinite(means).all() or (means < 0).any():
        raise ValueError("invalid expected counts for the frozen observation role")
    result = np.empty(expected, dtype=np.float32)
    for key in noise_keys(config, role):
        view, channel = key[-2:]
        sample = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key))).poisson(
            means[view, channel].astype(np.float64)
        )
        if (sample >= 2**24).any():
            raise ValueError("count draw exceeds the exact FP32 integer range; never clip")
        result[view, channel] = sample
    return result


def qualification_coverage(reports: dict[str, dict[str, Any]]) -> int:
    """Require the full declared Cartesian product, not a trusted summary count."""
    if set(reports) != set(FAMILIES):
        raise ValueError("missing or additional primary qualification family")
    count = 0
    for name, (n, samples) in FAMILIES.items():
        report = reports[name]
        expected: set[tuple[str, int, int, int]] = set()
        for view in range(n):
            for kind, channels in (("scalar", range(1)), ("spectral", range(3))):
                for channel in channels:
                    expected.add(("doubled_" + kind, view, channel, samples[0]))
                    for sample in samples:
                        expected.add(("independent_" + kind, view, channel, sample))
        actual: set[tuple[Any, ...]] = set()
        for gate in report["gates"]:
            key = tuple(gate[k] for k in ("role", "view", "channel", "samples"))
            p95, maximum = gate["p95_poisson_sd"], gate["maximum_poisson_sd"]
            if (
                key in actual
                or not gate["passed"]
                or not (math.isfinite(p95) and math.isfinite(maximum))
                or not (0 <= p95 <= maximum and p95 < 0.05 and maximum < 0.2)
            ):
                raise ValueError("duplicate, failed or invalid primary sampling gate")
            actual.add(key)
        if (
            actual != expected
            or not report["passed"]
            or report["gate_count"] != len(expected)
            or tuple(report["samples"]) != samples
            or report["error_exposure_photons_per_ray"] != 800000.0
        ):
            raise ValueError("primary sampling coverage or declared exposure changed")
        count += len(actual)
    if count != 4032:
        raise ValueError("incorrect complete primary gate count")
    return count


def verify_physics(
    physics: dict[str, Any], metadata: dict[str, Any], config: dict[str, Any]
) -> None:
    pins = config["physics_array_pins"]
    if metadata["model_id"] != config["physics_model_id"] or set(physics) != set(pins):
        raise ValueError("the openly reproduced physical model differs")
    for name, pin in pins.items():
        array = physics[name]
        if (
            list(array.shape) != pin["shape"]
            or array.dtype.str != pin["dtype"]
            or hashlib.sha256(array.tobytes(order="C")).hexdigest() != pin["sha256_c_order_bytes"]
        ):
            raise ValueError(f"reproduced physical array differs from its fixed model: {name}")


def source_files(script: Path, extra: dict[str, Path] | None = None) -> dict[str, Path]:
    root = script.resolve().parents[2]
    return experiment_sources(
        script,
        CONFIG_PATH,
        extra={
            **(extra or {}),
            **{f"reconstruction-study/{p.name}": p for p in script.parent.glob("*.py")},
            "helpers/spectral-run.py": root / "experiments/spectral-reconstruction/run.py",
            "primary-config.json": CONFIG_PATH,
        },
    )


def loaded_source_identity(root: Path) -> dict[str, dict[str, str]]:
    """Reject accidental imports from an installed or private sibling DPT tree."""
    root = root.resolve()
    study = root / "experiments/reconstruction-study"
    study_names = {p.stem for p in study.glob("*.py")}
    result: dict[str, dict[str, str]] = {}
    for name, module in tuple(sys.modules.items()):
        if name == "dpt" or name.startswith("dpt."):
            base = root / "python"
            relative = Path(*name.split("."))
        elif name in study_names:
            base, relative = study, Path(name)
        else:
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            raise ValueError(f"loaded source module has no file identity: {name}")
        path = Path(filename).resolve()
        expected = base / relative.with_suffix(".py")
        if not expected.is_file():
            expected = base / relative / "__init__.py"
        if path != expected.resolve():
            raise ValueError(f"loaded source module is outside this execution tree: {name}")
        if name == "dpt" and [Path(p).resolve() for p in module.__path__] != [root / "python/dpt"]:
            raise ValueError("loaded DPT package search path differs from this execution tree")
        result[name] = {"path": str(path), "sha256": digest(path)}
    return result


def fitting_inputs(folder: Path, root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    loaded_source_identity(root)
    public, sources = load_fitting_inputs(folder, root)
    record = complete_record(folder)
    validate_config(public["config"])
    actual = [
        {k: row[k] for k in ("id", "tilt_degrees", "yaw_degrees", "pose")}
        for row in public["views"]
    ]
    config = public["config"]
    height, width = config["detector_shape_hw"]
    pitch, sid, sdd = (
        config["detector_pitch_mm"],
        config["source_isocentre_mm"],
        config["source_detector_mm"],
    )
    expected_geometry = {
        "source_mm": [0.0, -sid, 0.0],
        "origin_mm": [-(width - 1) * pitch / 2, sdd - sid, -(height - 1) * pitch / 2],
        "u": [1, 0, 0],
        "v": [0, 0, 1],
        "spacing_mm": [pitch, pitch],
        "shape": [height, width],
    }
    if (
        public["role"] != "primary fitting observations and admissible support only"
        or actual != views(public["config"], "fitting")
        or any(set(row["sha256"]) != {"spectral"} for row in public["views"])
        or public["inverse_grid"] != json.loads(json.dumps(asdict(inverse_grid())))
        or public["inverse_samples"] != 512
        or public["geometry"] != expected_geometry
        or public["metric"]["matrix"] != config["material_metric_matrix"]
    ):
        raise ValueError("primary fitting geometry, support or role changed")
    with np.load(folder / "physics/physics.npz", allow_pickle=False) as archive:
        verify_physics(
            dict(archive), json.loads((folder / "physics/metadata.json").read_text()), config
        )
    if not np.array_equal(
        np.load(folder / public["open_beam"], allow_pickle=False),
        np.full((height, width), config["incident_photons_per_ray"], dtype=np.float32),
    ):
        raise ValueError("fitting open beam differs from the fixed source normalisation")
    config_source = checked_file(folder / "sources", "primary-config.json", record["source_sha256"])
    if digest(config_source) != digest(CONFIG_PATH):
        raise ValueError("the source-bound primary configuration bytes changed")
    sources["primary-config.json"] = config_source
    # Read only source-bound gate JSON here; no reference or mean NPY payloads.
    reports: dict[str, dict[str, Any]] = {}
    for name in ("run", "qualification", *FAMILIES):
        key = f"qualification/{name}.json"
        path = checked_file(folder / "sources", key, record["source_sha256"])
        sources[key] = path
        if name in FAMILIES:
            reports[name] = json.loads(path.read_text())
    qualification_coverage(reports)
    qrun = json.loads(sources["qualification/run.json"].read_text())
    qreport = json.loads(sources["qualification/qualification.json"].read_text())
    if (
        qrun["status"] != "complete"
        or not qrun["sources_unchanged"]
        or not qrun["recorded_files_unchanged"]
        or not qreport["passed"]
        or qreport["counts_drawn"]
        or qreport["gate_count"] != 4032
        or digest(sources["qualification/qualification.json"])
        != qrun["output_sha256"]["qualification.json"]
    ):
        raise ValueError("primary qualification ancestry is incomplete")
    for name in FAMILIES:
        if (
            digest(sources[f"qualification/{name}.json"])
            != qrun["output_sha256"][f"qualification/{name}.json"]
        ):
            raise ValueError("primary family report differs from its completed qualification")
    for name, want in record["source_sha256"].items():
        if name.startswith(("python/dpt/", "reconstruction-study/")) or name in (
            "helpers/spectral-run.py",
            "primary-config.json",
        ):
            if qrun["source_sha256"].get(name) != want:
                raise ValueError("qualification and fitting bundle source identities differ")
    return public, sources


def verify_completed_fit(folder: Path, fitting: Path, public: dict[str, Any]) -> dict[str, Any]:
    record = complete_record(folder)
    fitting_record = complete_record(fitting)
    for name in record["output_sha256"]:
        checked_file(folder, name, record["output_sha256"])
    for name in record["source_sha256"]:
        checked_file(folder / "sources", name, record["source_sha256"])
    for name, want in fitting_record["source_sha256"].items():
        if name.startswith(("python/dpt/", "reconstruction-study/")) or name in (
            "helpers/spectral-run.py",
            "primary-config.json",
        ):
            if record["source_sha256"].get(name) != want:
                raise ValueError("fit source differs from the qualified observation source")
    config = record["configuration"]
    if (
        config["method"] != "spectral_metric"
        or config["config"] != public["config"]
        or config["inverse_grid"] != public["inverse_grid"]
        or config["observation_run_sha256"] != digest(fitting / "run.json")
        or config["observation_manifest_sha256"] != digest(fitting / "public-observations.json")
        or config["reference_or_generating_mean_access"]
    ):
        raise ValueError("completed primary fit is not bound to these fitting inputs")
    result = checked_json(folder, "solver-result.json", record)
    policy = public["config"]
    accepted = result.get("accepted_steps")
    termination = result.get("termination")
    if (
        type(accepted) is not int
        or not 0 <= accepted <= policy["pilot_steps"]
        or result.get("poisson_count_predictions_refreshed") is not True
        or result.get("reference_or_generating_mean_access") is not False
        or result.get("field_units") != "dimensionless fractions"
        or termination
        not in (
            "iteration_budget",
            "wall_time_budget",
            "line_search_failed",
            "projected_gradient_tolerance",
        )
        or (termination == "iteration_budget" and accepted != policy["pilot_steps"])
        or (termination == "wall_time_budget" and accepted == 0)
    ):
        raise ValueError("invalid primary completion or final prediction contract")
    checkpoints = sorted({0, *(i for i in policy["checkpoint_steps"] if i <= accepted)})
    required = {
        "solver-result.json",
        "fields.npy",
        "predictions.npy",
        "gradient.npy",
        "stationarity-gate.json",
        "evaluation-trace.json",
        *(f"history/accepted-{i:04d}.json" for i in range(1, accepted + 1)),
        *(f"checkpoints/fields-{i:04d}.npy" for i in checkpoints),
        *(f"checkpoints/state-{i:04d}.json" for i in checkpoints),
    }
    if missing := required.difference(record["output_sha256"]):
        raise ValueError(f"completed primary fit is missing required outputs: {sorted(missing)}")
    for name in ("initial_objective", "final_objective"):
        value = result.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid primary result: {name}")
    raw_history = result.get("callback_history")
    if not isinstance(raw_history, list):
        raise ValueError("primary accepted history is incomplete")
    history = cast(list[Any], raw_history)
    if len(history) != accepted:
        raise ValueError("primary accepted history is incomplete")
    previous = result["initial_objective"]
    for i, raw_row in enumerate(history, 1):
        if not isinstance(raw_row, dict):
            raise ValueError("primary accepted history contains a non-object state")
        row = cast(dict[str, Any], raw_row)
        if (
            row.get("iteration") != i
            or row.get("accepted") is not True
            or not all(math.isfinite(row[k]) for k in ("objective_before", "objective"))
            or not math.isclose(row["objective_before"], previous, rel_tol=1e-10, abs_tol=1e-7)
            or not 0 <= row["objective"] < row["objective_before"]
            or checked_json(folder, f"history/accepted-{i:04d}.json", record) != row
        ):
            raise ValueError("primary accepted history differs from recorded states")
        previous = row["objective"]
    if not math.isclose(result["final_objective"], previous, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError("primary final objective differs from its accepted history")
    gate = checked_json(folder, "stationarity-gate.json", record)
    expected_gate = stationarity_gate(
        gate["initial_mapping"],
        policy["spectral_mapping_step"],
        policy["relative_mapping_tolerance"],
        policy["maximum_mapped_fraction_displacement"],
    )
    raw_diagnostic = result.get("final_stationarity")
    if not isinstance(raw_diagnostic, dict):
        raise ValueError("primary stationarity is incomplete")
    diagnostic = cast(dict[str, Any], raw_diagnostic)
    if (
        gate != expected_gate
        or diagnostic != stationarity_result(diagnostic["final_mapping"], expected_gate)
        or (termination == "projected_gradient_tolerance" and not diagnostic["passed"])
    ):
        raise ValueError("primary stationarity or termination differs from the frozen rule")
    trace = checked_json(folder, "evaluation-trace.json", record)
    if not isinstance(trace, list) or not trace:
        raise ValueError("primary evaluation trace is missing")
    field_shape = (2, *public["inverse_grid"]["shape"])
    prediction_shape = (len(public["views"]), 3, *policy["detector_shape_hw"])
    array_shapes = {
        "fields.npy": field_shape,
        "gradient.npy": field_shape,
        "predictions.npy": prediction_shape,
        **{f"checkpoints/fields-{i:04d}.npy": field_shape for i in checkpoints},
    }
    # All fitted outputs are checked before the caller may open any reference.
    for name, shape in array_shapes.items():
        array = np.load(folder / name, mmap_mode="r", allow_pickle=False)
        if (
            array.shape != shape
            or array.dtype != np.dtype("<f4")
            or not array.flags.c_contiguous
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"invalid completed primary array: {name}")
        if name != "gradient.npy" and (array < 0).any():
            raise ValueError(f"negative completed primary array: {name}")
        if (
            name not in ("gradient.npy", "predictions.npy")
            and (array.sum(axis=0, dtype=np.float64) > 1 + 2**-24).any()
        ):
            raise ValueError(f"completed primary fractions violate the simplex: {name}")
    for i in checkpoints:
        state = checked_json(folder, f"checkpoints/state-{i:04d}.json", record)
        expected_state: dict[str, Any] = (
            {
                "iteration": 0,
                "objective": result["initial_objective"],
                "mapping": gate["initial_mapping"],
            }
            if i == 0
            else {**history[i - 1], "accepted_trial_bytes_equal": True}
        )
        if state != expected_state:
            raise ValueError("primary checkpoint metadata differs from its accepted state")
    if accepted in checkpoints:
        final_fields = np.load(folder / "fields.npy", mmap_mode="r", allow_pickle=False)
        checkpoint_fields = np.load(
            folder / f"checkpoints/fields-{accepted:04d}.npy", mmap_mode="r", allow_pickle=False
        )
        if not np.array_equal(final_fields.view(np.uint32), checkpoint_fields.view(np.uint32)):
            raise ValueError("primary final fields differ from their accepted checkpoint")
    return record


def expected_noise_namespace(config: dict[str, Any]) -> dict[str, Any]:
    groups = {role: noise_keys(config, role) for role in ("fitting", "withheld")}
    all_keys = list(itertools.chain.from_iterable(groups.values()))
    if len(all_keys) != 504 or len({tuple(key) for key in all_keys}) != 504:
        raise ValueError("illustration noise roles are not disjoint")
    return groups
