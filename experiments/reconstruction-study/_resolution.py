"""Host contracts for the separately recorded actual-resolution development stage."""

from __future__ import annotations

import importlib
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from _study import complete_record, digest

from dpt.geometry import Matrix3, RigidTransform

np: Any = importlib.import_module("numpy")


def child(folder: Path, name: str) -> Path:
    path = (folder / name).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError(f"missing or unbounded recorded path: {name}")
    return path


def checked_file(folder: Path, name: str, hashes: dict[str, str]) -> Path:
    path = child(folder, name)
    if digest(path) != hashes[name]:
        raise ValueError(f"recorded input changed: {name}")
    return path


def checked_json(folder: Path, name: str, record: dict[str, Any]) -> Any:
    return json.loads(checked_file(folder, name, record["output_sha256"]).read_text())


def current_sources(root: Path, freeze: dict[str, Any]) -> dict[str, Path]:
    expected = freeze["files"]
    canonical = {str(p.relative_to(root)) for p in (root / "python/dpt").rglob("*.py")}
    if not canonical.issubset(expected):
        raise ValueError("source freeze omits canonical Python files")
    return {name: checked_file(root, name, expected) for name in expected}


def check_source_amendment(
    observed: dict[str, str], current: dict[str, str], allowed: dict[str, Any]
) -> dict[str, Any]:
    """Allow only exact reviewed old/new pairs; never waive arbitrary drift."""
    names = {name for name in observed if name.startswith("python/dpt/")}
    if names != {name for name in current if name.startswith("python/dpt/")}:
        raise ValueError("observation/current canonical inventories differ")
    changed: dict[str, Any] = {}
    for name in sorted(names):
        if observed[name] == current[name]:
            continue
        pair = {"observation": observed[name], "current": current[name]}
        if allowed.get(name) != pair:
            raise ValueError(f"unreviewed observation-to-execution source change: {name}")
        changed[name] = pair
    return changed


def required_outcomes() -> dict[str, tuple[str, str, int]]:
    rows = {
        f"case2_dense48_{method}": ("dense48", method, 144)
        for method in ("scalar_poisson", "scalar_pwls", "spectral_metric", "spectral_euclidean")
    }
    rows["case2_dense64_spectral_metric"] = ("dense64", "spectral_metric", 144)
    for regime in ("sparse", "limited"):
        rows[f"case2_{regime}48_spectral_metric"] = (f"{regime}48", "spectral_metric", 36)
    return rows


def validate_schedule(config: dict[str, Any]) -> None:
    expected = required_outcomes()
    rows = config["outcomes"]
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("prescribed seven-outcome schedule is incomplete or duplicated")
    for row in rows:
        group, method, views = expected[row["id"]]
        shape = [64] * 3 if group == "dense64" else [48] * 3
        if (
            row["method"] != method
            or row["views"] != views
            or row["shape_zyx"] != shape
            or row["regime"] != group.removesuffix("48").removesuffix("64")
            or row["maximum_accepted_updates"] != 20
            or row["soft_solve_seconds"] != 120.0
        ):
            raise ValueError("a prescribed outcome changed its method/grid/budget")
    if config["total_soft_fit_seconds"] != 840.0 or config["stage_soft_seconds"] != 1800.0:
        raise ValueError("development stage reservation changed")
    if config["new_noise_phase_id"] != 2 or config["new_noise_regime_ids"] != {
        "sparse": 1,
        "limited": 2,
    }:
        raise ValueError("the frozen development noise roles changed")
    fixed = {
        "pilot_steps": 20,
        "pilot_soft_seconds_per_fit": 120.0,
        "scalar_initial_trial_step_mm_inverse_squared": 3.125e-10,
        "scalar_initial_and_mapping_step_mm_inverse_squared": 1e-8,
        "spectral_mapping_step": 1e-6,
        "spectral_initial_step": 1e-6,
        "spectral_maximum_step": 1.0,
        "fraction_regularisation_mm_inverse": 0.1,
        "relative_mapping_tolerance": 1e-4,
        "maximum_mapped_fraction_displacement": 1e-5,
        "maximum_mapped_scalar_displacement_water_ratio": 1e-5,
        "samples_native": [1024, 2048],
        "samples_inverse": [512, 1024],
    }
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("a frozen solver or sampling setting changed")


def view_records(regime: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    def pose_at(tilt: float, yaw: float) -> RigidTransform:
        # Identical Rz@Rx host convention as the existing acquisition helper.
        a, b = math.radians(tilt), math.radians(yaw)
        rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        rz = np.array([[math.cos(b), -math.sin(b), 0], [math.sin(b), math.cos(b), 0], [0, 0, 1]])
        return RigidTransform(rotation=cast(Matrix3, tuple(float(v) for v in (rz @ rx).ravel())))

    if regime == "dense":
        angles = [7.5 * k for k in range(48)]
    elif regime == "sparse":
        angles = [30.0 * k for k in range(12)]
    elif regime == "limited":
        angles = [-60.0 + 120.0 * k / 11 for k in range(12)]
    else:
        raise ValueError("unknown development acquisition regime")
    return [
        {
            "id": i * len(angles) + j,
            "tilt_degrees": tilt,
            "yaw_degrees": angle,
            "pose": {key: list(value) for key, value in asdict(pose_at(tilt, angle)).items()},
        }
        for i, tilt in enumerate(config["tilts_degrees"])
        for j, angle in enumerate(angles)
    ]


def check_coverage(grid: Any, geometry: Any, views: list[dict[str, Any]]) -> dict[str, Any]:
    corners = [
        grid.grid_to_object(tuple(p)) for p in itertools.product(*zip(*grid.support, strict=True))
    ]
    source = np.asarray(geometry.source_mm, dtype=np.float64)
    origin = np.asarray(geometry.origin_mm, dtype=np.float64)
    u, v = np.asarray(geometry.u), np.asarray(geometry.v)
    normal = np.cross(u, v)
    plane = float(np.dot(origin - source, normal))
    bounds = np.asarray([geometry.shape[1], geometry.shape[0]], dtype=float) - 0.5
    minimum = np.full(2, np.inf)
    for row in views:
        pose = RigidTransform(**{k: tuple(x) for k, x in row["pose"].items()})
        for corner in corners:
            ray = np.asarray(pose.point(corner)) - source
            factor = plane / float(np.dot(ray, normal))
            if factor <= 1:
                raise ValueError("support lies beyond the detector or source plane")
            point = source + factor * ray - origin
            pixel = np.asarray([np.dot(point, u), np.dot(point, v)]) / geometry.spacing_mm
            minimum = np.minimum(minimum, np.minimum(pixel + 0.5, bounds - pixel))
    if np.any(minimum <= 0):
        raise ValueError("prescribed support is detector-truncated")
    return {"views": len(views), "minimum_cell_clearance_uv": minimum.tolist()}


def scaled_physics(physics: dict[str, Any], scale: float) -> dict[str, Any]:
    if scale != 4.0:
        raise ValueError("only the prescribed fourfold development exposure is supported")
    result = {name: value.copy() for name, value in physics.items()}
    for name in ("incident_weights", "weights"):
        result[name] = np.ascontiguousarray(result[name] * np.float32(scale))
        if not np.array_equal(
            result[name].astype(np.float64), physics[name].astype(np.float64) * scale
        ):
            raise ValueError("fourfold integrated weights are not represented exactly")
    if not np.allclose(result["weights"], result["incident_weights"][None], rtol=0, atol=0):
        raise ValueError("channel weights must replicate the single integrated incident spectrum")
    if not np.isclose(result["incident_weights"].astype(np.float64).sum(), 800000, rtol=2e-7):
        raise ValueError("scaled incident photon budget differs from 800000")
    return result


def regime_counts(means: Any, regime: str, config: dict[str, Any]) -> tuple[Any, list[list[int]]]:
    if means.shape[0:2] != (36, 3) or not np.isfinite(means).all() or (means < 0).any():
        raise ValueError("expected finite nonnegative 36-view three-channel means")
    result = np.empty_like(means, dtype=np.float32)
    keys: list[list[int]] = []
    for view in range(36):
        for channel in range(3):
            key = [
                config["noise_root_seed"],
                2,
                2,
                config["new_noise_regime_ids"][regime],
                0,
                2,
                view,
                channel,
            ]
            samples = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key))).poisson(
                means[view, channel].astype(np.float64)
            )
            if (samples > 2**24).any():
                raise ValueError("sampled count exceeds exact FP32 integer range")
            result[view, channel] = samples
            keys.append(key)
    return result, keys


def load_fitting_inputs(folder: Path, root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    record = complete_record(folder)
    public = checked_json(folder, "public-observations.json", record)
    if public["reference_access_permitted_for_fitting"]:
        raise ValueError("fitting manifest permits forbidden reference access")
    study_names = {
        f"reconstruction-study/{p.name}"
        for p in (root / "experiments/reconstruction-study").glob("*.py")
    }
    if study_names != {
        name for name in record["source_sha256"] if name.startswith("reconstruction-study/")
    }:
        raise ValueError("experiment helper inventory changed after preparation")
    for name, want in record["source_sha256"].items():
        if name.startswith("python/dpt/"):
            actual = child(root, name)
        elif name.startswith("reconstruction-study/"):
            actual = child(root / "experiments/reconstruction-study", name.split("/", 1)[1])
        elif name == "helpers/spectral-run.py":
            actual = root / "experiments/spectral-reconstruction/run.py"
        else:
            continue
        if digest(actual) != want:
            raise ValueError(f"source changed since prepared observations: {name}")
    files = {
        "run.json": folder / "run.json",
        "public-observations.json": folder / "public-observations.json",
    }
    for view in public["views"]:
        for model, want in view["sha256"].items():
            name = view[f"{model}_counts"]
            path = checked_file(folder, name, record["output_sha256"])
            values = np.load(path, allow_pickle=False)
            expected = tuple(public["geometry"]["shape"])
            if model == "spectral":
                expected = (3, *expected)
            if (
                digest(path) != want
                or values.dtype != np.float32
                or values.shape != expected
                or not np.isfinite(values).all()
                or (values < 0).any()
                or (values % 1 != 0).any()
                or (values > 2**24).any()
            ):
                raise ValueError("fitting counts violate their fixed input contract")
            files[name] = path
    for name, want in {
        public["open_beam"]: public["open_beam_sha256"],
        **{f"physics/{k}": v for k, v in public["physics"].items()},
    }.items():
        files[name] = checked_file(folder, name, record["output_sha256"])
        if digest(files[name]) != want:
            raise ValueError("physical input manifest differs from recorded bytes")
    return public, {f"acquisition/{name}": path for name, path in files.items()}


def qualification_coverage(reports: list[dict[str, Any]]) -> int:
    expected_families = {
        "dense48": (144, (512, 1024)),
        "dense64": (144, (512, 1024)),
        "sparse-native": (36, (1024, 2048)),
        "sparse48": (36, (512, 1024)),
        "limited-native": (36, (1024, 2048)),
        "limited48": (36, (512, 1024)),
    }
    found: set[str] = set()
    count = 0
    for report in reports:
        name = report["comparison_arrays"].removeprefix("qualification/").removesuffix("/")
        if name not in expected_families or name in found:
            raise ValueError("unknown or duplicate qualification family")
        found.add(name)
        views, pair = expected_families[name]
        expected = {
            (role, view, channel, sample)
            for view in range(views)
            for role, channels, samples in (
                ("independent_scalar", range(1), pair),
                ("independent_spectral", range(3), pair),
                ("doubled_scalar", range(1), pair[:1]),
                ("doubled_spectral", range(3), pair[:1]),
            )
            for channel in channels
            for sample in samples
        }
        gates = report["gates"]
        if (
            len(gates) != len(expected)
            or report["gate_count"] != len(expected)
            or {(g["role"], g["view"], g["channel"], g["samples"]) for g in gates} != expected
            or not report["passed"]
            or not all(
                g["passed"]
                and 0 <= g["p95_poisson_sd"] < 0.05
                and g["p95_poisson_sd"] <= g["maximum_poisson_sd"] < 0.2
                for g in gates
            )
        ):
            raise ValueError("qualification Cartesian coverage or numerical gate failed")
        count += len(gates)
    if found != set(expected_families) or count != 5184:
        raise ValueError("all six qualification families must precede new count draws")
    return count


def verify_prepared(prepared: Path, config: Path, freeze: Path) -> dict[str, Any]:
    receipt = json.loads((prepared / "prepared.json").read_text())
    if (
        receipt["status"] != "complete"
        or not receipt["all_5184_sampling_gates_passed"]
        or receipt["new_noise_keys"] != 216
        or receipt["config_sha256"] != digest(config)
        or receipt["canonical_freeze_sha256"] != digest(freeze)
        or set(receipt["bundles"]) != {"dense48", "dense64", "sparse48", "limited48"}
    ):
        raise ValueError("prepared stage identity or qualification gate changed")
    qualification = complete_record(prepared / "qualification")
    gate = checked_json(prepared / "qualification", "qualification.json", qualification)
    qualification_coverage(
        [
            checked_json(prepared / "qualification", f"qualification/{name}.json", qualification)
            for name in (
                "dense48",
                "dense64",
                "sparse-native",
                "sparse48",
                "limited-native",
                "limited48",
            )
        ]
    )
    if (
        digest(prepared / "qualification/run.json") != receipt["qualification_run_sha256"]
        or not gate["passed"]
        or gate["gate_count"] != 5184
        or gate["counts_drawn"]
    ):
        raise ValueError("qualification must finish before all new count draws")
    for group, hashes in receipt["bundles"].items():
        record = complete_record(prepared / group)
        if (
            digest(prepared / group / "run.json") != hashes["run_sha256"]
            or digest(prepared / group / "public-observations.json") != hashes["manifest_sha256"]
            or record["configuration"]["qualification_run_sha256"]
            != receipt["qualification_run_sha256"]
        ):
            raise ValueError("fitting inputs are not bound to the accepted qualification")
        for name in record["source_sha256"]:
            checked_file(prepared / group / "sources", name, record["source_sha256"])
    return receipt


def require_coverage(rows: list[dict[str, Any]]) -> None:
    expected = required_outcomes()
    if len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
        raise ValueError("recorded outcome set differs from the seven prescribed runs")
    for row in rows:
        group, method, _ = expected[row["id"]]
        if row["group"] != group or row["method"] != method:
            raise ValueError("an outcome changed its prescribed method or observations")
        if row["status"] not in ("recorded", "failed", "stage_budget_not_started"):
            raise ValueError("unrecognised development outcome status")


def count_metrics(prediction: Any, counts: Any, means: Any) -> dict[str, Any]:
    p, k, mean = (np.asarray(a, dtype=np.float64) for a in (prediction, counts, means))
    if (
        p.shape != k.shape
        or k.shape != mean.shape
        or not all(np.isfinite(a).all() for a in (p, k, mean))
    ):
        raise ValueError("invalid matched prediction/count/mean arrays")
    if (p <= 0).any() or (mean <= 0).any() or (k < 0).any():
        raise ValueError("this development metric requires positive model means")
    half = p - k
    active = k > 0
    half[active] -= k[active] * np.log1p((p[active] - k[active]) / k[active])
    error = (p - mean) / np.sqrt(mean)
    return {
        "pixels": p.size,
        "observed_half_deviance": float(np.sum(half, dtype=np.float64)),
        "mean_error_poisson_sd_rms": float(np.sqrt(np.mean(error**2))),
        "mean_error_poisson_sd_p95_absolute": float(np.quantile(np.abs(error), 0.95)),
        "mean_error_poisson_sd_maximum_absolute": float(np.max(np.abs(error))),
        "mean_error_relative_l2": float(np.linalg.norm(p - mean) / np.linalg.norm(mean)),
        "mean_signed_count_error": float(np.mean(p - mean)),
    }
