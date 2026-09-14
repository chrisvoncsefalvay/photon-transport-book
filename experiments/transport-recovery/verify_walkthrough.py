"""Check completion and recorded acceptance of the fixed Chapter 10 walkthrough.

Standard-library only: verify existing receipts and decisions without running
transport, recomputing statistics or changing the controller's acceptance rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    def members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON value in {path.name}: {value}")

    return json.loads(path.read_text(), object_pairs_hook=members, parse_constant=reject_constant)


def recorded_path(directory: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts:
        raise ValueError(f"noncanonical recorded path: {name}")
    if directory.is_symlink():
        raise ValueError("recorded directory is a symlink")
    path = directory
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"recorded path is a symlink: {name}")
    if not path.is_file():
        raise ValueError(f"recorded file is missing: {name}")
    return path


def verify_record(directory: Path, payload: str) -> dict[str, Any]:
    if directory.is_symlink():
        raise ValueError("run directory is a symlink")
    record = read(recorded_path(directory, "run.json"))
    if (
        record["schema_version"] != 2
        or record["status"] != "complete"
        or record["sources_unchanged"] is not True
        or record["recorded_files_unchanged"] is not True
        or record["changed_sources"]
        or record["changed_recorded_files"]
    ):
        raise ValueError(f"{directory.name}: execution is incomplete or changed")
    encoded = json.dumps(record["configuration"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    if hashlib.sha256(encoded.encode()).hexdigest() != record["configuration_sha256"]:
        raise ValueError(f"{directory.name}: configuration hash mismatch")
    if not {"configuration.json", payload}.issubset(record["output_sha256"]):
        raise ValueError(f"{directory.name}: required payload is not bound by its receipt")
    if record["output_sha256"]["configuration.json"] != record["configuration_sha256"]:
        raise ValueError(f"{directory.name}: stored configuration differs from receipt")
    for field, prefix in (("source_sha256", "sources"), ("output_sha256", "")):
        if not record[field]:
            raise ValueError(f"{directory.name}: empty {field}")
        for name, expected in record[field].items():
            if digest(recorded_path(directory / prefix, name)) != expected:
                raise ValueError(f"{directory.name}: recorded bytes changed: {name}")
    return record


def verify_walkthrough(recovery_root: Path, reference_path: Path) -> dict[str, Any]:
    protocol = read(Path(__file__).with_name("worked-example.json"))
    reference_record = verify_record(reference_path.parent, reference_path.name)
    reference = read(recorded_path(reference_path.parent, reference_path.name))
    reference_digest = digest(reference_path)
    reference_run_digest = digest(reference_path.parent / "run.json")
    sampling = dict(reference_record["configuration"])
    if (
        sampling.pop("seed") != protocol["reference_seed"]
        or reference["seed"] != protocol["reference_seed"]
    ):
        raise ValueError("reference seed differs from the prescribed independent seed")
    if json.dumps(sampling, sort_keys=True) != json.dumps(
        protocol["configuration"], sort_keys=True
    ):
        raise ValueError(
            "the complete physical and sampling configuration differs from the protocol"
        )
    policy = sampling["policy"]
    if (
        sampling["sampling_multiplier"] != 16
        or sampling["reference_batch"] != 1_048_576
        or sampling["reference_replicates"] != 256
        or sampling["observation"] != 0.08
        or sampling["initial_amplitude"] != 0.08
        or policy["gradient_tolerance"] != 1e-5
        or policy["replicates"] != 8
        or policy["standard_error_multiplier"] != 2.0
        or policy["final_validation_batch"] != 1_048_576
        or reference["batch"] != 1_048_576
        or reference["sampling_multiplier"] != 16
        or len(reference["replicates"]) != 256
    ):
        raise ValueError("the recorded configuration is not the prescribed walkthrough")
    names = [f"replicate-{index:02d}" for index in range(protocol["repetitions"])]
    if sorted(path.name for path in recovery_root.glob("replicate-*")) != names:
        raise ValueError("the walkthrough requires all 32 prescribed recovery directories")
    for index, name in enumerate(names):
        directory = recovery_root / name
        manifest = verify_record(directory, "recovery.json")
        configuration = dict(manifest["configuration"])
        if (
            configuration.pop("seed") != protocol["seed_base"] + protocol["seed_stride"] * index
            or configuration.pop("evaluation_index") != index
            or configuration.pop("repetitions") != protocol["repetitions"]
            or configuration.pop("reference") != reference
            or configuration != sampling
        ):
            raise ValueError(f"{name}: seed, reference or configuration differs from the protocol")
        sources = manifest["source_sha256"]
        if (
            sources.get("evaluation-reference.json") != reference_digest
            or sources.get("evaluation-reference-run.json") != reference_run_digest
            or any(
                sources.get(key) != value
                for key, value in reference_record["source_sha256"].items()
            )
        ):
            raise ValueError(f"{name}: reference or executed source binding differs")
        recovery = read(directory / "recovery.json")
        if (
            recovery["seed"] != manifest["configuration"]["seed"]
            or recovery["sampling_multiplier"] != 16
            or recovery["policy"] != policy
            or recovery["deterministic_sampling"] is not False
        ):
            raise ValueError(f"{name}: recovery payload differs from its configuration")
        result = recovery["result"]
        attempts = result["final_validation_attempts"]
        if not attempts:
            raise ValueError(f"{name}: no reserved final-validation decision")
        final = attempts[-1]
        if (
            result["reason"] != "gradient_band"
            or result["termination_detail"] != "held_out_gradient_band"
            or final["decision"] != "gradient_band"
            or final["pool"] != "final_validation"
            or final["parameters"] != result["parameters"]
            or final["batch_size"] != policy["final_validation_batch"]
            or final["histories_used"] != 2 * policy["replicates"] * final["batch_size"]
        ):
            raise ValueError(f"{name}: controller did not certify its returned point")
        values = [
            final[key] for key in ("gradient_norm", "standard_error_norm", "numerical_allowance")
        ]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"{name}: invalid recorded final gradient band")
        upper = values[0] + policy["standard_error_multiplier"] * values[1] + values[2]
        interval = recovery["reference_gradient_interval"]
        if (
            upper > policy["gradient_tolerance"]
            or recovery["reference_stationarity_pass"] is not True
            or len(interval) != 2
            or interval[0] > interval[1]
            or any(
                not math.isfinite(value) or abs(value) > policy["gradient_tolerance"]
                for value in interval
            )
        ):
            raise ValueError(f"{name}: controller or independent reference check did not pass")
    return {
        "passed": True,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": digest(Path(__file__).with_name("worked-example.json")),
        "runs": 32,
        "designated_replicate": "replicate-00",
        "controller_and_reference_checks_passed": 32,
        "recorded_source_and_output_hashes": "verified",
        "uncertainty_scope": "recorded heuristic rules; no new statistical certification",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_walkthrough(args.input, args.reference), indent=2))
    except (KeyError, IndexError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"Walkthrough verification failed: {error}\n")


if __name__ == "__main__":
    main()
