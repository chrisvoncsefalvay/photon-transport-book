"""Host-only record-contract fixtures; these are not scientific measurements."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "experiments/transport-recovery/verify_walkthrough.py"
)
SPEC = importlib.util.spec_from_file_location("verify_stochastic_walkthrough", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def seal(directory: Path, configuration: dict[str, Any], payload: str) -> None:
    """Bind test fixture bytes using the actual RunRecorder JSON encoding."""
    write(directory / "configuration.json", configuration)
    write(
        directory / "run.json",
        {
            "schema_version": 2,
            "status": "complete",
            "sources_unchanged": True,
            "recorded_files_unchanged": True,
            "changed_sources": [],
            "changed_recorded_files": [],
            "configuration": configuration,
            "configuration_sha256": VERIFY.digest(directory / "configuration.json"),
            "source_sha256": {
                p.relative_to(directory / "sources").as_posix(): VERIFY.digest(p)
                for p in (directory / "sources").rglob("*")
                if p.is_file()
            },
            "output_sha256": {
                name: VERIFY.digest(directory / name) for name in ("configuration.json", payload)
            },
        },
    )


@pytest.fixture
def records(tmp_path: Path) -> tuple[Path, Path]:
    """Small synthetic receipts test admission logic, never transport accuracy."""
    sampling = VERIFY.read(SCRIPT.with_name("worked-example.json"))["configuration"]
    policy = sampling["policy"]
    reference = {
        "seed": 419003,
        "batch": 1_048_576,
        "sampling_multiplier": 16,
        "replicates": [0.5] * 256,
    }
    reference_path = tmp_path / "reference/reference.json"
    write(reference_path, reference)
    source = reference_path.parent / "sources/fixture.txt"
    source.parent.mkdir()
    source.write_text("Record-contract fixture only.\n")
    seal(reference_path.parent, {"seed": 419003, **sampling}, "reference.json")
    root = tmp_path / "recovery"
    for index in range(32):
        directory = root / f"replicate-{index:02d}"
        (directory / "sources").mkdir(parents=True)
        (directory / "sources/fixture.txt").write_bytes(source.read_bytes())
        (directory / "sources/evaluation-reference.json").write_bytes(reference_path.read_bytes())
        (directory / "sources/evaluation-reference-run.json").write_bytes(
            (reference_path.parent / "run.json").read_bytes()
        )
        seed = 2026091501 + 104729 * index
        configuration = {
            **sampling,
            "seed": seed,
            "evaluation_index": index,
            "repetitions": 32,
            "reference": reference,
        }
        write(
            directory / "recovery.json",
            {
                "seed": seed,
                "sampling_multiplier": 16,
                "policy": policy,
                "deterministic_sampling": False,
                "reference_stationarity_pass": True,
                "reference_gradient_interval": [-1e-6, 1e-6],
                "result": {
                    "reason": "gradient_band",
                    "termination_detail": "held_out_gradient_band",
                    "parameters": [-2.0],
                    "final_validation_attempts": [
                        {
                            "parameters": [-2.0],
                            "pool": "final_validation",
                            "decision": "gradient_band",
                            "batch_size": 1_048_576,
                            "histories_used": 16_777_216,
                            "gradient_norm": 1e-6,
                            "standard_error_norm": 1e-6,
                            "numerical_allowance": 0.0,
                        }
                    ],
                },
            },
        )
        seal(directory, configuration, "recovery.json")
    return root, reference_path


def test_complete_prescribed_receipts_pass_without_optional_dependencies(
    records: tuple[Path, Path],
) -> None:
    root, reference = records
    assert (
        VERIFY.verify_walkthrough(root, reference)["controller_and_reference_checks_passed"] == 32
    )
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--input", str(root), "--reference", str(reference)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["passed"] is True


@pytest.mark.parametrize(
    "change",
    [
        "incomplete",
        "source_bytes",
        "output_bytes",
        "missing_payload",
        "missing_replicate",
        "unbound_payload",
    ],
)
def test_changed_or_incomplete_records_fail(records: tuple[Path, Path], change: str) -> None:
    root, reference = records
    directory = root / "replicate-00"
    manifest = VERIFY.read(directory / "run.json")
    if change == "incomplete":
        manifest["status"] = "failed"
    elif change == "source_bytes":
        (directory / "sources/fixture.txt").write_text("Changed source\n")
    elif change == "output_bytes":
        with (directory / "recovery.json").open("a") as stream:
            stream.write("\n")
    elif change == "missing_payload":
        (directory / "recovery.json").unlink()
    elif change == "missing_replicate":
        directory.rename(root / "omitted-first-run")
    else:
        del manifest["output_sha256"]["recovery.json"]
    if change in ("incomplete", "unbound_payload"):
        write(directory / "run.json", manifest)
    with pytest.raises(ValueError):
        VERIFY.verify_walkthrough(root, reference)


@pytest.mark.parametrize(
    "change",
    [
        "unresolved",
        "reference_failed",
        "reference_interval",
        "final_band",
        "no_final",
        "wrong_final_point",
        "wrong_seed",
        "wrong_tolerance",
        "reference_binding",
    ],
)
def test_hash_valid_but_unaccepted_or_wrong_protocol_records_fail(
    records: tuple[Path, Path], change: str
) -> None:
    root, reference = records
    directory = root / "replicate-00"
    manifest = VERIFY.read(directory / "run.json")
    recovery = VERIFY.read(directory / "recovery.json")
    if change == "unresolved":
        recovery["result"]["reason"] = "sampling_unresolved"
    elif change == "reference_failed":
        recovery["reference_stationarity_pass"] = False
    elif change == "reference_interval":
        recovery["reference_gradient_interval"][1] = 1.01e-5
    elif change == "final_band":
        recovery["result"]["final_validation_attempts"][0]["standard_error_norm"] = 1e-5
    elif change == "no_final":
        recovery["result"]["final_validation_attempts"] = []
    elif change == "wrong_final_point":
        recovery["result"]["parameters"] = [-2.1]
    elif change == "wrong_seed":
        recovery["seed"] += 1
        manifest["configuration"]["seed"] += 1
    elif change == "wrong_tolerance":
        recovery["policy"]["gradient_tolerance"] = 1e-3
        manifest["configuration"]["policy"]["gradient_tolerance"] = 1e-3
    else:
        (directory / "sources/evaluation-reference.json").write_text("Different reference\n")
    write(directory / "recovery.json", recovery)
    seal(directory, manifest["configuration"], "recovery.json")
    with pytest.raises(ValueError):
        VERIFY.verify_walkthrough(root, reference)


def test_cli_exits_nonzero_for_unresolved_fit(records: tuple[Path, Path]) -> None:
    root, reference = records
    directory = root / "replicate-00"
    manifest = VERIFY.read(directory / "run.json")
    recovery = VERIFY.read(directory / "recovery.json")
    recovery["result"]["reason"] = "sampling_unresolved"
    write(directory / "recovery.json", recovery)
    seal(directory, manifest["configuration"], "recovery.json")
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--input", str(root), "--reference", str(reference)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "did not certify" in result.stderr


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("policy", "initial_batch", 4096),
        ("policy", "maximum_batch", 65536),
        ("policy", "unique_history_budget", 16_000_000),
        ("policy", "proposal", "gradient"),
        ("physical_spec", "energy_kev", 60.0),
        ("source", "weight", 2.0),
    ],
)
def test_consistently_changed_physics_or_controller_is_not_the_fixed_walkthrough(
    records: tuple[Path, Path], section: str, key: str, value: object
) -> None:
    root, reference = records
    reference_manifest = VERIFY.read(reference.parent / "run.json")
    reference_manifest["configuration"][section][key] = value
    seal(reference.parent, reference_manifest["configuration"], "reference.json")
    for directory in root.glob("replicate-*"):
        manifest = VERIFY.read(directory / "run.json")
        manifest["configuration"][section][key] = value
        if section == "policy":
            recovery = VERIFY.read(directory / "recovery.json")
            recovery["policy"][key] = value
            write(directory / "recovery.json", recovery)
        (directory / "sources/evaluation-reference-run.json").write_bytes(
            (reference.parent / "run.json").read_bytes()
        )
        seal(directory, manifest["configuration"], "recovery.json")
    with pytest.raises(ValueError, match="complete physical and sampling configuration"):
        VERIFY.verify_walkthrough(root, reference)
