"""Evaluate fixed completed poses against separately supplied geometric references."""

from __future__ import annotations

import argparse
from pathlib import Path

from _study import digest, read_json, rigid, write_json

from dpt.validation.recovery import pose_error


def geometric_errors(run: Path, reference_path: Path) -> dict:
    record = read_json(run / "run.json")
    if record["status"] != "complete":
        raise ValueError("refusing to evaluate an incomplete pose run")
    output = run / "optimisation.json"
    if digest(output) != record["output_sha256"]["optimisation.json"]:
        raise ValueError("accepted-pose output changed")
    result = read_json(output)
    generator = read_json(reference_path.parent / "run.json")
    if (
        generator["status"] != "complete"
        or digest(reference_path) != generator["output_sha256"][reference_path.name]
    ):
        raise ValueError("reference generator is incomplete or its recorded bytes changed")
    reference = read_json(reference_path)
    recovered = rigid(result["pose_object_to_world"])
    target = rigid(reference["pose_object_to_world"])
    from dataclasses import asdict

    probes = tuple(tuple(point) for point in reference["geometric_probes_object_mm"])
    errors = pose_error(recovered, target, probes)
    initial_errors = pose_error(rigid(result["chart"]["anchor"]), target, probes)
    return {
        "run_sha256": digest(run / "run.json"),
        "pose_output_sha256": digest(output),
        "reference_sha256": digest(reference_path),
        "errors": asdict(errors),
        "initial_errors": asdict(initial_errors),
        "stationary": result["stationary"],
        "termination": result["result"]["reason"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, geometric_errors(args.run, args.reference))


if __name__ == "__main__":
    main()
