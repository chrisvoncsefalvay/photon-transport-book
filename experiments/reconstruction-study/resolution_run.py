"""Run seven fixed development outcomes from qualified, count-only input folders."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from _resolution import (
    current_sources,
    load_fitting_inputs,
    require_coverage,
    required_outcomes,
    validate_schedule,
    verify_prepared,
)
from _study import digest, scalar_case
from pilot import run_method, wp

from dpt.experiments import RunRecorder, experiment_sources, private_output, repository_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "config", "freeze", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = repository_root(__file__)
    config = json.loads(args.config.read_text())
    validate_schedule(config)
    freeze = json.loads(args.freeze.read_text())
    current_sources(root, freeze)
    prepared = verify_prepared(args.prepared, args.config, args.freeze)
    if prepared["clock_boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        raise ValueError("soft stage clock cannot be resumed across host boots")
    stage_start = prepared["stage_start_monotonic_seconds"]
    if time.perf_counter() < stage_start:
        raise ValueError("invalid cross-process monotonic stage clock")
    # Resolve and validate every fitting-only input before any run is started.
    inputs = {
        group: load_fitting_inputs(args.prepared / group, root) for group in prepared["bundles"]
    }
    sources = experiment_sources(
        __file__,
        args.config,
        extra={
            "canonical-freeze.json": args.freeze,
            "prepared.json": args.prepared / "prepared.json",
            "qualification/run.json": args.prepared / "qualification/run.json",
            **{f"reconstruction-study/{p.name}": p for p in Path(__file__).parent.glob("*.py")},
            **{f"inputs/{group}/run.json": args.prepared / group / "run.json" for group in inputs},
        },
    )
    output = private_output(args.output, root)
    rows: list[dict[str, Any]] = []
    wp.init()
    with RunRecorder(
        output,
        configuration={
            "config": config,
            "prepared_sha256": digest(args.prepared / "prepared.json"),
            "stage_start_monotonic_seconds": stage_start,
            "known_field_or_generating_mean_access": False,
            "scope": "seven actual-grid development outcomes, not final study",
        },
        sources=sources,
    ) as run:
        for planned in config["outcomes"]:
            name = planned["id"]
            group, method, _ = required_outcomes()[name]
            row: dict[str, Any] = {"id": name, "group": group, "method": method}
            elapsed = time.perf_counter() - stage_start
            if elapsed >= config["stage_soft_seconds"]:
                row.update(status="stage_budget_not_started", elapsed_stage_seconds=elapsed)
            else:
                public, source_inputs = inputs[group]
                if (
                    public["inverse_grid"]["shape"] != planned["shape_zyx"]
                    or len(public["views"]) != planned["views"]
                    or public["config"]["pilot_steps"] != planned["maximum_accepted_updates"]
                    or public["config"]["pilot_soft_seconds_per_fit"]
                    != planned["soft_solve_seconds"]
                ):
                    raise ValueError("prepared fit differs from the reviewed grid/budget")
                tick = time.perf_counter()
                try:
                    # Only scalar methods need a supplied attenuation case.
                    case_path = output / f"unused-{name}"
                    if not method.startswith("spectral_"):
                        case_path = scalar_case(
                            output / f"input-{name}",
                            args.prepared / group,
                            public,
                            public["config"],
                        )
                    result = run_method(
                        output / name,
                        method,
                        args.prepared / group,
                        public,
                        {
                            **source_inputs,
                            "stage-config.json": args.config,
                            "stage-prepared.json": args.prepared / "prepared.json",
                            "stage-freeze.json": args.freeze,
                        },
                        case_path,
                        args.device,
                    )
                    row.update(status="recorded", **result)
                except Exception as error:
                    # Preserve the failed child RunRecorder and a required outcome row.
                    row.update(status="failed", error=repr(error))
                    if (output / name / "run.json").is_file():
                        row["run_sha256"] = digest(output / name / "run.json")
                row["whole_fit_segment_seconds"] = time.perf_counter() - tick
                row["elapsed_stage_seconds"] = time.perf_counter() - stage_start
            rows.append(row)
            run.write_json(f"outcomes/{name}.json", row)
            print(json.dumps(row), flush=True)
        require_coverage(rows)
        current_sources(root, freeze)
        run.write_json(
            "batch.json",
            {
                "status": "complete prescribed development accounting",
                "outcomes": rows,
                "all_outcomes_recorded": all(row["status"] == "recorded" for row in rows),
                "stage_elapsed_seconds": time.perf_counter() - stage_start,
                "stage_soft_seconds": config["stage_soft_seconds"],
                "stage_overshoot_seconds": max(
                    0, time.perf_counter() - stage_start - config["stage_soft_seconds"]
                ),
                "reference_or_generating_mean_access": False,
                "preparation_seconds": prepared["preparation_seconds"],
            },
        )
    print(f"Resolution fit accounting complete: {output}", flush=True)


if __name__ == "__main__":
    main()
