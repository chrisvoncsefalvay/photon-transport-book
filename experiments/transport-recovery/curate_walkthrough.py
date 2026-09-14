"""Curate the prescribed stochastic walkthrough from verified execution records.

This reads recorded estimates and independent assessment; it runs no transport
or optimisation. Full local paths and environment inventories are not exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def verify(directory: Path) -> dict:
    record = read(directory / "run.json")
    if (
        record["status"] != "complete"
        or not record["sources_unchanged"]
        or not record["recorded_files_unchanged"]
    ):
        raise ValueError("curation requires completed unchanged execution records")
    encoded = json.dumps(record["configuration"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    if hashlib.sha256(encoded.encode()).hexdigest() != record["configuration_sha256"]:
        raise ValueError("configuration hash mismatch")
    for field, prefix in (("source_sha256", "sources"), ("output_sha256", "")):
        for name, expected in record[field].items():
            path = directory / prefix / name
            if path.is_symlink() or digest(path) != expected:
                raise ValueError(f"recorded bytes changed: {name}")
    return record


def write(directory: Path, name: str, value: object) -> None:
    (directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plot(output: Path, study: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#24231f",
            "text.color": "#24231f",
            "axes.edgecolor": "#68665e",
            "figure.facecolor": "#faf8f3",
            "axes.facecolor": "#faf8f3",
            "svg.fonttype": "none",
            "svg.hashsalt": "dpt-stochastic-worked-v2",
            "pdf.use14corefonts": True,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), layout="constrained")
    amplitudes = study["designated"]["amplitudes"]
    axes[0].plot(range(len(amplitudes)), amplitudes, "o-", color="#0072B2", linewidth=1.8)
    axes[0].set(
        xlabel="Accepted update (0 = initial)",
        ylabel="Source amplitude",
        title="A  The designated first run",
    )
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylim(0.075, 0.122)
    runs = study["all_runs"]
    indices = [row["index"] for row in runs]
    axes[1].plot(
        indices,
        [row["final_gradient_upper"] for row in runs],
        "o",
        color="#0072B2",
        markersize=4,
        label="Held-out gradient bound",
    )
    axes[1].axhline(
        1e-5, color="#24231f", linestyle="--", linewidth=1.4, label="Unchanged tolerance"
    )
    axes[1].plot(
        0,
        runs[0]["final_gradient_upper"],
        "o",
        color="#D55E00",
        markersize=6,
        label="Predesignated run 00",
    )
    axes[1].set(
        xlabel="Independent run (prescribed order)",
        ylabel="Absolute gradient + 2 SE",
        title="B  All 32 stopping checks",
        ylim=(0, 1.15e-5),
    )
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-5, -5), useMathText=True)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=8, ncols=1)
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
    fig.savefig(
        output / "walkthrough.svg",
        metadata={"Date": None, "Creator": "Matplotlib; canonical recorded stochastic walkthrough"},
    )
    fig.savefig(
        output / "walkthrough.pdf",
        metadata={
            "CreationDate": None,
            "ModDate": None,
            "Creator": "Matplotlib; canonical recorded stochastic walkthrough",
        },
    )
    fig.savefig(output / "walkthrough.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = read(args.freeze)
    assessment = read(args.assessment)
    reference_record = verify(args.reference.parent)
    reference = read(args.reference)
    directories = sorted(args.input.glob("replicate-*"))
    if len(directories) != 32 or frozen["designated_tutorial_replicate"] != 0:
        raise ValueError("retain the complete prescribed set and its designated first run")
    records, manifests, rows = [], [], []
    independent = assessment["stochastic"]["runs"]
    audited = {row["run"]: row["manifest_sha256"] for row in assessment["manifests"]["runs"]}
    for index, directory in enumerate(directories):
        manifest = verify(directory)
        if audited[directory.name] != digest(directory / "run.json"):
            raise ValueError("the independent assessment belongs to a different run record")
        for name, expected in frozen["source_sha256"].items():
            if name in manifest["source_sha256"] and manifest["source_sha256"][name] != expected:
                raise ValueError("executed source differs from the frozen implementation")
        record = read(directory / "recovery.json")
        if record["seed"] != frozen["evaluation_seeds"][index]:
            raise ValueError("the observed seed differs from the prospective sequence")
        if record["policy"] != frozen["configuration"]["policy"]:
            raise ValueError("the policy changed after the protocol was frozen")
        if manifest["source_sha256"]["evaluation-reference.json"] != digest(args.reference):
            raise ValueError("the recovery used a different reference")
        assessed = independent[index]
        if (
            assessed["seed"] != record["seed"]
            or assessed["run"] != directory.name
            or assessed["accounting"]["errors"]
            or assessed["any_false_acceptance"] != "no_false_acceptance"
            or assessed["any_insufficient_decrease"] != "no_false_acceptance"
        ):
            raise ValueError("independent assessment does not support this accepted walkthrough")
        final = record["result"]["final_validation_attempts"][-1]
        bound = final["gradient_norm"] + 2 * final["standard_error_norm"]
        passed = (
            record["result"]["reason"] == "gradient_band"
            and assessed["independent_stationarity"] == "pass"
            and bound <= 1e-5
        )
        if not passed:
            raise ValueError(
                "do not label an unresolved or unverified fit as this successful suite"
            )
        rows.append(
            {
                "index": index,
                "seed": record["seed"],
                "amplitude": record["amplitude"],
                "final_gradient_upper": bound,
                "reference_gradient_interval": assessed["reference_gradient_interval"],
                "controller_passed": True,
                "reference_passed": True,
                "accepted_steps": sum(row["accepted"] for row in record["result"]["steps"]),
                "unique_histories": record["result"]["unique_histories"],
                "traced_histories_including_replay": record["histories_traced_including_replay"],
            }
        )
        records.append(record)
        manifests.append(manifest)
    first = records[0]
    attempts = first["result"]["acceptance_attempts"]
    amplitudes = [
        0.08,
        *[math.exp(row["candidate"][0]) for row in attempts if row["outcome"] == "accepted"],
    ]
    final = first["result"]["final_validation_attempts"][-1]
    study = {
        "schema_version": 1,
        "protocol_id": frozen["protocol_id"],
        "passed": True,
        "scope": (
            "Fixed mathematical isotropic-scattering cube; prescribed expected score, "
            "not acquired data or anatomy"
        ),
        "configuration": frozen["configuration"],
        "designated": {
            "index": 0,
            "seed": first["seed"],
            "amplitudes": amplitudes,
            "final_gradient_estimate": final["gradient"][0],
            "final_gradient_standard_error": final["standard_error"][0],
            "final_gradient_upper": rows[0]["final_gradient_upper"],
            "reference_gradient_interval": rows[0]["reference_gradient_interval"],
            "unique_histories": first["result"]["unique_histories"],
            "traced_histories_including_replay": first["histories_traced_including_replay"],
            "allocated_device_bytes": first["allocated_device_bytes"],
        },
        "all_runs": rows,
        "reference": {key: value for key, value in reference.items() if key != "replicates"},
        "uncertainty_scope": (
            "Two-SE controller and seven-SE reference are empirical heuristic intervals, "
            "not simultaneous or distribution-free guarantees."
        ),
        "amplitude_series": [
            {
                "label": "Accepted source amplitude",
                "x": list(range(len(amplitudes))),
                "y": amplitudes,
                "tone": "technical",
            }
        ],
        "certification_series": [
            {
                "label": "Final gradient upper bound",
                "x": [row["index"] for row in rows],
                "y": [row["final_gradient_upper"] for row in rows],
                "mode": "markers",
                "tone": "technical",
            },
            {
                "label": "Unchanged tolerance",
                "x": [0, 31],
                "y": [1e-5, 1e-5],
                "dash": True,
                "tone": "ink",
            },
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output, "study.json", study)
    (args.output / "replicate-00.json").write_bytes((directories[0] / "recovery.json").read_bytes())
    (args.output / "reference.json").write_bytes(args.reference.read_bytes())
    write(args.output, "all-runs.json", independent)
    plot(args.output, study)
    provenance = {
        "schema_version": 1,
        "protocol_id": frozen["protocol_id"],
        "frozen_utc": frozen["frozen_utc"],
        "freeze_sha256": digest(args.freeze),
        "assessment_sha256": digest(args.assessment),
        "reference_run_sha256": digest(args.reference.parent / "run.json"),
        "run_sha256": {directory.name: digest(directory / "run.json") for directory in directories},
        "source_sha256": manifests[0]["source_sha256"],
        "reference_source_sha256": reference_record["source_sha256"],
        "curator_sha256": digest(Path(__file__)),
        "content_license": "CC-BY-NC-4.0",
        "code_license": "Apache-2.0",
        "third_party_inputs": (
            "None: analytic geometry and coefficients; no medical data, measured spectrum, "
            "model weights or external images."
        ),
        "fonts": (
            "SVG stores text; PDF uses standard PDF core fonts; "
            "no external font file is distributed."
        ),
        "output_sha256": {
            path.name: digest(path)
            for path in sorted(args.output.iterdir())
            if path.name != "provenance.json"
        },
    }
    write(args.output, "provenance.json", provenance)
    print(
        json.dumps(
            {"output": str(args.output), "runs": 32, "designated": study["designated"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
