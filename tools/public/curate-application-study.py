"""Curate the reviewed application study without fitting or new forward computation.

Only native PNG displays and explicit numerical metadata enter a new destination.
The private receipt binds source locations; public metadata contains hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_SHA = "930c66f2842a2853b53c75448da1c67c6bf532cce8d28918f5f2e6c0b87dfbef"
DISPLAY_SHA = "04b82f140e184100eeb431930dedd076fce4e60d09c2f42cd45cc7683ee70009"
REVIEW_SHA = "3e88cb1e58f27a81fd11fecc29b3e7cc74345a51b8798ee418a73ebd658dbdec"
RIGHTS = "CC BY 3.0; derived from Rister et al., CT-ORG, TCIA"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def serialise(value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if re.search(rb"/(?:home|fshare|tmp|Users|root)/|file://|tailscale|100\.105\.", data):
        raise ValueError("Private location in public metadata")
    return data


def inside(root, name):
    relative = Path(name)
    target = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not target.resolve().is_relative_to(root)
    ):
        raise ValueError("Source path escapes its declared root")
    if any(p.is_symlink() for p in (target, *target.parents) if p.is_relative_to(root)):
        raise ValueError("Symlink in source path")
    return target


class Intake:
    def __init__(self):
        self.inputs = {}
        self.payloads = {}
        self.origins = {}
        self.verified_files = 0

    def read(self, tag, path, expected=None):
        data = path.read_bytes()
        actual = sha(data)
        if expected is not None and actual != expected:
            raise ValueError(f"Changed input: {tag}")
        self.inputs[tag] = {"path": str(path.resolve()), "sha256": actual}
        return data

    def record(self, tag, path, expected=None):
        return json.loads(self.read(tag, path, expected))

    def run(self, tag, root, expected=None):
        run = self.record(tag + "/run", root / "run.json", expected)
        if (
            run["status"] != "complete"
            or not run["sources_unchanged"]
            or not run["recorded_files_unchanged"]
        ):
            raise ValueError("Producer did not complete with unchanged sources and outputs")
        for name, value in {
            **run["output_sha256"],
            **{"sources/" + n: h for n, h in run["source_sha256"].items()},
        }.items():
            if sha(inside(root, name).read_bytes()) != value:
                raise ValueError(f"Changed archived producer file: {tag}/{name}")
            self.verified_files += 1
        return run

    def array(self, tag, path, expected):
        value = np.load(io.BytesIO(self.read(tag, path, expected)), allow_pickle=False)
        if value.dtype != np.float32 or value.shape != (128, 128) or not np.isfinite(value).all():
            raise ValueError("Expected native finite FP32 detector array")
        return value.astype(np.float64)

    def add(self, name, data, origin):
        if name in self.payloads or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Duplicate or unsafe output path")
        self.payloads[name] = data
        self.origins[name] = origin
        return {"file": name, "sha256": sha(data), "bytes": len(data)}

    def image(self, name, values, beam, residual=False, valid_mask=None):
        valid = np.ones(values.shape, dtype=bool) if valid_mask is None else valid_mask.astype(bool)
        shown = values if residual else -np.log((values + 0.5) / beam)
        low, high = (-500.0, 500.0) if residual else (0.0, 8.0)
        if not np.isfinite(shown).all():
            raise ValueError("Nonfinite display input")
        rgba = matplotlib.colormaps["RdBu_r" if residual else "gray"](
            np.clip((shown - low) / (high - low), 0, 1), bytes=True
        )
        rgba[~valid, :3] = 0
        rgba[~valid, 3] = 0
        stream = io.BytesIO()
        Image.fromarray(np.flipud(rgba)).save(stream, format="PNG", optimize=False)
        return {
            **self.add(
                name + ".png",
                stream.getvalue(),
                "Native recorded detector image; fixed display mapping",
            ),
            "window": [low, high],
            "display": "observed minus final prediction, counts"
            if residual
            else "-log((count + 0.5) / open_beam)",
            "open_beam_counts_per_pixel": beam,
            "below_window_pixels": int(np.count_nonzero((shown < low) & valid)),
            "above_window_pixels": int(np.count_nonzero((shown > high) & valid)),
            "stored_value_range": [float(values[valid].min()), float(values[valid].max())],
            "valid_pixels": int(valid.sum()),
            "invalid_pixels": int((~valid).sum()),
            "invalid_display": "transparent RGBA(0,0,0,0); not measured zero counts",
            "shape_hw": [128, 128],
        }


def coverage(protocol, rows):
    expected = set()
    for noise in range(protocol["registration"]["noise_replicates_per_case_and_start"]):
        for start in range(len(protocol["registration"]["initial_offsets_local_se3_mm_rad"])):
            for comparison in protocol["registration"]["view_sets_degrees"]:
                expected.add((f"reg-{comparison}-n{noise:02d}-s{start}", "reg", noise))
    for noise in range(protocol["registration"]["stress"]["noise_replicates_per_case"]):
        for start in range(len(protocol["registration"]["initial_offsets_local_se3_mm_rad"])):
            expected.add((f"stress-n{noise:02d}-s{start}", "stress", noise))
    for role in ("broad", "constrained"):
        settings = protocol["acquisition"] if role == "broad" else protocol["acquisition"][role]
        for noise in range(settings["replicates_per_case"]):
            for policy in ("base", "selected", "fixed", "random"):
                expected.add((f"{role}-n{noise:02d}-{policy}", role, noise))
    actual = {(r["name"], r["role"], r["replicate"]) for r in rows}
    if actual != expected or len(actual) != len(rows) or len(rows) != 68:
        raise ValueError("Missing, duplicate or incorrectly assigned prespecified outcome")


def selected_fields(row, names):
    return {key: row[key] for key in names if key in row}


def series(label, rows, x, y, mode="markers"):
    return {"label": label, "x": [x(r) for r in rows], "y": [y(r) for r in rows], "mode": mode}


def geometric_trajectory(bundle, name, child, run, optimisation, reference, evaluated):
    """Evaluate fixed mathematical probes at saved poses, without a forward model."""
    trajectory = bundle.record(
        name + "/accepted-trajectory",
        child / "trajectory.json",
        run["output_sha256"]["trajectory.json"],
    )
    probes = np.asarray(reference["geometric_probes_object_mm"], dtype=np.float64)
    expected = np.asarray(
        [
            [x, y, z]
            for x in (-60, -30, 0, 30, 60)
            for y in (-60, -30, 0, 30, 60)
            for z in (-60, -30, 0, 30, 60)
        ],
        dtype=np.float64,
    )
    if not np.array_equal(probes, expected):
        raise ValueError("Geometric probes differ from the fixed 125-point lattice")

    def positions(pose):
        rotation = np.asarray(pose["rotation"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(pose["translation_mm"], dtype=np.float64)
        if (
            translation.shape != (3,)
            or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-10)
            or np.linalg.det(rotation) <= 0
        ):
            raise ValueError("Recorded pose is not a finite proper rigid transform")
        return probes @ rotation.T + translation

    target = positions(reference["pose_object_to_world"])
    history = optimisation["result"]["history"]
    if (
        not trajectory
        or len(history) != len(trajectory) + 1
        or [r["iteration"] for r in history] != list(range(len(history)))
        or trajectory[-1]["pose_object_to_world"] != optimisation["pose_object_to_world"]
        or trajectory[-1]["parameters"] != optimisation["result"]["parameters"]
    ):
        raise ValueError("Accepted trajectory does not close on the reported final state")
    for pose, state in zip(trajectory, history[1:], strict=True):
        if any(pose[key] != value for key, value in state.items()):
            raise ValueError("Accepted trajectory differs from scalar optimisation history")
    poses = [optimisation["chart"]["anchor"]] + [r["pose_object_to_world"] for r in trajectory]
    records = []
    for iteration, pose in enumerate(poses):
        distances2 = np.sum((positions(pose) - target) ** 2, axis=1)
        records.append(
            {
                "iteration": iteration,
                "probe_rms_mm": float(np.sqrt(np.mean(distances2))),
                "probe_max_mm": float(np.sqrt(np.max(distances2))),
            }
        )
    for record, key in ((records[0], "initial_errors"), (records[-1], "errors")):
        for metric in ("rms", "max"):
            if not np.isclose(
                record[f"probe_{metric}_mm"],
                evaluated[key][f"landmark_{metric}_mm"],
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError("Trajectory endpoint disagrees with independent final evaluation")
    return records


def curate(bundle, root, display, review):
    # Verify every producer before taking any final image array into the display path.
    review_record = bundle.record("independent-review", review, REVIEW_SHA)
    if not review_record["status"].startswith("accepted"):
        raise ValueError("Independent numerical review has not accepted this study")
    windows = bundle.record("display-protocol", display, DISPLAY_SHA)
    if windows["log_window"] != [0, 8] or windows["final_residual_window"] != [-500, 500]:
        raise ValueError("Frozen display windows differ")
    eval_root = inside(root, "final-evaluation-v1")
    evaluated = bundle.run("evaluation", eval_root, EVALUATION_SHA)
    protocol = evaluated["configuration"]["protocol"]
    results = bundle.record(
        "evaluation/results",
        eval_root / "evaluation.json",
        evaluated["output_sha256"]["evaluation.json"],
    )
    if (
        results["evaluated_rows"] != 136
        or len(results["results"]) != 136
        or not results["no_fitting_after_reference_access"]
    ):
        raise ValueError("Final evaluation is incomplete")
    batches, observations, optimisations = {}, {}, {}
    for case in (0, 1):
        batch_root = inside(root, f"final-batch-case{case}-v1")
        obs_root = inside(root, f"final-observations-case{case}-v1")
        batch = bundle.run(
            f"case{case}/batch", batch_root, evaluated["source_sha256"][f"case{case}/batch.json"]
        )
        observation = bundle.run(
            f"case{case}/observations", obs_root, batch["source_sha256"]["generation"]
        )
        summary = bundle.record(
            f"case{case}/summary",
            batch_root / "batch-summary.json",
            batch["output_sha256"]["batch-summary.json"],
        )
        public = bundle.record(
            f"case{case}/observations/manifest",
            obs_root / "public-observations.json",
            batch["configuration"]["observation_manifest_sha256"],
        )
        if (
            observation["output_sha256"]["public-observations.json"]
            != batch["configuration"]["observation_manifest_sha256"]
            or batch["configuration"]["protocol"] != protocol
        ):
            raise ValueError("Fitting observation or protocol lineage differs")
        case_rows = [r for r in results["results"] if r["case"] == case]
        coverage(protocol, case_rows)
        coverage(protocol, summary["rows"])
        for row, original in zip(case_rows, summary["rows"], strict=True):
            if any(row.get(key) != value for key, value in original.items()):
                raise ValueError("Evaluation row differs from the frozen fitting record")
            if row["status"] != "complete":
                continue
            child = inside(batch_root, row["output"])
            run = bundle.run(f"case{case}/{row['name']}", child, row["run_sha256"])
            optimisation = bundle.record(
                f"case{case}/{row['name']}/optimisation",
                child / "optimisation.json",
                row["result_sha256"],
            )
            if (
                run["output_sha256"]["optimisation.json"] != row["result_sha256"]
                or optimisation["stationary"] != row["stationary"]
                or optimisation["result"]["reason"] != row["termination"]
            ):
                raise ValueError("Solver outcome differs from evaluated outcome")
            optimisations[case, row["name"]] = (child, run, optimisation)
        batches[case] = (batch_root, summary)
        observations[case] = (obs_root, public)

    retained = []
    for row in results["results"]:
        item = selected_fields(
            row,
            (
                "case",
                "name",
                "role",
                "comparison",
                "replicate",
                "start",
                "policy",
                "selected_angle",
                "fit_view_ids",
                "status",
                "error_type",
                "termination",
                "stationary",
                "geometric_success",
                "errors",
                "initial_errors",
                "reserved",
                "reserved_half_deviance",
                "reserved_expectation_standardised_rms",
                "run_sha256",
                "result_sha256",
            ),
        )
        if (row["case"], row["name"]) in optimisations:
            optimisation = optimisations[row["case"], row["name"]][2]
            item["final_gradient_infinity_norm"] = max(
                abs(v) for v in optimisation["result"]["evaluation"]["gradient"]
            )
            item["gradient_tolerance"] = optimisation["policy"]["gradient_tolerance"]
        retained.append(item)

    reference = bundle.record(
        "case0/disclosed-reference",
        observations[0][0] / "generator-reference.json",
        evaluated["source_sha256"]["case0/reference.json"],
    )
    if reference["case"] != 0:
        raise ValueError("Trajectory reference belongs to a different case")
    examples = []
    for comparison, name in (
        ("orthogonal", "reg-orthogonal-n00-s0"),
        ("near_parallel", "reg-near_parallel-n00-s0"),
        ("stress", "stress-n00-s0"),
    ):
        child, run, optimisation = optimisations[0, name]
        config = run["configuration"]
        panels = []
        for view in config["registration"]["views"]:
            identifier = view["id"]
            source = observations[0][1]["views"][identifier]
            if (
                config["arrays"][view["observation"]]["sha256"] != source["observation"]["sha256"]
                or config["arrays"][view["mask"]]["sha256"] != source["mask"]["sha256"]
                or source["geometry"] != view["geometry"]
            ):
                raise ValueError("Displayed observations differ from actual fitting inputs")
            observed = bundle.array(
                identifier + "/observation",
                inside(observations[0][0], source["observation"]["file"]),
                source["observation"]["sha256"],
            )
            initial = bundle.array(
                name + "/initial/" + identifier,
                child / f"initial-{identifier}.npy",
                run["output_sha256"][f"initial-{identifier}.npy"],
            )
            final = bundle.array(
                name + "/final/" + identifier,
                child / f"prediction-{identifier}.npy",
                run["output_sha256"][f"prediction-{identifier}.npy"],
            )
            mask = bundle.array(
                name + "/mask/" + identifier,
                inside(observations[0][0], source["mask"]["file"]),
                source["mask"]["sha256"],
            )
            expected_mask = np.ones((128, 128))
            if comparison == "stress":
                expected_mask[:] = 0
                expected_mask[:, 60:68] = 1
            if not np.array_equal(mask, expected_mask):
                raise ValueError("Unexpected fixed fitting support")
            geometry = view["geometry"]
            if (
                geometry["shape"] != [128, 128]
                or geometry["spacing_mm"] != [8, 8]
                or geometry["v"] != [0, 0, 1]
            ):
                raise ValueError("Unexpected native detector geometry")
            beam = view["open_beam_counts"]
            if beam != (200 if comparison == "stress" else 10000):
                raise ValueError("Unexpected recorded photon population")
            observed_name = identifier + "-observed"
            if observed_name + ".png" not in bundle.payloads:
                observed_image = bundle.image(
                    observed_name,
                    observed,
                    beam,
                    valid_mask=mask if comparison == "stress" else None,
                )
            else:
                observed_image = next(
                    p["observed"]
                    for e in examples
                    for p in e["views"]
                    if p["view_id"] == identifier
                )
            panel = {
                "view_id": identifier,
                "angle_degrees": int(identifier.rsplit("a", 1)[1]) - 180,
                "geometry": geometry,
                "fitted_columns_half_open": [60, 68] if comparison == "stress" else [0, 128],
                "outside_fit_support": (
                    "Observed pixels are invalid placeholders, displayed transparently. "
                    "Initial and final expectations outside the strip are model context only."
                    if comparison == "stress"
                    else "All displayed detector pixels were used for fitting."
                ),
                "observed": observed_image,
                "initial": bundle.image(name + "-" + identifier + "-initial", initial, beam),
                "final": bundle.image(name + "-" + identifier + "-final", final, beam),
                "input_sha256": {
                    "observed": source["observation"]["sha256"],
                    "initial": run["output_sha256"][f"initial-{identifier}.npy"],
                    "final": run["output_sha256"][f"prediction-{identifier}.npy"],
                    "mask": source["mask"]["sha256"],
                },
            }
            if comparison != "stress":
                panel["residual"] = bundle.image(
                    name + "-" + identifier + "-residual", observed - final, beam, residual=True
                )
            panels.append(panel)
        examples.append(
            {
                "comparison": comparison,
                "case": 0,
                "noise": 0,
                "start": 0,
                "name": name,
                "views": panels,
                "termination": optimisation["result"]["reason"],
                "stationary": optimisation["stationary"],
                "history": optimisation["result"]["history"],
                "trajectory": geometric_trajectory(
                    bundle,
                    name,
                    child,
                    run,
                    optimisation,
                    reference,
                    next(r for r in results["results"] if r["case"] == 0 and r["name"] == name),
                ),
            }
        )

    decisions = []
    for case, (batch_root, summary) in batches.items():
        for selection in summary["selections"]:
            item = {
                "case": case,
                **selected_fields(
                    selection,
                    (
                        "job",
                        "status",
                        "selected_angle",
                        "future_counts_access",
                        "design_run_sha256",
                        "design_result_sha256",
                    ),
                ),
            }
            if selection["status"] == "complete":
                design_root = inside(batch_root, "designs/" + selection["job"] + "/run")
                run = bundle.run(
                    f"case{case}/{selection['job']}/design",
                    design_root,
                    selection["design_run_sha256"],
                )
                design = bundle.record(
                    f"case{case}/{selection['job']}/decision",
                    design_root / "design.json",
                    selection["design_result_sha256"],
                )
                if (
                    run["output_sha256"]["design.json"] != selection["design_result_sha256"]
                    or int(design["selected_candidate"][1:]) - 180 != selection["selected_angle"]
                ):
                    raise ValueError("Selected angle differs from actual decision")
                item.update(
                    selected_candidate=design["selected_candidate"],
                    additional_incident_photons=design["maximum_cost"],
                    total_incident_photons=2 * design["maximum_cost"],
                    current_mean_target_variance_mm2=design["current_mean_target_variance_mm2"],
                    candidates=[
                        selected_fields(
                            c,
                            (
                                "name",
                                "cost",
                                "mean_target_variance_mm2",
                                "local_fisher_rank",
                                "posterior_precision_condition",
                            ),
                        )
                        for c in design["candidates"]
                    ],
                )
            decisions.append(item)

    def rms(row):
        return row.get("errors", {}).get("landmark_rms_mm")

    plots = {}
    registration = [r for r in retained if r["role"] in ("reg", "stress")]
    plots["capture"] = {
        "title": "Prescribed starts and final geometric error",
        "xLabel": "Initial 125-probe RMS displacement / mm",
        "yLabel": "Final 125-probe RMS displacement / mm",
        "series": [
            series(
                label,
                [r for r in registration if r["comparison"] == key],
                lambda r: r.get("initial_errors", {}).get("landmark_rms_mm"),
                rms,
            )
            for key, label in (
                ("orthogonal", "Orthogonal"),
                ("near_parallel", "Near-parallel"),
                ("stress", "Restricted low-dose stress"),
            )
        ],
    }
    plots["gradient"] = {
        "title": "Accepted iterates and the unchanged stationarity test",
        "xLabel": "Accepted iteration",
        "yLabel": "Scaled gradient infinity norm",
        "yScale": "log",
        "series": [
            series(
                e["comparison"],
                e["history"],
                lambda r: r["iteration"],
                lambda r: r["gradient_infinity_norm"],
                "line",
            )
            for e in examples
        ]
        + [
            {
                "label": "Original stationarity threshold",
                "x": [0, max(e["history"][-1]["iteration"] for e in examples)],
                "y": [0.001, 0.001],
                "mode": "line",
                "dash": True,
            }
        ],
    }
    plots["trajectory"] = {
        "title": "Geometric error at the recorded accepted poses",
        "xLabel": "Accepted iteration (0 is the initial anchor)",
        "yLabel": "125-probe RMS displacement / mm",
        "yScale": "log",
        "scope": "Fixed mathematical probes; initial anchor and every saved accepted pose. "
        "CPU evaluation after disclosure of the reference; no fitting or interpolated poses.",
        "series": [
            series(
                e["comparison"],
                e["trajectory"],
                lambda r: r["iteration"],
                lambda r: r["probe_rms_mm"],
                "line",
            )
            for e in examples
        ],
    }
    pairs = []
    for case in (0, 1):
        for noise in range(2):
            for start in range(4):
                matched = {
                    r["comparison"]: r
                    for r in registration
                    if r["case"] == case
                    and r["role"] == "reg"
                    and r["replicate"] == noise
                    and r["start"] == start
                }
                pairs.append(
                    {
                        "case": case,
                        "replicate": noise,
                        "start": start,
                        "orthogonal": rms(matched["orthogonal"]),
                        "near_parallel": rms(matched["near_parallel"]),
                    }
                )
    plots["view-pairs"] = {
        "title": "Paired full-view registration errors",
        "xLabel": "Orthogonal 125-probe RMS displacement / mm",
        "yLabel": "Near-parallel 125-probe RMS displacement / mm",
        "series": [
            series(
                f"Case {case}",
                [p for p in pairs if p["case"] == case],
                lambda p: p["orthogonal"],
                lambda p: p["near_parallel"],
            )
            for case in (0, 1)
        ],
        "pairs": pairs,
    }
    for role in ("broad", "constrained"):
        for case in (0, 1):
            plots[f"{role}-case{case}"] = {
                "title": f"Case {case}: {role} candidate set",
                "xLabel": "Paired noise replicate",
                "yLabel": "Final 125-probe RMS displacement / mm",
                "series": [
                    series(
                        policy.capitalize(),
                        [
                            r
                            for r in retained
                            if r["case"] == case and r["role"] == role and r["policy"] == policy
                        ],
                        lambda r: r["replicate"],
                        rms,
                    )
                    for policy in ("selected", "fixed", "random")
                ],
            }
    if any(r.get("stationary") or r.get("termination") != "line_search_failed" for r in retained):
        raise ValueError("The pinned study's explicit stopping qualification no longer holds")
    return {
        "schema_version": 1,
        "title": "Recorded CT-derived registration and finite-view selection",
        "source_doi": protocol["data"]["source_doi"],
        "licence": RIGHTS,
        "scope": (
            "Two assigned CT-derived 80 keV attenuation phantoms and simulated Poisson "
            "radiographs; mathematical probes, not clinical landmarks or patient validation."
        ),
        "qualifications": [
            "Geometric success and numerical stationarity are separate; all 136 accepted "
            "fits ended line_search_failed and none is stationary.",
            "The stress arm jointly changes photon population, view count and detector "
            "support; it does not isolate an angular effect.",
            "Fisher scores are local proxies at nonstationary accepted poses; they do not "
            "consistently improve downstream error relative to fixed-angle choices.",
            "Paired intervals describe noise within each of two phantoms; no patient-population "
            "inference. Broad and constrained roles have independent observation streams.",
            "Incident detector photon population is the cost proxy; it is not absorbed dose.",
        ],
        "geometry": {
            "shape_hw": [128, 128],
            "pixel_pitch_uv_mm": [8, 8],
            "detector_cell_face_extent_uv_mm": [-512, 512, -512, 512],
            "orientation": (
                "PNG top is +v; native rows flipped exactly once; "
                "detector physical aspect is square"
            ),
        },
        "display_protocol_sha256": DISPLAY_SHA,
        "geometric_gate": protocol["registration"]["geometric_success"],
        "geometric_probes": protocol["evaluation"]["geometric_probes"],
        "counts": {
            "outcomes": len(retained),
            "geometric_success": sum(r["geometric_success"] for r in retained),
            "stationary": sum(r.get("stationary", False) for r in retained),
            "termination": dict(Counter(r.get("termination", r["status"]) for r in retained)),
        },
        "figures": {
            "figure-11-2": {"examples": ["orthogonal", "near_parallel"]},
            "figure-11-3": {"examples": ["stress"], "plots": ["gradient", "trajectory"]},
            "figure-11-4": {"plots": ["capture"]},
            "figure-13-3": {"plots": ["broad-case0", "broad-case1"], "decision_role": "broad"},
            "figure-13-4": {
                "plots": ["constrained-case0", "constrained-case1"],
                "decision_role": "constrained",
            },
        },
        "examples": examples,
        "plots": plots,
        "outcomes": retained,
        "paired_acquisition": results["paired_acquisition"],
        "decisions": decisions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "display-protocol", "review", "output", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root, output, receipt = args.input_root.resolve(), args.output.resolve(), args.receipt.resolve()
    if (
        output.exists()
        or receipt.exists()
        or receipt.is_relative_to(output)
        or output.is_relative_to(root)
    ):
        raise ValueError("Use new output and receipt paths outside the source tree")
    bundle = Intake()
    study = curate(bundle, root, args.display_protocol, args.review)
    bundle.add(
        "study.json",
        serialise(study),
        "Positive-allowlisted recorded outcomes, decisions and display metadata",
    )
    provenance = {
        "schema_version": 1,
        "generator": "tools/public/curate-application-study.py",
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "evaluation_run_sha256": EVALUATION_SHA,
        "independent_review_sha256": REVIEW_SHA,
        "display_protocol_sha256": DISPLAY_SHA,
        "rights_record_precedes_payloads": True,
        "rights": {
            "licence": RIGHTS,
            "url": "https://creativecommons.org/licenses/by/3.0/",
            "doi": "10.7937/tcia.2019.tt7f4v7o",
            "changes": (
                "Assigned 80 keV monochromatic fields, simulated Poisson radiographs, "
                "rigid pose estimation, scalar summaries and fixed image display mapping"
            ),
            "software": (
                "xraylib 4.3.0 BSD-3-Clause coefficients; see "
                "experiments/application-study/XRAYLIB-NOTICE.txt; no source tables copied"
            ),
        },
        "scope": (
            "Native detector PNGs and explicit scalar records only; no raw CT, labels, "
            "volumetric arrays, full run dumps or private locations"
        ),
        "inputs": {tag: value["sha256"] for tag, value in bundle.inputs.items()},
        "rendering_versions": {
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": Image.__version__,
        },
        "allowlist": [
            {
                "file": name,
                "sha256": sha(data),
                "bytes": len(data),
                "origin": bundle.origins[name],
                "licence": RIGHTS,
            }
            for name, data in sorted(bundle.payloads.items())
        ],
    }
    # All rights and allowed filenames exist before the first image write.
    output.mkdir(parents=True)
    (output / "provenance.json").write_bytes(serialise(provenance))
    for name, data in bundle.payloads.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
        if (output / name).read_bytes() != data:
            raise ValueError("Written payload differs from curated bytes")
    files = {p.name: sha(p.read_bytes()) for p in output.iterdir()}
    if set(files) != set(bundle.payloads) | {"provenance.json"}:
        raise ValueError("Unexpected output file")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x") as stream:
        json.dump(
            {
                "output": str(output),
                "files": files,
                "inputs": bundle.inputs,
                "verified_archived_files": bundle.verified_files,
                "generator_sha256": sha(Path(__file__).read_bytes()),
                "all_written_bytes_verified": True,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print(
        json.dumps(
            {
                "files": len(files),
                "outcomes": len(study["outcomes"]),
                "study_sha256": files["study.json"],
            }
        )
    )


if __name__ == "__main__":
    main()
