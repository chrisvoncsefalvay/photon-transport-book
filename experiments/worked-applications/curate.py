"""Verify and plot the accepted application record; never runs or alters a fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#253c4a"
BLUE = "#376b82"
RUST = "#a35539"
PAPER = "#faf8f3"
RECIPE = (
    "https://github.com/chrisvoncsefalvay/photon-transport-"
    "book/tree/main/experiments/worked-applications"
)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def verify(root):
    records = {}
    count = 0
    for path in sorted(root.rglob("run.json")):
        if "sources" in path.parts:
            continue
        record = read(path)
        if not (
            record["status"] == "complete"
            and record["sources_unchanged"]
            and record["recorded_files_unchanged"]
        ):
            raise ValueError(f"unaccepted execution record: {path}")
        for key, base in (
            ("source_sha256", path.parent / "sources"),
            ("output_sha256", path.parent),
        ):
            for name, expected in record[key].items():
                if digest(base / name) != expected:
                    raise ValueError(f"changed evidence: {path}: {name}")
                count += 1
        records[str(path.relative_to(root))] = digest(path)
    summary = read(root / "summary.json")
    if summary["development"] or not summary["passed"] or len(records) != 34:
        raise ValueError("requires the complete accepted 25-fit, eight-decision record")
    return records, count


def save(fig, folder, name, alt):
    fig.savefig(folder / f"{name}.pdf", facecolor=PAPER)
    fig.savefig(folder / f"{name}.png", dpi=300, facecolor=PAPER)
    width, height = fig.get_size_inches() * 300
    plt.close(fig)
    return {
        "file": f"{name}.png",
        "pdf": f"{name}.pdf",
        "alt": alt,
        "width": int(width),
        "height": int(height),
    }


def curate(root, destination):
    records, hashes = verify(root)
    destination.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "pdf.fonttype": 42,
        }
    )
    evaluation = read(root / "evaluation.json")
    summary = read(root / "summary.json")
    protocol = read(root / "run.json")["configuration"]["protocol"]
    fit = read(root / "fits/registration/optimisation.json")
    reg = evaluation["registration"]
    for name in ("registration", "acquisition"):
        (destination / name).mkdir()
    folder = destination / "registration"
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.4), layout="constrained")
    for row, suffix in enumerate(("a180", "a270")):
        view = f"r1-n00-{suffix}"
        paths = [
            root / f"fits/registration/initial-{view}.npy",
            root / f"observations/{view}.npy",
            root / f"fits/registration/prediction-{view}.npy",
        ]
        for col, (path, label) in enumerate(
            zip(
                paths,
                ("Initial prediction", "Observed counts", "Recovered prediction"),
                strict=True,
            )
        ):
            values = np.load(path).reshape(128, 128)
            im = axes[row, col].imshow(
                np.log10(1 + values),
                vmin=0,
                vmax=np.log10(10001),
                cmap="gray",
                interpolation="nearest",
                origin="lower",
                extent=(-512, 512, -512, 512),
            )
            axes[row, col].set_title(f"{label} · {row * 90}°", fontsize=10)
            axes[row, col].set_xlabel("Detector column coordinate (mm)")
            if col == 0:
                axes[row, col].set_ylabel("Detector row coordinate (mm)")
    fig.colorbar(im, ax=axes, label="Display only: log₁₀(1 + counts)", shrink=0.72)
    images = [
        save(
            fig,
            folder,
            "images",
            (
                "Both orthogonal views show initial predictions, supplied Poisson observations "
                "and recovered predictions in one fixed logarithmic window."
            ),
        )
    ]
    history = fit["result"]["history"]
    trajectory = read(root / "fits/registration/trajectory.json")
    reference = read(root / "evaluation-reference.json")
    points = np.array(reference["task_points_object_mm"])

    def transformed(pose):
        return points @ np.array(pose["rotation"]).reshape(3, 3).T + pose["translation_mm"]

    truth = transformed(reference["pose_object_to_world"])
    errors = [reg["initial_errors"]["landmark_rms_mm"]] + [
        float(
            np.sqrt(np.mean(np.sum((transformed(t["pose_object_to_world"]) - truth) ** 2, axis=1)))
        )
        for t in trajectory
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), layout="constrained")
    steps = [h["iteration"] for h in history]
    axes[0].semilogy(
        steps, [h["gradient_infinity_norm"] for h in history], "o-", color=BLUE, markersize=3
    )
    axes[0].axhline(0.001, color=RUST, linestyle="--", label="Stopping threshold: 0.001")
    axes[0].set_ylabel("Scaled gradient infinity norm")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].semilogy(steps, errors, "s-", color=INK, markersize=3)
    axes[1].set_ylabel("Independent RMS target error (mm)")
    for ax in axes:
        ax.set_xlabel("Accepted update (0 = initial)")
        ax.grid(alpha=0.15)
    images.append(
        save(
            fig,
            folder,
            "convergence",
            (
                f"{len(history) - 1} accepted updates reduce the scaled gradient below 0.001. "
                f"RMS target error falls from {errors[0]:.2f} mm to {errors[-1]:.3f} mm."
            ),
        )
    )
    grad = max(map(abs, fit["result"]["evaluation"]["gradient"]))
    write(
        folder / "study.json",
        {
            "schema_version": 1,
            "passed": True,
            "title": "From two radiographs to a recovered pose",
            "description": (
                "Recorded CT-ORG case 2 teaching example: assigned 80 keV attenuation, simulated "
                "counts, identity start and independently evaluated recovery."
            ),
            "images": images,
            "measurements": [
                {
                    "label": "Solver termination",
                    "value": f"Gradient tolerance; {len(history) - 1} accepted updates",
                    "criterion": "Scaled gradient ≤ 0.001",
                },
                {"label": "Final scaled gradient", "value": f"{grad:.6g}", "criterion": "≤ 0.001"},
                {
                    "label": "RMS target error",
                    "value": f"{reg['initial_errors']['landmark_rms_mm']:.3f} → {
                        reg['errors'][('landmark_rms_mm')]:.5f} mm",
                    "criterion": "≤ 1 mm",
                },
                {
                    "label": "Worst target error",
                    "value": f"{reg['errors']['landmark_max_mm']:.5f} mm",
                    "criterion": "≤ 2 mm",
                },
                {
                    "label": "Reserved-view mean error",
                    "value": (
                        f"{max(x['expected_error_poisson_sd_rms'] for x in reg['held_out']):.5f}"
                        " Poisson SD RMS"
                    ),
                    "criterion": "≤ 1 Poisson SD RMS in both views",
                },
            ],
            "reading": (
                "The optimiser sees the two count images, the known volume and fixed calibration."
                " After fixing the result, the evaluator compares its action on the eight target "
                "points with their generating positions. The final gradient and target error "
                "answer different questions, and both pass."
            ),
            "reproduction": RECIPE,
        },
    )
    write(
        folder / "data.json",
        {
            "evaluation": reg,
            "history": history,
            "independent_rms_error_mm": errors,
            "accepted_pose": fit["pose_object_to_world"],
        },
    )
    folder = destination / "acquisition"
    design = read(root / "decision-00/run/design.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    angles = protocol["acquisition"]["candidate_angles_degrees"]
    variances = [c["mean_target_variance_mm2"] for c in design["candidates"]]
    axes[0].bar(
        [str(int(a)) for a in angles],
        variances,
        color=[
            BLUE if f"a{int(a) + 180}" == design["selected_candidate"] else "#aaa69f"
            for a in angles
        ],
    )
    axes[0].set_xlabel("Candidate view (degrees)")
    axes[0].set_ylabel("Predicted mean target variance (mm²)")
    axes[0].set_title("First prescribed decision", fontsize=11)
    selected = np.array([p["selected"]["errors"]["landmark_rms_mm"] for p in evaluation["pairs"]])
    fixed = np.array([p["near_parallel"]["errors"]["landmark_rms_mm"] for p in evaluation["pairs"]])
    for i, (s, f) in enumerate(zip(selected, fixed, strict=True)):
        axes[1].plot([i, i], [f, s], "-", color="#aaa69f", alpha=0.8)
        axes[1].scatter(
            [i], [f], color=RUST, marker="s", s=24, label="Near-parallel 5°" if i == 0 else None
        )
        axes[1].scatter(
            [i], [s], color=BLUE, marker="o", s=24, label="Selected view" if i == 0 else None
        )
    axes[1].set_xticks(range(8))
    axes[1].set_xlabel("Prescribed noise replicate")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_ylabel("Measured RMS target error (mm)")
    axes[1].set_title("All eight prescribed noise pairs", fontsize=11)
    images = [
        save(
            fig,
            folder,
            "decision-and-recovery",
            (
                f"The decision favours {int(design['selected_candidate'][1:]) - 180} degrees. "
                "All eight paired outcomes compare the "
                "selected view with the near-parallel 5-degree baseline at equal exposure."
            ),
        )
    ]
    write(
        folder / "study.json",
        {
            "schema_version": 1,
            "passed": True,
            "title": "Choose a view, acquire its counts, check the recovery",
            "description": (
                "Equal-exposure comparison on the same eight object-frame targets. The future "
                "count image is generated only after each decision is recorded."
            ),
            "images": images,
            "measurements": [
                {
                    "label": "Completed solves",
                    "value": (
                        "25 / 25 stationary, geometrically accurate and reserved-view checks passed"
                    ),
                    "criterion": "Every prescribed solve passes",
                },
                {
                    "label": "Selected mean squared target error",
                    "value": f"{summary['selected_mean_squared_task_error_mm2']:.7f} mm²",
                    "criterion": "≤ 75% of near-parallel baseline",
                },
                {
                    "label": "Near-parallel mean squared target error",
                    "value": f"{summary['near_parallel_mean_squared_task_error_mm2']:.7f} mm²",
                    "criterion": "Same exposure and solver",
                },
                {
                    "label": "Selected / baseline",
                    "value": f"{summary['selected_to_near_parallel_ratio']:.4f}",
                    "criterion": "≤ 0.75",
                },
                {
                    "label": "Paired descriptive 95% upper bound",
                    "value": f"{summary['paired_descriptive_t95_upper_mm2']:.7f} mm²",
                    "criterion": "Below zero",
                },
            ],
            "reading": (
                f"The selected view reduces mean squared target error by "
                f"{100 * (1 - summary['selected_to_near_parallel_ratio']):.0f}% relative to the "
                f"near-parallel view. {int(np.count_nonzero(selected > fixed))} of the "
                f"{len(selected)} pairs have higher error after "
                "selection: the criterion predicts an average over possible count images, while "
                "each pair receives one noise realisation. This controlled example demonstrates "
                "the average benefit; it does not promise to beat every well-chosen fixed view."
            ),
            "reproduction": RECIPE,
        },
    )
    write(
        folder / "data.json",
        {"summary": summary, "pairs": evaluation["pairs"], "first_decision": design},
    )
    provenance = {
        "schema_version": 1,
        "verified_records": records,
        "verified_hash_count": hashes,
        "source_sha256": read(root / "run.json")["source_sha256"],
        "protocol": protocol,
        "input_sha256": protocol["known_attenuation_sha256"],
        "curator_sha256": digest(Path(__file__)),
        "source_rights": (
            "CT-ORG case 2, Rister et al., DOI 10.7937/tcia.2019.tt7f4v7o, CC BY 3.0. "
            "Assigned water/bone 80 keV attenuation; xraylib 4.3.0 BSD-3-Clause notice "
            "accompanies the recipe."
        ),
        "display": (
            "Full detectors; fixed log10(1+counts) window [0,log10(10001)], origin lower; no "
            "interpolation or smoothing of measurements. All noise pairs retained."
        ),
        "numerical_record_sha256": {
            str(p.relative_to(root)): digest(p)
            for p in (root / "summary.json", root / "evaluation.json", root / "sampling.json")
        },
    }
    for name in ("registration", "acquisition"):
        write(destination / name / "provenance.json", provenance)
    print(
        f"Verified {len(records)} complete records and {hashes} hashes; "
        f"plots written to {destination}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    curate(args.run, args.output)
