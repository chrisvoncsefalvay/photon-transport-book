"""Render an accepted worked reconstruction, with reproducible numerical inputs.

This command writes a new external staging directory. It does not admit assets,
publish a site or manufacture a successful result from a stopped fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from run import checked_array, checked_json, complete_record, digest


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_figure(fig: Any, output: Path, name: str, alt: str) -> dict[str, Any]:
    fig.savefig(output / f"{name}.png", dpi=300)
    fig.savefig(output / f"{name}.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    with Image.open(output / f"{name}.png") as image:
        width, height = image.size
    return {
        "file": f"{name}.png",
        "pdf": f"{name}.pdf",
        "alt": alt,
        "width": width,
        "height": height,
    }


def curate(source: Path, output: Path) -> None:
    outer = complete_record(source)
    summary = checked_json(source, "summary.json", outer)
    if summary.get("development") or not summary.get("accepted"):
        raise ValueError("curation requires a scientifically accepted final two-replicate run")
    if [row["replicate"] for row in summary["replicates"]] != [0, 1]:
        raise ValueError("every prescribed replicate must be present in order")
    for relative, expected in summary["child_records_sha256"].items():
        if digest(source / relative) != expected:
            raise ValueError(f"a completed child record changed: {relative}")
    acquisition = source / "acquisition"
    acquisition_record = complete_record(acquisition)
    public = checked_json(acquisition, "fitting.json", acquisition_record)
    reference = checked_array(acquisition, "evaluation/reference-fractions.npy", acquisition_record)
    if reference.shape != (2, 16, 16, 16):
        raise ValueError(
            "this fixed display protocol requires the admitted two-material 16-cubed basis"
        )
    fields, initials, checkpoints, results, evaluations = [], [], [], [], []
    for replicate in (0, 1):
        fit, evaluation = source / f"fit-rep{replicate}", source / f"evaluation-rep{replicate}"
        fit_record, evaluation_record = complete_record(fit), complete_record(evaluation)
        result = checked_json(fit, "solver-result.json", fit_record)
        assessment = checked_json(evaluation, "evaluation.json", evaluation_record)
        if not assessment["accepted"] or not result["stationarity"]["passed"]:
            raise ValueError("a prescribed field fails scientific acceptance")
        if assessment != summary["replicates"][replicate]:
            raise ValueError("summary and independent evaluation differ")
        fields.append(checked_array(fit, "fields.npy", fit_record))
        initials.append(checked_array(fit, "checkpoints/fields-0000.npy", fit_record))
        checkpoint = fit / "checkpoints/fields-0100.npy"
        checkpoints.append(
            checked_array(fit, "checkpoints/fields-0100.npy", fit_record)
            if checkpoint.exists()
            else None
        )
        results.append(result)
        evaluations.append(assessment)
    if output.exists():
        raise ValueError("curation requires a new staging directory")
    output.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )
    images = []
    colours, styles = ["#0072B2", "#D55E00"], ["-", "--"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), constrained_layout=True)
    curves = []
    for replicate, result in enumerate(results):
        history = [row for row in result["history"] if row["accepted"]]
        steps = [0] + [row["iteration"] for row in history]
        objective = [result["initial_objective"]] + [row["objective"] for row in history]
        mapping = [row["gradient_mapping_before"] for row in history] + [
            result["stationarity"]["final_mapping"]
        ]
        axes[0].semilogy(
            steps,
            objective,
            styles[replicate],
            color=colours[replicate],
            label=f"Replicate {replicate}",
        )
        axes[1].semilogy(
            steps,
            mapping,
            styles[replicate],
            color=colours[replicate],
            label=f"Replicate {replicate}",
        )
        curves.append(
            {"replicate": replicate, "steps": steps, "objective": objective, "mapping": mapping}
        )
    axes[0].set(
        title="Fitting objective",
        xlabel="Accepted update",
        ylabel="Poisson half-deviance + penalty",
    )
    axes[1].set(
        title="Stationarity at the accepted field",
        xlabel="Accepted update",
        ylabel="Euclidean projected mapping",
    )
    threshold = results[0]["stationarity"]["absolute_threshold"]
    axes[1].axhline(threshold, color="#333333", linestyle=":", label=f"Required ≤ {threshold:g}")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    images.append(
        save_figure(
            fig,
            output,
            "convergence",
            "Both prescribed final reconstructions reduce the fitting objective and reach "
            "the unchanged Euclidean stationarity threshold.",
        )
    )

    def plane(field: np.ndarray) -> np.ndarray:
        return 0.5 * (field[:, 7].astype(np.float64) + field[:, 8].astype(np.float64))

    grid = public["grid"]
    width, height = (
        grid["shape"][2] * grid["spacing_mm"][0],
        grid["shape"][1] * grid["spacing_mm"][1],
    )
    extent = (-width / 2, width / 2, -height / 2, height / 2)
    display = {
        "plane": "object-frame z=0 mm; equal interpolation of sample planes 7 and 8",
        "fraction_window": [0, 1],
        "signed_error_window": [-0.1, 0.1],
        "pixel_interpolation": "nearest for displaying the recorded coarse plane",
    }
    display_arrays: dict[str, Any] = {"reference": reference}
    for replicate, recovered in enumerate(fields):
        snapshots = [plane(reference), plane(initials[replicate])]
        titles = ["Assigned reference", "Uniform start"]
        if checkpoints[replicate] is not None:
            snapshots.append(plane(checkpoints[replicate]))
            titles.append("Update 100")
        snapshots.extend([plane(recovered), plane(recovered) - plane(reference)])
        titles.extend(["Stationary field", "Signed final error"])
        fig, axes = plt.subplots(2, len(snapshots), figsize=(10.5, 4.7), constrained_layout=True)
        for material, name in enumerate(("Water", "Bone")):
            for column, values in enumerate(snapshots):
                error = column == len(snapshots) - 1
                image = axes[material, column].imshow(
                    values[material],
                    origin="lower",
                    extent=extent,
                    interpolation="nearest",
                    cmap="RdBu_r" if error else "viridis",
                    vmin=-0.1 if error else 0,
                    vmax=0.1 if error else 1,
                )
                axes[material, column].set_title(titles[column], fontsize=9)
                axes[material, column].set_xlabel("x (mm)")
                axes[material, column].set_ylabel(f"{name}\ny (mm)" if column == 0 else "")
                if error:
                    fig.colorbar(
                        image,
                        ax=axes[material, column],
                        label="Fraction error",
                        shrink=0.8,
                        extend="both",
                    )
            fraction_map = axes[material, 0].images[0]
            fig.colorbar(
                fraction_map, ax=list(axes[material, :-1]), label="Volume fraction", shrink=0.8
            )
        images.append(
            save_figure(
                fig,
                output,
                f"materials-rep{replicate}",
                f"Fixed physical zero-plane fields for replicate {replicate}: assigned water "
                "and bone, uniform start, available update-100 checkpoint, stationary recovery "
                "and signed error. Coarse matched-model assigned fractions, "
                "not patient composition.",
            )
        )
        display_arrays[f"initial_rep{replicate}"] = initials[replicate]
        display_arrays[f"final_rep{replicate}"] = recovered
        if checkpoints[replicate] is not None:
            display_arrays[f"update100_rep{replicate}"] = checkpoints[replicate]
    np.savez_compressed(output / "fields.npz", **display_arrays)
    measurements = []
    for replicate, (result, evaluation) in enumerate(zip(results, evaluations, strict=True)):
        measurements.extend(
            [
                {
                    "label": f"Replicate {replicate}: accepted updates",
                    "value": str(result["accepted_steps"]),
                    "criterion": "Stop by projected-gradient tolerance",
                },
                {
                    "label": f"Replicate {replicate}: final mapping",
                    "value": f"{result['stationarity']['final_mapping']:.5g}",
                    "criterion": f"≤ {result['stationarity']['absolute_threshold']:g}",
                },
                {
                    "label": f"Replicate {replicate}: water / bone RMSE",
                    "value": " / ".join(
                        f"{value:.5f}" for value in evaluation["material_rmse_water_bone"]
                    ),
                    "criterion": "Each ≤ 0.05; whole coarse box",
                },
                {
                    "label": f"Replicate {replicate}: withheld mean error",
                    "value": f"{evaluation['withheld']['rms_poisson_sd']:.5f} Poisson SD RMS",
                    "criterion": "≤ 1 across all 12 withheld views",
                },
            ]
        )
    study = {
        "schema_version": 1,
        "passed": True,
        "title": "From spectral counts to two recovered material fields",
        "description": (
            "CT-ORG case 2 supplies assigned water/bone anatomy in a declared matched "
            "16³ trilinear basis. Both independent prescribed noise replicates are retained."
        ),
        "images": images,
        "measurements": measurements,
        "reading": (
            "The starting field is uniform water. Only the 48 fitting views enter either solve. "
            "Both fields and their stopping diagnostics are fixed before the evaluator opens "
            "assigned references and 12 disjoint withheld views. The mapping tests constrained "
            "stationarity; material and withheld errors test recovery under the declared "
            "matched model. All must pass."
        ),
        "reproduction": "https://github.com/chrisvoncsefalvay/photon-transport-book/tree/main/experiments/worked-reconstruction",
    }
    write_json(output / "study.json", study)
    write_json(
        output / "data.json",
        {
            "protocol": public["protocol"],
            "curves": curves,
            "evaluations": evaluations,
            "display": display,
        },
    )
    files = {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    write_json(
        output / "provenance.json",
        {
            "source": "https://doi.org/10.7937/tcia.2019.tt7f4v7o",
            "rights": (
                "CC BY 3.0; Rister et al., CT-ORG, TCIA; assigned-material display derivatives"
            ),
            "source_file_sha256": public["protocol"]["source_sha256"],
            "execution_record_sha256": digest(source / "run.json"),
            "child_records_sha256": summary["child_records_sha256"],
            "curator_sha256": digest(Path(__file__)),
            "display": display,
            "output_sha256": files,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    curate(args.source, args.output)
