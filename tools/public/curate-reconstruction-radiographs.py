"""Render separate book panels from pinned, completed radiograph records.

This stages display derivatives only. It never evaluates a projector or solver.
The source arrays and previous annotated figures remain unchanged. Rights and a
complete output allowlist are written before any image payload.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Mathematical minus and range signs are intentional reader-facing labels.
# ruff: noqa: RUF001

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "curation", ROOT / "tools/public/curate-reconstruction.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("The verified reconstruction curation helper is unavailable")
curation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curation)


def array(bundle, tag, path, expected):
    result = np.load(io.BytesIO(bundle.read(tag, path, expected)), allow_pickle=False)
    if not isinstance(result, np.ndarray) or not np.isfinite(result).all():
        raise ValueError("Expected a finite recorded numerical array")
    return result


def panel(bundle, case, identifier, label, values, limits, rights, *, residual=False, valid=None):
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("A panel must contain finite recorded scalar values")
    palette = plt.colormaps["RdBu_r" if residual else "gray"].with_extremes(bad="#d9b4d5")
    displayed = np.where(valid, values, np.nan) if valid is not None else values
    output = io.BytesIO()
    plt.imsave(
        output,
        displayed,
        format="png",
        cmap=palette,
        vmin=limits[0],
        vmax=limits[1],
        origin="lower",
    )
    record = bundle.image(
        case,
        identifier,
        label,
        label,
        output.getvalue(),
        "Fixed-window raster display of the verified selected numerical source",
        rights,
    )
    record["limits"] = limits
    record["palette"] = "RdBu_r" if residual else "gray"
    record["masked_pixels"] = int(np.count_nonzero(~valid)) if valid is not None else 0
    record["low_clipped_pixels"] = int(
        np.count_nonzero((values < limits[0]) & (valid if valid is not None else True))
    )
    record["high_clipped_pixels"] = int(
        np.count_nonzero((values > limits[1]) & (valid if valid is not None else True))
    )
    return record


def projection_rig(settings, images):
    """Rigid display coordinates for recorded rays; no transport is evaluated."""
    geometry = settings["geometry"]
    height, width = geometry["shape"]
    du, dv = geometry["spacing_mm"]
    source_world = np.asarray(geometry["source_mm"], dtype=np.float64)
    origin_world = np.asarray(geometry["origin_mm"], dtype=np.float64)
    views = []
    for image, colour in zip(images, np.eye(3), strict=True):
        index = image["view_zero_based"]
        pose = settings["poses"][index]
        rotation = np.asarray(pose["rotation"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(pose["translation_mm"], dtype=np.float64)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0):
            raise ValueError("A recorded pose must be rigid")
        source = rotation.T @ (source_world - translation)
        origin = rotation.T @ (origin_world - translation)
        u, v = (rotation.T @ np.asarray(geometry[axis]) for axis in ("u", "v"))
        centre = origin + (width - 1) * du * u / 2 + (height - 1) * dv * v / 2
        normal = (centre - source) / np.linalg.norm(centre - source)
        distance = float(normal @ (centre - source))
        if not np.allclose(np.cross(u, v), -normal, atol=1e-12, rtol=0):
            raise ValueError("The pinned detector orientation changed")
        # Full cell faces are UV 0/1; pixel (0,0) is half a cell inside them.
        a = ((source - origin) @ u + du / 2) / (width * du)
        b = ((source - origin) @ v + dv / 2) / (height * dv)
        rows = [
            distance * u / (width * du) + a * normal,
            distance * v / (height * dv) + b * normal,
            normal,
        ]
        matrix = [np.r_[row, -row @ source].tolist() for row in rows]
        corners = [
            (origin + col * du * u + row * dv * v).tolist()
            for row, col in (
                (-0.5, -0.5),
                (-0.5, width - 0.5),
                (height - 0.5, width - 0.5),
                (height - 0.5, -0.5),
            )
        ]
        views.append(
            {
                "view_zero_based": index,
                "label": image["label"],
                "colour_rgb": colour.tolist(),
                "source_mm": source.tolist(),
                "detector_corners_mm": corners,
                "source_detector_distance_mm": distance,
                "uv_matrix_rows": matrix,
                "image": image,
            }
        )
    return {
        "coordinate_frame": "Centred object XYZ in mm, shared with the recorded surfaces",
        "pose_kind": "Known acquisition poses, fixed during material reconstruction",
        "texture_coordinates": "Native bottom-origin UV; upload PNG with exactly one vertical flip",
        "display": "RGB identifies views; windowed radiograph projection is a display cue only",
        "views": views,
    }


def ct_case(bundle, root):
    folder = root / "reconstruction-vertebrae-v1"
    run = bundle.record("ct/run", folder / "run.json", curation.CT_RUN)
    if run["status"] != "complete":
        raise ValueError("Only completed CT-derived records are admitted")
    config = run["configuration"]
    acquisition = Path(config["acquisition"])
    acquired = bundle.record(
        "ct/acquisition", acquisition / "run.json", config["acquisition_sha256"]
    )
    if acquired["status"] != "complete":
        raise ValueError("Acquisition is incomplete")
    settings = acquired["configuration"]
    observed = array(
        bundle,
        "ct/counts",
        acquisition / "observed-counts.npy",
        acquired["output_sha256"]["observed-counts.npy"],
    )
    expected = array(
        bundle,
        "ct/expectation",
        acquisition / "expected-counts.npy",
        acquired["output_sha256"]["expected-counts.npy"],
    )
    predicted = array(
        bundle,
        "ct/predictions",
        folder / "heldout-predictions.npy",
        run["output_sha256"]["heldout-predictions.npy"],
    )
    physics = Path(settings["physics"])
    with np.load(
        io.BytesIO(
            bundle.read("ct/physics", physics / "physics.npz", settings["physics_npz_sha256"])
        ),
        allow_pickle=False,
    ) as inputs:
        open_beam = (inputs["weights"] * inputs["response"]).sum(axis=1)
    metadata = bundle.record(
        "ct/physics-metadata", physics / "metadata.json", settings["physics_metadata_sha256"]
    )
    view = settings["heldout_views"][0]
    if (
        view in settings["fit_views"]
        or view != 4
        or observed.shape != expected.shape
        or predicted.shape[0] != len(settings["heldout_views"])
    ):
        raise ValueError("The pinned CT withheld-view layout changed")
    if min(observed.min(), expected.min(), predicted.min()) < 0 or not np.array_equal(
        observed, np.floor(observed)
    ):
        raise ValueError("Recorded count domain changed")
    h, w = settings["geometry"]["shape"]
    du, dv = settings["geometry"]["spacing_mm"]
    case = {
        "id": "vertebrae",
        "title": "What the spectral reconstruction predicts",
        "kind": "ct-derived-simulation",
        "view_zero_based": view,
        "role": "whole withheld view",
        "extent_uv_mm": [-w * du / 2, w * du / 2, -h * dv / 2, h * dv / 2],
        "extent_convention": "Full detector cell-face extent in centred detector coordinates",
        "aspect": w * du / (h * dv),
        "channels": [],
        "input_views": [],
        "licence": curation.CT_RIGHTS,
        "source_url": "https://doi.org/10.7937/tcia.2019.tt7f4v7o",
        "observation_display": (
            "-log((counts+0.5)/open_beam);0.5 is a display-only continuity correction"
        ),
        "residual_display": "(observed-predicted)/sqrt(max(predicted,1)); limits[-5,5]",
        "qualification": (
            "Assigned CT-derived material phantom with simulated independent Poisson channels; "
            "finite-budget reconstruction, not measured patient composition."
        ),
    }
    edges = metadata["acquisition"]["thresholds_kev"]
    for input_view in (0, 16, 32):
        if input_view not in settings["fit_views"]:
            raise ValueError("An opening radiograph must actually belong to the fitting set")
        image = panel(
            bundle,
            case,
            f"input-view-{input_view}",
            f"Fitting view {input_view + 1}",
            -np.log((observed[input_view, 0] + 0.5) / open_beam[0]),
            [0.0, 8.0],
            curation.CT_RIGHTS,
        )
        image["view_zero_based"] = input_view
        image["channel"] = "20–55 keV"
        case["input_views"].append(image)
    case["projection_rig"] = projection_rig(settings, case["input_views"])
    for c in range(3):
        expected_display = -np.log((expected[view, c] + 0.5) / open_beam[c])
        upper = max(1.0, float(np.ceil(np.percentile(expected_display, 99.5) * 2) / 2))
        if upper != [8, 5.5, 4][c]:
            raise ValueError("The original shared reference-derived window changed")
        channel = {
            "id": str(c),
            "label": f"{edges[c]:g}–{edges[c + 1]:g} keV",
            "open_beam": float(open_beam[c]),
            "panels": [],
        }
        for key, label, values in [
            ("observed", "Simulated observation", observed[view, c]),
            ("expected", "Generating expectation", expected[view, c]),
            ("predicted", "Withheld-view prediction", predicted[0, c]),
        ]:
            channel["panels"].append(
                panel(
                    bundle,
                    case,
                    f"channel-{c}-{key}",
                    label,
                    -np.log((values + 0.5) / open_beam[c]),
                    [0.0, upper],
                    curation.CT_RIGHTS,
                )
            )
        residual = (observed[view, c].astype(np.float64) - predicted[0, c]) / np.sqrt(
            np.maximum(predicted[0, c], 1)
        )
        channel["panels"].append(
            panel(
                bundle,
                case,
                f"channel-{c}-residual",
                "Observation − prediction (Poisson scaled)",
                residual,
                [-5.0, 5.0],
                curation.CT_RIGHTS,
                residual=True,
            )
        )
        case["channels"].append(channel)
    return case


def hap_case(bundle, root):
    folder = root / "root/refinement-recovery-v1"
    run = bundle.record("hap/run", folder / "run.json", curation.HAP_RUN)
    if run["status"] != "complete":
        raise ValueError("Only completed acquired records are admitted")
    prepared = Path(run["prepared"])
    sampled = Path(run["sampled"])
    observed = array(
        bundle,
        "hap/counts",
        sampled / "HAP_Phantom-counts.npy",
        run["input_hashes"]["observations"],
    )
    predicted = array(
        bundle, "hap/predictions", folder / "fitting-predictions.npy", run["predictions_sha256"]
    )
    with np.load(
        io.BytesIO(
            bundle.read("hap/prepared", prepared / "prepared.npz", run["input_hashes"]["prepared"])
        ),
        allow_pickle=False,
    ) as inputs:
        air, masks = inputs["air_total_high"], inputs["valid"].astype(bool)
    if not np.array_equal(masks[0], masks[1]):
        raise ValueError("The original shared detector mask changed")
    valid = masks[0]
    figures = bundle.record(
        "hap/figure-receipt",
        root / "root/figures-refinement-v1/figures.json",
        "f9f491f50624057592ac0b7e460087fbe03916d8d9b721e40411ba73aac37a60",
    )
    extent = figures["detector_display_extent_mm"]
    view = run["fitting_view_ids"][0]
    if view != 1 or observed.shape[1:] != predicted.shape[1:] or np.count_nonzero(valid) != 15583:
        raise ValueError("The pinned acquired training preview changed")
    measured, model = observed[view].astype(np.float64), predicted[0].astype(np.float64)
    case = {
        "id": "hap",
        "title": "The acquired phantom radiographs",
        "kind": "acquired-phantom",
        "view_zero_based": view,
        "role": "fitting view; not a withheld evaluation",
        "extent_uv_mm": extent,
        "extent_convention": (
            "Native pixel-zero-centre coordinates; the extent spans sampled-cell spacing, "
            "not the physical pixel aperture"
        ),
        "aspect": (extent[1] - extent[0]) / (extent[3] - extent[2]),
        "channels": [],
        "licence": curation.HAP_RIGHTS,
        "source_url": "https://doi.org/10.5281/zenodo.17328375",
        "observation_display": "-log(max(transmission,1e-6)); limits[0,2.5]; masked pixels purple",
        "residual_display": "predicted minus observed normalised transmission; limits[-0.15,0.15]",
        "qualification": (
            "Acquired cumulative Total/High observations; displayed Low is their difference. "
            "The fitted effective response fails a reserved calibration criterion. "
            "The shown frame was used for fitting."
        ),
    }
    for c, label in enumerate(["Low (Total − High)", "High"]):
        denominator = air[0] - air[1] if c == 0 else air[1]
        first = measured[0] - measured[1] if c == 0 else measured[1]
        second = model[0] - model[1] if c == 0 else model[1]
        if np.any(denominator[valid] <= 0):
            raise ValueError("Invalid recorded open-beam normalisation")
        transmission = [first / denominator, second / denominator]
        channel = {"id": str(c), "label": label, "panels": []}
        for i, (key, name) in enumerate(
            [("observed", "Acquired observation"), ("predicted", "Fitted prediction")]
        ):
            channel["panels"].append(
                panel(
                    bundle,
                    case,
                    f"channel-{c}-{key}",
                    name,
                    -np.log(np.maximum(transmission[i], 1e-6)),
                    [0.0, 2.5],
                    curation.HAP_RIGHTS,
                    valid=valid,
                )
            )
        channel["panels"].append(
            panel(
                bundle,
                case,
                f"channel-{c}-residual",
                "Prediction − observation (normalised transmission)",
                transmission[1] - transmission[0],
                [-0.15, 0.15],
                curation.HAP_RIGHTS,
                residual=True,
                valid=valid,
            )
        )
        case["channels"].append(channel)
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-root", type=Path, required=True)
    parser.add_argument("--hap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.receipt.exists()
        or args.receipt.resolve().is_relative_to(args.output.resolve())
    ):
        raise ValueError("Use separate new output and receipt paths")
    bundle = curation.Curation()
    study = {
        "schema_version": 1,
        "cases": [ct_case(bundle, args.spectral_root), hap_case(bundle, args.hap_root)],
    }
    bundle.add(
        "study.json",
        curation.json_bytes(study),
        "Verified radiograph display records",
        "Project metadata; cited source licences retained",
    )
    provenance = {
        "schema_version": 1,
        "generator_sha256": curation.sha(Path(__file__).read_bytes()),
        "helper_sha256": curation.sha(
            (ROOT / "tools/public/curate-reconstruction.py").read_bytes()
        ),
        "inputs": bundle.inputs,
        "rights_record_precedes_payloads": True,
        "source_rights": [
            {"case": c["id"], "licence": c["licence"], "source_url": c["source_url"]}
            for c in study["cases"]
        ],
        "transformation": (
            "Fixed-window scalar rasterisation at original selected detector dimensions; "
            "lower-origin rows stored for ordinary top-origin image display; "
            "recorded rigid acquisition poses inverted into the shared object frame; "
            "full detector cell-face corners and bottom-origin projective UV maps; "
            "no smoothing or reconstructed image synthesis"
        ),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "allowlist": [
            {"file": name, "sha256": curation.sha(data), "bytes": len(data), **bundle.origins[name]}
            for name, data in sorted(bundle.payloads.items())
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "provenance.json").write_bytes(curation.json_bytes(provenance))
    for name, data in bundle.payloads.items():
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as file:
            file.write(data)
        if target.read_bytes() != data:
            raise ValueError("Rendered output changed during writing")
    files = {
        p.relative_to(args.output).as_posix(): curation.sha(p.read_bytes())
        for p in args.output.rglob("*")
        if p.is_file()
    }
    if set(files) != set(bundle.payloads) | {"provenance.json"}:
        raise ValueError("Output differs from the positive allowlist")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        __import__("json").dumps(
            {
                "inputs": bundle.private_inputs,
                "output": str(args.output.resolve()),
                "files": files,
                "count": len(files),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Rendered {len(files)} verified files into {args.output}")


if __name__ == "__main__":
    main()
