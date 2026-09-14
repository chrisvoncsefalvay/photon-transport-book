"""Stage the recorded CT -> simulated radiograph -> recovered bone introduction.

Render native CT samples and the archived mesh; perform no reconstruction or
medical-image resampling. Reuse the existing fitting radiograph by URL and hash.
Only the two PNGs and positively selected metadata may enter the new destination.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from PIL import __version__ as pillow_version

# Ranges in display labels intentionally use typographic en dashes.
# ruff: noqa: RUF001

ROOT = Path(__file__).resolve().parents[2]
HELPER_SHA = "8bcbee171edaa702f1a79b4ecbcb864a900a3b2a27b33ac5b0e2558747437557"
RUN_SHA = "fab471b81eb132b798ef62cd958a02e35f2a8a06d85d52d808bb2bc682f4e869"
FIGURES_SHA = "e86f08ce4d6e38c72842fe7845a757f15a71645c4ce6dee97f24f7812a630bc8"
ANATOMY_SHA = "b604e4dc2d5dac018d69f06ca734c490dec04c9d92a9361d2560f11b83203966"
PUBLIC_STUDY_SHA = "9e45151567fa0dea8cdbed589c33ee41170c09db54c4b9da376471bed8ff093d"
RADIOGRAPH_STUDY_SHA = "64a1e772493d2bec8593a78251296c66dc7d725075ba4c551ffcdce56dd0c9e8"
RADIOGRAPH_SHA = "091cdabf198f0808bed919b8d55cad8562b9f2819c085d34a4f2dfe3d2e474f4"
RIGHTS = "CC BY 3.0; derived from Rister et al., CT-ORG, TCIA"
PREFIX = "generated/introduction-study/"


def helper_module():
    import hashlib

    path = ROOT / "tools/public/curate-reconstruction.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError("The reviewed curation helper has changed")
    spec = importlib.util.spec_from_file_location("introduction_curation_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the fixed curation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_record(bundle, name, data, alt, origin):
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        image.verify()
    record = bundle.add(name, data, origin, RIGHTS)
    return {**record, "file": PREFIX + name, "width": width, "height": height, "alt": alt}


def render_surface(vertices, faces, poster):
    lower, upper = np.asarray(poster["bounds_xyz_mm"], dtype=np.float64)
    figure = plt.figure(figsize=(7, 5.5), dpi=100, facecolor="#eeebe3")
    try:
        axes = figure.add_axes([0, 0, 1, 1], projection="3d", facecolor="#eeebe3")
        axes.add_collection3d(
            Poly3DCollection(
                vertices[faces],
                facecolors="#ffffff",
                linewidth=0,
                shade=True,
                lightsource=LightSource(315, 45),
                antialiased=False,
            )
        )
        axes.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
        axes.set_box_aspect(upper - lower, zoom=1.1)
        axes.set_proj_type("ortho")
        axes.view_init(elev=poster["elevation_degrees"], azim=poster["azimuth_degrees"], roll=0)
        axes.set_axis_off()
        stream = io.BytesIO()
        figure.savefig(stream, format="png", dpi=100, facecolor="#eeebe3")
        return stream.getvalue()
    finally:
        plt.close(figure)


def curate(bundle, helpers, spectral_root):
    anatomy_dir = spectral_root / "anatomy-vertebrae-v1"
    run_dir = spectral_root / "reconstruction-vertebrae-v1"
    figures_dir = spectral_root / "figures-vertebrae-v1"
    run = bundle.record("completed-run", run_dir / "run.json", RUN_SHA)
    figures = bundle.record("original-figure-receipt", figures_dir / "figures.json", FIGURES_SHA)
    anatomy = bundle.record("anatomy-receipt", anatomy_dir / "anatomy.json", ANATOMY_SHA)
    bundle.record(
        "source-rights-receipt",
        anatomy_dir / "source-license-manifest.json",
        anatomy["implementation"]["source_license_manifest_sha256"],
    )
    bundle.read("original-renderer", figures_dir / "render.py", figures["renderer_sha256"])
    acquired = bundle.record(
        "acquisition-receipt",
        spectral_root / "acquisition-vertebrae-v1/run.json",
        run["configuration"]["acquisition_sha256"],
    )
    solver = bundle.record(
        "solver-report", run_dir / "solver-report.json", run["output_sha256"]["solver-report.json"]
    )
    bundle.read(
        "recovered-fractions", run_dir / "fractions.npy", run["output_sha256"]["fractions.npy"]
    )
    if (
        run["status"] != "complete"
        or acquired["status"] != "complete"
        or not run["sources_unchanged"]
        or not run["recorded_files_unchanged"]
        or figures["run_sha256"] != RUN_SHA
        or figures["anatomy_sha256"] != ANATOMY_SHA
        or figures["acquisition_sha256"] != bundle.inputs["acquisition-receipt"]
        or solver["accepted_steps"] != 1000
        or solver["termination"] != "iteration_budget"
        or figures["surface_fraction"] != 0.325
        or 0 not in run["configuration"]["fit_views"]
    ):
        raise ValueError("The pinned completed-study identity or display protocol differs")

    public_root = ROOT / "public/generated/reconstruction"
    public = bundle.record("accepted-public-study", public_root / "study.json", PUBLIC_STUDY_SHA)
    case = next(case for case in public["cases"] if case["id"] == "vertebrae")
    grid = helpers.grid_record(anatomy["forward_grid"])
    inverse = helpers.grid_record(anatomy["inverse_grid"])
    if inverse != case["grid"] or inverse != figures["grid"]:
        raise ValueError("Original and current display grids differ")
    ct = bundle.array(
        "native-ct-context",
        anatomy_dir / "evaluation-native-ct-hu.npy",
        anatomy["outputs"]["evaluation-native-ct-hu.npy"]["sha256"],
    )
    if ct.shape != (192, 192, 192) or grid["shape"] != list(ct.shape):
        raise ValueError("Unexpected native CT geometry")
    # The archived renderer chooses the nearest native plane to its inverse-grid
    # coronal plane. Here these centres coincide exactly in binary64 millimetres.
    inverse_index = figures["material_slice_indices"]["coronal"]
    coordinate = inverse["origin_mm"][1] + inverse_index * inverse["spacing_mm"][1]
    native_index = int(np.rint((coordinate - grid["origin_mm"][1]) / grid["spacing_mm"][1]))
    if (
        native_index != figures["ct_context"]["indices_zyx"][1]
        or native_index != 103
        or coordinate != 5.8154296875
        or grid["origin_mm"][1] + native_index * grid["spacing_mm"][1] != coordinate
        or figures["ct_context"]["window_hu"] != [-300, 1300]
    ):
        raise ValueError("Original coronal CT plane or display window changed")
    spacing = np.asarray(grid["spacing_mm"])
    lower = np.asarray(grid["origin_mm"]) - spacing / 2
    upper = lower + np.asarray(grid["shape"])[::-1] * spacing
    extent = [float(lower[0]), float(upper[0]), float(lower[2]), float(upper[2])]
    plane = ct[:, native_index, :]
    stream = io.BytesIO()
    plt.imsave(stream, plane, format="png", origin="lower", cmap="gray", vmin=-300, vmax=1300)
    ct_image = image_record(
        bundle,
        "ct.png",
        stream.getvalue(),
        "Native acquired coronal CT crop at y=5.81543 mm; x increases right and z increases up.",
        "Exact native CT coronal samples, original recorded HU window, lower origin; no resampling",
    )
    mesh = next(mesh for mesh in case["meshes"] if mesh["id"] == "bone-reconstructed")
    mesh_name = "reconstructed-bone-surface.npz"
    with np.load(
        io.BytesIO(
            bundle.read(
                "recorded-recovered-mesh", figures_dir / mesh_name, figures["outputs"][mesh_name]
            )
        ),
        allow_pickle=False,
    ) as archive:
        vertices, faces = archive["vertices_xyz_mm"], archive["triangles"]
    if (
        vertices.shape != (16109, 3)
        or vertices.dtype != np.float64
        or faces.shape != (31788, 3)
        or not np.issubdtype(faces.dtype, np.integer)
        or not np.isfinite(vertices).all()
        or faces.min() < 0
        or faces.max() >= len(vertices)
        or mesh["threshold"] != 0.325
    ):
        raise ValueError("Invalid or changed recorded recovered mesh")
    for name, values, dtype in (("positions", vertices, "<f8"), ("indices", faces, "<u4")):
        metadata = mesh[name]
        data = bundle.read(
            "accepted-mesh-" + name, public_root / metadata["file"], metadata["sha256"]
        )
        if (
            len(data) != metadata["bytes"]
            or np.ascontiguousarray(values, dtype=dtype).tobytes() != data
        ):
            raise ValueError("Accepted public mesh differs from the original recorded triangles")
    poster = case["acquisition"]["display"]["poster"]
    if (
        poster["bounds_xyz_mm"] != [lower.tolist(), upper.tolist()]
        or poster["projection"] != "orthographic"
        or poster["elevation_degrees"] != 12
        or poster["azimuth_degrees"] != -75
        or poster["roll_degrees"] != 0
        or poster["background"] != "#eeebe3"
        or poster["material_colours"] != ["#ffffff"] * 2
        or poster["light_azimuth_elevation_degrees"] != [315, 45]
        or any(
            poster[key]
            for key in ("raster_antialiasing", "geometry_smoothing", "decimation", "capping")
        )
    ):
        raise ValueError("Accepted paper poster rendering protocol changed")
    surface = image_record(
        bundle,
        "reconstruction.png",
        render_surface(vertices, faces, poster),
        "Recovered vertebral-region bone surface, including cropped ribs and open boundaries.",
        "New white-on-paper rendering of exact recorded recovered triangles at the accepted camera",
    )
    radiograph_root = ROOT / "public/generated/reconstruction-radiographs"
    radio = bundle.record(
        "accepted-radiograph-study", radiograph_root / "study.json", RADIOGRAPH_STUDY_SHA
    )
    radio_case = next(case for case in radio["cases"] if case["id"] == "vertebrae")
    frame = next(frame for frame in radio_case["input_views"] if frame["view_zero_based"] == 0)
    data = bundle.read("existing-radiograph", radiograph_root / frame["file"], RADIOGRAPH_SHA)
    if (
        frame["sha256"] != RADIOGRAPH_SHA
        or len(data) != frame["bytes"]
        or frame["width"] != 96
        or frame["height"] != 96
        or frame["limits"] != [0, 8]
        or frame["channel"] != "20–55 keV"
        or frame["masked_pixels"] != 0
        or frame["file"] != "vertebrae/input-view-0.png"
    ):
        raise ValueError("Existing fitting radiograph or display identity changed")
    radiograph = {key: frame[key] for key in ("sha256", "bytes", "width", "height")} | {
        "file": "generated/reconstruction-radiographs/" + frame["file"],
        "alt": "Simulated fitting radiograph 1 of the assigned phantom, 20–55 keV channel.",
    }
    source = {
        "title": "Rister et al., CT-ORG (2019), case 2 acquired CT and bone labels",
        "url": "https://doi.org/10.7937/tcia.2019.tt7f4v7o",
        "licence": RIGHTS,
        "source_sha256": anatomy["source_files"]["volume-2.nii.gz"]["sha256"],
        "label_source_sha256": anatomy["source_files"]["labels-2.nii.gz"]["sha256"],
    }
    study = {
        "schema_version": 1,
        "id": "vertebrae-fixed-pose-introduction",
        "panels": [
            {
                "id": "ct",
                "label": "Acquired CT",
                "image": ct_image,
                "aspect": (extent[1] - extent[0]) / (extent[3] - extent[2]),
                "note": (
                    "Coronal plane, y=5.81543 mm · −300 to 1,300 HU. "
                    "x increases right; z increases up."
                ),
            },
            {
                "id": "radiograph",
                "label": "Simulated radiograph",
                "image": radiograph,
                "aspect": 1,
                "note": "Fitting view 1 · 20–55 keV · negative-log display 0–8.",
            },
            {
                "id": "reconstruction",
                "label": "Recovered bone",
                "image": surface,
                "aspect": 700 / 550,
                "note": "Recorded surface at fraction 0.325 · 1,000 accepted updates.",
            },
        ],
        "ct_display": {
            "plane": "coronal",
            "native_axis_zyx": 1,
            "native_index": native_index,
            "coordinate_y_mm": coordinate,
            "window_hu": [-300, 1300],
            "extent_xz_mm": extent,
            "sample_spacing_xz_mm": [float(spacing[0]), float(spacing[2])],
            "array_shape_zx": list(plane.shape),
            "origin": "lower",
            "png_rows": "PNG top row is native z=191; columns retain native increasing x",
            "orientation": (
                "Centred object XYZ; lossless LAS-to-RAS X reversal from the source header"
            ),
            "laterality": "Header-derived; patient laterality not independently established",
            "css": (
                "Render at the declared physical aspect, filling both dimensions; native "
                "sample array is square"
            ),
            "resampling": (
                "none; native samples mapped once through the fixed grayscale display window"
            ),
        },
        "radiograph_display": {
            "view_zero_based": 0,
            "role": "fitting",
            "channel_kev": [20, 55],
            "formula": "-log((counts+0.5)/open_beam)",
            "window": [0, 8],
            "pseudocount": "0.5 for display only",
            "detector_extent_uv_mm": [-192, 192, -192, 192],
            "origin": "lower; already encoded in the reused PNG",
            "copied": False,
        },
        "surface_display": {
            **{
                key: poster[key]
                for key in (
                    "projection",
                    "elevation_degrees",
                    "azimuth_degrees",
                    "roll_degrees",
                    "bounds_xyz_mm",
                    "background",
                    "light_azimuth_elevation_degrees",
                    "rendering",
                    "raster_antialiasing",
                    "geometry_smoothing",
                    "decimation",
                    "capping",
                )
            },
            "material_colour": "#ffffff",
            "box_zoom": 1.1,
            "threshold_fraction": 0.325,
            "vertices": len(vertices),
            "triangles": len(faces),
            "mesh_sha256": bundle.inputs["recorded-recovered-mesh"],
        },
        "scientific_scope": {
            "source_case": "CT-ORG case 2",
            "material_assignment": (
                "Bone=0.65 on source label 5; water=1-bone throughout the finite box"
            ),
            "geometry": (
                "Fixed recorded source/detector poses; no fitted registration supplies this "
                "reconstruction"
            ),
            "physics": (
                "Historical NIST SRD126 water and ICRU-44 cortical bone; SpekPy 2.5.4; ideal "
                "three-channel counts"
            ),
            "physics_npz_sha256": figures["physics_sha256"],
            "physics_metadata_sha256": figures["physics_metadata_sha256"],
            "accepted_steps": solver["accepted_steps"],
            "termination": solver["termination"],
            "converged": False,
            "completed_run_sha256": RUN_SHA,
        },
        "qualification": [
            (
                "The CT is acquired; the radiograph is simulated from an assigned "
                "bone-in-water phantom derived from its labels."
            ),
            (
                "Assigned and recovered fractions are not measured patient composition; the "
                "CT HU values are display context only."
            ),
            (
                "Fixed poses were supplied. This sequence does not contain a recorded "
                "registration-to-reconstruction step."
            ),
            (
                "The historical 1,000-update run reached its iteration budget; convergence "
                "and noise-limited recovery are not established."
            ),
            (
                "The vertebral region includes cropped ribs. Recorded surfaces remain "
                "unsmoothed and open at cropped boundaries."
            ),
            (
                "NIST source and derived attenuation tables are excluded; this display is not"
                " a reproduction using the later xraylib model."
            ),
        ],
        "provenance": [source, *case["provenance"][1:]],
    }
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    output, receipt_path = args.output.resolve(), args.receipt.resolve()
    if output.exists() or receipt_path.exists() or receipt_path.is_relative_to(output):
        raise ValueError("Use a new output directory and a separate new private receipt")
    if output.is_relative_to(ROOT) or receipt_path.is_relative_to(ROOT):
        raise ValueError("This bounded curator stages outside the repository only")
    helpers = helper_module()
    bundle = helpers.Curation()
    bundle.read("curation-helper", ROOT / "tools/public/curate-reconstruction.py", HELPER_SHA)
    study = curate(bundle, helpers, args.spectral_root.resolve())
    bundle.add(
        "study.json",
        helpers.json_bytes(study),
        "Positive allowlist of recorded study/display metadata",
        RIGHTS,
    )
    generator_sha = helpers.sha(Path(__file__).read_bytes())
    provenance = {
        "schema_version": 1,
        "generator": "tools/public/curate-introduction-study.py",
        "generator_sha256": generator_sha,
        "helper_sha256": HELPER_SHA,
        "rights_record_precedes_payloads": True,
        "complete_file_inventory": [
            "ct.png",
            "reconstruction.png",
            "study.json",
            "provenance.json",
        ],
        "rights": {
            "licence": RIGHTS,
            "url": "https://creativecommons.org/licenses/by/3.0/",
            "sources": study["provenance"],
        },
        "scope": (
            "Two rendered PNGs and selected display metadata; no raw CT, masks, fields, "
            "meshes, attenuation tables or run records"
        ),
        "changes": (
            "Native CT grayscale mapping and new exact-triangle raster rendering; "
            "existing radiograph referenced without copying"
        ),
        "inputs": bundle.inputs,
        "external_references": [study["panels"][1]["image"]],
        "rendering_versions": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "Pillow": pillow_version,
        },
        "allowlist": [
            {"file": name, "sha256": helpers.sha(data), "bytes": len(data), **bundle.origins[name]}
            for name, data in sorted(bundle.payloads.items())
        ],
    }
    # Rights and the full named inventory are the first persistent output. All
    # rendered payloads already exist in memory; existing destinations are refused.
    output.mkdir(parents=True)
    with (output / "provenance.json").open("xb") as stream:
        stream.write(helpers.json_bytes(provenance))
    for name, data in sorted(bundle.payloads.items()):
        with (output / name).open("xb") as stream:
            stream.write(data)
        if (output / name).read_bytes() != data:
            raise ValueError("Written bytes differ from curated memory payload")
    actual = sorted(path.name for path in output.iterdir())
    if actual != sorted(provenance["complete_file_inventory"]):
        raise ValueError("Unexpected staged output")
    receipt = {
        "output": str(output),
        "inputs": bundle.private_inputs,
        "files": {name: helpers.sha((output / name).read_bytes()) for name in actual},
        "generator_sha256": generator_sha,
        "only_allowlisted_files": True,
        "all_written_bytes_verified": True,
        "command": [
            str(Path(__file__).resolve()),
            "--spectral-root",
            str(args.spectral_root.resolve()),
            "--output",
            str(output),
            "--receipt",
            str(receipt_path),
        ],
        "rendering_versions": provenance["rendering_versions"],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "files": len(actual),
                "study_sha256": receipt["files"]["study.json"],
            }
        )
    )


if __name__ == "__main__":
    main()
