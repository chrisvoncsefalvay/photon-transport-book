"""Curate completed reconstruction displays by an explicit, hash-verified allowlist.

No reconstruction, medical-image resampling or attenuation-table export occurs.
The destination must be new. Rights and the complete asset allowlist are written
before numerical or image payloads. Private source locations go only to --receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
CT_RUN = "fab471b81eb132b798ef62cd958a02e35f2a8a06d85d52d808bb2bc682f4e869"
HAP_RUN = "f83e1c19577a22b6b8fb6a7cbcd5edf0f84878f5c852ca205dc2e74699108b64"
CT_RIGHTS = "CC BY 3.0; derived from Rister et al., CT-ORG, TCIA"
HAP_RIGHTS = "CC BY 4.0; derived from Zhou et al., calibration phantom release"


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value) -> bytes:
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if re.search(rb"/(?:home|fshare|tmp|Users|root)/|file://|tailscale|100\.105\.", data):
        raise ValueError("Private location or infrastructure found in curated metadata")
    return data


def grid_record(value):
    grid = {key: value[key] for key in ("shape", "spacing_mm", "origin_mm", "orientation")}
    if (
        len(grid["shape"]) != 3
        or any(type(n) is not int or n < 2 for n in grid["shape"])
        or np.shape(grid["spacing_mm"]) != (3,)
        or np.shape(grid["origin_mm"]) != (3,)
        or not np.isfinite(grid["spacing_mm"]).all()
        or not np.isfinite(grid["origin_mm"]).all()
        or min(grid["spacing_mm"]) <= 0
        or grid["orientation"] != IDENTITY
    ):
        raise ValueError("Expected a recorded finite axis-aligned XYZ grid with ZYX storage")
    return grid


class Curation:
    def __init__(self):
        self.payloads = {}
        self.origins = {}
        self.inputs = {}
        self.private_inputs = {}
        self.mesh_arrays = {}

    def read(self, tag, path, expected=None):
        data = path.read_bytes()
        actual = sha(data)
        if expected is not None and actual != expected:
            raise ValueError(f"Changed completed source: {tag}")
        self.inputs[tag] = actual
        self.private_inputs[tag] = {"path": str(path.resolve()), "sha256": actual}
        return data

    def record(self, tag, path, expected=None):
        return json.loads(self.read(tag, path, expected))

    def array(self, tag, path, expected):
        value = np.load(io.BytesIO(self.read(tag, path, expected)), allow_pickle=False)
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Expected unchanged finite FP32 scalar source: {tag}")
        return value

    def add(self, name, value, origin, rights):
        if name in self.payloads or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Duplicate or unsafe public asset name")
        self.payloads[name] = value
        self.origins[name] = {"origin": origin, "licence": rights}
        return {"file": name, "sha256": sha(value), "bytes": len(value)}

    def scalar(self, case, material, role, label, values, window, origin, rights):
        if values.shape != tuple(case["grid"]["shape"]) or values.dtype != np.float32:
            raise ValueError("Scalar and declared physical grid disagree")
        identifier = f"{material}-{role}"
        record = self.add(
            f"{case['id']}/{identifier}.f32",
            np.ascontiguousarray(values, dtype="<f4").tobytes(),
            origin,
            rights,
        )
        return {
            "id": identifier,
            "label": label,
            "role": role,
            "material": material,
            "units": "dimensionless fraction"
            if case["id"] == "vertebrae"
            else "dimensionless equivalent coefficient",
            **record,
            "dtype": "float32-le",
            "window": window,
        }

    def mesh(self, case, identifier, label, scalar_id, threshold, source, expected, rights):
        tag = f"{case['id']}/recorded-mesh/{identifier}"
        with np.load(io.BytesIO(self.read(tag, source, expected)), allow_pickle=False) as archive:
            vertices, triangles = archive["vertices_xyz_mm"], archive["triangles"]
            if "threshold" in archive and float(archive["threshold"]) != threshold:
                raise ValueError("Recorded mesh threshold differs")
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not np.isfinite(vertices).all()
            or triangles.ndim != 2
            or triangles.shape[1] != 3
            or not np.issubdtype(triangles.dtype, np.integer)
            or triangles.min() < 0
            or triangles.max() >= len(vertices)
        ):
            raise ValueError("Invalid recorded mesh")
        # Conversion is exact: these sources already contain binary64 coordinates
        # and nonnegative indices representable by uint32. Do not reorder faces.
        positions = np.ascontiguousarray(vertices, dtype="<f8")
        indices = np.ascontiguousarray(triangles, dtype="<u4")
        if not np.array_equal(vertices, positions) or not np.array_equal(triangles, indices):
            raise ValueError("Public mesh conversion would change numerical values")
        self.mesh_arrays[(case["id"], identifier)] = (positions, indices)
        return {
            "id": identifier,
            "label": label,
            "scalar_id": scalar_id,
            "threshold": threshold,
            "positions": {
                **self.add(
                    f"{case['id']}/{identifier}-positions.f64", positions.tobytes(), tag, rights
                ),
                "dtype": "float64-le",
            },
            "indices": {
                **self.add(
                    f"{case['id']}/{identifier}-indices.u32", indices.tobytes(), tag, rights
                ),
                "dtype": "uint32-le",
            },
        }

    def image(self, case, identifier, label, alt, data, origin, rights):
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image.verify()
        return {
            "id": identifier,
            "label": label,
            **self.add(f"{case['id']}/{identifier}.png", data, origin, rights),
            "width": width,
            "height": height,
            "alt": alt,
        }

    def poster(self, case, colours, rights):
        grid = case["grid"]
        spacing = np.asarray(grid["spacing_mm"])
        lower = np.asarray(grid["origin_mm"]) - spacing / 2
        upper = lower + np.asarray(grid["shape"])[::-1] * spacing
        display = case["acquisition"]["display"]
        direction = np.asarray(display["camera_direction"], dtype=np.float64)
        azimuth = float(np.degrees(np.arctan2(direction[1], direction[0])))
        elevation = float(np.degrees(np.arctan2(direction[2], np.linalg.norm(direction[:2]))))
        fig = plt.figure(figsize=(14, 5.5), dpi=100, facecolor="#eeebe3")
        for index, (mesh, colour) in enumerate(zip(case["meshes"], colours, strict=True)):
            ax = fig.add_axes([index / 2, 0, 0.5, 1], projection="3d", facecolor="#eeebe3")
            vertices, faces = self.mesh_arrays[(case["id"], mesh["id"])]
            collection = Poly3DCollection(
                vertices[faces],
                facecolors=colour,
                linewidth=0,
                shade=True,
                lightsource=LightSource(315, 45),
                antialiased=False,
            )
            ax.add_collection3d(collection)
            ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
            ax.set_box_aspect(upper - lower, zoom=1.1)
            ax.set_proj_type("ortho")
            ax.view_init(elev=elevation, azim=azimuth, roll=0)
            ax.set_axis_off()
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=100, facecolor="#eeebe3")
        plt.close(fig)
        display["poster"] = {
            "projection": "orthographic",
            "elevation_degrees": elevation,
            "azimuth_degrees": azimuth,
            "roll_degrees": 0,
            "bounds_xyz_mm": [lower.tolist(), upper.tolist()],
            "panel_shape_pixels": [550, 700],
            "background": "#eeebe3",
            "material_colours": colours,
            "light_azimuth_elevation_degrees": [315, 45],
            "rendering": "Recorded triangles with flat face lighting; no raster triangle seams",
            "raster_antialiasing": False,
            "geometry_smoothing": False,
            "decimation": False,
            "capping": False,
        }
        return self.image(
            case,
            "poster",
            "Recorded material surfaces",
            (
                "Two material surfaces at the same physical scale and camera; original "
                "open boundaries."
            ),
            stream.getvalue(),
            "New raster rendering of the case's exact exported mesh arrays",
            rights,
        )


def vertebrae(bundle, root):
    run_dir, anatomy_dir = root / "reconstruction-vertebrae-v1", root / "anatomy-vertebrae-v1"
    figures_dir = root / "figures-vertebrae-v1"
    run = bundle.record("vertebrae/completed-run", run_dir / "run.json", CT_RUN)
    figures = bundle.record(
        "vertebrae/figure-receipt",
        figures_dir / "figures.json",
        "e86f08ce4d6e38c72842fe7845a757f15a71645c4ce6dee97f24f7812a630bc8",
    )
    if run["status"] != "complete" or figures["run_sha256"] != CT_RUN:
        raise ValueError("CT reconstruction and figure identities differ")
    anatomy = bundle.record(
        "vertebrae/anatomy-receipt", anatomy_dir / "anatomy.json", figures["anatomy_sha256"]
    )
    reference = bundle.array(
        "vertebrae/assigned-reference",
        anatomy_dir / "evaluation-fractions.npy",
        anatomy["outputs"]["evaluation-fractions.npy"]["sha256"],
    )
    recovered = bundle.array(
        "vertebrae/recovered-fractions",
        run_dir / "fractions.npy",
        run["output_sha256"]["fractions.npy"],
    )
    evaluation = bundle.record(
        "vertebrae/evaluation", run_dir / "evaluation.json", run["output_sha256"]["evaluation.json"]
    )
    solver = bundle.record(
        "vertebrae/solver-report",
        run_dir / "solver-report.json",
        run["output_sha256"]["solver-report.json"],
    )
    if reference.shape != (2, 64, 64, 64) or recovered.shape != reference.shape:
        raise ValueError("Unexpected CT material-volume dimensions")
    for values in (reference, recovered):
        if values.min() < 0 or values.max() > 1 or np.max(values.sum(axis=0)) > 1 + 2e-7:
            raise ValueError("CT fraction source violates its recorded simplex")
    if solver["accepted_steps"] != 1000 or figures["surface_fraction"] != 0.325:
        raise ValueError("CT solver or display protocol changed")
    grid = grid_record(anatomy["inverse_grid"])
    if grid != figures["grid"] or grid != evaluation["grid"]:
        raise ValueError("Reference, reconstructed and figure grids differ")
    # Match the frozen evaluator: subtract recorded FP32 fields, then reduce in FP64.
    rmse = np.sqrt(np.mean((recovered - reference).astype(np.float64) ** 2, axis=(1, 2, 3)))
    if not np.allclose(rmse, evaluation["material_rmse"], rtol=0, atol=1e-12):
        raise ValueError("Recomputed matched-array errors differ from recorded evaluation")
    a, b = reference[1] > 0.325, recovered[1] > 0.325
    dice = float(2 * np.count_nonzero(a & b) / (np.count_nonzero(a) + np.count_nonzero(b)))
    azimuth, elevation = np.radians([-75, 12])
    case = {
        "id": "vertebrae",
        "title": "CT-derived vertebral-region reconstruction",
        "kind": "ct-derived-simulation",
        "description": (
            "Assigned bone and water reconstructed from simulated spectral "
            "radiographs of CT-derived anatomy."
        ),
        "qualification": [
            "Assigned material phantom; fractions are not measured patient composition.",
            (
                "1,000-update budget reached; stationarity and noise-limited "
                "performance are not established."
            ),
            (
                "Vertebral region includes cropped ribs; source-header laterality is "
                "not independently established."
            ),
            (
                "Reference and recovered arrays share the inverse grid; surfaces "
                "remain unsmoothed and open at cropped boundaries."
            ),
        ],
        "grid": grid,
        "scalars": [],
        "meshes": [],
        "images": [],
        "metrics": {
            "water_rmse": float(rmse[0]),
            "bone_rmse": float(rmse[1]),
            "bone_dice": dice,
            "accepted_steps": solver["accepted_steps"],
            "termination": solver["termination"],
            "converged": False,
        },
        "acquisition": {
            "observation_kind": (
                "Simulated independent Poisson photon counts; ideal three-channel response"
            ),
            "fitting_views": 126,
            "total_views": 144,
            "angular_span_degrees": 360,
            "incident_count_protocol": (
                "200,000 photons per ray integrated across three ideal channels"
            ),
            "heldout_views": 18,
            "detector_shape": [96, 96],
            "forward_grid_shape": [192, 192, 192],
            "inverse_grid_shape": [64, 64, 64],
            "tube_potential_kvp": 120,
            "energy_channel_edges_kev": [20, 55, 80, 120],
            "incident_photons_per_ray": 200000,
            "source_generator": "SpekPy 2.5.4",
            "material_assignment": (
                "Bone fraction 0.65 on source bone labels, water=1-bone throughout the known box"
            ),
            "reference_resampling": (
                "Recorded aligned 3 x 3 x 3 native-cell volume means; no further "
                "resampling for display"
            ),
            "display": {
                "slice_indices_xyz": [32, 34, 30],
                "camera_direction": [
                    float(np.cos(elevation) * np.cos(azimuth)),
                    float(np.cos(elevation) * np.sin(azimuth)),
                    float(np.sin(elevation)),
                ],
                "surface_threshold": 0.325,
                "radiograph": figures["radiograph_display"],
            },
        },
        "provenance": [
            {
                "title": (
                    "Rister et al. CT-ORG (2019), case 2 bone labels; cropped, assigned "
                    "and volume averaged"
                ),
                "url": "https://doi.org/10.7937/tcia.2019.tt7f4v7o",
                "licence": "CC BY 3.0",
                "source_sha256": anatomy["source_files"]["labels-2.nii.gz"]["sha256"],
            },
            {
                "title": "SpekPy 2.5.4 spectrum generator",
                "url": "https://pypi.org/project/spekpy/2.5.4/",
                "licence": "MIT; software not bundled in this data package",
                "source_sha256": "2f01e805721281b5a3e332500db577cedaf71bde0f0d3838745b915e6f1c258a",
            },
            {
                "title": (
                    "Hubbell and Seltzer, NIST SRD126; water coefficient source used for simulation"
                ),
                "url": "https://doi.org/10.18434/T4D01F",
                "licence": "SRD copyright reserved; source and derived attenuation tables excluded",
                "source_sha256": "19a80f2609b6aa4a3ff057d57ad1e17001eeaa5932553c15c66522e673c3f76f",
            },
            {
                "title": "NIST SRD126 ICRU-44 cortical bone coefficient source used for simulation",
                "url": "https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/bone.html",
                "licence": "SRD copyright reserved; source and derived attenuation tables excluded",
                "source_sha256": "1df054fa5e8eef770a2fbcd8fd6a28f68b6c1f426d65cfa41d55ebf68041144a",
            },
        ],
    }
    for material, index, window in (("water", 0, [0.0, 1.0]), ("bone", 1, [0.0, 0.65])):
        for role, array in (("reference", reference), ("reconstructed", recovered)):
            case["scalars"].append(
                bundle.scalar(
                    case,
                    material,
                    role,
                    f"{material.title()} {role}",
                    array[index],
                    window,
                    f"vertebrae/{role} material array",
                    CT_RIGHTS,
                )
            )
    for role, name in (
        ("reference", "reference-bone-surface.npz"),
        ("reconstructed", "reconstructed-bone-surface.npz"),
    ):
        case["meshes"].append(
            bundle.mesh(
                case,
                f"bone-{role}",
                f"Bone {role}",
                f"bone-{role}",
                0.325,
                figures_dir / name,
                figures["outputs"][name],
                CT_RIGHTS,
            )
        )
    case["images"].append(bundle.poster(case, ["#ffffff", "#ffffff"], CT_RIGHTS))
    for identifier, name, label in (
        ("surfaces", "bone-volumes.png", "Recorded bone surfaces"),
        ("sections", "bone-sections.png", "Matched bone sections and errors"),
        (
            "radiographs",
            "heldout-radiographs.png",
            "Simulated spectral radiographs and held-out errors",
        ),
    ):
        data = bundle.read(
            f"vertebrae/plate/{identifier}", figures_dir / name, figures["outputs"][name]
        )
        case["images"].append(
            bundle.image(
                case,
                identifier,
                label,
                label,
                data,
                f"Exact recorded {name}; display mappings preserved",
                CT_RIGHTS,
            )
        )
    return case


def hap(bundle, root):
    runs = root / "root"
    run_dir, figures_dir = runs / "refinement-recovery-v1", runs / "figures-refinement-v1"
    run = bundle.record("hap/completed-run", run_dir / "run.json", HAP_RUN)
    figures = bundle.record(
        "hap/figure-receipt",
        figures_dir / "figures.json",
        "f9f491f50624057592ac0b7e460087fbe03916d8d9b721e40411ba73aac37a60",
    )
    if run["status"] != "complete" or figures["run_sha256"] != HAP_RUN:
        raise ValueError("Acquired reconstruction and figure identities differ")
    fields = bundle.array("hap/reconstructed-fields", run_dir / "fields.npy", run["fields_sha256"])
    solver = bundle.record(
        "hap/solver-report", run_dir / "solver-result.json", run["solver_result_sha256"]
    )
    assets = figures_dir / "material-viewer-v2-assets"
    viewer = bundle.record(
        "hap/accepted-viewer-receipt",
        assets / "build-receipt.json",
        "5dcebdac36815bd2f4926c1e9ac18bac444aea1e1fdf88d610c99cccfdabd788",
    )
    if viewer["provenance"]["fieldsSha256"] != run["fields_sha256"]:
        raise ValueError("Acquired mesh and scalar identities differ")
    evaluation_dir = runs / "evaluation-refinement-v1"
    evaluated = bundle.record(
        "hap/evaluation-receipt",
        evaluation_dir / "run.json",
        "dd735da9f4fdcad9e2a2ecb6586e55e70f09324c7634dc8f81bbedf3facdbec0",
    )
    if evaluated["status"] != "complete":
        raise ValueError("Acquired independent evaluation is incomplete")
    evaluation_inputs = evaluated["configuration"]["input_sha256"]
    if (
        evaluation_inputs[str((run_dir / "fields.npy").resolve())] != run["fields_sha256"]
        or evaluation_inputs[str((run_dir / "run.json").resolve())] != HAP_RUN
    ):
        raise ValueError("Acquired evaluation was not computed from these completed fields")
    metrics = bundle.record(
        "hap/evaluation",
        evaluation_dir / "metrics.json",
        evaluated["output_sha256"]["metrics.json"],
    )
    grid = grid_record(run["grid"])
    if fields.shape != (2, *grid["shape"]):
        raise ValueError("Acquired scalar array shape disagrees with recorded grid")
    if fields.min() < 0 or solver["accepted_steps"] != 342:
        raise ValueError("Acquired nonnegativity or retained update count changed")
    if solver["cumulative_executed_accepted_steps"] != 344:
        raise ValueError("Acquired recovery accounting changed")
    z = grid["origin_mm"][2] + np.arange(grid["shape"][0]) * grid["spacing_mm"][2]
    selected = np.flatnonzero((z >= -4) & (z <= 4))
    if not np.array_equal(selected, np.arange(16, 32)):
        raise ValueError("Expected the exact completed central sixteen-cell slab")
    crop = fields[:, 16:32]
    grid["shape"] = [16, *grid["shape"][1:]]
    grid["origin_mm"] = [*grid["origin_mm"][:2], float(z[16])]
    primary = metrics["heldout_primary_unpreviewed"]
    case = {
        "id": "hap",
        "title": "Acquired phantom material reconstruction",
        "kind": "acquired-diagnostic",
        "description": (
            "PMMA- and aluminium-equivalent fields fitted to measured "
            "photon-counting CT intensities."
        ),
        "qualification": [
            (
                "Response calibration v1 fails its frozen gate; absolute material "
                "quantification is unaccepted."
            ),
            (
                "Equivalent coefficients are not HAP concentrations; the rod carrier "
                "composition is undocumented."
            ),
            (
                "342 retained updates, 344 executed including two lost before "
                "checkpoint recovery; time budget reached without convergence."
            ),
            (
                "Only the central z = -4 to +4 mm slab is shown; surfaces remain open "
                "and do not recover physical rod ends."
            ),
            (
                "Weak separation and speckle remain in the recorded fields; display "
                "levels do not classify material."
            ),
        ],
        "grid": grid,
        "scalars": [],
        "meshes": [],
        "images": [],
        "metrics": {
            "accepted_steps": solver["accepted_steps"],
            "executed_updates": solver["cumulative_executed_accepted_steps"],
            "termination": solver["termination"],
            "converged": False,
            "calibration_pass": False,
            "primary_heldout_views": metrics["primary_previously_unpreviewed_count"],
            "heldout_total_rms": primary["Total"]["normalised_transmission_residual"]["rms"],
            "heldout_high_rms": primary["High"]["normalised_transmission_residual"]["rms"],
            "quadrature_p95_gate_pass": metrics["quadrature_gate_pass"],
        },
        "acquisition": {
            "observation_kind": (
                "Recorded mean photon-counting intensities; deterministic WLS on "
                "cumulative Total/High channels"
            ),
            "fitting_views": len(run["fitting_view_ids"]),
            "total_views": 1440,
            "heldout_views": 120,
            "angular_span_degrees": 360,
            "incident_count_protocol": (
                "Recorded mean cumulative Total/High intensities; frozen air normalisation"
            ),
            "development_union_views": 144,
            "primary_heldout_views": 116,
            "previously_previewed_excluded_views": 4,
            "energy_thresholds_kev": [15, 30],
            "tube_potential_kvp": 80,
            "material_bases": (
                "Nominal PMMA 1.19 g/cm3 and Al 2.699 g/cm3; no fraction-sum constraint"
            ),
            "scalar_crop": "Exact source z indices 16:32; no resampling or intensity normalisation",
            "display": {
                "slice_indices_xyz": [64, 64, 7],
                "camera_direction": [0.8, 0.6, 0.9],
                "surface_thresholds": {"pmma": 0.5, "al": 0.02},
                "surface_crop_z_mm": [-4, 4],
                "radiograph_extent_mm": figures["detector_display_extent_mm"],
                "radiograph_origin": figures["detector_display_origin"],
            },
        },
        "provenance": [
            {
                "title": (
                    "Zhou et al. (2025), calibration phantom measured data; fixed masks, "
                    "native sample selection and project reconstruction"
                ),
                "url": "https://doi.org/10.5281/zenodo.17328375",
                "licence": "CC BY 4.0",
                "source_sha256": "ea6c12a7d624128eb182da6d11b51edbe9d514a02684f317ecfb9650e86a3c7e",
            },
            {
                "title": "xraylib 4.3.0 physical coefficient generator",
                "url": "https://pypi.org/project/xraylib/4.3.0/",
                "licence": "BSD-3-Clause; software not bundled in this data package",
                "source_sha256": "328bdccff6b7ea511757d6eb22ab7983de13b41e70369a0967e2c543c2bfed1e",
            },
        ],
    }
    for index, material, label in ((0, "pmma", "PMMA equivalent"), (1, "al", "Al equivalent")):
        case["scalars"].append(
            bundle.scalar(
                case,
                material,
                "reconstructed",
                label,
                crop[index],
                figures["display_limits"][index],
                "Exact central crop of hap/reconstructed-fields",
                HAP_RIGHTS,
            )
        )
    mesh_digests = viewer["assetSha256"]
    for material, label, threshold, name in (
        ("pmma", "PMMA equivalent", 0.5, "pmma-surface.npz"),
        ("al", "Al equivalent", 0.02, "al-surface-02.npz"),
    ):
        case["meshes"].append(
            bundle.mesh(
                case,
                f"{material}-reconstructed",
                label,
                f"{material}-reconstructed",
                threshold,
                assets / name,
                mesh_digests[name],
                HAP_RIGHTS,
            )
        )
    case["images"].append(bundle.poster(case, ["#b9cdd1", "#dccba7"], HAP_RIGHTS))
    for identifier, name, label in (
        ("sections", "material-slices.png", "Recorded equivalent-material sections"),
        (
            "radiographs",
            "acquired-radiographs.png",
            "Measured radiographs, fitted predictions and errors",
        ),
        ("calibration", "calibration.png", "Recorded response calibration failure"),
    ):
        data = bundle.read(f"hap/plate/{identifier}", figures_dir / name, figures["outputs"][name])
        case["images"].append(
            bundle.image(
                case,
                identifier,
                label,
                label,
                data,
                f"Exact recorded {name}; display mappings preserved",
                HAP_RIGHTS,
            )
        )
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-root", type=Path, required=True)
    parser.add_argument("--hap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "public/generated/reconstruction")
    parser.add_argument(
        "--receipt", type=Path, required=True, help="New private verification receipt"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or args.receipt.exists() or args.receipt.resolve().is_relative_to(output):
        raise ValueError("Use a new output directory and separate new private receipt")
    bundle = Curation()
    study = {
        "schema_version": 1,
        "cases": [vertebrae(bundle, args.spectral_root), hap(bundle, args.hap_root)],
    }
    bundle.add(
        "study.json",
        json_bytes(study),
        "Positive allowlist of verified scientific and display metadata",
        "Project metadata; underlying cases retain CC BY 3.0 and CC BY 4.0 attribution",
    )
    provenance = {
        "schema_version": 1,
        "generator": "tools/public/curate-reconstruction.py",
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "rights_record_precedes_payloads": True,
        "inputs": bundle.inputs,
        "scope": (
            "Selected computed material arrays and exact recorded meshes/plates; "
            "no raw patient CT, labels, attenuation tables, software packages or "
            "raw run records"
        ),
        "changes": (
            "Header-free little-endian numeric encoding; exact HAP central crop; "
            "newly rendered mesh posters; no scalar resampling or normalisation"
        ),
        "scalar_storage": (
            "C-order ZYX, voxel centres from grid origin; all original FP32 values retained"
        ),
        "mesh_storage": (
            "Original physical XYZ in mm, float64 coordinates and uint32 triangle indices"
        ),
        "rights": {
            "vertebrae": {
                "licence": CT_RIGHTS,
                "url": "https://creativecommons.org/licenses/by/3.0/",
            },
            "hap": {"licence": HAP_RIGHTS, "url": "https://creativecommons.org/licenses/by/4.0/"},
            "nist": (
                "Simulation input tables have no asserted redistribution clearance and "
                "are excluded. New numerical results are not labelled as source "
                "attenuation tables."
            ),
        },
        "rendering_versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "allowlist": [
            {"file": name, "sha256": sha(data), "bytes": len(data), **bundle.origins[name]}
            for name, data in sorted(bundle.payloads.items())
        ],
    }
    # These rights and the complete positive allowlist must exist before copying data.
    output.mkdir(parents=True)
    (output / "provenance.json").write_bytes(json_bytes(provenance))
    for name, data in sorted(bundle.payloads.items()):
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
        if destination.read_bytes() != data:
            raise ValueError("Written curated asset differs from its verified memory buffer")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual != set(bundle.payloads) | {"provenance.json"}:
        raise ValueError("Curated directory differs from the explicit asset allowlist")
    receipt = {
        "output": str(output),
        "inputs": bundle.private_inputs,
        "files": {name: sha((output / name).read_bytes()) for name in sorted(actual)},
        "all_written_bytes_verified": True,
        "only_allowlisted_files": True,
        "source_generator_sha256": sha(Path(__file__).read_bytes()),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "files": len(actual),
                "bytes": sum((output / name).stat().st_size for name in actual),
                "study_sha256": sha((output / "study.json").read_bytes()),
            }
        )
    )


if __name__ == "__main__":
    main()
