"""Render recorded material volumes and radiographs; never invent result images."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid_geometry(record):
    """Recorded ZYX shape with XYZ voxel-centre geometry and voxel-face support."""
    shape = np.asarray(record["shape"], dtype=int)
    spacing = np.asarray(record["spacing_mm"], dtype=float)
    origin = np.asarray(record["origin_mm"], dtype=float)
    orientation = np.asarray(record["orientation"], dtype=float).reshape(3, 3)
    if shape.shape != (3,) or np.any(shape <= 0) or np.any(spacing <= 0):
        raise ValueError("invalid recorded grid geometry")
    # Current intake reorients losslessly to axis-aligned RAS. An oblique grid
    # needs a separate plane-coordinate display, not silently incorrect axes.
    if not np.array_equal(orientation, np.eye(3)):
        raise ValueError("slice renderer requires the recorded axis-aligned RAS grid")
    lower = origin - spacing / 2
    upper = lower + shape[::-1] * spacing
    return shape, spacing, origin, orientation, lower, upper


def plane(volume, axis, index):
    return np.take(volume, index, axis=axis)


def plane_extent(lower, upper, axis):
    horizontal, vertical = ((0, 1), (0, 2), (1, 2))[axis]
    return (lower[horizontal], upper[horizontal], lower[vertical], upper[vertical])


def describe_plane(axis, index, origin, spacing):
    names = ("Axial", "Coronal", "Sagittal")
    direction = ("z", "y", "x")[axis]
    coordinate = origin[2 - axis] + index * spacing[2 - axis]
    return f"{names[axis]} · {direction} = {coordinate:g} mm"


def show_plane(ax, values, axis, lower, upper, *, difference=False):
    artist = ax.imshow(
        values,
        origin="lower",
        extent=plane_extent(lower, upper, axis),
        interpolation="nearest",
        aspect="equal",
        cmap="RdBu_r" if difference else "cividis",
        vmin=-0.5 if difference else 0,
        vmax=0.5 if difference else 1,
    )
    horizontal, vertical = (("x", "y"), ("x", "z"), ("y", "z"))[axis]
    ax.set(xlabel=f"{horizontal} (mm)", ylabel=f"{vertical} (mm)")
    ax.tick_params(labelsize=8)
    return artist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-comparison",
        action="store_true",
        help="include a deterministic comparison of hashed checkpoints from the completed run",
    )
    args = parser.parse_args()
    record = json.loads((args.run / "run.json").read_text())
    if record["status"] != "complete":
        raise ValueError("only completed records can supply these result figures")
    for name, expected in record["output_sha256"].items():
        if digest(args.run / name) != expected:
            raise ValueError(f"recorded output changed: {name}")
    acquisition = Path(record["configuration"]["acquisition"])
    if digest(acquisition / "run.json") != record["configuration"]["acquisition_sha256"]:
        raise ValueError("acquisition identity differs from the reconstruction record")
    source = json.loads((acquisition / "run.json").read_text())
    if source["status"] != "complete":
        raise ValueError("radiographs require a completed acquisition record")
    for name, expected_hash in source["output_sha256"].items():
        if digest(acquisition / name) != expected_hash:
            raise ValueError(f"acquisition output changed: {name}")
    anatomy = Path(source["configuration"]["anatomy"])
    frozen_anatomy = acquisition / "sources/inputs/anatomy.json"
    if digest(frozen_anatomy) != source["source_sha256"]["inputs/anatomy.json"]:
        raise ValueError("acquisition reference snapshot changed")
    if digest(anatomy / "anatomy.json") != digest(frozen_anatomy):
        raise ValueError("external anatomy differs from the frozen reference")
    anatomy_meta = json.loads((anatomy / "anatomy.json").read_text())
    physics_path = Path(source["configuration"]["physics"])
    for filename, field in [
        ("physics.npz", "physics_npz_sha256"),
        ("metadata.json", "physics_metadata_sha256"),
    ]:
        if digest(physics_path / filename) != source["configuration"][field]:
            raise ValueError("physics bytes or provenance changed")
    for name, entry in anatomy_meta["outputs"].items():
        if digest(anatomy / name) != entry["sha256"]:
            raise ValueError(f"anatomical reference changed: {name}")
    recovered = np.load(args.run / "fractions.npy", allow_pickle=False)
    reference = np.load(anatomy / "evaluation-fractions.npy", allow_pickle=False)
    shape, spacing, origin, orientation, lower, upper = grid_geometry(anatomy_meta["inverse_grid"])
    if source["configuration"]["inverse_grid"] != anatomy_meta["inverse_grid"]:
        raise ValueError("reconstruction and anatomical evaluation grid differ")
    if recovered.shape != (2, *shape) or reference.shape != recovered.shape:
        raise ValueError("material arrays differ from the recorded two-material grid")
    if not np.isfinite(recovered).all() or not np.isfinite(reference).all():
        raise ValueError("material figure inputs must be finite")
    indices = tuple(anatomy_meta.get("display_indices_zyx", (shape // 2).tolist()))
    if len(indices) != 3 or any(
        type(index) is not int or not 0 <= index < count
        for index, count in zip(indices, shape, strict=True)
    ):
        raise ValueError("display_indices_zyx must lie inside the inverse grid")
    study_title = anatomy_meta.get("title", "CT-derived assigned material phantom")
    grid_label = " x ".join(str(int(value)) for value in shape[::-1])
    spacing_label = " x ".join(f"{value:.3f}" for value in spacing)
    report = json.loads((args.run / "solver-report.json").read_text())
    completion_label = (
        f"{report['accepted_steps']} accepted steps · {report['termination'].replace('_', ' ')}"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "axes.titleweight": "medium",
            "axes.titlesize": 11,
            "figure.facecolor": "#fafaf7",
        }
    )
    outputs = {}
    renderer_source = args.output / "render.py"
    renderer_source.write_bytes(Path(__file__).read_bytes())
    outputs[renderer_source.name] = digest(renderer_source)
    rendering_versions = {
        name: importlib.metadata.version(name)
        for name in ("matplotlib", "scikit-image", "nibabel", "numpy")
    }
    volume_affine = np.eye(4)
    volume_affine[:3, :3] = orientation @ np.diag(spacing)
    volume_affine[:3, 3] = origin + np.asarray(anatomy_meta["object_centre_source_ras_mm"])
    for index, name in enumerate(("water", "bone")):
        volume = nib.Nifti1Image(
            np.ascontiguousarray(recovered[index].transpose(2, 1, 0)), volume_affine
        )
        volume.header.set_xyzt_units("mm")
        volume.header["descrip"] = (
            b"Reconstructed assigned phantom fraction; not patient composition"
        )
        target = args.output / f"reconstructed-{name}-fraction.nii.gz"
        nib.save(volume, target)
        outputs[target.name] = digest(target)

    def save(fig, stem):
        for extension in ("png", "pdf"):
            path = args.output / f"{stem}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
            outputs[path.name] = digest(path)
        plt.close(fig)

    # Display positions are fixed by the intake manifest, not chosen by fitting
    # quality. Reference, reconstruction and error use identical physical axes.
    z, y, x = indices
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, material in enumerate((0, 1)):
        values = [
            reference[material, z],
            recovered[material, z],
            recovered[material, z] - reference[material, z],
        ]
        titles = [
            "Assigned reference",
            "Reconstructed",
            "Reconstruction - reference",
        ]
        for column, (value, title) in enumerate(zip(values, titles, strict=True)):
            ax = axes[row, column]
            diff = column == 2
            artist = show_plane(ax, value, 0, lower, upper, difference=diff)
            ax.set_title(title)
            ax.set_ylabel(("Water" if material == 0 else "Bone") + " · y (mm)")
            fig.colorbar(
                artist,
                ax=ax,
                shrink=0.66,
                label="Fraction difference" if diff else "Volume fraction",
                extend="both" if diff else "neither",
            )
    fig.suptitle(
        f"{study_title}\n{describe_plane(0, z, origin, spacing)} · {completion_label}", fontsize=15
    )
    save(fig, "material-slices")

    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for axis, index in enumerate(indices):
        expected_plane = plane(reference[1], axis, index)
        recovered_plane = plane(recovered[1], axis, index)
        for column, (value, label) in enumerate(
            zip(
                (expected_plane, recovered_plane, recovered_plane - expected_plane),
                ("Assigned bone", "Recovered bone", "Fraction difference"),
                strict=True,
            )
        ):
            artist = show_plane(
                axes[axis, column], value, axis, lower, upper, difference=column == 2
            )
            axes[axis, column].set_title(f"{label}\n{describe_plane(axis, index, origin, spacing)}")
            fig.colorbar(
                artist,
                ax=axes[axis, column],
                shrink=0.65,
                label="Fraction difference" if column == 2 else "Volume fraction",
                extend="both" if column == 2 else "neither",
            )
    fig.suptitle(f"{study_title}\nBone material in three recorded planes", fontsize=15)
    save(fig, "bone-sections")

    # Surfaces are actual isosurfaces at one fixed half-assigned bone fraction.
    # No post-reconstruction smoothing, geometry repair or reference substitution.
    fig = plt.figure(figsize=(12, 7), facecolor="#101a24")
    threshold = 0.325
    surfaces = {}
    for column, (volume, title) in enumerate(
        [(reference[1], "Assigned reference"), (recovered[1], "Recovered bone fraction")]
    ):
        ax = fig.add_subplot(1, 2, column + 1, projection="3d", facecolor="#101a24")
        if volume.min() < threshold < volume.max():
            vertices, faces, _, _ = marching_cubes(
                volume, level=threshold, spacing=tuple(spacing[::-1])
            )
            vertices = vertices[:, ::-1] + origin
            faces = faces[:, ::-1]
            surface_path = args.output / (
                "reference-bone-surface.npz" if column == 0 else "reconstructed-bone-surface.npz"
            )
            np.savez_compressed(surface_path, vertices_xyz_mm=vertices, triangles=faces)
            outputs[surface_path.name] = digest(surface_path)
            surfaces[surface_path.name] = {"vertices": len(vertices), "triangles": len(faces)}
            mesh = Poly3DCollection(
                vertices[faces],
                facecolors="#e7c99b",
                linewidth=0,
                shade=True,
                lightsource=matplotlib.colors.LightSource(310, 40),
                rasterized=True,
            )
            ax.add_collection3d(mesh)
        else:
            ax.text2D(
                0.1, 0.5, "No surface at the fixed threshold", transform=ax.transAxes, color="white"
            )
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]))
        ax.set_box_aspect(upper - lower)
        ax.set_proj_type("ortho")
        ax.view_init(elev=12, azim=-75)
        ax.set_axis_off()
        ax.set_title(title, color="white", fontsize=15)
    fig.suptitle(
        f"{study_title}\nBone material reconstructed from simulated spectral radiographs",
        color="white",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.07,
        f"Fixed fraction {threshold:g} isosurface · no smoothing · {spacing_label} mm voxels\n"
        f"{grid_label} grid · {completion_label}\n"
        "Assigned phantom, not measured patient composition. Cropped surfaces remain open.",
        ha="center",
        color="#b4c4d1",
        fontsize=10,
    )
    save(fig, "bone-volumes")

    observed = np.load(acquisition / "observed-counts.npy", allow_pickle=False)
    expected = np.load(acquisition / "expected-counts.npy", allow_pickle=False)
    predictions = np.load(args.run / "heldout-predictions.npy", allow_pickle=False)
    heldout_views = source["configuration"]["heldout_views"]
    if (
        observed.ndim != 4
        or expected.shape != observed.shape
        or predictions.shape != (len(heldout_views), *observed.shape[1:])
        or not heldout_views
    ):
        raise ValueError("recorded radiograph arrays do not match the withheld-view layout")
    if any(
        not np.isfinite(values).all() or np.any(values < 0)
        for values in (observed, expected, predictions)
    ):
        raise ValueError("radiograph counts and expectations must be finite and nonnegative")
    with np.load(physics_path / "physics.npz", allow_pickle=False) as physics:
        open_beam = (physics["weights"] * physics["response"]).sum(axis=1)
    physics_metadata = json.loads((physics_path / "metadata.json").read_text())
    thresholds = physics_metadata["acquisition"]["thresholds_kev"]
    detector = source["configuration"]["geometry"]
    height, width = detector["shape"]
    du, dv = detector["spacing_mm"]
    extent = (-width * du / 2, width * du / 2, -height * dv / 2, height * dv / 2)
    view = heldout_views[0]
    channels = observed.shape[1]
    if len(thresholds) != channels + 1 or np.any(open_beam <= 0):
        raise ValueError("energy channel labels/open beam differ from the recorded counts")
    radiograph_windows = []
    fig, axes = plt.subplots(
        channels, 4, figsize=(14, 3.5 * channels), constrained_layout=True, squeeze=False
    )
    for channel in range(channels):
        reference_display = -np.log((expected[view, channel] + 0.5) / open_beam[channel])
        upper_log = max(1.0, float(np.ceil(np.percentile(reference_display, 99.5) * 2) / 2))
        radiograph_windows.append([0.0, upper_log])
        for column, (values, title) in enumerate(
            [
                (observed[view, channel], "Simulated photon counts"),
                (expected[view, channel], "Reference expectation"),
                (predictions[0, channel], "Held-out prediction"),
            ]
        ):
            # Continuity correction is display-only and never enters fitting.
            log_t = -np.log((values + 0.5) / open_beam[channel])
            artist = axes[channel, column].imshow(
                log_t,
                origin="lower",
                cmap="gray",
                vmin=0,
                vmax=upper_log,
                interpolation="nearest",
                extent=extent,
                aspect="equal",
            )
            energy_label = f"{thresholds[channel]:g}-{thresholds[channel + 1]:g} keV"
            axes[channel, column].set_title(title + "\n" + energy_label)
            axes[channel, column].set_axis_off()
        fig.colorbar(
            artist,
            ax=axes[channel, :3].tolist(),
            shrink=0.7,
            label="Display -log transmission",
            extend="both",
        )
        residual = (observed[view, channel].astype(float) - predictions[0, channel]) / np.sqrt(
            np.maximum(predictions[0, channel], 1.0)
        )
        residual_artist = axes[channel, 3].imshow(
            residual,
            origin="lower",
            cmap="RdBu_r",
            vmin=-5,
            vmax=5,
            interpolation="nearest",
            extent=extent,
            aspect="equal",
        )
        axes[channel, 3].set_title("Observation - prediction\nPoisson-scaled residual")
        axes[channel, 3].set_axis_off()
        fig.colorbar(
            residual_artist,
            ax=axes[channel, 3],
            shrink=0.7,
            label="(count - mean) / √max(mean, 1)",
            extend="both",
        )
    fig.suptitle(
        f"{study_title}\nHeld-out view {view} · {channels} disjoint energy channels",
        fontsize=15,
    )
    fig.supxlabel(
        "Identical reference-derived windows for observation, expectation and prediction;\n"
        "0.5 display correction only.",
        fontsize=9,
    )
    save(fig, "heldout-radiographs")

    history = [h for h in report["history"] if "objective" in h and h.get("accepted", True)]
    if history:
        fig, (ax, gradient_ax) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
        ax.plot(
            [h.get("iteration", i + 1) for i, h in enumerate(history)],
            [h["objective"] for h in history],
            color="#0072b2",
        )
        ax.set(
            xlabel="Accepted iteration",
            ylabel="Penalised Poisson objective",
            title="Recorded objective",
        )
        if all(h["objective"] > 0 for h in history):
            ax.set_yscale("log")
        ax.grid(alpha=0.2)
        gradient_ax.plot(
            [h["iteration"] - 1 for h in history] + [report["accepted_steps"]],
            [h["gradient_mapping_before"] for h in history] + [report["final_gradient_mapping"]],
            color="#009e73",
        )
        gradient_ax.axhline(
            report["gradient_mapping_threshold"],
            color="#d55e00",
            linestyle="--",
            label="Stopping threshold",
        )
        gradient_ax.set(
            xlabel="Accepted iteration",
            ylabel="Euclidean gradient mapping, infinity norm",
            title=f"Fixed mapping step {report['mapping_step']:g}",
        )
        gradient_ax.set_yscale("symlog", linthresh=max(report["gradient_mapping_threshold"], 1e-12))
        gradient_ax.grid(alpha=0.2)
        gradient_ax.legend(fontsize=8)
        fig.suptitle(f"Training record · {completion_label}")
        save(fig, "convergence")
    checkpoint_sources = []
    if args.checkpoint_comparison:
        # Only completed-record entries are eligible. Deterministic quarter and
        # halfway checkpoints show optimisation without selecting a favourable
        # reconstruction against its reference.
        available = sorted(
            (int(Path(name).stem.removeprefix("fractions-")), name)
            for name in record["output_sha256"]
            if name.startswith("checkpoints/fractions-") and name.endswith(".npy")
        )
        chosen = []
        for fraction in (0.25, 0.5):
            if available:
                item = min(
                    available,
                    key=lambda pair: (abs(pair[0] - fraction * report["accepted_steps"]), pair[0]),
                )
                if item not in chosen and item[0] < report["accepted_steps"]:
                    chosen.append(item)
        if chosen:
            panels = []
            for iteration, name in chosen:
                checkpoint = np.load(args.run / name, allow_pickle=False)
                if checkpoint.shape != recovered.shape or not np.isfinite(checkpoint).all():
                    raise ValueError(f"invalid recorded checkpoint: {name}")
                panels.append((checkpoint[1], f"Iteration {iteration}"))
                checkpoint_sources.append({"name": name, "sha256": record["output_sha256"][name]})
            panels += [
                (recovered[1], f"Final · {report['accepted_steps']} steps"),
                (reference[1], "Assigned reference"),
            ]
            fig, axes = plt.subplots(
                2,
                len(panels),
                figsize=(3.5 * len(panels), 7),
                constrained_layout=True,
                squeeze=False,
            )
            for row, axis in enumerate((0, 1)):
                for column, (volume, label) in enumerate(panels):
                    artist = show_plane(
                        axes[row, column], plane(volume, axis, indices[axis]), axis, lower, upper
                    )
                    axes[row, column].set_title(
                        f"{label}\n{describe_plane(axis, indices[axis], origin, spacing)}"
                    )
                fig.colorbar(
                    artist, ax=axes[row].tolist(), shrink=0.7, label="Bone volume fraction"
                )
            fig.suptitle(
                f"{study_title}\nRecorded checkpoints · identical planes and fraction scale",
                fontsize=15,
            )
            save(fig, "bone-checkpoints")

    # Prefer the unchanged native CT crop when intake retained it. Displayed
    # native planes are nearest to the fixed inverse-grid locations; no new CT
    # resampling, mask overlay or material inference enters this context plate.
    native_ct = "evaluation-native-ct-hu.npy" in anatomy_meta["outputs"]
    ct_name = "evaluation-native-ct-hu.npy" if native_ct else "evaluation-ct-hu.npy"
    ct = np.load(anatomy / ct_name, allow_pickle=False)
    ct_shape, ct_spacing, ct_origin, _, ct_lower, ct_upper = grid_geometry(
        anatomy_meta["forward_grid" if native_ct else "inverse_grid"]
    )
    if tuple(ct.shape) != tuple(ct_shape) or not np.isfinite(ct).all():
        raise ValueError("CT context differs from its recorded grid")
    ct_indices = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=True)
    for axis, index in enumerate(indices):
        coordinate = origin[2 - axis] + index * spacing[2 - axis]
        ct_index = int(np.rint((coordinate - ct_origin[2 - axis]) / ct_spacing[2 - axis]))
        if not 0 <= ct_index < ct_shape[axis]:
            raise ValueError("recorded display plane lies outside the CT context")
        ct_indices.append(ct_index)
        artist = axes[axis].imshow(
            plane(ct, axis, ct_index),
            origin="lower",
            cmap="gray",
            vmin=-300,
            vmax=1300,
            extent=plane_extent(ct_lower, ct_upper, axis),
            interpolation="nearest",
            aspect="equal",
        )
        horizontal, vertical = (("x", "y"), ("x", "z"), ("y", "z"))[axis]
        axes[axis].set(
            title=describe_plane(axis, ct_index, ct_origin, ct_spacing),
            xlabel=f"{horizontal} (mm)",
            ylabel=f"{vertical} (mm)",
        )
    fig.colorbar(artist, ax=axes.tolist(), shrink=0.75, label="CT attenuation (HU)", extend="both")
    fig.suptitle(
        f"Acquired CT context · {'native crop' if native_ct else 'recorded volume average'}\n"
        "Source anatomy only; the water/bone composition is assigned separately",
        fontsize=14,
    )
    save(fig, "acquired-ct-context")
    metadata = {
        "run": str(args.run),
        "run_sha256": digest(args.run / "run.json"),
        "renderer_sha256": digest(Path(__file__)),
        "rendering_library_versions": rendering_versions,
        "outputs": outputs,
        "title": study_title,
        "scientific_role": anatomy_meta.get(
            "scientific_role", "CT-derived assigned material phantom"
        ),
        "anatomical_scope": anatomy_meta.get("roi", {}),
        "acquisition_sha256": digest(acquisition / "run.json"),
        "anatomy_sha256": digest(frozen_anatomy),
        "physics_sha256": digest(physics_path / "physics.npz"),
        "physics_metadata_sha256": digest(physics_path / "metadata.json"),
        "material_slice_indices": {"axial": z, "coronal": y, "sagittal": x},
        "slice_selection": "intake display_indices_zyx; grid midpoint fallback for older manifests",
        "grid": anatomy_meta["inverse_grid"],
        "surface_fraction": threshold,
        "surface_smoothing": False,
        "surface_geometry": surfaces,
        "surface_camera": {
            "projection": "orthographic",
            "elevation_degrees": 12,
            "azimuth_degrees": -75,
        },
        "surface_coordinates": "object XYZ in millimetres; cropped boundaries are not capped",
        "nifti_affine": volume_affine.tolist(),
        "nifti_units": "dimensionless fractions; mm geometry; source-header laterality",
        "radiograph_display": {
            "formula": "-log((value+0.5)/open_beam); applied only for display",
            "view": view,
            "windows_by_channel": radiograph_windows,
            "window_rule": (
                "0 to the reference expectation display's 99.5th percentile, rounded up to 0.5, "
                "minimum upper limit 1; shared across each channel's three images"
            ),
            "residual": "(observed-predicted)/sqrt(max(predicted,1)); fixed limits [-5,5]",
            "detector_extent_uv_mm": list(extent),
        },
        "ct_context": {"array": ct_name, "indices_zyx": ct_indices, "window_hu": [-300, 1300]},
        "checkpoint_comparison_sources": checkpoint_sources,
        "attribution": {"source": anatomy_meta["source"], "rights": anatomy_meta["rights"]},
    }
    (args.output / "figures.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"figures": len(outputs), "output": str(args.output)}))


if __name__ == "__main__":
    main()
