"""Build a private, self-contained viewer from hash-verified figure exports."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import subprocess
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def checked_file(folder, name, expected):
    path = (folder / name).resolve()
    if not path.is_relative_to(folder) or not path.is_file() or digest(path) != expected:
        raise ValueError(f"figure export is missing or changed: {name}")
    return path


def solver_status(report):
    steps, reason = report["accepted_steps"], report["termination"]
    if type(steps) is not int or steps < 0 or not isinstance(reason, str) or not reason:
        raise ValueError("solver report requires a nonnegative accepted-step count and termination")
    explanation = {
        "iteration_budget": "iteration budget reached; convergence not established",
        "projected_gradient_tolerance": "projected-gradient stopping tolerance met",
        "line_search_failed": "line search failed; convergence not established",
    }.get(reason, f"stopped ({reason}); convergence not established")
    noun = "step" if steps == 1 else "steps"
    return f"{steps:,} accepted {noun} · {explanation}"


def assemble(figures, output):
    figures = figures.resolve()
    output = output.resolve()
    repository = Path(__file__).resolve().parents[2]
    if output.parent != figures or output.suffix.lower() != ".html":
        raise ValueError("write the self-contained HTML inside its private figure folder")
    if output.is_relative_to(repository):
        raise ValueError("private viewer outputs must remain outside the repository")
    if output.exists():
        raise FileExistsError(output)
    manifest_path = figures / "figures.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = {
        name: checked_file(figures, name, expected)
        for name, expected in manifest["outputs"].items()
    }
    names = ("reference-bone-surface.npz", "reconstructed-bone-surface.npz")
    if any(name not in inputs for name in (*names, "bone-volumes.png")):
        raise ValueError("viewer requires both verified exported bone meshes and bone-volumes.png")
    if manifest.get("surface_smoothing") is not False or manifest.get("surface_fraction") != 0.325:
        raise ValueError("viewer requires the recorded unsmoothed 0.325 fraction surfaces")
    run_path = Path(manifest["run"]) / "run.json"
    if digest(run_path) != manifest["run_sha256"]:
        raise ValueError("reconstruction identity differs from the verified figures")
    run = json.loads(run_path.read_text())
    if run["status"] != "complete":
        raise ValueError("an unfinished reconstruction cannot supply the comparison viewer")
    report_path = checked_file(
        run_path.parent.resolve(), "solver-report.json", run["output_sha256"]["solver-report.json"]
    )
    report = json.loads(report_path.read_text())
    run_status = solver_status(report)
    grid = manifest["grid"]
    shape = np.asarray(grid["shape"], dtype=int)
    spacing = np.asarray(grid["spacing_mm"], dtype=float)
    origin = np.asarray(grid["origin_mm"], dtype=float)
    orientation = np.asarray(grid["orientation"], dtype=float).reshape(3, 3)
    if not np.array_equal(orientation, np.eye(3)):
        raise ValueError("surface exports must use the recorded axis-aligned object coordinates")
    lower = origin - spacing / 2
    upper = lower + shape[::-1] * spacing
    centre = (lower + upper) / 2
    meshes = []
    for name, label in zip(names, ("Assigned reference", "Reconstructed bone"), strict=True):
        with np.load(inputs[name], allow_pickle=False) as archive:
            vertices = np.asarray(archive["vertices_xyz_mm"], dtype=np.float64)
            triangles = np.asarray(archive["triangles"])
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not len(vertices)
            or len(vertices) > np.iinfo(np.uint32).max
            or not np.isfinite(vertices).all()
            or triangles.ndim != 2
            or triangles.shape[1] != 3
            or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer)
            or triangles.min() < 0
            or triangles.max() >= len(vertices)
        ):
            raise ValueError(f"invalid exported mesh: {name}")
        if np.any(vertices < lower - 1e-5) or np.any(vertices > upper + 1e-5):
            raise ValueError(f"mesh extends beyond its recorded physical grid: {name}")
        centred = vertices - centre
        rounding = float(np.max(np.abs(centred.astype(np.float32).astype(float) - centred)))
        meshes.append(
            {
                "name": name,
                "label": label,
                "sha256": manifest["outputs"][name],
                "vertexCount": len(vertices),
                "triangleCount": len(triangles),
                "positionsFloat64LE": encode(vertices, "<f8"),
                "trianglesUint32LE": encode(triangles, "<u4"),
                "gpuPositionRoundingMaxMm": rounding,
            }
        )
    entry = Path(__file__).with_name("viewer.js")
    three = repository / "node_modules/three"
    three_version = json.loads((three / "package.json").read_text())["version"]
    if three_version != "0.185.1":
        raise ValueError(
            f"reviewed viewer requires the installed Three.js 0.185.1, got {three_version}"
        )
    bundles = list(
        (repository / "node_modules/.pnpm").glob("esbuild@*/node_modules/esbuild/bin/esbuild")
    )
    if len(bundles) != 1:
        raise ValueError("expected one existing esbuild installation; install no new dependencies")
    result = subprocess.run(
        [
            str(bundles[0]),
            str(entry),
            "--bundle",
            "--format=iife",
            "--platform=browser",
            "--target=es2020",
            "--minify",
            "--legal-comments=inline",
        ],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    )
    bundle_script = result.stdout.replace("</script", "<\\/script")
    provenance = {
        "figureManifestSha256": digest(manifest_path),
        "runSha256": manifest["run_sha256"],
        "solverReportSha256": digest(report_path),
        "solverSummary": {
            "acceptedSteps": report["accepted_steps"],
            "termination": report["termination"],
            "display": run_status,
        },
        "acquisitionSha256": manifest["acquisition_sha256"],
        "anatomySha256": manifest["anatomy_sha256"],
        "physicsSha256": manifest["physics_sha256"],
        "physicsMetadataSha256": manifest["physics_metadata_sha256"],
        "rendererSha256": manifest["renderer_sha256"],
        "builderSha256": digest(Path(__file__)),
        "viewerSourceSha256": digest(entry),
        "threeVersion": three_version,
        "threeModuleSha256": digest(three / "build/three.module.js"),
        "threeCoreSha256": digest(three / "build/three.core.js"),
        "threeLicenseSha256": digest(three / "LICENSE"),
        "threeLicense": (three / "LICENSE").read_text(),
        "orbitControlsSha256": digest(three / "examples/jsm/controls/OrbitControls.js"),
        "inlineBundleSha256": hashlib.sha256(bundle_script.encode()).hexdigest(),
        "esbuildVersion": json.loads((bundles[0].parent.parent / "package.json").read_text())[
            "version"
        ],
        "sourceFiles": {name: manifest["outputs"][name] for name in (*names, "bone-volumes.png")},
        "grid": grid,
        "surfaceFraction": manifest["surface_fraction"],
        "geometrySmoothing": False,
        "surfaceCoordinates": manifest["surface_coordinates"],
        "rendering": (
            "Original indexed vertices and triangles; centred FP32 GPU positions, interpolated "
            "lighting normals; no geometry smoothing, decimation or boundary capping"
        ),
        "gpuPositionRoundingMaxMm": [mesh["gpuPositionRoundingMaxMm"] for mesh in meshes],
        "meshTriangleCounts": {mesh["label"]: mesh["triangleCount"] for mesh in meshes},
        "attribution": manifest["attribution"],
        "anatomicalScope": manifest.get("anatomical_scope", {}),
    }
    payload = {
        "title": manifest["title"],
        "meshes": meshes,
        "centre": centre.tolist(),
        "boundsSize": (upper - lower).tolist(),
        "provenance": provenance,
    }
    payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    fallback = "data:image/png;base64," + base64.b64encode(
        inputs["bone-volumes.png"].read_bytes()
    ).decode("ascii")
    spacing_label = " x ".join(f"{value:.3f}" for value in spacing)
    title = html.escape(manifest["title"])
    document = HTML.replace("{{TITLE}}", title).replace("{{SPACING}}", spacing_label)
    document = document.replace("{{RUN_STATUS}}", html.escape(run_status))
    document = document.replace("{{FALLBACK}}", fallback).replace("{{DATA}}", payload_json)
    document = document.replace("{{PROVENANCE}}", html.escape(json.dumps(provenance, indent=2)))
    document = document.replace("{{BUNDLE}}", bundle_script)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(document)
    return {
        "output": str(output),
        "sha256": digest(output),
        "bytes": output.stat().st_size,
        "provenance": provenance,
    }


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>{{TITLE}} · material reconstruction</title>
<style>
:root{color-scheme:dark;--ink:#edf0eb;--muted:#a2b0b4;--line:#29383d;--paper:#111d23;--accent:#e3c99f}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;background:#0a1318;color:var(--ink);font:15px/1.55 "Avenir Next",Avenir,"Segoe UI",sans-serif}
main{max-width:1500px;margin:auto;padding:38px 38px 26px}
.eyebrow{margin:0 0 12px;color:var(--accent);font:11px/1.5 "SFMono-Regular",Consolas,monospace;letter-spacing:.16em;text-transform:uppercase}
h1{margin:0 0 12px;max-width:1000px;font:400 clamp(29px,3.5vw,49px)/1.1 "Iowan Old Style","Palatino Linotype",Georgia,serif;letter-spacing:-.035em}
.intro{max-width:1000px;margin:0;color:var(--muted)}
.facts{display:flex;gap:10px 26px;flex-wrap:wrap;margin:24px 0 18px;font-size:12px;color:#c1ccc9}
.facts span{border-left:2px solid #596a65;padding-left:10px}
.run-status{margin:-4px 0 20px;color:var(--accent);font-size:13px}
.toolbar{display:flex;justify-content:space-between;gap:16px;align-items:center;flex-wrap:wrap;margin:0 0 14px}
.controls{display:flex;gap:6px;flex-wrap:wrap}
button{appearance:none;border:1px solid #43504f;border-radius:5px;background:#17242a;color:#e7e9de;padding:9px 13px;font:inherit;font-size:12px;cursor:pointer}
button:hover{background:#26383d;border-color:#87968c}button:focus-visible,summary:focus-visible,.stage:focus-visible{outline:2px solid var(--accent);outline-offset:4px}
button[aria-pressed=true]{background:var(--accent);color:#182125;border-color:var(--accent)}
.help{margin:0;color:var(--muted);font-size:12px}
.stage{position:relative;display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden;touch-action:none;isolation:isolate}
.stage canvas{position:absolute;inset:0;width:100%;height:100%;z-index:0}
.viewport{position:relative;height:clamp(360px,48vw,660px);pointer-events:none;z-index:1}
.panel-caption{position:absolute;left:22px;right:20px;top:18px;display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.panel-caption h2{margin:0;font-size:16px;font-weight:500}
.axis-label{position:absolute;bottom:14px;left:22px;font:10px/1.4 Consolas,monospace;color:#8c9f9e;letter-spacing:.06em}
.status{min-height:24px;margin:10px 0 20px;color:#a8b8b4;font-size:12px}
.footer-note{margin:12px 0 22px;color:var(--muted);font-size:12px;max-width:1000px}
details{border-top:1px solid var(--line);padding:15px 0;color:var(--muted)}summary{cursor:pointer;color:#d3dcd7;font-size:13px;width:fit-content}
.fallback img{display:block;width:100%;height:auto;max-width:1250px;margin:16px auto;border-radius:6px}
.fallback p{font-size:12px}.provenance pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.65 Consolas,monospace;color:#a9beb7;max-height:520px;overflow:auto}
.provenance p{font-size:12px}.bottom{display:flex;justify-content:space-between;gap:20px;color:#71888a;font-size:11px;margin-top:18px}
@media(max-width:760px){main{padding:24px 16px}.stage{grid-template-columns:1fr}.viewport{height:420px}.panel-caption{left:16px;top:15px}.facts{gap:10px 16px}.toolbar{gap:10px}.help{font-size:11px}.bottom{display:block}.bottom span{display:block}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
@media print{body{background:white;color:black}.interactive,.status{display:none!important}.fallback{display:block!important}.fallback summary{display:none}.fallback img{display:block}main{padding:0}.facts,.intro,.footer-note,.run-status{color:#333}}
</style></head><body><main>
<p class="eyebrow">Differentiable photon transport / recorded material reconstruction</p>
<h1>{{TITLE}}</h1>
<p class="intro">The assigned reference and reconstructed bone share one camera and the same physical scale.
Drag either view to inspect the recovered geometry from every angle.</p>
<div class="facts"><span>Bone fraction isosurface: 0.325</span><span>{{SPACING}} mm voxels</span><span>No geometry smoothing or added caps</span></div>
<p class="run-status">{{RUN_STATUS}}</p>
<div class="interactive" id="interactive" hidden>
<div class="toolbar"><div class="controls" aria-label="Camera presets">
<button type="button" data-view="oblique" aria-pressed="true">Oblique</button>
<button type="button" data-view="coronal" aria-pressed="false">Coronal</button>
<button type="button" data-view="sagittal" aria-pressed="false">Sagittal</button>
<button type="button" id="reset">Reset view</button></div>
<div class="controls" aria-label="Keyboard-accessible camera controls">
<button type="button" data-action="left" aria-label="Rotate left">←</button><button type="button" data-action="right" aria-label="Rotate right">→</button>
<button type="button" data-action="up" aria-label="Tilt up">↑</button><button type="button" data-action="down" aria-label="Tilt down">↓</button>
<button type="button" data-action="in" aria-label="Zoom in">+</button><button type="button" data-action="out" aria-label="Zoom out">-</button></div></div>
<p class="help">Drag to orbit · Scroll or pinch to zoom · Right-drag to pan · Arrow keys, +/- and R also work when the view is focused</p>
<div class="stage" id="stage" tabindex="0" role="group" aria-label="Linked three-dimensional bone views. Arrow keys rotate; plus and minus zoom; R resets.">
<section class="viewport" data-mesh="0" aria-label="Assigned material reference"><div class="panel-caption"><h2>Assigned material reference</h2></div><span class="axis-label">OBJECT XYZ · MILLIMETRES</span></section>
<section class="viewport" data-mesh="1" aria-label="Recovered from spectral projections"><div class="panel-caption"><h2>Recovered from spectral projections</h2></div><span class="axis-label">OBJECT XYZ · MILLIMETRES</span></section>
</div></div>
<p class="status" id="status" role="status" aria-live="polite">The verified static comparison is available below.</p>
<p class="footer-note">This is a controlled CT-derived assigned phantom with simulated spectral radiographs, not a measurement of patient composition.
Surfaces at cropped volume boundaries remain open. Source laterality follows the archived image header.</p>
<details class="fallback" id="static-plate" open><summary>Verified static comparison</summary><img src="{{FALLBACK}}" alt="The recorded assigned and reconstructed bone isosurfaces at the same fixed threshold and physical scale"><p>This PNG comes from the same verified figure export as the interactive geometry.</p></details>
<details class="provenance"><summary>Data and rendering provenance</summary><p>Original mesh topology is retained. Surface lighting interpolates normals; vertex positions are not smoothed. All scripts, geometry and the fallback image are embedded in this file. No network request is required.</p><pre id="provenance">{{PROVENANCE}}</pre></details>
<div class="bottom"><span>Private research artefact · recorded data, no browser transport solver</span><span>Linked camera / identical physical scale</span></div>
<script type="application/json" id="viewer-data">{{DATA}}</script><script>{{BUNDLE}}</script>
</main></body></html>
"""  # noqa: E501 - The self-contained HTML keeps one CSS rule or element per line.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(assemble(args.figures, args.output or args.figures / "bone-viewer.html")))


if __name__ == "__main__":
    main()
