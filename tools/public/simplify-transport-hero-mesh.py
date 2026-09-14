"""Prepare a checked display LOD; never alter the canonical pelvic meshes.

Optional authoring dependencies: numpy, pyvista 0.47.1 and VTK 9.6.1.
Static site builds use the checked-in JSON and binary. This tool simplifies
display geometry only; it does not change any CT or transport calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "public/generated/introduction-pose-sensitivity/meshes.json"
OUTPUT = ROOT / "public/generated/transport-hero"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def triangles(mesh: pv.PolyData) -> np.ndarray:
    cells = np.asarray(mesh.faces).reshape(-1, 4)
    if not np.all(cells[:, 0] == 3):
        raise ValueError("A triangle-only surface is required.")
    return cells[:, 1:]


def components(edges: np.ndarray, vertices: np.ndarray) -> int:
    """Count connected components without repairing or modifying the surface."""
    parent = {int(vertex): int(vertex) for vertex in vertices}

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for a, b in edges:
        parent[find(int(a))] = find(int(b))
    return len({find(vertex) for vertex in parent})


def topology(mesh: pv.PolyData) -> dict:
    faces = triangles(mesh)
    edges, incidence = np.unique(
        np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), 1),
        axis=0,
        return_counts=True,
    )
    boundary = edges[incidence == 1]
    boundary_vertices, degree = np.unique(boundary, return_counts=True)
    return {
        "components": components(edges, np.unique(faces)),
        "eulerCharacteristic": int(mesh.n_points - len(edges) + len(faces)),
        "boundaryLoops": components(boundary, boundary_vertices),
        "boundaryEdges": len(boundary),
        "nonManifoldEdges": int(np.count_nonzero(incidence > 2)),
        "nonCycleBoundaryVertices": int(np.count_nonzero(degree != 2)),
    }


def surface_samples(mesh: pv.PolyData) -> np.ndarray:
    """Probe vertices, unique edge midpoints and triangle centroids, deterministically."""
    faces = triangles(mesh)
    edges = np.unique(
        np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), 1),
        axis=0,
    )
    points = np.asarray(mesh.points)
    return np.concatenate((points, points[edges].mean(axis=1), points[faces].mean(axis=1)))


def distances(source: pv.PolyData, target: pv.PolyData) -> dict:
    samples = surface_samples(source)
    distance = np.abs(pv.PolyData(samples).compute_implicit_distance(target)["implicit_distance"])
    return {
        "samples": len(samples),
        "meanMm": float(np.mean(distance)),
        "rmsMm": float(np.sqrt(np.mean(np.square(distance)))),
        "p95Mm": float(np.percentile(distance, 95)),
        "maxSampledMm": float(np.max(distance)),
    }


def simplify(source: pv.PolyData, mesh_id: str) -> tuple[pv.PolyData, dict]:
    if mesh_id == "sacrum":
        # A low feature angle protects the small sacral foramina and detached
        # source components. The filter deliberately stops above its target.
        settings = {
            "algorithm": "vtkDecimatePro",
            "targetTriangles": 1400,
            "featureAngleDegrees": 26,
            "preserveTopology": True,
            "splitting": False,
            "boundaryVertexDeletion": False,
        }
        result = source.decimate_pro(
            1 - settings["targetTriangles"] / source.n_cells,
            feature_angle=settings["featureAngleDegrees"],
            preserve_topology=True,
            splitting=False,
            boundary_vertex_deletion=False,
        )
    else:
        # Quadric decimation retains hip curvature better at this budget.
        # Its topology is verified below; the algorithm alone is no guarantee.
        settings = {
            "algorithm": "vtkQuadricDecimation",
            "targetTriangles": 1100,
            "volumePreservation": True,
            "boundaryWeight": 1,
            "inputOrdering": "VTK connected component 0, dataset_surface extraction",
        }
        connected = source.connectivity()
        ordered = connected.extract_cells(connected.cell_data["RegionId"] == 0).extract_surface(
            algorithm="dataset_surface"
        )
        if ordered.n_cells != source.n_cells:
            raise ValueError("Hip simplification would omit a source component.")
        result = ordered.decimate(
            1 - settings["targetTriangles"] / source.n_cells,
            volume_preservation=True,
            boundary_weight=1,
        )
    # Normals are generated on the final surface without splitting vertices.
    result = result.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=False,
    )
    return result, settings


def preview(originals: list, simplified: list, output: Path) -> None:
    """Write an optional local comparison, outside normal build artefacts."""
    plot = pv.Plotter(off_screen=True, shape=(2, 3), window_size=(1500, 1100))
    views = [(0, 1, 0.06), (-1.1, 1.5, 0.2), (-1, 0, 0.06)]
    names = ["Anterior", "Hero oblique", "Lateral"]
    for row, collection in enumerate((originals, simplified)):
        for col, direction in enumerate(views):
            plot.subplot(row, col)
            plot.set_background("#f4f1e8")
            for mesh in collection:
                plot.add_mesh(
                    mesh,
                    color="#e0ddd3",
                    smooth_shading=True,
                    ambient=0.3,
                    diffuse=0.7,
                    specular=0,
                )
            plot.camera_position = [np.asarray(direction) * 1000, (0, 35, -40), (0, 0, 1)]
            plot.enable_parallel_projection()
            plot.camera.parallel_scale = 180
            label = "Original" if row == 0 else "Display LOD"
            plot.add_text(f"{label}: {names[col]}", font_size=13, color="#425d68")
    output.parent.mkdir(parents=True, exist_ok=True)
    plot.screenshot(str(output))
    plot.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, help="Optional local six-view comparison PNG.")
    args = parser.parse_args()
    source_bytes = SOURCE.read_bytes()
    payload = json.loads(source_bytes)
    if payload["coordinate_system"] != "RAS-mm-pivot-centred":
        raise ValueError("Unexpected source coordinate system.")
    if [item["id"] for item in payload["meshes"]] != ["sacrum", "left-hip", "right-hip"]:
        raise ValueError("The three original labelled pelvic components are required.")
    binary = bytearray()
    meshes, checks, originals, simplified = [], [], [], []

    def append(values: np.ndarray, dtype: str) -> dict:
        while len(binary) % 4:
            binary.append(0)
        array = np.asarray(values, dtype=dtype).reshape(-1)
        descriptor = {"byteOffset": len(binary), "count": int(array.size)}
        binary.extend(array.tobytes())
        return descriptor

    for item in payload["meshes"]:
        positions = np.asarray(item["positions"], dtype=np.float64).reshape(-1, 3)
        indices = np.asarray(item["indices"], dtype=np.int64).reshape(-1, 3)
        original = pv.PolyData(positions, np.column_stack((np.full(len(indices), 3), indices)))
        lod, settings = simplify(original, item["id"])
        before, after = topology(original), topology(lod)
        for key in (
            "components",
            "eulerCharacteristic",
            "boundaryLoops",
            "nonManifoldEdges",
            "nonCycleBoundaryVertices",
        ):
            if before[key] != after[key]:
                raise ValueError(f"{item['id']}: simplification changed topology {key}.")
        if after["nonManifoldEdges"] != 0 or after["nonCycleBoundaryVertices"] != 0:
            raise ValueError(
                "The display LOD must have no non-manifold edges or branched boundaries."
            )
        normals = np.asarray(lod.point_data["Normals"])
        if not np.isfinite(lod.points).all() or not np.isfinite(normals).all():
            raise ValueError("LOD positions and normals must be finite.")
        if not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5):
            raise ValueError("LOD vertex normals must be unit length.")
        if lod.n_points > 65535:
            raise ValueError("LOD exceeds Uint16 index capacity.")
        forward, reverse = distances(original, lod), distances(lod, original)
        if max(forward["p95Mm"], reverse["p95Mm"]) > 1.5:
            raise ValueError("LOD exceeds the 1.5 mm sampled p95 surface-distance limit.")
        if max(forward["maxSampledMm"], reverse["maxSampledMm"]) > 4:
            raise ValueError("LOD exceeds the 4 mm maximum sampled surface-distance limit.")
        meshes.append(
            {
                "id": item["id"],
                "label": item["label"],
                "vertexCount": lod.n_points,
                "triangleCount": lod.n_cells,
                "positions": append(lod.points, "<f4"),
                "normals": append(normals, "<f4"),
                "indices": append(triangles(lod), "<u2"),
            }
        )
        checks.append(
            {
                "id": item["id"],
                "originalTriangles": original.n_cells,
                "simplifiedTriangles": lod.n_cells,
                "settings": settings,
                "originalTopology": before,
                "simplifiedTopology": after,
                "originalToSimplified": forward,
                "simplifiedToOriginal": reverse,
            }
        )
        originals.append(original)
        simplified.append(lod)
    count = sum(mesh["triangleCount"] for mesh in meshes)
    if count > 6000 or len(binary) > 150000:
        raise ValueError("Display LOD exceeds its triangle or binary size budget.")
    if SOURCE.read_bytes() != source_bytes:
        raise ValueError("The source mesh changed during generation; rerun on one source revision.")
    metadata = {
        "schemaVersion": 1,
        "coordinateSystem": payload["coordinate_system"],
        "buffer": "pelvis-lod.bin",
        "bufferByteLength": len(binary),
        "encoding": {
            "byteOrder": "little-endian",
            "positions": "float32 xyz, RAS millimetres",
            "normals": "float32 xyz unit vectors",
            "indices": "uint16 triangle vertex indices, local to each component",
            "descriptors": (
                "byteOffset is 4-byte aligned; count is the number of scalar array values"
            ),
        },
        "meshes": meshes,
        "provenance": {
            "kind": "Simplified display geometry; not an attenuation or transport input",
            "sourcePath": str(SOURCE.relative_to(ROOT)),
            "sourceSha256": digest(source_bytes),
            "generatorSha256": digest(Path(__file__).read_bytes()),
            "bufferSha256": digest(bytes(binary)),
            "attribution": payload["attribution"],
            "changes": (
                "Existing three display meshes decimated in unchanged RAS mm coordinates; "
                "vertex normals recomputed without vertex splitting. No anatomy is generated, "
                "components added, pose applied or original source edited."
            ),
            "libraries": {
                "numpy": np.__version__,
                "pyvista": pv.__version__,
                "vtk": vtk.vtkVersion.GetVTKVersion(),
            },
            "validation": {
                "distanceMethod": (
                    "Unsigned closest-point distance to the other triangle surface, evaluated "
                    "in both directions at all vertices, unique edge midpoints and face centroids. "
                    "The reported maximum is sampled, not a certified Hausdorff bound."
                ),
                "topologyMethod": (
                    "Connected component count, Euler characteristic, boundary loop count, "
                    "edge incidence and boundary vertex degree compared before and after."
                ),
                "meshes": checks,
            },
            "regeneration": "python3 tools/public/simplify-transport-hero-mesh.py",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "pelvis-lod.bin").write_bytes(binary)
    (OUTPUT / "pelvis-lod.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.preview:
        preview(originals, simplified, args.preview)
    print(json.dumps({"triangles": count, "binaryBytes": len(binary), "checks": checks}))


if __name__ == "__main__":
    main()
