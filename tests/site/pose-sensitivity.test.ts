import { describe, expect, it } from "vitest";
import { Matrix4, Vector3 } from "three";
import {
  assetUrl,
  decodePlanes,
  decodeFiniteDifferenceStatus,
  pixelIndex,
  posesForParameter,
  validateData,
  validateMeshes,
  unresolvedCoordinates,
  type SensitivityData,
} from "../../src/lib/pose-sensitivity/data";

// Small labelled records exercise display contracts only; they are never
// exported, rendered in the book or presented as scientific execution evidence.
function record(): SensitivityData {
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  return {
    schema_version: 1,
    id: "parser-fixture",
    title: "Parser fixture",
    anatomy_label: "Test metadata",
    signal_label: "Expected counts",
    signal_unit: "counts",
    width: 3,
    height: 2,
    detector: {
      source_mm: [0, 750, 0],
      centre_mm: [0, -450, 0],
      u_axis: [1, 0, 0],
      v_axis: [0, 0, -1],
      pixel_spacing_mm: [1.25, 1.25],
    },
    pivot_ras_mm: [0, 0, 0],
    reference: {
      projection: "reference.png",
      channels_f32: "reference.f32",
      translation_limit: 5,
      rotation_limit: 10,
    },
    sensitivities: (["tx", "ty", "tz", "rx", "ry", "rz"] as const).map(
      (id, i) => ({
        id,
        label: id,
        unit: `counts/${i < 3 ? "mm" : "rad"}`,
        image: `${id}.png`,
      }),
    ),
    poses: [
      {
        id: "positive",
        label: "Positive x",
        parameter: "tx",
        value: 1,
        matrix: identity.map((v, i) => (i === 3 ? 1 : v)),
        projection: "positive.png",
        projection_f32: "positive.f32",
      },
      {
        id: "reference",
        label: "Reference",
        parameter: "reference",
        value: 0,
        matrix: identity,
        projection: "reference.png",
        projection_f32: "reference-pose.f32",
      },
      {
        id: "negative",
        label: "Negative x",
        parameter: "tx",
        value: -1,
        matrix: identity.map((v, i) => (i === 3 ? -1 : v)),
        projection: "negative.png",
        projection_f32: "negative.f32",
      },
    ],
    provenance: { label: "Run provenance", url: "artifact.manifest.json" },
  };
}

describe("recorded pose sensitivity display contracts", () => {
  it("reads coordinate-interleaved finite-difference status without relabelling it as an invalid derivative", () => {
    const bytes = new Uint8Array(3 * 2 * 6);
    bytes[6 + 4] = 1;
    const status = decodeFiniteDifferenceStatus(bytes.buffer, 3, 2);
    expect(unresolvedCoordinates(status, 0)).toEqual([]);
    expect(unresolvedCoordinates(status, 1)).toEqual(["ry"]);
    expect(() => decodeFiniteDifferenceStatus(bytes.buffer, 3, 3)).toThrow(
      "byte count",
    );
    bytes[0] = 2;
    expect(() => decodeFiniteDifferenceStatus(bytes.buffer, 3, 2)).toThrow(
      "zero or one",
    );
  });
  it("accepts explicit display stretches without changing data units or recorded limits", () => {
    const data = record();
    data.reference.display = {
      projection: "log1p",
      sensitivities: "asinh",
      asinh_softening_fraction: 0.02,
      projection_max: 1000,
    };
    expect(validateData(data).reference.translation_limit).toBe(5);
    expect(data.sensitivities[0]!.unit).toBe("counts/mm");
    data.reference.display.asinh_softening_fraction = 0;
    expect(() => validateData(data)).toThrow("softening");
  });
  it("keeps translation and rotation units separate and sorts only the selected recorded axis", () => {
    const data = validateData(record());
    expect(posesForParameter(data, "tx").map((pose) => pose.value)).toEqual([
      -1, 0, 1,
    ]);
    expect(posesForParameter(data, "ry").map((pose) => pose.id)).toEqual([
      "reference",
    ]);
    data.sensitivities[4]!.unit = "counts/mm";
    expect(() => validateData(data)).toThrow("units");
  });
  it("rejects reordered channels, invalid geometry, scaled poses and reflections", () => {
    const reordered = record();
    reordered.sensitivities.reverse();
    expect(() => validateData(reordered)).toThrow("channel order");
    const geometry = record();
    geometry.detector.v_axis = [1, 0, 0];
    expect(() => validateData(geometry)).toThrow("orthonormal");
    const scaled = record();
    scaled.poses[0]!.matrix[0] = 2;
    expect(() => validateData(scaled)).toThrow("rigid");
    const mirrored = record();
    mirrored.poses[0]!.matrix[0] = -1;
    expect(() => validateData(mirrored)).toThrow("handedness");
  });
  it("maps stored row-major matrices into Three without transposing the physical translation", () => {
    const pose = validateData(record()).poses[0]!;
    const transform = new Matrix4().fromArray(pose.matrix).transpose();
    expect(new Vector3(4, 5, 6).applyMatrix4(transform).toArray()).toEqual([
      5, 5, 6,
    ]);
  });
  it("decodes signed little-endian samples without changing zeros or channel/pixel ordering", () => {
    const bytes = new ArrayBuffer(2 * 3 * 7 * 4);
    const view = new DataView(bytes);
    for (let i = 0; i < 42; i++) view.setFloat32(i * 4, i - 20, true);
    const planes = decodePlanes(bytes, 3, 2, 7);
    expect(planes[0]).toBe(-20);
    expect(planes[20]).toBe(0);
    expect(planes[3 * 6 + pixelIndex(2, 1, 3, 2)]).toBe(3);
    expect(() => decodePlanes(bytes, 2, 2, 7)).toThrow("byte count");
    view.setFloat32(0, NaN, true);
    expect(() => decodePlanes(bytes, 3, 2, 7)).toThrow("nonfinite");
  });
  it("uses top-to-bottom row-major detector indices and clamps the image boundary", () => {
    expect(pixelIndex(0, 0, 480, 256)).toBe(0);
    expect(pixelIndex(479, 255, 480, 256)).toBe(480 * 256 - 1);
    expect(pixelIndex(480, 256, 480, 256)).toBe(480 * 256 - 1);
    expect(pixelIndex(-1, -1, 480, 256)).toBe(0);
    expect(pixelIndex(NaN, Infinity, 480, 256)).toBe(0);
  });
  it("rejects remote/traversal assets and mesh indices outside the supplied vertices", () => {
    expect(assetUrl("maps/tx.png")).toBe(
      "/generated/introduction-pose-sensitivity/maps/tx.png",
    );
    for (const file of [
      "../private.nii",
      "/etc/passwd",
      "https://example.com/image.png",
      "maps/../x.png",
      "maps//x.png",
    ])
      expect(() => assetUrl(file)).toThrow();
    expect(() =>
      validateMeshes({
        schema_version: 1,
        coordinate_system: "RAS-mm-pivot-centred",
        meshes: [
          {
            id: "test",
            label: "Index fixture",
            positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 3],
          },
        ],
      }),
    ).toThrow("triangles");
  });
});
