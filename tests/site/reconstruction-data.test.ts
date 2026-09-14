import { describe, expect, it } from "vitest";
import {
  colour,
  decodeScalars,
  dimensions,
  lineValues,
  physicalPoint,
  planeValues,
  planes,
  pointOnPlane,
  validateGrid,
  voxelOffset,
  type RecordedGrid,
} from "../../src/lib/reconstruction-data";

// Asymmetric display fixture: each address has an independently recognisable
// value. These tests validate indexing; they are not reconstruction evidence.
const grid: RecordedGrid = {
  shape: [2, 3, 4],
  spacing_mm: [2, 3, 5],
  origin_mm: [10, -20, 30],
  orientation: [0, -1, 0, 1, 0, 0, 0, 0, 1],
};
const field = new Float32Array([
  0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 100, 101, 102, 103, 110, 111, 112,
  113, 120, 121, 122, 123,
]);

describe("recorded reconstruction display", () => {
  it("uses XYZ physical spacing and orientation on ZYX scalar storage", () => {
    validateGrid(grid);
    expect(dimensions(grid)).toEqual([4, 3, 2]);
    expect(voxelOffset(grid, [3, 2, 1])).toBe(23);
    expect(physicalPoint(grid, [3, 2, 1])).toEqual([4, -14, 35]);
  });
  it("preserves all three plane axes and displays increasing vertical coordinates upwards", () => {
    expect([...planeValues(field, grid, [2, 1, 1], planes[0])]).toEqual([
      120, 121, 122, 123, 110, 111, 112, 113, 100, 101, 102, 103,
    ]);
    expect([...planeValues(field, grid, [2, 1, 1], planes[1])]).toEqual([
      110, 111, 112, 113, 10, 11, 12, 13,
    ]);
    expect([...planeValues(field, grid, [2, 1, 1], planes[2])]).toEqual([
      102, 112, 122, 2, 12, 22,
    ]);
  });
  it("maps the displayed corner and out-of-frame clicks to exact voxel centres", () => {
    expect(pointOnPlane(grid, [2, 1, 1], planes[0], 0, 0)).toEqual([0, 2, 1]);
    expect(pointOnPlane(grid, [2, 1, 1], planes[0], 1, 1)).toEqual([3, 0, 1]);
    expect(pointOnPlane(grid, [2, 1, 1], planes[1], -1, 2)).toEqual([0, 1, 0]);
    expect(() => pointOnPlane(grid, [2, 1, 1], planes[0], NaN, 0)).toThrow();
  });
  it("subtracts exact stored FP32 values in double precision without clipping signed errors", () => {
    const reference = new Float32Array(field.length).fill(0.1);
    const values = new Float32Array(field.length).fill(0.3);
    const errors = planeValues(values, grid, [2, 1, 1], planes[2], reference);
    expect(errors[0]).toBe(Number(values[0]) - Number(reference[0]));
    expect(planeValues(reference, grid, [2, 1, 1], planes[2], values)[0]).toBe(
      -errors[0]!,
    );
  });
  it("reports line distance in millimetres along the selected grid axis", () => {
    expect(lineValues(field, grid, [2, 1, 1], 1)).toEqual([
      { distance_mm: 0, value: 102 },
      { distance_mm: 3, value: 112 },
      { distance_mm: 6, value: 122 },
    ]);
  });
  it("decodes the declared little-endian byte order and rejects invalid numerical payloads", () => {
    const bytes = new ArrayBuffer(96),
      view = new DataView(bytes);
    for (let i = 0; i < field.length; i++)
      view.setFloat32(i * 4, field[i]!, true);
    expect(decodeScalars(bytes, grid)).toEqual(field);
    expect(() => decodeScalars(bytes.slice(0, -4), grid)).toThrow();
    view.setFloat32(0, NaN, true);
    expect(() => decodeScalars(bytes, grid)).toThrow();
    view.setFloat32(0, -0.1, true);
    expect(() => decodeScalars(bytes, grid)).toThrow();
  });
  it("rejects malformed, excessive, reflected or nonorthogonal grids", () => {
    expect(() =>
      validateGrid({ ...grid, shape: [1024, 1024, 1024] }),
    ).toThrow();
    expect(() => validateGrid({ ...grid, spacing_mm: [2, 0, 5] })).toThrow();
    expect(() =>
      validateGrid({ ...grid, orientation: [-1, 0, 0, 0, 1, 0, 0, 0, 1] }),
    ).toThrow("right-handed");
    expect(() =>
      validateGrid({ ...grid, orientation: [1, 0.1, 0, 0, 1, 0, 0, 0, 1] }),
    ).toThrow("orthonormal");
    expect(() =>
      planeValues(field.subarray(1), grid, [2, 1, 1], planes[0]),
    ).toThrow();
    expect(() => voxelOffset(grid, [4, 1, 1])).toThrow();
  });
  it("keeps the declared greyscale and signed error endpoints fixed", () => {
    expect(colour(0, [0, 1])).toEqual([0, 0, 0]);
    expect(colour(1, [0, 1])).toEqual([255, 255, 255]);
    expect(colour(0, [-1, 1], true)).toEqual([245, 245, 245]);
    expect(colour(-1, [-1, 1], true)).toEqual([53, 104, 137]);
    expect(colour(1, [-1, 1], true)).toEqual([159, 53, 41]);
    expect(() => colour(0, [1, 1])).toThrow();
  });
});
