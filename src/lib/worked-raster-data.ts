export interface RecordedArray {
  shape: number[];
  dtype: string;
  values: number[];
}
export interface RasterPayload {
  schema_version: number;
  kind: "registration" | "reconstruction";
  arrays: Record<string, RecordedArray>;
  grid?: { shape: number[]; spacing_mm: number[]; origin_mm: number[] };
}
export interface RasterPlane {
  width: number;
  height: number;
  values: number[];
}
export type RGB = [number, number, number];
export type RasterScale = "counts" | "fraction" | "error";

/** The book's palette uses six-digit screen colours and shorthand print white. */
export function paletteColour(value: string): RGB {
  let hex = value.trim().replace(/^#/, "");
  if (/^[\da-f]{3}$/i.test(hex))
    hex = [...hex].map((digit) => digit + digit).join("");
  if (!/^[\da-f]{6}$/i.test(hex))
    throw new Error("Invalid recorded figure palette colour");
  return [0, 2, 4].map((offset) =>
    parseInt(hex.slice(offset, offset + 2), 16),
  ) as RGB;
}

export function validateRasterPayload(
  value: unknown,
  kind: RasterPayload["kind"],
): RasterPayload {
  const data = value as RasterPayload;
  if (!data || data.schema_version !== 1 || data.kind !== kind || !data.arrays)
    throw new Error("Unrecognised recorded array data");
  const required =
    kind === "registration"
      ? [
          "initial_a180",
          "observed_a180",
          "final_a180",
          "initial_a270",
          "observed_a270",
          "final_a270",
        ]
      : [
          "reference",
          "initial_rep0",
          "update100_rep0",
          "final_rep0",
          "initial_rep1",
          "update100_rep1",
          "final_rep1",
        ];
  for (const name of required) {
    const array = data.arrays[name];
    if (
      !array ||
      !Array.isArray(array.shape) ||
      !Array.isArray(array.values) ||
      !array.shape.every((n) => Number.isSafeInteger(n) && n > 0) ||
      array.shape.reduce((a, b) => a * b, 1) !== array.values.length ||
      !array.values.every(Number.isFinite) ||
      (kind === "registration"
        ? array.shape.join() !== "128,128"
        : array.shape.join() !== "2,16,16,16")
    )
      throw new Error(`Invalid recorded array: ${name}`);
  }
  if (
    kind === "reconstruction" &&
    (!data.grid ||
      !Array.isArray(data.grid.shape) ||
      data.grid.shape.join() !== "16,16,16" ||
      !Array.isArray(data.grid.spacing_mm) ||
      !Array.isArray(data.grid.origin_mm) ||
      data.grid.spacing_mm.length !== 3 ||
      data.grid.origin_mm.length !== 3 ||
      !data.grid.spacing_mm.every((x) => Number.isFinite(x) && x > 0) ||
      !data.grid.origin_mm.every(Number.isFinite))
  )
    throw new Error("Invalid recorded grid");
  return data;
}

/** The grid origin is the first sample centre, not the outer voxel corner. */
export function materialPlane(
  array: RecordedArray,
  material: number,
  zMm: number,
  originZ: number,
  spacingZ: number,
): RasterPlane {
  const [materials, depth, height, width] = array.shape as [
    number,
    number,
    number,
    number,
  ];
  if (
    !Number.isInteger(material) ||
    material < 0 ||
    material >= materials ||
    !Number.isFinite(zMm) ||
    !Number.isFinite(originZ) ||
    !Number.isFinite(spacingZ) ||
    !(spacingZ > 0)
  )
    throw new Error("Invalid recorded slice coordinate");
  const position = (zMm - originZ) / spacingZ;
  if (position < -1e-10 || position > depth - 1 + 1e-10)
    throw new Error("Slice outside sample centres");
  const bounded = Math.max(0, Math.min(depth - 1, position));
  const low = Math.floor(bounded),
    high = Math.min(depth - 1, low + 1),
    weight = bounded - low;
  const count = width * height,
    start = material * depth * count;
  return {
    width,
    height,
    values: Array.from(
      { length: count },
      (_, i) =>
        (1 - weight) * array.values[start + low * count + i]! +
        weight * array.values[start + high * count + i]!,
    ),
  };
}
export function differencePlane(a: RasterPlane, b: RasterPlane): RasterPlane {
  if (a.width !== b.width || a.height !== b.height)
    throw new Error("Unmatched recorded planes");
  return {
    width: a.width,
    height: a.height,
    values: a.values.map((v, i) => v - b.values[i]!),
  };
}
const mix = (a: RGB, b: RGB, t: number): RGB =>
  a.map((v, i) => Math.round(v + (b[i]! - v) * t)) as RGB;
export function rasterColour(
  value: number,
  scale: RasterScale,
  paper: RGB,
  blue: RGB,
  rust: RGB,
): RGB {
  if (!Number.isFinite(value))
    throw new Error("Nonfinite recorded display value");
  if (scale === "counts") {
    if (value < 0) throw new Error("Negative recorded count");
    const grey = Math.round(
      255 * Math.max(0, Math.min(1, Math.log10(1 + value) / Math.log10(10001))),
    );
    return [grey, grey, grey];
  }
  if (scale === "fraction")
    return mix(paper, blue, Math.max(0, Math.min(1, value)));
  return value < 0
    ? mix(paper, blue, Math.min(1, -value / 0.1))
    : mix(paper, rust, Math.min(1, value / 0.1));
}
export function rasterPixels(
  plane: RasterPlane,
  scale: RasterScale,
  paper: RGB,
  blue: RGB,
  rust: RGB,
): Uint8ClampedArray<ArrayBuffer> {
  const pixels = new Uint8ClampedArray(plane.width * plane.height * 4);
  for (let row = 0; row < plane.height; row++)
    for (let col = 0; col < plane.width; col++) {
      const colour = rasterColour(
        plane.values[row * plane.width + col]!,
        scale,
        paper,
        blue,
        rust,
      );
      const target = ((plane.height - 1 - row) * plane.width + col) * 4;
      pixels.set([...colour, 255], target);
    }
  return pixels;
}
