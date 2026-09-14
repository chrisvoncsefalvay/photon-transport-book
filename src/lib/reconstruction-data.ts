/** Display of recorded scalar fields. No projection or reconstruction runs here. */
export type Axis = 0 | 1 | 2;
export type Point = [number, number, number];
export interface RecordedGrid {
  shape: Point;
  spacing_mm: Point;
  origin_mm: Point;
  orientation: number[];
}
export interface RecordedBuffer {
  file: string;
  sha256: string;
  bytes: number;
  dtype: string;
}
export interface RecordedScalar extends RecordedBuffer {
  id: string;
  label: string;
  role: "reference" | "reconstructed";
  material: string;
  units: string;
  window: [number, number];
}
export interface RecordedMesh {
  id: string;
  label: string;
  scalar_id: string;
  threshold: number;
  positions: RecordedBuffer;
  indices: RecordedBuffer;
}
export interface RecordedImage {
  id: string;
  label: string;
  file: string;
  sha256: string;
  bytes: number;
  width: number;
  height: number;
  alt: string;
}
export interface RecordedProjection {
  view_zero_based: number;
  label: string;
  colour_rgb: Point;
  source_mm: Point;
  detector_corners_mm: [Point, Point, Point, Point];
  source_detector_distance_mm: number;
  /** Three row-major homogeneous rows: u numerator, v numerator, depth. */
  uv_matrix_rows: [number[], number[], number[]];
  image: RecordedImage;
}
export interface ReconstructionCase {
  id: string;
  title: string;
  kind: string;
  description: string;
  qualification: string[];
  grid: RecordedGrid;
  scalars: RecordedScalar[];
  meshes: RecordedMesh[];
  images: RecordedImage[];
  projections?: RecordedProjection[];
  metrics: {
    accepted_steps: number;
    termination: string;
    converged: boolean;
    calibration_pass?: boolean;
    bone_rmse?: number;
    water_rmse?: number;
    bone_dice?: number;
    heldout_total_rms?: number;
    heldout_high_rms?: number;
  };
  acquisition: {
    fitting_views: number;
    total_views: number;
    heldout_views: number;
    primary_heldout_views?: number;
    angular_span_degrees?: number;
    incident_count_protocol: string;
    display: { slice_indices_xyz: Point; camera_direction: Point };
  };
  provenance: {
    title: string;
    url: string;
    licence: string;
    source_sha256: string;
  }[];
}

export const planes = [
  { name: "Axial", fixed: 2, horizontal: 0, vertical: 1 },
  { name: "Coronal", fixed: 1, horizontal: 0, vertical: 2 },
  { name: "Sagittal", fixed: 0, horizontal: 1, vertical: 2 },
] as const;

export function dimensions(grid: RecordedGrid): Point {
  return [grid.shape[2], grid.shape[1], grid.shape[0]];
}

export function voxelOffset(grid: RecordedGrid, point: Point): number {
  const size = dimensions(grid);
  if (
    point.some((n, axis) => !Number.isInteger(n) || n < 0 || n >= size[axis]!)
  )
    throw new Error("Slice coordinate lies outside the recorded grid.");
  return (point[2] * grid.shape[1] + point[1]) * grid.shape[2] + point[0];
}

export function physicalPoint(grid: RecordedGrid, point: Point): Point {
  voxelOffset(grid, point);
  return grid.origin_mm.map(
    (origin, row) =>
      origin +
      point.reduce(
        (sum, index, axis) =>
          sum +
          grid.orientation[row * 3 + axis]! * index * grid.spacing_mm[axis]!,
        0,
      ),
  ) as Point;
}

export function planeValues(
  values: Float32Array,
  grid: RecordedGrid,
  point: Point,
  plane: (typeof planes)[number],
  reference?: Float32Array,
): Float64Array {
  const size = dimensions(grid);
  const width = size[plane.horizontal];
  const height = size[plane.vertical];
  const expected = grid.shape.reduce((a, b) => a * b, 1);
  if (
    values.length !== expected ||
    (reference && reference.length !== expected)
  )
    throw new Error("The scalar field does not match its recorded grid.");
  voxelOffset(grid, point);
  const output = new Float64Array(width * height);
  const index: Point = [...point];
  for (let row = 0; row < height; row++) {
    index[plane.vertical] = height - 1 - row;
    for (let column = 0; column < width; column++) {
      index[plane.horizontal] = column;
      const offset = voxelOffset(grid, index);
      output[row * width + column] =
        values[offset]! - (reference?.[offset] ?? 0);
    }
  }
  return output;
}

export function pointOnPlane(
  grid: RecordedGrid,
  point: Point,
  plane: (typeof planes)[number],
  horizontal: number,
  vertical: number,
): Point {
  if (!Number.isFinite(horizontal) || !Number.isFinite(vertical))
    throw new Error("Invalid slice position.");
  voxelOffset(grid, point);
  const result: Point = [...point];
  const size = dimensions(grid);
  const clamp = (value: number, count: number): number =>
    Math.max(0, Math.min(count - 1, Math.floor(value * count)));
  result[plane.horizontal] = clamp(horizontal, size[plane.horizontal]);
  result[plane.vertical] =
    size[plane.vertical] - 1 - clamp(vertical, size[plane.vertical]);
  return result;
}

export function colour(
  value: number,
  window: [number, number],
  error = false,
): Point {
  if (
    !Number.isFinite(value) ||
    !Number.isFinite(window[0]) ||
    !Number.isFinite(window[1]) ||
    window[1] <= window[0]
  )
    throw new Error("Invalid display range.");
  const t = Math.max(
    0,
    Math.min(1, (value - window[0]) / (window[1] - window[0])),
  );
  if (!error) {
    const shade = Math.round(t * 255);
    return [shade, shade, shade];
  }
  const mix = t < 0.5 ? t * 2 : (1 - t) * 2;
  const end = t < 0.5 ? [53, 104, 137] : [159, 53, 41];
  return end.map((c) => Math.round(c + (245 - c) * mix)) as Point;
}

export function validateGrid(grid: RecordedGrid): void {
  if (
    grid.shape.length !== 3 ||
    grid.shape.some((n) => !Number.isInteger(n) || n < 1) ||
    grid.shape.reduce((a, b) => a * b, 1) > 4_194_304 ||
    grid.spacing_mm.length !== 3 ||
    grid.spacing_mm.some((n) => !Number.isFinite(n) || n <= 0) ||
    grid.origin_mm.length !== 3 ||
    grid.origin_mm.some((n) => !Number.isFinite(n)) ||
    grid.orientation.length !== 9 ||
    grid.orientation.some((n) => !Number.isFinite(n))
  )
    throw new Error("Invalid recorded display grid.");
  for (let a = 0; a < 3; a++)
    for (let b = 0; b < 3; b++) {
      let dot = 0;
      for (let i = 0; i < 3; i++)
        dot += grid.orientation[i * 3 + a]! * grid.orientation[i * 3 + b]!;
      if (Math.abs(dot - Number(a === b)) > 1e-7)
        throw new Error("Recorded grid axes are not orthonormal.");
    }
  const m = grid.orientation;
  const determinant =
    m[0]! * (m[4]! * m[8]! - m[5]! * m[7]!) -
    m[1]! * (m[3]! * m[8]! - m[5]! * m[6]!) +
    m[2]! * (m[3]! * m[7]! - m[4]! * m[6]!);
  if (Math.abs(determinant - 1) > 1e-7)
    throw new Error("Recorded display grid must be right-handed.");
}

export function lineValues(
  values: Float32Array,
  grid: RecordedGrid,
  point: Point,
  axis: Axis,
): { distance_mm: number; value: number }[] {
  if (values.length !== grid.shape.reduce((a, b) => a * b, 1))
    throw new Error("The scalar field does not match its recorded grid.");
  voxelOffset(grid, point);
  const cursor: Point = [...point];
  return Array.from({ length: dimensions(grid)[axis] }, (_, index) => {
    cursor[axis] = index;
    return {
      distance_mm: index * grid.spacing_mm[axis],
      value: values[voxelOffset(grid, cursor)]!,
    };
  });
}

const buffers = new Map<string, Promise<ArrayBuffer>>();
export function recordedBuffer(
  base: string,
  record: RecordedBuffer,
): Promise<ArrayBuffer> {
  if (
    !/^[a-zA-Z0-9_./-]+$/.test(record.file) ||
    record.file.startsWith("/") ||
    record.file.split("/").includes("..") ||
    !/^[a-f0-9]{64}$/.test(record.sha256) ||
    !Number.isInteger(record.bytes) ||
    record.bytes < 1 ||
    record.bytes > 25 * 1024 * 1024
  )
    return Promise.reject(new Error("Invalid recorded asset declaration."));
  const url = new URL(record.file, new URL(base, document.baseURI)).href;
  const key = `${url}:${record.sha256}`;
  let pending = buffers.get(key);
  if (!pending) {
    pending = (async () => {
      const response = await fetch(url);
      if (!response.ok)
        throw new Error("The recorded volume could not be loaded.");
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength !== record.bytes)
        throw new Error("Recorded asset length changed.");
      // HTTPS/localhost additionally verify transport bytes. Authoring over plain
      // HTTP still uses the build-verified manifest, size and numeric-domain gates.
      if (globalThis.crypto?.subtle) {
        const hash = await crypto.subtle.digest("SHA-256", buffer);
        const hex = Array.from(new Uint8Array(hash), (b) =>
          b.toString(16).padStart(2, "0"),
        ).join("");
        if (hex !== record.sha256)
          throw new Error("Recorded asset checksum changed.");
      }
      return buffer;
    })();
    buffers.set(key, pending);
    void pending.catch(() => buffers.delete(key));
  }
  return pending;
}

export function decodeScalars(
  buffer: ArrayBuffer,
  grid: RecordedGrid,
): Float32Array {
  const length = grid.shape.reduce((a, b) => a * b, 1);
  if (buffer.byteLength !== length * 4)
    throw new Error("Wrong scalar field size.");
  const view = new DataView(buffer);
  const values = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const value = view.getFloat32(i * 4, true);
    if (!Number.isFinite(value) || value < 0)
      throw new Error("Invalid recorded material coefficient.");
    values[i] = value;
  }
  return values;
}
