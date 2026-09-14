export const PARAMETERS = ["tx", "ty", "tz", "rx", "ry", "rz"] as const;
export type Parameter = (typeof PARAMETERS)[number];
export function parameterLabel(parameter: Parameter): string {
  return {
    tx: "Translation x (right)",
    ty: "Translation y (anterior)",
    tz: "Translation z (superior)",
    rx: "Rotation about x",
    ry: "Rotation about y",
    rz: "Rotation about z",
  }[parameter];
}
export function poseLabel(pose: RecordedPose): string {
  return pose.parameter === "reference"
    ? "Reference pose"
    : `${parameterLabel(pose.parameter)}: ${pose.value > 0 ? "+" : ""}${formatValue(pose.value)} ${pose.parameter.startsWith("t") ? "mm" : "rad"}`;
}
export type Vec3 = [number, number, number];
export interface RecordedPose {
  id: string;
  label: string;
  parameter: Parameter | "reference";
  value: number;
  matrix: number[];
  projection: string;
  projection_f32: string;
}
export interface SensitivityData {
  schema_version: 1;
  id: string;
  title: string;
  anatomy_label: string;
  signal_label: string;
  signal_unit: string;
  width: number;
  height: number;
  detector: {
    source_mm: Vec3;
    centre_mm: Vec3;
    u_axis: Vec3;
    v_axis: Vec3;
    pixel_spacing_mm: [number, number];
  };
  pivot_ras_mm: Vec3;
  reference: {
    projection: string;
    channels_f32: string;
    fd_status_u8?: string;
    translation_limit: number;
    rotation_limit: number;
    display?: {
      projection: "log1p";
      sensitivities: "asinh";
      asinh_softening_fraction: number;
      projection_max: number;
    };
  };
  sensitivities: {
    id: Parameter;
    label: string;
    unit: string;
    image: string;
  }[];
  poses: RecordedPose[];
  provenance: { label: string; url: string };
}
export interface PelvisMeshes {
  schema_version: 1;
  coordinate_system: "RAS-mm-pivot-centred";
  meshes: {
    id: string;
    label: string;
    positions: number[];
    indices: number[];
  }[];
}

function invariant(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Pose sensitivity: ${message}`);
}
function finiteVector(value: unknown, length: number): value is number[] {
  return (
    Array.isArray(value) &&
    value.length === length &&
    value.every(Number.isFinite)
  );
}
export function assetUrl(file: string): string {
  invariant(
    typeof file === "string" && /^[a-zA-Z0-9][a-zA-Z0-9._/-]*$/.test(file),
    "invalid asset path",
  );
  invariant(
    file
      .split("/")
      .every((part) => part !== ".." && part !== "." && part !== ""),
    "invalid asset path",
  );
  return `/generated/introduction-pose-sensitivity/${file}`;
}

/** Validate recorded metadata before it can drive a display or a rigid transform. */
export function validateData(input: unknown): SensitivityData {
  invariant(input !== null && typeof input === "object", "missing record");
  const data = input as SensitivityData;
  invariant(data.schema_version === 1, "unsupported schema");
  for (const name of [
    "id",
    "title",
    "anatomy_label",
    "signal_label",
    "signal_unit",
  ] as const)
    invariant(
      typeof data[name] === "string" && data[name].length,
      `missing ${name}`,
    );
  invariant(
    Number.isInteger(data.width) &&
      data.width > 0 &&
      Number.isInteger(data.height) &&
      data.height > 0 &&
      data.width * data.height <= 4_194_304,
    "invalid detector dimensions",
  );
  invariant(finiteVector(data.pivot_ras_mm, 3), "invalid sacral pivot");
  invariant(
    data.detector &&
      finiteVector(data.detector.source_mm, 3) &&
      finiteVector(data.detector.centre_mm, 3),
    "invalid source or detector",
  );
  invariant(
    finiteVector(data.detector.u_axis, 3) &&
      finiteVector(data.detector.v_axis, 3),
    "invalid detector axes",
  );
  const u = data.detector.u_axis,
    v = data.detector.v_axis;
  invariant(
    Math.abs(Math.hypot(...u) - 1) < 1e-6 &&
      Math.abs(Math.hypot(...v) - 1) < 1e-6 &&
      Math.abs(u.reduce((sum, n, i) => sum + n * v[i]!, 0)) < 1e-6,
    "detector axes must be orthonormal",
  );
  invariant(
    finiteVector(data.detector.pixel_spacing_mm, 2) &&
      data.detector.pixel_spacing_mm.every((n) => n > 0),
    "invalid detector spacing",
  );
  invariant(
    data.reference &&
      Number.isFinite(data.reference.translation_limit) &&
      data.reference.translation_limit >= 0 &&
      Number.isFinite(data.reference.rotation_limit) &&
      data.reference.rotation_limit >= 0,
    "invalid sensitivity scales",
  );
  assetUrl(data.reference.projection);
  assetUrl(data.reference.channels_f32);
  if (data.reference.fd_status_u8 !== undefined)
    assetUrl(data.reference.fd_status_u8);
  if (data.reference.display) {
    const display = data.reference.display;
    invariant(
      display.projection === "log1p" && display.sensitivities === "asinh",
      "unsupported image display transform",
    );
    invariant(
      Number.isFinite(display.projection_max) && display.projection_max > 0,
      "invalid projection display maximum",
    );
    invariant(
      Number.isFinite(display.asinh_softening_fraction) &&
        display.asinh_softening_fraction > 0 &&
        display.asinh_softening_fraction <= 1,
      "invalid sensitivity display softening",
    );
  }
  invariant(
    Array.isArray(data.sensitivities) && data.sensitivities.length === 6,
    "exactly six sensitivity images are required",
  );
  data.sensitivities.forEach((channel, index) => {
    invariant(
      channel.id === PARAMETERS[index],
      "sensitivity channel order must be tx ty tz rx ry rz",
    );
    invariant(
      channel.unit === `${data.signal_unit}/${index < 3 ? "mm" : "rad"}`,
      "sensitivity units do not match their coordinate",
    );
    invariant(
      typeof channel.label === "string" && channel.label.length,
      "missing sensitivity label",
    );
    assetUrl(channel.image);
  });
  invariant(
    Array.isArray(data.poses) && data.poses.length > 0,
    "no recorded poses",
  );
  const ids = new Set<string>();
  for (const pose of data.poses) {
    invariant(
      typeof pose.id === "string" && pose.id.length && !ids.has(pose.id),
      "duplicate or empty pose ID",
    );
    ids.add(pose.id);
    invariant(
      typeof pose.label === "string" && pose.label.length,
      "missing pose label",
    );
    invariant(
      pose.parameter === "reference" || PARAMETERS.includes(pose.parameter),
      "unknown pose parameter",
    );
    invariant(Number.isFinite(pose.value), "invalid pose value");
    invariant(
      finiteVector(pose.matrix, 16),
      "pose must contain a row-major 4 by 4 matrix",
    );
    const m = pose.matrix;
    invariant(
      m[12] === 0 && m[13] === 0 && m[14] === 0 && m[15] === 1,
      "pose is not an affine transform",
    );
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        const dot = [0, 1, 2].reduce(
          (sum, k) => sum + m[i * 4 + k]! * m[j * 4 + k]!,
          0,
        );
        invariant(
          Math.abs(dot - Number(i === j)) < 1e-5,
          "pose is not a rigid transform",
        );
      }
    const determinant =
      m[0]! * (m[5]! * m[10]! - m[6]! * m[9]!) -
      m[1]! * (m[4]! * m[10]! - m[6]! * m[8]!) +
      m[2]! * (m[4]! * m[9]! - m[5]! * m[8]!);
    invariant(Math.abs(determinant - 1) < 1e-5, "pose changes handedness");
    assetUrl(pose.projection);
    assetUrl(pose.projection_f32);
  }
  invariant(
    data.poses.filter((pose) => pose.parameter === "reference").length === 1,
    "exactly one reference pose is required",
  );
  const reference = data.poses.find((pose) => pose.parameter === "reference")!;
  invariant(
    reference.value === 0 &&
      reference.matrix.every(
        (v, i) => Math.abs(v - Number(i % 5 === 0)) < 1e-8,
      ),
    "reference pose must be identity",
  );
  invariant(
    data.provenance && typeof data.provenance.label === "string",
    "missing provenance",
  );
  assetUrl(data.provenance.url);
  return data;
}

export function validateMeshes(input: unknown): PelvisMeshes {
  invariant(
    input !== null && typeof input === "object",
    "missing anatomy meshes",
  );
  const data = input as PelvisMeshes;
  invariant(
    data.schema_version === 1 &&
      data.coordinate_system === "RAS-mm-pivot-centred",
    "unsupported mesh coordinates",
  );
  invariant(
    Array.isArray(data.meshes) && data.meshes.length > 0,
    "empty anatomy mesh set",
  );
  for (const mesh of data.meshes) {
    invariant(
      Array.isArray(mesh.positions) &&
        mesh.positions.length >= 9 &&
        mesh.positions.length % 3 === 0 &&
        mesh.positions.every(Number.isFinite),
      "invalid mesh vertices",
    );
    invariant(
      Array.isArray(mesh.indices) &&
        mesh.indices.length >= 3 &&
        mesh.indices.length % 3 === 0 &&
        mesh.indices.every(
          (i) => Number.isInteger(i) && i >= 0 && i < mesh.positions.length / 3,
        ),
      "invalid mesh triangles",
    );
  }
  return data;
}

/** The published planes are little-endian float32, channel-major, row-major. */
export function decodePlanes(
  buffer: ArrayBuffer,
  width: number,
  height: number,
  channels: number,
): Float32Array {
  invariant(
    buffer.byteLength === width * height * channels * 4,
    "recorded plane byte count does not match detector dimensions",
  );
  const view = new DataView(buffer);
  const values = new Float32Array(buffer.byteLength / 4);
  for (let i = 0; i < values.length; i++) {
    const value = view.getFloat32(i * 4, true);
    invariant(
      Number.isFinite(value),
      "recorded plane contains a nonfinite value",
    );
    values[i] = value;
  }
  return values;
}

/** FD status is pixel-major, with six coordinate-interleaved uint8 flags. */
export function decodeFiniteDifferenceStatus(
  buffer: ArrayBuffer,
  width: number,
  height: number,
): Uint8Array {
  invariant(
    buffer.byteLength === width * height * 6,
    "finite-difference status byte count does not match detector dimensions",
  );
  const values = new Uint8Array(buffer);
  invariant(
    values.every((value) => value === 0 || value === 1),
    "finite-difference status must contain only zero or one",
  );
  return values;
}
export function unresolvedCoordinates(
  status: Uint8Array,
  pixel: number,
): Parameter[] {
  invariant(
    Number.isInteger(pixel) && pixel >= 0 && pixel * 6 + 5 < status.length,
    "finite-difference pixel index is out of bounds",
  );
  return PARAMETERS.filter((_, index) => status[pixel * 6 + index] === 1);
}

export function pixelIndex(
  column: number,
  row: number,
  width: number,
  height: number,
): number {
  const clamp = (value: number, max: number) =>
    Math.min(
      max - 1,
      Math.max(0, Number.isFinite(value) ? Math.floor(value) : 0),
    );
  return clamp(row, height) * width + clamp(column, width);
}
export function posesForParameter(
  data: SensitivityData,
  parameter: Parameter,
): RecordedPose[] {
  return data.poses
    .filter(
      (pose) => pose.parameter === parameter || pose.parameter === "reference",
    )
    .sort((a, b) => a.value - b.value);
}
export function formatValue(value: number): string {
  return value === 0 ? "0" : Number(value.toPrecision(4)).toString();
}
