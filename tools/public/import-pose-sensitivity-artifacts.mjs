import { createHash } from "node:crypto";
import {
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { inflateSync } from "node:zlib";

import {
  assertNoSymlinks,
  parseUniqueJson,
  readRegularFile,
  walkFiles,
} from "./lib/files.mjs";
import { assertSchema } from "./lib/schema.mjs";

const TARGET = "public/generated/introduction-pose-sensitivity";
const GENERATOR = "experiments/pose-sensitivity/run.py";
const PARAMETERS = ["tx", "ty", "tz", "rx", "ry", "rz"];
const CHECKS = [
  "input_hashes",
  "finite",
  "quadrature_counts",
  "quadrature_derivatives",
  "display_quadrature",
  "discrete_fd",
  "independent_reference",
  "contractions",
];
const EXTRA_SOURCES = [
  "run.py",
  "common.py",
  "prepare.py",
  "mesh.py",
  "config.json",
  "requirements-volume.txt",
].map((name) => `experiments/pose-sensitivity/${name}`);
const RUNTIME_KEYS = [
  "python",
  "warp",
  "numpy",
  "nibabel",
  "scipy",
  "skimage",
  "device",
  "device_name",
  "compute_capability",
  "cuda_toolkit",
  "cuda_driver_api",
  "cuda_driver",
];
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const jsonBytes = (value) => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
const invariant = (condition, message) => {
  if (!condition) throw new Error(message);
};
const isObject = (value) =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const sameSet = (actual, expected) =>
  actual.length === expected.length &&
  expected.every((item) => actual.includes(item));
const finiteVector = (value, size) =>
  Array.isArray(value) && value.length === size && value.every(Number.isFinite);
const close = (a, b, tolerance = 1e-8) => Math.abs(a - b) <= tolerance;

function object(value, required, optional = [], label = "payload") {
  invariant(isObject(value), `${label} must be an object`);
  invariant(
    required.every((key) => Object.hasOwn(value, key)),
    `${label} is missing a required field`,
  );
  invariant(
    Object.keys(value).every((key) => [...required, ...optional].includes(key)),
    `${label} contains an unexpected field`,
  );
}

function text(value, label) {
  invariant(
    typeof value === "string" &&
      value.length > 0 &&
      value.length <= 8192 &&
      !/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(value),
    `${label} must be nonempty text`,
  );
}

function publicMetadata(value, label = "metadata") {
  if (typeof value === "number")
    invariant(Number.isFinite(value), `${label} contains a nonfinite number`);
  else if (typeof value === "string") {
    invariant(
      !/(?:\/home\/|\/Users\/|\/private\/|\/tmp\/|\/mnt\/|\/Volumes\/|file:\/\/|[A-Za-z]:\\|\.nii(?:\.gz)?(?:\b|$)|\.pt(?:\b|$)|\.safetensors(?:\b|$))/.test(
        value,
      ),
      `${label} contains a private path or raw/model filename`,
    );
  } else if (Array.isArray(value))
    value.forEach((item) => publicMetadata(item, label));
  else if (isObject(value)) {
    for (const [key, child] of Object.entries(value)) {
      invariant(
        !/(?:^|_)(?:path|paths|hostname|command|token|secret|environment|source_status|nvidia_smi)(?:_|$)/i.test(
          key,
        ),
        `${label} contains a private metadata field: ${key}`,
      );
      publicMetadata(child, `${label}.${key}`);
    }
  }
}

function filename(value, extension) {
  invariant(
    typeof value === "string" &&
      /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value) &&
      value.endsWith(extension),
    `Invalid flat figure filename: ${JSON.stringify(value)}`,
  );
  return value;
}

function timestamp(value, label) {
  invariant(
    typeof value === "string" &&
      /^\d{4}-\d{2}-\d{2}T/.test(value) &&
      Number.isFinite(Date.parse(value)),
    `Invalid ${label}`,
  );
}

function rigidMatrix(pose) {
  invariant(finiteVector(pose.matrix, 16), `Invalid pose matrix: ${pose.id}`);
  const expected = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const coordinate = PARAMETERS.indexOf(pose.parameter);
  if (coordinate >= 0 && coordinate < 3)
    expected[coordinate * 4 + 3] = pose.value;
  else if (coordinate >= 3) {
    const axis = coordinate - 3,
      a = (axis + 1) % 3,
      b = (axis + 2) % 3;
    expected[a * 4 + a] = expected[b * 4 + b] = Math.cos(pose.value);
    expected[a * 4 + b] = -Math.sin(pose.value);
    expected[b * 4 + a] = Math.sin(pose.value);
  } else
    invariant(
      pose.parameter === "reference" && pose.value === 0,
      "Invalid reference pose",
    );
  invariant(
    pose.matrix.every((value, i) => close(value, expected[i])),
    `Pose matrix does not match its declared coordinate/value: ${pose.id}`,
  );
}

function configuredCoordinates(configuration) {
  const coordinates = ["reference:0"];
  for (const [field, parameters] of [
    ["display_translation_mm", PARAMETERS.slice(0, 3)],
    ["display_rotation_rad", PARAMETERS.slice(3)],
  ]) {
    const steps = configuration[field];
    invariant(
      Array.isArray(steps) && steps.length >= 3 && steps.every(Number.isFinite),
      `${field} must contain at least three finite recorded display steps`,
    );
    invariant(
      steps.every((step, index) => index === 0 || step > steps[index - 1]),
      `${field} must be strictly increasing with unique steps`,
    );
    invariant(
      steps.includes(0) && steps[0] < 0 && steps.at(-1) > 0,
      `${field} must contain zero between negative and positive steps`,
    );
    for (const parameter of parameters)
      for (const step of steps)
        if (step !== 0) coordinates.push(`${parameter}:${step}`);
  }
  return coordinates;
}

function validateData(data, configuration) {
  object(
    data,
    [
      "schema_version",
      "id",
      "title",
      "anatomy_label",
      "signal_label",
      "signal_unit",
      "width",
      "height",
      "detector",
      "pivot_ras_mm",
      "reference",
      "sensitivities",
      "poses",
      "provenance",
    ],
    [],
    "data.json",
  );
  invariant(
    data.schema_version === 1 &&
      data.width === 480 &&
      data.height === 256 &&
      data.width === configuration.width &&
      data.height === configuration.height,
    "Unexpected final detector dimensions or schema",
  );
  for (const key of ["id", "title", "anatomy_label", "signal_label"])
    text(data[key], key);
  invariant(data.signal_unit === "counts", "Signal unit must be counts");
  invariant(
    finiteVector(data.pivot_ras_mm, 3) &&
      finiteVector(configuration.pivot_ras_mm, 3) &&
      data.pivot_ras_mm.every((value, i) =>
        close(value, configuration.pivot_ras_mm[i]),
      ),
    "Pivot differs from the recorded configuration",
  );
  object(
    data.detector,
    ["source_mm", "centre_mm", "u_axis", "v_axis", "pixel_spacing_mm"],
    [],
    "detector",
  );
  for (const [key, expected] of Object.entries({
    source_mm: configuration.source_mm,
    centre_mm: configuration.detector_centre_mm,
    u_axis: [1, 0, 0],
    v_axis: [0, 0, -1],
    pixel_spacing_mm: [
      configuration.detector_width_mm / data.width,
      configuration.detector_height_mm / data.height,
    ],
  })) {
    invariant(
      finiteVector(data.detector[key], expected?.length) &&
        finiteVector(expected, key === "pixel_spacing_mm" ? 2 : 3) &&
        data.detector[key].every((value, i) => close(value, expected[i])),
      `Detector ${key} differs from the recorded geometry`,
    );
  }
  invariant(
    data.detector.pixel_spacing_mm.every((value) => value > 0) &&
      data.detector.source_mm[1] > data.detector.centre_mm[1],
    "Invalid detector spacing or beam direction",
  );
  object(
    data.reference,
    [
      "projection",
      "channels_f32",
      "translation_limit",
      "rotation_limit",
      "display",
      "fd_status_u8",
    ],
    [],
    "reference",
  );
  object(
    data.reference.display,
    [
      "projection",
      "sensitivities",
      "asinh_softening_fraction",
      "projection_max",
    ],
    [],
    "display mapping",
  );
  invariant(
    data.reference.display.projection === "log1p" &&
      data.reference.display.sensitivities === "asinh" &&
      data.reference.display.asinh_softening_fraction === 0.02 &&
      data.reference.display.projection_max === configuration.n0,
    "Unsupported or inconsistent display mapping",
  );
  for (const key of ["translation_limit", "rotation_limit"])
    invariant(
      Number.isFinite(data.reference[key]) && data.reference[key] >= 0,
      "Invalid sensitivity scale",
    );
  const png = new Set([filename(data.reference.projection, ".png")]);
  const floats = new Set([filename(data.reference.channels_f32, ".f32")]);
  filename(data.reference.fd_status_u8, ".u8");
  invariant(
    Array.isArray(data.sensitivities) && data.sensitivities.length === 6,
    "Exactly six ordered sensitivity channels are required",
  );
  data.sensitivities.forEach((channel, i) => {
    object(
      channel,
      ["id", "label", "unit", "image"],
      [],
      "sensitivity channel",
    );
    invariant(
      channel.id === PARAMETERS[i] &&
        channel.unit === `counts/${i < 3 ? "mm" : "rad"}`,
      "Sensitivity channel order or units do not match",
    );
    text(channel.label, "channel label");
    filename(channel.image, ".png");
    invariant(
      !png.has(channel.image),
      "Sensitivity images must have distinct filenames",
    );
    png.add(channel.image);
  });
  const expectedCoordinates = configuredCoordinates(configuration);
  invariant(
    Array.isArray(data.poses) &&
      data.poses.length === expectedCoordinates.length,
    `Recorded pose count must match the configured coordinate sweep (${expectedCoordinates.length})`,
  );
  const ids = new Set(),
    coordinates = new Set(),
    poseImages = new Set(),
    poseFloats = new Set();
  for (const pose of data.poses) {
    object(
      pose,
      [
        "id",
        "label",
        "parameter",
        "value",
        "matrix",
        "projection",
        "projection_f32",
      ],
      [],
      "recorded pose",
    );
    text(pose.id, "pose ID");
    text(pose.label, "pose label");
    invariant(
      !ids.has(pose.id) && Number.isFinite(pose.value),
      "Duplicate pose ID or nonfinite pose value",
    );
    ids.add(pose.id);
    rigidMatrix(pose);
    const key = `${pose.parameter}:${pose.value}`;
    invariant(
      !coordinates.has(key),
      "Duplicate recorded pose coordinate/value",
    );
    coordinates.add(key);
    filename(pose.projection, ".png");
    filename(pose.projection_f32, ".f32");
    invariant(
      !poseImages.has(pose.projection) &&
        !poseFloats.has(pose.projection_f32) &&
        !floats.has(pose.projection_f32),
      "Recorded poses must have distinct PNG and float payloads",
    );
    if (pose.parameter === "reference")
      invariant(
        pose.projection === data.reference.projection,
        "Reference pose must use the reference projection",
      );
    else
      invariant(
        !png.has(pose.projection),
        "Pose projection aliases a different panel",
      );
    poseImages.add(pose.projection);
    poseFloats.add(pose.projection_f32);
  }
  invariant(
    sameSet([...coordinates], expectedCoordinates),
    "Recorded poses do not cover the configured coordinate sweep",
  );
  object(data.provenance, ["label", "url"], [], "provenance");
  text(data.provenance.label, "provenance label");
  invariant(
    data.provenance.url === "artifact.manifest.json",
    "Provenance must link to the curated manifest",
  );
  publicMetadata(data, "data.json");
  return {
    png: new Set([...png, ...poseImages]),
    floats: new Set([...floats, ...poseFloats]),
  };
}

function validateMeshes(data, configuration) {
  object(
    data,
    ["schema_version", "coordinate_system", "meshes", "attribution"],
    [],
    "meshes.json",
  );
  invariant(
    data.schema_version === 1 &&
      data.coordinate_system === "RAS-mm-pivot-centred",
    "Unexpected mesh coordinate system",
  );
  invariant(
    Array.isArray(data.meshes) && data.meshes.length === 3,
    "Exactly three attributed pelvic meshes are required",
  );
  const ids = [];
  for (const mesh of data.meshes) {
    object(mesh, ["id", "label", "positions", "indices"], [], "mesh");
    ids.push(mesh.id);
    text(mesh.label, "mesh label");
    invariant(
      Array.isArray(mesh.positions) &&
        mesh.positions.length >= 9 &&
        mesh.positions.length <= 6_000_000 &&
        mesh.positions.length % 3 === 0 &&
        mesh.positions.every(
          (value) => Number.isFinite(value) && Math.abs(value) <= 2000,
        ),
      "Invalid finite mesh positions",
    );
    invariant(
      Array.isArray(mesh.indices) &&
        mesh.indices.length >= 3 &&
        mesh.indices.length <= 12_000_000 &&
        mesh.indices.length % 3 === 0 &&
        mesh.indices.every(
          (value) =>
            Number.isInteger(value) &&
            value >= 0 &&
            value < mesh.positions.length / 3,
        ),
      "Invalid mesh triangle indices",
    );
  }
  invariant(
    sameSet(ids, ["sacrum", "left-hip", "right-hip"]),
    "Unexpected or duplicate pelvic mesh IDs",
  );
  object(
    data.attribution,
    ["title", "creator", "source", "licence", "changes", "label_sha256"],
    [],
    "mesh attribution",
  );
  for (const key of ["title", "creator", "changes"])
    text(data.attribution[key], `mesh attribution ${key}`);
  invariant(
    data.attribution.source === "https://doi.org/10.5281/zenodo.10047292" &&
      data.attribution.licence ===
        "https://creativecommons.org/licenses/by/4.0/",
    "Missing verified mesh dataset or licence URL",
  );
  invariant(
    /^[a-f0-9]{64}$/.test(data.attribution.label_sha256) &&
      data.attribution.label_sha256 === configuration.label_sha256,
    "Mesh input hash differs from the recorded label input",
  );
  publicMetadata(data, "meshes.json");
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const value of bytes) {
    crc ^= value;
    for (let i = 0; i < 8; i++) crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function validatePng(bytes, name, expectedWidth, expectedHeight) {
  invariant(
    bytes.length >= 45 &&
      bytes
        .subarray(0, 8)
        .equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])),
    `Invalid PNG signature: ${name}`,
  );
  let offset = 8,
    width,
    height,
    channels,
    ended = false;
  const compressed = [];
  while (offset < bytes.length) {
    invariant(offset + 12 <= bytes.length, `Truncated PNG chunk: ${name}`);
    const size = bytes.readUInt32BE(offset),
      type = bytes.toString("ascii", offset + 4, offset + 8),
      end = offset + 12 + size;
    invariant(
      end <= bytes.length &&
        crc32(bytes.subarray(offset + 4, end - 4)) ===
          bytes.readUInt32BE(end - 4),
      `PNG chunk checksum mismatch: ${name}`,
    );
    if (offset === 8) {
      invariant(type === "IHDR" && size === 13, `Missing PNG header: ${name}`);
      width = bytes.readUInt32BE(offset + 8);
      height = bytes.readUInt32BE(offset + 12);
      const colour = bytes[offset + 17];
      channels = { 0: 1, 2: 3, 4: 2, 6: 4 }[colour];
      invariant(
        width > 0 &&
          height > 0 &&
          width * height <= 16_777_216 &&
          bytes[offset + 16] === 8 &&
          channels &&
          bytes[offset + 18] === 0 &&
          bytes[offset + 19] === 0 &&
          bytes[offset + 20] === 0,
        `Unsupported PNG dimensions or encoding: ${name}`,
      );
      if (expectedWidth !== undefined)
        invariant(
          width === expectedWidth && height === expectedHeight,
          `PNG dimensions disagree with data.json: ${name}`,
        );
    } else if (type === "IDAT")
      compressed.push(bytes.subarray(offset + 8, end - 4));
    else if (type === "IEND") {
      invariant(
        size === 0 && end === bytes.length && compressed.length > 0,
        `Invalid PNG ending: ${name}`,
      );
      ended = true;
    } else
      invariant(
        ["sRGB", "gAMA", "pHYs"].includes(type),
        `Unexpected PNG chunk or private metadata: ${name}/${type}`,
      );
    offset = end;
  }
  invariant(ended, `PNG missing IEND: ${name}`);
  const stride = width * channels + 1,
    raw = inflateSync(Buffer.concat(compressed), {
      maxOutputLength: stride * height,
    });
  invariant(
    raw.length === stride * height,
    `PNG pixel byte count mismatch: ${name}`,
  );
  for (let row = 0; row < height; row++)
    invariant(raw[row * stride] <= 4, `Invalid PNG row filter: ${name}`);
}

function floatPlanes(bytes, pixels, channels, n0, label) {
  invariant(
    bytes.length === pixels * channels * 4,
    `Float plane byte count mismatch: ${label}`,
  );
  const maxima = Array(channels).fill(0);
  for (let index = 0; index < pixels * channels; index++) {
    const value = bytes.readFloatLE(index * 4),
      channel = Math.floor(index / pixels);
    invariant(Number.isFinite(value), `Nonfinite float plane value: ${label}`);
    if (channel === 0)
      invariant(
        value >= 0 && value <= n0 + 1e-4,
        `Expected counts outside [0,n0]: ${label}`,
      );
    maxima[channel] = Math.max(maxima[channel], Math.abs(value));
  }
  return maxima;
}

export async function importPoseSensitivityArtifacts({
  root = process.cwd(),
  runDir,
} = {}) {
  invariant(
    typeof runDir === "string" && path.isAbsolute(runDir),
    "--run-dir must be an absolute private run directory",
  );
  root = path.resolve(root);
  invariant(
    (await lstat(runDir)).isDirectory(),
    "Run directory must be a real directory",
  );
  const raw = parseUniqueJson(
    (await readRegularFile(runDir, "run.json")).toString("utf8"),
    "run.json",
  );
  invariant(
    raw.schema_version === 2 &&
      raw.status === "complete" &&
      raw.sources_unchanged === true &&
      raw.recorded_files_unchanged === true,
    "Only a complete version-2 run with unchanged snapshots can be imported",
  );
  const configurationBytes = await readRegularFile(
    runDir,
    "configuration.json",
  );
  invariant(
    sha(configurationBytes) === raw.configuration_sha256 &&
      sha(configurationBytes) === raw.output_sha256?.["configuration.json"] &&
      isDeepStrictEqual(
        parseUniqueJson(
          configurationBytes.toString("utf8"),
          "configuration.json",
        ),
        raw.configuration,
      ),
    "Run effective configuration SHA256 or payload mismatch",
  );
  const configuration = raw.configuration;
  invariant(
    isObject(configuration) &&
      Number.isFinite(configuration.n0) &&
      configuration.n0 > 0,
    "Invalid incident count configuration",
  );
  const metadata = raw.metadata ?? {};
  invariant(
    /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(metadata.source_commit ?? ""),
    "A full source_commit is required",
  );
  timestamp(raw.started_utc, "started_utc");
  timestamp(raw.finished_utc, "finished_utc");
  const expectedSources = [
    ...(await walkFiles(path.join(root, "python/dpt")))
      .filter((name) => name.endsWith(".py"))
      .map((name) => `python/dpt/${name}`),
    ...EXTRA_SOURCES,
    "pyproject.toml",
    "uv.lock",
  ];
  invariant(
    isObject(raw.source_sha256) &&
      sameSet(Object.keys(raw.source_sha256), [
        ...expectedSources,
        "configuration.json",
      ]),
    "Run source_sha256 must cover exactly the declared pose-sensitivity source files",
  );
  const sourceDigests = {};
  for (const file of [...expectedSources, "configuration.json"]) {
    const expected = raw.source_sha256[file];
    invariant(
      /^[a-f0-9]{64}$/.test(expected ?? ""),
      `Invalid source SHA256: ${file}`,
    );
    const snapshot = await readRegularFile(runDir, `sources/${file}`);
    invariant(
      sha(snapshot) === expected,
      `Run source snapshot SHA256 mismatch: ${file}`,
    );
    if (file === "configuration.json") {
      // The recorder serialises the effective configuration canonically; the
      // source snapshot preserves the original configuration file's bytes.
      invariant(
        isDeepStrictEqual(
          parseUniqueJson(
            snapshot.toString("utf8"),
            "configuration source snapshot",
          ),
          configuration,
        ),
        "Configuration source snapshot differs from the effective configuration",
      );
    } else
      invariant(
        sha(await readRegularFile(root, file)) === expected,
        `Current source SHA256 mismatch: ${file}`,
      );
    if (file !== "configuration.json") sourceDigests[file] = expected;
  }
  const files = new Map();
  async function load(name) {
    if (!files.has(name)) {
      const bytes = await readRegularFile(runDir, `figure/${name}`);
      invariant(
        sha(bytes) === raw.output_sha256?.[`figure/${name}`],
        `Run output SHA256 mismatch: ${name}`,
      );
      files.set(name, bytes);
    }
    return files.get(name);
  }
  const data = parseUniqueJson(
    (await load("data.json")).toString("utf8"),
    "data.json",
  );
  const inventory = validateData(data, configuration);
  const meshes = parseUniqueJson(
    (await load("meshes.json")).toString("utf8"),
    "meshes.json",
  );
  validateMeshes(meshes, configuration);
  const validation = parseUniqueJson(
    (await load("validation.json")).toString("utf8"),
    "validation.json",
  );
  object(
    validation,
    [
      "schema_version",
      "status",
      "checks",
      "budgets",
      "samples_per_ray",
      "quadrature_refinements",
      "display_quadrature",
      "finite_differences",
      "independent_rays",
      "contraction_relative_error",
      "mixed_taylor",
      "fd_unresolved_pixels_per_axis",
      "fd_pixel_tolerance",
      "limitations",
    ],
    [],
    "validation.json",
  );
  invariant(
    isObject(validation) &&
      validation.schema_version === 1 &&
      validation.status === "passed" &&
      isObject(validation.checks) &&
      CHECKS.every((key) => validation.checks[key] === true) &&
      Object.values(validation.checks).every((value) => value === true),
    "Validation must pass every required scientific check",
  );
  publicMetadata(validation, "validation.json");
  invariant(
    isDeepStrictEqual(validation.budgets, configuration.acceptance) &&
      validation.samples_per_ray === configuration.samples_per_ray &&
      Number.isInteger(validation.samples_per_ray) &&
      validation.samples_per_ray > 0,
    "Validation budgets or quadrature differ from the recorded configuration",
  );
  const budgets = validation.budgets;
  invariant(
    isObject(budgets) &&
      Object.values(budgets).every(
        (value) => Number.isFinite(value) && value > 0,
      ),
    "Invalid numerical acceptance budgets",
  );
  invariant(
    Array.isArray(validation.quadrature_refinements) &&
      validation.quadrature_refinements.length >= 2,
    "At least two quadrature refinements are required",
  );
  for (const row of validation.quadrature_refinements.slice(-2)) {
    invariant(
      Number.isInteger(row.samples) &&
        row.samples > 0 &&
        Number.isFinite(row.count_max) &&
        row.count_max >= 0 &&
        row.count_max <= budgets.count_max &&
        Number.isFinite(row.count_p99) &&
        row.count_p99 >= 0 &&
        row.count_p99 <= budgets.count_p99 &&
        finiteVector(row.jacobian_relative_l2, 6) &&
        row.jacobian_relative_l2.every(
          (value) => value >= 0 && value <= budgets.jacobian_relative_l2,
        ) &&
        finiteVector(row.jacobian_normalised_max, 6) &&
        row.jacobian_normalised_max.every(
          (value) => value >= 0 && value <= budgets.jacobian_normalised_max,
        ),
      "Recorded quadrature results fail their declared budgets",
    );
  }
  invariant(
    validation.quadrature_refinements.at(-1).samples ===
      configuration.samples_per_ray,
    "Last quadrature refinement must be the exported sample count",
  );
  invariant(
    Array.isArray(configuration.sampling_sweep) &&
      configuration.sampling_sweep.length >= 2 &&
      configuration.sampling_sweep.every(
        (samples, index, sweep) =>
          Number.isInteger(samples) &&
          samples > 0 &&
          (index === 0 || samples > sweep[index - 1]),
      ) &&
      configuration.sampling_sweep.at(-1) === configuration.samples_per_ray,
    "Invalid configured quadrature sampling sweep",
  );
  invariant(
    Array.isArray(validation.display_quadrature) &&
      validation.display_quadrature.length === data.poses.length,
    "Display quadrature must cover every configured pose",
  );
  const displayCoordinates = new Set();
  for (const row of validation.display_quadrature) {
    object(
      row,
      [
        "parameter",
        "value",
        "coarse_samples",
        "fine_samples",
        "count_max",
        "count_p99",
      ],
      [],
      "display quadrature row",
    );
    invariant(
      Number.isFinite(row.value) &&
        row.coarse_samples === configuration.sampling_sweep.at(-2) &&
        row.fine_samples === configuration.samples_per_ray &&
        Number.isFinite(row.count_max) &&
        row.count_max >= 0 &&
        row.count_max <= budgets.count_max &&
        Number.isFinite(row.count_p99) &&
        row.count_p99 >= 0 &&
        row.count_p99 <= row.count_max &&
        row.count_p99 <= budgets.count_p99,
      "Display quadrature results fail their configured samples or count budgets",
    );
    const coordinate = `${row.parameter}:${row.value}`;
    invariant(
      !displayCoordinates.has(coordinate),
      "Duplicate display quadrature coordinate/value",
    );
    displayCoordinates.add(coordinate);
  }
  invariant(
    sameSet(
      [...displayCoordinates],
      data.poses.map((pose) => `${pose.parameter}:${pose.value}`),
    ),
    "Display quadrature does not cover the configured coordinate sweep",
  );
  invariant(
    Array.isArray(validation.finite_differences) &&
      validation.finite_differences.length === 6,
    "All six finite-difference records are required",
  );
  validation.finite_differences.forEach((row, i) => {
    invariant(
      row.axis === PARAMETERS[i] &&
        Number.isFinite(row.best_relative_l2) &&
        row.best_relative_l2 >= 0 &&
        row.best_relative_l2 <= budgets.discrete_fd_relative_l2 &&
        Array.isArray(row.steps) &&
        row.steps.length >= 2 &&
        row.steps.every(
          (step) =>
            Number.isFinite(step.step) &&
            step.step > 0 &&
            Number.isFinite(step.relative_l2) &&
            step.relative_l2 >= 0,
        ) &&
        row.best_relative_l2 ===
          Math.min(...row.steps.map((step) => step.relative_l2)),
      "Finite-difference results fail their declared budgets or step evidence",
    );
  });
  invariant(
    Array.isArray(validation.independent_rays) &&
      validation.independent_rays.length >= 7,
    "Independent reference rays are missing",
  );
  for (const row of validation.independent_rays) {
    invariant(
      Number.isInteger(row.row) &&
        row.row >= 0 &&
        row.row < data.height &&
        Number.isInteger(row.column) &&
        row.column >= 0 &&
        row.column < data.width &&
        Number.isFinite(row.count_discrete_error) &&
        row.count_discrete_error >= 0 &&
        row.count_discrete_error <= budgets.reference_count_absolute &&
        Number.isFinite(row.count_quadrature_error) &&
        row.count_quadrature_error >= 0 &&
        row.count_quadrature_error <= budgets.count_max &&
        Array.isArray(row.derivatives) &&
        row.derivatives.length === 6 &&
        row.derivatives.every(
          (derivative, i) =>
            derivative.axis === PARAMETERS[i] &&
            Number.isFinite(derivative.best_normalised_error) &&
            derivative.best_normalised_error >= 0 &&
            derivative.best_normalised_error <=
              budgets.reference_gradient_relative,
        ),
      "Independent reference results fail their declared budgets",
    );
  }
  invariant(
    finiteVector(validation.contraction_relative_error, 6) &&
      validation.contraction_relative_error.every(
        (value) => value >= 0 && value <= budgets.contraction_relative,
      ),
    "Contraction results fail their declared budget",
  );
  invariant(
    finiteVector(validation.fd_unresolved_pixels_per_axis, 6) &&
      validation.fd_unresolved_pixels_per_axis.every(
        (value) =>
          Number.isInteger(value) &&
          value >= 0 &&
          value <= data.width * data.height,
      ),
    "Invalid per-axis unresolved pixel counts",
  );
  invariant(
    isDeepStrictEqual(validation.fd_pixel_tolerance, {
      relative: 0.05,
      axis_rms_fraction: 0.002,
    }),
    "Unexpected per-pixel finite-difference status criterion",
  );
  invariant(
    Array.isArray(validation.mixed_taylor) &&
      validation.mixed_taylor.length >= 4 &&
      validation.mixed_taylor.every(
        (row) =>
          Number.isFinite(row.step) &&
          row.step !== 0 &&
          Number.isFinite(row.remainder_l2) &&
          row.remainder_l2 >= 0 &&
          Number.isFinite(row.remainder_per_step) &&
          row.remainder_per_step >= 0,
      ),
    "Mixed Taylor evidence is missing or invalid",
  );
  const expectedFiles = [
    "data.json",
    "meshes.json",
    "validation.json",
    "panel.png",
    data.reference.fd_status_u8,
    ...inventory.png,
    ...inventory.floats,
  ];
  invariant(
    new Set(expectedFiles).size === expectedFiles.length,
    "Figure filenames overlap reserved metadata",
  );
  invariant(
    sameSet(await walkFiles(path.join(runDir, "figure")), expectedFiles),
    "Figure directory must contain exactly the referenced files",
  );
  invariant(
    sameSet(
      Object.keys(raw.output_sha256 ?? {}).filter((name) =>
        name.startsWith("figure/"),
      ),
      expectedFiles.map((name) => `figure/${name}`),
    ),
    "Recorded figure output inventory differs from its payload references",
  );
  for (const name of expectedFiles) await load(name);
  for (const name of inventory.png)
    validatePng(files.get(name), name, data.width, data.height);
  validatePng(files.get("panel.png"), "panel.png");
  const pixels = data.width * data.height;
  const statuses = files.get(data.reference.fd_status_u8),
    statusCounts = [0, 0, 0, 0, 0, 0];
  invariant(statuses.length === pixels * 6, "FD status byte count mismatch");
  for (let i = 0; i < statuses.length; i++) {
    invariant(
      statuses[i] === 0 || statuses[i] === 1,
      "FD status payload contains an unknown code",
    );
    statusCounts[i % 6] += statuses[i];
  }
  invariant(
    isDeepStrictEqual(statusCounts, validation.fd_unresolved_pixels_per_axis),
    "FD status counts disagree with validation.json",
  );
  const maxima = floatPlanes(
    files.get(data.reference.channels_f32),
    pixels,
    7,
    configuration.n0,
    data.reference.channels_f32,
  );
  for (const [key, value] of [
    ["translation_limit", Math.max(...maxima.slice(1, 4))],
    ["rotation_limit", Math.max(...maxima.slice(4, 7))],
  ])
    invariant(
      Math.abs(data.reference[key] - value) <= Math.max(1e-7, value * 3e-7),
      `Sensitivity colour range differs from recorded values: ${key}`,
    );
  for (const pose of data.poses)
    floatPlanes(
      files.get(pose.projection_f32),
      pixels,
      1,
      configuration.n0,
      pose.projection_f32,
    );
  const reference = data.poses.find((pose) => pose.parameter === "reference");
  invariant(
    files
      .get(data.reference.channels_f32)
      .subarray(0, pixels * 4)
      .equals(files.get(reference.projection_f32)),
    "Reference projection differs from the first inspector channel",
  );
  const publicRuntime = Object.fromEntries(
    RUNTIME_KEYS.filter((key) => Object.hasOwn(metadata, key)).map((key) => [
      key,
      metadata[key],
    ]),
  );
  publicMetadata(publicRuntime, "runtime");
  files.set(
    "run.json",
    jsonBytes({
      schema_version: 2,
      status: "complete",
      started_utc: raw.started_utc,
      finished_utc: raw.finished_utc,
      source_commit: metadata.source_commit,
      source_sha256: sourceDigests,
      configuration: {
        width: data.width,
        height: data.height,
        n0: configuration.n0,
        samples_per_ray: configuration.samples_per_ray,
        image_sha256: configuration.image_sha256,
        label_sha256: configuration.label_sha256,
        pivot_ras_mm: data.pivot_ras_mm,
        display_translation_mm: configuration.display_translation_mm,
        display_rotation_rad: configuration.display_rotation_rad,
      },
      metadata: publicRuntime,
      output_sha256: Object.fromEntries(
        [...files].map(([name, bytes]) => [name, sha(bytes)]),
      ),
    }),
  );
  const manifest = {
    schema_version: 1,
    id: "introduction-pose-sensitivity",
    chapter: "introduction",
    experiment: "pose-sensitivity",
    source_commit: metadata.source_commit,
    generator: GENERATOR,
    mode: "deterministic-precomputed-sweep",
    device: metadata.device_name ?? null,
    cuda_version:
      metadata.cuda_toolkit == null
        ? null
        : [].concat(metadata.cuda_toolkit).join("."),
    warp_version: metadata.warp ?? null,
    precision:
      "FP32 field/count storage; FP64 pose derivatives; FP32 display planes",
    parameters: {
      width: data.width,
      height: data.height,
      samples_per_ray: configuration.samples_per_ray,
      n0: configuration.n0,
      input_image_sha256: configuration.image_sha256,
      input_label_sha256: configuration.label_sha256,
      display_translation_mm: configuration.display_translation_mm,
      display_rotation_rad: configuration.display_rotation_rad,
    },
    outputs: [...files.keys()].map((name) => `${TARGET}/${name}`),
    stochastic: false,
    created_at: raw.finished_utc,
    source_digests: sourceDigests,
    output_digests: Object.fromEntries(
      [...files].map(([name, bytes]) => [`${TARGET}/${name}`, sha(bytes)]),
    ),
    notes:
      "Recorded canonical CUDA count projections and pose derivatives of a synthetic water-equivalent field. Display meshes are attributed augmented conditioning-mask derivatives. Raw volumes and private run metadata remain excluded.",
  };
  assertSchema(
    manifest,
    JSON.parse(
      await readFile(
        path.join(root, "tools/public/schemas/generated-artifact.schema.json"),
        "utf8",
      ),
    ),
    "pose-sensitivity manifest",
  );
  files.set("artifact.manifest.json", jsonBytes(manifest));
  await assertNoSymlinks(root, TARGET);
  const destination = path.join(root, TARGET);
  try {
    const existing = await walkFiles(destination);
    invariant(
      sameSet(existing, [...files.keys()]),
      "Existing pose-sensitivity artefact inventory differs",
    );
    for (const [name, bytes] of files)
      invariant(
        bytes.equals(await readRegularFile(destination, name)),
        `Existing pose-sensitivity artefact differs: ${name}`,
      );
    return { target: TARGET, files: files.size, status: "unchanged" };
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  await mkdir(path.dirname(destination), { recursive: true });
  const temporary = await mkdtemp(
    path.join(path.dirname(destination), ".pose-sensitivity-import-"),
  );
  try {
    for (const [name, bytes] of files)
      await writeFile(path.join(temporary, name), bytes);
    await rename(temporary, destination);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
  return { target: TARGET, files: files.size, status: "imported" };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    invariant(
      process.argv.length === 4 && process.argv[2] === "--run-dir",
      "Usage: node tools/public/import-pose-sensitivity-artifacts.mjs --run-dir /absolute/private/run",
    );
    process.stdout.write(
      `${JSON.stringify(await importPoseSensitivityArtifacts({ runDir: process.argv[3] }))}\n`,
    );
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
