import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { deflateSync } from "node:zlib";

import { importPoseSensitivityArtifacts } from "./import-pose-sensitivity-artifacts.mjs";
import { validateGeneratedArtifacts } from "./validate-generated-artifacts.mjs";

const TARGET = "public/generated/introduction-pose-sensitivity";
const AXES = ["tx", "ty", "tz", "rx", "ry", "rz"];
const EXPANDED_SWEEP = {
  display_translation_mm: [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
  display_rotation_rad: [
    -0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05,
  ],
};
const SOURCE_FILES = [
  "python/dpt/projection.py",
  "python/dpt/kernels/projection.py",
  ...[
    "run.py",
    "common.py",
    "prepare.py",
    "mesh.py",
    "config.json",
    "requirements-volume.txt",
  ].map((name) => `experiments/pose-sensitivity/${name}`),
  "pyproject.toml",
  "uv.lock",
];
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const json = (value) => Buffer.from(JSON.stringify(value));
const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function png(width = 480, height = 256) {
  function chunk(type, data) {
    const bytes = Buffer.alloc(data.length + 12);
    bytes.writeUInt32BE(data.length);
    bytes.write(type, 4);
    data.copy(bytes, 8);
    let crc = -1;
    for (const byte of bytes.subarray(4, -4)) {
      crc ^= byte;
      for (let i = 0; i < 8; i++) crc = (crc >>> 1) ^ ((crc & 1) * 0xedb88320);
    }
    bytes.writeUInt32BE((crc ^ -1) >>> 0, bytes.length - 4);
    return bytes;
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  // Valid monochrome fixture; these bytes are never scientific evidence.
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", header),
    chunk("IDAT", deflateSync(Buffer.alloc((width + 1) * height))),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

async function fixture(t, sweep = {}) {
  const base = await mkdtemp(path.join(os.tmpdir(), "dpt-pose-import-test-"));
  t.after(() => rm(base, { recursive: true, force: true }));
  const root = path.join(base, "repo"),
    runDir = path.join(base, "private-run");
  await mkdir(path.join(root, "tools/public/schemas"), { recursive: true });
  await copyFile(
    new URL("./schemas/generated-artifact.schema.json", import.meta.url),
    path.join(root, "tools/public/schemas/generated-artifact.schema.json"),
  );
  await mkdir(path.join(runDir, "figure"), { recursive: true });
  const sources = {};
  for (const name of SOURCE_FILES) {
    const bytes = Buffer.from(`Synthetic source fixture ${name}\n`);
    for (const destination of [
      path.join(root, name),
      path.join(runDir, "sources", name),
    ]) {
      await mkdir(path.dirname(destination), { recursive: true });
      await writeFile(destination, bytes);
    }
    sources[name] = sha(bytes);
  }
  const configuration = {
    schema_version: 1,
    width: 480,
    height: 256,
    n0: 1000,
    image_sha256: "1".repeat(64),
    label_sha256: "2".repeat(64),
    mu_water_mm_inv: 0.01837,
    energy_kev: 80,
    pivot_ras_mm: [8.5, 65, -246],
    source_mm: [0, 750, 0],
    detector_centre_mm: [0, -450, 0],
    detector_width_mm: 600,
    detector_height_mm: 320,
    samples_per_ray: 8192,
    sampling_sweep: [1024, 2048, 4096, 8192],
    display_translation_mm: [-1, -0.5, 0, 0.5, 1],
    display_rotation_rad: [-0.01, -0.005, 0, 0.005, 0.01],
    ...sweep,
    acceptance: {
      count_max: 0.25,
      count_p99: 0.05,
      jacobian_relative_l2: 0.02,
      jacobian_normalised_max: 0.05,
      discrete_fd_relative_l2: 0.02,
      reference_count_absolute: 0.0002,
      reference_gradient_relative: 0.001,
      contraction_relative: 1e-10,
    },
  };
  const data = {
    schema_version: 1,
    id: "introduction-pose-sensitivity",
    title: "Synthetic importer fixture",
    anatomy_label: "Test arrays",
    signal_label: "Expected counts",
    signal_unit: "counts",
    width: 480,
    height: 256,
    detector: {
      source_mm: [0, 750, 0],
      centre_mm: [0, -450, 0],
      u_axis: [1, 0, 0],
      v_axis: [0, 0, -1],
      pixel_spacing_mm: [1.25, 1.25],
    },
    pivot_ras_mm: configuration.pivot_ras_mm,
    reference: {
      projection: "projection.png",
      channels_f32: "reference.f32",
      translation_limit: 0,
      rotation_limit: 0,
      fd_status_u8: "fd-status.u8",
      display: {
        projection: "log1p",
        sensitivities: "asinh",
        asinh_softening_fraction: 0.02,
        projection_max: 1000,
      },
    },
    sensitivities: AXES.map((id, i) => ({
      id,
      label: id,
      unit: `counts/${i < 3 ? "mm" : "rad"}`,
      image: `${id}.png`,
    })),
    poses: [
      {
        id: "reference",
        label: "Reference",
        parameter: "reference",
        value: 0,
        matrix: identity(),
        projection: "projection.png",
        projection_f32: "projection.f32",
      },
    ],
    provenance: {
      label: "Synthetic importer fixture",
      url: "artifact.manifest.json",
    },
  };
  for (const [axis, parameter] of AXES.entries()) {
    for (const [index, value] of (axis < 3
      ? configuration.display_translation_mm
      : configuration.display_rotation_rad
    ).entries()) {
      if (!value) continue;
      const matrix = identity();
      if (axis < 3) matrix[axis * 4 + 3] = value;
      else {
        const k = axis - 3,
          a = (k + 1) % 3,
          b = (k + 2) % 3;
        matrix[a * 4 + a] = matrix[b * 4 + b] = Math.cos(value);
        matrix[a * 4 + b] = -Math.sin(value);
        matrix[b * 4 + a] = Math.sin(value);
      }
      const id = `${parameter}-${index}`;
      data.poses.push({
        id,
        label: id,
        parameter,
        value,
        matrix,
        projection: `${id}.png`,
        projection_f32: `${id}.f32`,
      });
    }
  }
  const meshes = {
    schema_version: 1,
    coordinate_system: "RAS-mm-pivot-centred",
    meshes: ["sacrum", "left-hip", "right-hip"].map((id) => ({
      id,
      label: id,
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
    })),
    attribution: {
      title: "TotalSegmentator dataset v2",
      creator: "Jakob Wasserthal and University Hospital Basel",
      source: "https://doi.org/10.5281/zenodo.10047292",
      licence: "https://creativecommons.org/licenses/by/4.0/",
      changes: "Synthetic fixture for importer behaviour only",
      label_sha256: configuration.label_sha256,
    },
  };
  const validation = {
    schema_version: 1,
    status: "passed",
    checks: Object.fromEntries(
      [
        "input_hashes",
        "finite",
        "quadrature_counts",
        "quadrature_derivatives",
        "display_quadrature",
        "discrete_fd",
        "independent_reference",
        "contractions",
      ].map((key) => [key, true]),
    ),
    budgets: configuration.acceptance,
    samples_per_ray: 8192,
    quadrature_refinements: [4096, 8192].map((samples) => ({
      samples,
      count_max: 0,
      count_p99: 0,
      jacobian_relative_l2: [0, 0, 0, 0, 0, 0],
      jacobian_normalised_max: [0, 0, 0, 0, 0, 0],
    })),
    display_quadrature: data.poses.map(({ parameter, value }) => ({
      parameter,
      value,
      coarse_samples: 4096,
      fine_samples: 8192,
      count_max: 0,
      count_p99: 0,
    })),
    finite_differences: AXES.map((axis) => ({
      axis,
      steps: [
        { step: 0.1, relative_l2: 0 },
        { step: 0.01, relative_l2: 0 },
      ],
      best_relative_l2: 0,
    })),
    independent_rays: Array.from({ length: 7 }, (_, row) => ({
      row,
      column: row,
      count_discrete_error: 0,
      count_quadrature_error: 0,
      derivatives: AXES.map((axis) => ({ axis, best_normalised_error: 0 })),
    })),
    contraction_relative_error: [0, 0, 0, 0, 0, 0],
    mixed_taylor: [-1, -0.1, 0.1, 1].map((step) => ({
      step,
      remainder_l2: 0,
      remainder_per_step: 0,
    })),
    fd_unresolved_pixels_per_axis: [0, 0, 0, 0, 0, 0],
    fd_pixel_tolerance: { relative: 0.05, axis_rms_fraction: 0.002 },
    limitations: "Synthetic test fixture, not numerical evidence",
  };
  const files = new Map([
    ["data.json", json(data)],
    ["meshes.json", json(meshes)],
    ["validation.json", json(validation)],
    ["panel.png", png()],
    ["reference.f32", Buffer.alloc(480 * 256 * 7 * 4)],
    ["fd-status.u8", Buffer.alloc(480 * 256 * 6)],
  ]);
  for (const name of new Set([
    data.reference.projection,
    ...data.sensitivities.map((channel) => channel.image),
    ...data.poses.map((pose) => pose.projection),
  ]))
    files.set(name, png());
  for (const pose of data.poses)
    files.set(pose.projection_f32, Buffer.alloc(480 * 256 * 4));
  const configurationBytes = json(configuration);
  const configurationSource = Buffer.from(
    `${JSON.stringify(configuration, null, 2)}\n`,
  );
  await writeFile(path.join(runDir, "configuration.json"), configurationBytes);
  await writeFile(
    path.join(runDir, "sources/configuration.json"),
    configurationSource,
  );
  sources["configuration.json"] = sha(configurationSource);
  const record = {
    schema_version: 2,
    status: "complete",
    sources_unchanged: true,
    recorded_files_unchanged: true,
    started_utc: "2026-09-10T10:00:00Z",
    finished_utc: "2026-09-10T11:00:00Z",
    source_sha256: sources,
    configuration,
    configuration_sha256: sha(configurationBytes),
    output_sha256: { "configuration.json": sha(configurationBytes) },
    metadata: {
      source_commit: "a".repeat(40),
      warp: "1.17.0",
      device: "cuda:0",
      device_name: "Synthetic fixture device",
      private_input_records: { preparation: "/private/input" },
    },
  };
  const save = () => writeFile(path.join(runDir, "run.json"), json(record));
  async function payload(name, value) {
    const bytes = Buffer.isBuffer(value) ? value : json(value);
    await writeFile(path.join(runDir, "figure", name), bytes);
    record.output_sha256[`figure/${name}`] = sha(bytes);
    await save();
  }
  async function configure(sweep) {
    Object.assign(configuration, sweep);
    const bytes = json(configuration);
    const source = Buffer.from(`${JSON.stringify(configuration, null, 2)}\n`);
    await writeFile(path.join(runDir, "configuration.json"), bytes);
    await writeFile(path.join(runDir, "sources/configuration.json"), source);
    record.configuration_sha256 = sha(bytes);
    record.output_sha256["configuration.json"] = sha(bytes);
    record.source_sha256["configuration.json"] = sha(source);
    await save();
  }
  for (const [name, bytes] of files) await payload(name, bytes);
  return {
    root,
    runDir,
    record,
    data,
    meshes,
    validation,
    payload,
    save,
    configure,
  };
}

async function noOutput(root) {
  await assert.rejects(readdir(path.join(root, TARGET)), /ENOENT/);
}

test("imports only validated flat figure assets and public metadata atomically", async (t) => {
  const f = await fixture(t),
    before = await readFile(path.join(f.runDir, "run.json"));
  await writeFile(
    path.join(f.runDir, "raw-private.nii.gz"),
    "not a public payload",
  );
  assert.equal((await importPoseSensitivityArtifacts(f)).status, "imported");
  const files = await readdir(path.join(f.root, TARGET));
  assert.ok(
    files.includes("fd-status.u8") && files.includes("artifact.manifest.json"),
  );
  const publicRun = await readFile(
    path.join(f.root, TARGET, "run.json"),
    "utf8",
  );
  assert.ok(
    !publicRun.includes("private_input_records") &&
      !publicRun.includes("/private/"),
  );
  assert.deepEqual(await readFile(path.join(f.runDir, "run.json")), before);
  assert.equal((await importPoseSensitivityArtifacts(f)).status, "unchanged");
  const checked = await validateGeneratedArtifacts({ root: f.root });
  assert.equal(checked.manifests, 1);
  assert.equal(checked.sources, SOURCE_FILES.length);
});

test("imports all 55 expanded poses and preserves their configured sweep in provenance", async (t) => {
  const f = await fixture(t, EXPANDED_SWEEP);
  assert.equal(f.data.poses.length, 55);
  assert.equal((await importPoseSensitivityArtifacts(f)).status, "imported");
  const directory = path.join(f.root, TARGET);
  assert.equal((await readdir(directory)).length, 124);
  const imported = JSON.parse(
    await readFile(path.join(directory, "data.json"), "utf8"),
  );
  assert.deepEqual(imported.poses, f.data.poses);
  const publicRun = JSON.parse(
    await readFile(path.join(directory, "run.json"), "utf8"),
  );
  const manifest = JSON.parse(
    await readFile(path.join(directory, "artifact.manifest.json"), "utf8"),
  );
  for (const [field, values] of Object.entries(EXPANDED_SWEEP)) {
    assert.deepEqual(publicRun.configuration[field], values);
    assert.deepEqual(manifest.parameters[field], values);
  }
  assert.equal((await importPoseSensitivityArtifacts(f)).status, "unchanged");
  assert.equal(
    (await validateGeneratedArtifacts({ root: f.root })).manifests,
    1,
  );
});

test("rejects missing expanded endpoints and axes instead of accepting a smaller sweep", async (t) => {
  const f = await fixture(t, EXPANDED_SWEEP);
  const complete = [...f.data.poses];
  for (const incomplete of [
    complete.filter(
      (pose) => !(pose.parameter === "rz" && pose.value === 0.05),
    ),
    complete.filter((pose) => pose.parameter !== "tz"),
    complete.filter((pose) => pose.parameter !== "reference"),
  ]) {
    f.data.poses = incomplete;
    await f.payload("data.json", f.data);
    await assert.rejects(
      importPoseSensitivityArtifacts(f),
      /pose count.*configured coordinate sweep \(55\)/,
    );
    await noOutput(f.root);
  }
});

test("rejects a same-size but different coordinate grid even when its matrices are correct", async (t) => {
  const f = await fixture(t, EXPANDED_SWEEP);
  const pose = f.data.poses.find(
    (pose) => pose.parameter === "tx" && pose.value === -2,
  );
  pose.value = -2.5;
  pose.matrix[3] = -2.5;
  await f.payload("data.json", f.data);
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /do not cover the configured coordinate sweep/,
  );
  await noOutput(f.root);
});

test("rejects an old 25-pose record when the hashed configuration requires 55 poses", async (t) => {
  const f = await fixture(t);
  await f.configure(EXPANDED_SWEEP);
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /pose count.*configured coordinate sweep \(55\)/,
  );
  await noOutput(f.root);
});

test("rejects malformed, unsorted, duplicate and zero-free configured grids", async (t) => {
  const f = await fixture(t);
  for (const field of ["display_translation_mm", "display_rotation_rad"]) {
    for (const steps of [
      null,
      [],
      [-1, 0],
      [-1, 0, "1"],
      [-1, 0, null],
      [-1, 1, 0],
      [-1, 0, 0, 1],
      [-1, 0, 0.5, 0.5, 1],
      [-1, -0.5, 0.5, 1],
      [0, 0.5, 1],
      [-1, -0.5, 0],
    ]) {
      await f.configure({
        display_translation_mm: [-1, -0.5, 0, 0.5, 1],
        display_rotation_rad: [-0.01, -0.005, 0, 0.005, 0.01],
        [field]: steps,
      });
      await assert.rejects(
        importPoseSensitivityArtifacts(f),
        new RegExp(`${field} must`),
      );
      await noOutput(f.root);
    }
  }
});

test("requires positive display-quadrature acceptance and evidence for every expanded pose", async (t) => {
  const f = await fixture(t, EXPANDED_SWEEP);
  delete f.validation.checks.display_quadrature;
  await f.payload("validation.json", f.validation);
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /required scientific check/,
  );
  f.validation.checks.display_quadrature = true;
  const complete = f.validation.display_quadrature;
  delete f.validation.display_quadrature;
  await f.payload("validation.json", f.validation);
  await assert.rejects(importPoseSensitivityArtifacts(f), /required field/);
  f.validation.display_quadrature = complete.slice(0, 1);
  await f.payload("validation.json", f.validation);
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /Display quadrature must cover every configured pose/,
  );
  f.validation.display_quadrature = [...complete.slice(0, -1), complete[0]];
  await f.payload("validation.json", f.validation);
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /Duplicate display quadrature/,
  );
  f.validation.display_quadrature = complete.map((row, index) =>
    index === complete.length - 1 ? { ...row, value: 0.06 } : row,
  );
  await f.payload("validation.json", f.validation);
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /Display quadrature does not cover/,
  );
  await noOutput(f.root);
});

test("rejects failed or incorrectly sampled display quadrature despite a true acceptance flag", async (t) => {
  const f = await fixture(t, EXPANDED_SWEEP);
  const row = f.validation.display_quadrature.at(-1);
  const original = { ...row };
  for (const patch of [
    { coarse_samples: 2048 },
    { fine_samples: 4096 },
    { count_max: 0.25001 },
    { count_max: 0.1, count_p99: 0.05001 },
    { count_max: -1 },
    { count_p99: -1 },
    { count_max: null },
    { count_p99: "0" },
    { count_max: 0.01, count_p99: 0.02 },
  ]) {
    Object.assign(row, original, patch);
    await f.payload("validation.json", f.validation);
    await assert.rejects(
      importPoseSensitivityArtifacts(f),
      /Display quadrature results fail/,
    );
    await noOutput(f.root);
  }
});

test("rejects inconsistent configured sampling sweeps for display evidence", async (t) => {
  const f = await fixture(t);
  for (const sampling_sweep of [
    [8192],
    [4096, 4096, 8192],
    [4096, 2048, 8192],
    [1024, 2048, 4096],
    [1024, 4096.5, 8192],
    [0, 4096, 8192],
  ]) {
    await f.configure({ sampling_sweep });
    await assert.rejects(
      importPoseSensitivityArtifacts(f),
      /Invalid configured quadrature sampling sweep/,
    );
    await noOutput(f.root);
  }
});

test("rejects incomplete runs and missing positive validation flags", async (t) => {
  const f = await fixture(t);
  for (const field of ["sources_unchanged", "recorded_files_unchanged"]) {
    f.record[field] = "true";
    await f.save();
    await assert.rejects(
      importPoseSensitivityArtifacts(f),
      /complete version-2/,
    );
    f.record[field] = true;
  }
  await f.save();
  delete f.validation.checks.discrete_fd;
  await f.payload("validation.json", f.validation);
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /required scientific check/,
  );
  await noOutput(f.root);
});

test("rejects source drift, missing source coverage and tampered snapshots", async (t) => {
  const f = await fixture(t),
    file = SOURCE_FILES[0],
    original = await readFile(path.join(f.root, file));
  await writeFile(path.join(f.root, file), "changed");
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /Current source SHA256/,
  );
  await writeFile(path.join(f.root, file), original);
  const digest = f.record.source_sha256[file];
  delete f.record.source_sha256[file];
  await f.save();
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /exactly the declared/,
  );
  f.record.source_sha256[file] = digest;
  await f.save();
  await writeFile(path.join(f.runDir, "sources", file), "changed");
  await assert.rejects(importPoseSensitivityArtifacts(f), /snapshot SHA256/);
  await noOutput(f.root);
});

test("rejects changed configuration and altered output bytes", async (t) => {
  const f = await fixture(t);
  f.record.configuration.n0++;
  await f.save();
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /effective configuration/,
  );
  f.record.configuration.n0--;
  await f.save();
  await writeFile(path.join(f.runDir, "figure/projection.png"), "tampered");
  await assert.rejects(importPoseSensitivityArtifacts(f), /output SHA256/);
  await noOutput(f.root);
});

test("rejects undeclared figure files and source/output symlinks", async (t) => {
  const f = await fixture(t),
    extra = path.join(f.runDir, "figure/extra.json");
  await writeFile(extra, "{}");
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /exactly the referenced/,
  );
  await rm(extra);
  const file = path.join(f.runDir, "figure/projection.png");
  await rm(file);
  await symlink(path.join(f.runDir, "figure/tx.png"), file);
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /Symbolic links|symlinks/,
  );
  await noOutput(f.root);
});

test("rejects traversal URLs, unknown metadata and missing attribution", async (t) => {
  const f = await fixture(t);
  f.data.reference.projection = "../projection.png";
  await f.payload("data.json", f.data);
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /flat figure filename/,
  );
  f.data.reference.projection = "projection.png";
  f.data.private_path = "/private/volume";
  await f.payload("data.json", f.data);
  await assert.rejects(importPoseSensitivityArtifacts(f), /unexpected field/);
  delete f.data.private_path;
  await f.payload("data.json", f.data);
  delete f.meshes.attribution.licence;
  await f.payload("meshes.json", f.meshes);
  await assert.rejects(importPoseSensitivityArtifacts(f), /required field/);
  await noOutput(f.root);
});

test("rejects wrong units, matrix values and mesh triangles", async (t) => {
  const f = await fixture(t);
  f.data.sensitivities[3].unit = "counts/mm";
  await f.payload("data.json", f.data);
  await assert.rejects(importPoseSensitivityArtifacts(f), /units/);
  f.data.sensitivities[3].unit = "counts/rad";
  f.data.poses[1].matrix[3] = 99;
  await f.payload("data.json", f.data);
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /declared coordinate/,
  );
  f.data.poses[1].matrix[3] = -1;
  await f.payload("data.json", f.data);
  f.meshes.meshes[0].indices[0] = 3;
  await f.payload("meshes.json", f.meshes);
  await assert.rejects(importPoseSensitivityArtifacts(f), /triangle indices/);
  await noOutput(f.root);
});

test("rejects malformed PNGs, nonfinite/short floats and inconsistent scales", async (t) => {
  const f = await fixture(t);
  await f.payload("projection.png", png(479, 256));
  await assert.rejects(importPoseSensitivityArtifacts(f), /PNG dimensions/);
  await f.payload("projection.png", png());
  await f.payload("reference.f32", Buffer.alloc(4));
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /Float plane byte count/,
  );
  const floats = Buffer.alloc(480 * 256 * 7 * 4);
  floats.writeFloatLE(NaN, 0);
  await f.payload("reference.f32", floats);
  await assert.rejects(importPoseSensitivityArtifacts(f), /Nonfinite float/);
  floats.writeFloatLE(0, 0);
  floats.writeFloatLE(1, 480 * 256 * 4);
  await f.payload("reference.f32", floats);
  await assert.rejects(importPoseSensitivityArtifacts(f), /colour range/);
  await noOutput(f.root);
});

test("rejects failed numerical evidence even when booleans claim success", async (t) => {
  const f = await fixture(t);
  f.validation.quadrature_refinements[1].count_max = 1;
  await f.payload("validation.json", f.validation);
  await assert.rejects(importPoseSensitivityArtifacts(f), /quadrature results/);
  f.validation.quadrature_refinements[1].count_max = 0;
  f.validation.independent_rays = [];
  await f.payload("validation.json", f.validation);
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /Independent reference rays/,
  );
  await noOutput(f.root);
});

test("rejects status counts which hide unresolved pixels and private validation fields", async (t) => {
  const f = await fixture(t),
    statuses = Buffer.alloc(480 * 256 * 6);
  statuses[8] = 1;
  await f.payload("fd-status.u8", statuses);
  await assert.rejects(importPoseSensitivityArtifacts(f), /FD status counts/);
  statuses[8] = 0;
  await f.payload("fd-status.u8", statuses);
  f.validation.limitations = ["source at ", "home", "user", "private"].join(
    "/",
  );
  await f.payload("validation.json", f.validation);
  await assert.rejects(importPoseSensitivityArtifacts(f), /private path/);
  await noOutput(f.root);
});

test("preserves an existing nonidentical target without replacement", async (t) => {
  const f = await fixture(t);
  await importPoseSensitivityArtifacts(f);
  const target = path.join(f.root, TARGET, "projection.png");
  await writeFile(target, "existing user bytes");
  await assert.rejects(
    importPoseSensitivityArtifacts(f),
    /Existing pose-sensitivity artefact differs/,
  );
  assert.equal(await readFile(target, "utf8"), "existing user bytes");
});
