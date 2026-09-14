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

import {
  assertNoSymlinks,
  readRegularFile,
  walkFiles,
  parseUniqueJson,
} from "./lib/files.mjs";
import { assertSchema } from "./lib/schema.mjs";

const TARGET = "public/generated/transmission-contract";
const GENERATOR = "experiments/transmission-contract/run.py";
const SOURCE_FILES = [
  "python/dpt/transmission.py",
  "python/dpt/kernels/transmission.py",
  "python/dpt/validation/transmission.py",
  GENERATOR,
  "experiments/transmission-contract/profile_cuda.py",
  "experiments/transmission-contract/config.json",
  "pyproject.toml",
  "uv.lock",
];
const FIGURES = [
  "transmission-sweep.svg",
  "transmission-weak-attenuation.svg",
  "transmission-tail-rescue.svg",
];
const PAYLOADS = [...FIGURES, "validation.json"];
const RUNTIME_KEYS = [
  "python",
  "warp",
  "numpy",
  "matplotlib",
  "device",
  "device_name",
  "compute_capability",
  "cuda_toolkit",
  "cuda_driver_api",
];
const NUMERIC_KEYS = [
  "storage",
  "intermediates",
  "fast_math",
  "fuse_fp",
  "oracle_decimal_digits",
  "reduction_tile_size",
];
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const jsonBytes = (value) => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
const select = (value, keys) =>
  Object.fromEntries(
    keys
      .filter((key) => Object.hasOwn(value ?? {}, key))
      .map((key) => [key, value[key]]),
  );

function requireTimestamp(value, name) {
  if (
    typeof value !== "string" ||
    !/^\d{4}-\d{2}-\d{2}T/.test(value) ||
    Number.isNaN(Date.parse(value))
  ) {
    throw new Error(`A valid ${name} is required`);
  }
}

export async function importTransmissionArtifacts({
  root = process.cwd(),
  runDir,
} = {}) {
  if (typeof runDir !== "string" || !path.isAbsolute(runDir)) {
    throw new Error("--run-dir must be an absolute private run directory");
  }
  root = path.resolve(root);
  if (!(await lstat(runDir)).isDirectory())
    throw new Error("Run directory must be a real directory");
  const raw = parseUniqueJson(
    (await readRegularFile(runDir, "run.json")).toString("utf8"),
    "run.json",
  );
  const legacy = raw.schema_version === 1 && raw.status === "passed";
  const current = raw.schema_version === 2 && raw.status === "complete";
  if (!legacy && !current)
    throw new Error(
      "Only a passed version-1 or complete version-2 run can be imported",
    );
  if (
    current &&
    (raw.sources_unchanged !== true || raw.recorded_files_unchanged !== true)
  )
    throw new Error(
      "Version-2 completion requires unchanged source and output snapshots",
    );
  if (current) {
    const configuration = await readRegularFile(runDir, "configuration.json");
    if (
      sha(configuration) !== raw.configuration_sha256 ||
      sha(configuration) !== raw.output_sha256?.["configuration.json"] ||
      !isDeepStrictEqual(
        parseUniqueJson(configuration.toString("utf8"), "configuration.json"),
        raw.configuration,
      )
    ) {
      throw new Error("Run effective configuration SHA256 or payload mismatch");
    }
  }
  const record = current
    ? {
        ...raw.metadata,
        ...raw,
        config: raw.configuration,
        completed_utc: raw.finished_utc,
      }
    : raw;
  const expectedSources = current
    ? [
        ...(await walkFiles(path.join(root, "python/dpt")))
          .filter((file) => file.endsWith(".py"))
          .map((file) => `python/dpt/${file}`),
        GENERATOR,
        "experiments/transmission-contract/profile_cuda.py",
        "experiments/transmission-contract/config.json",
        "pyproject.toml",
        "uv.lock",
      ]
    : SOURCE_FILES;
  if (!/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(record.source_commit ?? ""))
    throw new Error("A full source_commit is required");
  requireTimestamp(record.started_utc, "started_utc");
  requireTimestamp(record.completed_utc, "completed_utc");
  const sources = Object.keys(record.source_sha256 ?? {}).filter(
    (file) => !current || file !== "configuration.json",
  );
  if (
    sources.length !== expectedSources.length ||
    expectedSources.some((file) => !sources.includes(file))
  ) {
    throw new Error(
      "Run source_sha256 must cover exactly the declared transmission source files",
    );
  }
  const sourceDigests = {};
  for (const file of expectedSources) {
    const expected = record.source_sha256[file];
    if (
      !/^[0-9a-f]{64}$/.test(expected ?? "") ||
      sha(await readRegularFile(root, file)) !== expected
    ) {
      throw new Error(`Current source SHA256 mismatch: ${file}`);
    }
    if (
      current &&
      sha(await readRegularFile(runDir, `sources/${file}`)) !== expected
    )
      throw new Error(`Run source snapshot SHA256 mismatch: ${file}`);
    sourceDigests[file] = expected;
  }
  if (
    current &&
    sha(await readRegularFile(runDir, "sources/configuration.json")) !==
      record.source_sha256["configuration.json"]
  )
    throw new Error("Run configuration snapshot SHA256 mismatch");
  const files = new Map();
  for (const name of PAYLOADS) {
    const bytes = await readRegularFile(runDir, name);
    if (sha(bytes) !== record.output_sha256?.[name])
      throw new Error(`Run output SHA256 mismatch: ${name}`);
    files.set(name, bytes);
  }
  const validation = JSON.parse(files.get("validation.json").toString("utf8"));
  if (validation.status !== "passed")
    throw new Error("Validation data must report passed status");
  const sanitised = {
    schema_version: 2,
    status: "complete",
    started_utc: record.started_utc,
    finished_utc: record.completed_utc,
    source_commit: record.source_commit,
    source_sha256: sourceDigests,
    configuration: select(record.config, ["schema_version", "sweep_points"]),
    metadata: {
      ...select(record, RUNTIME_KEYS),
      numerics: select(record.numerics, NUMERIC_KEYS),
    },
    input_provenance:
      "deterministic mathematical stress inputs; no anatomical data",
    figures: FIGURES,
    output_sha256: Object.fromEntries(
      [...files].map(([name, bytes]) => [name, sha(bytes)]),
    ),
  };
  files.set("run.json", jsonBytes(sanitised));
  const manifest = {
    schema_version: 1,
    id: "transmission-contract",
    chapter: "transmission",
    experiment: "transmission-contract",
    source_commit: record.source_commit,
    generator: GENERATOR,
    mode: "deterministic-precomputed-sweep",
    device: record.device_name ?? null,
    cuda_version:
      record.cuda_toolkit == null
        ? null
        : Array.isArray(record.cuda_toolkit)
          ? record.cuda_toolkit.join(".")
          : String(record.cuda_toolkit),
    warp_version: record.warp ?? null,
    precision: "binary32 storage; binary64 range-sensitive intermediates",
    parameters: { sweep_points: record.config?.sweep_points },
    outputs: [...files.keys()].map((name) => `${TARGET}/${name}`),
    stochastic: false,
    created_at: record.completed_utc,
    source_digests: sourceDigests,
    output_digests: Object.fromEntries(
      [...files].map(([name, bytes]) => [`${TARGET}/${name}`, sha(bytes)]),
    ),
    notes:
      "Generated CUDA sweeps and independent mathematical validation. Raw profiling, benchmark records and private run metadata remain outside this projection.",
  };
  const schema = JSON.parse(
    await readFile(
      path.join(root, "tools/public/schemas/generated-artifact.schema.json"),
      "utf8",
    ),
  );
  assertSchema(manifest, schema, "transmission artefact manifest");
  files.set("artifact.manifest.json", jsonBytes(manifest));

  // Every input is verified before any destination is created or changed.
  await assertNoSymlinks(root, TARGET);
  const destination = path.join(root, TARGET);
  try {
    const existing = await walkFiles(destination);
    if (
      existing.length !== files.size ||
      existing.some((name) => !files.has(name))
    ) {
      throw new Error(
        "Existing transmission artefacts differ; use a separate checkout or explicitly remove the old generated set",
      );
    }
    for (const [name, bytes] of files) {
      if (!bytes.equals(await readRegularFile(destination, name))) {
        throw new Error(`Existing transmission artefact differs: ${name}`);
      }
    }
    return { target: TARGET, files: files.size, status: "unchanged" };
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  await mkdir(path.dirname(destination), { recursive: true });
  const temporary = await mkdtemp(
    path.join(path.dirname(destination), ".transmission-import-"),
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
    if (process.argv.length !== 4 || process.argv[2] !== "--run-dir") {
      throw new Error(
        "Usage: node tools/public/import-transmission-artifacts.mjs --run-dir /absolute/private/run",
      );
    }
    process.stdout.write(
      `${JSON.stringify(await importTransmissionArtifacts({ runDir: process.argv[3] }))}\n`,
    );
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
