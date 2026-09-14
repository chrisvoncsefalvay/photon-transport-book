import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { importTransmissionArtifacts } from "./import-transmission-artifacts.mjs";
import { validateGeneratedArtifacts } from "./validate-generated-artifacts.mjs";

const TARGET = "public/generated/transmission-contract";
const sha = (value) => createHash("sha256").update(value).digest("hex");
const sourceNames = [
  "python/dpt/transmission.py",
  "python/dpt/kernels/transmission.py",
  "python/dpt/validation/transmission.py",
  "experiments/transmission-contract/run.py",
  "experiments/transmission-contract/profile_cuda.py",
  "experiments/transmission-contract/config.json",
  "pyproject.toml",
  "uv.lock",
];
const figures = [
  "transmission-sweep.svg",
  "transmission-weak-attenuation.svg",
  "transmission-tail-rescue.svg",
];

async function fixture(t) {
  const base = await mkdtemp(path.join(os.tmpdir(), "dpt-projection-test-"));
  t.after(() => rm(base, { recursive: true, force: true }));
  const root = path.join(base, "repo"),
    runDir = path.join(base, "private-run");
  await mkdir(runDir, { recursive: true });
  await mkdir(path.join(root, "tools/public/schemas"), { recursive: true });
  await copyFile(
    new URL("./schemas/generated-artifact.schema.json", import.meta.url),
    path.join(root, "tools/public/schemas/generated-artifact.schema.json"),
  );
  const sources = {};
  // Deliberately synthetic bytes exercise projection integrity, never CUDA correctness.
  for (const name of sourceNames) {
    const bytes = `synthetic source: ${name}\n`;
    await mkdir(path.dirname(path.join(root, name)), { recursive: true });
    await writeFile(path.join(root, name), bytes);
    sources[name] = sha(bytes);
  }
  const payloads = Object.fromEntries(
    figures.map((name) => [
      name,
      '<svg xmlns="http://www.w3.org/2000/svg"><text>synthetic test fixture</text></svg>',
    ]),
  );
  payloads["validation.json"] = JSON.stringify({
    status: "passed",
    synthetic_fixture: true,
  });
  for (const [name, bytes] of Object.entries(payloads))
    await writeFile(path.join(runDir, name), bytes);
  await writeFile(
    path.join(runDir, "benchmark.json"),
    "private benchmark fixture",
  );
  await mkdir(path.join(runDir, "sources"));
  await writeFile(
    path.join(runDir, "sources/private.txt"),
    "private source copy",
  );
  const record = {
    schema_version: 1,
    status: "passed",
    source_commit: "a".repeat(40),
    source_sha256: sources,
    started_utc: "2026-09-09T00:00:00Z",
    completed_utc: "2026-09-09T00:01:00Z",
    output_sha256: Object.fromEntries(
      Object.entries(payloads).map(([name, bytes]) => [name, sha(bytes)]),
    ),
    command: ["/private/run.py"],
    platform: "private hostname",
    source_status: "private working files",
    nvidia_smi: "raw private diagnostics",
    nvcc: "raw compiler output",
    benchmark_rows: 100,
    python: "3.12.13",
    warp: "1.17.0",
    device: "cuda:0",
    device_name: "Synthetic test device",
    cuda_toolkit: [12, 9],
    cuda_driver_api: 13000,
    compute_capability: 121,
    config: {
      schema_version: 1,
      sweep_points: 129,
      benchmark_sizes: [100],
      private_path: "/private/input",
    },
    numerics: {
      storage: "binary32",
      intermediates: "binary64",
      fast_math: false,
      fuse_fp: true,
      oracle_decimal_digits: 100,
      reduction_tile_size: 256,
      private_path: "/private/input",
    },
  };
  const save = () =>
    writeFile(path.join(runDir, "run.json"), JSON.stringify(record));
  await save();
  return { root, runDir, record, save };
}

async function noOutput(root) {
  await assert.rejects(readFile(path.join(root, TARGET, "run.json")), /ENOENT/);
}

test("projects only verified payloads and allowlisted metadata, retaining raw input", async (t) => {
  const { root, runDir } = await fixture(t);
  const original = await readFile(path.join(runDir, "run.json"));
  assert.equal(
    (await importTransmissionArtifacts({ root, runDir })).status,
    "imported",
  );
  assert.deepEqual(
    (await readdir(path.join(root, TARGET))).sort(),
    [
      ...figures,
      "run.json",
      "validation.json",
      "artifact.manifest.json",
    ].sort(),
  );
  const metadata = await readFile(path.join(root, TARGET, "run.json"), "utf8");
  for (const forbidden of [
    "command",
    "platform",
    "source_status",
    "nvidia_smi",
    "nvcc",
    "benchmark",
    "private_path",
    "/private/",
  ])
    assert.ok(!metadata.includes(forbidden), forbidden);
  const projected = JSON.parse(metadata);
  assert.equal(projected.metadata.warp, "1.17.0");
  assert.deepEqual(projected.configuration, {
    schema_version: 1,
    sweep_points: 129,
  });
  assert.deepEqual(await readFile(path.join(runDir, "run.json")), original);
  assert.deepEqual(await validateGeneratedArtifacts({ root }), {
    manifests: 1,
    outputs: 5,
    sources: 8,
  });
  assert.equal(
    (await importTransmissionArtifacts({ root, runDir })).status,
    "unchanged",
  );
});

test("rejects an unfinished or failed run before writing", async (t) => {
  const { root, runDir, record, save } = await fixture(t);
  for (const status of ["running", "failed"]) {
    record.status = status;
    await save();
    await assert.rejects(
      importTransmissionArtifacts({ root, runDir }),
      /Only a passed/,
    );
    await noOutput(root);
  }
});

test("rejects changed current source before writing", async (t) => {
  const { root, runDir } = await fixture(t);
  await writeFile(path.join(root, sourceNames[0]), "changed source");
  await assert.rejects(
    importTransmissionArtifacts({ root, runDir }),
    /Current source SHA256 mismatch/,
  );
  await noOutput(root);
});

test("rejects altered figure bytes before writing", async (t) => {
  const { root, runDir } = await fixture(t);
  await writeFile(path.join(runDir, figures[2]), "altered final payload");
  await assert.rejects(
    importTransmissionArtifacts({ root, runDir }),
    /Run output SHA256 mismatch/,
  );
  await noOutput(root);
});

test("requires all declared source digests, even when omitted source would pass", async (t) => {
  const { root, runDir, record, save } = await fixture(t);
  delete record.source_sha256[sourceNames[0]];
  await save();
  await assert.rejects(
    importTransmissionArtifacts({ root, runDir }),
    /exactly the declared/,
  );
  await noOutput(root);
});

test("does not overwrite an existing different projection", async (t) => {
  const { root, runDir } = await fixture(t);
  await importTransmissionArtifacts({ root, runDir });
  const file = path.join(root, TARGET, figures[0]);
  await writeFile(file, "existing different content");
  await assert.rejects(
    importTransmissionArtifacts({ root, runDir }),
    /Existing transmission artefact differs/,
  );
  assert.equal(await readFile(file, "utf8"), "existing different content");
});

test("requires an absolute run path and a full source commit", async (t) => {
  const { root, runDir, record, save } = await fixture(t);
  await assert.rejects(
    importTransmissionArtifacts({ root, runDir: "relative" }),
    /absolute/,
  );
  record.source_commit = "not-a-commit";
  await save();
  await assert.rejects(
    importTransmissionArtifacts({ root, runDir }),
    /full source_commit/,
  );
  await noOutput(root);
});

async function upgradeFixture(f) {
  const { record, root, runDir } = f;
  record.schema_version = 2;
  record.status = "complete";
  record.finished_utc = record.completed_utc;
  record.configuration = record.config;
  record.metadata = {
    source_commit: record.source_commit,
    warp: record.warp,
    device_name: record.device_name,
    cuda_toolkit: record.cuda_toolkit,
    numerics: record.numerics,
  };
  record.sources_unchanged = true;
  record.recorded_files_unchanged = true;
  delete record.completed_utc;
  delete record.config;
  delete record.source_commit;
  for (const name of sourceNames) {
    const file = path.join(runDir, "sources", name);
    await mkdir(path.dirname(file), { recursive: true });
    await copyFile(path.join(root, name), file);
  }
  const configuration = JSON.stringify(record.configuration);
  record.configuration_sha256 = sha(configuration);
  record.output_sha256["configuration.json"] = sha(configuration);
  await writeFile(path.join(runDir, "configuration.json"), configuration);
  record.source_sha256["configuration.json"] = sha(configuration);
  await writeFile(
    path.join(runDir, "sources/configuration.json"),
    configuration,
  );
  await f.save();
}

test("imports the common version-2 completed envelope and verifies its snapshots", async (t) => {
  const f = await fixture(t);
  await upgradeFixture(f);
  assert.equal((await importTransmissionArtifacts(f)).status, "imported");
  const projected = JSON.parse(
    await readFile(path.join(f.root, TARGET, "run.json"), "utf8"),
  );
  assert.equal(projected.schema_version, 2);
  assert.equal(projected.status, "complete");
  assert.equal(projected.finished_utc, f.record.finished_utc);
  assert.equal(projected.metadata.warp, "1.17.0");
});

test("rejects version-2 completion with changed or tampered source snapshots", async (t) => {
  const f = await fixture(t);
  await upgradeFixture(f);
  f.record.sources_unchanged = false;
  await f.save();
  await assert.rejects(importTransmissionArtifacts(f), /unchanged/);
  f.record.sources_unchanged = true;
  await f.save();
  await writeFile(path.join(f.runDir, "sources", sourceNames[0]), "tampered");
  await assert.rejects(
    importTransmissionArtifacts(f),
    /snapshot SHA256 mismatch/,
  );
  await noOutput(f.root);
});

test("rejects modified effective configuration and non-boolean completion flags", async (t) => {
  const f = await fixture(t);
  await upgradeFixture(f);
  f.record.sources_unchanged = "false";
  await f.save();
  await assert.rejects(importTransmissionArtifacts(f), /unchanged/);
  f.record.sources_unchanged = true;
  f.record.configuration.sweep_points++;
  await f.save();
  await assert.rejects(
    importTransmissionArtifacts(f),
    /effective configuration/,
  );
  await noOutput(f.root);
});
