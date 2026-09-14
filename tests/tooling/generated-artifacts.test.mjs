import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { validateGeneratedArtifacts } from "../../tools/public/validate-generated-artifacts.mjs";

const source = "experiments/synthetic/run.py";
const output = "public/generated/synthetic/payload.txt";
const manifestPath = "public/generated/synthetic/artifact.manifest.json";
const sha = (value) => createHash("sha256").update(value).digest("hex");

async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), "dpt-artifact-integrity-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, "tools/public/schemas"), { recursive: true });
  await copyFile(
    new URL(
      "../../tools/public/schemas/generated-artifact.schema.json",
      import.meta.url,
    ),
    path.join(root, "tools/public/schemas/generated-artifact.schema.json"),
  );
  await mkdir(path.dirname(path.join(root, source)), { recursive: true });
  await mkdir(path.dirname(path.join(root, output)), { recursive: true });
  // These bytes test filesystem integrity only; no scientific computation is claimed.
  await writeFile(path.join(root, source), "synthetic source bytes\n");
  await writeFile(path.join(root, output), "synthetic payload bytes\n");
  const manifest = {
    schema_version: 1,
    id: "synthetic-integrity-test",
    chapter: "fixture",
    experiment: "synthetic",
    source_commit: "1234567",
    generator: source,
    mode: "deterministic-precomputed-sweep",
    parameters: {},
    outputs: [output],
    stochastic: false,
    created_at: "2026-09-09T00:00:00Z",
    source_digests: { [source]: sha("synthetic source bytes\n") },
    output_digests: { [output]: sha("synthetic payload bytes\n") },
  };
  const save = () =>
    writeFile(path.join(root, manifestPath), JSON.stringify(manifest));
  await save();
  return { root, manifest, save };
}

test("discovers real manifests and checks their exact source and payload bytes", async (t) => {
  const { root } = await fixture(t);
  assert.deepEqual(await validateGeneratedArtifacts({ root }), {
    manifests: 1,
    outputs: 1,
    sources: 1,
  });
});

test("an absent generated directory has no generated results", async (t) => {
  const { root } = await fixture(t);
  await rm(path.join(root, "public/generated"), { recursive: true });
  assert.deepEqual(await validateGeneratedArtifacts({ root }), {
    manifests: 0,
    outputs: 0,
    sources: 0,
  });
});

for (const item of [source, output]) {
  test(`rejects changed bytes in ${item}`, async (t) => {
    const { root } = await fixture(t);
    await writeFile(path.join(root, item), "changed bytes");
    await assert.rejects(
      validateGeneratedArtifacts({ root }),
      /SHA256 mismatch/,
    );
  });
  test(`rejects a missing ${item}`, async (t) => {
    const { root } = await fixture(t);
    await rm(path.join(root, item));
    await assert.rejects(validateGeneratedArtifacts({ root }), /ENOENT/);
  });
  test(`rejects a symlink in ${item}`, async (t) => {
    const { root } = await fixture(t);
    await rm(path.join(root, item));
    await symlink("/etc/hosts", path.join(root, item));
    await assert.rejects(validateGeneratedArtifacts({ root }), /symlink/);
  });
}

for (const invalid of [
  "../outside.txt",
  "/tmp/outside.txt",
  "public/generated/../payload.txt",
  "public//generated/payload.txt",
  "public\\generated\\payload.txt",
  "C:/payload.txt",
]) {
  test(`rejects unsafe output path ${JSON.stringify(invalid)}`, async (t) => {
    const { root, manifest, save } = await fixture(t);
    manifest.outputs = [invalid];
    manifest.output_digests = { [invalid]: sha("synthetic payload bytes\n") };
    await save();
    await assert.rejects(validateGeneratedArtifacts({ root }), /Non-canonical/);
  });
}

test("rejects output outside the generated directory", async (t) => {
  const { root, manifest, save } = await fixture(t);
  manifest.outputs = [source];
  manifest.output_digests = { ...manifest.source_digests };
  await save();
  await assert.rejects(
    validateGeneratedArtifacts({ root }),
    /Output must lie under/,
  );
});

test("requires the listed outputs to equal the digested outputs", async (t) => {
  const { root, manifest, save } = await fixture(t);
  manifest.outputs = [];
  await save();
  await assert.rejects(validateGeneratedArtifacts({ root }), /must match/);
});

test("requires generator source coverage", async (t) => {
  const { root, manifest, save } = await fixture(t);
  manifest.source_digests = {};
  await save();
  await assert.rejects(
    validateGeneratedArtifacts({ root }),
    /generator must be included/,
  );
});

test("rejects duplicate output entries", async (t) => {
  const { root, manifest, save } = await fixture(t);
  manifest.outputs.push(output);
  await save();
  await assert.rejects(validateGeneratedArtifacts({ root }), /duplicate items/);
});

test("rejects duplicate JSON members before their overwritten value can hide bytes", async (t) => {
  const { root } = await fixture(t);
  const file = path.join(root, manifestPath);
  const text = await readFile(file, "utf8");
  await writeFile(
    file,
    text.replace('"schema_version":1', '"schema_version":0,"schema_version":1'),
  );
  await assert.rejects(
    validateGeneratedArtifacts({ root }),
    /duplicate JSON key/,
  );
});

test("rejects duplicate IDs across manifests", async (t) => {
  const { root } = await fixture(t);
  const second = path.join(root, "public/generated/second");
  await mkdir(second);
  await copyFile(
    path.join(root, manifestPath),
    path.join(second, "artifact.manifest.json"),
  );
  await assert.rejects(
    validateGeneratedArtifacts({ root }),
    /Duplicate generated artefact ID/,
  );
});

test("rejects two manifests claiming the same payload", async (t) => {
  const { root, manifest } = await fixture(t);
  const second = path.join(root, "public/generated/second");
  await mkdir(second);
  await writeFile(
    path.join(second, "artifact.manifest.json"),
    JSON.stringify({ ...manifest, id: "second" }),
  );
  await assert.rejects(
    validateGeneratedArtifacts({ root }),
    /Duplicate generated output ownership/,
  );
});

test("rejects bootstrap metadata in the generated result tree", async (t) => {
  const { root, manifest, save } = await fixture(t);
  manifest.bootstrap = true;
  await save();
  await assert.rejects(
    validateGeneratedArtifacts({ root }),
    /bootstrap fixtures/,
  );
});

test("rejects a generated directory symlink even without a manifest", async (t) => {
  const { root } = await fixture(t);
  await symlink("/tmp", path.join(root, "public/generated/elsewhere"));
  await assert.rejects(validateGeneratedArtifacts({ root }), /symlink/);
});

test("requires seeds for stochastic independent realisations", async (t) => {
  const { root, manifest, save } = await fixture(t);
  manifest.mode = "stochastic-independent-realisations";
  manifest.stochastic = true;
  await save();
  await assert.rejects(
    validateGeneratedArtifacts({ root }),
    /requires independent-realisation seeds/,
  );
});
