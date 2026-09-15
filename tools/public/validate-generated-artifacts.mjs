import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { canonicalPath, checkedPath, parseUniqueJson } from "./lib/files.mjs";
import { assertSchema } from "./lib/schema.mjs";
import { isPrivateExperimentPath } from "./lib/private-experiments.mjs";

const GENERATED = "public/generated";
const DIGEST = /^[0-9a-f]{64}$/;

async function discover(root) {
  try {
    await checkedPath(root, GENERATED, "directory");
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  const manifests = [];
  async function visit(relative) {
    const entries = await readdir(path.join(root, relative), {
      withFileTypes: true,
    });
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      const child = `${relative}/${entry.name}`;
      if (entry.isSymbolicLink()) {
        throw new Error(`Generated artefact tree contains a symlink: ${child}`);
      }
      if (entry.isDirectory()) await visit(child);
      else if (entry.name === "artifact.manifest.json") manifests.push(child);
    }
  }
  await visit(GENERATED);
  return manifests;
}

async function verifyDigests(root, digests, label, outputs) {
  if (!digests || Object.keys(digests).length === 0) {
    throw new Error(`${label} requires a non-empty digest map`);
  }
  for (const [relative, expected] of Object.entries(digests)) {
    canonicalPath(relative);
    if (outputs && !relative.startsWith(`${GENERATED}/`)) {
      throw new Error(`Output must lie under ${GENERATED}: ${relative}`);
    }
    if (!DIGEST.test(expected))
      throw new Error(`Invalid SHA256 for ${relative}`);
    const file = await checkedPath(root, relative);
    const actual = createHash("sha256")
      .update(await readFile(file))
      .digest("hex");
    if (actual !== expected) throw new Error(`SHA256 mismatch for ${relative}`);
  }
}

export async function validateGeneratedArtifacts({
  root = process.cwd(),
} = {}) {
  root = path.resolve(root);
  const schema = JSON.parse(
    await readFile(
      path.join(root, "tools/public/schemas/generated-artifact.schema.json"),
      "utf8",
    ),
  );
  const manifests = await discover(root);
  const ids = new Set();
  const ownedOutputs = new Set();
  let outputs = 0;
  let sources = 0;
  let privateSources = 0;
  for (const relative of manifests) {
    const manifest = parseUniqueJson(
      await readFile(await checkedPath(root, relative), "utf8"),
      relative,
    );
    assertSchema(manifest, schema, relative);
    if (manifest.bootstrap) {
      throw new Error(
        `${relative}: bootstrap fixtures do not belong in generated results`,
      );
    }
    if (ids.has(manifest.id))
      throw new Error(`Duplicate generated artefact ID: ${manifest.id}`);
    ids.add(manifest.id);
    if (manifest.mode === "stochastic-independent-realisations") {
      if (!manifest.stochastic || !manifest.seeds?.length) {
        throw new Error(`${relative} requires independent-realisation seeds`);
      }
    } else if (manifest.stochastic) {
      throw new Error(`${relative} marks a deterministic mode as stochastic`);
    }
    const listed = manifest.outputs;
    const digested = Object.keys(manifest.output_digests ?? {});
    if (
      listed.length !== digested.length ||
      listed.some((item) => !digested.includes(item))
    ) {
      throw new Error(`${relative}: outputs must match output_digests exactly`);
    }
    canonicalPath(manifest.generator);
    const privateDigests = manifest.private_source_digests ?? {};
    for (const [file, digest] of Object.entries(privateDigests)) {
      canonicalPath(file);
      if (!isPrivateExperimentPath(file) || !file.startsWith("experiments/"))
        throw new Error(
          `${relative}: private source must lie under experiments/: ${file}`,
        );
      if (Object.hasOwn(manifest.source_digests ?? {}, file))
        throw new Error(
          `${relative}: source cannot be both public and private: ${file}`,
        );
      try {
        await verifyDigests(root, { [file]: digest }, "private source", false);
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
    }
    if (
      !Object.hasOwn(manifest.source_digests ?? {}, manifest.generator) &&
      !Object.hasOwn(privateDigests, manifest.generator)
    ) {
      throw new Error(
        `${relative}: generator must be included in source_digests or private_source_digests`,
      );
    }
    if (
      Object.keys(manifest.source_digests ?? {}).length ||
      !Object.keys(privateDigests).length
    ) {
      await verifyDigests(
        root,
        manifest.source_digests,
        `${relative}: source_digests`,
        false,
      );
    }
    await verifyDigests(
      root,
      manifest.output_digests,
      `${relative}: output_digests`,
      true,
    );
    for (const output of listed) {
      if (manifests.includes(output))
        throw new Error(`A manifest cannot be its own payload: ${output}`);
      if (ownedOutputs.has(output))
        throw new Error(`Duplicate generated output ownership: ${output}`);
      ownedOutputs.add(output);
    }
    outputs += listed.length;
    sources += Object.keys(manifest.source_digests ?? {}).length;
    privateSources += Object.keys(privateDigests).length;
  }
  return {
    manifests: manifests.length,
    outputs,
    sources,
    ...(privateSources ? { privateSources } : {}),
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    process.stdout.write(
      `${JSON.stringify(await validateGeneratedArtifacts())}\n`,
    );
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
