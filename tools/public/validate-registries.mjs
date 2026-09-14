import path from "node:path";
import { fileURLToPath } from "node:url";

import { readJsonFile } from "./lib/files.mjs";
import { assertSchema } from "./lib/schema.mjs";

const REGISTRIES = [
  ["terminology/terms.yml", "terminology/schemas/terms.schema.json"],
  ["terminology/symbols.yml", "terminology/schemas/symbols.schema.json"],
  ["terminology/frames.yml", "terminology/schemas/frames.schema.json"],
];

function assertUniqueIds(entries, label) {
  const ids = entries.map((entry) => entry.id);
  const duplicates = ids.filter((id, index) => ids.indexOf(id) !== index);
  if (duplicates.length > 0) {
    throw new Error(
      `${label} contains duplicate IDs: ${[...new Set(duplicates)].join(", ")}`,
    );
  }
}

export async function validateRegistries({
  root = process.cwd(),
  artifactFiles = [],
} = {}) {
  const library = await readJsonFile(
    path.join(root, "references/library.json"),
  );
  const librarySchema = await readJsonFile(
    path.join(root, "tools/public/schemas/csl-library.schema.json"),
  );
  assertSchema(library, librarySchema, "references/library.json");
  assertUniqueIds(library, "references/library.json");
  const citationKeys = new Set(library.map((item) => item.id));

  for (const [registryPath, schemaPath] of REGISTRIES) {
    const entries = await readJsonFile(path.join(root, registryPath));
    const schema = await readJsonFile(path.join(root, schemaPath));
    assertSchema(entries, schema, registryPath);
    assertUniqueIds(entries, registryPath);
    for (const entry of entries) {
      for (const key of entry.source_keys ?? []) {
        if (!citationKeys.has(key)) {
          throw new Error(
            `${registryPath}:${entry.id} refers to unknown citation key ${key}`,
          );
        }
      }
    }
  }

  const artifactSchema = await readJsonFile(
    path.join(root, "tools/public/schemas/generated-artifact.schema.json"),
  );
  for (const artifactPath of artifactFiles) {
    const manifest = await readJsonFile(path.join(root, artifactPath));
    assertSchema(manifest, artifactSchema, artifactPath);
    if (manifest.mode === "stochastic-independent-realisations") {
      if (
        !manifest.stochastic ||
        !Array.isArray(manifest.seeds) ||
        manifest.seeds.length === 0
      ) {
        throw new Error(
          `${artifactPath} must record seeds for stochastic independent realisations`,
        );
      }
    } else if (manifest.stochastic) {
      throw new Error(
        `${artifactPath} marks a non-stochastic mode as stochastic`,
      );
    }
    if (!manifest.bootstrap && manifest.outputs.length === 0) {
      throw new Error(
        `${artifactPath} must list outputs unless it is an explicit bootstrap fixture`,
      );
    }
  }

  return {
    registries: REGISTRIES.length,
    citations: library.length,
    artifacts: artifactFiles.length,
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    const artifactFiles = process.argv.slice(2);
    const result = await validateRegistries({ artifactFiles });
    process.stdout.write(
      `Validated ${result.registries} registries, ${result.citations} citation(s), and ${result.artifacts} artefact manifest(s).\n`,
    );
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
