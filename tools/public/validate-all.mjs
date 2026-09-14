import { fileURLToPath } from "node:url";

import { validateGeneratedArtifacts } from "./validate-generated-artifacts.mjs";
import { validateCitationKeys } from "./validate-citations.mjs";
import { validateInternalLinks } from "./validate-links.mjs";
import { validateRegistries } from "./validate-registries.mjs";
import { validateSourceReferences } from "./validate-source-references.mjs";

export async function validateAll({ root = process.cwd() } = {}) {
  const registries = await validateRegistries({
    root,
    artifactFiles: [
      "tests/tooling/fixtures/generated-artifact.metadata-only.json",
    ],
  });
  const generated = await validateGeneratedArtifacts({ root });
  const citations = await validateCitationKeys({ root });
  const sources = await validateSourceReferences({ root });
  const links = await validateInternalLinks({ root });
  return { registries, generated, citations, sources, links };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    const result = await validateAll();
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
