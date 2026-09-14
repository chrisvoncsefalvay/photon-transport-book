import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  checkDeployment,
  checkDeploymentOutput,
  PUBLIC_REPOSITORY,
  SOURCE_MANIFEST,
  WEBSITE_SNAPSHOT,
} from "./check-deployment.mjs";
import { generateSourceManifest } from "./generate-source-manifest.mjs";
import { validateAll } from "./validate-all.mjs";

export async function buildDeployment({ root = process.cwd() } = {}) {
  const checked = await checkDeployment({ root });
  // The author commit identifies provenance; same-site pages expose the exact
  // admitted source bytes without pretending that SHA exists in the public repo.
  await generateSourceManifest({
    root,
    commit: checked.sourceCommit,
    linkRef: checked.publicCommit ?? "main",
    localSourceLinks: true,
  });
  await validateAll({ root });
  const snapshot = {
    schema_version: 1,
    kind: "website-snapshot",
    source_commit: checked.sourceCommit,
    ...(checked.publicCommit ? { public_commit: checked.publicCommit } : {}),
    repository_url: `https://github.com/${PUBLIC_REPOSITORY}`,
    canonical_site: checked.site,
    citation_version: checked.citation.version,
    source_link_mode: "same-site",
    source_regions_sha256: createHash("sha256")
      .update(await readFile(path.join(root, SOURCE_MANIFEST)))
      .digest("hex"),
  };
  await writeFile(
    path.join(root, WEBSITE_SNAPSHOT),
    `${JSON.stringify(snapshot, null, 2)}\n`,
  );
  execFileSync("corepack", ["pnpm", "run", "build:site"], {
    cwd: root,
    stdio: "inherit",
  });
  return {
    snapshot,
    output: await checkDeploymentOutput({ root, site: checked.site }),
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    process.stdout.write(`${JSON.stringify(await buildDeployment())}\n`);
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
