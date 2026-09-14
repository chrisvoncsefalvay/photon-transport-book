import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  mkdtemp,
  mkdir,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { stringify } from "yaml";

import {
  PUBLIC_REPOSITORY,
  SOURCE_MANIFEST,
  WEBSITE_SNAPSHOT,
  canonicalSiteRoot,
  checkDeployment,
  checkDeploymentInputs,
  checkDeploymentOutput,
} from "./check-deployment.mjs";
import { loadBookCitation } from "./page-citation.mjs";

// Configuration fixtures only; these are not claimed publication records.
const SOURCE = "a".repeat(40);
const PUBLIC = "b".repeat(40);
const SITE = "https://book.example.invalid/";
const gitEnvironment = {
  VERCEL: "1",
  VERCEL_GIT_REPO_OWNER: "chrisvoncsefalvay",
  VERCEL_GIT_REPO_SLUG: "photon-transport-book",
  VERCEL_GIT_COMMIT_SHA: PUBLIC,
  VERCEL_URL: "temporary-preview.example.invalid",
};

async function fixture(t) {
  const root = await mkdtemp(
    path.join(os.tmpdir(), "dpt-deployment-config-test-"),
  );
  t.after(() => rm(root, { recursive: true, force: true }));
  const cff = {
    "cff-version": "1.2.0",
    title: "Synthetic website fixture",
    authors: [{ "family-names": "Example", "given-names": "Test" }],
    version: "0.0.0-bootstrap",
    "date-released": "2032-02-29",
    "repository-code": `https://github.com/${PUBLIC_REPOSITORY}`,
    url: SITE,
  };
  const pkg = {
    engines: { node: "24.x" },
    packageManager: "pnpm@12.3.4",
    scripts: { "build:site": "astro build" },
  };
  const sources = {
    schema_version: 1,
    source_commit: SOURCE,
    repository_url: `https://github.com/${PUBLIC_REPOSITORY}`,
    regions: [
      {
        file: "python/example.py",
        region: "example",
        start_line: 1,
        url: "/generated/source-files/python/example.py.html#L1",
      },
    ],
  };
  await mkdir(path.join(root, "public/generated"), { recursive: true });
  await writeFile(path.join(root, "package.json"), JSON.stringify(pkg));
  await writeFile(path.join(root, "CITATION.cff"), stringify(cff));
  await writeFile(path.join(root, SOURCE_MANIFEST), JSON.stringify(sources));
  return { root, cff, pkg, sources };
}

test("a CLI website snapshot keeps draft citation and author identity without a DOI or Git checkout", async (t) => {
  const { root } = await fixture(t);
  const result = await checkDeployment({ root, environment: { VERCEL: "1" } });
  assert.equal(result.sourceCommit, SOURCE);
  assert.equal(result.publicCommit, undefined);
  assert.equal(result.site, SITE);
  assert.equal(result.citation.version, "0.0.0-bootstrap");
  assert.equal(result.citation.released, false);
  assert.equal(result.citation.doi, undefined);
});

test("a Git preview keeps the production URL and distinct author/public commits", async (t) => {
  const { root } = await fixture(t);
  const result = await checkDeployment({ root, environment: gitEnvironment });
  assert.equal(result.site, SITE);
  assert.equal(result.sourceCommit, SOURCE);
  assert.equal(result.publicCommit, PUBLIC);
});

test("a website snapshot does not weaken the formal version-specific DOI gate", async (t) => {
  const { root } = await fixture(t);
  await checkDeployment({ root, environment: {} });
  await assert.rejects(
    loadBookCitation({ root, requireDoi: true }),
    /version-specific Zenodo DOI/,
  );
  await writeFile(path.join(root, "public/release-metadata.json"), "{}");
  await assert.rejects(
    checkDeployment({ root, environment: {} }),
    /Prepared release metadata is invalid/,
  );
});

test("missing, stale-format or wrong-repository source manifests fail preflight", async (t) => {
  const { root, sources } = await fixture(t);
  for (const replacement of [
    { ...sources, source_commit: "main" },
    { ...sources, repository_url: "https://github.com/example/authoring" },
    { ...sources, regions: [] },
  ]) {
    await writeFile(
      path.join(root, SOURCE_MANIFEST),
      JSON.stringify(replacement),
    );
    await assert.rejects(
      checkDeploymentInputs({ root, environment: {} }),
      /generated public source manifest/,
    );
  }
  await rm(path.join(root, SOURCE_MANIFEST));
  await assert.rejects(
    checkDeploymentInputs({ root, environment: {} }),
    /ENOENT/,
  );
});

test("authoring paths and dangling symlinks fail before dependency installation", async (t) => {
  for (const relative of [
    "AGENTS.md",
    "task_plan.md",
    "notes",
    "release",
    "tools/promotion",
  ]) {
    const { root } = await fixture(t);
    await mkdir(path.dirname(path.join(root, relative)), { recursive: true });
    await symlink("missing-private-target", path.join(root, relative));
    await assert.rejects(
      checkDeploymentInputs({ root, environment: {} }),
      /Deploy only the prepared public tree/,
    );
  }
});

test("source manifest inputs cannot traverse symlinks", async (t) => {
  const { root } = await fixture(t);
  await rm(path.join(root, SOURCE_MANIFEST));
  await symlink("missing-manifest", path.join(root, SOURCE_MANIFEST));
  await assert.rejects(
    checkDeploymentInputs({ root, environment: {} }),
    /symlinks/,
  );
});

test("incorrect repository and malformed Git commit metadata are rejected", async (t) => {
  const { root } = await fixture(t);
  await assert.rejects(
    checkDeploymentInputs({
      root,
      environment: { ...gitEnvironment, VERCEL_GIT_REPO_SLUG: "authoring" },
    }),
    /designated public repository/,
  );
  await assert.rejects(
    checkDeploymentInputs({
      root,
      environment: { ...gitEnvironment, VERCEL_GIT_COMMIT_SHA: "main" },
    }),
    /full commit SHA/,
  );
});

test("hosting rejects inherited authoring identity overrides", async (t) => {
  const { root } = await fixture(t);
  for (const name of [
    "DPT_LOCAL_SOURCE_LINKS",
    "DPT_SOURCE_COMMIT",
    "DPT_SOURCE_LINK_REF",
  ])
    await assert.rejects(
      checkDeploymentInputs({ root, environment: { [name]: "override" } }),
      /authoring override/,
    );
});

test("Node major, exact package manager, sanitised scripts and local environment files are checked", async (t) => {
  const { root, pkg } = await fixture(t);
  await assert.rejects(
    checkDeploymentInputs({ root, environment: {}, nodeVersion: "22.12.0" }),
    /Node.js 24/,
  );
  pkg.packageManager = "pnpm@latest";
  await writeFile(path.join(root, "package.json"), JSON.stringify(pkg));
  await assert.rejects(
    checkDeploymentInputs({ root, environment: {} }),
    /exact pnpm/,
  );
  pkg.packageManager = "pnpm@12.3.4";
  pkg.scripts["release:prepare"] = "node tools/promotion/prepare-release.mjs";
  await writeFile(path.join(root, "package.json"), JSON.stringify(pkg));
  await assert.rejects(
    checkDeploymentInputs({ root, environment: {} }),
    /promotion commands/,
  );
  delete pkg.scripts["release:prepare"];
  await writeFile(path.join(root, "package.json"), JSON.stringify(pkg));
  await writeFile(path.join(root, ".env.local"), "TEST_ONLY=1");
  await assert.rejects(
    checkDeploymentInputs({ root, environment: {} }),
    /environment file/,
  );
});

test("canonical root must be an explicit HTTPS production origin", () => {
  assert.equal(canonicalSiteRoot(SITE), SITE);
  for (const value of [
    undefined,
    "",
    "http://book.example.invalid",
    `${SITE}book/`,
    `${SITE}?preview=1`,
    `${SITE}#fragment`,
    "https://test:password@book.example.invalid",
    "https://localhost/",
  ])
    assert.throws(() => canonicalSiteRoot(value), /HTTPS/);
});

async function outputFixture(t) {
  const data = await fixture(t);
  const { root, sources } = data;
  const snapshot = {
    schema_version: 1,
    kind: "website-snapshot",
    source_commit: SOURCE,
    canonical_site: SITE,
    source_regions_sha256: createHash("sha256")
      .update(JSON.stringify(sources))
      .digest("hex"),
  };
  await mkdir(path.join(root, "dist/generated/source-files/python"), {
    recursive: true,
  });
  await mkdir(path.join(root, "dist/chapters/example"), { recursive: true });
  await writeFile(path.join(root, WEBSITE_SNAPSHOT), JSON.stringify(snapshot));
  await writeFile(
    path.join(root, "dist/website-snapshot.json"),
    JSON.stringify(snapshot),
  );
  await writeFile(
    path.join(root, "dist/generated/source-regions.json"),
    JSON.stringify(sources),
  );
  await writeFile(
    path.join(root, "dist/generated/source-files/python/example.py.html"),
    '<span id="L1">configuration fixture</span>',
  );
  for (const route of ["", "chapters/example/"])
    await writeFile(
      path.join(root, "dist", route, "index.html"),
      `<html><head><link rel="canonical" href="${SITE}${route}"></head></html>`,
    );
  return { ...data, snapshot };
}

test("static output preserves snapshot bytes, source lines and canonical URLs", async (t) => {
  const { root } = await outputFixture(t);
  const result = await checkDeploymentOutput({ root, site: SITE });
  assert.equal(result.files, 5);
  assert.equal(result.pages, 2);
});

test("changed snapshot, source-manifest bytes and missing source lines are rejected", async (t) => {
  const { root, sources, snapshot } = await outputFixture(t);
  await writeFile(path.join(root, "dist/website-snapshot.json"), "{}");
  await assert.rejects(
    checkDeploymentOutput({ root, site: SITE }),
    /changed the website snapshot/,
  );
  await writeFile(
    path.join(root, "dist/website-snapshot.json"),
    JSON.stringify(snapshot),
  );
  await writeFile(path.join(root, "dist/generated/source-regions.json"), "{}");
  await assert.rejects(
    checkDeploymentOutput({ root, site: SITE }),
    /changed the source manifest/,
  );
  await writeFile(
    path.join(root, "dist/generated/source-regions.json"),
    JSON.stringify(sources),
  );
  await writeFile(
    path.join(root, "dist/generated/source-files/python/example.py.html"),
    '<span id="L2">other line</span>',
  );
  await assert.rejects(
    checkDeploymentOutput({ root, site: SITE }),
    /missing line/,
  );
});

test("a mismatched snapshot digest cannot attest to a different source manifest", async (t) => {
  const { root, snapshot } = await outputFixture(t);
  snapshot.source_regions_sha256 = "0".repeat(64);
  for (const relative of [WEBSITE_SNAPSHOT, "dist/website-snapshot.json"])
    await writeFile(path.join(root, relative), JSON.stringify(snapshot));
  await assert.rejects(
    checkDeploymentOutput({ root, site: SITE }),
    /source identity disagrees/,
  );
});

test("preview canonical URLs, raw CT files and output symlinks are rejected", async (t) => {
  const { root } = await outputFixture(t);
  const index = path.join(root, "dist/index.html");
  const html = await readFile(index, "utf8");
  await writeFile(
    index,
    html.replace(SITE, "https://preview.example.invalid/"),
  );
  await assert.rejects(
    checkDeploymentOutput({ root, site: SITE }),
    /canonical production/,
  );
  await writeFile(index, html);
  await writeFile(
    path.join(root, "dist/volume.nii.gz"),
    "extension-only fixture",
  );
  await assert.rejects(
    checkDeploymentOutput({ root, site: SITE }),
    /Ineligible static output/,
  );
  await rm(path.join(root, "dist/volume.nii.gz"));
  await symlink("missing-output", path.join(root, "dist/hidden.json"));
  await assert.rejects(
    checkDeploymentOutput({ root, site: SITE }),
    /Symbolic links/,
  );
});

test("static output rejects private files and deployment environment files", async (t) => {
  for (const relative of [
    "task_plan.md",
    ".env.local",
    ".git/config",
    "tools/promotion/script.mjs",
  ]) {
    const { root } = await outputFixture(t);
    const file = path.join(root, "dist", relative);
    await mkdir(path.dirname(file), { recursive: true });
    await writeFile(file, "forbidden-path fixture");
    await assert.rejects(
      checkDeploymentOutput({ root, site: SITE }),
      /Ineligible static output/,
    );
  }
});
