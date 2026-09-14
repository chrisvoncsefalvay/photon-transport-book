import { createHash } from "node:crypto";
import { lstat, readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const PUBLIC_REPOSITORY = "chrisvoncsefalvay/photon-transport-book";
export const SOURCE_MANIFEST = "public/generated/source-regions.json";
export const WEBSITE_SNAPSHOT = "public/website-snapshot.json";
const COMMIT = /^[0-9a-f]{40}$/;
const PRIVATE_PATHS = [
  "AGENTS.md",
  "task_plan.md",
  "notes.md",
  "notes",
  "research",
  "agents",
  "release",
  "assets-src",
  "tools/promotion",
  "tests/promotion",
  "experiments/acquired-hap",
  ".beagle",
  "wandb",
];

async function regularFile(root, relative) {
  let current = path.resolve(root);
  const parts = relative.split("/");
  for (const [index, part] of parts.entries()) {
    current = path.join(current, part);
    const info = await lstat(current);
    if (info.isSymbolicLink())
      throw new Error(
        `Deployment inputs may not traverse symlinks: ${relative}`,
      );
    if (index < parts.length - 1 ? !info.isDirectory() : !info.isFile())
      throw new Error(`Expected a regular deployment input: ${relative}`);
  }
  return readFile(current, "utf8");
}

export function canonicalSiteRoot(value) {
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error(
      "Set the final canonical HTTPS site root in CITATION.cff url",
    );
  }
  if (
    url.protocol !== "https:" ||
    url.username ||
    url.password ||
    url.pathname !== "/" ||
    url.search ||
    url.hash ||
    url.hostname === "localhost" ||
    url.hostname.endsWith(".localhost")
  ) {
    throw new Error(
      "CITATION.cff url must be an HTTPS origin without a path prefix",
    );
  }
  return url.href;
}

/** A wrong-tree guard, not a replacement for the private allowlist export scan. */
export async function checkDeploymentInputs({
  root = process.cwd(),
  environment = process.env,
  nodeVersion = process.versions.node,
} = {}) {
  if (nodeVersion.split(".")[0] !== "24")
    throw new Error("The deployment build requires Node.js 24");
  for (const relative of PRIVATE_PATHS) {
    try {
      await lstat(path.join(root, relative));
    } catch (error) {
      if (error.code === "ENOENT") continue;
      throw error;
    }
    throw new Error(`Deploy only the prepared public tree: found ${relative}`);
  }
  const rootEntries = await readdir(root);
  if (
    rootEntries.some(
      (name) =>
        name === ".env" ||
        (name.startsWith(".env.") && name !== ".env.example"),
    )
  )
    throw new Error("Deployment source contains a local environment file");
  for (const key of [
    "DPT_LOCAL_SOURCE_LINKS",
    "DPT_SOURCE_COMMIT",
    "DPT_SOURCE_LINK_REF",
  ]) {
    if (environment[key])
      throw new Error(
        `Remove the authoring override ${key} from deployment settings`,
      );
  }
  const owner = environment.VERCEL_GIT_REPO_OWNER;
  const repository = environment.VERCEL_GIT_REPO_SLUG;
  if ((owner || repository) && `${owner}/${repository}` !== PUBLIC_REPOSITORY)
    throw new Error(
      "Vercel Git metadata must identify the designated public repository",
    );
  if (
    environment.VERCEL_GIT_COMMIT_SHA &&
    !COMMIT.test(environment.VERCEL_GIT_COMMIT_SHA)
  )
    throw new Error("Vercel Git metadata must contain a full commit SHA");

  const pkg = JSON.parse(await regularFile(root, "package.json"));
  if (
    pkg.engines?.node !== "24.x" ||
    !/^pnpm@\d+\.\d+\.\d+$/.test(pkg.packageManager ?? "")
  )
    throw new Error("Pin Node 24.x and an exact pnpm version in package.json");
  if (
    Object.entries(pkg.scripts ?? {}).some(
      ([name, command]) =>
        name.startsWith("release:") ||
        /(?:tools|tests)\/promotion/.test(command),
    )
  )
    throw new Error(
      "Deployment package.json still contains promotion commands",
    );

  const sources = JSON.parse(await regularFile(root, SOURCE_MANIFEST));
  if (
    sources.schema_version !== 1 ||
    !COMMIT.test(sources.source_commit ?? "") ||
    sources.repository_url !== `https://github.com/${PUBLIC_REPOSITORY}` ||
    !Array.isArray(sources.regions) ||
    sources.regions.length === 0
  )
    throw new Error(
      "Deployment requires a generated public source manifest with its full author commit SHA",
    );
  return {
    sourceCommit: sources.source_commit,
    publicCommit:
      owner && repository ? environment.VERCEL_GIT_COMMIT_SHA : undefined,
    packageManager: pkg.packageManager,
  };
}

export async function checkDeployment({
  root = process.cwd(),
  environment = process.env,
} = {}) {
  const checked = await checkDeploymentInputs({ root, environment });
  const { loadBookCitation } = await import("./page-citation.mjs");
  // A website snapshot does not claim a formal edition. If prepared release
  // metadata is present, loadBookCitation still verifies its frozen DOI contract.
  const citation = await loadBookCitation({ root });
  if (citation.repositoryCode !== `https://github.com/${PUBLIC_REPOSITORY}`)
    throw new Error(
      "CITATION.cff must identify the designated public repository",
    );
  if (citation.sourceCommit && citation.sourceCommit !== checked.sourceCommit)
    throw new Error("Prepared release and source manifest commits disagree");
  return { ...checked, citation, site: canonicalSiteRoot(citation.url) };
}

/** Validate the files that the static host will actually serve. */
export async function checkDeploymentOutput({
  root = process.cwd(),
  site,
} = {}) {
  const { walkFiles } = await import("./lib/files.mjs");
  const output = path.join(root, "dist");
  const files = await walkFiles(output);
  if (!files.includes("index.html"))
    throw new Error("Static output lacks index.html");
  if (
    (await regularFile(root, WEBSITE_SNAPSHOT)) !==
    (await regularFile(output, "website-snapshot.json"))
  )
    throw new Error("Static output changed the website snapshot metadata");
  if (
    (await regularFile(root, SOURCE_MANIFEST)) !==
    (await regularFile(output, "generated/source-regions.json"))
  )
    throw new Error("Static output changed the source manifest");
  const sources = JSON.parse(
    await regularFile(output, "generated/source-regions.json"),
  );
  const snapshot = JSON.parse(
    await regularFile(output, "website-snapshot.json"),
  );
  if (
    snapshot.schema_version !== 1 ||
    snapshot.kind !== "website-snapshot" ||
    snapshot.source_commit !== sources.source_commit ||
    snapshot.canonical_site !== site ||
    snapshot.source_regions_sha256 !==
      createHash("sha256")
        .update(await regularFile(output, "generated/source-regions.json"))
        .digest("hex")
  )
    throw new Error(
      "Static output source identity disagrees with its website snapshot",
    );
  for (const region of sources.regions) {
    const match = /^\/generated\/source-files\/(.+\.html)#L([1-9]\d*)$/.exec(
      region.url ?? "",
    );
    if (!match || Number(match[2]) !== region.start_line)
      throw new Error(
        "Deployment source links must identify same-site source lines",
      );
    const relative = decodeURIComponent(match[1]);
    if (
      relative
        .split("/")
        .some((part) => !part || part === "." || part === "..") ||
      relative.includes("\\")
    )
      throw new Error("Deployment source link contains a non-canonical path");
    const html = await regularFile(
      output,
      `generated/source-files/${relative}`,
    );
    if (!html.includes(`id="L${region.start_line}"`))
      throw new Error("Deployment source link points to a missing line");
  }
  let bytes = 0;
  let largest = { path: "", bytes: 0 };
  let pages = 0;
  for (const file of files) {
    if (
      PRIVATE_PATHS.some(
        (entry) => file === entry || file.startsWith(`${entry}/`),
      ) ||
      /(?:^|\/)(?:\.env(?:\.[^/]*)?|\.git|\.vercel)(?:\/|$)|\.(?:dcm|nii(?:\.gz)?|nrrd|raw|h5|hdf5|map)$/i.test(
        file,
      )
    )
      throw new Error(`Ineligible static output: ${file}`);
    const size = (await lstat(path.join(output, file))).size;
    if (size > 25 * 1024 * 1024)
      throw new Error(
        `Static file exceeds the project's 25 MiB limit: ${file}`,
      );
    bytes += size;
    if (size > largest.bytes) largest = { path: file, bytes: size };
    if (file === "index.html" || file.endsWith("/index.html")) {
      const html = await regularFile(output, file);
      const expected = new URL(file.replace(/index\.html$/, ""), site).href;
      if (!html.includes(`<link rel="canonical" href="${expected}">`))
        throw new Error(
          `Static page lacks its canonical production URL: ${file}`,
        );
      pages += 1;
    }
  }
  return { files: files.length, pages, bytes, largest };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    if (process.argv.length !== 3 || process.argv[2] !== "--preinstall")
      throw new Error(
        "Usage: node tools/public/check-deployment.mjs --preinstall",
      );
    const { sourceCommit, packageManager } = await checkDeploymentInputs();
    process.stdout.write(
      `${sourceCommit}: exported-tree preflight passed (${packageManager})\n`,
    );
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
