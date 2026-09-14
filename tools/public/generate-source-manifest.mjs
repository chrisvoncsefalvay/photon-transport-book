import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

import { writeLocalSourcePage } from "./local-source-browser.mjs";
import { walkFiles } from "./lib/files.mjs";
import { extractSourceRegion } from "./source-regions.mjs";
import { parseSourceVariants } from "./source-variants.mjs";

const LANGUAGE_BY_EXTENSION = new Map([
  [".c", "c"],
  [".cc", "cpp"],
  [".cpp", "cpp"],
  [".cu", "cuda"],
  [".h", "c"],
  [".hpp", "cpp"],
  [".js", "javascript"],
  [".mjs", "javascript"],
  [".py", "python"],
  [".ts", "typescript"],
  [".tsx", "tsx"],
]);

export function discoverSourceReferences(source, filePath = "<content>") {
  const references = [];
  for (const match of source.matchAll(
    /<Source\b((?:"[^"]*"|'[^']*'|[^'">])*)(?:\/>|>)/g,
  )) {
    const attributes = new Map();
    for (const attribute of match[1].matchAll(
      /([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')/g,
    )) {
      attributes.set(attribute[1], attribute[2] ?? attribute[3]);
    }
    if (!attributes.has("file") || !attributes.has("region")) {
      throw new Error(
        `${filePath} contains a Source component without literal file and region attributes`,
      );
    }
    references.push({
      file: attributes.get("file"),
      region: attributes.get("region"),
    });
    if (/\bvariants\s*=/.test(match[1]) && !attributes.has("variants")) {
      throw new Error(
        `${filePath}: source variants must be literal JSON, not an expression.`,
      );
    }
    references.push(
      ...parseSourceVariants(attributes.get("variants"), filePath).map(
        ({ file, region }) => ({ file, region }),
      ),
    );
  }
  return references;
}

async function currentCommit(root) {
  if (process.env.DPT_SOURCE_COMMIT) return process.env.DPT_SOURCE_COMMIT;
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: root,
      encoding: "utf8",
    }).trim();
  } catch {
    try {
      const metadata = JSON.parse(
        await readFile(path.join(root, "public/release-metadata.json"), "utf8"),
      );
      if (metadata.source_commit) return metadata.source_commit;
    } catch {
      // Report one stable error below.
    }
    throw new Error(
      "A source commit is required; pass --commit, set DPT_SOURCE_COMMIT, retain release metadata, or run inside a committed Git checkout",
    );
  }
}

export async function generateSourceManifest({
  root = process.cwd(),
  contentDirectory = "src",
  output = "public/generated/source-regions.json",
  commit,
  linkRef,
  repositoryUrl = "https://github.com/chrisvoncsefalvay/photon-transport-book",
  publicBuild = true,
  localSourceLinks = process.env.DPT_LOCAL_SOURCE_LINKS === "1",
}) {
  const absoluteContent = path.join(root, contentDirectory);
  const contentFiles = (await walkFiles(absoluteContent)).filter((file) =>
    /\.mdx?$/.test(file),
  );
  const requested = [];
  for (const relativeContentFile of contentFiles) {
    const source = await readFile(
      path.join(absoluteContent, relativeContentFile),
      "utf8",
    );
    requested.push(
      ...discoverSourceReferences(
        source,
        path.posix.join(contentDirectory, relativeContentFile),
      ),
    );
  }

  const unique = new Map();
  for (const reference of requested) {
    unique.set(`${reference.file}\0${reference.region}`, reference);
  }
  const resolvedCommit = commit ?? (await currentCommit(root));
  const resolvedLinkRef =
    linkRef ??
    (commit === undefined ? process.env.DPT_SOURCE_LINK_REF : undefined);
  const regions = [];
  for (const reference of [...unique.values()].sort((left, right) =>
    `${left.file}:${left.region}`.localeCompare(
      `${right.file}:${right.region}`,
    ),
  )) {
    const extracted = await extractSourceRegion({
      root,
      filePath: reference.file,
      region: reference.region,
      repositoryUrl,
      commit: resolvedLinkRef ?? resolvedCommit,
      publicBuild,
    });
    regions.push({
      file: extracted.file,
      region: extracted.region,
      language:
        LANGUAGE_BY_EXTENSION.get(path.extname(extracted.file)) ?? "text",
      code: extracted.code,
      start_line: extracted.start_line,
      end_line: extracted.end_line,
      url: localSourceLinks
        ? await writeLocalSourcePage({
            root,
            file: extracted.file,
            startLine: extracted.start_line,
          })
        : extracted.url,
    });
  }
  const manifest = {
    schema_version: 1,
    source_commit: resolvedCommit,
    repository_url: repositoryUrl,
    regions,
  };
  const absoluteOutput = path.join(root, output);
  await mkdir(path.dirname(absoluteOutput), { recursive: true });
  await writeFile(absoluteOutput, `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

function parseArguments(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index]
      ?.replace(/^--/, "")
      .replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    if (!key || argv[index + 1] === undefined)
      throw new Error(`Invalid argument: ${argv[index] ?? ""}`);
    options[key] = argv[index + 1];
  }
  return options;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    const manifest = await generateSourceManifest(
      parseArguments(process.argv.slice(2)),
    );
    process.stdout.write(
      `Generated ${manifest.regions.length} source region(s).\n`,
    );
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
