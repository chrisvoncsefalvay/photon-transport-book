import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { normaliseRepositoryPath } from "./lib/files.mjs";
import { isPrivateExperimentPath } from "./lib/private-experiments.mjs";

const START_MARKER =
  /^\s*(?:#|\/\/)\s*region\s+book:([A-Za-z][A-Za-z0-9._-]*)\s*$/;
const END_MARKER =
  /^\s*(?:#|\/\/)\s*endregion\s+book:([A-Za-z][A-Za-z0-9._-]*)\s*$/;
const MARKER_HINT = /^\s*(?:#|\/\/)\s*(?:end)?region\b/;

export class SourceRegionError extends Error {
  constructor(message, { filePath, line } = {}) {
    const suffix = [filePath, line ? `line ${line}` : undefined]
      .filter(Boolean)
      .join(":");
    super(suffix ? `${message} (${suffix})` : message);
    this.name = "SourceRegionError";
    this.filePath = filePath;
    this.line = line;
  }
}

export function parseSourceRegions(source, { filePath = "<source>" } = {}) {
  const lines = source.split(/\r?\n/);
  const regions = new Map();
  let openRegion = null;

  lines.forEach((line, zeroBasedLine) => {
    const lineNumber = zeroBasedLine + 1;
    const start = line.match(START_MARKER);
    const end = line.match(END_MARKER);

    if (start) {
      if (openRegion) {
        throw new SourceRegionError(
          `Nested source region ${start[1]} inside ${openRegion.id} is not supported`,
          { filePath, line: lineNumber },
        );
      }
      if (regions.has(start[1])) {
        throw new SourceRegionError(`Duplicate source region ${start[1]}`, {
          filePath,
          line: lineNumber,
        });
      }
      openRegion = {
        id: start[1],
        markerStart: lineNumber,
        bodyStart: lineNumber + 1,
      };
      return;
    }

    if (end) {
      if (!openRegion) {
        throw new SourceRegionError(
          `Closing marker for unopened region ${end[1]}`,
          {
            filePath,
            line: lineNumber,
          },
        );
      }
      if (end[1] !== openRegion.id) {
        throw new SourceRegionError(
          `Closing marker ${end[1]} does not match open region ${openRegion.id}`,
          { filePath, line: lineNumber },
        );
      }
      regions.set(openRegion.id, {
        id: openRegion.id,
        markerStart: openRegion.markerStart,
        markerEnd: lineNumber,
        startLine: openRegion.bodyStart,
        endLine: lineNumber - 1,
        code: lines.slice(openRegion.bodyStart - 1, lineNumber - 1).join("\n"),
      });
      openRegion = null;
      return;
    }

    if (MARKER_HINT.test(line)) {
      throw new SourceRegionError("Malformed source-region marker", {
        filePath,
        line: lineNumber,
      });
    }
  });

  if (openRegion) {
    throw new SourceRegionError(`Unclosed source region ${openRegion.id}`, {
      filePath,
      line: openRegion.markerStart,
    });
  }

  return regions;
}

export function sourceUrl({
  repositoryUrl,
  commit,
  filePath,
  startLine,
  endLine,
  publicBuild = false,
}) {
  const relativePath = normaliseRepositoryPath(filePath, "source file");
  if (publicBuild && isPrivateExperimentPath(relativePath)) {
    throw new SourceRegionError(
      "Private experiment source cannot be used in a public build",
      { filePath: relativePath },
    );
  }
  if (!repositoryUrl || !commit) {
    return null;
  }
  const base = repositoryUrl.replace(/\.git$/, "").replace(/\/$/, "");
  if (!/^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(base)) {
    throw new SourceRegionError(
      `Unsupported GitHub repository URL: ${repositoryUrl}`,
    );
  }
  if (
    publicBuild &&
    /(?:photon-transport-dev|differentiable-photon-transport-dev)/i.test(base)
  ) {
    throw new SourceRegionError(
      "A private development repository URL cannot be used in a public build",
    );
  }
  if (!/^[A-Za-z0-9._/-]+$/.test(commit) || commit.includes("..")) {
    throw new SourceRegionError(`Invalid source commit: ${commit}`);
  }
  const encodedPath = relativePath.split("/").map(encodeURIComponent).join("/");
  const encodedRef = commit.split("/").map(encodeURIComponent).join("/");
  return `${base}/blob/${encodedRef}/${encodedPath}#L${startLine}-L${endLine}`;
}

export async function extractSourceRegion({
  root = process.cwd(),
  filePath,
  region,
  stripMarkers = true,
  repositoryUrl,
  commit,
  publicBuild = false,
}) {
  const relativePath = normaliseRepositoryPath(filePath, "source file");
  if (publicBuild && isPrivateExperimentPath(relativePath)) {
    throw new SourceRegionError(
      "Private experiment source cannot be used in a public build",
      { filePath: relativePath },
    );
  }
  const absoluteRoot = path.resolve(root);
  const absolutePath = path.resolve(absoluteRoot, relativePath);
  if (
    absolutePath !== absoluteRoot &&
    !absolutePath.startsWith(`${absoluteRoot}${path.sep}`)
  ) {
    throw new SourceRegionError("Source file escapes the repository", {
      filePath: relativePath,
    });
  }

  let source;
  try {
    source = await readFile(absolutePath, "utf8");
  } catch (error) {
    throw new SourceRegionError(`Cannot read source file: ${error.message}`, {
      filePath: relativePath,
    });
  }
  const regions = parseSourceRegions(source, { filePath: relativePath });
  const selected = regions.get(region);
  if (!selected) {
    throw new SourceRegionError(`Source region ${region} does not exist`, {
      filePath: relativePath,
    });
  }
  const lines = source.split(/\r?\n/);
  const startLine = stripMarkers ? selected.startLine : selected.markerStart;
  const endLine = stripMarkers ? selected.endLine : selected.markerEnd;
  const code = stripMarkers
    ? selected.code
    : lines.slice(selected.markerStart - 1, selected.markerEnd).join("\n");

  return {
    file: relativePath,
    region,
    code,
    start_line: startLine,
    end_line: endLine,
    url: sourceUrl({
      repositoryUrl,
      commit,
      filePath: relativePath,
      startLine,
      endLine,
      publicBuild,
    }),
  };
}

function parseArguments(argv) {
  const options = { stripMarkers: true, publicBuild: false };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--include-markers") options.stripMarkers = false;
    else if (argument === "--public") options.publicBuild = true;
    else if (argument.startsWith("--")) {
      const key = argument
        .slice(2)
        .replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      options[key] = argv[index + 1];
      index += 1;
    } else if (!options.filePath) options.filePath = argument;
    else if (!options.region) options.region = argument;
    else throw new Error(`Unexpected argument: ${argument}`);
  }
  return options;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    const result = await extractSourceRegion(
      parseArguments(process.argv.slice(2)),
    );
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
