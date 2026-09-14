import { lstat, readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { parseDocument } from "yaml";

/** Canonical paths are shared by source previews and scientific artefacts. */
export function canonicalPath(value) {
  if (
    typeof value !== "string" ||
    !value ||
    /[\\:\u0000-\u001f]/.test(value) ||
    path.posix.isAbsolute(value) ||
    value.split("/").some((part) => !part || part === "." || part === "..")
  ) {
    throw new Error(`Non-canonical artefact path: ${JSON.stringify(value)}`);
  }
  return value;
}

export async function checkedPath(
  root,
  relative,
  expected = "file",
  allowMissing = false,
) {
  canonicalPath(relative);
  let current = path.resolve(root);
  const parts = relative.split("/");
  for (const [index, part] of parts.entries()) {
    current = path.join(current, part);
    let info;
    try {
      info = await lstat(current);
    } catch (error) {
      if (allowMissing && error.code === "ENOENT")
        return path.join(root, relative);
      throw error;
    }
    if (info.isSymbolicLink())
      throw new Error(`Paths must not traverse symlinks: ${relative}`);
    if (index < parts.length - 1 || expected === "directory") {
      if (!info.isDirectory())
        throw new Error(`Expected directory: ${relative}`);
    } else if (expected === "file" && !info.isFile()) {
      throw new Error(`Expected regular file: ${relative}`);
    }
  }
  return current;
}

export async function assertNoSymlinks(root, relative) {
  await checkedPath(root, relative, "any", true);
}

export async function readRegularFile(root, relative) {
  return readFile(await checkedPath(root, relative));
}

export function parseUniqueJson(text, label) {
  const value = JSON.parse(text); // Enforce JSON grammar, not YAML extensions.
  const document = parseDocument(text, { schema: "json", uniqueKeys: true });
  if (document.errors.length) {
    const error = document.errors[0];
    throw new Error(
      `${label}: ${error.code === "DUPLICATE_KEY" ? "duplicate JSON key" : error.message}`,
      { cause: error },
    );
  }
  return value;
}

export async function walkFiles(root) {
  const files = [];

  async function visit(relativeDirectory) {
    const absoluteDirectory = path.join(root, relativeDirectory);
    const entries = await readdir(absoluteDirectory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));

    for (const entry of entries) {
      const relativePath = path.posix.join(
        relativeDirectory.split(path.sep).join("/"),
        entry.name,
      );
      if (entry.isSymbolicLink()) {
        throw new Error(
          `Symbolic links are not supported in public inputs: ${relativePath}`,
        );
      }
      if (entry.isDirectory()) {
        await visit(relativePath);
      } else if (entry.isFile()) {
        files.push(relativePath);
      }
    }
  }

  const stat = await lstat(root);
  if (!stat.isDirectory()) {
    throw new Error(`Expected a directory: ${root}`);
  }
  await visit("");
  return files;
}

export async function readJsonFile(filePath) {
  let source;
  try {
    source = await readFile(filePath, "utf8");
  } catch (error) {
    throw new Error(`Cannot read ${filePath}: ${error.message}`, {
      cause: error,
    });
  }

  try {
    return JSON.parse(source);
  } catch (strictError) {
    // Prettier emits legal YAML flow collections with trailing commas. Removing
    // only commas immediately before a closing collection preserves the
    // intentionally small JSON-compatible YAML subset used by this repository.
    try {
      return JSON.parse(source.replace(/,(\s*[}\]])/g, "$1"));
    } catch (error) {
      throw new Error(
        `${filePath} must use JSON-compatible YAML during bootstrap: ${error.message}`,
        { cause: strictError },
      );
    }
  }
}

export function normaliseRepositoryPath(value, label = "path") {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  if (value.includes("\\")) {
    throw new Error(`${label} must use forward slashes: ${value}`);
  }
  if (path.posix.isAbsolute(value)) {
    throw new Error(`${label} must be repository-relative: ${value}`);
  }
  const normalised = path.posix.normalize(value);
  if (
    normalised === ".." ||
    normalised.startsWith("../") ||
    normalised.includes("/../")
  ) {
    throw new Error(`${label} escapes the repository: ${value}`);
  }
  if (normalised === "." || normalised.startsWith("./")) {
    throw new Error(`${label} must not contain redundant segments: ${value}`);
  }
  return normalised;
}
