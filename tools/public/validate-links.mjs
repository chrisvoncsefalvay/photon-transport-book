import { access, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { walkFiles } from "./lib/files.mjs";

async function exists(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

function linksInSource(source) {
  const links = [];
  const withoutFences = source.replace(/```[\s\S]*?```/g, "");
  for (const match of withoutFences.matchAll(
    /(?<!!)\[[^\]]*\]\(([^\s)]+)(?:\s+["'][^"']*["'])?\)/g,
  )) {
    links.push(match[1]);
  }
  for (const match of withoutFences.matchAll(
    /\b(?:href|src)=["']([^"']+)["']/g,
  ))
    links.push(match[1]);
  return links;
}

async function routeCandidates(root, sourceFile, target) {
  const withoutQuery = target.split(/[?#]/, 1)[0];
  if (!withoutQuery) return [];
  if (withoutQuery.startsWith("/")) {
    const route = withoutQuery.slice(1).replace(/\/$/, "");
    return [
      path.join(root, "src/pages", `${route}.astro`),
      path.join(root, "src/pages", `${route}.md`),
      path.join(root, "src/pages", `${route}.mdx`),
      path.join(root, "src/pages", route, "index.astro"),
      path.join(root, "src/pages", route, "index.mdx"),
      path.join(root, "public", route),
      path.join(root, "public", route, "index.html"),
    ];
  }
  const resolved = path.resolve(path.dirname(sourceFile), withoutQuery);
  return [
    resolved,
    `${resolved}.md`,
    `${resolved}.mdx`,
    `${resolved}.astro`,
    path.join(resolved, "index.mdx"),
  ];
}

export async function validateInternalLinks({
  root = process.cwd(),
  sourceDirectory = "src",
} = {}) {
  const sourceRoot = path.join(root, sourceDirectory);
  const files = (await walkFiles(sourceRoot)).filter((file) =>
    /\.(?:astro|md|mdx)$/.test(file),
  );
  let checked = 0;
  for (const file of files) {
    const absoluteFile = path.join(sourceRoot, file);
    const source = await readFile(absoluteFile, "utf8");
    for (const link of linksInSource(source)) {
      if (/^(?:https?:|mailto:|tel:|data:|javascript:|#|\{)/.test(link))
        continue;
      checked += 1;
      const candidates = await routeCandidates(root, absoluteFile, link);
      if (!(await Promise.all(candidates.map(exists))).some(Boolean)) {
        throw new Error(
          `${path.posix.join(sourceDirectory, file)} contains a broken internal link: ${link}`,
        );
      }
    }
  }
  return { files: files.length, checked };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    const result = await validateInternalLinks();
    process.stdout.write(
      `Validated ${result.checked} internal link(s) in ${result.files} file(s).\n`,
    );
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
