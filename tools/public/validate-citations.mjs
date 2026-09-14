import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { readJsonFile, walkFiles } from "./lib/files.mjs";

export function citationKeysInSource(source) {
  const keys = [];
  const withoutFences = source.replace(/```[\s\S]*?```/g, "");
  for (const match of withoutFences.matchAll(
    /\[@([A-Za-z][A-Za-z0-9._:-]*)\]/g,
  ))
    keys.push(match[1]);
  for (const match of withoutFences.matchAll(
    /\{cite:([A-Za-z][A-Za-z0-9._:-]*)\}/g,
  ))
    keys.push(match[1]);
  for (const component of withoutFences.matchAll(
    /<Citation\b([\s\S]*?)(?:\/>|>)/g,
  )) {
    const keysExpression = component[1].match(
      /keys\s*=\s*\{\s*\[([\s\S]*?)\]\s*\}/,
    )?.[1];
    if (!keysExpression) {
      throw new Error(
        'Citation components must use a literal keys={["citation-key"]} expression',
      );
    }
    for (const quoted of keysExpression.matchAll(
      /["']([A-Za-z][A-Za-z0-9._:-]*)["']/g,
    )) {
      keys.push(quoted[1]);
    }
  }
  return keys;
}

export async function validateCitationKeys({
  root = process.cwd(),
  contentDirectory = "src",
} = {}) {
  const library = await readJsonFile(
    path.join(root, "references/library.json"),
  );
  const available = new Set(library.map((entry) => entry.id));
  const duplicates = library
    .map((entry) => entry.id)
    .filter((key, index, all) => all.indexOf(key) !== index);
  if (duplicates.length > 0) {
    throw new Error(
      `Duplicate citation keys: ${[...new Set(duplicates)].join(", ")}`,
    );
  }
  for (const item of library) {
    if (item.DOI && item.URL !== `https://doi.org/${item.DOI}`) {
      throw new Error(`${item.id} DOI and URL disagree`);
    }
  }

  const contentRoot = path.join(root, contentDirectory);
  const files = (await walkFiles(contentRoot)).filter((file) =>
    /\.(?:astro|md|mdx)$/.test(file),
  );
  let references = 0;
  for (const file of files) {
    const source = await readFile(path.join(contentRoot, file), "utf8");
    for (const key of citationKeysInSource(source)) {
      references += 1;
      if (!available.has(key))
        throw new Error(
          `${path.posix.join(contentDirectory, file)} uses unknown citation key ${key}`,
        );
    }
  }
  return { files: files.length, references };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    const result = await validateCitationKeys();
    process.stdout.write(
      `Validated ${result.references} citation reference(s) in ${result.files} file(s).\n`,
    );
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
