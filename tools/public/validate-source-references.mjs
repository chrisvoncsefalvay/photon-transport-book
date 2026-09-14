import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { discoverSourceReferences } from "./generate-source-manifest.mjs";
import { walkFiles } from "./lib/files.mjs";
import { extractSourceRegion } from "./source-regions.mjs";

export async function validateSourceReferences({
  root = process.cwd(),
  contentDirectory = "src",
} = {}) {
  const contentRoot = path.join(root, contentDirectory);
  const files = (await walkFiles(contentRoot)).filter((file) =>
    /\.mdx?$/.test(file),
  );
  let references = 0;
  for (const file of files) {
    const source = await readFile(path.join(contentRoot, file), "utf8");
    for (const reference of discoverSourceReferences(
      source,
      path.posix.join(contentDirectory, file),
    )) {
      references += 1;
      await extractSourceRegion({
        root,
        filePath: reference.file,
        region: reference.region,
      });
    }
  }
  return { files: files.length, references };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    const result = await validateSourceReferences();
    process.stdout.write(
      `Validated ${result.references} source reference(s) in ${result.files} file(s).\n`,
    );
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}
