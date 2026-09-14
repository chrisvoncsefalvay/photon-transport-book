import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  normaliseRepositoryPath,
  assertNoSymlinks,
  readRegularFile,
} from "./lib/files.mjs";

const PREFIX = "public/generated/source-files";

export function escapeSourceHtml(value) {
  return value.replace(
    /[&<>"']/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[character],
  );
}

export function renderSourcePage(file, source, digest) {
  const label = escapeSourceHtml(file);
  const lines = source
    .split(/\r?\n/)
    .map((line, index) => {
      const number = index + 1;
      return `<span class="line" id="L${number}"><a class="number" href="#L${number}" aria-label="Line ${number}">${number}</a><code>${escapeSourceHtml(line)}</code></span>`;
    })
    .join("");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${label} — canonical source</title>
<style>body{margin:0;background:#faf9f6;color:#172127;font:16px system-ui,sans-serif}header{padding:1.5rem 2rem;border-bottom:1px solid #ccc}h1{font-size:1.2rem;overflow-wrap:anywhere}p{font-size:.85rem;overflow-wrap:anywhere}pre{margin:0;padding:1rem 0;overflow:auto;font:14px/1.6 ui-monospace,monospace;tab-size:4}.line{display:block;min-width:max-content;padding-right:2rem;scroll-margin-top:1rem}.line:target{background:#fff1b8}.number{display:inline-block;width:4rem;padding-right:1rem;color:#647079;text-align:right;text-decoration:none;user-select:none}code{font:inherit}</style>
</head><body><header><h1>${label}</h1><p>Generated from the full canonical file for this source snapshot. Line numbers match the library source.</p><p>Source SHA256: <code>${escapeSourceHtml(digest)}</code></p></header><pre>${lines}</pre></body></html>\n`;
}

export async function writeLocalSourcePage({ root, file, startLine }) {
  const relative = normaliseRepositoryPath(file, "local source file");
  if (!Number.isInteger(startLine) || startLine < 1)
    throw new Error("Invalid source line");
  const bytes = await readRegularFile(root, relative);
  const destination = `${PREFIX}/${relative}.html`;
  await assertNoSymlinks(root, destination);
  await mkdir(path.dirname(path.join(root, destination)), { recursive: true });
  await writeFile(
    path.join(root, destination),
    renderSourcePage(
      relative,
      bytes.toString("utf8"),
      createHash("sha256").update(bytes).digest("hex"),
    ),
  );
  const encoded = relative.split("/").map(encodeURIComponent).join("/");
  return `/generated/source-files/${encoded}.html#L${startLine}`;
}
