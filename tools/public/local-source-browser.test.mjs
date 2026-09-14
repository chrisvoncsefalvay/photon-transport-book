import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { generateSourceManifest } from "./generate-source-manifest.mjs";
import {
  escapeSourceHtml,
  renderSourcePage,
  writeLocalSourcePage,
} from "./local-source-browser.mjs";

const file = "python/dpt/canonical.py";
const source =
  '# ordinary first line\n# region book:sample\nvalue = "<script>& \\\"quoted\\\""\n# endregion book:sample\n';
async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), "dpt-local-source-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, "python/dpt"), { recursive: true });
  await mkdir(path.join(root, "src"));
  await writeFile(path.join(root, file), source);
  await writeFile(
    path.join(root, "src/chapter.mdx"),
    `<Source file="${file}" region="sample" />`,
  );
  return root;
}

test("escapes source text and file labels without active HTML", () => {
  const html = renderSourcePage(
    '<file>&".py',
    '<script>alert("x")</script>\na < b && c > d',
    "abc123",
  );
  assert.ok(!html.includes("<script>"));
  assert.ok(html.includes("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"));
  assert.ok(html.includes("&lt;file&gt;&amp;&quot;.py"));
  assert.equal(escapeSourceHtml("&<>\"'"), "&amp;&lt;&gt;&quot;&#39;");
  assert.ok(html.includes('id="L1"'));
  assert.ok(html.includes('href="#L2"'));
});

test("local source links resolve to the full canonical file and exact line anchor", async (t) => {
  const root = await fixture(t);
  const manifest = await generateSourceManifest({
    root,
    commit: "abc123",
    localSourceLinks: true,
  });
  const entry = manifest.regions[0];
  assert.equal(entry.url, `/generated/source-files/${file}.html#L3`);
  assert.equal(entry.code, source.split("\n")[2]);
  const html = await readFile(
    path.join(root, "public", entry.url.split("#")[0]),
    "utf8",
  );
  assert.ok(html.includes('id="L3"'));
  for (const line of source.split("\n"))
    assert.ok(html.includes(`<code>${escapeSourceHtml(line)}</code>`));
  assert.ok(html.includes(createHash("sha256").update(source).digest("hex")));
});

test("default source generation retains the pinned GitHub link", async (t) => {
  const root = await fixture(t);
  const manifest = await generateSourceManifest({
    root,
    commit: "abc123",
    localSourceLinks: false,
  });
  assert.equal(
    manifest.regions[0].url,
    `https://github.com/chrisvoncsefalvay/photon-transport-book/blob/abc123/${file}#L3-L3`,
  );
  await assert.rejects(
    readFile(path.join(root, "public/generated/source-files", `${file}.html`)),
    /ENOENT/,
  );
});

test("local browser rejects paths that escape the repository", async (t) => {
  const root = await fixture(t);
  await assert.rejects(
    writeLocalSourcePage({ root, file: "../secret.py", startLine: 1 }),
    /must not traverse|relative repository path|escape/,
  );
});

test("local browser rejects a symlink source", async (t) => {
  const root = await fixture(t);
  await rm(path.join(root, file));
  await symlink("/etc/hosts", path.join(root, file));
  await assert.rejects(
    writeLocalSourcePage({ root, file, startLine: 1 }),
    /symlinks/,
  );
});

test("the opt-in environment flag controls the generator CLI", async (t) => {
  const root = await fixture(t);
  await promisify(execFile)(
    process.execPath,
    [
      fileURLToPath(new URL("./generate-source-manifest.mjs", import.meta.url)),
      "--root",
      root,
      "--commit",
      "abc123",
    ],
    { env: { ...process.env, DPT_LOCAL_SOURCE_LINKS: "1" } },
  );
  const manifest = JSON.parse(
    await readFile(
      path.join(root, "public/generated/source-regions.json"),
      "utf8",
    ),
  );
  assert.equal(
    manifest.regions[0].url,
    `/generated/source-files/${file}.html#L3`,
  );
});

test("local browser rejects a symlink destination", async (t) => {
  const root = await fixture(t);
  await mkdir(path.join(root, "public/generated"), { recursive: true });
  await symlink("/tmp", path.join(root, "public/generated/source-files"));
  await assert.rejects(
    writeLocalSourcePage({ root, file, startLine: 1 }),
    /symlinks/,
  );
});
