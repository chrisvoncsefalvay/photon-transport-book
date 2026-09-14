import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  discoverSourceReferences,
  generateSourceManifest,
} from "./generate-source-manifest.mjs";

test("discovers literal Source references in either attribute order", () => {
  assert.deepEqual(
    discoverSourceReferences(
      '<Source region="one" file="python/one.py" />\n<Source file="python/two.py" region="two" />',
    ),
    [
      { file: "python/one.py", region: "one" },
      { file: "python/two.py", region: "two" },
    ],
  );
  assert.throws(
    () => discoverSourceReferences('<Source file="x.py" />'),
    /without literal/,
  );
});

test("generates the public source-manifest contract deterministically", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "dpt-source-manifest-"));
  await mkdir(path.join(root, "src/content"), { recursive: true });
  await mkdir(path.join(root, "python"), { recursive: true });
  await writeFile(
    path.join(root, "src/content/chapter.mdx"),
    '<Source file="python/demo.py" region="demo" />\n',
  );
  await writeFile(
    path.join(root, "python/demo.py"),
    "# region book:demo\ndef demo():\n    return 1\n# endregion book:demo\n",
  );

  const manifest = await generateSourceManifest({
    root,
    commit: "0123456789abcdef",
    repositoryUrl: "https://github.com/example/public-book",
  });
  assert.equal(manifest.schema_version, 1);
  assert.equal(manifest.regions.length, 1);
  assert.deepEqual(Object.keys(manifest.regions[0]), [
    "file",
    "region",
    "language",
    "code",
    "start_line",
    "end_line",
    "url",
  ]);
  assert.equal(manifest.regions[0].language, "python");
  assert.match(manifest.regions[0].url, /public-book\/blob\/0123456789abcdef/);
  assert.deepEqual(
    JSON.parse(
      await readFile(
        path.join(root, "public/generated/source-regions.json"),
        "utf8",
      ),
    ),
    manifest,
  );
});

test("discovers every canonical variant without evaluating MDX expressions", () => {
  assert.deepEqual(
    discoverSourceReferences(
      `<Source file="python/base.py" region="base" variants='[{"label":"CUDA","file":"python/alternate.cu","region":"alternate"}]' />`,
    ),
    [
      { file: "python/base.py", region: "base" },
      { file: "python/alternate.cu", region: "alternate" },
    ],
  );
  for (const variants of [
    "{someVariable}",
    "'[1]'",
    "'{}'",
    `'[{"label":"CUDA"}]'`,
  ]) {
    assert.throws(
      () =>
        discoverSourceReferences(
          `<Source file="python/base.py" region="base" variants=${variants} />`,
        ),
      /variants|variant/,
    );
  }
});

test("variant source files must exist and contain the requested canonical region", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "dpt-source-variants-"));
  await mkdir(path.join(root, "src"), { recursive: true });
  await writeFile(
    path.join(root, "base.py"),
    "# region book:base\n# structural extraction fixture\n# endregion book:base\n",
  );
  await writeFile(
    path.join(root, "src/chapter.mdx"),
    `<Source file="base.py" region="base" variants='[{"label":"CUDA","file":"missing.cu","region":"missing"}]' />`,
  );
  await assert.rejects(
    generateSourceManifest({ root, commit: "0123456789abcdef" }),
    /Cannot read source file/,
  );
});
