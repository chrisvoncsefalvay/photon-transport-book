import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  extractSourceRegion,
  parseSourceRegions,
  sourceUrl,
} from "./source-regions.mjs";

test("extracts exact source while preserving indentation and line numbers", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "dpt-source-"));
  await writeFile(
    path.join(root, "fixture.py"),
    [
      "before",
      "# region book:identity",
      "def identity(value):",
      "    return value",
      "# endregion book:identity",
      "after",
    ].join("\n"),
  );

  const result = await extractSourceRegion({
    root,
    filePath: "fixture.py",
    region: "identity",
    repositoryUrl: "https://github.com/example/public-book",
    commit: "abc123",
    publicBuild: true,
  });

  assert.equal(result.code, "def identity(value):\n    return value");
  assert.equal(result.start_line, 3);
  assert.equal(result.end_line, 4);
  assert.equal(
    result.url,
    "https://github.com/example/public-book/blob/abc123/fixture.py#L3-L4",
  );
});

test("can include region markers in the selected range", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "dpt-source-"));
  await writeFile(
    path.join(root, "fixture.py"),
    "# region book:x\nx = 1\n# endregion book:x\n",
  );
  const result = await extractSourceRegion({
    root,
    filePath: "fixture.py",
    region: "x",
    stripMarkers: false,
  });
  assert.equal(result.code, "# region book:x\nx = 1\n# endregion book:x");
  assert.deepEqual([result.start_line, result.end_line], [1, 3]);
});

test("rejects missing, duplicate, nested, mismatched, and malformed regions", () => {
  assert.throws(
    () => parseSourceRegions("# region book:x\na = 1\n", { filePath: "x.py" }),
    /Unclosed/,
  );
  assert.throws(
    () => parseSourceRegions("# region book:x\na = 1\n# endregion book:y"),
    /does not match/,
  );
  assert.throws(
    () =>
      parseSourceRegions(
        "# region book:x\n# endregion book:x\n# region book:x\n# endregion book:x",
      ),
    /Duplicate/,
  );
  assert.throws(
    () =>
      parseSourceRegions(
        "# region book:x\n# region book:y\n# endregion book:y\n# endregion book:x",
      ),
    /Nested/,
  );
  assert.throws(() => parseSourceRegions("# region book:\nx = 1"), /Malformed/);
});

test("missing files and regions fail explicitly", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "dpt-source-"));
  await assert.rejects(
    extractSourceRegion({ root, filePath: "absent.py", region: "x" }),
    /Cannot read source file/,
  );
  await writeFile(path.join(root, "fixture.py"), "x = 1\n");
  await assert.rejects(
    extractSourceRegion({ root, filePath: "fixture.py", region: "x" }),
    /does not exist/,
  );
});

test("public source URLs reject private repository targets", () => {
  const privateRepository = ["photon", "transport", "dev"].join("-");
  assert.throws(
    () =>
      sourceUrl({
        repositoryUrl: `https://github.com/chrisvoncsefalvay/${privateRepository}`,
        commit: "abc123",
        filePath: "python/dpt/bootstrap.py",
        startLine: 1,
        endLine: 2,
        publicBuild: true,
      }),
    /private development repository URL/,
  );
});
