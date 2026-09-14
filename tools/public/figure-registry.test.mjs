import test from "node:test";
import assert from "node:assert/strict";
import {
  mkdtempSync,
  mkdirSync,
  writeFileSync,
  renameSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createFigureRegistry,
  getFigureRegistry,
  clearFigureRegistry,
  resolveFigureHref,
} from "./lib/figure-registry.mjs";

const source = (body, chapter = "02") =>
  `---\nlayout: ../../layouts/ChapterLayout.astro\nchapterNumber: ${chapter}\n---\nimport Plate from "../../components/figures/FigurePlate.astro";\nimport Content from "../../components/figures/InteractiveFigure.astro";\n\n${body}`;
const document = (body, chapter = "02", slug = "sample") => ({
  filePath: `/book/src/pages/chapters/${slug}.mdx`,
  source: source(body, chapter),
});
const plate = (id) => `<Plate id="${id}" title="Fixture" />`;

test("numbers literal IDs in reading order, including generic nested wrappers", () => {
  const registry = createFigureRegistry([
    document(
      `${plate("a")}\n\n<div>\n${plate("b")}\n</div>\n\n<Plate id="c" title="Fixture"><Content /></Plate>`,
    ),
  ]);
  assert.deepEqual(
    registry.figures.map(({ id, number }) => [id, number]),
    [
      ["a", "2.1"],
      ["b", "2.2"],
      ["c", "2.3"],
    ],
  );
});

test("insertion and reordering change display numbers while preserving opaque IDs", () => {
  const before = createFigureRegistry([
    document(`${plate("figure-2-1")}\n\n${plate("historic-anchor")}`),
  ]);
  const after = createFigureRegistry([
    document(
      `${plate("new")}\n\n${plate("historic-anchor")}\n\n${plate("figure-2-1")}`,
    ),
  ]);
  assert.equal(before.get("/chapters/sample/", "figure-2-1").number, "2.1");
  assert.equal(after.get("/chapters/sample/", "figure-2-1").number, "2.3");
  assert.equal(after.get("/chapters/sample/", "historic-anchor").number, "2.2");
});

test("resets per chapter, supports appendices and permits IDs scoped to different pages", () => {
  const registry = createFigureRegistry([
    document(plate("shared")),
    document(plate("shared"), "A", "appendix"),
  ]);
  assert.equal(registry.get("/chapters/sample", "shared").number, "2.1");
  assert.equal(registry.get("/chapters/appendix/", "shared").number, "A.1");
});

test("rejects duplicate IDs, manual numbers, missing/dynamic IDs and spreads with source locations", () => {
  for (const [body, expected] of [
    [`${plate("same")}\n\n${plate("same")}`, /Duplicate figure id #same/],
    ['<Plate id="fixed" number="9.9" />', /assigned automatically/],
    ["<Plate />", /literal stable id/],
    ['<Plate id={"computed"} />', /literal stable id/],
    ['<Plate id="fixed" {...props} />', /cannot be spread/],
  ]) {
    assert.throws(
      () => createFigureRegistry([document(body)]),
      (error) =>
        error.message.includes("/book/src/pages/chapters/sample.mdx:") &&
        expected.test(error.message),
    );
  }
});

test("rejects conditional figures and actual nested plates rather than guessing their order", () => {
  assert.throws(
    () =>
      createFigureRegistry([document('{true && <Plate id="conditional" />}')]),
    /runtime expressions/,
  );
  assert.throws(
    () =>
      createFigureRegistry([
        document('<Plate id="outer"><Plate id="inner" /></Plate>'),
      ]),
    /nested inside/,
  );
});

test("uses imported component identity, ignores unrelated names, code examples and ordinary pages", () => {
  const unrelated = document(
    '<FigurePlate id="not-imported" />\n\n```mdx\n<Plate id="example" />\n```\n\n' +
      plate("actual"),
  );
  const ordinary = {
    filePath: "/book/src/pages/about.mdx",
    source: '<Plate number="1" />',
  };
  assert.deepEqual(
    createFigureRegistry([unrelated, ordinary]).figures.map(({ id }) => id),
    ["actual"],
  );
});

test("recognises new application diagrams through aliased imports", () => {
  const item = document(
    'import Application from "../../components/figures/ApplicationFigures.astro";\n\n<Application id="figure-11-1" />',
    "11",
  );
  assert.equal(createFigureRegistry([item]).figures[0].number, "11.1");
});

test("filesystem cache detects edits, added pages and removed pages without restart", () => {
  const root = mkdtempSync(join(tmpdir(), "dpt-figure-registry-"));
  const pages = join(root, "src/pages/chapters");
  mkdirSync(pages, { recursive: true });
  const path = join(pages, "sample.mdx");
  try {
    writeFileSync(path, source(plate("a")));
    const initial = getFigureRegistry(root);
    assert.equal(getFigureRegistry(root), initial);
    writeFileSync(path, source(`${plate("b")}\n\n${plate("a")}`));
    assert.equal(
      getFigureRegistry(root).get("/chapters/sample/", "a").number,
      "2.2",
    );
    writeFileSync(join(pages, "extra.mdx"), source(plate("other"), "3"));
    assert.equal(
      getFigureRegistry(root).get("/chapters/extra/", "other").number,
      "3.1",
    );
    renameSync(join(pages, "extra.mdx"), join(pages, "extra.txt"));
    assert.equal(getFigureRegistry(root).hasId("other"), false);
    // A failed parse must not poison the previous successful cache entry.
    writeFileSync(path, source("<Plate id={bad} />"));
    assert.throws(() => getFigureRegistry(root), /literal stable id/);
    writeFileSync(path, source(plate("repaired")));
    assert.equal(
      getFigureRegistry(root).get("/chapters/sample/", "repaired").number,
      "2.1",
    );
  } finally {
    clearFigureRegistry(root);
    rmSync(root, { recursive: true });
  }
});

test("resolves stable local, relative and absolute chapter fragments without external rewriting", () => {
  assert.deepEqual(resolveFigureHref("#old-anchor", "/chapters/one/"), {
    route: "/chapters/one/",
    id: "old-anchor",
  });
  assert.deepEqual(resolveFigureHref("../two/#figure-2-1", "/chapters/one/"), {
    route: "/chapters/two/",
    id: "figure-2-1",
  });
  assert.deepEqual(
    resolveFigureHref(
      "/appendix/x-ray-theory/#c-arm-positioning",
      "/chapters/one/",
    ),
    { route: "/appendix/x-ray-theory/", id: "c-arm-positioning" },
  );
  assert.equal(
    resolveFigureHref("https://example.com/#figure-2-1", "/chapters/one/"),
    null,
  );
  assert.throws(
    () => resolveFigureHref("#%E0%A4", "/chapters/one/"),
    /Malformed figure fragment/,
  );
});
