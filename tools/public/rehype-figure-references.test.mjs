import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { createFigureRegistry } from "./lib/figure-registry.mjs";
import rehypeFigureReferences, {
  figureReferenceDevPlugin,
} from "./rehype-figure-references.mjs";

const item = (slug, chapter, ids) => ({
  filePath: `/book/src/pages/chapters/${slug}.mdx`,
  source: `---\nlayout: ../../layouts/ChapterLayout.astro\nchapterNumber: ${chapter}\n---\nimport Plate from "../../components/figures/FigurePlate.astro";\n\n${ids.map((id) => `<Plate id="${id}" />`).join("\n\n")}`,
});
const registry = createFigureRegistry([
  item("one", "02", ["first", "figure-2-1", "last"]),
  item("two", "A", ["historic-anchor"]),
]);
const file = {
  path: "/book/src/pages/chapters/one.mdx",
  data: {
    astro: {
      frontmatter: {
        layout: "../../layouts/ChapterLayout.astro",
        chapterNumber: "02",
      },
    },
  },
};
const text = (value) => ({ type: "text", value });
const link = (href, ...children) => ({
  type: "element",
  tagName: "a",
  properties: { href },
  children,
});
const textContent = (node) =>
  node.type === "text"
    ? node.value
    : (node.children ?? []).map(textContent).join("");
const transform = (tree) => rehypeFigureReferences({ registry })(tree, file);

test("replaces bare and legacy Figure tokens while retaining surrounding words and href", () => {
  for (const [value, expected] of [
    ["Figure", "Figure 2.2"],
    ["Figure 99.42", "Figure 2.2"],
    ["The chain in Figure A.8", "The chain in Figure 2.2"],
    ["figure", "figure 2.2"],
  ]) {
    const node = link("#figure-2-1", text(value));
    transform({ children: [node] });
    assert.equal(textContent(node), expected);
    assert.equal(node.properties.href, "#figure-2-1");
    assert.equal(node.properties["data-figure-reference"], "figure-2-1");
  }
});

test("retains mixed nested inline elements when the token crosses text nodes", () => {
  const strong = {
    type: "element",
    tagName: "strong",
    properties: {},
    children: [
      text("Fig"),
      { type: "element", tagName: "em", children: [text("ure")] },
    ],
  };
  const node = link(
    "#figure-2-1",
    text("See "),
    strong,
    text(" 7.8 for the result"),
  );
  transform({ children: [node] });
  assert.equal(textContent(node), "See Figure 2.2 for the result");
  assert.equal(node.children[1], strong);
  assert.equal(strong.children[1].tagName, "em");
});

test("resolves cross-chapter numbers even when the target page has not rendered", () => {
  const relative = link("../two/#historic-anchor", text("Figure"));
  const absolute = link("/chapters/two/#historic-anchor", text("Figure 2.3"));
  transform({ children: [relative, absolute] });
  assert.equal(textContent(relative), "Figure A.1");
  assert.equal(textContent(absolute), "Figure A.1");
});

test("validates descriptive links without forcing the word Figure", () => {
  const node = link("#first", text("the selected ray"));
  transform({ children: [node] });
  assert.equal(textContent(node), "the selected ray");
  assert.equal(node.properties["data-figure-number"], "2.1");
});

test("an opaque figure ID in another chapter does not reserve an ordinary heading", () => {
  const node = link("#historic-anchor", text("the normal section"));
  transform({ children: [node] });
  assert.equal(textContent(node), "the normal section");
  assert.equal(node.properties["data-figure-reference"], undefined);
});

test("handles authored JSX anchors and leaves external and unrelated chapter links intact", () => {
  const node = {
    type: "mdxJsxTextElement",
    name: "a",
    attributes: [{ type: "mdxJsxAttribute", name: "href", value: "#last" }],
    children: [text("Figure")],
  };
  const external = link(
    "https://example.com/#figure-missing",
    text("Figure 9.1"),
  );
  const section = link("#ordinary-section", text("the derivation"));
  transform({ children: [node, external, section] });
  assert.equal(textContent(node), "Figure 2.3");
  assert.equal(textContent(external), "Figure 9.1");
  assert.equal(section.properties["data-figure-number"], undefined);
});

test("rejects missing fragments and wrong-page historical IDs with useful source diagnostics", () => {
  for (const href of [
    "#figure-missing",
    "#historic-anchor",
    "../missing/#first",
  ])
    assert.throws(
      () => transform({ children: [link(href, text("Figure"))] }),
      (error) =>
        error.message.includes(file.path) &&
        error.message.includes(href) &&
        /unresolved figure reference/.test(error.message),
    );
});

test("ordinary pages are untouched and repeated transformations are idempotent", () => {
  const ordinary = link("#figure-missing", text("Figure"));
  rehypeFigureReferences({ registry })({ children: [ordinary] }, {});
  assert.equal(textContent(ordinary), "Figure");
  const node = link("#last", text("Figure 9.2 and Figure"));
  transform({ children: [node] });
  transform({ children: [node] });
  assert.equal(textContent(node), "Figure 2.3 and Figure 2.3");
});

test("dev invalidation reaches client and SSR chapter modules on edits, additions and deletions", () => {
  const watcher = new EventEmitter();
  const httpServer = new EventEmitter();
  const invalidated = [];
  const sent = [];
  const chapter = { file: "/book/src/pages/chapters/one.mdx" };
  const target = { file: "/book/src/pages/chapters/two.mdx" };
  const unrelated = { file: "/book/src/lib/other.ts" };
  const graph = (modules) => ({
    idToModuleMap: new Map(modules.map((module) => [module.file, module])),
    invalidateModule(module) {
      invalidated.push(module);
    },
  });
  const client = graph([chapter, unrelated]);
  const ssr = graph([target]);
  figureReferenceDevPlugin({ root: "/book" }).configureServer({
    watcher,
    httpServer,
    moduleGraph: client,
    environments: {
      client: { moduleGraph: client },
      ssr: { moduleGraph: ssr },
    },
    ws: {
      send(message) {
        sent.push(message);
      },
    },
  });
  watcher.emit("all", "change", unrelated.file);
  assert.equal(sent.length, 0);
  for (const event of ["change", "add", "unlink"])
    watcher.emit("all", event, target.file);
  assert.deepEqual(invalidated, [
    chapter,
    target,
    chapter,
    target,
    chapter,
    target,
  ]);
  assert.deepEqual(sent, Array(3).fill({ type: "full-reload", path: "*" }));
  httpServer.emit("close");
  assert.equal(watcher.listenerCount("all"), 0);
});
