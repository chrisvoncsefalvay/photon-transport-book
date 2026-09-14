import assert from "node:assert/strict";
import test from "node:test";
import { rehypeHeadingIds, unified } from "@astrojs/markdown-remark";

import rehypeHeadingPermalinks, {
  headingPermalinkPath,
} from "./rehype-heading-permalinks.mjs";

const layout = "../../layouts/ChapterLayout.astro";

function elements(node, predicate) {
  return [
    ...(predicate(node) ? [node] : []),
    ...(node.children ?? []).flatMap((child) => elements(child, predicate)),
  ];
}

function textContent(node) {
  return (
    (node.type === "text" ? node.value : "") +
    (node.children ?? []).map(textContent).join("")
  );
}

function sectionHeadings(tree) {
  return elements(
    tree,
    (node) => node.type === "element" && /^h[2-6]$/.test(node.tagName),
  );
}

function fixture(headings, frontmatter = { layout }) {
  return {
    tree: { type: "root", children: headings },
    file: { data: { astro: { frontmatter } }, history: ["/book/chapter.mdx"] },
  };
}

function heading(tagName = "h2", id = "a-section", text = "A section") {
  return {
    type: "element",
    tagName,
    properties: id === undefined ? {} : { id },
    children: [{ type: "text", value: text }],
  };
}

async function renderMdx(content, plugins, frontmatter = { layout }) {
  let captured;
  let file;
  const processor = unified({
    rehypePlugins: [
      ...plugins,
      () => (tree, source) => {
        captured = structuredClone(tree);
        file = source;
      },
    ],
  });
  const renderer = await processor.createMdxRenderer(
    { syntaxHighlight: false },
    { srcDir: new URL("../../src/", import.meta.url) },
  );
  const result = await renderer.process(
    content,
    "/book/chapter.mdx",
    frontmatter,
  );
  return {
    ...result,
    tree: captured,
    headings: structuredClone(file.data.astro.headings),
  };
}

test("Astro MDX preserves duplicate slugs, rich heading text and TOC metadata", async () => {
  const codeMark = String.fromCharCode(96);
  const content = [
    "# Chapter title",
    "",
    "## 2.7 Checks before adding geometry",
    "",
    "## Repeated heading",
    "",
    "## Repeated heading",
    "",
    "### Detailed *physical* " + codeMark + "n0" + codeMark + " checks",
    "",
    "#### Further details",
    "",
    "###### Last level",
  ].join("\n");
  const baseline = await renderMdx(content, []);
  const linked = await renderMdx(content, [
    rehypeHeadingIds,
    rehypeHeadingPermalinks,
  ]);

  assert.deepEqual(linked.headings, baseline.headings);
  assert.deepEqual(
    linked.headings.map((item) => item.slug),
    [
      "chapter-title",
      "27-checks-before-adding-geometry",
      "repeated-heading",
      "repeated-heading-1",
      "detailed-physical-n0-checks",
      "further-details",
      "last-level",
    ],
  );
  const headings = sectionHeadings(linked.tree);
  assert.equal(headings.length, 6);
  for (const node of headings) {
    const link = node.children.at(-1);
    assert.equal(link.tagName, "a");
    assert.deepEqual(link.properties.className, ["heading-permalink"]);
    assert.equal(link.properties.href, "#" + node.properties.id);
    assert.equal(
      link.properties.ariaLabel,
      "Link to section " + textContent(node),
    );
    assert.equal(textContent(link), "");
    assert.equal(link.properties.tabIndex, undefined);
    assert.equal(link.children[0].properties.ariaHidden, "true");
    assert.equal(link.children[0].properties.focusable, "false");
    assert.equal(
      link.children[0].children[0].properties.d,
      headingPermalinkPath,
    );
  }
  const title = elements(linked.tree, (node) => node.tagName === "h1")[0];
  assert.equal(title.children.length, 1);
  assert.match(linked.code, /heading-permalink/);
});

test("repeated heading-ID and permalink transforms preserve the tree and TOC", () => {
  const { tree, file } = fixture([
    heading("h2", null, "Repeated"),
    heading("h2", null, "Repeated"),
    heading("h3", "author-defined", "A named subsection"),
  ]);
  const ids = rehypeHeadingIds();
  const links = rehypeHeadingPermalinks();
  ids(tree, file);
  links(tree, file);
  const once = structuredClone(tree);
  const headings = structuredClone(file.data.astro.headings);
  ids(tree, file);
  links(tree, file);
  assert.deepEqual(tree, once);
  assert.deepEqual(file.data.astro.headings, headings);
  assert.deepEqual(
    tree.children.map((node) => node.properties.id),
    ["repeated", "repeated-1", "author-defined"],
  );
});

test("Markdown emits native fragment links without JavaScript", async () => {
  const renderer = await unified({
    rehypePlugins: [rehypeHeadingIds, rehypeHeadingPermalinks],
  }).createRenderer({ syntaxHighlight: false });
  const result = await renderer.render("## A.1 Photon interactions", {
    fileURL: new URL("file:///book/appendix.md"),
    frontmatter: { layout, chapterNumber: "A" },
  });
  assert.match(result.code, /<h2 id="a1-photon-interactions">/);
  assert.match(
    result.code,
    /<a class="heading-permalink" href="#a1-photon-interactions" aria-label="Link to section A\.1 Photon interactions">/,
  );
  assert.doesNotMatch(result.code, /<script|onclick|tabindex="-1"/);
});

test("existing static MDX heading IDs and nested sections are supported", async () => {
  const content = [
    '<section><h4 id="custom-section">A custom section</h4></section>',
    "",
    '<h3 id="energy-μ">Energy response</h3>',
  ].join("\n");
  const linked = await renderMdx(content, [
    rehypeHeadingIds,
    rehypeHeadingPermalinks,
  ]);
  const anchors = elements(linked.tree, (node) => node.tagName === "a");
  assert.equal(anchors.length, 2);
  assert.equal(anchors[0].properties.href, "#custom-section");
  assert.equal(anchors[1].properties.href, "#energy-%CE%BC");
  assert.equal(
    anchors[1].properties.ariaLabel,
    "Link to section Energy response",
  );
});

test("only chapter-layout sections with static nonempty IDs receive links", () => {
  const source = [
    heading("h1", "title", "Title"),
    heading("h2", "", "No ID"),
    heading("h3", null, "No ID"),
    heading("h4", "details", "Details"),
    heading("h5", "more-details", "More details"),
    heading("h6", "last-details", "Last details"),
  ];
  for (const frontmatter of [
    {},
    { layout: "../../layouts/BaseLayout.astro" },
    { layout: "../../layouts/NotChapterLayout.astro" },
  ]) {
    const { tree, file } = fixture(structuredClone(source), frontmatter);
    const original = structuredClone(tree);
    rehypeHeadingPermalinks()(tree, file);
    assert.deepEqual(tree, original);
  }
  const { tree, file } = fixture(structuredClone(source));
  rehypeHeadingPermalinks()(tree, file);
  assert.deepEqual(
    tree.children.map((node) => node.children.length),
    [1, 1, 1, 2, 2, 2],
  );
});

test("permalinks preserve authored heading children and existing links", () => {
  const node = heading("h2", "units & paths", "Units ");
  node.children.push({
    type: "element",
    tagName: "a",
    properties: { href: "/appendix/" },
    children: [{ type: "text", value: "and paths" }],
  });
  const original = structuredClone(node.children);
  const { tree, file } = fixture([node]);
  rehypeHeadingPermalinks()(tree, file);
  assert.deepEqual(node.children.slice(0, -1), original);
  assert.equal(node.children.at(-1).properties.href, "#units%20%26%20paths");
  assert.equal(
    node.children.at(-1).properties.ariaLabel,
    "Link to section Units and paths",
  );
});
