import assert from "node:assert/strict";
import test from "node:test";
import { unified } from "@astrojs/markdown-remark";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

import rehypeEquationNumbers from "./rehype-equation-numbers.mjs";

const chapterLayout = "../../layouts/ChapterLayout.astro";

function elementsWithClass(node, name) {
  const found = node.properties?.className?.includes(name) ? [node] : [];
  return found.concat(
    (node.children ?? []).flatMap((child) => elementsWithClass(child, name)),
  );
}

async function createRenderer() {
  const snapshots = {};
  const processor = unified({
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeKatex,
      () => (tree) => {
        snapshots.before = structuredClone(tree);
      },
      rehypeEquationNumbers,
      () => (tree) => {
        snapshots.after = structuredClone(tree);
      },
    ],
  });
  const renderer = await processor.createMdxRenderer(
    { syntaxHighlight: false },
    { srcDir: new URL("../../src/", import.meta.url) },
  );
  return { renderer, snapshots };
}

test("Astro MDX numbers displays while preserving KaTeX and inline mathematics", async () => {
  const { renderer, snapshots } = await createRenderer();
  const content = [
    "Inline $T_p$ is unnumbered.",
    "",
    "$$",
    "\\lambda_p = n_{0,p} T_p.",
    "$$",
    "",
    "$$",
    "T_p = \\exp(-L_p).",
    "$$",
  ].join("\n");
  const result = await renderer.process(content, "/book/chapter.mdx", {
    layout: chapterLayout,
    chapterNumber: "02",
  });

  const equations = elementsWithClass(snapshots.after, "equation");
  assert.deepEqual(
    equations.map((node) => node.properties["data-equation-number"]),
    ["2.1", "2.2"],
  );
  assert.deepEqual(
    elementsWithClass(snapshots.before, "katex-display"),
    elementsWithClass(snapshots.after, "katex-display"),
  );
  assert.deepEqual(
    elementsWithClass(snapshots.before, "katex"),
    elementsWithClass(snapshots.after, "katex"),
  );
  assert.equal(elementsWithClass(snapshots.after, "katex").length, 3);
  assert.match(result.code, /equation-2-1/);
  assert.match(result.code, /equation-2-2/);

  for (const [index, equation] of equations.entries()) {
    const [scrollRegion, number] = equation.children;
    assert.equal(scrollRegion.properties.tabIndex, 0);
    assert.equal(scrollRegion.properties.ariaLabel, `Equation 2.${index + 1}`);
    assert.equal(
      scrollRegion.children[0].properties.className[0],
      "katex-display",
    );
    assert.equal(number.tagName, "a");
    assert.equal(number.properties.href, `#${equation.properties.id}`);
    assert.equal(number.children[0].value, `(2.${index + 1})`);
  }
});

test("numbering restarts for each chapter and supports appendix prefixes", async () => {
  const { renderer, snapshots } = await createRenderer();
  for (const [chapterNumber, expected] of [
    ["02", ["2.1", "2.2"]],
    ["01", ["1.1", "1.2"]],
    ["A", ["A.1", "A.2"]],
    [10, ["10.1", "10.2"]],
  ]) {
    await renderer.process("$$\nx=1\n$$\n\n$$\ny=2\n$$", "/book/chapter.mdx", {
      layout: chapterLayout,
      chapterNumber,
    });
    assert.deepEqual(
      elementsWithClass(snapshots.after, "equation").map(
        (node) => node.properties["data-equation-number"],
      ),
      expected,
    );
  }
});

test("non-chapter content and absent or invalid chapter numbers stay unnumbered", async () => {
  const { renderer, snapshots } = await createRenderer();
  for (const frontmatter of [
    {},
    { chapterNumber: "02" },
    { layout: "../../layouts/BaseLayout.astro", chapterNumber: "02" },
    { layout: chapterLayout },
    { layout: chapterLayout, chapterNumber: "2.1" },
  ]) {
    await renderer.process("$$\nx=1\n$$", "/book/page.mdx", frontmatter);
    assert.equal(elementsWithClass(snapshots.after, "equation").length, 0);
    assert.deepEqual(snapshots.before, snapshots.after);
  }
});

test("the transform handles nested displays and does not wrap its output twice", () => {
  const tree = {
    type: "root",
    children: [
      {
        type: "element",
        tagName: "section",
        properties: {},
        children: [
          {
            type: "element",
            tagName: "span",
            properties: { className: ["katex-display"] },
            children: [{ type: "text", value: "rendered maths" }],
          },
        ],
      },
    ],
  };
  const file = {
    data: {
      astro: { frontmatter: { layout: chapterLayout, chapterNumber: "02" } },
    },
  };
  const transform = rehypeEquationNumbers();
  transform(tree, file);
  const once = structuredClone(tree);
  transform(tree, file);
  assert.deepEqual(tree, once);
  assert.equal(elementsWithClass(tree, "equation").length, 1);
});
