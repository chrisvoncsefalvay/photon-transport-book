import test from "node:test";
import assert from "node:assert/strict";
import rehypeTableNumbers from "./rehype-table-numbers.mjs";

function fixture(
  chapter = "02",
  href = "#table-2-1",
  label = "Quantities and units",
) {
  const table = {
    type: "element",
    tagName: "table",
    properties: {},
    children: [],
  };
  const tree = {
    children: [
      {
        type: "element",
        tagName: "p",
        children: [
          { type: "element", tagName: "a", properties: { href }, children: [] },
        ],
      },
      {
        type: "mdxJsxFlowElement",
        name: "div",
        attributes: [{ name: "aria-label", value: label }],
        children: [table],
      },
    ],
  };
  const file = {
    data: {
      astro: {
        frontmatter: {
          layout: "../../layouts/ChapterLayout.astro",
          chapterNumber: chapter,
        },
      },
    },
  };
  return { tree, file, table };
}

test("creates a native linked caption for MDX tables", () => {
  const { tree, file, table } = fixture();
  rehypeTableNumbers()(tree, file);
  assert.equal(table.properties.id, "table-2-1");
  assert.equal(table.children[0].tagName, "caption");
  assert.equal(table.children[0].children[0].children[0].value, "Table 2.1");
});
test("supports appendix numbering", () => {
  const { tree, file, table } = fixture("A", "#table-A-1");
  rehypeTableNumbers()(tree, file);
  assert.equal(table.properties.id, "table-A-1");
});
test("rejects missing labels and broken references", () => {
  for (const f of [
    fixture("2", "#table-2-1", ""),
    fixture("2", "#table-2-2"),
    fixture("2", "#other"),
  ]) {
    assert.throws(() => rehypeTableNumbers()(f.tree, f.file));
  }
});
test("ignores non-chapter pages", () => {
  const { tree, table } = fixture();
  rehypeTableNumbers()(tree, {});
  assert.equal(table.children.length, 0);
});

test("numbers multiple tables and requires a reference for each", () => {
  const first = fixture();
  const second = fixture("02", "#table-2-2", "Operator checks");
  first.tree.children.push(...second.tree.children);
  rehypeTableNumbers()(first.tree, first.file);
  assert.equal(first.table.properties.id, "table-2-1");
  assert.equal(second.table.properties.id, "table-2-2");
  const missing = fixture();
  missing.tree.children.push(fixture().tree.children[1]);
  assert.throws(
    () => rehypeTableNumbers()(missing.tree, missing.file),
    /needs an in-text reference/,
  );
});

test("counts literal MDX JSX anchors and rejects broken JSX targets", () => {
  for (const href of ["#table-2-1", "#table-2-2"]) {
    const { tree, file } = fixture();
    tree.children[0].children = [
      {
        type: "mdxJsxTextElement",
        name: "a",
        attributes: [{ type: "mdxJsxAttribute", name: "href", value: href }],
        children: [],
      },
    ];
    if (href === "#table-2-1")
      assert.doesNotThrow(() => rehypeTableNumbers()(tree, file));
    else
      assert.throws(
        () => rehypeTableNumbers()(tree, file),
        /Unknown table reference/,
      );
  }
});
