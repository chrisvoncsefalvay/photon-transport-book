import test from "node:test";
import assert from "node:assert/strict";
import rehypeListingNumbers from "./rehype-listing-numbers.mjs";

const listing = () => ({
  type: "mdxJsxFlowElement",
  name: "Source",
  attributes: [],
});
const file = (chapterNumber = "02") => ({
  data: {
    astro: {
      frontmatter: {
        layout: "../../layouts/ChapterLayout.astro",
        chapterNumber,
      },
    },
  },
});
const number = (node) =>
  node.attributes.find((a) => a.name === "listingNumber")?.value;

test("numbers nested listings in reading order and resets for each chapter", () => {
  const a = listing(),
    b = listing(),
    c = listing();
  const transform = rehypeListingNumbers();
  transform({ children: [a, { children: [b] }] }, file());
  transform({ children: [c] }, file("03"));
  assert.deepEqual([number(a), number(b), number(c)], ["2.1", "2.2", "3.1"]);
});
test("supports appendix chapters", () => {
  const a = listing();
  rehypeListingNumbers()({ children: [a] }, file("D"));
  assert.equal(number(a), "D.1");
});
test("ignores ordinary pages and invalid chapter metadata", () => {
  for (const metadata of [{}, file(""), file("2.6")]) {
    const a = listing();
    rehypeListingNumbers()({ children: [a] }, metadata);
    assert.equal(number(a), undefined);
  }
});
test("rejects manually assigned numbers", () => {
  const a = listing();
  a.attributes.push({ name: "listingNumber", value: "8.8" });
  assert.throws(
    () => rehypeListingNumbers()({ children: [a] }, file()),
    /automatically/,
  );
});
