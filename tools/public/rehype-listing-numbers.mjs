import { chapterNumber } from "./lib/chapter.mjs";

/** Number canonical source listings in chapter document order at build time. */
export default function rehypeListingNumbers() {
  return (tree, file) => {
    const chapter = chapterNumber(file);
    if (chapter === null) return;
    let index = 0;
    function visit(node) {
      if (node.type === "mdxJsxFlowElement" && node.name === "Source") {
        node.attributes ??= [];
        if (
          node.attributes.some(
            (attribute) => attribute.name === "listingNumber",
          )
        ) {
          throw new Error(
            "Source listing numbers are assigned automatically; remove listingNumber.",
          );
        }
        node.attributes.push({
          type: "mdxJsxAttribute",
          name: "listingNumber",
          value: `${chapter}.${++index}`,
        });
      }
      for (const child of node.children ?? []) visit(child);
    }
    visit(tree);
  };
}
