import { chapterNumber } from "./lib/chapter.mjs";

/** Give each chapter table a native caption and a linkable chapter number. */
export default function rehypeTableNumbers() {
  return (tree, file) => {
    const chapter = chapterNumber(file);
    if (chapter === null) return;
    let index = 0;
    const ids = new Set();
    function visit(node, label) {
      const attributes = node.attributes ?? [];
      const ownLabel =
        node.properties?.ariaLabel ??
        attributes.find((a) => a.name === "aria-label")?.value;
      label = ownLabel ?? label;
      if (node.type === "element" && node.tagName === "table") {
        const number = `${chapter}.${++index}`;
        const id = `table-${chapter}-${index}`;
        if (!label)
          throw new Error(
            `Table ${number} needs a descriptive aria-label on its wrapper.`,
          );
        if (node.children.some((child) => child.tagName === "caption"))
          throw new Error(`Table ${number} already has a caption.`);
        node.properties ??= {};
        node.properties.id = id;
        node.properties["data-table-number"] = number;
        ids.add(id);
        node.children.unshift({
          type: "element",
          tagName: "caption",
          properties: {},
          children: [
            {
              type: "element",
              tagName: "a",
              properties: { href: `#${id}`, className: ["table-number"] },
              children: [{ type: "text", value: `Table ${number}` }],
            },
            { type: "text", value: `. ${label}.` },
          ],
        });
        return;
      }
      for (const child of node.children ?? []) visit(child, label);
    }
    visit(tree);
    const referenced = new Set();
    function references(node) {
      if (node.tagName === "table") return;
      const href =
        node.properties?.href ??
        node.attributes?.find(
          (attribute) =>
            attribute.type === "mdxJsxAttribute" && attribute.name === "href",
        )?.value;
      if (typeof href === "string" && href.startsWith("#table-")) {
        if (!ids.has(href.slice(1)))
          throw new Error(`Unknown table reference ${href}`);
        referenced.add(href.slice(1));
      }
      for (const child of node.children ?? []) references(child);
    }
    references(tree);
    for (const id of ids)
      if (!referenced.has(id))
        throw new Error(`Table #${id} needs an in-text reference.`);
  };
}
