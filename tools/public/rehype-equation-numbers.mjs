import { chapterNumber } from "./lib/chapter.mjs";

function hasClass(node, name) {
  return (
    node.type === "element" &&
    Array.isArray(node.properties?.className) &&
    node.properties.className.includes(name)
  );
}

/** Number KaTeX display equations after rendering, without changing its tree. */
export default function rehypeEquationNumbers() {
  return function transform(tree, file) {
    // Astro supplies this metadata to both Markdown and MDX rehype plugins.
    const prefix = chapterNumber(file);
    if (prefix === null) return;
    let index = 0;

    function visit(parent) {
      if (!Array.isArray(parent.children) || hasClass(parent, "equation"))
        return;

      for (let position = 0; position < parent.children.length; position += 1) {
        const child = parent.children[position];
        if (!hasClass(child, "katex-display")) {
          visit(child);
          continue;
        }

        const number = `${prefix}.${++index}`;
        const id = `equation-${prefix}-${index}`;
        parent.children[position] = {
          type: "element",
          tagName: "div",
          properties: {
            className: ["equation"],
            id,
            "data-equation-number": number,
          },
          children: [
            {
              type: "element",
              tagName: "div",
              properties: {
                className: ["equation__content"],
                role: "group",
                ariaLabel: `Equation ${number}`,
                tabIndex: 0,
              },
              children: [child],
            },
            {
              type: "element",
              tagName: "a",
              properties: {
                className: ["equation__number"],
                href: `#${id}`,
                ariaLabel: `Link to equation ${number}`,
              },
              children: [{ type: "text", value: `(${number})` }],
            },
          ],
        };
      }
    }

    visit(tree);
  };
}
