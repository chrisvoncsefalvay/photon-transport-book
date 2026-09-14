const sectionHeading = /^h[2-6]$/;

export const headingPermalinkPath =
  "M10 8H8a4 4 0 0 0 0 8h2M14 8h2a4 4 0 0 1 0 8h-2M8 12h8";

function headingId(node) {
  if (node.type === "element" && sectionHeading.test(node.tagName)) {
    return node.properties?.id;
  }
  if (
    (node.type === "mdxJsxFlowElement" || node.type === "mdxJsxTextElement") &&
    sectionHeading.test(node.name)
  ) {
    return node.attributes?.find((attribute) => attribute.name === "id")?.value;
  }
}

function headingText(node) {
  if (node.type === "text") return node.value;
  if (
    node.tagName === "annotation" ||
    node.properties?.ariaHidden === "true" ||
    node.properties?.ariaHidden === true
  )
    return "";
  if (node.tagName === "img") return node.properties?.alt ?? "";
  return (node.children ?? []).map(headingText).join("");
}

function isPermalink(node) {
  return (
    node.type === "element" &&
    node.tagName === "a" &&
    Array.isArray(node.properties?.className) &&
    node.properties.className.includes("heading-permalink")
  );
}

function permalink(id, label) {
  return {
    type: "element",
    tagName: "a",
    properties: {
      className: ["heading-permalink"],
      href: "#" + encodeURIComponent(id),
      ariaLabel: "Link to section " + label,
    },
    children: [
      {
        type: "element",
        tagName: "svg",
        properties: {
          viewBox: "0 0 24 24",
          fill: "none",
          stroke: "currentColor",
          strokeWidth: 2,
          strokeLinecap: "round",
          strokeLinejoin: "round",
          ariaHidden: "true",
          focusable: "false",
        },
        children: [
          {
            type: "element",
            tagName: "path",
            properties: {
              d: headingPermalinkPath,
            },
            children: [],
          },
        ],
      },
    ],
  };
}

/** Append static links without adding text to headings or changing their IDs. */
export default function rehypeHeadingPermalinks() {
  return (tree, file) => {
    const layout = file.data?.astro?.frontmatter?.layout;
    if (
      typeof layout !== "string" ||
      !/(^|\/)ChapterLayout\.astro$/.test(layout)
    ) {
      return;
    }

    function visit(node) {
      const id = headingId(node);
      if (typeof id === "string" && id.length > 0) {
        node.children ??= [];
        if (!node.children.some(isPermalink)) {
          const label = headingText(node).replace(/\s+/g, " ").trim() || id;
          node.children.push(permalink(id, label));
        }
        return;
      }
      for (const child of node.children ?? []) visit(child);
    }

    visit(tree);
  };
}
