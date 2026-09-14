import { resolve, sep } from "node:path";
import { chapterNumber } from "./lib/chapter.mjs";
import {
  clearFigureRegistry,
  figureFilePath,
  figureRouteForFile,
  getFigureRegistry,
  resolveFigureHref,
} from "./lib/figure-registry.mjs";

function textLeaves(node, output = []) {
  if (node.type === "text") output.push(node);
  for (const child of node.children ?? []) textLeaves(child, output);
  return output;
}

/** Change only the reference token; retain links, prose and inline elements. */
function replaceFigureTokens(node, number) {
  const leaves = textLeaves(node);
  const spans = [];
  let text = "";
  for (const leaf of leaves) {
    spans.push({
      leaf,
      start: text.length,
      end: text.length + leaf.value.length,
    });
    text += leaf.value;
  }
  const matches = [
    ...text.matchAll(/\b([Ff]igure)(?:\s+(?:\d+|[A-Z]+)\.\d+)?\b/g),
  ];
  for (const match of matches.reverse()) {
    const start = match.index;
    const end = start + match[0].length;
    let inserted = false;
    for (const span of spans) {
      if (span.end <= start || span.start >= end) continue;
      const localStart = Math.max(start - span.start, 0);
      const localEnd = Math.min(end - span.start, span.end - span.start);
      span.leaf.value =
        span.leaf.value.slice(0, localStart) +
        (inserted ? "" : `${match[1]} ${number}`) +
        span.leaf.value.slice(localEnd);
      inserted = true;
    }
  }
}

/** Resolve figure numbers through stable IDs, before Astro renders components. */
export default function rehypeFigureReferences({
  root,
  registry: suppliedRegistry,
} = {}) {
  return (tree, file) => {
    if (chapterNumber(file) === null) return;
    const filePath = figureFilePath(file, root);
    const route = figureRouteForFile(filePath);
    const registry = suppliedRegistry ?? getFigureRegistry(root);
    function visit(node) {
      const isLink =
        (node.type === "element" && node.tagName === "a") ||
        ((node.type === "mdxJsxTextElement" ||
          node.type === "mdxJsxFlowElement") &&
          node.name === "a");
      if (isLink) {
        const href =
          node.properties?.href ??
          node.attributes?.find((attribute) => attribute.name === "href")
            ?.value;
        let target;
        try {
          target = resolveFigureHref(href, route);
        } catch (error) {
          throw new Error(`${filePath}: ${error.message}`, { cause: error });
        }
        const text = textLeaves(node)
          .map((leaf) => leaf.value)
          .join("");
        if (
          target &&
          (registry.has(target.route, target.id) ||
            target.id.startsWith("figure-") ||
            /\b[Ff]igure\b/.test(text))
        ) {
          let figure;
          try {
            figure = registry.get(target.route, target.id);
          } catch (error) {
            throw new Error(
              `${filePath}${node.position?.start?.line ? `:${node.position.start.line}` : ""}: unresolved figure reference ${href}: ${error.message}`,
              { cause: error },
            );
          }
          replaceFigureTokens(node, figure.number);
          if (node.type === "element") {
            node.properties["data-figure-reference"] = figure.id;
            node.properties["data-figure-number"] = figure.number;
          } else {
            node.attributes ??= [];
            for (const [name, value] of [
              ["data-figure-reference", figure.id],
              ["data-figure-number", figure.number],
            ]) {
              node.attributes = node.attributes.filter(
                (attribute) => attribute.name !== name,
              );
              node.attributes.push({ type: "mdxJsxAttribute", name, value });
            }
          }
        }
      }
      for (const child of node.children ?? []) visit(child);
    }
    visit(tree);
  };
}

/** A target edit must invalidate reference-bearing pages already compiled by Vite. */
export function figureReferenceDevPlugin({ root }) {
  const pages = resolve(root, "src", "pages") + sep;
  const isChapter = (file) =>
    typeof file === "string" &&
    resolve(file.split("?")[0]).startsWith(pages) &&
    file.split("?")[0].endsWith(".mdx");
  return {
    name: "dpt-figure-reference-invalidation",
    apply: "serve",
    configureServer(server) {
      const changed = (event, file) => {
        if (!["change", "add", "unlink"].includes(event) || !isChapter(file))
          return;
        clearFigureRegistry(root);
        const graphs = new Set([
          server.moduleGraph,
          ...Object.values(server.environments ?? {}).map(
            (environment) => environment.moduleGraph,
          ),
        ]);
        for (const graph of graphs) {
          if (!graph) continue;
          for (const module of graph.idToModuleMap.values())
            if (isChapter(module.file)) graph.invalidateModule(module);
        }
        server.ws.send({ type: "full-reload", path: "*" });
      };
      server.watcher.on("all", changed);
      server.httpServer?.once("close", () =>
        server.watcher.off("all", changed),
      );
    },
  };
}
