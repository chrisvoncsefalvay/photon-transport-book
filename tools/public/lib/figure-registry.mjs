import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { createProcessor } from "@mdx-js/mdx";
import { parseFrontmatter } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import { chapterNumber } from "./chapter.mjs";

/** Only complete plates belong here; charts and other plate contents do not. */
export const figureComponents = new Set([
  "FigurePlate",
  "FigurePlaceholder",
  "AnatomyFigure",
  "ApplicationFigures",
  "AcquisitionResults",
  "CArmFigure",
  "ConceptualFigures",
  "DirectionalChecksFigure",
  "ExactIllustrations",
  "GeometryIllustrations",
  "IntroductionStudy",
  "MaterialResponseFigure",
  "PixelCentresFigure",
  "PoseSensitivityFigure",
  "ProjectionSandbox",
  "RadiographFigure",
  "RadiographStudy",
  "RecordedFigure",
  "RegistrationResults",
  "ReconstructionComparison",
  "ReconstructionFigure",
  "ReconstructionRadiographs",
  "SpectralRadiographs",
  "SelectedRayFigure",
  "TransmissionAveragingFigure",
  "WorkedExample",
]);

const parser = createProcessor({ format: "mdx", remarkPlugins: [remarkMath] });
const caches = new Map();

export function normaliseFigureRoute(route) {
  const path = route.replace(/\\/g, "/").replace(/\/+$/, "");
  return `${path || ""}/`;
}

export function figureRouteForFile(filePath) {
  const path = filePath.replace(/\\/g, "/");
  const marker = "/src/pages/";
  const start = path.lastIndexOf(marker);
  if (start < 0 || !path.endsWith(".mdx"))
    throw new Error(`Cannot determine figure route for ${filePath}`);
  const route = path
    .slice(start + marker.length)
    .slice(0, -4)
    .replace(/(?:^|\/)index$/, "");
  return normaliseFigureRoute(`/${route}`);
}

function fail(filePath, node, message) {
  const line = node.position?.start?.line;
  throw new Error(`${filePath}${line ? `:${line}` : ""}: ${message}`);
}

function collectDocument(filePath, source) {
  const { frontmatter, content } = parseFrontmatter(source, {
    frontmatter: "empty-with-lines",
  });
  const prefix = chapterNumber({ data: { astro: { frontmatter } } });
  if (prefix === null) return null;
  const tree = parser.parse(content);
  const names = new Set();
  for (const node of tree.children) {
    for (const statement of node.data?.estree?.body ?? []) {
      if (statement.type !== "ImportDeclaration") continue;
      const imported = String(statement.source.value).match(
        /(?:^|\/)components\/figures\/([^/]+)\.astro$/,
      )?.[1];
      if (!figureComponents.has(imported)) continue;
      for (const specifier of statement.specifiers)
        if (specifier.type === "ImportDefaultSpecifier")
          names.add(specifier.local.name);
    }
  }
  const route = figureRouteForFile(filePath);
  const figures = [];
  const ids = new Set();
  function visit(node, insidePlate = false) {
    const isPlate =
      (node.type === "mdxJsxFlowElement" ||
        node.type === "mdxJsxTextElement") &&
      names.has(node.name);
    if (isPlate) {
      if (insidePlate)
        fail(
          filePath,
          node,
          "A numbered figure cannot be nested inside another figure; use a surface component inside the outer plate.",
        );
      const attributes = node.attributes ?? [];
      if (attributes.some((attribute) => attribute.name === "number"))
        fail(
          filePath,
          node,
          "Figure numbers are assigned automatically; remove number.",
        );
      if (
        attributes.some(
          (attribute) => attribute.type === "mdxJsxExpressionAttribute",
        )
      )
        fail(
          filePath,
          node,
          "Figure props cannot be spread: declare a literal stable id.",
        );
      const id = attributes.find((attribute) => attribute.name === "id")?.value;
      if (typeof id !== "string" || !/^[A-Za-z][\w:.-]*$/.test(id))
        fail(filePath, node, `${node.name} requires a literal stable id.`);
      if (ids.has(id)) fail(filePath, node, `Duplicate figure id #${id}.`);
      ids.add(id);
      figures.push(
        Object.freeze({
          id,
          number: `${prefix}.${figures.length + 1}`,
          route,
          filePath,
        }),
      );
    }
    // Runtime conditionals/maps have no single static reading order.
    if (
      node.type === "mdxFlowExpression" ||
      node.type === "mdxTextExpression"
    ) {
      function expression(value) {
        if (!value || typeof value !== "object") return;
        if (value.type === "JSXOpeningElement" && names.has(value.name?.name))
          fail(
            filePath,
            node,
            "Figure declarations must appear directly in the chapter, outside runtime expressions.",
          );
        for (const child of Object.values(value)) {
          if (Array.isArray(child)) child.forEach(expression);
          else expression(child);
        }
      }
      expression(node.data?.estree);
    }
    for (const child of node.children ?? [])
      visit(child, insidePlate || isPlate);
  }
  visit(tree);
  return { route, filePath, figures };
}

function assemble(documents) {
  const pages = new Map();
  const knownIds = new Set();
  for (const document of documents) {
    if (!document) continue;
    if (pages.has(document.route))
      throw new Error(`Duplicate figure page route ${document.route}`);
    pages.set(
      document.route,
      new Map(document.figures.map((figure) => [figure.id, figure])),
    );
    document.figures.forEach((figure) => knownIds.add(figure.id));
  }
  return Object.freeze({
    get(route, id) {
      const normalised = normaliseFigureRoute(route);
      const figure = pages.get(normalised)?.get(id);
      if (!figure) throw new Error(`Unknown figure #${id} on ${normalised}`);
      return figure;
    },
    has(route, id) {
      return pages.get(normaliseFigureRoute(route))?.has(id) ?? false;
    },
    hasId(id) {
      return knownIds.has(id);
    },
    figures: Object.freeze(
      documents.flatMap((document) => document?.figures ?? []),
    ),
  });
}

/** Pure source-to-registry boundary, shared by tests and the filesystem cache. */
export function createFigureRegistry(documents) {
  return assemble(
    documents.map(({ filePath, source }) => collectDocument(filePath, source)),
  );
}

function chapterFiles(root) {
  const files = [];
  function walk(directory) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const filePath = join(directory, entry.name);
      if (entry.isDirectory()) walk(filePath);
      else if (entry.isFile() && entry.name.endsWith(".mdx"))
        files.push(filePath);
    }
  }
  walk(join(root, "src", "pages"));
  return files.sort();
}

/** Server/build-only. Check the complete file set and stamps on every lookup. */
export function getFigureRegistry(root) {
  if (typeof root !== "string" || !root)
    throw new Error(
      "Figure registry requires the configured DPT_PROJECT_ROOT.",
    );
  root = resolve(root);
  let cache = caches.get(root);
  if (!cache) {
    cache = { files: new Map(), registry: null };
    caches.set(root, cache);
  }
  const files = chapterFiles(root);
  let changed = files.length !== cache.files.size;
  const next = new Map();
  for (const filePath of files) {
    const stats = statSync(filePath, { bigint: true });
    const stamp = `${stats.mtimeNs}:${stats.ctimeNs}:${stats.size}`;
    let entry = cache.files.get(filePath);
    if (entry?.stamp !== stamp) {
      entry = {
        stamp,
        document: collectDocument(filePath, readFileSync(filePath, "utf8")),
      };
      changed = true;
    }
    next.set(filePath, entry);
  }
  if (changed || !cache.registry) {
    const registry = assemble(
      [...next.values()].map((entry) => entry.document),
    );
    cache.files = next;
    cache.registry = registry;
  }
  return cache.registry;
}

export function clearFigureRegistry(root) {
  caches.delete(resolve(root));
}

export function resolveFigureHref(href, currentRoute) {
  if (
    typeof href !== "string" ||
    !href.includes("#") ||
    /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(href)
  )
    return null;
  const url = new URL(
    href,
    `https://book.invalid${normaliseFigureRoute(currentRoute)}`,
  );
  if (!url.hash) return null;
  let id;
  try {
    id = decodeURIComponent(url.hash.slice(1));
  } catch {
    throw new Error(`Malformed figure fragment in ${href}`);
  }
  return { route: normaliseFigureRoute(url.pathname), id };
}

/** Absolute file paths make errors usable in both the build and the dev overlay. */
export function figureFilePath(file, root) {
  return typeof file.path === "string"
    ? resolve(file.path)
    : join(root, "src", "pages", "unknown.mdx");
}
