import { readFile } from "node:fs/promises";
import path from "node:path";

import { readJsonFile } from "./lib/files.mjs";

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function yearOf(item) {
  return item.issued?.["date-parts"]?.[0]?.[0] ?? "n.d.";
}

function authorLabel(item) {
  const authors = item.author?.length ? item.author : (item.editor ?? []);
  if (authors.length === 0) return item.title ?? item.id;
  if (authors.length === 1)
    return authors[0].family ?? authors[0].literal ?? "Unknown";
  if (authors.length === 2) {
    return authors
      .map((author) => author.family ?? author.literal ?? "Unknown")
      .join(" and ");
  }
  return `${authors[0].family ?? authors[0].literal ?? "Unknown"} et al.`;
}

function bibliographyAuthors(item) {
  const contributors = item.author?.length ? item.author : (item.editor ?? []);
  const names = contributors
    .map((author) => {
      if (author.literal) return author.literal;
      return [author.family, author.given].filter(Boolean).join(", ");
    })
    .filter(Boolean);
  const formatted = new Intl.ListFormat("en-GB", {
    type: "conjunction",
  }).format(names);
  if (!formatted || item.author?.length) return formatted;
  return `${formatted} (${contributors.length === 1 ? "ed." : "eds."})`;
}

function sentence(html, plainText = html) {
  if (!html) return "";
  return /[.!?]$/.test(plainText.trim()) ? html : `${html}.`;
}

function identifierLink(item) {
  if (item.DOI) {
    return `<a href="https://doi.org/${encodeURIComponent(item.DOI).replaceAll("%2F", "/")}">https://doi.org/${escapeHtml(item.DOI)}</a>`;
  }
  if (!item.URL) return "";
  const value = String(item.URL).trim();
  try {
    if (!["http:", "https:"].includes(new URL(value).protocol)) return "";
  } catch {
    return "";
  }
  return `<a href="${escapeHtml(value)}">${escapeHtml(value)}</a>`;
}

function styleMetadata(source, stylePath) {
  if (
    !/<style\b[^>]*xmlns="http:\/\/purl\.org\/net\/xbiblio\/csl"/.test(source)
  ) {
    throw new Error(`${stylePath} is not a CSL style`);
  }
  const title =
    source.match(/<title>([^<]+)<\/title>/)?.[1] ?? path.basename(stylePath);
  const format =
    source.match(/<category\b[^>]*citation-format="([^"]+)"/)?.[1] ??
    "author-date";
  return { title, format };
}

export async function loadCitationRenderer({
  root = process.cwd(),
  libraryPath = "references/library.json",
  stylePath = "references/styles/default.csl",
  overridesPath = "references/overrides.yml",
} = {}) {
  const library = await readJsonFile(path.join(root, libraryPath));
  const styleSource = await readFile(path.join(root, stylePath), "utf8");
  const overridesDocument = await readJsonFile(path.join(root, overridesPath));
  const overrides = overridesDocument.overrides ?? {};
  const entries = new Map(
    library.map((item) => [
      item.id,
      { ...item, ...(overrides[item.id] ?? {}), id: item.id },
    ]),
  );
  const style = styleMetadata(styleSource, stylePath);
  const canonicalKeys = [...entries.keys()];
  const citationNumbers = new Map(
    canonicalKeys.map((key, index) => [key, index + 1]),
  );

  function citationNumber(key) {
    const number = citationNumbers.get(key);
    if (number === undefined) throw new Error(`Unknown citation key: ${key}`);
    return number;
  }

  function requireItems(keys) {
    return keys.map((key) => {
      const item = entries.get(key);
      if (!item) throw new Error(`Unknown citation key: ${key}`);
      return item;
    });
  }

  return {
    style,
    keys: [...canonicalKeys],
    citationNumber,
    renderCitation(keys) {
      const items = requireItems(keys);
      if (style.format === "numeric") {
        return `[${keys.map(citationNumber).join(", ")}]`;
      }
      return `(${items.map((item) => `${authorLabel(item)}, ${yearOf(item)}`).join(" and ")})`;
    },
    renderBibliography(keys = canonicalKeys) {
      return requireItems(keys).map((item) => {
        const authors = bibliographyAuthors(item);
        const title = String(item.title ?? item.id).trim();
        const journalTitle = String(item["container-title"] ?? "").trim();
        const journal = journalTitle
          ? `<em>${escapeHtml(journalTitle)}</em>`
          : "";
        const volumeIssue = [
          item.volume,
          item.issue ? `(${item.issue})` : "",
        ].join("");
        const publication = [journal, volumeIssue, item.page]
          .map((part, index) => (index === 0 ? part : escapeHtml(part ?? "")))
          .filter(Boolean)
          .join(", ");
        const publicationText = [journalTitle, volumeIssue, item.page]
          .filter(Boolean)
          .join(", ");
        const publisher = [item["publisher-place"], item.publisher]
          .filter(Boolean)
          .join(": ");
        const parts = [
          authors
            ? `${escapeHtml(authors)} (${escapeHtml(yearOf(item))}).`
            : `${escapeHtml(title)} (${escapeHtml(yearOf(item))}).`,
          authors ? sentence(escapeHtml(title), title) : "",
          sentence(publication, publicationText),
          sentence(escapeHtml(publisher), publisher),
          identifierLink(item),
        ].filter(Boolean);
        return `<span data-citation-key="${escapeHtml(item.id)}">${parts.join(" ")}</span>`;
      });
    },
  };
}
