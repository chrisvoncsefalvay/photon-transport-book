import { readFile } from "node:fs/promises";
import path from "node:path";
import { isDeepStrictEqual } from "node:util";
import { parseDocument } from "yaml";

/**
 * @typedef {Object} BookCitation
 * @property {string} title
 * @property {{family: string, given: string}[]} authors
 * @property {string} version Version without a leading v.
 * @property {string} date Canonical CFF date in YYYY-MM-DD form.
 * @property {string} repositoryCode
 * @property {boolean} released Whether CFF supplies a DOI; no network publication check.
 * @property {string} [doi]
 * @property {string} [url]
 * @property {string} [publisher]
 * @property {string} [sourceCommit]
 */

/**
 * @typedef {Object} PageCitationOptions
 * @property {string} title
 * @property {string} pathname
 * @property {string | number} [chapterNumber]
 * @property {'chapter' | 'appendix' | 'page' | 'book'} part
 */

/**
 * @typedef {Object} PageCitation
 * @property {string} bibtex
 * @property {string} key
 * @property {string} filename
 * @property {string} label
 * @property {string} statusText
 * @property {{authors: string, year: string, title: string, container?: string, edition: string, url: string, draft: boolean}} formatted
 * @property {string} [doi]
 */

const VERSION =
  /^\d+\.\d+\.\d+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?(?:\+[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?$/;
const DOI = /^10\.\d{4,9}(?:\.\d+)?\/[A-Za-z0-9:/_;.()\[\]\\-]+$/;
const ZENODO_DOI = /^10\.5281\/zenodo\.[1-9]\d*$/;
const COMMIT = /^[0-9a-f]{40,64}$/;

function text(value, field) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${field} must be a non-empty string`);
  }
  if (/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/u.test(value)) {
    throw new Error(`${field} contains control characters`);
  }
  return value.trim().replace(/\s+/gu, " ");
}

function version(value, field) {
  const result = text(value, field).replace(/^v/, "");
  if (!VERSION.test(result)) {
    throw new Error(`${field} must be a version such as 0.1.0 or v0.1.0`);
  }
  return result;
}

function date(value) {
  const result = text(value, "CITATION.cff date-released");
  const parsed = new Date(`${result}T00:00:00.000Z`);
  if (
    !/^\d{4}-\d{2}-\d{2}$/.test(result) ||
    Number.isNaN(parsed.valueOf()) ||
    parsed.toISOString().slice(0, 10) !== result
  ) {
    throw new Error(
      "CITATION.cff date-released must be a valid YYYY-MM-DD date",
    );
  }
  return result;
}

function publicUrl(value, field) {
  const result = text(value, field);
  let parsed;
  try {
    parsed = new URL(result);
  } catch {
    throw new Error(`${field} must be an absolute HTTP(S) URL`);
  }
  if (
    !["https:", "http:"].includes(parsed.protocol) ||
    parsed.username ||
    parsed.password
  ) {
    throw new Error(
      `${field} must be an absolute HTTP(S) URL without credentials`,
    );
  }
  return result;
}

async function readPreparedMetadata(root) {
  let source;
  try {
    source = await readFile(
      path.join(root, "public/release-metadata.json"),
      "utf8",
    );
  } catch (error) {
    if (error.code === "ENOENT") return undefined;
    throw error;
  }
  try {
    return JSON.parse(source);
  } catch (error) {
    throw new Error(
      `Cannot parse public/release-metadata.json: ${error.message}`,
    );
  }
}

/**
 * Read authoritative citation fields; never use the build clock or preview URL.
 * @param {{root?: string, expectedVersion?: string, requireDoi?: boolean}} [options]
 * @returns {Promise<BookCitation>}
 */
export async function loadBookCitation({
  root = process.cwd(),
  expectedVersion,
  requireDoi = false,
} = {}) {
  const source = await readFile(path.join(root, "CITATION.cff"), "utf8");
  const document = parseDocument(source, { uniqueKeys: true });
  if (document.errors.length) {
    throw new Error(`Cannot parse CITATION.cff: ${document.errors[0].message}`);
  }
  const cff = document.toJS({ maxAliasCount: 100 });
  if (!cff || typeof cff !== "object" || Array.isArray(cff)) {
    throw new Error("CITATION.cff must contain a mapping");
  }
  if (!Array.isArray(cff.authors) || !cff.authors.length) {
    throw new Error("CITATION.cff authors must contain at least one person");
  }
  /** @type {BookCitation} */
  const book = {
    title: text(cff.title, "CITATION.cff title"),
    authors: cff.authors.map((author, index) => ({
      family: text(
        author?.["family-names"],
        `CITATION.cff authors[${index}].family-names`,
      ),
      given:
        author?.["given-names"] === undefined
          ? ""
          : text(
              author["given-names"],
              `CITATION.cff authors[${index}].given-names`,
            ),
    })),
    version: version(cff.version, "CITATION.cff version"),
    date: date(cff["date-released"]),
    repositoryCode: publicUrl(
      cff["repository-code"],
      "CITATION.cff repository-code",
    ),
    released: false,
  };
  if (
    expectedVersion !== undefined &&
    book.version !== version(expectedVersion, "Requested release version")
  ) {
    throw new Error(
      `CITATION.cff version ${book.version} does not match requested release ${expectedVersion}`,
    );
  }
  if (cff.url !== undefined) book.url = publicUrl(cff.url, "CITATION.cff url");
  if (cff.doi !== undefined) {
    const doi = text(cff.doi, "CITATION.cff doi");
    if (
      !DOI.test(doi) ||
      (/^10\.5281\/zenodo\./i.test(doi) && !ZENODO_DOI.test(doi))
    ) {
      throw new Error(
        "CITATION.cff doi must be a DOI identifier, without a resolver URL or placeholder",
      );
    }
    book.doi = doi;
    book.released = true;
  }
  if (requireDoi && !ZENODO_DOI.test(book.doi ?? "")) {
    throw new Error(
      "Release preparation requires a reserved version-specific Zenodo DOI in CITATION.cff",
    );
  }
  if (book.doi && Array.isArray(cff.identifiers)) {
    const concept = cff.identifiers.find(
      (identifier) =>
        identifier?.type === "doi" &&
        identifier.value === book.doi &&
        /concept|all[ -]?versions/i.test(identifier.description ?? ""),
    );
    if (concept)
      throw new Error(
        "CITATION.cff doi identifies all versions; use the specific edition's Zenodo DOI",
      );
  }
  const metadata = await readPreparedMetadata(root);
  if (metadata !== undefined) {
    if (
      metadata?.schema_version !== 1 ||
      metadata.public_schema_version !== 1 ||
      metadata.version !== `v${book.version}` ||
      !COMMIT.test(metadata.source_commit ?? "") ||
      !ZENODO_DOI.test(book.doi ?? "")
    ) {
      throw new Error(
        "Prepared release metadata is invalid or does not match CITATION.cff",
      );
    }
    const frozenCitation = { ...book, sourceCommit: metadata.source_commit };
    if (!isDeepStrictEqual(metadata.citation, frozenCitation)) {
      throw new Error(
        "Prepared release citation does not match CITATION.cff; rebuild the edition from its canonical citation",
      );
    }
    return frozenCitation;
  }
  return book;
}

// Commands for literal braces avoid unbalanced BibTeX delimiters even in an
// adversarial title. Replacing each original character once avoids re-escaping.
function bibtexText(value) {
  const escapes = {
    "\\": "\\textbackslash{}",
    "{": "\\textbraceleft{}",
    "}": "\\textbraceright{}",
    "&": "\\&",
    "%": "\\%",
    $: "\\$",
    "#": "\\#",
    _: "\\_",
    "^": "\\textasciicircum{}",
    "~": "\\textasciitilde{}",
  };
  return text(value, "BibTeX value").replace(
    /[\\{}&%$#_^~]/g,
    (character) => escapes[character],
  );
}

function slug(value) {
  return value
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function normaliseChapter(chapterNumber, part) {
  if (
    chapterNumber === undefined ||
    chapterNumber === null ||
    chapterNumber === ""
  )
    return undefined;
  const value = String(chapterNumber).trim();
  if (part === "appendix") {
    const letter = value.replace(/^appendix\s+/i, "").toUpperCase();
    if (!/^[A-Z]+$/.test(letter))
      throw new Error("An appendix number must be a letter such as A");
    return `Appendix ${letter}`;
  }
  const numeric = value.replace(/^chapter\s+/i, "");
  if (!/^(?:\d+|[A-Za-z]+)$/.test(numeric))
    throw new Error("A chapter label must be numeric or alphabetic");
  return numeric.replace(/^0+(?=\d)/, "").toUpperCase();
}

function pageUrl(book, pathname) {
  if (
    typeof pathname !== "string" ||
    !pathname.startsWith("/") ||
    pathname.startsWith("//") ||
    /[\\?#]/.test(pathname)
  ) {
    throw new Error(
      "Citation pathname must be an absolute local route without a query or fragment",
    );
  }
  for (const segment of pathname.split("/")) {
    const decoded = decodeURIComponent(segment);
    if ([".", ".."].includes(decoded) || /[/\\]/.test(decoded)) {
      throw new Error("Citation pathname must not contain traversal segments");
    }
  }
  if (book.url) {
    const base = new URL(publicUrl(book.url, "Citation URL"));
    if (base.search || base.hash)
      throw new Error(
        "The canonical book URL must not contain a query or fragment",
      );
    if (!base.pathname.endsWith("/")) base.pathname += "/";
    return new URL(pathname.slice(1), base).href;
  }
  return book.doi ? `https://doi.org/${book.doi}` : book.repositoryCode;
}

function apaAuthors(authors) {
  const names = authors.map(({ family, given }) => {
    const initials = given
      .split(/\s+/u)
      .filter(Boolean)
      .map((name) =>
        name
          .split("-")
          .filter(Boolean)
          .map((part) => `${Array.from(part)[0].toLocaleUpperCase("en")}.`)
          .join("-"),
      )
      .join(" ");
    return initials ? `${family}, ${initials}` : family;
  });
  if (names.length === 1) return names[0];
  return `${names.slice(0, -1).join(", ")}, & ${names.at(-1)}`;
}

/**
 * Generate a page's static downloadable citation from canonical book metadata.
 * @param {BookCitation} book
 * @param {PageCitationOptions} options
 * @returns {PageCitation}
 */
export function createPageCitation(book, options) {
  const { title, pathname, chapterNumber, part } = options;
  if (!["chapter", "appendix", "page", "book"].includes(part)) {
    throw new Error("Citation part must be chapter, appendix, page or book");
  }
  const pageTitle = part === "book" ? book.title : text(title, "Page title");
  const chapter = ["chapter", "appendix"].includes(part)
    ? normaliseChapter(chapterNumber, part)
    : undefined;
  const url = pageUrl(book, pathname);
  const pageId = chapter
    ? `${part}-${slug(chapter.replace(/^Appendix /, ""))}`
    : part === "book"
      ? "book"
      : `${part}-${slug(pathname) || "index"}`;
  const baseId = slug(book.title) || "book";
  const suffix = `${pageId}-v${slug(book.version)}`;
  const key = `${slug(book.authors[0].family) || "author"}-${book.date.slice(0, 4)}-${baseId}-${suffix}`;
  const fields = [];
  const add = (name, value) => {
    if (value !== undefined) fields.push(`  ${name} = {${bibtexText(value)}}`);
  };
  add("title", pageTitle);
  if (["chapter", "appendix"].includes(part)) add("booktitle", book.title);
  const authors = book.authors
    .map(
      ({ family, given }) =>
        `{${bibtexText(family)}}${given ? `, {${bibtexText(given)}}` : ""}`,
    )
    .join(" and ");
  fields.push(`  author = {${authors}}`);
  add("chapter", chapter);
  add("year", book.date.slice(0, 4));
  add("date", book.date);
  add("version", book.version);
  if (book.doi) {
    add("doi", book.doi);
    if (book.publisher) add("publisher", book.publisher);
    if (part === "page") add("howpublished", `Page in ${book.title}`);
    if (book.sourceCommit) add("note", `Source commit ${book.sourceCommit}`);
  } else {
    const location =
      part === "book"
        ? ""
        : `${chapter ? `${part === "appendix" ? chapter : `Chapter ${chapter}`}: ` : ""}${book.title}. `;
    add(
      "note",
      `${location}Unpublished draft, version ${book.version}, dated ${book.date}.${book.sourceCommit ? ` Source commit ${book.sourceCommit}.` : ""}`,
    );
  }
  add("url", url);
  const type = !book.doi
    ? "unpublished"
    : part === "book"
      ? "book"
      : ["chapter", "appendix"].includes(part)
        ? "incollection"
        : "misc";
  return {
    bibtex: `@${type}{${key},\n${fields.join(",\n")}\n}\n`,
    key,
    filename: `${baseId}-${suffix}.bib`,
    label: `Cite this ${part}`,
    formatted: {
      authors: apaAuthors(book.authors),
      year: book.date.slice(0, 4),
      title: pageTitle,
      ...(part !== "book" ? { container: book.title } : {}),
      edition: `${chapter ? `${part === "appendix" ? chapter : `Chapter ${chapter}`}, ` : ""}Version ${book.version}`,
      url: book.doi ? `https://doi.org/${book.doi}` : url,
      draft: !book.doi,
    },
    statusText: `${book.doi ? "Version" : "Unpublished draft · version"} ${book.version} · ${book.date}${book.doi ? ` · DOI ${book.doi}` : ""}`,
    ...(book.doi ? { doi: book.doi } : {}),
  };
}
