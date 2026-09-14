import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { stringify } from "yaml";

import { createPageCitation, loadBookCitation } from "./page-citation.mjs";

// Synthetic test metadata only. This number is not a claimed Zenodo deposit.
const SYNTHETIC_DOI = "10.5281/zenodo.999999999999";
const SOURCE_COMMIT = "a".repeat(40);
const syntheticCff = {
  "cff-version": "1.2.0",
  message: "Synthetic citation tests; not a publication.",
  type: "software",
  title: "Synthetic Transport Book",
  authors: [{ "family-names": "von Example", "given-names": "Chris James" }],
  version: "v1.2.3",
  "date-released": "2032-02-29",
  "repository-code": "https://example.invalid/public/book",
};
const page = {
  title: "A synthetic chapter",
  pathname: "/chapters/synthetic/",
  chapterNumber: "02",
  part: "chapter",
};

async function fixture(t, changes = {}) {
  const root = await mkdtemp(path.join(os.tmpdir(), "dpt-citation-synthetic-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const cff = { ...syntheticCff, ...changes };
  await writeFile(path.join(root, "CITATION.cff"), stringify(cff));
  return { root, cff };
}

async function freeze(root, citation, changes = {}) {
  const metadata = {
    schema_version: 1,
    version: `v${citation.version}`,
    source_commit: SOURCE_COMMIT,
    generated_at: "2032-02-29T12:00:00.000Z",
    public_schema_version: 1,
    citation: { ...citation, sourceCommit: SOURCE_COMMIT },
    ...changes,
  };
  await mkdir(path.join(root, "public"), { recursive: true });
  await writeFile(
    path.join(root, "public/release-metadata.json"),
    JSON.stringify(metadata),
  );
  return metadata;
}

test("synthetic stable edition uses canonical CFF metadata without requiring a DOI", async (t) => {
  const { root } = await fixture(t);
  const book = await loadBookCitation({ root, expectedVersion: "1.2.3" });
  assert.deepEqual(book, {
    title: syntheticCff.title,
    authors: [{ family: "von Example", given: "Chris James" }],
    version: "1.2.3",
    date: "2032-02-29",
    repositoryCode: syntheticCff["repository-code"],
    released: true,
  });
  assert.deepEqual(JSON.parse(JSON.stringify(book)), book);
  const citation = createPageCitation(book, page);
  assert.match(citation.bibtex, /^@incollection\{/);
  assert.match(citation.bibtex, /chapter = \{2\}/);
  assert.match(citation.bibtex, /year = \{2032\}/);
  assert.match(citation.bibtex, /version = \{1\.2\.3\}/);
  assert.match(citation.bibtex, /date = \{2032-02-29\}/);
  assert.doesNotMatch(citation.bibtex, /unpublished|draft/i);
  assert.match(
    citation.bibtex,
    /url = \{https:\/\/example\.invalid\/public\/book\}/,
  );
  assert.doesNotMatch(
    citation.bibtex,
    /doi =|publisher =|pages =|localhost|4173/,
  );
  assert.equal(citation.doi, undefined);
  assert.equal(citation.formatted.authors, "von Example, C. J.");
  assert.equal(citation.formatted.edition, "Chapter 2, Version 1.2.3");
  assert.equal(citation.formatted.draft, false);
  assert.equal(citation.formatted.container, syntheticCff.title);
  assert.equal(citation.statusText, "Version 1.2.3 · 2032-02-29");
});

test("synthetic released book and appendix use their canonical site URLs without a DOI", async (t) => {
  const { root } = await fixture(t, {
    version: "1.0.0",
    url: "https://example.invalid/edition/",
  });
  const book = await loadBookCitation({ root });
  for (const [options, type, url] of [
    [
      { title: "Homepage", pathname: "/", part: "book" },
      "book",
      "https://example.invalid/edition/",
    ],
    [
      {
        ...page,
        pathname: "/appendix/theory/",
        part: "appendix",
        chapterNumber: "A",
      },
      "incollection",
      "https://example.invalid/edition/appendix/theory/",
    ],
    [
      { title: "About", pathname: "/about/", part: "page" },
      "misc",
      "https://example.invalid/edition/about/",
    ],
  ]) {
    const citation = createPageCitation(book, options);
    assert.ok(citation.bibtex.startsWith(`@${type}{`));
    assert.equal(citation.formatted.url, url);
    assert.equal(citation.formatted.draft, false);
    assert.equal(citation.statusText, "Version 1.0.0 · 2032-02-29");
    assert.doesNotMatch(
      citation.bibtex,
      /doi =|publisher =|unpublished|draft/i,
    );
    if (options.part === "book")
      assert.equal(citation.formatted.title, syntheticCff.title);
    if (options.part === "appendix")
      assert.match(citation.bibtex, /chapter = \{Appendix A\}/);
  }
});

test("synthetic prerelease versions retain draft citations even with a reserved DOI", async (t) => {
  for (const editionVersion of [
    "0.0.0-bootstrap",
    "1.2.3-rc.1",
    "1.2.3-beta.1+preview-build",
  ]) {
    for (const doi of [undefined, SYNTHETIC_DOI]) {
      const { root } = await fixture(t, { version: editionVersion, doi });
      const book = await loadBookCitation({ root });
      const citation = createPageCitation(book, page);
      assert.equal(book.released, false);
      assert.match(citation.bibtex, /^@unpublished\{/);
      assert.ok(
        citation.bibtex.includes(
          `Unpublished draft, version ${editionVersion}`,
        ),
      );
      assert.equal(citation.formatted.draft, true);
      assert.ok(
        citation.statusText.startsWith(
          `Unpublished draft · version ${editionVersion}`,
        ),
      );
      assert.equal(citation.doi, doi);
      if (doi) assert.ok(citation.bibtex.includes(`doi = {${doi}}`));
    }
  }
});

test("synthetic build metadata alone does not turn a released edition into a draft", async (t) => {
  const { root } = await fixture(t, { version: "1.0.0+website-build" });
  const book = await loadBookCitation({ root });
  assert.equal(book.released, true);
  const citation = createPageCitation(book, page);
  assert.match(citation.bibtex, /^@incollection\{/);
  assert.equal(citation.formatted.draft, false);
  assert.equal(citation.statusText, "Version 1.0.0+website-build · 2032-02-29");
});

test("synthetic edition chapter retains book DOI and explicit canonical page URL", async (t) => {
  const { root } = await fixture(t, {
    doi: SYNTHETIC_DOI,
    url: "https://example.invalid/edition/v1.2.3",
  });
  const book = await loadBookCitation({ root, requireDoi: true });
  const citation = createPageCitation(book, page);
  assert.match(citation.bibtex, /^@incollection\{/);
  assert.match(citation.bibtex, /booktitle = \{Synthetic Transport Book\}/);
  assert.ok(citation.bibtex.includes(`doi = {${SYNTHETIC_DOI}}`));
  assert.match(
    citation.bibtex,
    /url = \{https:\/\/example\.invalid\/edition\/v1\.2\.3\/chapters\/synthetic\/\}/,
  );
  assert.equal(citation.formatted.url, `https://doi.org/${SYNTHETIC_DOI}`);
  assert.equal(citation.formatted.draft, false);
  assert.doesNotMatch(citation.statusText, /published|released/i);
});

test("synthetic edition without a canonical site links to DOI and permits Appendix A and chapter D", async (t) => {
  const { root } = await fixture(t, { doi: SYNTHETIC_DOI });
  const book = await loadBookCitation({ root });
  const appendix = createPageCitation(book, {
    ...page,
    part: "appendix",
    chapterNumber: "A",
  });
  assert.match(appendix.bibtex, /chapter = \{Appendix A\}/);
  assert.ok(
    appendix.bibtex.includes(`url = {https://doi.org/${SYNTHETIC_DOI}}`),
  );
  assert.equal(appendix.formatted.edition, "Appendix A, Version 1.2.3");
  assert.notEqual(appendix.key, createPageCitation(book, page).key);
  assert.match(
    createPageCitation(book, { ...page, chapterNumber: "D" }).bibtex,
    /chapter = \{D\}/,
  );
});

test("synthetic readable citation formats multiple authors and hyphenated initials", async (t) => {
  const { root } = await fixture(t, {
    authors: [
      { "family-names": "von Example", "given-names": "Chris" },
      { "family-names": "Test", "given-names": "Élodie Jean-Pierre" },
      { "family-names": "Fixture" },
    ],
  });
  const citation = createPageCitation(await loadBookCitation({ root }), page);
  assert.equal(
    citation.formatted.authors,
    "von Example, C., Test, É. J.-P., & Fixture",
  );
  assert.equal(citation.formatted.year, "2032");
  assert.equal(citation.formatted.title, page.title);
});

test("synthetic home citation uses canonical book title and ancillary pages keep page title", async (t) => {
  const { root } = await fixture(t, { doi: SYNTHETIC_DOI });
  const book = await loadBookCitation({ root });
  const wholeBook = createPageCitation(book, {
    title: "Homepage",
    pathname: "/",
    part: "book",
  });
  assert.match(wholeBook.bibtex, /^@book\{/);
  assert.equal(wholeBook.formatted.title, syntheticCff.title);
  assert.equal(wholeBook.formatted.container, undefined);
  const ancillary = createPageCitation(book, {
    title: "About this synthetic book",
    pathname: "/about/",
    part: "page",
  });
  assert.match(ancillary.bibtex, /^@misc\{/);
  assert.match(
    ancillary.bibtex,
    /howpublished = \{Page in Synthetic Transport Book\}/,
  );
  assert.doesNotMatch(ancillary.bibtex, /chapter =/);
  assert.notEqual(ancillary.key, wholeBook.key);
});

test("synthetic malformed or missing CFF metadata fails instead of inventing values", async (t) => {
  for (const changes of [
    { version: undefined },
    { version: "latest" },
    { "date-released": undefined },
    { "date-released": "2031-02-29" },
    { authors: [] },
    { title: "" },
    { doi: "10.5281/zenodo.TODO" },
    { doi: "10.5281/zenodo.00000" },
    { doi: `https://doi.org/${SYNTHETIC_DOI}` },
    { "repository-code": "javascript:alert(1)" },
  ]) {
    const { root } = await fixture(t, changes);
    await assert.rejects(loadBookCitation({ root }), /CITATION\.cff/);
  }
  const { root } = await fixture(t);
  await assert.rejects(
    loadBookCitation({ root, requireDoi: true }),
    /version-specific Zenodo DOI/,
  );
  await assert.rejects(
    loadBookCitation({ root, expectedVersion: "v2.0.0" }),
    /does not match/,
  );
});

test("synthetic explicitly identified concept DOI cannot stand in for a version DOI", async (t) => {
  const { root } = await fixture(t, {
    doi: SYNTHETIC_DOI,
    identifiers: [
      {
        type: "doi",
        value: SYNTHETIC_DOI,
        description: "All versions (concept DOI)",
      },
    ],
  });
  await assert.rejects(
    loadBookCitation({ root, requireDoi: true }),
    /all versions/,
  );
});

test("synthetic frozen citation is independent of caller mutations and the build clock", async (t) => {
  const { root } = await fixture(t, { doi: SYNTHETIC_DOI });
  const original = await loadBookCitation({ root });
  const metadata = await freeze(root, original);
  const loaded = await loadBookCitation({ root });
  assert.deepEqual(loaded, metadata.citation);
  const before = await readFile(
    path.join(root, "public/release-metadata.json"),
    "utf8",
  );
  loaded.title = "A caller changed its own object";
  loaded.authors[0].family = "Different";
  const reloaded = await loadBookCitation({ root });
  assert.deepEqual(reloaded, metadata.citation);
  assert.equal(
    await readFile(path.join(root, "public/release-metadata.json"), "utf8"),
    before,
  );
  assert.equal(reloaded.date, "2032-02-29");
  assert.match(
    createPageCitation(reloaded, page).bibtex,
    /Source commit a{40}/,
  );
});

test("synthetic prepared citation rejects stale DOI, version, date, authors and URLs", async (t) => {
  const changes = [
    { doi: "10.5281/zenodo.999999999998" },
    { version: "2.0.0" },
    { "date-released": "2032-03-01" },
    { title: "Changed title" },
    { authors: [{ "family-names": "Changed", "given-names": "Author" }] },
    { url: "https://example.invalid/new-edition/" },
    { "repository-code": "https://example.invalid/changed-repository" },
  ];
  for (const change of changes) {
    const { root, cff } = await fixture(t, { doi: SYNTHETIC_DOI });
    await freeze(root, await loadBookCitation({ root }));
    await writeFile(
      path.join(root, "CITATION.cff"),
      stringify({ ...cff, ...change }),
    );
    await assert.rejects(
      loadBookCitation({ root }),
      /does not match CITATION\.cff/,
    );
  }
});

test("synthetic missing snapshot and divergent source commits fail prepared metadata checks", async (t) => {
  const { root } = await fixture(t, { doi: SYNTHETIC_DOI });
  const original = await loadBookCitation({ root });
  await freeze(root, original, { citation: undefined });
  await assert.rejects(
    loadBookCitation({ root }),
    /does not match CITATION\.cff/,
  );
  await freeze(root, original, { source_commit: "b".repeat(40) });
  await assert.rejects(
    loadBookCitation({ root }),
    /does not match CITATION\.cff/,
  );
});

test("synthetic local routes cannot replace the canonical host or escape its path", async (t) => {
  const { root } = await fixture(t, { url: "https://example.invalid/book/" });
  const book = await loadBookCitation({ root });
  for (const pathname of [
    "https://preview.invalid/",
    "//preview.invalid/",
    "/../other/",
    "/%2e%2e/other/",
    "/\\preview.invalid/",
    "/chapters/?preview=1",
  ]) {
    assert.throws(
      () => createPageCitation(book, { ...page, pathname }),
      /pathname/,
    );
  }
});

test("synthetic BibTeX input escapes delimiters and TeX special characters without executing them", async (t) => {
  const { root } = await fixture(t);
  const book = await loadBookCitation({ root });
  const title = "Synthetic } @book{injected, \\input{private} & 50% $ # _ ^ ~";
  const citation = createPageCitation(book, { ...page, title });
  assert.equal(citation.formatted.title, title);
  assert.match(citation.bibtex, /\\textbraceright\{\}/);
  assert.match(
    citation.bibtex,
    /\\textbackslash\{\}input\\textbraceleft\{\}private\\textbraceright\{\}/,
  );
  for (const escaped of [
    "\\&",
    "\\%",
    "\\$",
    "\\#",
    "\\_",
    "\\textasciicircum{}",
    "\\textasciitilde{}",
  ]) {
    assert.ok(citation.bibtex.includes(escaped), escaped);
  }
  assert.doesNotMatch(citation.bibtex, /\\input\{/);
  assert.match(citation.filename, /^[a-z0-9-]+\.bib$/);
});

const bibtexAvailable =
  spawnSync("bibtex", ["--version"], { stdio: "ignore" }).status === 0;
test(
  "synthetic escaped citation is accepted by a real BibTeX parser when available",
  { skip: !bibtexAvailable },
  async (t) => {
    const { root } = await fixture(t);
    const citation = createPageCitation(await loadBookCitation({ root }), {
      ...page,
      title: "Synthetic } { \\path & # % _ ~ ^ $",
    });
    await writeFile(path.join(root, "citation.bib"), citation.bibtex);
    await writeFile(
      path.join(root, "citation.aux"),
      `\\relax\n\\citation{${citation.key}}\n\\bibstyle{plain}\n\\bibdata{citation}\n`,
    );
    execFileSync("bibtex", ["citation"], { cwd: root, stdio: "pipe" });
    const log = await readFile(path.join(root, "citation.blg"), "utf8");
    assert.doesNotMatch(log, /I was expecting|Illegal|error message/i);
    assert.ok(
      (await readFile(path.join(root, "citation.bbl"), "utf8")).includes(
        "Synthetic",
      ),
    );
  },
);
