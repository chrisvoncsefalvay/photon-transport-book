import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { loadCitationRenderer } from "./citations.mjs";

async function citationFixture(format, library) {
  const root = await mkdtemp(path.join(os.tmpdir(), "dpt-citation-"));
  await mkdir(path.join(root, "references/styles"), { recursive: true });
  await writeFile(
    path.join(root, "references/library.json"),
    JSON.stringify(
      library ?? [
        {
          id: "real-source",
          type: "article-journal",
          title: "A real title",
          author: [{ family: "Siddon", given: "Robert L." }],
          issued: { "date-parts": [[1985]] },
          DOI: "10.1118/1.595715",
        },
      ],
    ),
  );
  await writeFile(
    path.join(root, "references/overrides.yml"),
    '{"schema_version":1,"overrides":{}}',
  );
  await writeFile(
    path.join(root, "references/styles/default.csl"),
    `<style xmlns="http://purl.org/net/xbiblio/csl" version="1.0.2"><info><title>${format}</title><category citation-format="${format}"/></info></style>`,
  );
  return root;
}

test("renders a genuine fixture through the encapsulated citation API", async () => {
  const renderer = await loadCitationRenderer({
    root: await citationFixture("numeric"),
  });
  assert.equal(renderer.renderCitation(["real-source"]), "[1]");
  assert.match(
    renderer.renderBibliography(["real-source"])[0],
    /10\.1118\/1\.595715/,
  );
  assert.throws(
    () => renderer.renderCitation(["missing"]),
    /Unknown citation key/,
  );
});

test("swapping CSL changes citation form without changing citation keys", async () => {
  const numeric = await loadCitationRenderer({
    root: await citationFixture("numeric"),
  });
  const authorDate = await loadCitationRenderer({
    root: await citationFixture("author-date"),
  });
  assert.equal(numeric.renderCitation(["real-source"]), "[1]");
  assert.equal(authorDate.renderCitation(["real-source"]), "(Siddon, 1985)");
});

test("numeric IDs agree with full and subset bibliography order", async () => {
  const renderer = await loadCitationRenderer({
    root: await citationFixture("numeric", [
      { id: "z-last-alphabetically", title: "First canonical entry" },
      { id: "a-first-alphabetically", title: "Second canonical entry" },
      { id: "middle", title: "Third canonical entry" },
    ]),
  });
  assert.deepEqual(renderer.keys, [
    "z-last-alphabetically",
    "a-first-alphabetically",
    "middle",
  ]);
  assert.deepEqual(renderer.keys.map(renderer.citationNumber), [1, 2, 3]);
  assert.deepEqual(
    renderer
      .renderBibliography()
      .map((html) => html.match(/data-citation-key="([^"]+)"/)[1]),
    renderer.keys,
  );
  const subset = ["middle", "a-first-alphabetically"];
  assert.equal(renderer.renderCitation(subset), "[3, 2]");
  assert.deepEqual(subset.map(renderer.citationNumber), [3, 2]);
  assert.deepEqual(
    renderer
      .renderBibliography(subset)
      .map((html) => html.match(/data-citation-key="([^"]+)"/)[1]),
    subset,
  );
  assert.throws(
    () => renderer.citationNumber("missing"),
    /Unknown citation key/,
  );

  // Chapter subsets must not restart their HTML list numbering at one.
  const component = await readFile(
    "src/components/citations/Bibliography.astro",
    "utf8",
  );
  assert.match(
    component,
    /value=\{[\s\S]*?renderer\.citationNumber\(selectedKeys\[index\]!\)/,
  );
});

test("renders institutional and editor-only references without empty punctuation", async () => {
  const renderer = await loadCitationRenderer({
    root: await citationFixture("numeric", [
      {
        id: "institution",
        title: "Institutional report.",
        author: [{ literal: "Research & Standards Institute" }],
        issued: { "date-parts": [[2020]] },
        publisher: "Institute Press",
      },
      {
        id: "editors",
        title: "Collected methods",
        editor: [
          { family: "Example", given: "A." },
          { family: "Sample", given: "B." },
        ],
        issued: { "date-parts": [[2021]] },
        "publisher-place": "Cambridge",
        publisher: "Methods Press",
      },
      { id: "anonymous", title: "Unsigned report", publisher: "Public Office" },
    ]),
  });
  const [institution, editors, anonymous] = renderer.renderBibliography();
  assert.match(
    institution,
    /Research &amp; Standards Institute \(2020\)\. Institutional report\. Institute Press\./,
  );
  assert.match(
    editors,
    /Example, A\. and Sample, B\. \(eds\.\) \(2021\)\. Collected methods\. Cambridge: Methods Press\./,
  );
  assert.match(
    anonymous,
    />Unsigned report \(n\.d\.\)\. Public Office\.<\/span>/,
  );
  for (const html of [institution, editors, anonymous]) {
    assert.doesNotMatch(html, /\.\s+\.|\.\./);
  }
});

test("URL-only links and every displayed metadata field are HTML escaped", async () => {
  const url = 'https://example.test/record?q="source"&mode=<review>';
  const renderer = await loadCitationRenderer({
    root: await citationFixture("numeric", [
      {
        id: 'quoted"key',
        title: 'A <title> & "subtitle"',
        author: [{ literal: "A & B" }],
        "container-title": "Journal <label>",
        volume: "<b>2</b>",
        issue: '"3"',
        page: "1&2",
        publisher: "Publisher <script>",
        URL: url,
      },
    ]),
  });
  const [html] = renderer.renderBibliography();
  assert.match(html, /data-citation-key="quoted&quot;key"/);
  assert.match(
    html,
    /href="https:\/\/example\.test\/record\?q=&quot;source&quot;&amp;mode=&lt;review&gt;"/,
  );
  assert.match(html, /A &lt;title&gt; &amp; &quot;subtitle&quot;/);
  assert.match(html, /<em>Journal &lt;label&gt;<\/em>/);
  assert.match(html, /&lt;b&gt;2&lt;\/b&gt;\(&quot;3&quot;\), 1&amp;2/);
  assert.match(html, /Publisher &lt;script&gt;/);
  assert.doesNotMatch(html, /<script>|<b>|<title>/);
});

test("DOIs take priority over URLs and unsafe URL protocols never become links", async () => {
  const renderer = await loadCitationRenderer({
    root: await citationFixture("numeric", [
      {
        id: "doi",
        title: "DOI priority",
        DOI: "10.1118/1.595715",
        URL: "https://example.test/unused",
      },
      { id: "unsafe", title: "Unsafe URL", URL: "javascript:alert(1)" },
      { id: "invalid", title: "Invalid URL", URL: "not a URL" },
    ]),
  });
  const [doi, unsafe, invalid] = renderer.renderBibliography();
  assert.match(doi, /href="https:\/\/doi.org\/10.1118\/1.595715"/);
  assert.doesNotMatch(doi, /example\.test/);
  assert.doesNotMatch(unsafe, /<a\b|javascript:/);
  assert.doesNotMatch(invalid, /<a\b/);
});

test("grouped citations give every key its own label and reference link", async () => {
  const library = [
    {
      id: "first",
      title: "First reference",
      author: [{ family: "First" }],
      issued: { "date-parts": [[2001]] },
    },
    {
      id: "second",
      title: "Second reference",
      author: [{ family: "Second" }],
      issued: { "date-parts": [[2002]] },
    },
  ];
  for (const format of ["numeric", "author-date"]) {
    const renderer = await loadCitationRenderer({
      root: await citationFixture(format, library),
    });
    const labels = ["first", "second"].map((key) =>
      format === "numeric"
        ? String(renderer.citationNumber(key))
        : renderer.renderCitation([key]).slice(1, -1),
    );
    assert.deepEqual(
      labels,
      format === "numeric" ? ["1", "2"] : ["First, 2001", "Second, 2002"],
    );
  }
  const component = await readFile(
    "src/components/citations/Citation.astro",
    "utf8",
  );
  assert.match(component, /keys\.map\(\(key\)\s*=>/);
  assert.match(component, /\/references\/#\$\{encodeURIComponent\(key\)\}/);
  assert.match(component, /entries\.map/);
  assert.match(component, /<a href=\{entry\.href\}>\{entry\.label\}<\/a>/);
  assert.doesNotMatch(component, /keys\[0\]/);
});
