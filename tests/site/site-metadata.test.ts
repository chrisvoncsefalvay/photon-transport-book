import { readFileSync, readdirSync } from "node:fs";
import type { APIContext } from "astro";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";
import { loadBookCitation } from "../../tools/public/page-citation.mjs";
import { chapters } from "../../src/lib/chapters";
import {
  AUTHOR_NAME,
  BOOK_DESCRIPTION,
  BOOK_REPOSITORY,
  BOOK_TITLE,
  SITE_NAME,
  SITE_URL,
  SOCIAL_IMAGE,
  canonicalPageUrl,
  createSiteMetadata,
  readerPagePaths,
  serializeStructuredData,
} from "../../src/lib/site-metadata";
import { GET as robots } from "../../src/pages/robots.txt";
import { GET as sitemap } from "../../src/pages/sitemap.xml";

describe("site metadata", () => {
  it("keeps canonical URLs on the published origin and removes view state", () => {
    expect(canonicalPageUrl("/")).toBe("https://photontransport.com/");
    expect(
      canonicalPageUrl("/chapters/reconstruction?channel=bone#figure-12-3"),
    ).toBe("https://photontransport.com/chapters/reconstruction/");
    expect(canonicalPageUrl("/about/")).toBe(
      "https://photontransport.com/about/",
    );
    expect(canonicalPageUrl("/generated/source-files/example.py.html")).toBe(
      "https://photontransport.com/generated/source-files/example.py.html",
    );
    const config = readFileSync(
      new URL("../../astro.config.mjs", import.meta.url),
      "utf8",
    );
    expect(config).toContain(`site: "${SITE_URL}"`);
    expect(config).toContain('trailingSlash: "always"');
  });

  it.each([
    "https://preview.example/chapters/reconstruction/",
    "//preview.example/",
    "/\\preview.example/",
    "/\n/preview.example/",
    "about/",
  ])("rejects a path that could select another origin: %s", (path) => {
    expect(() => canonicalPageUrl(path)).toThrow(/local absolute paths/);
  });

  it("indexes every actual reader page exactly once", () => {
    const pagesRoot = new URL("../../src/pages/", import.meta.url);
    const actualPages = readdirSync(pagesRoot, { recursive: true })
      .map(String)
      .filter((path) => /\.(astro|mdx)$/.test(path))
      .filter((path) => path !== "chapters/design-system.mdx")
      .map((path) =>
        path === "index.astro"
          ? "/"
          : `/${path.replace(/\.(astro|mdx)$/, "")}/`,
      );
    expect([...readerPagePaths].sort()).toEqual(actualPages.sort());
    expect(new Set(readerPagePaths).size).toBe(readerPagePaths.length);
    for (const pathname of readerPagePaths) {
      expect(createSiteMetadata({ pathname }).indexable, pathname).toBe(true);
    }
  });

  it("uses the authored chapter text and connects each chapter to the book", () => {
    const descriptions = new Set<string>();
    const titles = new Set<string>();
    for (const chapter of chapters) {
      const source = readFileSync(
        new URL(
          `../../src/pages${chapter.href.slice(0, -1)}.mdx`,
          import.meta.url,
        ),
        "utf8",
      );
      const frontmatter = parse(source.split("---", 3)[1]!) as {
        title: string;
        scope: string;
        description?: string;
      };
      const metadata = createSiteMetadata({
        pathname: chapter.href,
        title: `${frontmatter.title} — ${SITE_NAME}`,
        description: frontmatter.description ?? frontmatter.scope,
      });
      const graph = metadata.structuredData!["@graph"];
      titles.add(metadata.title);
      descriptions.add(metadata.description);
      expect(metadata.openGraphType).toBe("article");
      expect(metadata.title).toContain(frontmatter.title);
      expect(metadata.description).toBe(
        (frontmatter.description ?? frontmatter.scope)
          .replace(/\s+/g, " ")
          .trim(),
      );
      expect(graph.find((node) => node["@type"] === "Chapter")).toMatchObject({
        name: frontmatter.title,
        url: metadata.canonical,
        isPartOf: { "@id": "https://photontransport.com/#book" },
        mainEntityOfPage: { "@id": `${metadata.canonical}#webpage` },
        position: chapter.part === "chapter" ? Number(chapter.number) : "A",
        inLanguage: "en-GB",
      });
      expect(
        graph.find((node) => node["@type"] === "BreadcrumbList"),
      ).toMatchObject({
        itemListElement: [
          { position: 1, item: "https://photontransport.com/" },
          {
            position: 2,
            item: metadata.canonical,
            name: expect.stringContaining(frontmatter.title),
          },
        ],
      });
    }
    expect(titles.size).toBe(chapters.length);
    expect(descriptions.size).toBe(chapters.length);
  });

  it("describes the living book without fabricated publication facts", () => {
    const home = createSiteMetadata({ pathname: "/" });
    expect(home).toMatchObject({
      title: BOOK_TITLE,
      description: BOOK_DESCRIPTION,
      canonical: SITE_URL,
      openGraphType: "book",
    });
    const graph = home.structuredData!["@graph"];
    expect(graph.find((node) => node["@type"] === "Book")).toMatchObject({
      name: BOOK_TITLE,
      url: SITE_URL,
      author: { "@id": `${SITE_URL}#author` },
      sameAs: BOOK_REPOSITORY,
      hasPart: chapters.map((chapter) => ({
        "@type": "Chapter",
        name: chapter.title,
        url: `${SITE_URL}${chapter.href.slice(1)}`,
      })),
    });
    expect(graph.find((node) => node["@type"] === "WebSite")).toMatchObject({
      name: SITE_NAME,
      url: SITE_URL,
    });
    expect(graph.find((node) => node["@type"] === "Person")).toMatchObject({
      name: AUTHOR_NAME,
      url: "https://chrisvoncsefalvay.com/",
    });
    expect(graph.some((node) => node["@type"] === "BreadcrumbList")).toBe(
      false,
    );
    for (const pathname of readerPagePaths) {
      expect(
        JSON.stringify(createSiteMetadata({ pathname }).structuredData),
      ).not.toMatch(
        /"(?:isbn|doi|datePublished|dateModified|dateCreated|review|aggregateRating)":/,
      );
    }
  });

  it("gives reference pages their own breadcrumbs without making them chapters", () => {
    const metadata = createSiteMetadata({
      pathname: "/about/",
      title: `About — ${SITE_NAME}`,
      description: "About the author and the three volumes.",
    });
    const graph = metadata.structuredData!["@graph"];
    expect(metadata.openGraphType).toBe("website");
    expect(graph.some((node) => node["@type"] === "Chapter")).toBe(false);
    expect(graph.find((node) => node["@type"] === "AboutPage")).toMatchObject({
      description: "About the author and the three volumes.",
    });
    expect(
      graph.find((node) => node["@type"] === "BreadcrumbList"),
    ).toMatchObject({
      itemListElement: [
        { position: 1, item: SITE_URL },
        { position: 2, name: "About", item: `${SITE_URL}about/` },
      ],
    });
  });

  it("derives edition metadata from the canonical CFF citation on every reader page", async () => {
    const citation = await loadBookCitation();
    for (const pathname of readerPagePaths) {
      const data = createSiteMetadata({
        pathname,
        bookCitation: citation,
      }).structuredData!;
      const book = data["@graph"].find((node) => node["@type"] === "Book")!;
      expect(book.bookEdition, pathname).toBe(citation.version);
      expect(book.version, pathname).toBe(citation.version);
      expect(book.datePublished, pathname).toBe(
        citation.released ? citation.date : undefined,
      );
      if (citation.doi) {
        expect(book.identifier, pathname).toEqual({
          "@type": "PropertyValue",
          propertyID: "DOI",
          value: citation.doi,
        });
        expect(book.sameAs, pathname).toEqual([
          BOOK_REPOSITORY,
          `https://doi.org/${citation.doi}`,
        ]);
      } else {
        expect(book.identifier, pathname).toBeUndefined();
      }
    }
  });

  it("can describe a released URL-only edition without inventing a DOI", () => {
    // Synthetic citation values exercise metadata mapping, not a publication record.
    const bookCitation = {
      version: "2.3.4",
      date: "2032-02-29",
      released: true,
    };
    const data = createSiteMetadata({
      pathname: "/",
      bookCitation,
    }).structuredData!;
    const book = data["@graph"].find((node) => node["@type"] === "Book")!;
    expect(book).toMatchObject({
      bookEdition: "2.3.4",
      version: "2.3.4",
      datePublished: "2032-02-29",
      sameAs: BOOK_REPOSITORY,
    });
    expect(book.identifier).toBeUndefined();
  });

  it("does not assign a publication date to a prerelease with a reserved DOI", () => {
    // Synthetic reserved identifier; no deposit or publication is claimed.
    const bookCitation = {
      version: "2.3.4-rc.1",
      date: "2032-02-29",
      released: false,
      doi: "10.5281/zenodo.999999999999",
    };
    const data = createSiteMetadata({
      pathname: "/",
      bookCitation,
    }).structuredData!;
    const book = data["@graph"].find((node) => node["@type"] === "Book")!;
    expect(book.bookEdition).toBe(bookCitation.version);
    expect(book.version).toBe(bookCitation.version);
    expect(book.datePublished).toBeUndefined();
    expect(book.identifier).toMatchObject({ value: bookCitation.doi });
    expect(book.sameAs).toContain(`https://doi.org/${bookCitation.doi}`);
  });

  it.each([
    "/chapters/design-system/",
    "/generated/source-files/python/example.py.html",
    "/dev/",
    "/unknown/",
  ])("keeps non-reader content out of the index: %s", (pathname) => {
    const metadata = createSiteMetadata({ pathname });
    expect(metadata.indexable).toBe(false);
    expect(metadata.robots).toBe("noindex, follow");
    expect(metadata.structuredData).toBeUndefined();
  });

  it("keeps JSON-LD text intact without allowing a closing script tag", () => {
    const description =
      'A </script><script>alert("x")</script> & <ray>\u2028\u2029';
    const data = createSiteMetadata({
      pathname: "/",
      description,
    }).structuredData!;
    data["@graph"].push({ text: description });
    const encoded = serializeStructuredData(data);
    expect(encoded).not.toMatch(/[<>&\u2028\u2029]/);
    expect(JSON.parse(encoded)).toEqual(data);
  });

  it("matches the social image metadata to the actual PNG", () => {
    const image = readFileSync(
      new URL("../../public/social/photon-transport-og.png", import.meta.url),
    );
    expect(image.subarray(0, 8)).toEqual(
      Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    );
    expect(image.readUInt32BE(16)).toBe(SOCIAL_IMAGE.width);
    expect(image.readUInt32BE(20)).toBe(SOCIAL_IMAGE.height);
    expect(SOCIAL_IMAGE.url).toBe(`${SITE_URL}social/photon-transport-og.png`);
    expect(SOCIAL_IMAGE.alt).toContain(BOOK_TITLE.split(" — ")[0]);
    expect(SOCIAL_IMAGE.alt).toContain("three rays");
  });
});

describe("crawler endpoints", () => {
  it("serves an XML sitemap of canonical reader URLs without synthetic dates", async () => {
    const response = await sitemap({} as APIContext);
    expect(response.status).toBe(200);
    expect(response.headers.get("Content-Type")).toBe(
      "application/xml; charset=utf-8",
    );
    const xml = await response.text();
    const urls = [...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(
      (match) => match[1],
    );
    expect(urls).toEqual(readerPagePaths.map(canonicalPageUrl));
    expect(xml).toContain(
      'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"',
    );
    expect(xml).not.toMatch(
      /lastmod|priority|changefreq|design-system|source-files/,
    );
  });

  it("advertises the sitemap and lets crawlers read page-level noindex", async () => {
    const response = await robots({} as APIContext);
    expect(response.status).toBe(200);
    expect(response.headers.get("Content-Type")).toBe(
      "text/plain; charset=utf-8",
    );
    const text = await response.text();
    expect(text).toContain("User-agent: *\nAllow: /");
    expect(text).toContain("Sitemap: https://photontransport.com/sitemap.xml");
    expect(text).not.toMatch(/Disallow:|Noindex:/i);
  });
});
