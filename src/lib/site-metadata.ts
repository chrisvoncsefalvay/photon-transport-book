import { chapters } from "./chapters";

export const SITE_URL = "https://photontransport.com/";
export const SITE_NAME = "Differentiable Photon Transport";
export const BOOK_TITLE = `${SITE_NAME} — Volume I: Foundations`;
export const BOOK_DESCRIPTION =
  "Learn differentiable photon transport in CUDA and Warp, with worked examples in X-ray registration, volume reconstruction and acquisition design.";
export const AUTHOR_NAME = "Chris von Csefalvay";
export const AUTHOR_URL = "https://chrisvoncsefalvay.com/";
export const BOOK_REPOSITORY =
  "https://github.com/chrisvoncsefalvay/photon-transport-book";
export const SITE_LANGUAGE = "en-GB";
export const SOCIAL_IMAGE = {
  url: new URL("social/photon-transport-og.png", SITE_URL).href,
  width: 1731,
  height: 909,
  type: "image/png",
  alt: "Typographic cover of Differentiable Photon Transport, Volume I: Foundations, by Chris von Csefalvay, beside three rays passing through an object to a detector; website and GitHub addresses below.",
} as const;

// The chapter catalogue owns reading order. The remaining entries are the
// reader's reference pages; development and generated source pages stay out.
export const readerPagePaths: readonly string[] = [
  "/",
  ...chapters.map((chapter) => chapter.href),
  "/notation/",
  "/references/",
  "/about/",
  "/acknowledgements/",
];

const readerPaths = new Set(readerPagePaths);
const authorId = `${SITE_URL}#author`;
const bookId = `${SITE_URL}#book`;
const websiteId = `${SITE_URL}#website`;

export interface StructuredData {
  "@context": "https://schema.org";
  "@graph": Record<string, unknown>[];
}

/** Use the published origin even when a page is built or read in a preview. */
export function canonicalPageUrl(pathname: string): string {
  if (
    !pathname.startsWith("/") ||
    pathname.startsWith("//") ||
    pathname.includes("\\")
  ) {
    throw new Error("Canonical page paths must be local absolute paths.");
  }
  const url = new URL(pathname, SITE_URL);
  if (url.origin !== new URL(SITE_URL).origin) {
    throw new Error("Canonical page paths must be local absolute paths.");
  }
  url.search = "";
  url.hash = "";
  if (!url.pathname.endsWith("/") && !/\.[^/]+$/.test(url.pathname)) {
    url.pathname += "/";
  }
  return url.href;
}

export function createSiteMetadata({
  pathname,
  title = BOOK_TITLE,
  description = BOOK_DESCRIPTION,
  bookCitation,
}: {
  pathname: string;
  title?: string;
  description?: string;
  bookCitation?: {
    version: string;
    date: string;
    released: boolean;
    doi?: string;
  };
}) {
  const canonical = canonicalPageUrl(pathname);
  const path = new URL(canonical).pathname;
  const indexable = readerPaths.has(path);
  const chapter = chapters.find((entry) => entry.href === path);
  const isHome = path === "/";
  const pageTitle = title.trim();
  const pageDescription = description.replace(/\s+/g, " ").trim();
  const pageName = chapter?.title ?? pageTitle.replace(` — ${SITE_NAME}`, "");
  const openGraphType = isHome ? "book" : chapter ? "article" : "website";
  const pageId = `${canonical}#webpage`;
  const chapterId = `${canonical}#chapter`;
  const breadcrumbId = `${canonical}#breadcrumb`;

  const graph: StructuredData["@graph"] = [
    {
      "@type": "Person",
      "@id": authorId,
      name: AUTHOR_NAME,
      url: AUTHOR_URL,
      sameAs: ["https://github.com/chrisvoncsefalvay/"],
    },
    {
      "@type": "WebSite",
      "@id": websiteId,
      url: SITE_URL,
      name: SITE_NAME,
      description: BOOK_DESCRIPTION,
      inLanguage: SITE_LANGUAGE,
      author: { "@id": authorId },
    },
    {
      "@type": "Book",
      "@id": bookId,
      name: BOOK_TITLE,
      url: SITE_URL,
      description: BOOK_DESCRIPTION,
      inLanguage: SITE_LANGUAGE,
      author: { "@id": authorId },
      image: SOCIAL_IMAGE.url,
      sameAs: bookCitation?.doi
        ? [BOOK_REPOSITORY, `https://doi.org/${bookCitation.doi}`]
        : BOOK_REPOSITORY,
      ...(bookCitation
        ? { bookEdition: bookCitation.version, version: bookCitation.version }
        : {}),
      ...(bookCitation?.released ? { datePublished: bookCitation.date } : {}),
      ...(bookCitation?.doi
        ? {
            identifier: {
              "@type": "PropertyValue",
              propertyID: "DOI",
              value: bookCitation.doi,
            },
          }
        : {}),
      isAccessibleForFree: true,
      ...(isHome
        ? {
            hasPart: chapters.map((entry) => ({
              "@type": "Chapter",
              "@id": `${canonicalPageUrl(entry.href)}#chapter`,
              name: entry.title,
              url: canonicalPageUrl(entry.href),
            })),
          }
        : {}),
    },
    {
      "@type": path === "/about/" ? "AboutPage" : "WebPage",
      "@id": pageId,
      url: canonical,
      name: pageTitle,
      description: pageDescription,
      inLanguage: SITE_LANGUAGE,
      isPartOf: { "@id": websiteId },
      about: { "@id": bookId },
      author: { "@id": authorId },
      ...(isHome || chapter
        ? { mainEntity: { "@id": isHome ? bookId : chapterId } }
        : {}),
      ...(!isHome ? { breadcrumb: { "@id": breadcrumbId } } : {}),
    },
  ];

  if (chapter) {
    graph.push({
      "@type": "Chapter",
      "@id": chapterId,
      name: chapter.title,
      url: canonical,
      description: pageDescription,
      position:
        chapter.part === "chapter" ? Number(chapter.number) : chapter.number,
      inLanguage: SITE_LANGUAGE,
      author: { "@id": authorId },
      isPartOf: { "@id": bookId },
      mainEntityOfPage: { "@id": pageId },
    });
  }

  if (!isHome) {
    graph.push({
      "@type": "BreadcrumbList",
      "@id": breadcrumbId,
      itemListElement: [
        {
          "@type": "ListItem",
          position: 1,
          name: SITE_NAME,
          item: SITE_URL,
        },
        {
          "@type": "ListItem",
          position: 2,
          name: chapter
            ? `${chapter.part === "appendix" ? "Appendix" : "Chapter"} ${chapter.part === "appendix" ? chapter.number : Number(chapter.number)}: ${pageName}`
            : pageName,
          item: canonical,
        },
      ],
    });
  }

  return {
    title: pageTitle,
    description: pageDescription,
    canonical,
    indexable,
    robots: indexable
      ? "index, follow, max-image-preview:large"
      : "noindex, follow",
    openGraphType,
    structuredData: indexable
      ? ({
          "@context": "https://schema.org",
          "@graph": graph,
        } satisfies StructuredData)
      : undefined,
  };
}

/** Keep authored text from ending the JSON-LD script element. */
export function serializeStructuredData(data: StructuredData): string {
  return JSON.stringify(data)
    .replace(/</g, "\\u003c")
    .replace(/>/g, "\\u003e")
    .replace(/&/g, "\\u0026")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
}

export function renderSitemap(): string {
  const entries = readerPagePaths.map((path) => {
    const url = canonicalPageUrl(path).replace(/&/g, "&amp;");
    return `  <url><loc>${url}</loc></url>`;
  });
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${entries.join("\n")}\n</urlset>\n`;
}

export function renderRobots(): string {
  // Crawlers must be allowed to read noindex metadata on excluded pages.
  return `User-agent: *\nAllow: /\n\nSitemap: ${SITE_URL}sitemap.xml\n`;
}
