/** Return the shared chapter/appendix numbering prefix, or null for other pages. */
export function chapterNumber(file) {
  const frontmatter = file.data?.astro?.frontmatter;
  if (
    typeof frontmatter?.layout !== "string" ||
    !/(^|\/)ChapterLayout\.astro$/.test(frontmatter.layout)
  )
    return null;
  const chapter = String(frontmatter.chapterNumber ?? "").replace(
    /^0+(?=\d)/,
    "",
  );
  return /^(?:\d+|[A-Z]+)$/.test(chapter) ? chapter : null;
}
