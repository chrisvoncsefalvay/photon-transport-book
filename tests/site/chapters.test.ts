import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";

import {
  chapterContinuation,
  chapterNeighbours,
  chapters,
} from "../../src/lib/chapters";
import { typedContinuation } from "../../src/lib/chapter-preview";

describe("chapter catalogue", () => {
  it("keeps full titles in sync with their pages and distinct labels for navigation", () => {
    for (const chapter of chapters) {
      const path = new URL(
        `../../src/pages${chapter.href.slice(0, -1)}.mdx`,
        import.meta.url,
      );
      const source = readFileSync(path, "utf8");
      const frontmatter = parse(source.split("---", 3)[1]!);
      expect(frontmatter.title, chapter.href).toBe(chapter.title);
      expect(String(frontmatter.chapterNumber), chapter.href).toBe(
        chapter.number,
      );
      expect(frontmatter.part ?? "chapter", chapter.href).toBe(chapter.part);
      expect(chapter.shortTitle.trim(), chapter.href).not.toBe("");
      expect(chapter.shortTitle, chapter.href).not.toBe(chapter.title);
    }
    expect(new Set(chapters.map((chapter) => chapter.shortTitle)).size).toBe(
      chapters.length,
    );
  });

  it("contains the foundations, three application chapters and appendix in order", () => {
    expect(chapters).toHaveLength(14);
    expect(chapters.map((chapter) => chapter.number)).toEqual([
      "01",
      "02",
      "03",
      "04",
      "05",
      "06",
      "07",
      "08",
      "09",
      "10",
      "11",
      "12",
      "13",
      "A",
    ]);
    expect(new Set(chapters.map((chapter) => chapter.href)).size).toBe(
      chapters.length,
    );
  });

  it("stores a revision in every reader chapter and appendix", () => {
    for (const chapter of chapters) {
      const source = readFileSync(
        new URL(
          `../../src/pages${chapter.href.slice(0, -1)}.mdx`,
          import.meta.url,
        ),
        "utf8",
      );
      const frontmatter = parse(source.split("---", 3)[1]!);
      expect(frontmatter.revision, chapter.href).toMatch(/^\d+\.\d+\.\d+$/);
      expect(frontmatter, chapter.href).not.toHaveProperty("status");
    }
  });

  it("resolves adjacent navigation without wrapping", () => {
    expect(chapterNeighbours(chapters[0].href)).toEqual({ next: chapters[1] });
    expect(chapterNeighbours(chapters[4].href)).toEqual({
      previous: chapters[3],
      next: chapters[5],
    });
    expect(chapterNeighbours(chapters[9].href)).toEqual({
      previous: chapters[8],
      next: chapters[10],
    });
    expect(chapterNeighbours("/chapters/registration/")).toEqual({
      previous: chapters[9],
      next: chapters[11],
    });
    expect(chapterNeighbours("/chapters/reconstruction/")).toEqual({
      previous: chapters[10],
      next: chapters[12],
    });
    expect(chapterNeighbours("/chapters/acquisition-design/")).toEqual({
      previous: chapters[11],
      next: chapters[13],
    });
    expect(chapterNeighbours(chapters[13].href)).toEqual({
      previous: chapters[12],
    });
  });

  it("provides distinct short and long descriptions for every chapter preview", () => {
    for (const chapter of chapters) {
      expect(chapter.scope.trim()).not.toBe("");
      expect(chapter.longScope.trim()).toBe(chapter.longScope);
      expect(chapter.longScope.length).toBeGreaterThan(chapter.scope.length);
      expect(chapter.longScope).not.toContain("<");
      const { prefix, suffix } = chapterContinuation(
        chapter.scope,
        chapter.longScope,
      );
      expect(`${prefix}.`).toBe(chapter.scope);
      expect(suffix).toMatch(/^, \S/);
      expect(prefix + suffix).toBe(chapter.longScope);
    }
    expect(new Set(chapters.map((chapter) => chapter.longScope)).size).toBe(
      chapters.length,
    );
  });

  it("rejects a longer paragraph that does not continue the original sentence", () => {
    expect(() =>
      chapterContinuation("A short sentence.", "A different paragraph."),
    ).toThrow();
  });

  it("types the suffix gradually without exposing it before the first character interval", () => {
    const continuation = ", continuing the same sentence.";
    expect(typedContinuation(continuation, 0)).toBe("");
    expect(typedContinuation(continuation, 39)).toBe("");
    expect(typedContinuation(continuation, 40)).toBe(",");
    expect(typedContinuation(continuation, 200)).toBe(", con");
    expect(typedContinuation(continuation, continuation.length * 40)).toBe(
      continuation,
    );
    expect(typedContinuation(continuation, 10000)).toBe(continuation);
  });
});
