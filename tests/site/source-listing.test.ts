import { readFile } from "node:fs/promises";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createSourceClickGesture } from "../../src/lib/source-listing";

describe("canonical listing presentation", () => {
  it("renders real variants and canonical line numbers without including numbers in copied text", async () => {
    const source = await readFile("src/components/source/Source.astro", "utf8");
    expect(source).toContain("parseSourceVariants");
    expect(source).toContain("getSourceRegion(entry.file, entry.region)");
    expect(source).toContain("entry.start_line + line - 1");
    expect(source).toContain('aria-label="Copy code"');
    expect(source).toContain('role="region"');
    expect(source).not.toContain('role="button"');
    expect(source).toContain(
      'class="source-listing__toggle source-listing__label"',
    );
    expect(source).toContain('class="source-listing__copy-status sr-only"');
    const interaction = await readFile("src/lib/source-listing.ts", "utf8");
    expect(interaction).toContain(
      'selected.querySelector("pre code")?.textContent',
    );
  });

  it("animates measured dimensions and respects reduced motion", async () => {
    const source = await readFile("src/lib/source-listing.ts", "utf8");
    expect(source).toContain("listing.getBoundingClientRect().width");
    expect(source).toContain("codeSurface.getBoundingClientRect().height");
    expect(source).toContain("listing.animate(");
    expect(source).toContain("codeSurface.animate(");
    expect(source).toContain("if (motion.matches) return");
    expect(source).toContain("animation.cancel()");
    expect(source).toContain("ResizeObserver");
  });
});

describe("source click disambiguation", () => {
  afterEach(() => vi.useRealTimers());

  it.each([120, 320, 500])(
    "preserves vertical state for a native double-click %i ms apart",
    (interval) => {
      vi.useFakeTimers();
      for (const initial of [false, true]) {
        let vertical = initial;
        let horizontal = false;
        const gesture = createSourceClickGesture({
          readVertical: () => vertical,
          toggleVertical: () => {
            vertical = !vertical;
          },
          toggleHorizontal: (restore) => {
            horizontal = !horizontal;
            vertical = restore ?? vertical;
          },
        });
        gesture.click(1);
        vi.advanceTimersByTime(interval);
        gesture.click(2);
        gesture.doubleClick();
        vi.advanceTimersByTime(1000);
        expect(horizontal).toBe(true);
        expect(vertical).toBe(initial);
      }
    },
  );

  it("keeps single clicks working and cancels them when keyboard navigation takes over", () => {
    vi.useFakeTimers();
    let vertical = false;
    const gesture = createSourceClickGesture({
      readVertical: () => vertical,
      toggleVertical: () => {
        vertical = !vertical;
      },
      toggleHorizontal: () => {},
    });
    gesture.click(1);
    vi.advanceTimersByTime(281);
    expect(vertical).toBe(true);
    gesture.click(1);
    gesture.cancel();
    vi.advanceTimersByTime(1000);
    expect(vertical).toBe(true);
  });
});
