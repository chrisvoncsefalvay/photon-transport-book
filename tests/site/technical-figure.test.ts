import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

describe("Three.js component boundary", () => {
  it("loads Three.js inside the figure implementation and disposes GPU resources", async () => {
    const source = await readFile("src/lib/technical-figure.ts", "utf8");

    expect(source).toContain('await import("three")');
    expect(source).toContain("ResizeObserver");
    expect(source).toContain("disconnectedCallback");
    expect(source).toContain(
      'getExtension("WEBGL_lose_context")?.loseContext()',
    );
    expect(source).toContain("renderer.dispose()");
    expect(source).toContain("renderer.forceContextLoss()");
  });

  it("keeps a static fallback in the Astro component", async () => {
    const source = await readFile(
      "src/components/figures/InteractiveFigure.astro",
      "utf8",
    );

    expect(source).toContain("interactive-figure__fallback");
    expect(source).toContain("Static fallback");
    expect(source).toContain("data-figure-controls hidden");
    expect(source).toContain('data-figure-action="reset"');
    expect(source).toContain('aria-label="Reset view"');
    expect(source).toContain('title="Reset view"');
    expect(source).not.toContain(">Reset view</button");
  });

  it("provides deliberate pointer and keyboard controls without an idle render loop", async () => {
    const source = await readFile("src/lib/technical-figure.ts", "utf8");
    expect(source).toContain('"pointerdown"');
    expect(source).toContain('"pointermove"');
    expect(source).toContain('"ArrowLeft"');
    expect(source).toContain('"Home"');
    expect(source).toContain('"webglcontextlost"');
    expect(source).toContain("listeners.abort()");
    expect(source).not.toContain("requestAnimationFrame");
    expect(source).not.toContain('addEventListener("wheel"');
  });
});
