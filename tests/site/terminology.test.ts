import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";
import { loadTerminology, manuscriptLinks } from "../../src/lib/terminology";
import {
  renderTerminologyText,
  terminologySearchText,
} from "../../src/lib/terminology-text";
import {
  createTermMatcher,
  notationMatches,
  typedDefinition,
} from "../../src/lib/definition-notes";

const root = fileURLToPath(new URL("../../", import.meta.url));
const registry = loadTerminology(root);

describe("canonical reader reference", () => {
  it("retains every record and its exact definition, units and qualifications", () => {
    expect([
      registry.symbols.length,
      registry.terms.length,
      registry.frames.length,
    ]).toEqual([250, 178, 6]);
    expect(new Set(registry.definitions.map((entry) => entry.id)).size).toBe(
      434,
    );
    const symbols = parse(
      readFileSync(
        new URL("../../terminology/symbols.yml", import.meta.url),
        "utf8",
      ),
    );
    for (const symbol of symbols) {
      const entry = registry.definitions.find(
        (record) => record.id === `symbol-${symbol.id}`,
      )!;
      expect(entry.definition).toBe(symbol.meaning);
      expect(entry.notes).toBe(symbol.notes);
      expect(entry.units).toBe(symbol.units);
      expect(entry.html).toContain('class="katex-mathml"');
      expect(
        entry.sources.some((source) => source.href === symbol.introduced_in),
      ).toBe(true);
    }
    for (const term of registry.terms) {
      const entry = registry.definitions.find(
        (record) => record.id === `term-${term.id}`,
      )!;
      expect(entry.definition).toBe(term.definition);
      expect(entry.notes).toBe(term.notes);
      expect(entry.aliases).toEqual(term.aliases);
    }
    for (const frame of registry.frames) {
      const entry = registry.definitions.find(
        (record) => record.id === `frame-${frame.id}`,
      )!;
      expect(entry.definition).toBe(frame.definition);
      expect(entry.notes).toBe(frame.notes);
    }
  });

  it("keeps distinct meanings of the same TeX as distinct records", () => {
    const reused = registry.symbols.filter((entry) => entry.latex === "d_p");
    expect(reused.map((entry) => entry.id)).toEqual([
      "processed-image-value",
      "source-pixel-distance",
    ]);
    expect(new Set(reused.map((entry) => entry.units)).size).toBe(2);
    expect(
      registry.definitions.filter((entry) =>
        reused.some((symbol) => entry.id === `symbol-${symbol.id}`),
      ),
    ).toHaveLength(2);
  });

  it("extracts only manuscript links and preserves actual bibliography anchors", () => {
    expect(
      manuscriptLinks(
        "Manuscript: /appendix/x-ray-theory/#a4-how-photons-interact-with-matter. Qualification.",
      ),
    ).toEqual(["/appendix/x-ray-theory/#a4-how-photons-interact-with-matter"]);
    expect(
      registry.definitions.find((entry) => entry.id === "frame-detector")
        ?.sources,
    ).toContainEqual({ href: "/references/#dicomPS33", label: "dicomPS33" });
  });

  it("uses short chapter labels and preserves the section's capitalisation", () => {
    expect(
      registry.definitions.find((entry) => entry.id === "symbol-optical-depth")
        ?.sources[0]?.label,
    ).toBe("Introduction: Before an image becomes a tensor");
    expect(
      registry.sections.find((section) => section.number === "A.11")?.label,
    ).toBe("X-ray physics: CT values and Hounsfield units");
  });

  it("links numbered manuscript references to their recorded sections", () => {
    const html = registry.definitions.find(
      (entry) => entry.id === "symbol-primary-transmission",
    )?.notesHtml;
    expect(html).toContain(
      '<a href="/chapters/introduction/#11-before-an-image-becomes-a-tensor">Manuscript §1.1</a>',
    );
    expect(renderTerminologyText("Manuscript §99.9.", registry.sections)).toBe(
      "Manuscript §99.9.",
    );
  });
});

describe("typeset reference prose", () => {
  it("preserves plain-text searches for typeset units and subscripts", () => {
    const searchable = terminologySearchText("$\\mathrm{mm}^{-1}$ and $T_{p}$");
    expect(searchable).toContain("mm^-1 and T_p");
    expect(searchable).toContain("$\\mathrm{mm}^{-1}$");
    for (const [id, query] of [
      ["symbol-relative-source-pose", "inverse"],
      ["symbol-detector-inner-product-matrix", "transpose"],
      ["symbol-measured-log-projection", "L-hat-p"],
    ]) {
      const entry = registry.definitions.find((entry) => entry.id === id)!;
      expect(
        terminologySearchText(`${entry.definition} ${entry.notes}`),
      ).toContain(query);
    }
  });
  it("renders explicit expressions as accessible inline mathematics", () => {
    const html = renderTerminologyText(
      "Equal to $\\exp(-L_p(E))$, with $T_p$ or $T$.",
    );
    expect(html.match(/class="katex-mathml"/gu)).toHaveLength(3);
    expect(html).toContain("<msub>");
    expect(html).toContain("Equal to ");
    expect(html).toContain(", with ");
  });

  it("escapes prose and rejects invalid maths instead of accepting authored HTML", () => {
    const html = renderTerminologyText(
      '<img src=x onerror="alert(1)"> and $x < y$.',
    );
    expect(html).toContain("&lt;img");
    expect(html).not.toContain("<img");
    expect(html).toContain('class="katex-mathml"');
    expect(() => renderTerminologyText("$\\unknowncommand{x}$")).toThrow();
  });

  it("keeps ordinary prose and arbitrary URLs out of the maths and link renderer", () => {
    expect(
      renderTerminologyText("CT values in mm; see https://example.com."),
    ).toBe("CT values in mm; see https://example.com.");
    expect(
      renderTerminologyText("$\\href{https://example.com}{x}$"),
    ).not.toContain('<a href="https://example.com');
  });
});

describe("defined term matching", () => {
  const match = createTermMatcher(registry.definitions);
  it("matches recorded names and aliases with Unicode word boundaries", () => {
    const text =
      "Computed tomography (CT) and a digitally reconstructed radiograph (DRR).";
    expect(
      match(text).map((item) => [text.slice(item.start, item.end), item.id]),
    ).toEqual([
      ["Computed tomography", "term-computed-tomography"],
      ["CT", "term-computed-tomography"],
      [
        "digitally reconstructed radiograph",
        "term-digitally-reconstructed-radiograph",
      ],
      ["DRR", "term-digitally-reconstructed-radiograph"],
    ]);
    expect(match("CTscan æCT CT_2 APposterior")).toEqual([]);
  });
  it("does not turn case variants of abbreviations into meanings", () => {
    expect(match("ct drr hu ap pa jvp vjp")).toEqual([]);
    expect(match("CT DRR HU AP PA JVP VJP")).toHaveLength(7);
  });
  it("prefers whole recorded phrases over their shorter contained terms", () => {
    const text = "Mass energy-absorption coefficient";
    const matches = match(text);
    expect(matches).toHaveLength(1);
    expect(matches[0]?.id).toBe("term-mass-energy-absorption-coefficient");
    expect(text.slice(matches[0]!.start, matches[0]!.end)).toBe(text);
  });
});

describe("definition search and animation", () => {
  it("combines search words, entry category and domain", () => {
    expect(
      notationMatches(
        "linear attenuation mm^-1",
        "attenuation mm",
        "symbol",
        "symbol",
        "transmission",
        "all",
      ),
    ).toBe(true);
    expect(
      notationMatches(
        "linear attenuation mm^-1",
        "attenuation",
        "term",
        "symbol",
        "transmission",
        "all",
      ),
    ).toBe(false);
    expect(
      notationMatches(
        "linear attenuation mm^-1",
        "attenuation",
        "symbol",
        "all",
        "transmission",
        "geometry",
      ),
    ).toBe(false);
  });
  it("finishes all qualifications within 1.6 seconds and respects reduced motion", () => {
    const text = "Full qualified meaning. ".repeat(200);
    expect(typedDefinition(text, 0)).toBe("");
    expect(typedDefinition(text, 800).length).toBeGreaterThan(0);
    expect(typedDefinition(text, 1600)).toBe(text);
    expect(typedDefinition(text, 0, true)).toBe(text);
    expect(typedDefinition("α🙂", 18)).toBe("α");
    expect(typedDefinition("", 100)).toBe("");
  });
});
