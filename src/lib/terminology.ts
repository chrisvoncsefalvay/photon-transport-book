import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";
import { renderToString } from "katex";
import { chapters } from "./chapters";
import {
  renderTerminologyText,
  type ManuscriptSection,
} from "./terminology-text";

export interface SymbolRecord {
  id: string;
  latex: string;
  spoken: string;
  meaning: string;
  units: string | null;
  domain: string;
  introduced_in: string | null;
  notes: string | null;
}

export interface TermRecord {
  id: string;
  term: string;
  definition: string;
  aliases: string[];
  avoid: string[];
  domain: string;
  source_keys: string[];
  notes: string | null;
}

export interface FrameRecord {
  id: string;
  label: string;
  definition: string;
  handedness: string | null;
  axes: Record<string, string>;
  relative_to: string | null;
  introduced_in: string | null;
  source_keys: string[];
  notes: string | null;
}

export interface DefinitionEntry {
  id: string;
  kind: "symbol" | "term" | "frame" | "local";
  label: string;
  html?: string;
  definition: string;
  definitionHtml?: string;
  notes: string | null;
  notesHtml?: string;
  units: string | null;
  unitsHtml?: string;
  domain: string;
  sources: { href: string; label: string }[];
  aliases: string[];
}

export function manuscriptLinks(notes: string | null): string[] {
  return [
    ...new Set(notes?.match(/\/(?:chapters|appendix)\/[\w/-]+#[\w-]+/gu) ?? []),
  ];
}

export function sectionLabel(href: string, heading?: string): string {
  const fragment = href.split("#")[1] ?? "";
  const words =
    heading ?? fragment.replace(/^a?\d+-?/i, "").replaceAll("-", " ");
  const chapter =
    chapters.find((entry) => entry.href === href.split("#")[0])?.shortTitle ??
    "Manuscript";
  const capitalised = words
    ? words[0]!.toLocaleUpperCase("en-GB") + words.slice(1)
    : "";
  return capitalised ? `${chapter}: ${capitalised}` : chapter;
}

export function registryAnchor(
  kind: DefinitionEntry["kind"],
  id: string,
): string {
  return `${kind}-${id}`;
}

function sources(
  introduced: string | null,
  notes: string | null,
  keys: string[] = [],
  sections: readonly ManuscriptSection[] = [],
) {
  return [
    ...[
      ...new Set([
        ...(introduced ? [introduced] : []),
        ...manuscriptLinks(notes),
      ]),
    ].map((href) => ({
      href,
      label:
        sections.find((section) => section.href === href)?.label ??
        sectionLabel(href),
    })),
    ...keys.map((key) => ({ href: `/references/#${key}`, label: key })),
  ];
}

export function loadTerminology(
  root = (import.meta.env && import.meta.env.DPT_PROJECT_ROOT) ??
    fileURLToPath(new URL("../../", import.meta.url)),
) {
  const read = <T>(name: string): T[] =>
    parse(readFileSync(path.join(root, "terminology", `${name}.yml`), "utf8"));
  const symbols = read<SymbolRecord>("symbols");
  const terms = read<TermRecord>("terms");
  const frames = read<FrameRecord>("frames");
  const manuscriptHrefs = [
    ...new Set(
      [
        ...symbols.flatMap((entry) => [
          entry.introduced_in,
          ...manuscriptLinks(entry.notes),
        ]),
        ...terms.flatMap((entry) => manuscriptLinks(entry.notes)),
        ...frames.flatMap((entry) => [
          entry.introduced_in,
          ...manuscriptLinks(entry.notes),
        ]),
      ].filter((href): href is string => Boolean(href)),
    ),
  ];
  const sections: ManuscriptSection[] = [];
  for (const chapter of chapters) {
    const file = `${path.join(root, "src/pages", ...chapter.href.split("/").filter(Boolean))}.mdx`;
    const source = readFileSync(file, "utf8");
    for (const match of source.matchAll(
      /^##\s+((?:[A-Z]|\d+)\.\d+)\.?\s+(.+)$/gmu,
    )) {
      const prefix = `${chapter.href}#${match[1]!.replace(".", "").toLowerCase()}-`;
      const href = manuscriptHrefs.find((candidate) =>
        candidate.toLowerCase().startsWith(prefix),
      );
      if (href)
        sections.push({
          number: match[1]!,
          href,
          label: sectionLabel(href, match[2]!),
        });
    }
  }
  const presentation = (
    definition: string,
    notes: string | null,
    units: string | null = null,
  ) => ({
    definitionHtml: renderTerminologyText(definition, sections),
    ...(notes ? { notesHtml: renderTerminologyText(notes, sections) } : {}),
    ...(units ? { unitsHtml: renderTerminologyText(units, sections) } : {}),
  });
  const definitions: DefinitionEntry[] = [
    ...symbols.map((entry): DefinitionEntry => ({
      id: registryAnchor("symbol", entry.id),
      kind: "symbol",
      label: entry.spoken,
      html: renderToString(entry.latex, {
        throwOnError: true,
        strict: "error",
        output: "htmlAndMathml",
      }),
      definition: entry.meaning,
      ...presentation(entry.meaning, entry.notes, entry.units),
      notes: entry.notes,
      units: entry.units,
      domain: entry.domain,
      sources: sources(entry.introduced_in, entry.notes, [], sections),
      aliases: [],
    })),
    ...terms.map((entry): DefinitionEntry => ({
      id: registryAnchor("term", entry.id),
      kind: "term",
      label: entry.term,
      definition: entry.definition,
      ...presentation(entry.definition, entry.notes),
      notes: entry.notes,
      units: null,
      domain: entry.domain,
      sources: sources(null, entry.notes, entry.source_keys, sections),
      aliases: entry.aliases,
    })),
    ...frames.map((entry): DefinitionEntry => ({
      id: registryAnchor("frame", entry.id),
      kind: "frame",
      label: entry.label,
      definition: entry.definition,
      ...presentation(entry.definition, entry.notes),
      notes: entry.notes,
      units: null,
      domain: "coordinate frames",
      sources: sources(
        entry.introduced_in,
        entry.notes,
        entry.source_keys,
        sections,
      ),
      aliases: [],
    })),
  ];
  return { symbols, terms, frames, definitions, sections };
}
