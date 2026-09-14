interface CitationStyle {
  title: string;
  format: string;
}

export interface CitationRenderer {
  style: CitationStyle;
  keys: string[];
  citationNumber(key: string): number;
  renderCitation(keys: string[]): string;
  renderBibliography(keys?: string[]): string[];
}

import { loadCitationRenderer } from "../../tools/public/citations.mjs";

let rendererPromise: Promise<CitationRenderer> | undefined;

export function getCitationRenderer(): Promise<CitationRenderer> {
  if (!rendererPromise) {
    rendererPromise = loadCitationRenderer();
  }
  return rendererPromise;
}
