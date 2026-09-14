declare module "*tools/public/citations.mjs" {
  interface CitationRenderer {
    style: { title: string; format: string };
    keys: string[];
    renderCitation(keys: string[]): string;
    renderBibliography(keys?: string[]): string[];
  }

  export function loadCitationRenderer(options?: {
    root?: string;
    libraryPath?: string;
    stylePath?: string;
    overridesPath?: string;
  }): Promise<CitationRenderer>;
}
