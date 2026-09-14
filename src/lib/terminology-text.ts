import { renderToString } from "katex";

export interface ManuscriptSection {
  number: string;
  href: string;
  label: string;
}

export function escapeTerminologyText(value: string): string {
  return value.replace(/[&<>"']/gu, (character) => {
    return {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[character]!;
  });
}

/** Keep familiar searches such as mm^-1 and T_p alongside the authored LaTeX. */
export function terminologySearchText(value: string): string {
  const aliases = new Set<string>();
  const plain = value.replace(/\$([^$\n]+)\$/gu, (_, latex: string) => {
    const unstyled = latex
      .replace(
        /\\(?:mathrm|mathbf|mathsf|mathtt|boldsymbol|mathcal|operatorname)\b\s*/gu,
        "",
      )
      .replace(/\\([A-Za-z]+)/gu, "$1");
    if (/\^(?:\{-1\}|-1(?!\d))/u.test(unstyled)) aliases.add("inverse");
    if (/\^(?:\{(?:T|top)\}|T|top\b)/u.test(unstyled)) aliases.add("transpose");
    const text = unstyled.replace(/[{}]/gu, "");
    for (const accent of text.matchAll(
      /\b(widehat|hat|overline|bar)\s*([A-Za-z]+)_([A-Za-z0-9]+)/gu,
    )) {
      aliases.add(
        `${accent[2]}-${accent[1]!.includes("hat") ? "hat" : "bar"}-${accent[3]}`,
      );
    }
    return text;
  });
  return `${value} ${plain} ${[...aliases].join(" ")}`;
}

/** Only authored inline maths and known manuscript references become markup. */
export function renderTerminologyText(
  value: string,
  sections: readonly ManuscriptSection[] = [],
): string {
  const byNumber = new Map(
    sections.map((section) => [section.number, section]),
  );
  const byHref = new Map(sections.map((section) => [section.href, section]));
  const tokens =
    /\$([^$\n]+)\$|(?:Manuscript\s+)?§{1,2}\s*((?:[A-Z]|\d+)\.\d+)|\/(?:chapters|appendix)\/[\w/-]+#[\w-]+/gu;
  let result = "";
  let offset = 0;
  for (const match of value.matchAll(tokens)) {
    result += escapeTerminologyText(value.slice(offset, match.index));
    if (match[1]) {
      result += renderToString(match[1], {
        throwOnError: true,
        strict: "error",
        trust: false,
        output: "htmlAndMathml",
      });
    } else {
      const section = match[2] ? byNumber.get(match[2]) : byHref.get(match[0]);
      result += section
        ? `<a href="${escapeTerminologyText(section.href)}">${escapeTerminologyText(match[2] ? match[0] : section.label)}</a>`
        : escapeTerminologyText(match[0]);
    }
    offset = match.index + match[0].length;
  }
  return result + escapeTerminologyText(value.slice(offset));
}
