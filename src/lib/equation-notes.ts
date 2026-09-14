const notationColours = [
  "#0072b2",
  "#d55e00",
  "#009e73",
  "#cc79a7",
  "#e69f00",
  "#56b4e9",
];

const mathText = (element: Element) =>
  (element.textContent ?? "").replace(/[\s\u200b-\u200d\ufeff]/gu, "");

// KaTeX carries mathematical font semantics on the glyph or its wrapper.
// Include them in matching: a scalar r cannot claim a vector bold r.
const mathFonts = (element: Element) =>
  [element, ...element.querySelectorAll("[class]")]
    .map((node) =>
      [...node.classList]
        .filter((name) =>
          /^(?:mathnormal|mathrm|mathbf|boldsymbol|mathit|mathsf|mathbb|mathcal|mathscr|mathtt|amsrm|textrm|textsf|texttt|textbf|textit)$/u.test(
            name,
          ),
        )
        .sort()
        .join(" "),
    )
    .filter(Boolean);

const mathDecorations = (element: Element) =>
  [element, ...element.querySelectorAll(".accent, .overline, .underline")]
    .filter((node) => node.matches(".accent, .overline, .underline"))
    .map((node) => [
      ["accent", "overline", "underline"].filter((kind) =>
        node.classList.contains(kind),
      ),
      // Stretchy accents use SVG paths: their text alone is empty, so a hat,
      // tilde and vector arrow must also compare the actual decoration shape.
      [...node.querySelectorAll(".accent-body")].map((body) => [
        mathText(body),
      ]),
      [...node.querySelectorAll("svg")].map((svg) => [
        svg.getAttribute("viewBox"),
        [...svg.querySelectorAll("path")].map((path) => path.getAttribute("d")),
      ]),
    ]);

// Compare complete rendered atoms, including their script structure. A subscript
// such as T_p must not match a product Tp, a superscript T^p or T_{pi}.
const atomSignature = (element: Element) =>
  JSON.stringify([
    mathText(element),
    mathFonts(element),
    [...element.querySelectorAll(".msupsub")].map((script) => [
      script.firstElementChild?.classList.contains("vlist-t2")
        ? "subscript"
        : "superscript",
      mathText(script),
    ]),
    element.querySelectorAll(".mfrac").length,
    mathDecorations(element),
  ]);

const atoms = (parent: Element) =>
  [...parent.children].filter(
    (element) =>
      !element.matches(".strut, .pstrut, .mspace, .vlist-s") &&
      mathText(element) !== "",
  );

export function findEquationSymbols(
  equation: Element,
  label: Element,
): HTMLElement[][] {
  const expected = [...label.querySelectorAll(".katex-html > .base")]
    .flatMap(atoms)
    .map(atomSignature);
  if (!expected.length) return [];

  const matches: HTMLElement[][] = [];
  for (const parent of equation.querySelectorAll(
    ".katex-html .base, .katex-html .mord, .katex-html .minner",
  )) {
    // Decorations belong to the symbol. Descending into their layout wrappers
    // would let plain T claim the contents of a distinct estimator, widehat T.
    if (parent.closest(".text, .accent, .overline, .underline")) continue;
    // Match a scripted atom as a whole; do not descend to its bare base and
    // silently treat T_p or T^p as T. Script contents remain separate atoms.
    if ([...parent.children].some((child) => child.matches(".msupsub")))
      continue;
    const candidates = atoms(parent);
    for (let i = 0; i <= candidates.length - expected.length; i++) {
      const sequence = candidates.slice(i, i + expected.length);
      if (
        sequence.every(
          (element, index) => atomSignature(element) === expected[index],
        ) &&
        !matches.some((match) => match[0] === sequence[0])
      ) {
        matches.push(sequence as HTMLElement[]);
      }
    }
  }
  return matches;
}

function precedingEquation(note: Element): Element | null {
  let sibling = note.previousElementSibling;
  while (sibling) {
    if (sibling.matches("h1, h2, h3, h4")) return null;
    if (sibling.matches(".equation")) return sibling;
    // Preserve the design-system fixture and unnumbered mathematical content.
    const display = sibling.matches(".katex-display")
      ? sibling
      : sibling.querySelector(".katex-display");
    if (display) return display;
    sibling = sibling.previousElementSibling;
  }
  return null;
}

export function initialiseEquationNotes(root: ParentNode = document) {
  const notes = [
    ...root.querySelectorAll<HTMLElement>(
      ".equation-note:not([data-notation-ready])",
    ),
  ];
  const bindings = notes.map((note, index) => {
    const equation = precedingEquation(note);
    const label = note.querySelector(".equation-note__symbol");
    return {
      note,
      equation,
      index,
      length: label
        ? mathText(label.querySelector(".katex-html") ?? label).length
        : 0,
      matches: equation && label ? findEquationSymbols(equation, label) : [],
    };
  });

  // Give composite terms priority over their constituent atoms (Delta s over s).
  bindings.sort((a, b) => b.length - a.length);
  for (const { note, equation, index, matches } of bindings) {
    note.dataset.notationReady = "true";
    const symbols = matches
      .filter((match) =>
        match.every(
          (element) =>
            !element.closest("[data-notation-symbol]") &&
            !element.querySelector("[data-notation-symbol]"),
        ),
      )
      .flat();
    if (!symbols.length) continue;

    const colour = notationColours[index % notationColours.length]!;
    note.style.setProperty("--notation-colour", colour);
    note.setAttribute("role", "button");
    note.tabIndex = 0;
    note.setAttribute("aria-pressed", "false");
    if (equation?.id) note.setAttribute("aria-controls", equation.id);
    symbols.forEach((symbol) => {
      symbol.dataset.notationSymbol = note.dataset.equationTerm ?? "";
      symbol.style.setProperty("--notation-colour", colour);
    });

    const setPreview = (active: boolean) => {
      for (const symbol of symbols) {
        if (active) symbol.dataset.notationPreview = "true";
        else delete symbol.dataset.notationPreview;
      }
    };
    const toggleLock = () => {
      const locked = note.getAttribute("aria-pressed") !== "true";
      note.dataset.notationLocked = String(locked);
      note.setAttribute("aria-pressed", String(locked));
      for (const symbol of symbols) {
        if (locked) symbol.dataset.notationLocked = "true";
        else delete symbol.dataset.notationLocked;
      }
    };
    note.addEventListener("pointerenter", () => setPreview(true));
    note.addEventListener("pointerleave", () => {
      if (note !== document.activeElement) setPreview(false);
    });
    note.addEventListener("focus", () => setPreview(true));
    note.addEventListener("blur", () => {
      if (!note.matches(":hover")) setPreview(false);
    });
    note.addEventListener("click", toggleLock);
    note.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      toggleLock();
    });
  }
}
