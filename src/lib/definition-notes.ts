import { findEquationSymbols } from "./equation-notes";
import type { DefinitionEntry } from "./terminology";

interface TermMatch {
  start: number;
  end: number;
  id: string;
}
interface TermName {
  name: string;
  id: string;
  caseSensitive: boolean;
}

const escapePattern = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
const compactMath = (node: Element) =>
  (node.textContent ?? "").replace(/[\s\u200b-\u200d\ufeff]/gu, "");

/** Only recorded names and aliases participate; abbreviations retain their case. */
export function createTermMatcher(entries: DefinitionEntry[]) {
  const names = entries
    .filter((entry) => entry.kind === "term")
    .flatMap((entry) =>
      [entry.label, ...entry.aliases].map((name): TermName => ({
        name,
        id: entry.id,
        caseSensitive: !/\s/u.test(name) && /[A-Z]{2}|[a-z][A-Z]/u.test(name),
      })),
    )
    .sort((a, b) => b.name.length - a.name.length);
  const byName = new Map(
    names.map((item) => [item.name.toLocaleLowerCase("en-GB"), item]),
  );
  const expression = names.length
    ? new RegExp(
        `(?<![\\p{L}\\p{N}_])(?:${names.map((item) => escapePattern(item.name)).join("|")})(?![\\p{L}\\p{N}_])`,
        "giu",
      )
    : null;
  return (text: string): TermMatch[] => {
    if (!expression) return [];
    expression.lastIndex = 0;
    return [...text.matchAll(expression)].flatMap((match) => {
      const item = byName.get(match[0].toLocaleLowerCase("en-GB"));
      if (!item || (item.caseSensitive && item.name !== match[0])) return [];
      return [
        { start: match.index, end: match.index + match[0].length, id: item.id },
      ];
    });
  };
}

/** A long qualification never makes the reader wait longer than 1.6 seconds. */
export function typedDefinition(
  text: string,
  elapsedMs: number,
  reducedMotion = false,
): string {
  const characters = Array.from(text);
  if (reducedMotion) return text;
  const duration = Math.min(1600, characters.length * 18);
  const count = duration
    ? Math.floor((characters.length * Math.max(0, elapsedMs)) / duration)
    : 0;
  return characters.slice(0, count).join("");
}

export function notationMatches(
  searchable: string,
  query: string,
  entryKind: string,
  kind: string,
  entryDomain: string,
  domain: string,
): boolean {
  return (
    (kind === "all" || kind === entryKind) &&
    (domain === "all" || domain === entryDomain) &&
    query
      .toLocaleLowerCase("en-GB")
      .trim()
      .split(/\s+/u)
      .every((word) => searchable.toLocaleLowerCase("en-GB").includes(word))
  );
}

export function initialiseNotationSearch(root: HTMLElement): () => void {
  const form = root.querySelector<HTMLFormElement>("[data-notation-filters]");
  if (!form) return () => {};
  const events = new AbortController();
  const { signal } = events;
  const query = form.elements.namedItem("query") as HTMLInputElement;
  const kind = form.elements.namedItem("kind") as HTMLSelectElement;
  const domain = form.elements.namedItem("domain") as HTMLSelectElement;
  const rows = [...root.querySelectorAll<HTMLElement>("[data-notation-entry]")];
  const update = () => {
    let visible = 0;
    rows.forEach((row) => {
      row.hidden = !notationMatches(
        row.dataset.search ?? "",
        query.value,
        row.dataset.kind ?? "",
        kind.value,
        row.dataset.domain ?? "",
        domain.value,
      );
      visible += Number(!row.hidden);
    });
    const status = form.querySelector("[data-notation-count]");
    if (status) status.textContent = `${visible} of ${rows.length} entries`;
    root
      .querySelectorAll<HTMLElement>("[data-notation-section]")
      .forEach((section) => {
        const empty = section.querySelector<HTMLElement>(
          "[data-notation-empty]",
        );
        if (empty)
          empty.hidden = Boolean(
            section.querySelector("[data-notation-entry]:not([hidden])"),
          );
      });
  };
  const showLinkedEntry = () => {
    let id: string;
    try {
      id = decodeURIComponent(window.location.hash.slice(1));
    } catch {
      return;
    }
    const target = document.getElementById(id);
    if (!target?.matches("[data-notation-entry]") || !root.contains(target))
      return;
    if (target.hidden) {
      query.value = "";
      kind.value = "all";
      domain.value = "all";
      update();
      target.scrollIntoView({ block: "center" });
    }
  };
  form.hidden = false;
  form.addEventListener("submit", (event) => event.preventDefault(), {
    signal,
  });
  form.addEventListener("input", update, { signal });
  form.addEventListener("change", update, { signal });
  form.addEventListener(
    "reset",
    (event) => {
      event.preventDefault();
      query.value = "";
      kind.value = "all";
      domain.value = "all";
      update();
    },
    { signal },
  );
  window.addEventListener("hashchange", showLinkedEntry, { signal });
  update();
  showLinkedEntry();
  return () => {
    events.abort();
    form.hidden = true;
    rows.forEach((row) => {
      row.hidden = false;
    });
  };
}

const excluded =
  "a, button, input, select, textarea, code, pre, table, figure, .figure-plate, .source-listing, .bibliography, .references, .margin-note, .equation-note, definition-notes, h1, h2, h3, h4, h5, h6, [data-definition-key]";

function precedingEquation(note: Element): Element | null {
  let sibling = note.previousElementSibling;
  while (sibling) {
    if (sibling.matches("h1, h2, h3, h4")) return null;
    if (sibling.matches(".equation, .katex-display")) return sibling;
    const display = sibling.querySelector(".katex-display");
    if (display) return display;
    sibling = sibling.previousElementSibling;
  }
  return null;
}

/** The enhancement is disposable; registry data and MathML remain unchanged. */
export function initialiseDefinitionNotes(host: HTMLElement): () => void {
  const prose = host.closest("article")?.querySelector<HTMLElement>(".prose");
  const panel = host.querySelector<HTMLElement>("[data-definition-panel]");
  const content = host.querySelector<HTMLElement>("[data-definition-content]");
  const source = host.querySelector("[data-definition-registry]");
  if (!prose || !panel || !content || !source?.textContent) return () => {};
  const started = performance.now();
  const entries: DefinitionEntry[] = JSON.parse(source.textContent);
  const entryMap = new Map(entries.map((entry) => [entry.id, entry]));
  const contexts = new Map<string, DefinitionEntry[]>();
  const termButtons: HTMLButtonElement[] = [];
  const marked = new Set<HTMLElement>();
  const events = new AbortController();
  const { signal } = events;
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const hover = window.matchMedia("(hover: hover)");
  panel.id ||= "definition-sidenote";
  content.id ||= `${panel.id}-content`;
  let serial = 0;
  let active: HTMLElement | null = null;
  let activeKey = "";
  let pinned = false;
  let restoringFocus = false;
  let typingFrame = 0;
  let placementFrame = 0;
  let closingTimer = 0;
  let typedParts: { node: HTMLElement; text: string; offset: number }[] = [];

  const newContext = (definitions: DefinitionEntry[]) => {
    const key = `definition-${++serial}`;
    contexts.set(key, definitions);
    return key;
  };
  const mark = (node: HTMLElement, key: string) => {
    node.dataset.definitionKey = key;
    marked.add(node);
  };
  const occupied = (nodes: HTMLElement[]) =>
    nodes.some(
      (node) =>
        node.closest("[data-definition-key]") ||
        node.querySelector("[data-definition-key]"),
    );

  // Nearby manuscript definitions outrank registry meanings, including aliases
  // deliberately shortened by the author in this particular equation.
  const locals = [...prose.querySelectorAll<HTMLElement>(".equation-note")]
    .flatMap((note, index) => {
      const equation = precedingEquation(note);
      const label = note.querySelector<HTMLElement>(".equation-note__symbol");
      const definition = note
        .querySelector(".equation-note__definition")
        ?.cloneNode(true) as HTMLElement | undefined;
      if (
        !equation ||
        !label ||
        !definition ||
        equation.closest("figure, table, .figure-plate")
      )
        return [];
      definition
        .querySelectorAll(".katex-mathml")
        .forEach((node) => node.remove());
      const entry: DefinitionEntry = {
        id: `local-${index}`,
        kind: "local",
        label: note.dataset.equationTerm ?? "Symbol",
        html: label.innerHTML,
        definition: definition.textContent?.trim() ?? "",
        notes: null,
        units: null,
        domain: "In this equation",
        sources: equation.id
          ? [{ href: `#${equation.id}`, label: "This equation" }]
          : [],
        aliases: [],
      };
      return [
        {
          equation,
          label,
          entry,
          length: compactMath(label.querySelector(".katex-html") ?? label)
            .length,
        },
      ];
    })
    .sort((a, b) => b.length - a.length);
  for (const local of locals) {
    const key = newContext([local.entry]);
    for (const match of findEquationSymbols(local.equation, local.label)) {
      if (occupied(match)) continue;
      match.forEach((node) => mark(node, key));
    }
  }

  // Group identical rendered MathML, not raw TeX spelling. The matcher retains
  // scripts, accents and mathematical font semantics. No scope is guessed.
  const labels = new Map<
    string,
    { label: HTMLElement; entries: DefinitionEntry[]; text: string }
  >();
  for (const entry of entries.filter((item) => item.kind === "symbol")) {
    const label = document.createElement("span");
    label.innerHTML = entry.html ?? "";
    const semantic =
      label.querySelector(".katex-mathml semantics")?.firstElementChild
        ?.outerHTML ??
      entry.html ??
      entry.id;
    const group = labels.get(semantic);
    if (group) group.entries.push(entry);
    else
      labels.set(semantic, {
        label,
        entries: [entry],
        text: compactMath(label.querySelector(".katex-html") ?? label),
      });
  }
  const candidates = [...labels.values()].sort(
    (a, b) => b.text.length - a.text.length,
  );
  const maths = [...prose.querySelectorAll<HTMLElement>(".katex")].filter(
    (math) => !math.closest(excluded),
  );
  for (const math of maths) {
    const text = compactMath(math.querySelector(".katex-html") ?? math);
    for (const candidate of candidates) {
      if (!text.includes(candidate.text)) continue;
      const matches = findEquationSymbols(math, candidate.label).filter(
        (match) => !occupied(match),
      );
      if (!matches.length) continue;
      const key = newContext(candidate.entries);
      for (const match of matches) match.forEach((node) => mark(node, key));
    }
  }

  const matchTerms = createTermMatcher(entries);
  const walker = document.createTreeWalker(prose, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.parentElement?.closest(
        `${excluded}, .katex, .equation, .katex-display, script, style`,
      )
        ? NodeFilter.FILTER_REJECT
        : NodeFilter.FILTER_ACCEPT,
  });
  const textNodes: Text[] = [];
  let next: Node | null;
  while ((next = walker.nextNode())) textNodes.push(next as Text);
  for (const node of textNodes) {
    const matches = matchTerms(node.data);
    if (!matches.length) continue;
    const fragment = document.createDocumentFragment();
    let cursor = 0;
    for (const match of matches) {
      const entry = entryMap.get(match.id);
      if (!entry) continue;
      fragment.append(node.data.slice(cursor, match.start));
      const button = document.createElement("button");
      button.type = "button";
      button.className = "defined-term";
      button.textContent = node.data.slice(match.start, match.end);
      button.setAttribute("aria-label", `Define ${button.textContent}`);
      button.setAttribute("aria-controls", panel.id);
      button.setAttribute("aria-expanded", "false");
      mark(button, newContext([entry]));
      termButtons.push(button);
      fragment.append(button);
      cursor = match.end;
    }
    fragment.append(node.data.slice(cursor));
    node.replaceWith(fragment);
  }

  function place() {
    if (!active || panel!.hidden) return;
    if (window.matchMedia("(min-width: 86rem)").matches) {
      const proseBounds = prose!.getBoundingClientRect();
      const bounds = active.getBoundingClientRect();
      panel!.style.setProperty(
        "--definition-left",
        `${Math.min(proseBounds.right + 40, window.innerWidth - 232)}px`,
      );
      panel!.style.setProperty(
        "--definition-top",
        `${Math.max(96, Math.min(bounds.top, window.innerHeight - panel!.offsetHeight - 16))}px`,
      );
    }
  }

  function stopTyping() {
    window.cancelAnimationFrame(typingFrame);
    typingFrame = 0;
  }
  function finishTyping() {
    stopTyping();
    typedParts.forEach((part) => {
      part.node.textContent = part.text;
    });
  }
  function hide(restoreFocus = false) {
    window.clearTimeout(closingTimer);
    stopTyping();
    const previous = active;
    const containedFocus = panel!.contains(document.activeElement);
    if (active instanceof HTMLButtonElement) {
      active.setAttribute("aria-expanded", "false");
      active.removeAttribute("aria-describedby");
    }
    active = null;
    activeKey = "";
    pinned = false;
    panel!.hidden = true;
    if (
      restoreFocus &&
      containedFocus &&
      previous instanceof HTMLButtonElement
    ) {
      restoringFocus = true;
      previous.focus();
      restoringFocus = false;
    }
  }

  function addTypedParagraph(
    parent: HTMLElement,
    value: string,
    className = "",
    html?: string,
  ) {
    const paragraph = document.createElement("p");
    paragraph.className = className;
    if (html && (html.includes('class="katex"') || html.includes("<a "))) {
      // Keep typeset expressions and native links intact, including their MathML.
      paragraph.innerHTML = html;
      parent.append(paragraph);
      return;
    }
    const full = document.createElement("span");
    full.className = "definition-sr-only";
    full.textContent = value;
    const visual = document.createElement("span");
    visual.setAttribute("aria-hidden", "true");
    paragraph.append(full, visual);
    parent.append(paragraph);
    typedParts.push({
      node: visual,
      text: value,
      offset: typedParts.reduce(
        (total, item) => total + Array.from(item.text).length,
        0,
      ),
    });
  }

  function show(
    trigger: HTMLElement,
    key: string,
    intent: "hover" | "focus" | "pin",
  ) {
    window.clearTimeout(closingTimer);
    if (pinned && intent === "hover") return;
    const definitions = contexts.get(key);
    if (!definitions?.length) return;
    if (activeKey === key && active === trigger) {
      if (intent === "pin") pinned = true;
      const status = panel!.querySelector("[data-definition-status]");
      if (status)
        status.textContent = pinned ? "Definition · pinned" : "Definition";
      return;
    }
    stopTyping();
    if (active instanceof HTMLButtonElement) {
      active.setAttribute("aria-expanded", "false");
      active.removeAttribute("aria-describedby");
    }
    active = trigger;
    activeKey = key;
    pinned = intent === "pin";
    typedParts = [];
    content!.replaceChildren();
    const status = panel!.querySelector("[data-definition-status]");
    if (status)
      status.textContent = pinned ? "Definition · pinned" : "Definition";
    if (definitions.length > 1) {
      const heading = document.createElement("p");
      heading.className = "definition-sidenote__multiple";
      heading.textContent = "Recorded meanings";
      content!.append(heading);
    }
    for (const definition of definitions) {
      const section = document.createElement("section");
      section.className = "definition-sidenote__entry";
      const heading = document.createElement("h3");
      if (definition.html) heading.innerHTML = definition.html;
      else heading.textContent = definition.label;
      section.append(heading);
      const domain = document.createElement("p");
      domain.className = "definition-sidenote__domain";
      domain.textContent = definition.domain.replaceAll("-", " ");
      section.append(domain);
      addTypedParagraph(
        section,
        definition.definition,
        "",
        definition.definitionHtml,
      );
      if (definition.units !== null) {
        const units = document.createElement("p");
        units.className = "definition-sidenote__units";
        if (definition.unitsHtml && definition.units !== "1")
          units.innerHTML = `Units: ${definition.unitsHtml}`;
        else
          units.textContent = `Units: ${definition.units === "1" ? "1 (dimensionless)" : definition.units}`;
        section.append(units);
      }
      if (definition.notes)
        addTypedParagraph(
          section,
          definition.notes,
          "definition-sidenote__qualification",
          definition.notesHtml,
        );
      const links = document.createElement("ul");
      links.className = "definition-sidenote__sources";
      for (const sourceLink of definition.sources) {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = sourceLink.href;
        link.textContent = sourceLink.label;
        item.append(link);
        links.append(item);
      }
      if (definition.kind !== "local") {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = `/notation/#${definition.id}`;
        link.textContent = "Reference entry";
        item.append(link);
        links.append(item);
      }
      section.append(links);
      content!.append(section);
    }
    panel!.hidden = false;
    panel!.scrollTop = 0;
    if (trigger instanceof HTMLButtonElement) {
      trigger.setAttribute("aria-expanded", "true");
      trigger.setAttribute("aria-describedby", content!.id);
    }
    const total = typedParts.map((part) => part.text).join("");
    const start = performance.now();
    const tick = () => {
      const count = Array.from(
        typedDefinition(total, performance.now() - start, motion.matches),
      ).length;
      typedParts.forEach((part) => {
        part.node.textContent = Array.from(part.text)
          .slice(0, Math.max(0, count - part.offset))
          .join("");
      });
      place();
      if (count < Array.from(total).length)
        typingFrame = window.requestAnimationFrame(tick);
      else typingFrame = 0;
    };
    if (motion.matches) finishTyping();
    else tick();
    place();
  }

  const triggerAt = (target: EventTarget | null): HTMLElement | null =>
    target instanceof Element
      ? target.closest<HTMLElement>("[data-definition-key]")
      : null;
  const delayHide = () => {
    window.clearTimeout(closingTimer);
    if (
      pinned ||
      panel!.contains(document.activeElement) ||
      active === document.activeElement
    )
      return;
    closingTimer = window.setTimeout(() => hide(), 220);
  };
  prose.addEventListener(
    "pointerover",
    (event) => {
      if (!hover.matches || event.pointerType === "touch") return;
      const trigger = triggerAt(event.target);
      if (trigger && !trigger.contains(event.relatedTarget as Node | null))
        show(trigger, trigger.dataset.definitionKey!, "hover");
    },
    { signal },
  );
  prose.addEventListener(
    "pointerout",
    (event) => {
      const trigger = triggerAt(event.target);
      if (trigger && !trigger.contains(event.relatedTarget as Node | null))
        delayHide();
    },
    { signal },
  );
  prose.addEventListener(
    "focusin",
    (event) => {
      if (restoringFocus) return;
      const trigger = triggerAt(event.target);
      if (trigger) show(trigger, trigger.dataset.definitionKey!, "focus");
    },
    { signal },
  );
  prose.addEventListener(
    "focusout",
    () => {
      window.clearTimeout(closingTimer);
      closingTimer = window.setTimeout(delayHide, 0);
    },
    { signal },
  );
  document.addEventListener(
    "click",
    (event) => {
      const trigger = triggerAt(event.target);
      if (trigger && prose.contains(trigger)) {
        event.preventDefault();
        if (active === trigger && pinned) hide();
        else {
          show(trigger, trigger.dataset.definitionKey!, "pin");
          if (event.detail === 0)
            panel
              .querySelector<HTMLButtonElement>("[data-definition-close]")
              ?.focus();
        }
      } else if (event.target instanceof Node && !panel.contains(event.target))
        hide();
    },
    { signal },
  );
  panel.addEventListener(
    "pointerenter",
    () => window.clearTimeout(closingTimer),
    { signal },
  );
  panel.addEventListener("pointerleave", delayHide, { signal });
  panel.addEventListener("focusin", () => window.clearTimeout(closingTimer), {
    signal,
  });
  panel.addEventListener(
    "focusout",
    () => {
      window.clearTimeout(closingTimer);
      closingTimer = window.setTimeout(delayHide, 0);
    },
    { signal },
  );
  panel
    .querySelector("[data-definition-close]")
    ?.addEventListener("click", () => hide(true), { signal });
  document.addEventListener(
    "keydown",
    (event) => {
      if (event.key === "Escape" && !panel.hidden) {
        event.preventDefault();
        hide(true);
      }
    },
    { signal },
  );
  const queuePlacement = () => {
    if (placementFrame || !active) return;
    placementFrame = window.requestAnimationFrame(() => {
      placementFrame = 0;
      if (!active) return;
      const bounds = active.getBoundingClientRect();
      if (!pinned && (bounds.bottom < 0 || bounds.top > window.innerHeight))
        hide();
      else place();
    });
  };
  window.addEventListener("scroll", queuePlacement, {
    passive: true,
    capture: true,
    signal,
  });
  window.addEventListener("resize", queuePlacement, { passive: true, signal });
  motion.addEventListener("change", finishTyping, { signal });
  host.dataset.definitionReady = "true";
  host.dataset.definitionInitMs = String(
    Math.round(performance.now() - started),
  );
  host.dataset.definitionSymbolBindings = String(
    [...marked].filter((node) => !node.matches("button")).length,
  );
  host.dataset.definitionTermBindings = String(termButtons.length);

  return () => {
    events.abort();
    hide();
    window.cancelAnimationFrame(placementFrame);
    termButtons.forEach((button) =>
      button.replaceWith(document.createTextNode(button.textContent ?? "")),
    );
    marked.forEach((node) => {
      delete node.dataset.definitionKey;
    });
    delete host.dataset.definitionReady;
    delete host.dataset.definitionInitMs;
    delete host.dataset.definitionSymbolBindings;
    delete host.dataset.definitionTermBindings;
  };
}
