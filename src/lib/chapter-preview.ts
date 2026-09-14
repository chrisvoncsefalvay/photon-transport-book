interface ChapterPreview {
  row: HTMLElement;
  link: HTMLAnchorElement;
  viewport: HTMLElement;
  content: HTMLElement;
  prefix: HTMLElement;
  measure: HTMLElement;
  text: HTMLElement;
  toggle: HTMLButtonElement;
  label: HTMLElement;
  scope: string;
  continuation: string;
}

/** Reveal only the continuation, at a steady forty milliseconds per character. */
export function typedContinuation(
  continuation: string,
  elapsedMs: number,
): string {
  const count = Math.floor(Math.max(0, elapsedMs) / 40);
  return Array.from(continuation).slice(0, count).join("");
}

/** One preview owns the typing clock; every line keeps the short summary's style. */
export function initialiseChapterPreviews(root: HTMLElement): () => void {
  const previews = Array.from(
    root.querySelectorAll<HTMLElement>("[data-chapter-preview]"),
  ).flatMap((row): ChapterPreview[] => {
    const link = row.querySelector("a");
    const viewport = row.querySelector<HTMLElement>("[data-preview-viewport]");
    const content = row.querySelector<HTMLElement>("[data-preview-content]");
    const prefix = row.querySelector<HTMLElement>("[data-preview-prefix]");
    const measure = row.querySelector<HTMLElement>("[data-preview-measure]");
    const text = row.querySelector<HTMLElement>("[data-preview-text]");
    const toggle = row.querySelector<HTMLButtonElement>(
      "[data-preview-toggle]",
    );
    const label = row.querySelector<HTMLElement>("[data-preview-label]");
    const scope = prefix?.textContent?.trim();
    const continuation = text?.dataset.previewContinuation;
    return link &&
      viewport &&
      content &&
      prefix &&
      measure &&
      text &&
      toggle &&
      label &&
      scope &&
      continuation
      ? [
          {
            row,
            link,
            viewport,
            content,
            prefix,
            measure,
            text,
            toggle,
            label,
            scope,
            continuation,
          },
        ]
      : [];
  });
  const hover = window.matchMedia("(hover: hover) and (pointer: fine)");
  const events = new AbortController();
  const { signal } = events;
  let active: ChapterPreview | null = null;
  let desired: ChapterPreview | null = null;
  let pointed: ChapterPreview | null = null;
  let focused: ChapterPreview | null = null;
  let pinned: ChapterPreview | null = null;
  let frame = 0;
  let visibilityFrame = 0;
  let openingTimer = 0;
  let closingTimer = 0;
  let closing: ChapterPreview | null = null;
  let closingUntil = 0;

  function previewAt(target: EventTarget | null): ChapterPreview | null {
    if (!(target instanceof Element)) return null;
    const row = target.closest("[data-chapter-preview]");
    return previews.find((preview) => preview.row === row) ?? null;
  }

  function resize(preview: ChapterPreview): void {
    preview.viewport.style.height = `${preview.content.getBoundingClientRect().height}px`;
  }

  function collapseHeight(preview: ChapterPreview): void {
    preview.viewport.style.height = `${preview.measure.getBoundingClientRect().height}px`;
  }

  function renderContinuation(
    preview: ChapterPreview,
    continuation: string,
  ): void {
    preview.prefix.textContent = continuation
      ? preview.scope.replace(/\.$/, "")
      : preview.scope;
    preview.text.textContent = continuation;
    resize(preview);
  }

  function reset(preview: ChapterPreview): void {
    preview.row.dataset.previewState = "closed";
    preview.text.textContent = "";
    preview.prefix.textContent = preview.scope;
    collapseHeight(preview);
    preview.toggle.setAttribute("aria-expanded", "false");
    preview.toggle.setAttribute(
      "aria-label",
      preview.toggle
        .getAttribute("aria-label")!
        .replace("Read less", "Read more"),
    );
    preview.label.textContent = "Read more";
  }

  function finishClosing(): void {
    window.clearTimeout(closingTimer);
    if (closing) reset(closing);
    closing = null;
    closingUntil = 0;
  }

  function closeActive(): void {
    window.cancelAnimationFrame(frame);
    if (!active) return;
    finishClosing();
    const previous = active;
    active = null;
    previous.row.dataset.previewState = "closing";
    previous.toggle.setAttribute("aria-expanded", "false");
    previous.toggle.setAttribute(
      "aria-label",
      previous.toggle
        .getAttribute("aria-label")!
        .replace("Read less", "Read more"),
    );
    previous.label.textContent = "Read more";
    collapseHeight(previous);
    const duration = Math.max(
      ...getComputedStyle(previous.viewport)
        .transitionDuration.split(",")
        .map(
          (value) =>
            parseFloat(value) * (value.trim().endsWith("ms") ? 1 : 1000),
        ),
    );
    closing = previous;
    closingUntil = performance.now() + duration;
    if (duration === 0) finishClosing();
    else closingTimer = window.setTimeout(finishClosing, duration);
  }

  function open(preview: ChapterPreview): void {
    if (desired !== preview) return;
    finishClosing();
    active = preview;
    preview.toggle.setAttribute("aria-expanded", "true");
    preview.toggle.setAttribute(
      "aria-label",
      preview.toggle
        .getAttribute("aria-label")!
        .replace("Read more", "Read less"),
    );
    preview.label.textContent = "Read less";

    preview.row.dataset.previewState = "typing";
    const start = performance.now();
    let written = "";
    renderContinuation(preview, "");

    const tick = (now: number): void => {
      if (active !== preview || desired !== preview) return;
      const continuation = typedContinuation(preview.continuation, now - start);
      if (continuation !== written) {
        renderContinuation(preview, continuation);
        written = continuation;
      }
      if (continuation !== preview.continuation)
        frame = window.requestAnimationFrame(tick);
      else preview.row.dataset.previewState = "expanded";
    };
    frame = window.requestAnimationFrame(tick);
  }

  function request(preview: ChapterPreview | null): void {
    if (desired === preview) return;
    desired = preview;
    window.clearTimeout(openingTimer);
    closeActive();
    if (!preview) return;
    // Let the previous row settle before another starts growing. Typing is
    // always enabled; reduced motion only removes the CSS height/opacity tween.
    const delay = Math.max(0, closingUntil - performance.now()) + 120;
    openingTimer = window.setTimeout(() => open(preview), delay);
  }

  function resolveIntent(): void {
    request(pinned ?? focused ?? pointed);
  }

  // Pointer movement, rather than pointerenter, avoids a stationary pointer
  // selecting new rows as the changing text height moves their boundaries.
  root.addEventListener(
    "pointermove",
    (event) => {
      if (!hover.matches || event.pointerType === "touch") return;
      pointed = previewAt(event.target);
      if ((focused && focused !== pointed) || (pinned && pinned !== pointed)) {
        focused = null;
        pinned = null;
      }
      resolveIntent();
    },
    { signal },
  );
  root.addEventListener(
    "pointerleave",
    () => {
      pointed = null;
      resolveIntent();
    },
    { signal },
  );
  root.addEventListener(
    "focusin",
    (event) => {
      if (!(event.target instanceof HTMLAnchorElement)) return;
      if (!event.target.matches(":focus-visible")) return;
      focused = previewAt(event.target);
      resolveIntent();
    },
    { signal },
  );
  root.addEventListener(
    "focusout",
    (event) => {
      const next = previewAt(event.relatedTarget);
      if (next === focused) return;
      focused = null;
      resolveIntent();
    },
    { signal },
  );
  root.addEventListener(
    "click",
    (event) => {
      if (!(event.target instanceof Element)) return;
      if (!event.target.closest("[data-preview-toggle]")) return;
      const preview = previewAt(event.target);
      const wasOpen = desired === preview;
      pinned = wasOpen ? null : preview;
      if (wasOpen) {
        pointed = null;
        focused = null;
      }
      resolveIntent();
    },
    { signal },
  );
  root.addEventListener(
    "keydown",
    (event) => {
      if (event.key !== "Escape") return;
      pinned = null;
      pointed = null;
      focused = null;
      request(null);
    },
    { signal },
  );

  const observer = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const preview = previews.find(
        (item) =>
          item.content === entry.target || item.measure === entry.target,
      );
      if (!preview) continue;
      if (preview === active) resize(preview);
      else collapseHeight(preview);
    }
  });
  previews.forEach((preview) => {
    observer.observe(preview.content);
    observer.observe(preview.measure);
    collapseHeight(preview);
  });
  window.addEventListener(
    "scroll",
    () => {
      if (visibilityFrame || !desired) return;
      visibilityFrame = window.requestAnimationFrame(() => {
        visibilityFrame = 0;
        const preview = active ?? desired;
        if (!preview) return;
        const bounds = preview.row.getBoundingClientRect();
        if (bounds.bottom > 0 && bounds.top < window.innerHeight) return;
        // Scrolling can leave the pointer stationary over a different row.
        // Dismiss the old intent without starting a new offscreen preview.
        pointed = null;
        focused = null;
        pinned = null;
        request(null);
      });
    },
    { capture: true, passive: true, signal },
  );
  root.dataset.previewReady = "true";

  return () => {
    events.abort();
    observer.disconnect();
    window.cancelAnimationFrame(frame);
    window.cancelAnimationFrame(visibilityFrame);
    window.clearTimeout(openingTimer);
    finishClosing();
    previews.forEach(reset);
    delete root.dataset.previewReady;
  };
}
