/** Native double-click intervals follow the user's OS settings, not our delay.
 * If a delayed single action already ran, restore its starting state inside the
 * double action. A slow double-click must never also change vertical expansion.
 */
export function createSourceClickGesture(callbacks: {
  readVertical(): boolean;
  toggleVertical(): void;
  toggleHorizontal(restoreVertical?: boolean): void;
}) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  let initialVertical: boolean | undefined;
  return {
    click(detail: number): void {
      clearTimeout(timer);
      if (detail > 1) return;
      initialVertical = callbacks.readVertical();
      timer = setTimeout(callbacks.toggleVertical, 280);
    },
    doubleClick(): void {
      clearTimeout(timer);
      callbacks.toggleHorizontal(initialVertical);
      initialVertical = undefined;
    },
    cancel(): void {
      clearTimeout(timer);
      initialVertical = undefined;
    },
  };
}

/** Progressive enhancement for canonical source, independent of its language. */
export function mountSourceListings(): void {
  const listings = document.querySelectorAll<HTMLElement>(
    ".source-listing:not([data-source-ready])",
  );

  for (const listing of listings) {
    const codeSurface = listing.querySelector<HTMLElement>(
      ".source-listing__code",
    );
    const copyButton = listing.querySelector<HTMLButtonElement>(
      ".source-listing__copy",
    );
    const header = listing.querySelector<HTMLElement>(
      ".source-listing__header",
    );
    const toggleButton = listing.querySelector<HTMLButtonElement>(
      ".source-listing__toggle",
    );
    if (!codeSurface || !copyButton || !header || !toggleButton) continue;
    listing.dataset.sourceReady = "true";
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const listeners = new AbortController();
    let copyTimer: number | undefined;
    let animations: Animation[] = [];
    let selected = listing.querySelector<HTMLElement>(
      '[data-source-variant="0"]',
    )!;

    const cancelAnimations = () => {
      for (const animation of animations) animation.cancel();
      animations = [];
    };
    const animateChange = (change: () => void) => {
      const beforeWidth = listing.getBoundingClientRect().width;
      const beforeHeight = codeSurface.getBoundingClientRect().height;
      cancelAnimations();
      change();
      const width = listing.getBoundingClientRect().width;
      const height = codeSurface.getBoundingClientRect().height;
      if (motion.matches) return;
      const timing = {
        duration: 460,
        easing: "cubic-bezier(0.22, 1, 0.36, 1)",
      };
      if (Math.abs(beforeWidth - width) > 0.5) {
        animations.push(
          listing.animate(
            [{ width: `${beforeWidth}px` }, { width: `${width}px` }],
            timing,
          ),
        );
      }
      if (Math.abs(beforeHeight - height) > 0.5) {
        animations.push(
          codeSurface.animate(
            [
              { height: `${beforeHeight}px`, maxHeight: `${beforeHeight}px` },
              { height: `${height}px`, maxHeight: `${height}px` },
            ],
            timing,
          ),
        );
      }
    };
    const toggleVerticalExpansion = () => {
      if (listing.dataset.collapsible !== "true") return;
      animateChange(() => {
        const expanded = listing.dataset.expandedVertical !== "true";
        listing.dataset.expandedVertical = String(expanded);
        toggleButton.setAttribute("aria-expanded", String(expanded));
      });
    };
    const toggleHorizontalExpansion = (restoreVertical?: boolean) =>
      animateChange(() => {
        if (
          restoreVertical !== undefined &&
          listing.dataset.collapsible === "true"
        ) {
          listing.dataset.expandedVertical = String(restoreVertical);
          toggleButton.setAttribute("aria-expanded", String(restoreVertical));
        }
        const expanded = listing.dataset.expandedHorizontal !== "true";
        listing.dataset.expandedHorizontal = String(expanded);
        listing.classList.toggle("breakout-full", expanded);
        copyButton.hidden = !expanded;
      });
    const gesture = createSourceClickGesture({
      readVertical: () => listing.dataset.expandedVertical === "true",
      toggleVertical: toggleVerticalExpansion,
      toggleHorizontal: toggleHorizontalExpansion,
    });
    const isControl = (event: MouseEvent) =>
      event.target instanceof Element &&
      !event.target.closest(".source-listing__toggle") &&
      Boolean(event.target.closest("a, button, select, option"));
    for (const surface of [codeSurface, header]) {
      surface.addEventListener(
        "click",
        (event) => {
          if (isControl(event)) return;
          gesture.click(event.detail);
        },
        { signal: listeners.signal },
      );
      surface.addEventListener(
        "dblclick",
        (event) => {
          if (isControl(event)) return;
          event.preventDefault();
          gesture.doubleClick();
        },
        { signal: listeners.signal },
      );
    }
    for (const keyboardSurface of [codeSurface, toggleButton])
      keyboardSurface.addEventListener(
        "keydown",
        (event) => {
          if (event.key !== "Enter" && event.key !== " ") return;
          event.preventDefault();
          gesture.cancel();
          if (event.shiftKey) toggleHorizontalExpansion();
          else toggleVerticalExpansion();
        },
        { signal: listeners.signal },
      );
    const selector = listing.querySelector<HTMLSelectElement>(
      ".source-listing__language",
    );
    selector?.addEventListener(
      "change",
      () => {
        const next = listing.querySelector<HTMLElement>(
          `[data-source-variant="${selector.value}"]`,
        );
        if (!next) return;
        gesture.cancel();
        animateChange(() => {
          selected.hidden = true;
          next.hidden = false;
          selected = next;
          listing.dataset.sourceRegion = `${next.dataset.file}#${next.dataset.region}`;
          listing.dataset.collapsible = next.dataset.collapsible;
          listing.dataset.expandedVertical = "false";
          toggleButton.setAttribute(
            "aria-expanded",
            String(next.dataset.collapsible !== "true"),
          );
          codeSurface.setAttribute(
            "aria-label",
            `Source code from ${next.dataset.file}. Click or Enter to toggle height. Double-click or Shift+Enter to toggle page width.`,
          );
          listing.querySelector("[data-source-file]")!.textContent =
            next.dataset.file!;
          listing.querySelector(".source-listing__lines")!.textContent =
            next.dataset.lines!;
          listing.querySelector<HTMLAnchorElement>(
            ".source-listing__link",
          )!.href = next.dataset.url!;
        });
      },
      { signal: listeners.signal },
    );

    const copyText = async (text: string) => {
      if (window.isSecureContext && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
        return;
      }
      const active = document.activeElement;
      const temporaryInput = document.createElement("textarea");
      temporaryInput.value = text;
      temporaryInput.setAttribute("readonly", "");
      temporaryInput.style.position = "fixed";
      temporaryInput.style.opacity = "0";
      document.body.append(temporaryInput);
      temporaryInput.select();
      const legacyDocument = document as unknown as {
        execCommand(command: string): boolean;
      };
      const copied = legacyDocument.execCommand("copy");
      temporaryInput.remove();
      if (active instanceof HTMLElement) active.focus({ preventScroll: true });
      if (!copied) throw new Error("The browser rejected the copy command.");
    };
    copyButton.addEventListener(
      "click",
      async () => {
        window.clearTimeout(copyTimer);
        copyButton.dataset.copyState = "pending";
        let message = "Code copied";
        try {
          await copyText(selected.querySelector("pre code")?.textContent ?? "");
          copyButton.dataset.copyState = "success";
        } catch {
          copyButton.dataset.copyState = "failure";
          message = "Copy failed. Select the code to copy it manually.";
        }
        copyButton.setAttribute("aria-label", message);
        copyButton.title = message;
        listing.querySelector(".source-listing__copy-status")!.textContent =
          message;
        copyTimer = window.setTimeout(() => {
          delete copyButton.dataset.copyState;
          copyButton.setAttribute("aria-label", "Copy code");
          copyButton.title = "Copy code";
          listing.querySelector(".source-listing__copy-status")!.textContent =
            "";
        }, 4000);
      },
      { signal: listeners.signal },
    );
    let parentWidth = listing.parentElement?.clientWidth;
    const resizeObserver = new ResizeObserver(() => {
      const nextWidth = listing.parentElement?.clientWidth;
      if (nextWidth === parentWidth) return;
      parentWidth = nextWidth;
      if (animations.some((animation) => animation.playState === "running"))
        animateChange(() => {});
    });
    if (listing.parentElement) resizeObserver.observe(listing.parentElement);
    motion.addEventListener(
      "change",
      () => {
        if (motion.matches) cancelAnimations();
      },
      { signal: listeners.signal },
    );
    window.addEventListener(
      "pagehide",
      (event) => {
        if (event.persisted) return;
        cancelAnimations();
        gesture.cancel();
        window.clearTimeout(copyTimer);
        resizeObserver.disconnect();
        listeners.abort();
      },
      { signal: listeners.signal },
    );
  }
}
