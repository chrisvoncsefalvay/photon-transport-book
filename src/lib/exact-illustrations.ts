/** Exact display model for Appendix A's explicitly uniform scatter floor. */
export function scatterState(spr: number) {
  if (!Number.isFinite(spr) || spr < 0 || spr > 6)
    throw new RangeError("SPR must lie between 0 and 6");
  return { spr, totalBackground: 1 + spr, contrastPercent: 20 / (1 + spr) };
}

export function scatterPath(spr: number): string {
  const { totalBackground } = scatterState(spr);
  return Array.from({ length: 181 }, (_, i) => {
    const position = -3 + i / 30;
    const primary = 1 - 0.2 * Math.exp(-0.5 * (position / 0.5) ** 2);
    const normalised = (primary + spr) / totalBackground;
    const x = 76 + ((position + 3) / 6) * 324;
    const y = 244 - ((normalised - 0.75) / 0.28) * 226;
    return `${i ? "L" : "M"}${x.toFixed(3)} ${y.toFixed(3)}`;
  }).join(" ");
}

export function registerExactScatterFigure() {
  if (customElements.get("exact-scatter-figure")) return;
  class ExactScatterFigure extends HTMLElement {
    private events?: AbortController;
    connectedCallback() {
      this.events?.abort();
      this.events = new AbortController();
      const input = this.querySelector<HTMLInputElement>("[data-spr-input]");
      if (!input) return;
      const update = () => {
        const value = Number(input.value);
        const state = scatterState(value);
        this.querySelector("[data-scatter-profile]")?.setAttribute(
          "d",
          scatterPath(value),
        );
        const setText = (selector: string, text: string) => {
          const node = this.querySelector(selector);
          if (node) node.textContent = text;
        };
        setText("[data-spr-readout]", value.toFixed(1));
        setText(
          "[data-contrast-readout]",
          `${state.contrastPercent.toFixed(2)}%`,
        );
        setText("[data-floor-readout]", value.toFixed(1));
        setText("[data-total-readout]", state.totalBackground.toFixed(1));
        setText(
          "[data-profile-desc]",
          `The primary profile has 20 percent contrast. With scatter-to-primary ratio ${value.toFixed(1)}, the observed contrast is ${state.contrastPercent.toFixed(2)} percent. Each profile is divided by its own background signal.`,
        );
        const scatterPart = this.querySelector<HTMLElement>(
          "[data-scatter-part]",
        );
        if (scatterPart) scatterPart.style.flexGrow = String(value);
        input.setAttribute(
          "aria-valuetext",
          `SPR ${value.toFixed(1)}, observed contrast ${state.contrastPercent.toFixed(2)} percent`,
        );
      };
      input.disabled = false;
      input.addEventListener("input", update, { signal: this.events.signal });
      update();
    }
    disconnectedCallback() {
      this.events?.abort();
    }
  }
  customElements.define("exact-scatter-figure", ExactScatterFigure);
}
