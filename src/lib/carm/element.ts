const ELEMENT = "dpt-carm-figure";

class CArmFigureElement extends HTMLElement {
  #observer: IntersectionObserver | undefined;
  #dispose: (() => void) | undefined;
  #generation = 0;

  connectedCallback(): void {
    const generation = ++this.#generation;
    // Custom elements can connect before their children have been parsed.
    const observe = (): void => {
      if (!this.isConnected || generation !== this.#generation) return;
      let started = false;
      const start = (): void => {
        if (started) return;
        started = true;
        this.#observer?.disconnect();
        void this.#mount(generation);
      };
      if (!("IntersectionObserver" in window)) start();
      else {
        this.#observer = new IntersectionObserver(
          (entries) => {
            if (entries.some((entry) => entry.isIntersecting)) start();
          },
          { rootMargin: "240px" },
        );
        this.#observer.observe(this);
      }
    };
    if (document.readyState === "loading")
      document.addEventListener("DOMContentLoaded", observe, { once: true });
    else queueMicrotask(observe);
  }

  disconnectedCallback(): void {
    ++this.#generation;
    this.#observer?.disconnect();
    this.#observer = undefined;
    this.#dispose?.();
    this.#dispose = undefined;
  }

  async #mount(generation: number): Promise<void> {
    try {
      const { mountCArm } = await import("./viewer");
      if (!this.isConnected || generation !== this.#generation) return;
      this.#dispose = mountCArm(this);
    } catch (error) {
      if (!this.isConnected || generation !== this.#generation) return;
      const status = this.querySelector<HTMLElement>('[data-carm-id="status"]');
      if (status) {
        status.hidden = false;
        status.textContent =
          "Unable to load the C-arm view. Reload this page to try again.";
      }
      console.error("C-arm figure could not start", error);
    }
  }
}

export function mountCArmFigureElements(): void {
  if (!customElements.get(ELEMENT))
    customElements.define(ELEMENT, CArmFigureElement);
}
