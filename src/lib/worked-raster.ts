import {
  differencePlane,
  materialPlane,
  paletteColour,
  rasterPixels,
  validateRasterPayload,
  type RasterPayload,
  type RasterPlane,
  type RGB,
  type RasterScale,
} from "./worked-raster-data";

export class WorkedRaster extends HTMLElement {
  private payload: RasterPayload | null = null;
  private planes = new Map<string, RasterPlane>();
  private observer: IntersectionObserver | null = null;
  private controller: AbortController | null = null;
  private selection: [number, number] | null = null;

  connectedCallback() {
    if (this.payload || this.controller) return;
    this.observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          this.observer?.disconnect();
          void this.load();
        }
      },
      { rootMargin: "240px" },
    );
    this.observer.observe(this);
  }
  disconnectedCallback() {
    this.observer?.disconnect();
    this.controller?.abort();
    this.controller = null;
  }
  private async load() {
    const controller = new AbortController();
    this.controller = controller;
    try {
      const response = await fetch(this.dataset.arrays!, {
        signal: controller.signal,
      });
      if (!response.ok)
        throw new Error(`Recorded array request failed: ${response.status}`);
      const payload = validateRasterPayload(
        await response.json(),
        this.dataset.kind as RasterPayload["kind"],
      );
      if (!this.isConnected || controller.signal.aborted) return;
      this.payload = payload;
      const slider = this.querySelector<HTMLInputElement>('[data-control="z"]');
      if (slider && this.payload.grid) {
        slider.min = String(this.payload.grid.origin_mm[2]);
        slider.max = String(
          this.payload.grid.origin_mm[2]! +
            15 * this.payload.grid.spacing_mm[2]!,
        );
        slider.step = String(this.payload.grid.spacing_mm[2]! / 2);
        slider.value = "0";
      }
      this.querySelectorAll<HTMLInputElement | HTMLSelectElement>(
        "[data-control]",
      ).forEach((control) => {
        control.addEventListener("input", () => {
          this.selection = null;
          this.draw();
        });
      });
      this.querySelectorAll<HTMLCanvasElement>("canvas").forEach((canvas) => {
        canvas.addEventListener("pointermove", (event) => {
          const rect = canvas.getBoundingClientRect();
          this.select(
            Math.floor(
              ((event.clientX - rect.left) / rect.width) * canvas.width,
            ),
            canvas.height -
              1 -
              Math.floor(
                ((event.clientY - rect.top) / rect.height) * canvas.height,
              ),
            false,
          );
        });
        canvas.addEventListener("keydown", (event) => {
          const move = (
            {
              ArrowLeft: [-1, 0],
              ArrowRight: [1, 0],
              ArrowUp: [0, 1],
              ArrowDown: [0, -1],
            } as Record<string, number[]>
          )[event.key];
          if (!move) return;
          event.preventDefault();
          const [x, y] = this.selection ?? [
            Math.floor(canvas.width / 2),
            Math.floor(canvas.height / 2),
          ];
          this.select(x + move[0]!, y + move[1]!, true);
        });
      });
      this.draw();
      this.querySelectorAll<HTMLElement>("[data-ready]").forEach((el) => {
        el.hidden = false;
      });
      this.querySelector<HTMLElement>("[data-load-status]")!.hidden = true;
      this.dataset.ready = "true";
    } catch (error) {
      if (controller.signal.aborted) return;
      this.querySelector<HTMLElement>("[data-load-status]")!.textContent =
        "The recorded arrays could not be loaded. Reload the page or download the data below.";
      this.dataset.ready = "error";
      console.error(error);
    } finally {
      if (this.controller === controller) this.controller = null;
    }
  }
  private value(name: string): string {
    return this.querySelector<HTMLInputElement | HTMLSelectElement>(
      `[data-control="${name}"]`,
    )!.value;
  }
  private draw() {
    const data = this.payload!;
    this.planes.clear();
    if (data.kind === "registration") {
      for (const state of ["initial", "observed", "final"]) {
        const array = data.arrays[`${state}_${this.value("view")}`]!;
        this.planes.set(state, {
          width: array.shape[1]!,
          height: array.shape[0]!,
          values: array.values,
        });
      }
    } else {
      const material = Number(this.value("material")),
        z = Number(this.value("z")),
        replicate = this.value("replicate");
      for (const state of ["reference", "initial", "update100", "final"]) {
        const name = state === "reference" ? state : `${state}_rep${replicate}`;
        this.planes.set(
          state,
          materialPlane(
            data.arrays[name]!,
            material,
            z,
            data.grid!.origin_mm[2]!,
            data.grid!.spacing_mm[2]!,
          ),
        );
      }
      this.planes.set(
        "error",
        differencePlane(
          this.planes.get("final")!,
          this.planes.get("reference")!,
        ),
      );
      this.querySelector<HTMLOutputElement>("[data-slice-label]")!.textContent =
        `z = ${z.toFixed(0)} mm`;
    }
    const styles = getComputedStyle(this);
    const colour = (name: string): RGB =>
      paletteColour(styles.getPropertyValue(name));
    const [paper, blue, rust] = [
      colour("--paper"),
      colour("--technical"),
      colour("--accent"),
    ];
    for (const canvas of this.querySelectorAll<HTMLCanvasElement>("canvas")) {
      const plane = this.planes.get(canvas.dataset.state!)!;
      canvas.width = plane.width;
      canvas.height = plane.height;
      const scale: RasterScale =
        data.kind === "registration"
          ? "counts"
          : canvas.dataset.state === "error"
            ? "error"
            : "fraction";
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Canvas is unavailable");
      context.imageSmoothingEnabled = false;
      context.putImageData(
        new ImageData(
          rasterPixels(plane, scale, paper!, blue!, rust!),
          plane.width,
          plane.height,
        ),
        0,
        0,
      );
      canvas.setAttribute(
        "aria-label",
        `${canvas.dataset.label}, ${data.kind === "reconstruction" ? `${this.value("material") === "0" ? "water" : "bone"}, replicate ${this.value("replicate")}, z ${this.value("z")} mm` : `${this.value("view") === "a180" ? "0" : "90"} degree view`}. Use arrow keys to inspect samples.`,
      );
    }
    this.querySelectorAll<HTMLElement>("[data-cursor]").forEach((el) => {
      el.hidden = true;
    });
    this.querySelector<HTMLOutputElement>("[data-inspection]")!.textContent =
      "Point to a sample, or focus a panel and use the arrow keys, to compare its recorded values.";
  }
  private select(col: number, row: number, announce: boolean) {
    const first = this.planes.values().next().value as RasterPlane;
    col = Math.max(0, Math.min(first.width - 1, col));
    row = Math.max(0, Math.min(first.height - 1, row));
    this.selection = [col, row];
    const data = this.payload!;
    const x =
      data.kind === "registration"
        ? -508 + col * 8
        : data.grid!.origin_mm[0]! + col * data.grid!.spacing_mm[0]!;
    const y =
      data.kind === "registration"
        ? -508 + row * 8
        : data.grid!.origin_mm[1]! + row * data.grid!.spacing_mm[1]!;
    const parts = [...this.planes].map(([name, plane]) => {
      const label = this.querySelector<HTMLCanvasElement>(
        `canvas[data-state="${name}"]`,
      )!.dataset.label;
      return `${label}: ${plane.values[row * plane.width + col]!.toPrecision(6)}`;
    });
    const output = this.querySelector<HTMLOutputElement>("[data-inspection]")!;
    output.setAttribute("aria-live", announce ? "polite" : "off");
    output.textContent = `${data.kind === "registration" ? "u" : "x"} = ${x.toFixed(2)} mm, ${data.kind === "registration" ? "v" : "y"} = ${y.toFixed(2)} mm. ${parts.join("; ")}. ${data.kind === "registration" ? "Counts." : "Volume fractions; error is recovered minus reference."}`;
    this.querySelectorAll<HTMLElement>("[data-cursor]").forEach((el) => {
      el.hidden = false;
      el.style.left = `${((col + 0.5) / first.width) * 100}%`;
      el.style.top = `${((first.height - row - 0.5) / first.height) * 100}%`;
    });
  }
}
