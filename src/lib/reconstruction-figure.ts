import {
  colour,
  decodeScalars,
  dimensions,
  lineValues,
  physicalPoint,
  planes,
  planeValues,
  pointOnPlane,
  recordedBuffer,
  validateGrid,
  voxelOffset,
  type Axis,
  type Point,
  type ReconstructionCase,
  type RecordedScalar,
} from "./reconstruction-data";
import type { ReconstructionRenderer } from "./reconstruction-three";

const svgNS = "http://www.w3.org/2000/svg";
const displayNumber = (value: number): string =>
  String(Number(value.toPrecision(4)));
function svgElement(
  name: string,
  attributes: Record<string, string>,
  text?: string,
): SVGElement {
  const node = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attributes))
    node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

class ReconstructionFigure extends HTMLElement {
  private abort: AbortController | undefined;
  private observer: IntersectionObserver | undefined;
  private renderer: ReconstructionRenderer | undefined;
  private record: ReconstructionCase | undefined;
  private fields = new Map<string, Float32Array>();
  private point: Point = [0, 0, 0];
  private material = "";
  private comparison = "reconstructed";
  private loading = false;
  private active = false;
  private inView = true;
  private generation = 0;
  private selectionGeneration = 0;
  private inspecting = false;

  private required<T extends Element>(selector: string): T {
    const node = this.querySelector<T>(selector);
    if (!node)
      throw new Error(`Missing reconstruction display element: ${selector}`);
    return node;
  }

  connectedCallback(): void {
    if (this.abort) return;
    this.abort = new AbortController();
    this.generation++;
    this.loading = false;
    this.inspecting = false;
    this.dataset["illumination"] = "true";
    const sceneControls = this.querySelector<HTMLElement>(
      ".reconstruction-scene-controls",
    );
    if (sceneControls) sceneControls.hidden = true;
    const framing = this.querySelector<HTMLSelectElement>(
      "[data-reconstruction-framing]",
    );
    if (framing) framing.value = "anatomy";
    this.querySelectorAll<HTMLInputElement>(
      "[data-reconstruction-light], [data-reconstruction-rig]",
    ).forEach((input) => (input.checked = true));
    const button = this.required<HTMLButtonElement>(
      "[data-reconstruction-start]",
    );
    button.hidden = false;
    button.disabled = false;
    this.required<HTMLElement>(".reconstruction-inspector").hidden = true;
    this.required<HTMLSelectElement>("[data-reconstruction-material]").value =
      this.dataset["material"] ?? "";
    this.required<HTMLSelectElement>("[data-reconstruction-render]").value =
      "surface";
    this.required<HTMLSelectElement>("[data-reconstruction-comparison]").value =
      "reconstructed";
    this.required<HTMLSelectElement>(
      "[data-reconstruction-profile-axis]",
    ).value = "0";
    this.required<HTMLInputElement>("[data-reconstruction-clip]").value = "1";
    this.required<HTMLInputElement>("[data-reconstruction-opacity]").value =
      "0.35";
    this.comparison = "reconstructed";
    this.enable3DControls(true);
    this.showPoster();
    const { signal } = this.abort;
    this.required<HTMLButtonElement>(
      "[data-reconstruction-start]",
    ).addEventListener("click", () => void this.start(), { signal });
    this.observer = new IntersectionObserver(([entry]) => {
      this.inView = Boolean(entry?.isIntersecting);
      this.renderer?.setVisible(this.inView && !document.hidden);
      if (
        this.inView &&
        !document.hidden &&
        this.dataset["autoThree"] === "true"
      )
        void this.start(false);
    });
    this.observer.observe(this);
    document.addEventListener(
      "visibilitychange",
      () => this.renderer?.setVisible(this.inView && !document.hidden),
      { signal },
    );
  }

  disconnectedCallback(): void {
    this.generation++;
    this.abort?.abort();
    this.abort = undefined;
    this.observer?.disconnect();
    this.observer = undefined;
    this.renderer?.dispose();
    this.renderer = undefined;
    this.fields.clear();
    this.active = false;
  }

  private status(message: string, announceOnly = false): void {
    const status = this.required<HTMLElement>("[data-reconstruction-status]");
    status.textContent = message;
    status.classList.toggle(
      "sr-only",
      announceOnly && this.dataset["quietStatus"] === "true",
    );
  }

  private showPoster(): void {
    this.required<HTMLCanvasElement>("[data-reconstruction-three]").hidden =
      true;
    this.required<HTMLImageElement>(".reconstruction-poster").hidden = false;
    this.required<HTMLElement>(".reconstruction-poster-labels").hidden = false;
    this.required<HTMLElement>(".reconstruction-live-labels").hidden = true;
  }

  private fail3D(message: string): void {
    this.renderer?.dispose();
    this.renderer = undefined;
    this.showPoster();
    this.enable3DControls(false);
    this.status(message);
  }

  private enable3DControls(enabled: boolean): void {
    this.querySelectorAll<
      HTMLInputElement | HTMLSelectElement | HTMLButtonElement
    >(
      "[data-reconstruction-render], [data-reconstruction-clip], [data-reconstruction-opacity], [data-reconstruction-action], [data-reconstruction-framing], [data-reconstruction-light], [data-reconstruction-rig]",
    ).forEach((control) => {
      control.disabled = !enabled;
    });
  }

  private async start(inspect = true): Promise<void> {
    if (inspect) {
      this.inspecting = true;
      if (this.active) {
        this.required<HTMLElement>(".reconstruction-inspector").hidden = false;
        this.required<HTMLButtonElement>("[data-reconstruction-start]").hidden =
          true;
      }
    }
    if (this.active || this.loading) return;
    this.loading = true;
    const generation = this.generation;
    const button = this.required<HTMLButtonElement>(
      "[data-reconstruction-start]",
    );
    button.disabled = true;
    this.status("Loading recorded fields and surfaces…");
    try {
      this.record = JSON.parse(
        this.required<HTMLScriptElement>("[data-reconstruction-record]")
          .textContent ?? "",
      ) as ReconstructionCase;
      validateGrid(this.record.grid);
      this.material =
        this.dataset["material"] ?? this.record.scalars[0]!.material;
      this.point = [...this.record.acquisition.display.slice_indices_xyz];
      voxelOffset(this.record.grid, this.point);
      const base = this.dataset["base"]!;
      const values = await Promise.all(
        this.record.scalars.map(async (scalar) => {
          if (scalar.dtype !== "float32-le")
            throw new Error("Unsupported recorded scalar encoding.");
          return [
            scalar.id,
            decodeScalars(
              await recordedBuffer(base, scalar),
              this.record!.grid,
            ),
          ] as const;
        }),
      );
      if (!this.isConnected || generation !== this.generation) return;
      this.fields = new Map(values);
      this.required<HTMLElement>(".reconstruction-inspector").hidden =
        !this.inspecting;
      this.bindControls();
      this.drawSlices();
      try {
        const { ReconstructionRenderer } =
          await import("./reconstruction-three");
        if (!this.isConnected || generation !== this.generation) return;
        const previousCanvas = this.required<HTMLCanvasElement>(
          "[data-reconstruction-three]",
        );
        // A previously lost WebGL context belongs to its canvas. Reconnection
        // gets a fresh display surface rather than reusing that lost context.
        const canvas = previousCanvas.cloneNode(false) as HTMLCanvasElement;
        previousCanvas.replaceWith(canvas);
        const renderer = new ReconstructionRenderer(
          canvas,
          this.record,
          base,
          this.fields,
          (labels) => {
            const container = this.required<HTMLElement>(
              ".reconstruction-live-labels",
            );
            container.replaceChildren(
              ...labels.map((label) => {
                const span = document.createElement("span");
                span.textContent = label;
                return span;
              }),
            );
          },
          this.dataset["projectionBase"],
        );
        this.renderer = renderer;
        await renderer.select(this.material);
        if (!this.isConnected || generation !== this.generation) {
          renderer.dispose();
          return;
        }
        canvas.hidden = false;
        const sceneControls = this.querySelector<HTMLElement>(
          ".reconstruction-scene-controls",
        );
        if (sceneControls) sceneControls.hidden = false;
        this.required<HTMLImageElement>(".reconstruction-poster").hidden = true;
        this.required<HTMLElement>(".reconstruction-poster-labels").hidden =
          true;
        this.required<HTMLElement>(".reconstruction-live-labels").hidden =
          false;
        renderer.setVisible(this.inView && !document.hidden);
        canvas.addEventListener(
          "webglcontextlost",
          (event) => {
            event.preventDefault();
            this.fail3D(
              "3D display interrupted. The static plate, slices and numerical profiles remain available.",
            );
          },
          { signal: this.abort!.signal },
        );
        this.status(
          this.record.scalars.some((s) => s.role === "reference")
            ? "Recorded fields loaded. Cameras, clipping and display ranges are shared."
            : "Recorded equivalent-basis fields loaded. No independent material reference is available.",
          true,
        );
      } catch {
        if (!this.isConnected || generation !== this.generation) return;
        this.fail3D(
          "3D is unavailable here. Inspect the recorded slices and profiles below the static plate.",
        );
      }
      this.active = true;
      this.updateDisplayNote();
      button.hidden = this.inspecting;
      button.disabled = false;
    } catch (error) {
      if (!this.isConnected || generation !== this.generation) return;
      this.status(
        error instanceof Error
          ? error.message
          : "The recorded figure could not be loaded.",
      );
      button.disabled = false;
    } finally {
      if (generation === this.generation) this.loading = false;
    }
  }

  private bindControls(): void {
    const { signal } = this.abort!;
    this.querySelector<HTMLSelectElement>(
      "[data-reconstruction-framing]",
    )?.addEventListener(
      "change",
      (event) => {
        this.renderer?.setOptions({
          framing: (event.currentTarget as HTMLSelectElement).value as
            "anatomy" | "acquisition",
        });
      },
      { signal },
    );
    this.querySelector<HTMLInputElement>(
      "[data-reconstruction-light]",
    )?.addEventListener(
      "change",
      (event) => {
        const enabled = (event.currentTarget as HTMLInputElement).checked;
        this.dataset["illumination"] = String(enabled);
        this.renderer?.setOptions({ illumination: enabled });
      },
      { signal },
    );
    this.querySelector<HTMLInputElement>(
      "[data-reconstruction-rig]",
    )?.addEventListener(
      "change",
      (event) => {
        this.renderer?.setOptions({
          rig: (event.currentTarget as HTMLInputElement).checked,
        });
      },
      { signal },
    );
    this.required<HTMLSelectElement>(
      "[data-reconstruction-material]",
    ).addEventListener(
      "change",
      (event) => {
        this.material = (event.currentTarget as HTMLSelectElement).value;
        this.drawSlices();
        this.updateDisplayNote();
        const renderer = this.renderer;
        const generation = this.generation;
        const selection = ++this.selectionGeneration;
        void renderer?.select(this.material).catch(() => {
          if (
            generation !== this.generation ||
            renderer !== this.renderer ||
            selection !== this.selectionGeneration
          )
            return;
          this.fail3D(
            "The selected 3D field could not be loaded. Its recorded slices remain available, and the static plate retains its original labels.",
          );
        });
      },
      { signal },
    );
    this.required<HTMLSelectElement>(
      "[data-reconstruction-render]",
    ).addEventListener(
      "change",
      (event) => {
        this.renderer?.setOptions({
          mode: (event.currentTarget as HTMLSelectElement).value as
            "surface" | "volume",
        });
        this.updateDisplayNote();
      },
      { signal },
    );
    this.required<HTMLInputElement>(
      "[data-reconstruction-clip]",
    ).addEventListener(
      "input",
      (event) =>
        this.renderer?.setOptions({
          clip: Number((event.currentTarget as HTMLInputElement).value),
        }),
      { signal },
    );
    this.required<HTMLInputElement>(
      "[data-reconstruction-opacity]",
    ).addEventListener(
      "input",
      (event) =>
        this.renderer?.setOptions({
          opacity: Number((event.currentTarget as HTMLInputElement).value),
        }),
      { signal },
    );
    this.querySelectorAll<HTMLButtonElement>(
      "[data-reconstruction-action]",
    ).forEach((button) =>
      button.addEventListener(
        "click",
        () => {
          const action = button.dataset["reconstructionAction"];
          if (action === "reset") this.renderer?.reset();
          else this.renderer?.setZoom(action === "zoom-in" ? 0.85 : 1.15);
        },
        { signal },
      ),
    );
    this.required<HTMLSelectElement>(
      "[data-reconstruction-comparison]",
    ).addEventListener(
      "change",
      (event) => {
        this.comparison = (event.currentTarget as HTMLSelectElement).value;
        this.renderer?.setOptions({
          focus:
            this.comparison === "reference" ? "reference" : "reconstructed",
        });
        this.drawSlices();
      },
      { signal },
    );
    this.querySelectorAll<HTMLInputElement>(
      "[data-reconstruction-coordinate]",
    ).forEach((input) =>
      input.addEventListener(
        "input",
        () => {
          this.point[
            Number(input.dataset["reconstructionCoordinate"]) as Axis
          ] = Number(input.value);
          this.drawSlices();
        },
        { signal },
      ),
    );
    this.required<HTMLSelectElement>(
      "[data-reconstruction-profile-axis]",
    ).addEventListener("change", () => this.drawProfile(), { signal });
    this.querySelectorAll<HTMLCanvasElement>(
      "[data-reconstruction-slice]",
    ).forEach((canvas) => {
      const plane = planes[Number(canvas.dataset["reconstructionSlice"])]!;
      canvas.addEventListener(
        "click",
        (event) => {
          const bounds = canvas.getBoundingClientRect();
          this.point = pointOnPlane(
            this.record!.grid,
            this.point,
            plane,
            (event.clientX - bounds.left) / bounds.width,
            (event.clientY - bounds.top) / bounds.height,
          );
          this.drawSlices();
        },
        { signal },
      );
      canvas.addEventListener(
        "keydown",
        (event) => {
          const moves: Record<string, [Axis, number]> = {
            ArrowLeft: [plane.horizontal, -1],
            ArrowRight: [plane.horizontal, 1],
            ArrowUp: [plane.vertical, 1],
            ArrowDown: [plane.vertical, -1],
          };
          const move = moves[event.key];
          if (!move) return;
          event.preventDefault();
          this.point[move[0]] = Math.max(
            0,
            Math.min(
              dimensions(this.record!.grid)[move[0]] - 1,
              this.point[move[0]] + move[1],
            ),
          );
          this.drawSlices();
        },
        { signal },
      );
    });
  }

  private selected(role = "reconstructed"): RecordedScalar | undefined {
    return this.record?.scalars.find(
      (item) => item.material === this.material && item.role === role,
    );
  }

  private updateDisplayNote(): void {
    const scalar = this.selected();
    if (!scalar) return;
    const mesh = this.record!.meshes.find(
      (item) => item.scalar_id === scalar.id,
    );
    const selector = this.required<HTMLSelectElement>(
      "[data-reconstruction-render]",
    );
    const surfaceOption = selector.querySelector<HTMLOptionElement>(
      'option[value="surface"]',
    );
    if (surfaceOption) surfaceOption.disabled = !mesh;
    if (!mesh && selector.value === "surface") {
      selector.value = "volume";
      this.renderer?.setOptions({ mode: "volume" });
    }
    const mode = selector.value;
    this.required<HTMLElement>(
      "[data-reconstruction-display-note]",
    ).textContent =
      mode === "surface" && mesh
        ? `Surface threshold: ${mesh.threshold} ${scalar.units}.`
        : `Volume range: ${displayNumber(scalar.window[0])} to ${displayNumber(scalar.window[1])} ${scalar.units}.`;
  }

  private drawSlices(): void {
    const record = this.record!;
    const scalar = this.selected(
      this.comparison === "reference" ? "reference" : "reconstructed",
    );
    if (!scalar) throw new Error("Requested reference is unavailable.");
    const values = this.fields.get(scalar.id)!;
    const reference =
      this.comparison === "error"
        ? this.fields.get(this.selected("reference")!.id)
        : undefined;
    const error = this.comparison === "error";
    const extent = scalar.window[1] - scalar.window[0];
    const window: [number, number] = error ? [-extent, extent] : scalar.window;
    const size = dimensions(record.grid);
    this.querySelectorAll<HTMLCanvasElement>(
      "[data-reconstruction-slice]",
    ).forEach((canvas) => {
      const plane = planes[Number(canvas.dataset["reconstructionSlice"])]!;
      const slice = planeValues(
        values,
        record.grid,
        this.point,
        plane,
        reference,
      );
      canvas.width = size[plane.horizontal];
      canvas.height = size[plane.vertical];
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Slice drawing is unavailable.");
      const pixels = context.createImageData(canvas.width, canvas.height);
      slice.forEach((value, index) => {
        pixels.data.set([...colour(value, window, error), 255], index * 4);
      });
      context.putImageData(pixels, 0, 0);
      const area = canvas.parentElement!;
      area.style.aspectRatio = String(
        (canvas.width * record.grid.spacing_mm[plane.horizontal]) /
          (canvas.height * record.grid.spacing_mm[plane.vertical]),
      );
      area.querySelector<HTMLElement>(
        ".reconstruction-crosshair-x",
      )!.style.left =
        `${(100 * (this.point[plane.horizontal] + 0.5)) / canvas.width}%`;
      area.querySelector<HTMLElement>(
        ".reconstruction-crosshair-y",
      )!.style.top =
        `${(100 * (canvas.height - 0.5 - this.point[plane.vertical])) / canvas.height}%`;
      canvas.setAttribute(
        "aria-label",
        `${plane.name}, ${scalar.label}${error ? " minus assigned reference" : ""}. Grid index ${this.point[plane.fixed]}. Click or use arrow keys to move the linked crosshair.`,
      );
    });
    this.querySelectorAll<HTMLInputElement>(
      "[data-reconstruction-coordinate]",
    ).forEach((input) => {
      input.value = String(
        this.point[Number(input.dataset["reconstructionCoordinate"]) as Axis],
      );
    });
    const offset = voxelOffset(record.grid, this.point);
    const value = values[offset]! - (reference?.[offset] ?? 0);
    this.required<HTMLOutputElement>(
      "[data-reconstruction-position]",
    ).textContent = `Object XYZ (${physicalPoint(record.grid, this.point)
      .map((n) => n.toFixed(2))
      .join(", ")}) mm · value ${value.toFixed(4)}`;
    this.required<HTMLElement>("[data-reconstruction-window]").textContent =
      `${error ? "Blue → zero → rust" : "Black → white"}: ${displayNumber(window[0])} to ${displayNumber(window[1])} ${scalar.units}${error ? " (reconstructed − reference)" : ""}`;
    this.drawProfile();
  }

  private drawProfile(): void {
    const record = this.record!;
    const axis = Number(
      this.required<HTMLSelectElement>("[data-reconstruction-profile-axis]")
        .value,
    ) as Axis;
    const records = [this.selected("reference"), this.selected()].filter(
      (s): s is RecordedScalar => Boolean(s),
    );
    const svg = this.required<SVGSVGElement>("[data-reconstruction-profile]");
    const curves = records.map((scalar) => ({
      scalar,
      points: lineValues(
        this.fields.get(scalar.id)!,
        record.grid,
        this.point,
        axis,
      ),
    }));
    const maximum = Math.max(
      ...curves.flatMap((curve) => curve.points.map((p) => p.value)),
      ...records.map((s) => s.window[1]),
    );
    const distance =
      (dimensions(record.grid)[axis] - 1) * record.grid.spacing_mm[axis];
    const x = (mm: number): number =>
      45 + (mm / Math.max(distance, 1e-12)) * 655;
    const y = (value: number): number =>
      136 - (value / Math.max(maximum, 1e-12)) * 124;
    const nodes: SVGElement[] = [];
    for (const t of [0, 0.5, 1]) {
      nodes.push(
        svgElement("line", {
          x1: "45",
          x2: "700",
          y1: String(y(maximum * t)),
          y2: String(y(maximum * t)),
          stroke: "var(--rule)",
        }),
      );
      nodes.push(
        svgElement(
          "text",
          {
            x: "38",
            y: String(y(maximum * t) + 4),
            "text-anchor": "end",
            fill: "var(--ink-muted)",
          },
          (maximum * t).toFixed(2),
        ),
      );
      nodes.push(
        svgElement(
          "text",
          {
            x: String(x(distance * t)),
            y: "156",
            "text-anchor": "middle",
            fill: "var(--ink-muted)",
          },
          (distance * t).toFixed(1),
        ),
      );
    }
    for (const { scalar, points } of curves)
      nodes.push(
        svgElement("path", {
          d: points
            .map(
              (point, i) =>
                `${i ? "L" : "M"}${x(point.distance_mm).toFixed(3)},${y(point.value).toFixed(3)}`,
            )
            .join(" "),
          fill: "none",
          stroke:
            scalar.role === "reference" ? "var(--technical)" : "var(--accent)",
          "stroke-width": "1.6",
          "vector-effect": "non-scaling-stroke",
        }),
      );
    nodes.push(
      svgElement("line", {
        x1: String(x(this.point[axis] * record.grid.spacing_mm[axis])),
        x2: String(x(this.point[axis] * record.grid.spacing_mm[axis])),
        y1: "12",
        y2: "136",
        stroke: "var(--ink-muted)",
        "stroke-dasharray": "3 4",
      }),
    );
    svg.replaceChildren(...nodes);
  }
}

export function mountReconstructionFigures(): void {
  if (!customElements.get("dpt-reconstruction"))
    customElements.define("dpt-reconstruction", ReconstructionFigure);
}
