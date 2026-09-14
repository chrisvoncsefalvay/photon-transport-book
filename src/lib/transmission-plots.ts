import { type PlotPanel } from "./transmission-plot-data";
import {
  recordedSample as sample,
  recordedValueLabel as valueLabel,
  transmissionPlotDrawing,
  TRANSMISSION_LAYOUT as LAYOUT,
} from "./transmission-plot-renderer";

const NS = "http://www.w3.org/2000/svg";
function svgElement(
  tag: string,
  attributes: Record<string, string | number> = {},
  text?: string,
): SVGElement {
  const element = document.createElementNS(NS, tag);
  for (const [name, value] of Object.entries(attributes))
    element.setAttribute(name, String(value));
  if (text !== undefined) element.textContent = text;
  return element;
}
export function mountTransmissionPlots(): void {
  for (const host of document.querySelectorAll<HTMLElement>(
    "[data-plot-panel]",
  )) {
    if (host.dataset.mounted) continue;
    const record = host.dataset.record;
    const canvas = host.querySelector<HTMLElement>(
      ".transmission-plot__canvas",
    );
    const slider = host.querySelector<HTMLInputElement>('input[type="range"]');
    const readout = host.querySelector<HTMLOutputElement>("output");
    const note = host.querySelector<HTMLElement>(".transmission-plot__note");
    const scrubber = host.querySelector<HTMLElement>(
      ".transmission-plot__scrubber",
    );
    if (!record || !canvas || !slider || !readout || !note || !scrubber)
      continue;
    const panel: PlotPanel = JSON.parse(record);
    host.dataset.mounted = "true";
    let selected = 0;
    let cursor: SVGElement;
    let currentX: (value: number) => number = () => 0;
    const select = (index: number) => {
      index = Math.max(
        0,
        Math.min(
          panel.x.length - 1,
          Number.isFinite(index) ? Math.round(index) : 0,
        ),
      );
      selected = index;
      slider.value = String(index);
      const label = [
        `${panel.xLabel} = ${valueLabel(sample(panel.x, index))}`,
        ...panel.series.map(
          (line) => `${line.label} = ${valueLabel(sample(line.values, index))}`,
        ),
      ].join("\n");
      readout.textContent = label;
      slider.setAttribute("aria-valuetext", label.replaceAll("\n", ". "));
      cursor?.setAttribute("x1", String(currentX(sample(panel.x, index))));
      cursor?.setAttribute("x2", String(currentX(sample(panel.x, index))));
    };
    const draw = (): void => {
      const drawing = transmissionPlotDrawing(
        panel,
        canvas.getBoundingClientRect().width || LAYOUT.minimumWidth,
      );
      const { width, x } = drawing;
      const { left, top, bottom } = LAYOUT;
      currentX = x;
      note.textContent = drawing.note;
      const svg = svgElement("svg", drawing.attributes);
      for (const node of drawing.nodes)
        svg.append(svgElement(node.tag, node.attributes, node.text));
      const add = (tag: string, attrs: Record<string, string | number>) => {
        const node = svgElement(tag, attrs);
        svg.append(node);
        return node;
      };
      cursor = add("line", {
        x1: left,
        x2: left,
        y1: top,
        y2: drawing.hasZeros ? LAYOUT.zeroRail + LAYOUT.textBaseline : bottom,
        class: "transmission-plot__cursor",
      });
      svg.addEventListener("pointermove", (event) => {
        const pointer = event as PointerEvent;
        const position =
          ((pointer.clientX - svg.getBoundingClientRect().left) * width) /
          svg.getBoundingClientRect().width;
        let nearest = 0,
          distance = Infinity;
        panel.x.forEach((value, index) => {
          const delta = Math.abs(x(value) - position);
          if (delta < distance) {
            nearest = index;
            distance = delta;
          }
        });
        select(nearest);
      });
      canvas.replaceChildren(svg);
      select(selected);
    };
    slider.addEventListener("input", () => select(Number(slider.value)));
    scrubber.hidden = false;
    new ResizeObserver(draw).observe(canvas);
    draw();
  }
}
