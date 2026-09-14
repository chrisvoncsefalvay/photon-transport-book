import {
  plotCoordinate,
  validatePlotPanel,
  type PlotPanel,
} from "./transmission-plot-data";

// SVG user units: reserve room for exponent ticks, an explicit zero rail,
// and the horizontal-axis title. CSS controls the responsive outer width.
export const TRANSMISSION_LAYOUT = {
  minimumWidth: 220,
  left: 47,
  right: 12,
  top: 12,
  bottom: 192,
  height: 274,
  zeroRail: 232,
  axisTitle: 268,
  tickGap: 7,
  textBaseline: 4,
  xTickGap: 17,
  intervals: 4,
};

export function recordedSample(values: number[], index: number): number {
  const value = values[index];
  if (value === undefined)
    throw new Error(`Missing recorded plot sample ${index}`);
  return value;
}

const superscript = (value: number) =>
  String(value).replace(/[-0-9]/g, (digit) =>
    "⁻⁰¹²³⁴⁵⁶⁷⁸⁹".charAt("-0123456789".indexOf(digit)),
  );
export const recordedValueLabel = (value: number) =>
  value === 0 ? "0" : Number(value.toPrecision(9)).toString();

type SvgAttributes = Record<string, string | number>;
export interface PlotSvgNode {
  tag: "line" | "text" | "path" | "circle";
  attributes: SvgAttributes;
  text?: string;
}

/** Shared static/client geometry. These transformations never alter recorded values. */
export function transmissionPlotDrawing(
  panel: PlotPanel,
  requestedWidth = 320,
) {
  validatePlotPanel(panel);
  if (!Number.isFinite(requestedWidth) || requestedWidth <= 0)
    throw new Error("Transmission plot width must be positive and finite");
  const zeroNotes =
    panel.yScale === "log"
      ? panel.series.flatMap((line) => {
          const first = line.values.indexOf(0);
          return first < 0
            ? []
            : [
                `${line.label}: ${line.values.filter((value) => value === 0).length} stored zeros, first sampled zero at ${panel.xLabel} = ${recordedValueLabel(recordedSample(panel.x, first))}.`,
              ];
        })
      : [];
  const note = [
    panel.note,
    ...zeroNotes,
    ...(zeroNotes.length
      ? [
          "Zeros appear on a separate labelled rail, outside the logarithmic scale.",
        ]
      : []),
  ]
    .filter(Boolean)
    .join(" ");
  const allValues = panel.series
    .flatMap((series) => series.values)
    .filter((value) => panel.yScale === "linear" || value > 0);
  let yDomain: [number, number] = [
    Math.min(...allValues),
    Math.max(...allValues),
  ];
  if (panel.yScale === "log")
    yDomain = [
      10 ** Math.floor(Math.log10(yDomain[0])),
      10 ** Math.ceil(Math.log10(yDomain[1])),
    ];
  if (yDomain[0] === yDomain[1])
    yDomain =
      panel.yScale === "log"
        ? [yDomain[0] / 10, yDomain[1] * 10]
        : [Math.min(0, yDomain[0]), Math.max(1, yDomain[1])];
  const xDomain: [number, number] = [
    recordedSample(panel.x, 0),
    recordedSample(panel.x, panel.x.length - 1),
  ];
  const width = Math.max(TRANSMISSION_LAYOUT.minimumWidth, requestedWidth);
  const { left, top, bottom } = TRANSMISSION_LAYOUT;
  const right = width - TRANSMISSION_LAYOUT.right;
  const x = (value: number) =>
    plotCoordinate(value, xDomain, [left, right], panel.xScale);
  const y = (value: number) =>
    plotCoordinate(value, yDomain, [bottom, top], panel.yScale);
  const attributes: SvgAttributes = {
    viewBox: `0 0 ${width} ${TRANSMISSION_LAYOUT.height}`,
    height: TRANSMISSION_LAYOUT.height,
    role: "img",
    "aria-label": `${panel.title}. ${panel.x.length} recorded samples. ${panel.xScale} horizontal scale. ${panel.yScale} vertical scale. Exact values are available in the execution data.`,
  };
  const nodes: PlotSvgNode[] = [];
  const add = (
    tag: PlotSvgNode["tag"],
    attrs: Record<string, string | number>,
    text?: string,
  ) => {
    nodes.push({
      tag,
      attributes: attrs,
      ...(text === undefined ? {} : { text }),
    });
  };
  const ticks = (
    domain: [number, number],
    scale: "linear" | "log",
  ): number[] => {
    if (scale === "linear")
      return Array.from(
        { length: TRANSMISSION_LAYOUT.intervals + 1 },
        (_, i) =>
          domain[0] +
          ((domain[1] - domain[0]) * i) / TRANSMISSION_LAYOUT.intervals,
      );
    const start = Math.ceil(Math.log10(domain[0])),
      end = Math.floor(Math.log10(domain[1]));
    const step = Math.max(
      1,
      Math.ceil((end - start) / TRANSMISSION_LAYOUT.intervals),
    );
    return Array.from(
      { length: Math.floor((end - start) / step) + 1 },
      (_, i) => 10 ** (start + i * step),
    );
  };
  const tickLabel = (value: number, scale: "linear" | "log") =>
    scale === "log"
      ? `10${superscript(Math.round(Math.log10(value)))}`
      : String(Number(value.toPrecision(3)));
  for (const tick of ticks(yDomain, panel.yScale)) {
    add("line", {
      x1: left,
      x2: right,
      y1: y(tick),
      y2: y(tick),
      class: "transmission-plot__gridline",
    });
    add(
      "text",
      {
        x: left - TRANSMISSION_LAYOUT.tickGap,
        y: y(tick) + TRANSMISSION_LAYOUT.textBaseline,
        "text-anchor": "end",
      },
      tickLabel(tick, panel.yScale),
    );
  }
  for (const tick of ticks(xDomain, panel.xScale)) {
    add("line", {
      x1: x(tick),
      x2: x(tick),
      y1: top,
      y2: bottom,
      class: "transmission-plot__gridline",
    });
    add(
      "text",
      {
        x: x(tick),
        y: bottom + TRANSMISSION_LAYOUT.xTickGap,
        "text-anchor": "middle",
      },
      tickLabel(tick, panel.xScale),
    );
  }
  add("line", {
    x1: left,
    x2: right,
    y1: bottom,
    y2: bottom,
    class: "transmission-plot__axis",
  });
  if (zeroNotes.length) {
    add("line", {
      x1: left,
      x2: right,
      y1: TRANSMISSION_LAYOUT.zeroRail,
      y2: TRANSMISSION_LAYOUT.zeroRail,
      class: "transmission-plot__gridline",
    });
    add(
      "text",
      {
        x: left - TRANSMISSION_LAYOUT.tickGap,
        y: TRANSMISSION_LAYOUT.zeroRail + TRANSMISSION_LAYOUT.textBaseline,
        "text-anchor": "end",
      },
      "zero",
    );
  }
  for (const series of panel.series) {
    let path = "",
      connected = false;
    for (const [index, value] of series.values.entries()) {
      if (panel.yScale === "log" && value === 0) {
        connected = false;
        continue;
      }
      path += `${connected ? "L" : "M"}${x(recordedSample(panel.x, index))} ${y(value)} `;
      connected = true;
    }
    add("path", {
      d: path,
      class: "transmission-plot__line",
      "data-style": series.style,
    });
    for (const [index, value] of series.values.entries()) {
      add("circle", {
        cx: x(recordedSample(panel.x, index)),
        cy:
          panel.yScale === "log" && value === 0
            ? TRANSMISSION_LAYOUT.zeroRail
            : y(value),
        r: series.style === "reference" ? 2.2 : 1.6,
        class: "transmission-plot__point",
        "data-style": series.style,
      });
    }
  }
  add(
    "text",
    {
      x: (left + right) / 2,
      y: TRANSMISSION_LAYOUT.axisTitle,
      "text-anchor": "middle",
    },
    panel.xLabel,
  );

  return { attributes, nodes, note, width, x, hasZeros: zeroNotes.length > 0 };
}

const escapeSvg = (value: string | number) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
const attributesText = (attributes: SvgAttributes) =>
  Object.entries(attributes)
    .map(([name, value]) => ` ${name}="${escapeSvg(value)}"`)
    .join("");

/** No browser DOM is needed to emit every recorded point into static HTML. */
export function transmissionPlotSvg(
  drawing: ReturnType<typeof transmissionPlotDrawing>,
): string {
  return `<svg xmlns="http://www.w3.org/2000/svg"${attributesText(drawing.attributes)}>${drawing.nodes
    .map(
      (node) =>
        `<${node.tag}${attributesText(node.attributes)}>${node.text === undefined ? "" : escapeSvg(node.text)}</${node.tag}>`,
    )
    .join("")}</svg>`;
}
