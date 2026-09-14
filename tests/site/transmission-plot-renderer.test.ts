import { describe, expect, it } from "vitest";
import recorded from "../../public/generated/transmission-contract/validation.json";
import {
  transmissionPanels,
  type PlotPanel,
} from "../../src/lib/transmission-plot-data";
import {
  transmissionPlotDrawing,
  transmissionPlotSvg,
  TRANSMISSION_LAYOUT,
} from "../../src/lib/transmission-plot-renderer";

describe("static recorded transmission plots", () => {
  it("renders all seven actual panels and every recorded point without a browser DOM", () => {
    const panels = (["sweep", "weak", "tail"] as const).flatMap((kind) =>
      transmissionPanels(recorded, kind),
    );
    expect(panels).toHaveLength(7);
    for (const panel of panels) {
      const before = JSON.stringify(panel);
      const drawing = transmissionPlotDrawing(panel);
      const svg = transmissionPlotSvg(drawing);
      const points = drawing.nodes.filter((node) => node.tag === "circle");
      expect(points).toHaveLength(panel.x.length * panel.series.length);
      expect(svg.match(/<circle /g)).toHaveLength(points.length);
      expect(svg).toContain('role="img"');
      expect(svg).not.toMatch(/NaN|Infinity|undefined/);
      expect(JSON.stringify(panel)).toBe(before);
      const zeros =
        panel.yScale === "log"
          ? panel.series
              .flatMap((series) => series.values)
              .filter((value) => value === 0).length
          : 0;
      expect(
        points.filter(
          (node) => node.attributes.cy === TRANSMISSION_LAYOUT.zeroRail,
        ),
      ).toHaveLength(zeros);
      if (zeros > 0) {
        expect(svg).toContain(">zero</text>");
        expect(drawing.note).toContain("outside the logarithmic scale");
      }
    }
  });

  it("breaks the line at a stored zero and restarts at the next positive sample", () => {
    const panel: PlotPanel = {
      title: "Recorded gap",
      xLabel: "L",
      x: [0, 1, 2, 3, 4],
      xScale: "linear",
      yScale: "log",
      note: "",
      series: [
        { label: "Values", style: "primary", values: [1, 0.1, 0, 0.01, 0.001] },
      ],
    };
    const drawing = transmissionPlotDrawing(panel, 320);
    const path = drawing.nodes.find((node) => node.tag === "path")!;
    expect(String(path.attributes.d).match(/[ML]/g)).toEqual([
      "M",
      "L",
      "M",
      "L",
    ]);
    const points = drawing.nodes.filter((node) => node.tag === "circle");
    expect(points.map((point) => point.attributes.cx)).toEqual([
      47, 112.25, 177.5, 242.75, 308,
    ]);
    expect(points.map((point) => point.attributes.cy)).toEqual([
      12, 72, 232, 132, 192,
    ]);
    expect(drawing.note).toContain(
      "1 stored zeros, first sampled zero at L = 2",
    );
  });

  it("changes only horizontal placement on resize and keeps all zero classifications", () => {
    const panel = transmissionPanels(recorded, "tail")[0]!;
    const narrow = transmissionPlotDrawing(panel, 220);
    const wide = transmissionPlotDrawing(panel, 640);
    const points = (drawing: ReturnType<typeof transmissionPlotDrawing>) =>
      drawing.nodes.filter((node) => node.tag === "circle");
    expect(points(narrow).map((node) => node.attributes.cy)).toEqual(
      points(wide).map((node) => node.attributes.cy),
    );
    expect(narrow.x(panel.x[0]!)).toBe(47);
    expect(wide.x(panel.x[0]!)).toBe(47);
    expect(narrow.x(panel.x.at(-1)!)).toBe(208);
    expect(wide.x(panel.x.at(-1)!)).toBe(628);
    expect(() => transmissionPlotDrawing(panel, NaN)).toThrow(
      "positive and finite",
    );
  });

  it("escapes recorded titles and axis labels in static SVG output", () => {
    const source = transmissionPanels(recorded, "sweep")[0]!;
    const drawing = transmissionPlotDrawing({
      ...source,
      title: 'T "quoted" <script>alert(1)</script> & counts',
      xLabel: '<img src=x onerror="alert(1)">',
    });
    const svg = transmissionPlotSvg(drawing);
    expect(svg).not.toContain("<script>");
    expect(svg).not.toContain("<img");
    expect(svg).toContain("&quot;quoted&quot;");
    expect(svg).toContain("&lt;script&gt;");
    expect(svg).toContain("&amp; counts");
  });
});
