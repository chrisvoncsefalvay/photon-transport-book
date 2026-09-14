/** Presentation records only: every ordinate comes from the recorded CUDA run or its oracle. */
export interface PlotSeries {
  label: string;
  values: number[];
  style: "primary" | "reference" | "diagnostic";
}
export interface PlotPanel {
  title: string;
  xLabel: string;
  x: number[];
  xScale: "linear" | "log";
  yScale: "linear" | "log";
  series: PlotSeries[];
  note: string;
}
interface Sweep {
  optical_depths: number[];
  outputs: {
    T: number[];
    counts: number[];
    log_T: number[];
    removed: number[];
  };
  cases: { checks: { removed: { reference_decimal: string } } }[];
}
interface RecordedValidation {
  schema_version: number;
  status: string;
  sweep: Sweep;
  weak: Sweep & { diagnostic_one_minus_stored_T: number[] };
  tail: Sweep;
  inverse: { decrement: number; actual: number; ulps: number }[];
}
export type TransmissionPlotKind = "sweep" | "weak" | "tail";

export function transmissionPanels(
  record: RecordedValidation,
  kind: TransmissionPlotKind,
): PlotPanel[] {
  if (record.schema_version !== 1 || record.status !== "passed") {
    throw new Error(
      "Transmission plots require a passed version-1 execution record",
    );
  }
  const series = (
    label: string,
    values: number[],
    style: PlotSeries["style"] = "primary",
  ): PlotSeries => ({ label, values, style });
  const panel = (
    title: string,
    x: number[],
    y: PlotSeries[],
    yScale: PlotPanel["yScale"] = "log",
    xScale: PlotPanel["xScale"] = "linear",
    note = "",
  ): PlotPanel => ({
    title,
    x,
    series: y,
    xLabel: "Optical depth L",
    xScale,
    yScale,
    note,
  });
  let panels: PlotPanel[];
  if (kind === "sweep") {
    panels = [
      panel("Transmission T", record.sweep.optical_depths, [
        series("CUDA transmission", record.sweep.outputs.T),
      ]),
      panel("Expected counts λ", record.sweep.optical_depths, [
        series("CUDA expected counts", record.sweep.outputs.counts),
      ]),
      panel(
        "Log transmission",
        record.sweep.optical_depths,
        [series("CUDA log transmission", record.sweep.outputs.log_T)],
        "linear",
      ),
    ];
  } else if (kind === "weak") {
    panels = [
      panel(
        "Removed-primary fraction",
        record.weak.optical_depths,
        [
          series("CUDA removed fraction", record.weak.outputs.removed),
          series(
            "100-digit reference",
            record.weak.cases.map((row) =>
              Number(row.checks.removed.reference_decimal),
            ),
            "reference",
          ),
          series(
            "1 − stored T (binary32)",
            record.weak.diagnostic_one_minus_stored_T,
            "diagnostic",
          ),
        ],
        "log",
        "log",
        "The reference curve is rounded to binary64.",
      ),
      {
        ...panel(
          "Inverse error (binary32 ULPs)",
          record.inverse.map((row) => row.decrement),
          [
            series(
              "CUDA inverse error",
              record.inverse.map((row) => row.ulps),
            ),
          ],
          "linear",
          "log",
          "Error against an independent high-precision inverse.",
        ),
        xLabel: "Supplied decrement δ",
      },
    ];
  } else {
    panels = [
      panel("Stored transmission T", record.tail.optical_depths, [
        series("CUDA transmission", record.tail.outputs.T),
      ]),
      panel("Stored expected counts λ", record.tail.optical_depths, [
        series("CUDA expected counts", record.tail.outputs.counts),
      ]),
    ];
  }
  panels.forEach(validatePlotPanel);
  return panels;
}

export function validatePlotPanel(item: PlotPanel): void {
  if (
    !item ||
    !Array.isArray(item.x) ||
    !Array.isArray(item.series) ||
    !item.series.length ||
    !["log", "linear"].includes(item.xScale) ||
    !["log", "linear"].includes(item.yScale)
  )
    throw new Error("Invalid recorded plot panel");
  if (
    item.x.length < 2 ||
    item.x.some(
      (x, i) =>
        !Number.isFinite(x) ||
        (item.xScale === "log" && x <= 0) ||
        (i > 0 && x <= (item.x[i - 1] ?? -Infinity)),
    )
  )
    throw new Error("Invalid recorded plot abscissae");
  if (item.series.some((line) => !line || !Array.isArray(line.values)))
    throw new Error("Invalid recorded plot series");
  if (
    item.yScale === "log" &&
    !item.series.some((line) => line.values.some((value) => value > 0))
  ) {
    throw new Error(
      "Recorded logarithmic panel has no positive ordinates; use a zero-only presentation",
    );
  }
  for (const line of item.series) {
    if (
      line.values.length !== item.x.length ||
      line.values.some(
        (y) => !Number.isFinite(y) || (item.yScale === "log" && y < 0),
      )
    )
      throw new Error("Invalid recorded plot ordinates");
  }
}

/** Axis transformation only; zero is never replaced with an arbitrary positive floor. */
export function plotCoordinate(
  value: number,
  domain: [number, number],
  range: [number, number],
  scale: "linear" | "log",
): number {
  if (scale === "log" && (value <= 0 || domain[0] <= 0)) return NaN;
  const transform = scale === "log" ? Math.log10 : (x: number) => x;
  return (
    range[0] +
    ((transform(value) - transform(domain[0])) /
      (transform(domain[1]) - transform(domain[0]))) *
      (range[1] - range[0])
  );
}
