import registration from "../../public/generated/worked-examples/registration/data.json";
import acquisition from "../../public/generated/worked-examples/acquisition/data.json";
import reconstruction from "../../public/generated/worked-examples/reconstruction/data.json";
import provenance from "../../public/generated/worked-examples/registration/provenance.json";
import type { RecordedSeries, Scale } from "./recorded-plot";

export type WorkedExampleKind =
  "registration" | "acquisition" | "reconstruction";
export interface WorkedPlot {
  key: string;
  title: string;
  xLabel: string;
  yLabel: string;
  series: RecordedSeries[];
  xDomain?: [number, number];
  yDomain?: [number, number];
  xTicks?: number[];
  yScale?: Scale;
}

export function workedExamplePlots(example: WorkedExampleKind): WorkedPlot[] {
  if (example === "registration") {
    if (!registration.evaluation.stationary)
      throw new Error("Registration is unfinished");
    const x = registration.history.map((h) => h.iteration);
    const tolerance = provenance.protocol.acceptance.gradient_tolerance;
    return [
      {
        key: "stationarity",
        title: "Reaching the stopping criterion",
        xLabel: "Accepted update",
        yLabel: "Scaled gradient infinity norm",
        yScale: "log",
        xDomain: [0, x.at(-1)!],
        xTicks: [0, 4, 8, 12, 16],
        series: [
          {
            label: "Accepted pose",
            x,
            y: registration.history.map((h) => h.gradient_infinity_norm),
          },
          {
            label: "Required ≤ 0.001",
            x: [0, x.at(-1)!],
            y: [tolerance, tolerance],
            dash: true,
            tone: "accent",
          },
        ],
      },
      {
        key: "recovery",
        title: "Recovering the generating pose",
        xLabel: "Accepted update",
        yLabel: "RMS target error (mm)",
        yScale: "log",
        xDomain: [0, x.at(-1)!],
        xTicks: [0, 4, 8, 12, 16],
        series: [
          {
            label: "Independent evaluation",
            x,
            y: registration.independent_rms_error_mm,
          },
        ],
      },
    ];
  }
  if (example === "reconstruction") {
    if (!reconstruction.evaluations.every((e) => e.accepted))
      throw new Error("Reconstruction is unfinished");
    const end = Math.max(...reconstruction.curves.map((c) => c.steps.at(-1)!));
    const series = (key: "objective" | "mapping"): RecordedSeries[] =>
      reconstruction.curves.map((c) => ({
        label: `Replicate ${c.replicate}`,
        x: c.steps,
        y: c[key],
        tone: c.replicate === 0 ? "technical" : "accent",
        dash: c.replicate === 1,
      }));
    const limit =
      reconstruction.evaluations[0]!.stationarity.absolute_threshold;
    return [
      {
        key: "objective",
        title: "Reducing the fitting objective",
        xLabel: "Accepted update",
        yLabel: "Poisson half-deviance + penalty",
        yScale: "log",
        xDomain: [0, end],
        series: series("objective"),
      },
      {
        key: "stationarity",
        title: "Reaching stationarity",
        xLabel: "Accepted update",
        yLabel: "Euclidean projected mapping",
        yScale: "log",
        xDomain: [0, end],
        series: [
          ...series("mapping"),
          {
            label: "Required ≤ 10",
            x: [0, end],
            y: [limit, limit],
            dash: true,
            tone: "muted",
          },
        ],
      },
    ];
  }
  if (!acquisition.summary.passed)
    throw new Error("Acquisition comparison is unfinished");
  const decision = acquisition.first_decision;
  const candidates: RecordedSeries[] = decision.candidates.map((c) => ({
    label:
      c.name === decision.selected_candidate
        ? "Selected view"
        : "Other candidates",
    legend:
      c.name === decision.selected_candidate || c === decision.candidates[0],
    x: [Number(c.name.slice(1)) - 180],
    y: [c.mean_target_variance_mm2],
    mode: "bars",
    tone: c.name === decision.selected_candidate ? "technical" : "muted",
  }));
  const pairs = acquisition.pairs;
  return [
    {
      key: "decision",
      title: "The first view choice",
      xLabel: "Candidate view (degrees)",
      yLabel: "Predicted target variance (mm²)",
      xDomain: [0, 100],
      xTicks: [5, 30, 60, 90],
      yDomain: [
        0,
        Math.max(
          ...decision.candidates.map((c) => c.mean_target_variance_mm2),
        ) * 1.12,
      ],
      series: candidates,
    },
    {
      key: "pairs",
      title: "All eight recovery outcomes",
      xLabel: "Prescribed noise replicate",
      yLabel: "RMS target error (mm)",
      xDomain: [-0.5, 7.5],
      xTicks: pairs.map((p) => p.replicate),
      yDomain: [
        0,
        Math.max(
          ...pairs.flatMap((p) => [
            p.selected.errors.landmark_rms_mm,
            p.near_parallel.errors.landmark_rms_mm,
          ]),
        ) * 1.1,
      ],
      series: [
        ...pairs.map((p): RecordedSeries => ({
          label: `Pair ${p.replicate}`,
          legend: false,
          tone: "muted",
          x: [p.replicate, p.replicate],
          y: [
            p.near_parallel.errors.landmark_rms_mm,
            p.selected.errors.landmark_rms_mm,
          ],
        })),
        {
          label: "Near-parallel 5°",
          x: pairs.map((p) => p.replicate),
          y: pairs.map((p) => p.near_parallel.errors.landmark_rms_mm),
          mode: "markers",
          marker: "square",
          tone: "accent",
        },
        {
          label: "Selected view",
          x: pairs.map((p) => p.replicate),
          y: pairs.map((p) => p.selected.errors.landmark_rms_mm),
          mode: "markers",
          tone: "technical",
        },
      ],
    },
  ];
}
