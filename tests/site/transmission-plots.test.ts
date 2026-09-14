import { describe, expect, it } from "vitest";
import recorded from "../../public/generated/transmission-contract/validation.json";
import {
  plotCoordinate,
  transmissionPanels,
} from "../../src/lib/transmission-plot-data";

describe("recorded transmission plots", () => {
  it("uses every recorded CUDA sample without resampling or replacing stored zeros", () => {
    const panels = transmissionPanels(recorded, "tail");
    expect(panels[0]!.x).toBe(recorded.tail.optical_depths);
    expect(panels[0]!.series[0]!.values).toBe(recorded.tail.outputs.T);
    expect(panels[1]!.series[0]!.values).toBe(recorded.tail.outputs.counts);
    expect(panels[0]!.series[0]!.values).toContain(0);
    const ordinary = transmissionPanels(recorded, "sweep");
    expect(ordinary.map((panel) => panel.series[0]!.values)).toEqual([
      recorded.sweep.outputs.T,
      recorded.sweep.outputs.counts,
      recorded.sweep.outputs.log_T,
    ]);
  });
  it("retains independent references and the recorded binary32 subtraction diagnostic", () => {
    const [weak, inverse] = transmissionPanels(recorded, "weak");
    expect(weak!.series[1]!.values).toEqual(
      recorded.weak.cases.map((row) =>
        Number(row.checks.removed.reference_decimal),
      ),
    );
    expect(weak!.series[2]!.values).toBe(
      recorded.weak.diagnostic_one_minus_stored_T,
    );
    expect(inverse!.series[0]!.values).toEqual(
      recorded.inverse.map((row) => row.ulps),
    );
    expect(inverse!.x).toEqual(recorded.inverse.map((row) => row.decrement));
  });
  it("rejects failed execution records and mismatched sample lengths", () => {
    expect(() =>
      transmissionPanels({ ...recorded, status: "failed" }, "sweep"),
    ).toThrow();
    expect(() =>
      transmissionPanels(
        { ...recorded, tail: { ...recorded.tail, optical_depths: [80] } },
        "tail",
      ),
    ).toThrow();
  });
  it("rejects an all-zero logarithmic panel rather than inventing a scale", () => {
    expect(() =>
      transmissionPanels(
        {
          ...recorded,
          tail: {
            ...recorded.tail,
            outputs: {
              ...recorded.tail.outputs,
              T: recorded.tail.outputs.T.map(() => 0),
            },
          },
        },
        "tail",
      ),
    ).toThrow("no positive ordinates");
  });
  it("maps linear and logarithmic axes without turning zero into positive data", () => {
    expect(plotCoordinate(10, [0, 20], [40, 240], "linear")).toBe(140);
    expect(plotCoordinate(1e-6, [1e-12, 1], [40, 240], "log")).toBeCloseTo(140);
    expect(plotCoordinate(0, [1e-12, 1], [40, 240], "log")).toBeNaN();
  });
});
