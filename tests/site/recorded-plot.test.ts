import { describe, it, expect } from "vitest";
import {
  extent,
  mapValue,
  plotPath,
  ticks,
  formatTick,
} from "../../src/lib/recorded-plot";
describe("recorded scientific plots", () => {
  it("preserves gaps and undefined values rather than connecting across them", () => {
    const d = plotPath(
      { label: "sample", x: [0, 1, 2, 3], y: [0, 1, null, 3] },
      [0, 3],
      [0, 3],
      "linear",
      "linear",
    );
    expect(d.match(/M/g)).toHaveLength(2);
    expect(d.match(/L/g)).toHaveLength(1);
    expect(d).not.toContain("NaN");
  });
  it("does not turn zero or negative values into positive log measurements", () => {
    const d = plotPath(
      { label: "log", x: [1, 2, 3, 4, 5], y: [1, 0, 10, -1, 100] },
      [1, 5],
      [1, 100],
      "linear",
      "log",
    );
    expect(d.match(/M/g)).toHaveLength(3);
    expect(d).not.toContain("L");
    expect(extent([null, 0, -1, 2, 20], "log")).toEqual([2, 20]);
  });
  it("maps decades uniformly and preserves axis endpoints", () => {
    expect(mapValue(10, [1, 100], [0, 10], "log")).toBe(5);
    expect(mapValue(100, [1, 100], [0, 10], "log")).toBe(10);
  });
  it("draws step discontinuities without diagonal interpolation", () => {
    expect(
      plotPath(
        { label: "step", x: [0, 1], y: [0, 1], mode: "step" },
        [0, 1],
        [0, 1],
        "linear",
        "linear",
      ),
    ).toBe("M76.000,244.000H400.000V18.000");
  });
  it("rejects unpaired arrays and empty valid domains", () => {
    expect(() =>
      plotPath(
        { label: "bad", x: [0], y: [] },
        [0, 1],
        [0, 1],
        "linear",
        "linear",
      ),
    ).toThrow("Unpaired");
    expect(() => extent([0, null, -1], "log")).toThrow();
  });
  it("bounds log tick density and formats extremes without losing signs", () => {
    const actual = ticks([1e-16, 1], "log");
    const expected = [1e-16, 1e-12, 1e-8, 1e-4, 1];
    expect(actual).toHaveLength(expected.length);
    actual.forEach((tick, index) => {
      // Exponentiation can differ by an ULP between JS runtimes. Relative
      // error still rejects zero and misplaced decades at the smallest tick.
      expect(Math.abs(tick / expected[index]! - 1)).toBeLessThanOrEqual(
        4 * Number.EPSILON,
      );
    });
    expect(actual.map(formatTick)).toEqual([
      "10⁻¹⁶",
      "10⁻¹²",
      "10⁻⁸",
      "10⁻⁴",
      "1",
    ]);
    expect(formatTick(1e-12)).toBe("10⁻¹²");
    expect(formatTick(-0.25)).toBe("−0.25");
    expect(formatTick(0)).toBe("0");
  });
});
