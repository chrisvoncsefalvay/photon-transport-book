import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  differencePlane,
  materialPlane,
  paletteColour,
  rasterColour,
  rasterPixels,
  validateRasterPayload,
  type RGB,
} from "../../src/lib/worked-raster-data";
import { workedExamplePlots } from "../../src/lib/worked-example-plots";
import registration from "../../public/generated/worked-examples/registration/data.json";
import reconstruction from "../../public/generated/worked-examples/reconstruction/data.json";
import acquisition from "../../public/generated/worked-examples/acquisition/data.json";
import registrationProvenance from "../../public/generated/worked-examples/registration/provenance.json";

const load = (kind: "registration" | "reconstruction") =>
  validateRasterPayload(
    JSON.parse(
      readFileSync(
        new URL(
          `../../public/generated/worked-examples/${kind}/arrays.json`,
          import.meta.url,
        ),
        "utf8",
      ),
    ),
    kind,
  );
const fields = load("reconstruction");
const paper: RGB = [244, 240, 230];
const blue: RGB = [38, 84, 105];
const rust: RGB = [157, 61, 39];

describe("recorded material planes", () => {
  it("interpolates physical zero halfway between sample centres 7 and 8 for every field/material", () => {
    for (const array of Object.values(fields.arrays)) {
      for (const material of [0, 1]) {
        const plane = materialPlane(array, material, 0, -90, 12);
        expect([plane.width, plane.height]).toEqual([16, 16]);
        for (let i = 0; i < 256; i++) {
          const offset = material * 4096 + i;
          expect(plane.values[i]).toBe(
            0.5 * array.values[offset + 7 * 256]! +
              0.5 * array.values[offset + 8 * 256]!,
          );
        }
      }
    }
  });

  it("preserves exact endpoint planes and rejects positions outside the sample-centre interval", () => {
    const array = fields.arrays.reference!;
    for (const material of [0, 1]) {
      for (const slice of [0, 15]) {
        const offset = material * 4096 + slice * 256;
        expect(
          materialPlane(array, material, -90 + slice * 12, -90, 12).values,
        ).toEqual(array.values.slice(offset, offset + 256));
      }
    }
    expect(() => materialPlane(array, 0, -96, -90, 12)).toThrow(/outside/);
    expect(() => materialPlane(array, 0, 96, -90, 12)).toThrow(/outside/);
    expect(() => materialPlane(array, 2, 0, -90, 12)).toThrow(/coordinate/);
    expect(() => materialPlane(array, 0, 0, -90, Infinity)).toThrow(
      /coordinate/,
    );
  });

  it("subtracts reference from recovered fractions without clipping the underlying values", () => {
    const a = { width: 2, height: 1, values: [0.9, 0.1] };
    const b = { width: 2, height: 1, values: [0.1, 0.9] };
    const error = differencePlane(a, b);
    expect(error.values).toEqual([0.8, -0.8]);
    rasterPixels(error, "error", paper, blue, rust);
    expect(error.values).toEqual([0.8, -0.8]);
    expect(() => differencePlane(a, { ...b, width: 1 })).toThrow(/Unmatched/);
  });
});

describe("fixed display windows and orientation", () => {
  it("uses the same RGB values for shorthand print white and full palette colours", () => {
    expect(paletteColour(" #fff ")).toEqual([255, 255, 255]);
    expect(paletteColour("#abcdef")).toEqual([171, 205, 239]);
    expect(paletteColour("#ABC")).toEqual([170, 187, 204]);
    expect(() => paletteColour("#ffff")).toThrow(/palette/);
  });
  it("maps low stored rows to the bottom of the canvas", () => {
    expect(
      Array.from(
        rasterPixels(
          { width: 2, height: 2, values: [0, 1, 1, 0] },
          "fraction",
          paper,
          blue,
          rust,
        ),
      ),
    ).toEqual([...blue, 255, ...paper, 255, ...paper, 255, ...blue, 255]);
  });

  it("keeps signed-error zero neutral and saturates only its colours", () => {
    expect(rasterColour(-0.1, "error", paper, blue, rust)).toEqual(blue);
    expect(rasterColour(0, "error", paper, blue, rust)).toEqual(paper);
    expect(rasterColour(0.1, "error", paper, blue, rust)).toEqual(rust);
    expect(rasterColour(-0.8, "error", paper, blue, rust)).toEqual(blue);
    expect(rasterColour(0.8, "error", paper, blue, rust)).toEqual(rust);
  });

  it("uses one logarithmic count window for all recorded views and stages", () => {
    const detectors = load("registration");
    expect(
      Object.values(detectors.arrays).reduce((n, a) => n + a.values.length, 0),
    ).toBe(98304);
    expect(rasterColour(0, "counts", paper, blue, rust)).toEqual([0, 0, 0]);
    expect(rasterColour(10000, "counts", paper, blue, rust)).toEqual([
      255, 255, 255,
    ]);
    const mid = Math.sqrt(10001) - 1;
    expect(rasterColour(mid, "counts", paper, blue, rust)).toEqual([
      128, 128, 128,
    ]);
    expect(() => rasterColour(-1, "counts", paper, blue, rust)).toThrow(
      /Negative/,
    );
    expect(() => rasterColour(NaN, "counts", paper, blue, rust)).toThrow(
      /Nonfinite/,
    );
  });

  it("rejects incomplete arrays and malformed physical grids", () => {
    expect(() =>
      validateRasterPayload({ ...fields, grid: {} }, "reconstruction"),
    ).toThrow(/grid/);
    expect(() =>
      validateRasterPayload({ ...fields, arrays: {} }, "reconstruction"),
    ).toThrow(/array/);
    const array = fields.arrays.reference!;
    expect(() =>
      validateRasterPayload(
        {
          ...fields,
          arrays: {
            ...fields.arrays,
            reference: { ...array, values: [NaN, ...array.values.slice(1)] },
          },
        },
        "reconstruction",
      ),
    ).toThrow(/array/);
  });
});

describe("worked-example curve and comparison data", () => {
  it("retains every accepted material update and the independently recorded threshold", () => {
    const [objective, mapping] = workedExamplePlots("reconstruction");
    for (const [i, curve] of reconstruction.curves.entries()) {
      expect(objective!.series[i]!.x).toEqual(curve.steps);
      expect(objective!.series[i]!.y).toEqual(curve.objective);
      expect(mapping!.series[i]!.x).toEqual(curve.steps);
      expect(mapping!.series[i]!.y).toEqual(curve.mapping);
    }
    expect(mapping!.series[2]!.y).toEqual(
      Array(2).fill(
        reconstruction.evaluations[0]!.stationarity.absolute_threshold,
      ),
    );
  });

  it("uses the registration norm, independent geometric errors and protocol gate unchanged", () => {
    const [stationarity, recovery] = workedExamplePlots("registration");
    expect(stationarity!.series[0]!.y).toEqual(
      registration.history.map((h) => h.gradient_infinity_norm),
    );
    expect(stationarity!.series[1]!.y).toEqual(
      Array(2).fill(
        registrationProvenance.protocol.acceptance.gradient_tolerance,
      ),
    );
    expect(recovery!.series[0]!.y).toEqual(
      registration.independent_rms_error_mm,
    );
  });

  it("joins only the two outcomes within each noise replicate", () => {
    const [decision, outcomes] = workedExamplePlots("acquisition");
    for (const [
      i,
      candidate,
    ] of acquisition.first_decision.candidates.entries()) {
      expect(decision!.series[i]!.x).toEqual([
        Number(candidate.name.slice(1)) - 180,
      ]);
      expect(decision!.series[i]!.y).toEqual([
        candidate.mean_target_variance_mm2,
      ]);
    }
    const connectors = outcomes!.series.filter((s) => s.mode !== "markers");
    expect(connectors).toHaveLength(8);
    for (const [i, pair] of acquisition.pairs.entries()) {
      expect(connectors[i]!.x).toEqual([pair.replicate, pair.replicate]);
      expect(connectors[i]!.y).toEqual([
        pair.near_parallel.errors.landmark_rms_mm,
        pair.selected.errors.landmark_rms_mm,
      ]);
    }
    expect(
      outcomes!.series.filter((s) => s.mode === "markers").map((s) => s.y),
    ).toEqual([
      acquisition.pairs.map((p) => p.near_parallel.errors.landmark_rms_mm),
      acquisition.pairs.map((p) => p.selected.errors.landmark_rms_mm),
    ]);
  });
});
