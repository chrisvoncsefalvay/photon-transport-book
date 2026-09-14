import { describe, expect, it } from "vitest";
import {
  SANDBOX_DEFAULT,
  SANDBOX_PHANTOM,
  SANDBOX_DEPTH_WINDOW,
  ellipsoidChord,
  ellipsoidInterval,
  sandboxOpticalDepth,
  sandboxSourceOffsets,
  sandboxTransmission,
  sandboxImage,
  validateSandboxGeometry,
  type SandboxEllipsoid,
  type SandboxVector,
} from "../../src/lib/projection-sandbox";

const sphere: SandboxEllipsoid = {
  name: "Independent sphere",
  centre: [0, 0, 0],
  radii: [2, 2, 2],
  coefficient: 1,
};
describe("exact projection sandbox", () => {
  it("matches independent sphere chords, including clipped and tangent segments", () => {
    expect(ellipsoidChord([0, 0, -10], [0, 0, 10], sphere)).toBeCloseTo(4, 12);
    expect(ellipsoidChord([1, 0, -10], [1, 0, 10], sphere)).toBeCloseTo(
      2 * Math.sqrt(3),
      12,
    );
    expect(ellipsoidChord([0, 0, 0], [0, 0, 10], sphere)).toBeCloseTo(2, 12);
    expect(ellipsoidChord([0, 0, -1], [0, 0, 1], sphere)).toBeCloseTo(2, 12);
    expect(ellipsoidInterval([0, 0, -10], [0, 0, -3], sphere)).toBeNull();
    expect(ellipsoidInterval([2, 0, -10], [2, 0, 10], sphere)).toBeNull();
    expect(ellipsoidChord([0, 0, 0], [0, 0, 0], sphere)).toBe(0);
  });
  it("preserves anisotropic physical lengths under segment reversal", () => {
    const ellipsoid: SandboxEllipsoid = {
      name: "Anisotropic",
      centre: [4, -3, 2],
      radii: [2, 3, 5],
      coefficient: 1,
    };
    expect(ellipsoidChord([-10, -3, 2], [10, -3, 2], ellipsoid)).toBeCloseTo(
      4,
      12,
    );
    expect(ellipsoidChord([4, -3, -10], [4, -3, 10], ellipsoid)).toBeCloseTo(
      10,
      12,
    );
    expect(ellipsoidChord([4, -3, 10], [4, -3, -10], ellipsoid)).toBeCloseTo(
      10,
      12,
    );
  });
  it("proves the fixed inserts are contained and their axis boxes are disjoint", () => {
    const outer = SANDBOX_PHANTOM[0]!;
    for (const inner of SANDBOX_PHANTOM.slice(1)) {
      const bound =
        Math.hypot(
          ...inner.centre.map(
            (c, i) => (c - outer.centre[i]!) / outer.radii[i]!,
          ),
        ) + Math.max(...inner.radii.map((r, i) => r / outer.radii[i]!));
      expect(bound).toBeLessThan(1);
    }
    const inserts = SANDBOX_PHANTOM.slice(1);
    for (let i = 0; i < inserts.length; i++)
      for (let j = i + 1; j < inserts.length; j++) {
        const a = inserts[i]!,
          b = inserts[j]!;
        expect(
          [0, 1, 2].some(
            (axis) =>
              Math.abs(a.centre[axis]! - b.centre[axis]!) >
              a.radii[axis]! + b.radii[axis]!,
          ),
        ).toBe(true);
      }
  });
  it("replaces rather than double-counts material along independently known axial chords", () => {
    expect(sandboxOpticalDepth([0, 0, -100], [0, 0, 100])).toBeCloseTo(
      0.02 * 80,
      12,
    );
    const boneOuter = 80 * Math.sqrt(1 - (-20 / 55) ** 2 - (8 / 75) ** 2);
    expect(sandboxOpticalDepth([-20, 8, -100], [-20, 8, 100])).toBeCloseTo(
      0.02 * (boneOuter - 24) + 0.05 * 24,
      12,
    );
    const airOuter = 80 * Math.sqrt(1 - (-35 / 75) ** 2);
    expect(sandboxOpticalDepth([0, -35, -100], [0, -35, 100])).toBeCloseTo(
      0.02 * (airOuter - 28),
      12,
    );
  });
  it("keeps symmetric source quadrature centred, bounded and equal-area in radius", () => {
    const offsets = sandboxSourceOffsets(10);
    expect(offsets).toHaveLength(32);
    expect(offsets.reduce((s, p) => s + p[0], 0)).toBeCloseTo(0, 14);
    expect(offsets.reduce((s, p) => s + p[1], 0)).toBeCloseTo(0, 14);
    expect(
      Math.max(...offsets.map((p) => Math.hypot(p[0], p[1]))),
    ).toBeLessThan(5);
    expect(
      offsets.reduce((s, p) => s + p[0] ** 2 + p[1] ** 2, 0) / offsets.length,
    ).toBeCloseTo(25 / 2, 12);
    expect(sandboxSourceOffsets(0)).toEqual([[0, 0, 0]]);
  });
  it("takes a mean of transmissions and obeys the point-source limit and Jensen bound", () => {
    expect(sandboxTransmission(SANDBOX_DEFAULT, 0, 0)).toBeCloseTo(
      Math.exp(-1.6),
      12,
    );
    const geometry = { ...SANDBOX_DEFAULT, spotDiameter: 10 };
    const target: SandboxVector = [107, 0, 500];
    const depths = sandboxSourceOffsets(10).map((p) =>
      sandboxOpticalDepth([p[0], p[1], -500], target),
    );
    const expected =
      depths.reduce((s, d) => s + Math.exp(-d), 0) / depths.length;
    const wrong = Math.exp(-depths.reduce((s, d) => s + d, 0) / depths.length);
    expect(sandboxTransmission(geometry, target[0], target[1])).toBeCloseTo(
      expected,
      14,
    );
    expect(expected).toBeGreaterThan(wrong + 1e-5);
  });
  it("rejects invalid acquisition geometry and preserves the declared fixed display window", () => {
    expect(() =>
      validateSandboxGeometry({ ...SANDBOX_DEFAULT, sod: NaN }),
    ).toThrow();
    expect(() =>
      validateSandboxGeometry({ ...SANDBOX_DEFAULT, sdd: 500 }),
    ).toThrow();
    expect(() =>
      validateSandboxGeometry({ ...SANDBOX_DEFAULT, spotDiameter: -1 }),
    ).toThrow();
    const image = sandboxImage(SANDBOX_DEFAULT, 9);
    const centre = (4 * 9 + 4) * 4;
    expect(image[centre]).toBe(Math.round((255 * 1.6) / SANDBOX_DEPTH_WINDOW));
    for (let i = 0; i < image.length; i += 4) {
      expect(image[i]).toBe(image[i + 1]);
      expect(image[i]).toBe(image[i + 2]);
      expect(image[i + 3]).toBe(255);
    }
    expect(sandboxImage(SANDBOX_DEFAULT, 9, "surface")).not.toEqual(image);
  });
});
