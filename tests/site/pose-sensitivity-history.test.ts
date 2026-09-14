import { describe, expect, it } from "vitest";
import {
  committedCoordinates,
  deltaPixels,
  DisplayedProjectionHistory,
  projectionDelta,
  type DisplayedProjection,
} from "../../src/lib/pose-sensitivity/history";
import type { RecordedPose } from "../../src/lib/pose-sensitivity/data";

// Display-state fixtures only: these numbers are not transport results and are
// never exported as a figure or scientific validation evidence.
function frame(
  id: string,
  values: number[],
  parameter: RecordedPose["parameter"] = "tx",
  value = 1,
): DisplayedProjection {
  return {
    pose: {
      id,
      label: id,
      parameter,
      value,
      matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      projection: `${id}.png`,
      projection_f32: `${id}.f32`,
    },
    values: new Float32Array(values),
  };
}

describe("previous displayed projection history", () => {
  it("starts with a zero delta and no invented previous frame", () => {
    const history = new DisplayedProjectionHistory();
    const initial = frame("reference", [2, 7, 0], "reference", 0);
    history.initialise(initial);
    expect(history.current).toBe(initial);
    expect(history.previous).toBeUndefined();
    expect(Array.from(history.delta!.values)).toEqual([0, 0, 0]);
    expect(history.delta!.limit).toBe(0);
    expect(() => history.initialise(initial)).toThrow("already initialised");
  });

  it("subtracts the immediately preceding display, including returning to reference", () => {
    const history = new DisplayedProjectionHistory();
    const a = frame("reference", [10, 30], "reference", 0);
    const b = frame("b", [13, 22]);
    const c = frame("c", [9, 25]);
    history.initialise(a);
    expect(history.commit(history.request(), b)).toBe(true);
    expect(Array.from(history.delta!.values)).toEqual([3, -8]);
    expect(history.commit(history.request(), c)).toBe(true);
    expect(history.previous).toBe(b);
    expect(Array.from(history.delta!.values)).toEqual([-4, 3]);
    expect(history.commit(history.request(), a)).toBe(true);
    expect(history.previous).toBe(c);
    expect(Array.from(history.delta!.values)).toEqual([1, 5]);
  });

  it("leaves history intact on repeated poses and on presentation-only reads", () => {
    const history = new DisplayedProjectionHistory();
    const a = frame("a", [1, 4]);
    const b = frame("b", [3, 2]);
    history.initialise(a);
    history.commit(history.request(), b);
    const delta = history.delta;
    // Delta toggle, inspector selection and scene resume consume these getters.
    for (let i = 0; i < 3; i++) {
      expect(history.current).toBe(b);
      expect(history.previous).toBe(a);
      expect(history.delta).toBe(delta);
      expect(history.delta!.values[i % 2]).toBe(i % 2 ? -2 : 2);
    }
    expect(history.commit(history.request(), b)).toBe(false);
    expect(history.current).toBe(b);
    expect(history.previous).toBe(a);
    expect(history.delta).toBe(delta);
  });

  it("only accepts the latest successful request when loads finish out of order", () => {
    const history = new DisplayedProjectionHistory();
    const a = frame("a", [10]);
    const b = frame("b", [11]);
    const c = frame("c", [15]);
    history.initialise(a);
    const slow = history.request();
    const fast = history.request();
    expect(history.commit(slow, b)).toBe(false);
    expect(history.current).toBe(a);
    expect(history.commit(fast, c)).toBe(true);
    expect(history.commit(slow, b)).toBe(false);
    expect(history.fail(slow)).toBe(false);
    expect(history.current).toBe(c);
    expect(history.previous).toBe(a);
    expect(history.delta!.values[0]).toBe(5);
  });

  it("preserves the displayed pair after a failed request and permits a retry", () => {
    const history = new DisplayedProjectionHistory();
    const a = frame("a", [100]);
    const b = frame("b", [90]);
    const c = frame("c", [95]);
    history.initialise(a);
    history.commit(history.request(), b);
    const delta = history.delta;
    const failed = history.request();
    expect(history.fail(failed)).toBe(true);
    expect(history.commit(failed, c)).toBe(false);
    expect(history.current).toBe(b);
    expect(history.previous).toBe(a);
    expect(history.delta).toBe(delta);
    expect(history.commit(history.request(), c)).toBe(true);
    expect(history.previous).toBe(b);
    expect(history.delta!.values[0]).toBe(5);
  });

  it("a repeated current request cancels pending loads without advancing history", () => {
    const history = new DisplayedProjectionHistory();
    const a = frame("a", [10]);
    history.initialise(a);
    const pending = history.request();
    const repeatCurrent = history.request();
    history.fail(repeatCurrent);
    expect(history.commit(pending, frame("b", [20]))).toBe(false);
    expect(history.current).toBe(a);
    expect(history.previous).toBeUndefined();
    expect(history.delta!.limit).toBe(0);
  });

  it("discarding the controller rejects late completions", () => {
    const history = new DisplayedProjectionHistory();
    const a = frame("a", [10]);
    history.initialise(a);
    const pending = history.request();
    history.dispose();
    expect(history.isLatest(pending)).toBe(false);
    expect(history.commit(pending, frame("b", [20]))).toBe(false);
    expect(history.current).toBe(a);
    expect(history.previous).toBeUndefined();
  });

  it("validates numeric payloads before changing committed history", () => {
    const history = new DisplayedProjectionHistory();
    const a = frame("a", [10, 20]);
    history.initialise(a);
    const delta = history.delta;
    for (const invalid of [
      frame("bad-length", [1]),
      frame("bad-nan", [1, NaN]),
      frame("bad-infinity", [Infinity, 2]),
    ]) {
      const token = history.request();
      expect(() => history.commit(token, invalid)).toThrow();
      expect(history.current).toBe(a);
      expect(history.previous).toBeUndefined();
      expect(history.delta).toBe(delta);
      history.fail(token);
    }
  });
});

describe("recorded count differences and signed display", () => {
  it("subtracts binary32 counts in binary64 before applying any colour stretch", () => {
    const current = new Float32Array([1, 0.1, 1000]);
    const previous = new Float32Array([2 ** -25, 0.2, 999]);
    const delta = projectionDelta(current, previous);
    expect(delta.values).toBeInstanceOf(Float64Array);
    expect(Array.from(delta.values)).toEqual([
      1 - 2 ** -25,
      Math.fround(0.1) - Math.fround(0.2),
      1,
    ]);
    expect(delta.values[0]).not.toBe(Math.fround(delta.values[0]!));
    expect(delta.limit).toBe(1);
    expect(Array.from(current)).toEqual([1, Math.fround(0.1), 1000]);
    expect(Array.from(previous)).toEqual([2 ** -25, Math.fround(0.2), 999]);
  });

  it("uses the sensitivity palette at symmetric signed endpoints and neutral zero", () => {
    const delta = projectionDelta(
      new Float32Array([0, 5, 10]),
      new Float32Array([5, 5, 5]),
    );
    expect(delta.limit).toBe(5);
    expect(Array.from(deltaPixels(delta))).toEqual([
      53, 95, 120, 255, 247, 244, 238, 255, 169, 87, 50, 255,
    ]);
    expect(Array.from(delta.values)).toEqual([-5, 0, 5]);
    expect(
      Array.from(deltaPixels(projectionDelta(new Float32Array([100])))),
    ).toEqual([247, 244, 238, 255]);
  });

  it("uses the documented symmetric asinh softening without clipping valid samples", () => {
    const pixels = deltaPixels({
      values: new Float64Array([-0.02, 0.02]),
      limit: 1,
    });
    const weight = Math.asinh(1) / Math.asinh(50);
    const expected = new Uint8ClampedArray([
      247 + weight * (53 - 247),
      244 + weight * (95 - 244),
      238 + weight * (120 - 238),
      255,
      247 + weight * (169 - 247),
      244 + weight * (87 - 244),
      238 + weight * (50 - 238),
      255,
    ]);
    expect(pixels).toEqual(expected);
    expect(() =>
      deltaPixels({ values: new Float64Array([2]), limit: 1 }),
    ).toThrow("colour range");
    expect(() =>
      deltaPixels({ values: new Float64Array([NaN]), limit: 1 }),
    ).toThrow("finite");
  });

  it("reports only the committed recorded coordinate and resets all other coordinates", () => {
    expect(
      committedCoordinates(frame("rotated", [1], "ry", -0.001).pose),
    ).toEqual({
      tx: 0,
      ty: 0,
      tz: 0,
      rx: 0,
      ry: -0.001,
      rz: 0,
    });
    expect(committedCoordinates(frame("shifted", [1], "tz", 0.5).pose)).toEqual(
      {
        tx: 0,
        ty: 0,
        tz: 0.5,
        rx: 0,
        ry: 0,
        rz: 0,
      },
    );
    expect(
      committedCoordinates(frame("reference", [1], "reference", 0).pose),
    ).toEqual({
      tx: 0,
      ty: 0,
      tz: 0,
      rx: 0,
      ry: 0,
      rz: 0,
    });
  });
});
