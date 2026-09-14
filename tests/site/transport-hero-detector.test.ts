import { describe, expect, it } from "vitest";
import {
  advanceFrameClock,
  FLASH_TAU_SECONDS,
  PHOSPHOR_KERNEL,
  type FrameClock,
  type PhosphorRecord,
} from "../../src/lib/transport-hero-detector";

const record: PhosphorRecord = {
  width: 192,
  height: 160,
  frameCount: 128,
  frameRate: 30,
  atlasColumns: 16,
  tauSeconds: 1.8,
};
const newClock = (): FrameClock => ({ playedFrames: 0, phase: 0 });
const recordedCounts = [0, 1, 4, 0, 2];
const smallRecord = { ...record, frameCount: recordedCounts.length };

function playback(steps: number[]) {
  const clock = newClock();
  let memory = 0,
    flash = 0;
  for (const dt of steps) {
    const plan = advanceFrameClock(clock, dt, smallRecord);
    memory *= plan.decay[0];
    flash *= plan.decay[1];
    for (const impact of plan.impacts) {
      memory += recordedCounts[impact.frameIndex]! * impact.memoryWeight;
      flash += recordedCounts[impact.frameIndex]! * impact.flashWeight;
    }
  }
  return { memory, flash, clock };
}

// These are display time/accumulation checks, not scientific GPU validation.
describe("recorded-impact phosphor schedule", () => {
  it("places frame 0 at the first exact recorded-frame boundary", () => {
    const clock = newClock();
    expect(advanceFrameClock(clock, 1 / 60, record).impacts).toEqual([]);
    expect(clock.playedFrames).toBe(0);
    const boundary = advanceFrameClock(clock, 1 / 60, record);
    expect(boundary.impacts).toEqual([
      { frameIndex: 0, memoryWeight: 1, flashWeight: 1 },
    ]);
    expect(clock).toEqual({ playedFrames: 1, phase: 0 });
  });

  it("ages each observation from its exact arrival, including a partial interval", () => {
    const clock = newClock();
    const plan = advanceFrameClock(clock, 2.5 / 30, record);
    expect(plan.impacts.map((impact) => impact.frameIndex)).toEqual([0, 1]);
    for (let index = 0; index < 2; index++) {
      const age = (1.5 - index) / 30;
      expect(plan.impacts[index]!.memoryWeight).toBeCloseTo(
        Math.exp(-age / 1.8),
        14,
      );
      expect(plan.impacts[index]!.flashWeight).toBeCloseTo(
        Math.exp(-age / FLASH_TAU_SECONDS),
        14,
      );
    }
    expect(clock).toEqual({ playedFrames: 2, phase: 0.5 });
  });

  it("matches an independent event-time sum with looped recorded frames", () => {
    const end = 3.217;
    const actual = playback([end]);
    let memory = 0,
      flash = 0;
    for (let event = 1; event <= Math.floor(end * record.frameRate); event++) {
      const count = recordedCounts[(event - 1) % recordedCounts.length]!;
      const age = end - event / record.frameRate;
      memory += count * Math.exp(-age / record.tauSeconds);
      flash += count * Math.exp(-age / FLASH_TAU_SECONDS);
    }
    expect(actual.memory).toBeCloseTo(memory, 12);
    expect(actual.flash).toBeCloseTo(flash, 12);
  });

  it("is independent of 30/60/120 FPS and irregular visual update intervals", () => {
    const single = playback([4]);
    const alternatives = [
      playback(Array.from({ length: 120 }, () => 1 / 30)),
      playback(Array.from({ length: 240 }, () => 1 / 60)),
      playback(Array.from({ length: 480 }, () => 1 / 120)),
      playback([0.017, 0.092, 0.401, 1.73, 0.36, 1.4]),
    ];
    for (const actual of alternatives) {
      expect(actual.clock).toEqual(single.clock);
      expect(actual.memory).toBeCloseTo(single.memory, 10);
      expect(actual.flash).toBeCloseTo(single.flash, 10);
    }
  });

  it("decays without observations and never invents light in an empty field", () => {
    const plan = advanceFrameClock(newClock(), 0.01, record);
    expect(plan.impacts).toEqual([]);
    expect(7 * plan.decay[0]).toBeCloseTo(7 * Math.exp(-0.01 / 1.8), 14);
    expect(7 * plan.decay[1]).toBeCloseTo(7 * Math.exp(-0.01 / 0.08), 14);
    expect(0 * plan.decay[0]).toBe(0);
  });

  it("preserves exact playback and light while paused, regardless of wall time", () => {
    const clock = newClock();
    advanceFrameClock(clock, 0.08, record);
    const before = { ...clock };
    expect(advanceFrameClock(clock, 3600, record, true)).toEqual({
      decay: [1, 1],
      impacts: [],
    });
    expect(clock).toEqual(before);
    expect(advanceFrameClock(clock, 0, record)).toEqual({
      decay: [1, 1],
      impacts: [],
    });
    expect(clock).toEqual(before);
  });

  it("bounds catch-up to one pass per atlas frame without losing repeated impacts", () => {
    const clock = newClock();
    const plan = advanceFrameClock(clock, 24 * 3600, record);
    expect(plan.impacts).toHaveLength(record.frameCount);
    expect(new Set(plan.impacts.map((impact) => impact.frameIndex)).size).toBe(
      128,
    );
    expect(
      plan.impacts.every(
        (impact) => impact.frameIndex >= 0 && impact.frameIndex < 128,
      ),
    ).toBe(true);
    expect(clock.playedFrames).toBe(24 * 3600 * 30);
    const constantUnitCount = plan.impacts.reduce(
      (sum, impact) => sum + impact.memoryWeight,
      0,
    );
    expect(constantUnitCount).toBeCloseTo(1 / -Math.expm1(-1 / (30 * 1.8)), 11);
  });

  it("wraps atlas indices without resetting persistent memory or elapsed time", () => {
    const clock = newClock();
    advanceFrameClock(clock, 128 / 30, record);
    const next = advanceFrameClock(clock, 1 / 30, record);
    expect(next.impacts).toEqual([
      { frameIndex: 0, memoryWeight: 1, flashWeight: 1 },
    ]);
    expect(clock.playedFrames).toBe(129);
    expect(next.decay[0]).toBeGreaterThan(0.98);
  });

  it.each([NaN, Infinity, -1])(
    "rejects invalid elapsed time %s without changing the clock",
    (dt) => {
      const clock = newClock();
      expect(() => advanceFrameClock(clock, dt, record)).toThrow();
      expect(clock).toEqual(newClock());
    },
  );

  it("rejects invalid timing and persistence that could overflow FP16", () => {
    for (const changed of [
      { frameCount: 0 },
      { frameRate: 0 },
      { tauSeconds: NaN },
      { tauSeconds: 1e6 },
    ])
      expect(() =>
        advanceFrameClock(newClock(), 1, { ...record, ...changed }),
      ).toThrow();
  });

  it("uses a symmetric unit-sum display PSF without amplifying interior count totals", () => {
    expect(PHOSPHOR_KERNEL[0]).toBe(PHOSPHOR_KERNEL[2]);
    expect(PHOSPHOR_KERNEL[1]).toBeGreaterThan(PHOSPHOR_KERNEL[0]);
    const sum2d = PHOSPHOR_KERNEL.reduce(
      (total, x) => total + PHOSPHOR_KERNEL.reduce((row, y) => row + x * y, 0),
      0,
    );
    expect(sum2d).toBeCloseTo(1, 15);
  });
});
