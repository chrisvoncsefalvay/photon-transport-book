import { describe, expect, it } from "vitest";
import {
  draggedRecordedPose,
  nearestRecordedPose,
  poseParameter,
  steppedRecordedPose,
} from "../../src/lib/pose-sensitivity/interaction";
import type {
  Parameter,
  RecordedPose,
  SensitivityData,
} from "../../src/lib/pose-sensitivity/data";

// Interaction metadata only; no fixture geometry or pixels are scientific output.
const poses = [
  { id: "reference", parameter: "reference", value: 0 },
  { id: "tx-low", parameter: "tx", value: -1 },
  { id: "tx-half", parameter: "tx", value: -0.5 },
  { id: "tx-positive", parameter: "tx", value: 1 },
  { id: "ry-low", parameter: "ry", value: -0.01 },
  { id: "ry-half", parameter: "ry", value: 0.005 },
  { id: "ry-high", parameter: "ry", value: 0.01 },
] as RecordedPose[];
const data = { poses } as SensitivityData;
const reference = poses[0]!;
const parameters: Parameter[] = ["tx", "ty", "tz", "rx", "ry", "rz"];
const expanded = {
  poses: [
    reference,
    ...parameters.flatMap((parameter) => {
      const halfCount = parameter.startsWith("t") ? 4 : 5;
      const increment = parameter.startsWith("t") ? 0.5 : 0.01;
      return Array.from(
        { length: 2 * halfCount + 1 },
        (_, index) => index - halfCount,
      )
        .filter((index) => index !== 0)
        .map(
          (index) =>
            ({
              id: `${parameter}-${index}`,
              parameter,
              value: index * increment,
            }) as RecordedPose,
        );
    }),
  ],
} as SensitivityData;

describe("recorded pelvis manipulation", () => {
  it.each(parameters)(
    "walks every expanded %s record and clamps to the actual endpoints",
    (parameter) => {
      expect(expanded.poses).toHaveLength(55);
      const halfCount = parameter.startsWith("t") ? 4 : 5;
      const increment = parameter.startsWith("t") ? 0.5 : 0.01;
      let current = reference;
      for (let index = 1; index <= halfCount; index++) {
        current = steppedRecordedPose(expanded, parameter, current, 1);
        expect(current.value).toBeCloseTo(index * increment, 12);
        expect(current.parameter).toBe(parameter);
        expect(expanded.poses).toContain(current);
      }
      expect(steppedRecordedPose(expanded, parameter, current, 1)).toBe(
        current,
      );
      expect(draggedRecordedPose(expanded, parameter, reference, 1)).toBe(
        current,
      );
      for (let index = halfCount - 1; index >= -halfCount; index--) {
        current = steppedRecordedPose(expanded, parameter, current, -1);
        expect(current.value).toBeCloseTo(index * increment, 12);
        expect(current.parameter).toBe(index === 0 ? "reference" : parameter);
        expect(expanded.poses).toContain(current);
      }
      expect(steppedRecordedPose(expanded, parameter, current, -1)).toBe(
        current,
      );
      expect(draggedRecordedPose(expanded, parameter, reference, -1)).toBe(
        current,
      );
    },
  );
  it("converts world-axis gizmo modes into the recorded coordinate names", () => {
    expect(poseParameter("translate", "z")).toBe("tz");
    expect(poseParameter("rotate", "x")).toBe("rx");
  });
  it("returns existing records on an uneven grid and clamps beyond either end", () => {
    expect(nearestRecordedPose(data, "tx", -0.7)).toBe(poses[2]);
    expect(nearestRecordedPose(data, "tx", -100)).toBe(poses[1]);
    expect(nearestRecordedPose(data, "tx", 100)).toBe(poses[3]);
    expect(nearestRecordedPose(data, "tx", 0.5)).toBe(reference);
    expect(nearestRecordedPose(data, "tx", NaN)).toBe(reference);
  });
  it("maps a usable gesture range independently to millimetres and radians", () => {
    expect(draggedRecordedPose(data, "tx", reference, 1)).toBe(poses[3]);
    expect(draggedRecordedPose(data, "ry", reference, 1)).toBe(poses[6]);
    expect(draggedRecordedPose(data, "ry", reference, 0.5)).toBe(poses[5]);
    expect(draggedRecordedPose(data, "ry", poses[5]!, -0.5)).toBe(reference);
  });
  it("starts a different axis at zero instead of accumulating a compound pose", () => {
    expect(draggedRecordedPose(data, "ry", poses[3]!, 0)).toBe(reference);
    expect(draggedRecordedPose(data, "ry", poses[3]!, 0.5)).toBe(poses[5]);
    expect(steppedRecordedPose(data, "ry", poses[3]!, 1)).toBe(poses[5]);
    expect(steppedRecordedPose(data, "tx", poses[6]!, -1)).toBe(poses[2]);
  });
  it("steps actual adjacent records, stops at endpoints and tolerates a reference-only coordinate", () => {
    expect(steppedRecordedPose(data, "tx", poses[1]!, 1)).toBe(poses[2]);
    expect(steppedRecordedPose(data, "tx", poses[2]!, 1)).toBe(reference);
    expect(steppedRecordedPose(data, "tx", poses[3]!, 1)).toBe(poses[3]);
    expect(steppedRecordedPose(data, "tx", poses[1]!, -1)).toBe(poses[1]);
    expect(draggedRecordedPose(data, "rz", reference, 10)).toBe(reference);
    expect(steppedRecordedPose(data, "rz", reference, -1)).toBe(reference);
  });
});
