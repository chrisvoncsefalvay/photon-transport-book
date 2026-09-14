import {
  posesForParameter,
  type Parameter,
  type RecordedPose,
  type SensitivityData,
} from "./data";

export type TransformMode = "translate" | "rotate";
export type TransformAxis = "x" | "y" | "z";

export function poseParameter(
  mode: TransformMode,
  axis: TransformAxis,
): Parameter {
  return `${mode === "translate" ? "t" : "r"}${axis}`;
}

/** Return the actual record, never an interpolated or compound physical pose. */
export function nearestRecordedPose(
  data: SensitivityData,
  parameter: Parameter,
  value: number,
): RecordedPose {
  const poses = posesForParameter(data, parameter);
  const finiteValue = Number.isFinite(value) ? value : 0;
  return poses.reduce((nearest, pose) => {
    const distance = Math.abs(pose.value - finiteValue);
    const previous = Math.abs(nearest.value - finiteValue);
    return distance < previous ||
      (distance === previous && Math.abs(pose.value) < Math.abs(nearest.value))
      ? pose
      : nearest;
  });
}

/** A gesture spans the recorded range; this gain never scales the geometry. */
export function draggedRecordedPose(
  data: SensitivityData,
  parameter: Parameter,
  start: RecordedPose,
  rangeFraction: number,
): RecordedPose {
  const range = Math.max(
    ...posesForParameter(data, parameter).map((pose) => Math.abs(pose.value)),
  );
  const origin = start.parameter === parameter ? start.value : 0;
  return nearestRecordedPose(data, parameter, origin + rangeFraction * range);
}

/** Switching coordinate starts at zero, so keyboard requests cannot accumulate. */
export function steppedRecordedPose(
  data: SensitivityData,
  parameter: Parameter,
  current: RecordedPose,
  direction: -1 | 1,
): RecordedPose {
  const poses = posesForParameter(data, parameter);
  const value = current.parameter === parameter ? current.value : 0;
  const index = poses.findIndex((pose) => pose.value === value);
  return poses[Math.max(0, Math.min(poses.length - 1, index + direction))]!;
}
