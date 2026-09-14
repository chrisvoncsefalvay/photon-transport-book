import { Euler, MathUtils, Quaternion, Vector3 } from "three";

/** Supine patient: +X left, +Y anterior (up), +Z caudal (feet). */
export interface JointSettings {
  insertionMm: number;
  liftMm: number;
  /** Positive slides the chassis towards the feet along world +Z. */
  longitudinalMm: number;
  /** Positive rotates forward, taking the detector towards patient left (LAO). */
  orbitalDeg: number;
  /** Positive swings proximal, taking the detector towards the head (cranial). */
  angulationDeg: number;
}

export const DEFAULT_JOINTS: Readonly<JointSettings> = Object.freeze({
  insertionMm: 0,
  liftMm: 0,
  longitudinalMm: 0,
  orbitalDeg: 0,
  angulationDeg: 0,
});

export type TerminologyMode = "common" | "cardiac";
export const labelsByMode = {
  common: {
    insertionMm: { name: "In/out", negative: "Out", positive: "In" },
    liftMm: { name: "Raise/lower", negative: "Lower", positive: "Raise" },
    longitudinalMm: {
      name: "Move forward/back",
      negative: "Slide proximal",
      positive: "Slide distal",
    },
    orbitalDeg: {
      name: "Rotate back/forward",
      negative: "Rotate back",
      positive: "Rotate forward",
    },
    angulationDeg: {
      name: "Swing distal/proximal",
      negative: "Swing distal",
      positive: "Swing proximal",
    },
  },
  cardiac: {
    insertionMm: { name: "In/out", negative: "Out", positive: "In" },
    longitudinalMm: {
      name: "Slide distal/proximal",
      negative: "Slide proximal",
      positive: "Slide distal",
    },
    liftMm: {
      name: "Anterior/ posterior",
      negative: "Posterior",
      positive: "Anterior",
    },
    orbitalDeg: { name: "RAO/LAO", negative: "RAO", positive: "LAO" },
    angulationDeg: {
      name: "Caudal/ cranial (cephalad)",
      negative: "Caudal",
      positive: "Cranial (cephalad)",
    },
  },
} as const;

export interface ProjectionAngles {
  /** Signed DICOM-style primary angle, positive LAO; null at an axial pole. */
  obliqueDeg: number | null;
  /** Signed secondary angle, positive cranial, negative caudal. */
  cranialDeg: number;
  obliqueLabel: "Frontal" | "LAO" | "RAO" | "Posterior" | "Axial";
  angulationLabel: "Neutral" | "Cranial" | "Caudal";
}

const cleanZero = (value: number) => (Math.abs(value) < 1e-10 ? 0 : value);

/**
 * Extract clinical projection from source-to-detector direction expressed in
 * the patient frame. Camera orientation and source/detector translation do not
 * enter this calculation. See notes/c-arm-angle-conventions.md for the source.
 */
export function projectionAngles(direction: Vector3): ProjectionAngles {
  const length = direction.length();
  if (!Number.isFinite(length) || length === 0) {
    throw new RangeError("Detector direction must be a finite nonzero vector");
  }
  const d = direction.clone().divideScalar(length);
  const transverseLength = Math.hypot(d.x, d.y);
  const obliqueDeg =
    transverseLength < 1e-12
      ? null
      : cleanZero(MathUtils.radToDeg(Math.atan2(d.x, d.y)));
  const cranialDeg = cleanZero(
    MathUtils.radToDeg(Math.atan2(-d.z, transverseLength)),
  );
  return {
    obliqueDeg,
    cranialDeg,
    obliqueLabel:
      obliqueDeg === null
        ? "Axial"
        : obliqueDeg === 0
          ? "Frontal"
          : Math.abs(Math.abs(obliqueDeg) - 180) < 1e-10
            ? "Posterior"
            : obliqueDeg > 0
              ? "LAO"
              : "RAO",
    angulationLabel:
      cranialDeg === 0 ? "Neutral" : cranialDeg > 0 ? "Cranial" : "Caudal",
  };
}

export function worldEulerDegrees(rotation: Quaternion) {
  const euler = new Euler().setFromQuaternion(rotation, "XYZ");
  return {
    x: cleanZero(MathUtils.radToDeg(euler.x)),
    y: cleanZero(MathUtils.radToDeg(euler.y)),
    z: cleanZero(MathUtils.radToDeg(euler.z)),
  };
}

/** Absolute pose of the imaging frame relative to its frontal rest frame. */
export function createPose(settings: Partial<JointSettings> = {}) {
  const joints = { ...DEFAULT_JOINTS, ...settings };
  for (const [name, value] of Object.entries(joints)) {
    if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  }
  const orbital = new Quaternion().setFromAxisAngle(
    new Vector3(0, 0, 1),
    -MathUtils.degToRad(joints.orbitalDeg),
  );
  const angulation = new Quaternion().setFromAxisAngle(
    new Vector3(1, 0, 0),
    -MathUtils.degToRad(joints.angulationDeg),
  );
  // Support swing is the parent; the orbital rail is its child.
  const rotation = angulation.multiply(orbital);
  const detectorDirection = new Vector3(0, 1, 0).applyQuaternion(rotation);
  return {
    translationMm: new Vector3(
      joints.insertionMm,
      joints.liftMm,
      joints.longitudinalMm,
    ),
    rotation,
    detectorDirection,
    worldEulerDeg: worldEulerDegrees(rotation),
    projection: projectionAngles(detectorDirection),
  };
}
