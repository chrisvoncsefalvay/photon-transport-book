/** Exact endpoint geometry for the acquisition figure, in model-world metres. */
export type AcquisitionPoint = readonly [number, number, number];

export interface AcquisitionMetadata {
  joints: { orbital: { pivot: readonly number[] } };
  landmarks: {
    isocentre: readonly number[];
    source: readonly number[];
    detector: readonly number[];
  };
}

export interface AcquisitionGeometry {
  origin: AcquisitionPoint;
  sourceRadius: number;
  detectorRadius: number;
  clearanceCentre: AcquisitionPoint;
  clearanceRadius: number;
  boundaryAngle: number;
}

export const ACQUISITION_DEFAULT_ANGLE = Math.PI / 3;
export const ACQUISITION_DOMAIN_LIMIT = Math.PI / 2;

export function acquisitionGeometry(
  metadata: AcquisitionMetadata,
): AcquisitionGeometry {
  const point = (values: readonly number[]): AcquisitionPoint => {
    if (values.length !== 3 || values.some((v) => !Number.isFinite(v)))
      throw new Error("Acquisition landmarks must be finite XYZ triples");
    return [values[0]!, values[1]!, values[2]!];
  };
  const origin = point(metadata.joints.orbital.pivot);
  const source = point(metadata.landmarks.source);
  const detector = point(metadata.landmarks.detector);
  const isocentre = point(metadata.landmarks.isocentre);
  const sourceRadius = origin[1] - source[1];
  const detectorRadius = detector[1] - origin[1];
  if (
    sourceRadius <= 0 ||
    detectorRadius <= 0 ||
    origin.some((v, i) => Math.abs(v - isocentre[i]!) > 1e-9) ||
    [0, 2].some(
      (i) =>
        Math.abs(source[i]! - origin[i]!) > 1e-9 ||
        Math.abs(detector[i]! - origin[i]!) > 1e-9,
    )
  )
    throw new Error(
      "The acquisition figure requires the rig's Y-aligned rest endpoints",
    );
  // This disk and angular domain are authored illustration parameters, not
  // device specifications. The two unequal endpoint radii come from the model.
  const clearanceRadius = sourceRadius / 2;
  return {
    origin,
    sourceRadius,
    detectorRadius,
    clearanceCentre: source,
    clearanceRadius,
    boundaryAngle: Math.acos(
      1 -
        (clearanceRadius * clearanceRadius) / (2 * sourceRadius * sourceRadius),
    ),
  };
}

/** alpha rotates e0=(0,-1,0) towards e1=(1,0,0), about model +Z. */
export function acquisitionPoint(
  geometry: AcquisitionGeometry,
  radius: number,
  alpha: number,
): AcquisitionPoint {
  return [
    geometry.origin[0] + radius * Math.sin(alpha),
    geometry.origin[1] - radius * Math.cos(alpha),
    geometry.origin[2],
  ];
}

/** Detector positions for the same source-angle interval and clearance test. */
export function acquisitionDetectorArc(
  geometry: AcquisitionGeometry,
  start: number,
  end: number,
) {
  const count = Math.max(2, Math.ceil(Math.abs(end - start) * 36));
  return Array.from({ length: count + 1 }, (_, i) => {
    const alpha = start + ((end - start) * i) / count;
    return {
      alpha,
      point: acquisitionPoint(geometry, -geometry.detectorRadius, alpha),
    };
  });
}

export function acquisitionState(geometry: AcquisitionGeometry, alpha: number) {
  if (!Number.isFinite(alpha))
    throw new Error("Acquisition angle must be finite");
  const source = acquisitionPoint(geometry, geometry.sourceRadius, alpha);
  const detector = acquisitionPoint(geometry, -geometry.detectorRadius, alpha);
  const clearance = Math.hypot(
    ...source.map((v, i) => v - geometry.clearanceCentre[i]!),
  );
  const withinDomain = Math.abs(alpha) <= ACQUISITION_DOMAIN_LIMIT + 1e-12;
  const clearsDisk = clearance >= geometry.clearanceRadius - 1e-12;
  return {
    source,
    detector,
    clearance,
    withinDomain,
    clearsDisk,
    classification: !withinDomain
      ? ("outside" as const)
      : clearsDisk
        ? ("allowed" as const)
        : ("excluded" as const),
  };
}
