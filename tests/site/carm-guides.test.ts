import { describe, expect, it } from "vitest";
import { Matrix4, Quaternion, Vector3 } from "three";
import {
  detectorUV,
  imagingPlane,
  referenceBasis,
  rotationArc,
  translationDirection,
} from "../../src/lib/carm/guides";
import { createPose } from "../../src/lib/carm/kinematics";

function expectVector(actual: Vector3, expected: Vector3) {
  expect(actual.distanceTo(expected)).toBeLessThan(1e-12);
}
function imagingMatrix(orbital: number, swing: number, origin = new Vector3()) {
  return new Matrix4()
    .makeTranslation(origin.x, origin.y, origin.z)
    .multiply(new Matrix4().makeRotationX(-swing))
    .multiply(new Matrix4().makeRotationZ(-orbital));
}

describe("C-arm picture plane", () => {
  it.each([
    [0, 0],
    [0.7, -0.4],
    [-0.8, 0.5],
  ])(
    "contains detector U/V through the isocentre and is normal to the ray (%s,%s)",
    (orbital, swing) => {
      const transform = imagingMatrix(
        orbital,
        swing,
        new Vector3(1.2, 0.8, -0.3),
      );
      const plane = imagingPlane(transform);
      const origin = new Vector3().applyMatrix4(transform);
      const source = new Vector3(0, -0.35, 0).applyMatrix4(transform);
      const detector = new Vector3(0, 0.35, 0).applyMatrix4(transform);
      const ray = detector.clone().sub(source).normalize();
      expectVector(plane.normal, ray);
      for (const local of [
        new Vector3(),
        new Vector3(0.45, 0, 0),
        new Vector3(0, 0, -0.3),
      ])
        expect(
          plane.distanceToPoint(local.applyMatrix4(transform)),
        ).toBeCloseTo(0, 12);
      expect(
        plane.distanceToPoint(origin.addScaledVector(ray, 0.2)),
      ).toBeCloseTo(0.2, 12);
    },
  );
});

describe("visible reference frames and detector coordinates", () => {
  it.each(["world", "patient"] as const)(
    "preserves the %s XYZ basis and explicit origin",
    (reference) => {
      const matrix = imagingMatrix(0.3, 0.2, new Vector3(-1, 0.8, 2));
      const basis = referenceBasis(matrix, reference);
      expectVector(basis.origin, new Vector3(-1, 0.8, 2));
      expectVector(basis.x.clone().cross(basis.y), basis.z);
      expectVector(basis.x, new Vector3(1, 0, 0).transformDirection(matrix));
      expectVector(basis.y, new Vector3(0, 1, 0).transformDirection(matrix));
    },
  );

  it("makes the detector UVN triad right-handed with normal along the beam", () => {
    const matrix = imagingMatrix(-0.6, 0.45, new Vector3(0.2, 1, -0.1));
    const basis = referenceBasis(matrix, "detector");
    expectVector(basis.x.clone().cross(basis.y), basis.z);
    expectVector(basis.z, new Vector3(0, 1, 0).transformDirection(matrix));
    expectVector(basis.y, new Vector3(0, 0, -1).transformDirection(matrix));
    expect(basis.x.dot(basis.y)).toBeCloseTo(0, 12);
  });

  it("reports signed orthogonal plane coordinates including an offset body target", () => {
    // Detector +90 degrees around world Z: local X becomes +Y; local Y becomes -X.
    const detector = new Matrix4()
      .makeTranslation(1, 2, 3)
      .multiply(new Matrix4().makeRotationZ(Math.PI / 2));
    // World point relative detector is (0.4,0.12,0.07), so local=(0.12,-0.4,0.07).
    const coordinates = detectorUV(new Vector3(1.4, 2.12, 3.07), detector);
    expect(coordinates.u).toBeCloseTo(0.12, 12);
    expect(coordinates.v).toBeCloseTo(-0.07, 12);
    expect(coordinates.normalDistance).toBeCloseTo(-0.4, 12);
    expect(detectorUV(new Vector3(1, 2, 3), detector)).toEqual({
      u: 0,
      v: -0,
      normalDistance: 0,
    });
  });

  it("does not confuse orthogonal UV coordinates with a perspective detector projection", () => {
    const detector = new Matrix4().makeTranslation(0, 1, 0);
    const near = detectorUV(new Vector3(0.1, 0.9, 0.2), detector);
    const far = detectorUV(new Vector3(0.1, -0.7, 0.2), detector);
    expect(near.u).toBe(far.u);
    expect(near.v).toBe(far.v);
    expect(near.normalDistance).not.toBe(far.normalDistance);
  });
});

describe("movement direction arrows", () => {
  it.each([-1, 1] as const)(
    "points longitudinal guide %s along actual chassis travel while the arm is rotated",
    (sign) => {
      const before = createPose({ orbitalDeg: 40, angulationDeg: -25 });
      const after = createPose({
        orbitalDeg: 40,
        angulationDeg: -25,
        longitudinalMm: sign * 150,
      });
      const displacement = after.translationMm
        .clone()
        .sub(before.translationMm)
        .normalize();
      const direction = translationDirection("longitudinalMm", sign);
      expectVector(direction, new Vector3(0, 0, sign));
      expectVector(direction, displacement);
      expect(after.projection).toEqual(before.projection);
    },
  );
  it.each([
    ["support", new Vector3(1, 0, 0)],
    ["orbital", new Vector3(0, 0, 1)],
  ] as const)(
    "follows actual positive and negative %s joint rotations",
    (_name, axis) => {
      for (const sign of [-1, 1] as const) {
        const radial = new Vector3(0, 1, 0);
        const arc = rotationArc(axis, radial, sign, 0.4, 0.7);
        const independentlyRotated = radial
          .clone()
          .multiplyScalar(0.4)
          .applyQuaternion(
            new Quaternion().setFromAxisAngle(axis, -sign * 0.7),
          );
        expectVector(arc.end, independentlyRotated);
        for (const point of arc.points) {
          expect(point.length()).toBeCloseTo(0.4, 12);
          expect(point.dot(axis)).toBeCloseTo(0, 12);
        }
        // A finite mechanical step follows the arrowhead tangent at the endpoint.
        const next = arc.end
          .clone()
          .applyQuaternion(
            new Quaternion().setFromAxisAngle(axis, -sign * 1e-6),
          );
        expect(next.sub(arc.end).normalize().dot(arc.tangent)).toBeCloseTo(
          1,
          10,
        );
      }
    },
  );

  it("preserves positive directions when a parent rotates the joint plane", () => {
    const parent = new Matrix4().makeRotationX(-0.5);
    const arc = rotationArc(new Vector3(0, 0, 1), new Vector3(0, 1, 0), 1);
    const worldAxis = new Vector3(0, 0, 1).transformDirection(parent);
    const worldEnd = arc.end.clone().applyMatrix4(parent);
    const expectedTangent = new Vector3()
      .crossVectors(worldAxis, worldEnd)
      .negate()
      .normalize();
    expectVector(
      arc.tangent.clone().transformDirection(parent),
      expectedTangent,
    );
  });
});
