import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it } from "vitest";
import { Group, Mesh, Vector3 } from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import metadata from "../../public/assets/carm/c-arm-rig.json";
import {
  acquisitionDetectorArc,
  acquisitionGeometry,
  acquisitionState,
} from "../../src/lib/carm/acquisition-geometry";
import { attachAcquisitionOrbit } from "../../src/lib/carm/acquisition";

const geometry = acquisitionGeometry(metadata);
const closeVector = (actual: Vector3, expected: readonly number[]) => {
  expect(actual.distanceTo(new Vector3(...expected))).toBeLessThan(1e-10);
};

describe("acquisition illustration from the supplied C-arm metadata", () => {
  it("retains the actual unequal source and detector radii", () => {
    expect(geometry.sourceRadius * 1000).toBeCloseTo(353.72339, 8);
    expect(geometry.detectorRadius * 1000).toBeCloseTo(348.27661, 8);
    expect(geometry.sourceRadius + geometry.detectorRadius).toBeCloseTo(
      metadata.landmarks.source_detector_distance,
      12,
    );
    expect(geometry.sourceRadius).not.toBe(geometry.detectorRadius);
    expect(acquisitionState(geometry, 0).source).toEqual(
      metadata.landmarks.source,
    );
    expect(acquisitionState(geometry, 0).detector).toEqual(
      metadata.landmarks.detector,
    );
  });

  it("has the exact inclusive clearance boundary and authored angular domain", () => {
    const boundary = Math.acos(7 / 8);
    expect(geometry.boundaryAngle).toBeCloseTo(boundary, 14);
    expect(geometry.clearanceRadius * 1000).toBeCloseTo(176.861695, 8);
    for (const sign of [-1, 1]) {
      const onBoundary = acquisitionState(geometry, sign * boundary);
      expect(onBoundary.clearance).toBeCloseTo(geometry.clearanceRadius, 14);
      expect(onBoundary.classification).toBe("allowed");
      expect(
        acquisitionState(geometry, sign * (boundary - 1e-6)).classification,
      ).toBe("excluded");
      expect(
        acquisitionState(geometry, sign * (boundary + 1e-6)).classification,
      ).toBe("allowed");
      expect(
        acquisitionState(geometry, (sign * Math.PI) / 2).classification,
      ).toBe("allowed");
      expect(
        acquisitionState(geometry, sign * (Math.PI / 2 + 1e-6)).classification,
      ).toBe("outside");
    }
    expect(acquisitionState(geometry, 0).classification).toBe("excluded");
  });

  it("keeps the opposite detector outside the disk throughout the candidate domain", () => {
    // Independently evaluated lower bound: cos(alpha)>=0 makes the squared
    // detector-to-c distance at least Rs²+Rd² throughout [-pi/2,pi/2].
    const lowerBound = Math.hypot(
      geometry.sourceRadius,
      geometry.detectorRadius,
    );
    expect(lowerBound).toBeGreaterThan(geometry.clearanceRadius);
    for (let degrees = -90; degrees <= 90; degrees += 1) {
      const state = acquisitionState(geometry, (degrees * Math.PI) / 180);
      const distance = Math.hypot(
        ...state.detector.map(
          (value, i) => value - metadata.landmarks.source[i]!,
        ),
      );
      expect(distance).toBeGreaterThanOrEqual(lowerBound - 1e-12);
      const separation = Math.hypot(
        ...state.source.map((value, i) => value - state.detector[i]!),
      );
      expect(separation).toBeCloseTo(
        metadata.landmarks.source_detector_distance,
        12,
      );
    }
  });
});

describe("the actual GLB orbital transform", () => {
  let rig: Group;
  beforeEach(async () => {
    const bytes = readFileSync(
      new URL("../../public/assets/carm/c-arm-rig.glb", import.meta.url),
    );
    const gltf = await new GLTFLoader().parseAsync(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
      "",
    );
    rig = gltf.scene;
  });

  it("maps every coloured arc to the actual detector while retaining source-clearance classification", () => {
    const parent = new Group();
    const articulated = attachAcquisitionOrbit(parent, rig, geometry);
    const source = new Vector3();
    const detector = new Vector3();
    const centre = new Vector3(...metadata.landmarks.source);
    const origin = new Vector3(...metadata.joints.orbital.pivot);
    const boundary = Math.acos(7 / 8);
    for (const [classification, start, end] of [
      ["allowed", boundary, Math.PI / 2],
      ["allowed", -Math.PI / 2, -boundary],
      ["excluded", -boundary, boundary],
      ["outside", Math.PI / 2, (3 * Math.PI) / 2],
    ] as const) {
      const samples = acquisitionDetectorArc(geometry, start, end);
      expect(samples[0]!.alpha).toBeCloseTo(start, 14);
      expect(samples.at(-1)!.alpha).toBeCloseTo(end, 14);
      for (const [index, { alpha, point }] of samples.entries()) {
        articulated.pivot.rotation.z = alpha;
        parent.updateMatrixWorld(true);
        articulated.source.getWorldPosition(source);
        articulated.detector.getWorldPosition(detector);
        closeVector(detector, point);
        expect(detector.distanceTo(origin)).toBeCloseTo(
          geometry.detectorRadius,
          12,
        );
        expect(detector.distanceTo(source)).toBeCloseTo(
          metadata.landmarks.source_detector_distance,
          12,
        );
        // Classify independently from actual world source position. Outside
        // arc endpoints and the exact clearance boundaries are shared by arcs.
        if (index > 0 && index < samples.length - 1) {
          const wrappedAngle = Math.atan2(
            source.x - origin.x,
            origin.y - source.y,
          );
          const actual =
            Math.abs(wrappedAngle) > Math.PI / 2
              ? "outside"
              : source.distanceTo(centre) < geometry.clearanceRadius
                ? "excluded"
                : "allowed";
          expect(actual).toBe(classification);
        }
      }
    }
  });

  it("preserves all 46 mesh buffers and moves only the original orbital components", () => {
    const parent = new Group();
    parent.add(rig);
    parent.updateMatrixWorld(true);
    const meshes: Mesh[] = [];
    rig.traverse((object) => {
      if (object instanceof Mesh) meshes.push(object);
    });
    expect(meshes).toHaveLength(46);
    const digest = (mesh: Mesh) => {
      const hash = createHash("sha256");
      for (const attribute of [
        mesh.geometry.getAttribute("position"),
        mesh.geometry.index,
      ]) {
        if (attribute)
          hash.update(
            new Uint8Array(
              attribute.array.buffer,
              attribute.array.byteOffset,
              attribute.array.byteLength,
            ),
          );
      }
      return hash.digest("hex");
    };
    const baseline = meshes.map((mesh) => ({
      mesh,
      geometry: mesh.geometry,
      digest: digest(mesh),
      orbital:
        metadata.components.find((component) => component.name === mesh.name)!
          .link === "orbital",
      // The first and last original vertices of every supplied mesh provide
      // independent world-space witnesses, including both stationary cables.
      witnesses: [0, mesh.geometry.getAttribute("position").count - 1].map(
        (i) => ({
          index: i,
          rest: new Vector3()
            .fromBufferAttribute(mesh.geometry.getAttribute("position"), i)
            .applyMatrix4(mesh.matrixWorld),
        }),
      ),
    }));
    const orbital = attachAcquisitionOrbit(parent, rig, geometry);
    for (const alpha of [
      0,
      Math.PI / 3,
      -Math.PI / 3,
      Math.PI / 2,
      -Math.PI / 2,
      geometry.boundaryAngle,
      -geometry.boundaryAngle,
    ]) {
      orbital.pivot.rotation.z = alpha;
      parent.updateMatrixWorld(true);
      const state = acquisitionState(geometry, alpha);
      closeVector(orbital.source.getWorldPosition(new Vector3()), state.source);
      closeVector(
        orbital.detector.getWorldPosition(new Vector3()),
        state.detector,
      );
      for (const { mesh, orbital: moves, witnesses } of baseline) {
        for (const { index, rest } of witnesses) {
          const expected = rest.clone();
          if (moves) {
            const x = rest.x - geometry.origin[0];
            const y = rest.y - geometry.origin[1];
            expected.x =
              geometry.origin[0] + Math.cos(alpha) * x - Math.sin(alpha) * y;
            expected.y =
              geometry.origin[1] + Math.sin(alpha) * x + Math.cos(alpha) * y;
          }
          const actual = new Vector3()
            .fromBufferAttribute(mesh.geometry.getAttribute("position"), index)
            .applyMatrix4(mesh.matrixWorld);
          closeVector(actual, expected.toArray());
        }
      }
    }
    expect(baseline.filter((item) => item.orbital)).toHaveLength(11);
    for (const item of baseline) {
      expect(item.mesh.geometry).toBe(item.geometry);
      expect(digest(item.mesh)).toBe(item.digest);
      expect(parent.getObjectByName(item.mesh.name)).toBe(item.mesh);
    }
  });
});
