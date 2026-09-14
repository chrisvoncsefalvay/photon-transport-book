import { describe, expect, it } from "vitest";
import * as THREE from "three";
import study from "../../public/generated/reconstruction-radiographs/study.json";
import type { RecordedProjection } from "../../src/lib/reconstruction-data";
import {
  acquisitionDistance,
  projectiveMatrix,
} from "../../src/lib/reconstruction-projections";

const views = study.cases.find((item) => item.id === "vertebrae")!
  .projection_rig!.views as unknown as RecordedProjection[];

describe("recorded source–detector display", () => {
  it("maps native pixel-centre rays to one bottom-origin texture coordinate at every depth", () => {
    for (const view of views) {
      const corners = view.detector_corners_mm.map(
        (point) => new THREE.Vector3(...point),
      );
      const source = new THREE.Vector3(...view.source_mm);
      const u = corners[1]!
        .clone()
        .sub(corners[0]!)
        .divideScalar(view.image.width);
      const v = corners[3]!
        .clone()
        .sub(corners[0]!)
        .divideScalar(view.image.height);
      for (const [row, column] of [
        [0, 0],
        [0, 95],
        [95, 0],
        [95, 95],
        [47, 48],
      ]) {
        const detector = corners[0]!
          .clone()
          .addScaledVector(u, column! + 0.5)
          .addScaledVector(v, row! + 0.5);
        for (const fraction of [0.2, 0.5, 0.8, 1]) {
          const point = source.clone().lerp(detector, fraction);
          const h = new THREE.Vector4(
            point.x,
            point.y,
            point.z,
            1,
          ).applyMatrix4(projectiveMatrix(view));
          expect(h.x / h.z).toBeCloseTo((column! + 0.5) / view.image.width, 12);
          expect(h.y / h.z).toBeCloseTo((row! + 0.5) / view.image.height, 12);
          expect(h.z).toBeCloseTo(
            fraction * view.source_detector_distance_mm,
            9,
          );
        }
      }
    }
  });

  it("preserves signed depth for rejecting points outside the finite source–detector segment", () => {
    for (const view of views) {
      const s = new THREE.Vector3(...view.source_mm);
      const centre = view.detector_corners_mm
        .reduce(
          (sum, p) => sum.add(new THREE.Vector3(...p)),
          new THREE.Vector3(),
        )
        .multiplyScalar(0.25);
      for (const fraction of [-0.1, 1.1]) {
        const point = s.clone().lerp(centre, fraction);
        const h = new THREE.Vector4(point.x, point.y, point.z, 1).applyMatrix4(
          projectiveMatrix(view),
        );
        expect(h.z <= 0 || h.z > view.source_detector_distance_mm).toBe(true);
      }
    }
  });

  it("keeps actual rig endpoints inside narrow and wide views through full rotations", () => {
    const points = views
      .flatMap((view) => [view.source_mm, ...view.detector_corners_mm])
      .map((p) => new THREE.Vector3(...p));
    const radius = Math.max(...points.map((p) => p.length()));
    for (const aspect of [0.35, 0.6, 1, 2]) {
      const camera = new THREE.PerspectiveCamera(31, aspect, 0.1, 10000);
      camera.up.set(0, 0, 1);
      const distance = acquisitionDistance(radius, aspect, camera.fov);
      for (let i = 0; i < 24; i++) {
        const yaw = (i * Math.PI) / 12,
          pitch = ((i % 3) - 1) * 0.7;
        camera.position
          .set(
            Math.cos(yaw) * Math.cos(pitch),
            Math.sin(yaw) * Math.cos(pitch),
            Math.sin(pitch),
          )
          .multiplyScalar(distance);
        camera.lookAt(0, 0, 0);
        camera.updateMatrixWorld();
        for (const p of points) {
          const q = p.clone().project(camera);
          expect(Math.max(Math.abs(q.x), Math.abs(q.y))).toBeLessThan(1);
        }
      }
    }
  });
});
