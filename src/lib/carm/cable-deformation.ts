import * as T from "three";

type VectorAttribute = T.BufferAttribute | T.InterleavedBufferAttribute;

interface Samples {
  rest: Float32Array;
  weights: Float64Array;
  gradients: Float64Array;
}

function snapshot(attribute: VectorAttribute): Float32Array {
  const values = new Float32Array(attribute.count * 3);
  for (let i = 0; i < attribute.count; i++) {
    values[i * 3] = attribute.getX(i);
    values[i * 3 + 1] = attribute.getY(i);
    values[i * 3 + 2] = attribute.getZ(i);
  }
  return values;
}

function samples(
  attribute: VectorAttribute,
  a: T.Vector3,
  b: T.Vector3,
): Samples {
  const rest = snapshot(attribute);
  const weights = new Float64Array(attribute.count);
  const gradients = new Float64Array(attribute.count * 3);
  for (let i = 0; i < attribute.count; i++) {
    const j = i * 3;
    const ax = rest[j]! - a.x;
    const ay = rest[j + 1]! - a.y;
    const az = rest[j + 2]! - a.z;
    const bx = rest[j]! - b.x;
    const by = rest[j + 1]! - b.y;
    const bz = rest[j + 2]! - b.z;
    const da = Math.hypot(ax, ay, az);
    const db = Math.hypot(bx, by, bz);
    const sum = da + db;
    weights[i] = da / sum;
    // The distance ratio has no unique derivative at an exact endpoint.
    // Its position is exact there; use the undeformed normal at that point.
    if (da > 1e-12 && db > 1e-12) {
      const denominator = sum * sum;
      gradients[j] = ((db * ax) / da - (da * bx) / db) / denominator;
      gradients[j + 1] = ((db * ay) / da - (da * by) / db) / denominator;
      gradients[j + 2] = ((db * az) / da - (da * bz) / db) / denominator;
    }
  }
  return { rest, weights, gradients };
}

function restore(attribute: VectorAttribute, values: Float32Array): void {
  for (let i = 0; i < attribute.count; i++) {
    const j = i * 3;
    attribute.setXYZ(i, values[j]!, values[j + 1]!, values[j + 2]!);
  }
  attribute.needsUpdate = true;
}

function move(
  attribute: VectorAttribute,
  source: Samples,
  a: T.Vector3,
  difference: T.Vector3,
): void {
  for (let i = 0; i < attribute.count; i++) {
    const j = i * 3;
    const w = source.weights[i]!;
    attribute.setXYZ(
      i,
      source.rest[j]! + a.x + w * difference.x,
      source.rest[j + 1]! + a.y + w * difference.y,
      source.rest[j + 2]! + a.z + w * difference.z,
    );
  }
  attribute.needsUpdate = true;
}

function transformNormals(
  attribute: VectorAttribute,
  rest: Float32Array,
  source: Samples,
  difference: T.Vector3,
): void {
  for (let i = 0; i < attribute.count; i++) {
    const j = i * 3;
    const gx = source.gradients[j]!;
    const gy = source.gradients[j + 1]!;
    const gz = source.gradients[j + 2]!;
    const nx = rest[j]!;
    const ny = rest[j + 1]!;
    const nz = rest[j + 2]!;
    // J = I + (delta1-delta0) outer grad(w). Sherman-Morrison gives
    // J^-T n = n - grad(w) dot(delta1-delta0,n)/(1+dot(grad(w),delta1-delta0)).
    const determinant =
      1 + gx * difference.x + gy * difference.y + gz * difference.z;
    const dot = difference.x * nx + difference.y * ny + difference.z * nz;
    let x: number, y: number, z: number;
    if (Math.abs(determinant) > 1e-10) {
      const scale = dot / determinant;
      x = nx - gx * scale;
      y = ny - gy * scale;
      z = nz - gz * scale;
    } else {
      // At a singular illustrative deformation no inverse exists. The
      // cofactor still supplies a finite area-normal direction if one exists.
      x = determinant * nx - gx * dot;
      y = determinant * ny - gy * dot;
      z = determinant * nz - gz * dot;
    }
    const length = Math.hypot(x, y, z);
    if (length > 1e-20) attribute.setXYZ(i, x / length, y / length, z / length);
    else attribute.setXYZ(i, nx, ny, nz);
  }
  attribute.needsUpdate = true;
}

/**
 * Deform the supplied cable itself; no replacement tube or procedural cable.
 *
 * Endpoints and deltas must use the shared mesh/contour rest coordinate system.
 * Each update starts from immutable source snapshots, preventing accumulated
 * drift. This is an illustrative endpoint warp, not a cable mechanics model.
 */
export function createCableDeformer(
  mesh: T.Mesh,
  contour: T.LineSegments,
  endpoint0: T.Vector3,
  endpoint1: T.Vector3,
): { update(delta0: T.Vector3, delta1: T.Vector3): void } {
  if (
    ![...endpoint0.toArray(), ...endpoint1.toArray()].every(Number.isFinite)
  ) {
    throw new Error("Cable endpoints must be finite.");
  }
  if (endpoint0.distanceToSquared(endpoint1) < 1e-16) {
    throw new Error("Cable endpoints must be distinct.");
  }
  const position = mesh.geometry.getAttribute("position");
  const normal = mesh.geometry.getAttribute("normal");
  const linePosition = contour.geometry.getAttribute("position");
  const midpoint = contour.geometry.getAttribute("edgeMidpoint");
  const normal0 = contour.geometry.getAttribute("normal0");
  const normal1 = contour.geometry.getAttribute("normal1");
  if (
    !position ||
    !normal ||
    !linePosition ||
    !midpoint ||
    !normal0 ||
    !normal1
  ) {
    throw new Error(
      "Cable mesh and contour require positions and their original normal attributes.",
    );
  }
  const vertices = samples(position, endpoint0, endpoint1);
  const lineVertices = samples(linePosition, endpoint0, endpoint1);
  const midpoints = samples(midpoint, endpoint0, endpoint1);
  const meshNormals = snapshot(normal);
  const lineNormals0 = snapshot(normal0);
  const lineNormals1 = snapshot(normal1);
  const difference = new T.Vector3();
  let last0: [number, number, number] | undefined;
  let last1: [number, number, number] | undefined;

  return {
    update(delta0, delta1) {
      if (![...delta0.toArray(), ...delta1.toArray()].every(Number.isFinite)) {
        throw new Error("Cable endpoint displacements must be finite.");
      }
      if (
        last0 &&
        last1 &&
        delta0.equals(new T.Vector3(...last0)) &&
        delta1.equals(new T.Vector3(...last1))
      )
        return;
      last0 = delta0.toArray();
      last1 = delta1.toArray();
      difference.subVectors(delta1, delta0);
      if (delta0.lengthSq() === 0 && delta1.lengthSq() === 0) {
        restore(position, vertices.rest);
        restore(normal, meshNormals);
        restore(linePosition, lineVertices.rest);
        restore(midpoint, midpoints.rest);
        restore(normal0, lineNormals0);
        restore(normal1, lineNormals1);
      } else {
        move(position, vertices, delta0, difference);
        move(linePosition, lineVertices, delta0, difference);
        // Shader midpoint is the actual centre of the deformed straight edge.
        // Neighbour normals use J at the corresponding original midpoint.
        for (let i = 0; i < linePosition.count; i += 2) {
          const x = (linePosition.getX(i) + linePosition.getX(i + 1)) / 2;
          const y = (linePosition.getY(i) + linePosition.getY(i + 1)) / 2;
          const z = (linePosition.getZ(i) + linePosition.getZ(i + 1)) / 2;
          midpoint.setXYZ(i, x, y, z);
          midpoint.setXYZ(i + 1, x, y, z);
        }
        midpoint.needsUpdate = true;
        if (difference.lengthSq() === 0) {
          // Uniform translations retain the source normals bit-for-bit.
          restore(normal, meshNormals);
          restore(normal0, lineNormals0);
          restore(normal1, lineNormals1);
        } else {
          transformNormals(normal, meshNormals, vertices, difference);
          transformNormals(normal0, lineNormals0, midpoints, difference);
          transformNormals(normal1, lineNormals1, midpoints, difference);
        }
      }
      mesh.geometry.computeBoundingBox();
      mesh.geometry.computeBoundingSphere();
      contour.geometry.computeBoundingBox();
      contour.geometry.computeBoundingSphere();
    },
  };
}
