import * as T from "three";

// Each record: endpoints A/B, adjacent normals, boundary/sharp-edge flag.
// Transitions and front-facing boundaries/creases produce a contour.
export function conditionalLines(
  records: Float32Array,
  colour = 0x53676d,
): T.LineSegments {
  const count = records.length / 13;
  const positions = new Float32Array(count * 6),
    midpoints = new Float32Array(count * 6);
  const normal0 = new Float32Array(count * 6),
    normal1 = new Float32Array(count * 6),
    boundary = new Float32Array(count * 2);
  for (let i = 0; i < count; i++)
    for (let end = 0; end < 2; end++) {
      const offset = i * 13,
        out = i * 6 + end * 3;
      for (let k = 0; k < 3; k++) {
        positions[out + k] = records[offset + end * 3 + k]!;
        midpoints[out + k] =
          (records[offset + k]! + records[offset + 3 + k]!) / 2;
        normal0[out + k] = records[offset + 6 + k]!;
        normal1[out + k] = records[offset + 9 + k]!;
      }
      boundary[i * 2 + end] = records[offset + 12]!;
    }
  const geometry = new T.BufferGeometry();
  geometry.setAttribute("position", new T.BufferAttribute(positions, 3));
  geometry.setAttribute("edgeMidpoint", new T.BufferAttribute(midpoints, 3));
  geometry.setAttribute("normal0", new T.BufferAttribute(normal0, 3));
  geometry.setAttribute("normal1", new T.BufferAttribute(normal1, 3));
  geometry.setAttribute("boundary", new T.BufferAttribute(boundary, 1));
  const material = new T.ShaderMaterial({
    uniforms: { lineColour: { value: new T.Color(colour) } },
    vertexShader: `attribute vec3 edgeMidpoint;
   attribute vec3 normal0; attribute vec3 normal1; attribute float boundary;
   varying float visible;
   void main(){
    vec3 viewDirection=-(modelViewMatrix*vec4(edgeMidpoint,1.0)).xyz;
    float a=dot(normalMatrix*normal0,viewDirection);
    float b=dot(normalMatrix*normal1,viewDirection);
    visible=boundary>0.5 ? ((a>0.0 || b>0.0) ? 1.0:0.0) : ((a>0.0)!=(b>0.0) ? 1.0:0.0);
    gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);
   }`,
    fragmentShader: `uniform vec3 lineColour; varying float visible;
   void main(){if(visible<0.5)discard;gl_FragColor=vec4(lineColour,1.0);
   #include <tonemapping_fragment>
   #include <colorspace_fragment>
   }`,
    depthTest: true,
    depthWrite: false,
  });
  const lines = new T.LineSegments(geometry, material);
  lines.renderOrder = 2;
  return lines;
}

/** Build adjacency once for the original figure's already-triangulated body. */
export function bodyEdges(geometry: T.BufferGeometry): Float32Array {
  const position = geometry.getAttribute("position"),
    index = geometry.index;
  const edges = new Map<
    string,
    { a: T.Vector3; b: T.Vector3; normals: T.Vector3[] }
  >();
  const point = (i: number) =>
    new T.Vector3().fromBufferAttribute(position, index ? index.getX(i) : i);
  const key = (p: T.Vector3) =>
    `${Math.round(p.x * 1e6)},${Math.round(p.y * 1e6)},${Math.round(p.z * 1e6)}`;
  for (let i = 0; i < (index?.count ?? position.count); i += 3) {
    const v = [point(i), point(i + 1), point(i + 2)];
    const n = v[1]!
      .clone()
      .sub(v[0]!)
      .cross(v[2]!.clone().sub(v[0]!))
      .normalize();
    if (n.lengthSq() === 0) continue;
    for (let j = 0; j < 3; j++) {
      const a = v[j]!,
        b = v[(j + 1) % 3]!,
        ka = key(a),
        kb = key(b),
        id = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
      const edge = edges.get(id);
      if (edge) edge.normals.push(n);
      else edges.set(id, { a, b, normals: [n] });
    }
  }
  const output: number[] = [];
  for (const { a, b, normals } of edges.values()) {
    // This body is manifold apart from open boundaries; all normals retained in
    // the source. Pair extra incident faces if a non-manifold edge is encountered.
    for (let j = 1; j < Math.max(2, normals.length); j++)
      output.push(
        ...a.toArray(),
        ...b.toArray(),
        ...normals[0]!.toArray(),
        ...(normals[j] ?? normals[0]!).toArray(),
        normals.length === 1 ? 1 : 0,
      );
  }
  return new Float32Array(output);
}
