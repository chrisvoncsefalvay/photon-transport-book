import * as THREE from "three";
import { recordedBuffer, type RecordedProjection } from "./reconstruction-data";

export function projectiveMatrix(view: RecordedProjection): THREE.Matrix4 {
  return new THREE.Matrix4()
    .fromArray([...view.uv_matrix_rows.flat(), 0, 0, 0, 1])
    .transpose();
}

export function acquisitionDistance(
  radius: number,
  aspect: number,
  fovDegrees: number,
): number {
  const halfVertical = THREE.MathUtils.degToRad(fovDegrees / 2);
  const halfFov = Math.min(
    halfVertical,
    Math.atan(Math.tan(halfVertical) * aspect),
  );
  return (1.08 * radius) / Math.sin(halfFov);
}

// Projective image display only. The RGB colours identify recorded views;
// they do not model visible light, X-ray transport, occlusion or material response.
const declarations = `
uniform sampler2D rigImage0;
uniform sampler2D rigImage1;
uniform sampler2D rigImage2;
uniform mat4 rigUv0;
uniform mat4 rigUv1;
uniform mat4 rigUv2;
uniform vec3 rigSource0;
uniform vec3 rigSource1;
uniform vec3 rigSource2;
uniform vec3 rigDistance;
uniform float rigEnabled;
float recordedLight(sampler2D image, mat4 mapping, vec3 source, float farDepth,
                    vec3 point, vec3 normal) {
  vec3 h = (mapping * vec4(point, 1.0)).xyz;
  if (h.z <= 0.0 || h.z > farDepth) return 0.0;
  vec2 uv = h.xy / h.z;
  if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 0.0;
  float facing = 0.18 + 0.82 * max(0.0, dot(normal, normalize(source - point)));
  return texture2D(image, uv).r * facing;
}
vec3 recordedIllumination(vec3 point, vec3 normal) {
  return vec3(
    recordedLight(rigImage0, rigUv0, rigSource0, rigDistance.x, point, normal),
    recordedLight(rigImage1, rigUv1, rigSource1, rigDistance.y, point, normal),
    recordedLight(rigImage2, rigUv2, rigSource2, rigDistance.z, point, normal)
  );
}
`;

export class ReconstructionProjections {
  private resources = new Set<{ dispose(): void }>();
  private groups: THREE.Group[] = [];
  private uniforms: Record<string, THREE.IUniform> = {
    rigEnabled: { value: 1 },
  };
  private textures: THREE.Texture[] = [];
  private geometries: THREE.BufferGeometry[] = [];
  private lineGeometries: THREE.BufferGeometry[] = [];
  private disposed = false;
  private rigVisible = true;
  private detectorMaterials: {
    material: THREE.MeshBasicMaterial;
    colour: THREE.Color;
  }[] = [];

  private constructor(readonly views: RecordedProjection[]) {}

  static async load(
    views: RecordedProjection[],
    base: string,
  ): Promise<ReconstructionProjections> {
    if (views.length !== 3)
      throw new Error("Three recorded projections are required.");
    const display = new ReconstructionProjections(views);
    try {
      // Keep decoding bounded. Both panes reuse these three verified textures.
      for (let index = 0; index < views.length; index++) {
        const view = views[index]!;
        if (
          view.uv_matrix_rows.length !== 3 ||
          view.uv_matrix_rows.some(
            (row) => row.length !== 4 || row.some((v) => !Number.isFinite(v)),
          ) ||
          view.source_detector_distance_mm <= 0 ||
          view.colour_rgb.some(
            (value, axis) => value !== Number(index === axis),
          )
        )
          throw new Error("Invalid recorded projection geometry.");
        const buffer = await recordedBuffer(base, {
          ...view.image,
          dtype: "png",
        });
        const url = URL.createObjectURL(
          new Blob([buffer], { type: "image/png" }),
        );
        const image = new Image();
        try {
          image.src = url;
          await image.decode();
          if (
            image.naturalWidth !== view.image.width ||
            image.naturalHeight !== view.image.height
          )
            throw new Error("Recorded radiograph dimensions changed.");
        } finally {
          URL.revokeObjectURL(url);
        }
        const texture = new THREE.Texture(image);
        // PNG top row is the last native detector row. Texture upload flips
        // exactly once so v=0 denotes the original bottom detector cell face.
        texture.flipY = true;
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.minFilter = texture.magFilter = THREE.LinearFilter;
        texture.generateMipmaps = false;
        texture.needsUpdate = true;
        display.textures.push(texture);
        display.resources.add(texture);
        const matrix = projectiveMatrix(view);
        display.uniforms[`rigImage${index}`] = { value: texture };
        display.uniforms[`rigUv${index}`] = { value: matrix };
        display.uniforms[`rigSource${index}`] = {
          value: new THREE.Vector3(...view.source_mm),
        };
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(view.detector_corners_mm.flat(), 3),
        );
        geometry.setAttribute(
          "uv",
          new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2),
        );
        geometry.setIndex([0, 1, 2, 0, 2, 3]);
        display.geometries.push(geometry);
        display.resources.add(geometry);
        const lines: number[] = [];
        view.detector_corners_mm.forEach((corner, i) => {
          lines.push(
            ...view.source_mm,
            ...corner,
            ...corner,
            ...view.detector_corners_mm[(i + 1) % 4]!,
          );
        });
        const lineGeometry = new THREE.BufferGeometry();
        lineGeometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(lines, 3),
        );
        display.lineGeometries.push(lineGeometry);
        display.resources.add(lineGeometry);
      }
      display.uniforms["rigDistance"] = {
        value: new THREE.Vector3(
          ...(views.map((v) => v.source_detector_distance_mm) as [
            number,
            number,
            number,
          ]),
        ),
      };
      return display;
    } catch (error) {
      display.dispose();
      throw error;
    }
  }

  addTo(scene: THREE.Scene): void {
    const group = new THREE.Group();
    group.visible = this.rigVisible;
    group.name = "Recorded acquisition frustums";
    this.views.forEach((view, index) => {
      const colour = new THREE.Color(...view.colour_rgb);
      const planeMaterial = new THREE.MeshBasicMaterial({
        map: this.textures[index]!,
        alphaMap: this.textures[index]!,
        color: colour,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.95,
        depthWrite: false,
        blending: THREE.NormalBlending,
      });
      this.detectorMaterials.push({ material: planeMaterial, colour });
      if (!this.uniforms["rigEnabled"]!.value)
        planeMaterial.color.set(0xffffff);
      const linesMaterial = new THREE.LineBasicMaterial({
        color: colour.clone().multiplyScalar(0.5),
        transparent: true,
        opacity: 0.6,
      });
      this.resources.add(planeMaterial);
      this.resources.add(linesMaterial);
      const detector = new THREE.Mesh(this.geometries[index], planeMaterial);
      detector.name = `${view.label} detector plane`;
      group.add(
        detector,
        new THREE.LineSegments(this.lineGeometries[index], linesMaterial),
      );
      const sourceGeometry = new THREE.BufferGeometry();
      sourceGeometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(view.source_mm, 3),
      );
      const sourceMaterial = new THREE.PointsMaterial({
        color: colour,
        size: 6,
        sizeAttenuation: false,
      });
      this.resources.add(sourceGeometry);
      this.resources.add(sourceMaterial);
      group.add(new THREE.Points(sourceGeometry, sourceMaterial));
    });
    this.groups.push(group);
    scene.add(group);
  }

  illuminate(material: THREE.MeshStandardMaterial): void {
    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, this.uniforms);
      shader.vertexShader =
        `varying vec3 rigPosition;\n${shader.vertexShader}`.replace(
          "#include <worldpos_vertex>",
          "#include <worldpos_vertex>\nrigPosition = (modelMatrix * vec4(transformed, 1.0)).xyz;",
        );
      shader.fragmentShader =
        `varying vec3 rigPosition;\n${declarations}\n${shader.fragmentShader}`.replace(
          "#include <opaque_fragment>",
          `
          vec3 rigNormal = inverseTransformDirection(normal, viewMatrix);
          vec3 lit = outgoingLight * 0.45 + 1.65 * recordedIllumination(rigPosition, rigNormal);
          outgoingLight = mix(outgoingLight, lit, rigEnabled);
          #include <opaque_fragment>
        `,
        );
    };
    material.customProgramCacheKey = () => "recorded-rgb-radiographs-v1";
  }

  setEnabled(enabled: boolean): void {
    this.uniforms["rigEnabled"]!.value = enabled ? 1 : 0;
    for (const { material, colour } of this.detectorMaterials) {
      material.color.copy(enabled ? colour : new THREE.Color(1, 1, 1));
    }
  }

  setRigVisible(visible: boolean): void {
    this.rigVisible = visible;
    for (const group of this.groups) group.visible = visible;
  }

  radius(centre: THREE.Vector3): number {
    return Math.max(
      ...this.views.flatMap((view) =>
        [view.source_mm, ...view.detector_corners_mm].map((point) =>
          new THREE.Vector3(...point).distanceTo(centre),
        ),
      ),
    );
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const resource of this.resources) resource.dispose();
    this.resources.clear();
    for (const group of this.groups) group.removeFromParent();
    this.groups = [];
    this.detectorMaterials = [];
  }
}
