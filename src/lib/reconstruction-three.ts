import * as THREE from "three";
import {
  ReconstructionProjections,
  acquisitionDistance,
} from "./reconstruction-projections";
import {
  dimensions,
  recordedBuffer,
  type Point,
  type RecordedScalar,
  type ReconstructionCase,
} from "./reconstruction-data";

// This shader displays a recorded scalar field. Opacity is a display transfer
// function, not attenuation, a detector model, or a reconstruction operation.
const vertexShader = `
out vec3 localPosition;
void main() {
  localPosition = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const fragmentShader = `
precision highp float;
precision highp sampler3D;
in vec3 localPosition;
uniform sampler3D field;
uniform vec3 localCamera;
uniform vec3 fieldSize;
uniform vec3 texel;
uniform vec3 tint;
uniform vec2 windowRange;
uniform float opacityScale;
uniform float clipPosition;
out vec4 outColour;
void main() {
  vec3 direction = normalize(localPosition - localCamera);
  // Keep reciprocal finite for rays parallel to a face.
  vec3 safeDirection = mix(direction, vec3(1e-7), lessThan(abs(direction), vec3(1e-7)));
  vec3 a = (vec3(-0.5) - localCamera) / safeDirection;
  vec3 b = (vec3(0.5, 0.5, clipPosition) - localCamera) / safeDirection;
  vec3 nearFace = min(a, b);
  vec3 farFace = max(a, b);
  float entry = max(max(nearFace.x, nearFace.y), max(nearFace.z, 0.0));
  float leave = min(min(farFace.x, farFace.y), farFace.z);
  if (leave <= entry) discard;
  float stepSize = (leave - entry) / 192.0;
  float physicalStep = length(direction * fieldSize) * stepSize;
  vec4 accumulated = vec4(0.0);
  for (int i = 0; i < 192; i++) {
    vec3 position = localCamera + direction * (entry + (float(i) + 0.5) * stepSize);
    float value = texture(field, position + 0.5).r;
    float level = clamp((value - windowRange.x) / (windowRange.y - windowRange.x), 0.0, 1.0);
    float alpha = 1.0 - exp(-level * opacityScale * physicalStep);
    float lighting = 0.6;
    if (alpha > 0.0001) {
      vec3 q = position + 0.5;
      vec3 gradient = vec3(
        texture(field, q + vec3(texel.x, 0.0, 0.0)).r - texture(field, q - vec3(texel.x, 0.0, 0.0)).r,
        texture(field, q + vec3(0.0, texel.y, 0.0)).r - texture(field, q - vec3(0.0, texel.y, 0.0)).r,
        texture(field, q + vec3(0.0, 0.0, texel.z)).r - texture(field, q - vec3(0.0, 0.0, texel.z)).r
      ) / (texel * fieldSize);
      if (length(gradient) > 1e-7) {
        vec3 lightDirection = normalize(localCamera * fieldSize + vec3(-0.4, -0.6, 0.8) * length(fieldSize));
        lighting = 0.32 + 0.68 * max(0.0, dot(-normalize(gradient), lightDirection));
      }
    }
    vec3 shade = tint * lighting;
    accumulated.rgb += (1.0 - accumulated.a) * alpha * shade;
    accumulated.a += (1.0 - accumulated.a) * alpha;
    if (accumulated.a > 0.995) break;
  }
  if (accumulated.a < 0.002) discard;
  outColour = linearToOutputTexel(vec4(accumulated.rgb / accumulated.a, accumulated.a));
}`;

type DisplayMode = "surface" | "volume";
interface Pane {
  scalar: RecordedScalar;
  scene: THREE.Scene;
  volume: THREE.Mesh<THREE.BoxGeometry, THREE.ShaderMaterial>;
  surface?: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;
}

export class ReconstructionRenderer {
  readonly volumeInterpolation: "trilinear" | "nearest neighbour";
  private renderer: THREE.WebGLRenderer;
  private camera = new THREE.PerspectiveCamera(31, 1, 0.1, 10_000);
  private panes: Pane[] = [];
  private centre: THREE.Vector3;
  private radius: number;
  private yaw = 0;
  private pitch = 0;
  private zoom = 1;
  private mode: DisplayMode = "surface";
  private opacity = 0.35;
  private clip = 1;
  private visible = true;
  private disposed = false;
  private frame = 0;
  private focused = "reconstructed";
  private resize: ResizeObserver;
  private abort = new AbortController();
  private generation = 0;
  private resources = new Set<{ dispose(): void }>();
  private textureCache = new Map<string, THREE.Data3DTexture>();
  private surfaceCache = new Map<string, THREE.BufferGeometry>();
  private paneCache = new Map<string, Promise<Pane | undefined>>();
  private initialDirection: Point;
  private projections?: ReconstructionProjections;
  private projectionPromise?: Promise<ReconstructionProjections>;
  private framing: "anatomy" | "acquisition" = "anatomy";

  constructor(
    private canvas: HTMLCanvasElement,
    private record: ReconstructionCase,
    private base: string,
    private fields: Map<string, Float32Array>,
    private onLayout: (labels: string[]) => void,
    private projectionBase = base,
  ) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
    });
    this.renderer.setClearColor(
      getComputedStyle(canvas).getPropertyValue("--paper-inset").trim() ||
        "#eeebe3",
    );
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.localClippingEnabled = true;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.volumeInterpolation = this.renderer.extensions.has(
      "OES_texture_float_linear",
    )
      ? "trilinear"
      : "nearest neighbour";
    const { grid } = record;
    const size = dimensions(grid);
    const local = new THREE.Vector3(
      ...(size.map((n, i) => ((n - 1) * grid.spacing_mm[i]!) / 2) as Point),
    );
    const o = grid.orientation;
    const rotation = new THREE.Matrix3().set(
      o[0]!,
      o[1]!,
      o[2]!,
      o[3]!,
      o[4]!,
      o[5]!,
      o[6]!,
      o[7]!,
      o[8]!,
    );
    this.centre = local
      .applyMatrix3(rotation)
      .add(new THREE.Vector3(...grid.origin_mm));
    this.radius =
      Math.hypot(...size.map((n, i) => n * grid.spacing_mm[i]!)) / 2;
    this.initialDirection = record.acquisition.display.camera_direction;
    this.reset();
    const { signal } = this.abort;
    let pointer: { id: number; x: number; y: number } | undefined;
    canvas.addEventListener(
      "pointerdown",
      (event) => {
        if (event.button !== 0) return;
        pointer = { id: event.pointerId, x: event.clientX, y: event.clientY };
        canvas.setPointerCapture(event.pointerId);
      },
      { signal },
    );
    canvas.addEventListener(
      "pointermove",
      (event) => {
        if (!pointer || pointer.id !== event.pointerId) return;
        this.yaw -= (event.clientX - pointer.x) * 0.008;
        this.pitch = Math.max(
          -1.45,
          Math.min(1.45, this.pitch + (event.clientY - pointer.y) * 0.008),
        );
        pointer.x = event.clientX;
        pointer.y = event.clientY;
        this.invalidate();
      },
      { signal },
    );
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"])
      canvas.addEventListener(
        name,
        () => {
          pointer = undefined;
        },
        { signal },
      );
    canvas.addEventListener(
      "keydown",
      (event) => {
        const actions: Record<string, () => void> = {
          ArrowLeft: () => {
            this.yaw -= 0.12;
          },
          ArrowRight: () => {
            this.yaw += 0.12;
          },
          ArrowUp: () => {
            this.pitch = Math.min(1.45, this.pitch + 0.12);
          },
          ArrowDown: () => {
            this.pitch = Math.max(-1.45, this.pitch - 0.12);
          },
          "+": () => this.setZoom(0.9),
          "=": () => this.setZoom(0.9),
          "-": () => this.setZoom(1.1),
          Home: () => this.reset(),
        };
        if (actions[event.key]) {
          event.preventDefault();
          actions[event.key]!();
          this.invalidate();
        }
      },
      { signal },
    );
    this.resize = new ResizeObserver(() => this.invalidate());
    this.resize.observe(canvas);
  }

  reset(): void {
    const direction = new THREE.Vector3(...this.initialDirection).normalize();
    this.yaw = Math.atan2(direction.x, direction.y);
    this.pitch = Math.asin(direction.z);
    this.zoom = this.record.id === "hap" ? 0.5 : 1;
    this.invalidate();
  }

  setZoom(factor: number): void {
    this.zoom = Math.max(0.35, Math.min(2.8, this.zoom * factor));
    this.invalidate();
  }

  setVisible(visible: boolean): void {
    this.visible = visible;
    if (!visible) {
      cancelAnimationFrame(this.frame);
      this.frame = 0;
    } else this.invalidate();
  }

  setOptions(options: {
    mode?: DisplayMode;
    opacity?: number;
    clip?: number;
    focus?: string;
    framing?: "anatomy" | "acquisition";
    illumination?: boolean;
    rig?: boolean;
  }): void {
    if (options.mode) this.mode = options.mode;
    if (options.opacity !== undefined) this.opacity = options.opacity;
    if (options.clip !== undefined) this.clip = options.clip;
    if (options.focus) this.focused = options.focus;
    if (options.framing && options.framing !== this.framing) {
      this.framing = options.framing;
      this.zoom = 1;
    }
    if (options.illumination !== undefined)
      this.projections?.setEnabled(options.illumination);
    if (options.rig !== undefined) this.projections?.setRigVisible(options.rig);
    this.invalidate();
  }

  async select(material: string): Promise<void> {
    const generation = ++this.generation;
    const selected = this.record.scalars.filter((s) => s.material === material);
    this.panes = [];
    this.invalidate();
    if (this.record.projections?.length && !this.projections) {
      this.projectionPromise ??= ReconstructionProjections.load(
        this.record.projections,
        this.projectionBase,
      );
      const projections = await this.projectionPromise;
      if (this.disposed) {
        projections.dispose();
        return;
      }
      this.projections = projections;
    }
    const panes = await Promise.all(
      selected.map((scalar) => {
        let pending = this.paneCache.get(scalar.id);
        if (!pending) {
          pending = this.createPane(scalar);
          this.paneCache.set(scalar.id, pending);
        }
        return pending;
      }),
    );
    if (this.disposed || generation !== this.generation) return;
    this.panes = panes.filter((pane): pane is Pane => Boolean(pane));
    this.invalidate();
  }

  private async createPane(scalar: RecordedScalar): Promise<Pane | undefined> {
    const material = scalar.material;
    const scene = new THREE.Scene();
    const tint = new THREE.Color(
      this.projections
        ? "#ffffff"
        : material === "water" || material === "pmma"
          ? "#b9cdd1"
          : "#dccba7",
    );
    const ambient = new THREE.HemisphereLight(0xe5efef, 0x303537, 2.4);
    const light = new THREE.DirectionalLight(0xffffff, 3.4);
    light.position.set(-this.radius, -this.radius, this.radius * 2);
    scene.add(ambient, light);
    const volume = this.volume(scalar, tint);
    scene.add(volume);
    const pane: Pane = { scalar, scene, volume };
    const mesh = this.record.meshes.find((m) => m.scalar_id === scalar.id);
    if (mesh) {
      let geometry = this.surfaceCache.get(mesh.id);
      if (!geometry) {
        const [positionBuffer, indexBuffer] = await Promise.all([
          recordedBuffer(this.base, mesh.positions),
          recordedBuffer(this.base, mesh.indices),
        ]);
        if (this.disposed) return;
        if (
          mesh.positions.dtype !== "float64-le" ||
          mesh.indices.dtype !== "uint32-le" ||
          positionBuffer.byteLength % 24 ||
          indexBuffer.byteLength % 12
        )
          throw new Error("Invalid recorded surface layout.");
        const p = new DataView(positionBuffer),
          ix = new DataView(indexBuffer);
        const positions = new Float32Array(positionBuffer.byteLength / 8);
        const indices = new Uint32Array(indexBuffer.byteLength / 4);
        for (let i = 0; i < positions.length; i++) {
          const value = p.getFloat64(i * 8, true);
          if (!Number.isFinite(value))
            throw new Error("Invalid surface vertex.");
          positions[i] = value;
        }
        for (let i = 0; i < indices.length; i++) {
          indices[i] = ix.getUint32(i * 4, true);
          if (indices[i]! >= positions.length / 3)
            throw new Error("Invalid surface index.");
        }
        geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
          "position",
          new THREE.BufferAttribute(positions, 3),
        );
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();
        this.surfaceCache.set(mesh.id, geometry);
        this.resources.add(geometry);
      }
      const surfaceMaterial = new THREE.MeshStandardMaterial({
        color: tint,
        roughness: 0.76,
        metalness: 0,
        side: THREE.DoubleSide,
      });
      this.projections?.illuminate(surfaceMaterial);
      this.resources.add(surfaceMaterial);
      pane.surface = new THREE.Mesh(geometry, surfaceMaterial);
      scene.add(pane.surface);
    }
    this.projections?.addTo(scene);
    return pane;
  }

  private volume(scalar: RecordedScalar, tint: THREE.Color): Pane["volume"] {
    const { grid } = this.record;
    const size = dimensions(grid);
    let texture = this.textureCache.get(scalar.id);
    if (!texture) {
      const values = this.fields.get(scalar.id);
      if (!values) throw new Error("Missing recorded scalar field.");
      texture = new THREE.Data3DTexture(values, ...size);
      texture.format = THREE.RedFormat;
      texture.type = THREE.FloatType;
      texture.minFilter = texture.magFilter =
        this.volumeInterpolation === "trilinear"
          ? THREE.LinearFilter
          : THREE.NearestFilter;
      texture.unpackAlignment = 1;
      texture.needsUpdate = true;
      this.textureCache.set(scalar.id, texture);
      this.resources.add(texture);
    }
    const extent = new THREE.Vector3(
      ...(size.map((n, i) => n * grid.spacing_mm[i]!) as Point),
    );
    const material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader,
      fragmentShader,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
      uniforms: {
        field: { value: texture },
        localCamera: { value: new THREE.Vector3() },
        fieldSize: { value: extent },
        texel: {
          value: new THREE.Vector3(...(size.map((n) => 1 / n) as Point)),
        },
        tint: { value: tint },
        windowRange: { value: new THREE.Vector2(...scalar.window) },
        opacityScale: { value: this.opacity },
        clipPosition: { value: this.clip - 0.5 },
      },
    });
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    this.resources.add(material);
    this.resources.add(geometry);
    const volume = new THREE.Mesh(geometry, material);
    const o = grid.orientation;
    volume.matrixAutoUpdate = false;
    volume.matrix.set(
      o[0]! * extent.x,
      o[1]! * extent.y,
      o[2]! * extent.z,
      this.centre.x,
      o[3]! * extent.x,
      o[4]! * extent.y,
      o[5]! * extent.z,
      this.centre.y,
      o[6]! * extent.x,
      o[7]! * extent.y,
      o[8]! * extent.z,
      this.centre.z,
      0,
      0,
      0,
      1,
    );
    volume.updateMatrixWorld(true);
    return volume;
  }

  private invalidate(): void {
    if (!this.visible || this.disposed || this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.draw();
    });
  }

  private draw(): void {
    if (this.disposed || !this.visible) return;
    const width = this.canvas.clientWidth,
      height = this.canvas.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    const paired =
      (width >= 640 || Boolean(this.projections)) && this.panes.length > 1;
    const stacked = paired && width < 640;
    this.canvas.dataset["layout"] = stacked ? "stacked" : "paired";
    const panes = paired
      ? this.panes
      : [
          this.panes.find((p) => p.scalar.role === this.focused) ??
            this.panes[0],
        ].filter((p): p is Pane => Boolean(p));
    this.onLayout(
      panes.map((p) =>
        p.scalar.role === "reference" ? "Assigned reference" : "Reconstructed",
      ),
    );
    const rigRadius = this.projections?.radius(this.centre) ?? this.radius;
    const aspect = stacked
      ? width / (height / panes.length)
      : width / Math.max(1, panes.length) / height;
    // Fit the actual rig in the narrower field of view. Only the camera moves.
    const distance =
      (this.framing === "acquisition"
        ? acquisitionDistance(rigRadius, aspect, this.camera.fov)
        : this.radius * 4.4) * this.zoom;
    this.camera.position
      .copy(this.centre)
      .add(
        new THREE.Vector3(
          Math.sin(this.yaw) * Math.cos(this.pitch),
          Math.cos(this.yaw) * Math.cos(this.pitch),
          Math.sin(this.pitch),
        ).multiplyScalar(distance),
      );
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(this.centre);
    this.camera.near = this.radius * 0.01;
    this.camera.far = distance + Math.max(this.radius * 4, rigRadius * 2);
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    this.renderer.setScissorTest(false);
    this.renderer.clear();
    this.renderer.setScissorTest(true);
    const o = this.record.grid.orientation;
    const normal = new THREE.Vector3(-o[2]!, -o[5]!, -o[8]!);
    const size = dimensions(this.record.grid);
    const cut = this.centre
      .clone()
      .addScaledVector(
        normal,
        -(this.clip - 0.5) * size[2] * this.record.grid.spacing_mm[2],
      );
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, cut);
    panes.forEach((pane, index) => {
      const volume = this.mode === "volume" || !pane.surface;
      pane.volume.visible = volume;
      if (pane.surface) {
        pane.surface.visible = !volume;
        pane.surface.material.clippingPlanes = this.clip < 1 ? [plane] : [];
      }
      const uniforms = pane.volume.material.uniforms;
      (uniforms["localCamera"]!.value as THREE.Vector3)
        .copy(this.camera.position)
        .applyMatrix4(
          new THREE.Matrix4().copy(pane.volume.matrixWorld).invert(),
        );
      uniforms["opacityScale"]!.value = this.opacity;
      uniforms["clipPosition"]!.value = this.clip - 0.5;
      const left = Math.round((index * width) / panes.length),
        right = Math.round(((index + 1) * width) / panes.length);
      if (stacked) {
        const top = Math.round(
          ((panes.length - index) * height) / panes.length,
        );
        const bottom = Math.round(
          ((panes.length - index - 1) * height) / panes.length,
        );
        this.renderer.setViewport(0, bottom, width, top - bottom);
        this.renderer.setScissor(0, bottom, width, top - bottom);
      } else {
        this.renderer.setViewport(left, 0, right - left, height);
        this.renderer.setScissor(left, 0, right - left, height);
      }
      this.renderer.render(pane.scene, this.camera);
    });
    this.renderer.setScissorTest(false);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.generation++;
    cancelAnimationFrame(this.frame);
    this.resize.disconnect();
    this.abort.abort();
    this.projections?.dispose();
    for (const resource of this.resources) resource.dispose();
    this.resources.clear();
    this.paneCache.clear();
    this.textureCache.clear();
    this.surfaceCache.clear();
    this.panes = [];
    this.renderer.dispose();
  }
}
