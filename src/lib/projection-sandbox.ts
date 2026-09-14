/** Closed-form display geometry for the appendix's declared ellipsoid phantom. */
export type SandboxVector = readonly [number, number, number];
export interface SandboxEllipsoid {
  readonly name: string;
  readonly centre: SandboxVector;
  readonly radii: SandboxVector;
  readonly coefficient: number;
}
export interface SandboxGeometry {
  sod: number;
  sdd: number;
  spotDiameter: number;
}
export type SandboxMode = "attenuation" | "surface";
export const SANDBOX_DEFAULT: Readonly<SandboxGeometry> = {
  sod: 500,
  sdd: 1000,
  spotDiameter: 0,
};
export const SANDBOX_DETECTOR_SIZE = 320;
export const SANDBOX_DEPTH_WINDOW = 4;
export const SANDBOX_SOURCE_SAMPLES = 32;
export const SANDBOX_PHANTOM: readonly SandboxEllipsoid[] = [
  {
    name: "Soft-tissue region",
    centre: [0, 0, 0],
    radii: [55, 75, 40],
    coefficient: 0.02,
  },
  {
    name: "Bone insert 1",
    centre: [-20, 8, 0],
    radii: [9, 28, 12],
    coefficient: 0.05,
  },
  {
    name: "Bone insert 2",
    centre: [20, 12, 3],
    radii: [9, 22, 10],
    coefficient: 0.05,
  },
  {
    name: "Air cavity",
    centre: [0, -35, -2],
    radii: [13, 10, 14],
    coefficient: 0,
  },
];

export function validateSandboxGeometry(geometry: SandboxGeometry): void {
  if (
    ![geometry.sod, geometry.sdd, geometry.spotDiameter].every(
      Number.isFinite,
    ) ||
    geometry.sod < 200 ||
    geometry.sod > 600 ||
    geometry.sdd < 800 ||
    geometry.sdd > 1400 ||
    geometry.spotDiameter < 0 ||
    geometry.spotDiameter > 10 ||
    geometry.sdd <= geometry.sod + 40
  ) {
    throw new RangeError(
      "Sandbox geometry requires SOD 200–600 mm, SDD 800–1400 mm and a 0–10 mm source diameter, with the detector beyond the phantom.",
    );
  }
}

/** The returned roots are clipped to the finite source-to-detector segment. */
export function ellipsoidInterval(
  source: SandboxVector,
  detector: SandboxVector,
  ellipsoid: SandboxEllipsoid,
): readonly [number, number] | null {
  let a = 0,
    b = 0,
    c = -1;
  for (let axis = 0; axis < 3; axis += 1) {
    const radius = ellipsoid.radii[axis]!;
    if (!(radius > 0) || !Number.isFinite(radius))
      throw new RangeError("Ellipsoid radii must be finite and positive");
    const origin = (source[axis]! - ellipsoid.centre[axis]!) / radius;
    const direction = (detector[axis]! - source[axis]!) / radius;
    a += direction * direction;
    b += 2 * origin * direction;
    c += origin * origin;
  }
  if (a === 0) return null;
  const discriminant = b * b - 4 * a * c;
  if (discriminant < 0) return null;
  const root = Math.sqrt(discriminant);
  // Stable quadratic roots avoid subtracting nearly equal large values.
  const q = -0.5 * (b + (b < 0 ? -root : root));
  const roots = q === 0 ? [-b / (2 * a), -b / (2 * a)] : [q / a, c / q];
  const enter = Math.max(0, Math.min(roots[0]!, roots[1]!));
  const exit = Math.min(1, Math.max(roots[0]!, roots[1]!));
  return exit > enter ? [enter, exit] : null;
}

export function ellipsoidChord(
  source: SandboxVector,
  detector: SandboxVector,
  ellipsoid: SandboxEllipsoid,
): number {
  const interval = ellipsoidInterval(source, detector, ellipsoid);
  return interval
    ? (interval[1] - interval[0]) *
        Math.hypot(
          detector[0] - source[0],
          detector[1] - source[1],
          detector[2] - source[2],
        )
    : 0;
}

export function sandboxOpticalDepth(
  source: SandboxVector,
  detector: SandboxVector,
): number {
  const background = SANDBOX_PHANTOM[0]!;
  let depth =
    background.coefficient * ellipsoidChord(source, detector, background);
  // Disjoint, fully contained inserts replace the background material. Adding
  // their full coefficients would count the outer material a second time.
  for (const insert of SANDBOX_PHANTOM.slice(1))
    depth +=
      (insert.coefficient - background.coefficient) *
      ellipsoidChord(source, detector, insert);
  if (depth < -1e-10)
    throw new Error(
      "The declared replacement geometry produced negative attenuation",
    );
  return Math.max(0, depth);
}

/** Equal-area radial nodes with opposite pairs: finite, deterministic quadrature. */
export function sandboxSourceOffsets(
  diameter: number,
  sampleCount = SANDBOX_SOURCE_SAMPLES,
): SandboxVector[] {
  if (
    !Number.isFinite(diameter) ||
    diameter < 0 ||
    !Number.isInteger(sampleCount) ||
    sampleCount < 2 ||
    sampleCount % 2 !== 0
  )
    throw new RangeError(
      "Source quadrature requires a nonnegative diameter and an even sample count",
    );
  if (diameter === 0) return [[0, 0, 0]];
  const offsets: SandboxVector[] = [];
  const pairs = sampleCount / 2;
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < pairs; i += 1) {
    const radius = (diameter / 2) * Math.sqrt((i + 0.5) / pairs);
    const angle = i * goldenAngle;
    const x = radius * Math.cos(angle),
      y = radius * Math.sin(angle);
    offsets.push([x, y, 0], [-x, -y, 0]);
  }
  return offsets;
}

export function sandboxTransmission(
  geometry: SandboxGeometry,
  detectorX: number,
  detectorY: number,
  sampleCount = SANDBOX_SOURCE_SAMPLES,
): number {
  validateSandboxGeometry(geometry);
  const target: SandboxVector = [
    detectorX,
    detectorY,
    geometry.sdd - geometry.sod,
  ];
  const offsets = sandboxSourceOffsets(geometry.spotDiameter, sampleCount);
  let transmission = 0;
  for (const offset of offsets)
    transmission += Math.exp(
      -sandboxOpticalDepth([offset[0], offset[1], -geometry.sod], target),
    );
  return transmission / offsets.length;
}

function surfaceLevel(source: SandboxVector, target: SandboxVector): number {
  const ellipsoid = SANDBOX_PHANTOM[0]!;
  const interval = ellipsoidInterval(source, target, ellipsoid);
  if (!interval) return 0;
  const normal = ellipsoid.radii.map(
    (radius, axis) =>
      (source[axis]! +
        interval[0] * (target[axis]! - source[axis]!) -
        ellipsoid.centre[axis]!) /
      (radius * radius),
  );
  const length = Math.hypot(...normal);
  const lighting = Math.max(
    0,
    (-normal[0]! + normal[1]! - 2 * normal[2]!) / (Math.sqrt(6) * length),
  );
  return Math.round(210 * (0.25 + 0.75 * lighting));
}

export function sandboxImage(
  geometry: SandboxGeometry,
  resolution = 80,
  mode: SandboxMode = "attenuation",
): Uint8ClampedArray {
  validateSandboxGeometry(geometry);
  if (!Number.isInteger(resolution) || resolution < 8 || resolution > 256)
    throw new RangeError(
      "Sandbox image resolution must be an integer from 8 to 256",
    );
  const rgba = new Uint8ClampedArray(resolution * resolution * 4);
  const offsets = sandboxSourceOffsets(
    mode === "surface" ? 0 : geometry.spotDiameter,
  );
  const sources = offsets.map((offset): SandboxVector => [
    offset[0],
    offset[1],
    -geometry.sod,
  ]);
  for (let row = 0; row < resolution; row += 1) {
    for (let column = 0; column < resolution; column += 1) {
      const target: SandboxVector = [
        ((column + 0.5) / resolution - 0.5) * SANDBOX_DETECTOR_SIZE,
        (0.5 - (row + 0.5) / resolution) * SANDBOX_DETECTOR_SIZE,
        geometry.sdd - geometry.sod,
      ];
      let level: number;
      if (mode === "surface") level = surfaceLevel(sources[0]!, target);
      else {
        let total = 0;
        for (const source of sources)
          total += Math.exp(-sandboxOpticalDepth(source, target));
        const transmission = total / sources.length;
        level = Math.round(
          255 * Math.min(1, -Math.log(transmission) / SANDBOX_DEPTH_WINDOW),
        );
      }
      const offset = (row * resolution + column) * 4;
      rgba[offset] = level;
      rgba[offset + 1] = level;
      rgba[offset + 2] = level;
      rgba[offset + 3] = 255;
    }
  }
  return rgba;
}

/** Compact static image, generated from the same exact geometry as the controls. */
export function sandboxSvg(
  geometry: SandboxGeometry,
  resolution = 80,
  mode: SandboxMode = "attenuation",
): string {
  const pixels = sandboxImage(geometry, resolution, mode);
  const paths = new Map<number, string[]>();
  for (let row = 0; row < resolution; row += 1) {
    let column = 0;
    while (column < resolution) {
      const level = pixels[(row * resolution + column) * 4]!;
      let end = column + 1;
      while (end < resolution && pixels[(row * resolution + end) * 4] === level)
        end += 1;
      if (level) {
        const parts = paths.get(level) ?? [];
        parts.push(`M${column} ${row}h${end - column}v1h-${end - column}Z`);
        paths.set(level, parts);
      }
      column = end;
    }
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${resolution} ${resolution}" shape-rendering="crispEdges"><rect width="${resolution}" height="${resolution}" fill="#000"/>${[...paths].map(([level, parts]) => `<path fill="rgb(${level},${level},${level})" d="${parts.join("")}"/>`).join("")}</svg>`;
}

export function sandboxDiagramProjection(geometry: SandboxGeometry) {
  const project = (point: SandboxVector) =>
    [
      0.75 * point[0] + 0.38 * point[2],
      -0.85 * point[1] + 0.18 * point[2],
    ] as const;
  const half = SANDBOX_DETECTOR_SIZE / 2;
  const corners: SandboxVector[] = [
    [-half, -half, geometry.sdd - geometry.sod],
    [-half, half, geometry.sdd - geometry.sod],
    [half, half, geometry.sdd - geometry.sod],
    [half, -half, geometry.sdd - geometry.sod],
  ];
  const bounds = [...corners, [0, 0, -geometry.sod] as const].map(project);
  const left = Math.min(...bounds.map((p) => p[0])),
    right = Math.max(...bounds.map((p) => p[0]));
  const top = Math.min(...bounds.map((p) => p[1])),
    bottom = Math.max(...bounds.map((p) => p[1]));
  const scale = Math.min(360 / (right - left), 235 / (bottom - top));
  return (point: SandboxVector) => {
    const projected = project(point);
    return [
      210 + (projected[0] - (left + right) / 2) * scale,
      160 + (projected[1] - (top + bottom) / 2) * scale,
    ] as const;
  };
}

export function sandboxEllipsoidOutline(
  geometry: SandboxGeometry,
  shape: SandboxEllipsoid,
  plane: readonly [number, number],
): string {
  const project = sandboxDiagramProjection(geometry);
  return Array.from({ length: 65 }, (_, i) => {
    const point = [...shape.centre];
    point[plane[0]]! +=
      shape.radii[plane[0]]! * Math.cos((i / 64) * 2 * Math.PI);
    point[plane[1]]! +=
      shape.radii[plane[1]]! * Math.sin((i / 64) * 2 * Math.PI);
    return project([point[0]!, point[1]!, point[2]!]).join(",");
  }).join(" ");
}

export function registerProjectionSandbox() {
  if (customElements.get("projection-sandbox")) return;
  class ProjectionSandbox extends HTMLElement {
    private observer: IntersectionObserver | undefined;
    private teardown: (() => void) | undefined;
    private startup: AbortController | undefined;
    private generation = 0;
    private starting = false;
    connectedCallback() {
      const start = () => {
        if (this.starting || !this.isConnected) return;
        this.starting = true;
        const generation = this.generation;
        this.startup = new AbortController();
        void mountSandbox(this, this.startup.signal).then((teardown) => {
          if (!this.isConnected || generation !== this.generation) teardown();
          else this.teardown = teardown;
        });
      };
      if (!("IntersectionObserver" in window)) start();
      else {
        this.observer = new IntersectionObserver(
          (entries) => {
            if (entries.some((entry) => entry.isIntersecting)) {
              this.observer?.disconnect();
              start();
            }
          },
          { rootMargin: "200px" },
        );
        this.observer.observe(this);
      }
    }
    disconnectedCallback() {
      this.generation += 1;
      this.observer?.disconnect();
      this.startup?.abort();
      this.teardown?.();
      this.observer = undefined;
      this.teardown = undefined;
      this.startup = undefined;
      this.starting = false;
    }
  }
  customElements.define("projection-sandbox", ProjectionSandbox);
}

async function mountSandbox(
  root: HTMLElement,
  startupSignal: AbortSignal,
): Promise<() => void> {
  const events = new AbortController();
  const imageCanvas = root.querySelector<HTMLCanvasElement>(
    "[data-sandbox-image]",
  );
  const sceneCanvas = root.querySelector<HTMLCanvasElement>(
    "[data-sandbox-scene]",
  );
  const imageFallback = root.querySelector<HTMLElement>(
    "[data-image-fallback]",
  );
  const sceneFallback = root.querySelector<SVGElement>("[data-scene-fallback]");
  const context = imageCanvas?.getContext("2d");
  const input = (name: string) =>
    root.querySelector<HTMLInputElement>(`[data-sandbox-input="${name}"]`);
  const sod = input("sod"),
    sdd = input("sdd"),
    spot = input("spot");
  if (!context || !imageCanvas || !sceneCanvas || !sod || !sdd || !spot)
    return () => events.abort();
  const setText = (selector: string, value: string) => {
    const node = root.querySelector(selector);
    if (node) node.textContent = value;
  };
  const currentGeometry = (): SandboxGeometry => ({
    sod: Number(sod.value),
    sdd: Number(sdd.value),
    spotDiameter: Number(spot.value),
  });
  let mode: SandboxMode =
    root.querySelector<HTMLInputElement>("[data-sandbox-mode]:checked")
      ?.value === "surface"
      ? "surface"
      : "attenuation";
  let dirty = true;
  let frame: number | undefined;
  let scenePaint: (() => void) | undefined;
  let sceneUpdate: (() => void) | undefined;
  let sceneDispose: (() => void) | undefined;
  let disposed = false;
  const updateFallback = (geometry: SandboxGeometry) => {
    if (!sceneFallback) return;
    const project = sandboxDiagramProjection(geometry);
    const sourcePoint = project([0, 0, -geometry.sod]);
    const half = SANDBOX_DETECTOR_SIZE / 2;
    const corners = [
      [-half, -half, geometry.sdd - geometry.sod],
      [-half, half, geometry.sdd - geometry.sod],
      [half, half, geometry.sdd - geometry.sod],
      [half, -half, geometry.sdd - geometry.sod],
    ] as const;
    sceneFallback.querySelectorAll(".sandbox-ray").forEach((line, index) => {
      const end = project(corners[index]!);
      line.setAttribute("x1", String(sourcePoint[0]));
      line.setAttribute("y1", String(sourcePoint[1]));
      line.setAttribute("x2", String(end[0]));
      line.setAttribute("y2", String(end[1]));
    });
    sceneFallback
      .querySelector(".sandbox-detector-plane")
      ?.setAttribute(
        "points",
        corners.map((point) => project(point).join(",")).join(" "),
      );
    sceneFallback
      .querySelectorAll(".sandbox-ellipsoid")
      .forEach((group, index) => {
        const planes = [
          [0, 1],
          [0, 2],
          [1, 2],
        ] as const;
        group
          .querySelectorAll("polyline")
          .forEach((line, plane) =>
            line.setAttribute(
              "points",
              sandboxEllipsoidOutline(
                geometry,
                SANDBOX_PHANTOM[index]!,
                planes[plane]!,
              ),
            ),
          );
      });
    const sourceMarker = sceneFallback.querySelector(".sandbox-source");
    sourceMarker?.setAttribute("cx", String(sourcePoint[0]));
    sourceMarker?.setAttribute("cy", String(sourcePoint[1]));
    const sourceLabel = sceneFallback.querySelector(
      "[data-fallback-source-label]",
    );
    sourceLabel?.setAttribute("x", String(sourcePoint[0]));
    sourceLabel?.setAttribute("y", String(sourcePoint[1] - 15));
    const phantomLabel = sceneFallback.querySelector(
      "[data-fallback-phantom-label]",
    );
    phantomLabel?.setAttribute("x", String(project([0, 0, 0])[0]));
    phantomLabel?.setAttribute("y", String(project([0, 95, 0])[1] - 12));
    const description = sceneFallback.querySelector("desc");
    if (description)
      description.textContent = `The phantom centre is ${geometry.sod} mm from the source. The detector is ${geometry.sdd} mm from the source and measures 320 by 320 mm. Three contained, disjoint regions replace the surrounding material.`;
  };
  const paint = () => {
    frame = undefined;
    if (disposed || !root.isConnected) return;
    if (dirty) {
      const geometry = currentGeometry();
      updateFallback(geometry);
      const pixels = sandboxImage(geometry, imageCanvas.width, mode);
      const image = context.createImageData(
        imageCanvas.width,
        imageCanvas.height,
      );
      image.data.set(pixels);
      context.putImageData(image, 0, 0);
      imageCanvas.hidden = false;
      if (imageFallback) imageFallback.hidden = true;
      setText("[data-sod-value]", `${geometry.sod} mm`);
      setText("[data-sdd-value]", `${geometry.sdd} mm`);
      setText("[data-spot-value]", `${geometry.spotDiameter.toFixed(1)} mm`);
      setText("[data-magnification]", (geometry.sdd / geometry.sod).toFixed(2));
      setText(
        "[data-image-title]",
        mode === "attenuation" ? "Attenuation image" : "Opaque surface image",
      );
      setText(
        "[data-image-domain]",
        mode === "attenuation"
          ? "Fixed display window: −log(mean T) from 0 to 4, where greater attenuation is brighter."
          : "Nearest visible surface, shaded from the central source viewpoint. The internal inserts are occluded.",
      );
      imageCanvas.setAttribute(
        "aria-label",
        `${mode === "attenuation" ? "Attenuation" : "Opaque surface"} image of the geometric ellipsoid phantom. SOD ${geometry.sod} mm, SDD ${geometry.sdd} mm, magnification ${(geometry.sdd / geometry.sod).toFixed(2)}.`,
      );
      spot.disabled = mode === "surface";
      sceneUpdate?.();
      dirty = false;
    }
    scenePaint?.();
  };
  const requestPaint = () => {
    if (frame === undefined && !disposed) frame = requestAnimationFrame(paint);
  };
  for (const element of [sod, sdd, spot]) {
    element.disabled = false;
    element.addEventListener(
      "input",
      () => {
        dirty = true;
        requestPaint();
      },
      { signal: events.signal },
    );
  }
  root
    .querySelectorAll<HTMLInputElement>("[data-sandbox-mode]")
    .forEach((radio) => {
      radio.disabled = false;
      radio.addEventListener(
        "change",
        () => {
          if (radio.checked) {
            mode = radio.value === "surface" ? "surface" : "attenuation";
            dirty = true;
            requestPaint();
          }
        },
        { signal: events.signal },
      );
    });
  root
    .querySelector<HTMLButtonElement>("[data-reset-geometry]")
    ?.addEventListener(
      "click",
      () => {
        sod.value = "500";
        sdd.value = "1000";
        spot.value = "0";
        dirty = true;
        requestPaint();
      },
      { signal: events.signal },
    );
  const geometryReset = root.querySelector<HTMLButtonElement>(
    "[data-reset-geometry]",
  );
  if (geometryReset) geometryReset.disabled = false;
  paint();
  const teardown = () => {
    if (disposed) return;
    disposed = true;
    events.abort();
    if (frame !== undefined) cancelAnimationFrame(frame);
    sceneDispose?.();
    sceneCanvas.hidden = true;
    imageCanvas.hidden = true;
    if (imageFallback) imageFallback.hidden = false;
    sceneFallback?.removeAttribute("hidden");
    root
      .querySelectorAll<HTMLInputElement | HTMLButtonElement>("input,button")
      .forEach((control) => {
        control.disabled = true;
      });
  };
  startupSignal.addEventListener("abort", teardown, { once: true });
  try {
    const [T, { OrbitControls }] = await Promise.all([
      import("three"),
      import("three/examples/jsm/controls/OrbitControls.js"),
    ]);
    if (disposed || !root.isConnected) return teardown;
    const renderer = new T.WebGLRenderer({
      canvas: sceneCanvas,
      antialias: true,
      alpha: false,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0xeeeae2, 1);
    const scene = new T.Scene();
    const camera = new T.PerspectiveCamera(32, 1, 1, 7000);
    const controls = new OrbitControls(camera, sceneCanvas);
    controls.enableDamping = false;
    controls.enablePan = false;
    controls.minDistance = 180;
    controls.maxDistance = 4000;
    const disposables: { dispose(): void }[] = [];
    const resize = new ResizeObserver(() => {
      const bounds = sceneCanvas.parentElement!.getBoundingClientRect();
      renderer.setSize(
        Math.max(1, bounds.width),
        Math.max(1, bounds.height),
        false,
      );
      camera.aspect = bounds.width / Math.max(1, bounds.height);
      camera.updateProjectionMatrix();
      requestPaint();
    });
    sceneDispose = () => {
      resize.disconnect();
      controls.dispose();
      disposables.forEach((resource) => resource.dispose());
      renderer.dispose();
      renderer.forceContextLoss();
      sceneCanvas.hidden = true;
      sceneCanvas.replaceWith(sceneCanvas.cloneNode(true));
    };
    scene.add(new T.AmbientLight(0xffffff, 2.3));
    const light = new T.DirectionalLight(0xffffff, 2.5);
    light.position.set(-500, 700, -600);
    scene.add(light);
    const phantomMeshes: import("three").Mesh[] = [];
    SANDBOX_PHANTOM.forEach((ellipsoid, index) => {
      const shape = new T.SphereGeometry(1, 40, 24);
      const material = new T.MeshLambertMaterial({
        color: index === 0 ? 0xb6afa2 : index === 3 ? 0x718990 : 0xe7e0ce,
        transparent: true,
        opacity: index === 0 ? 0.15 : 0.8,
        depthWrite: false,
      });
      const mesh = new T.Mesh(shape, material);
      mesh.position.set(...ellipsoid.centre);
      mesh.scale.set(...ellipsoid.radii);
      scene.add(mesh);
      phantomMeshes.push(mesh);
      disposables.push(shape, material);
      for (const plane of [
        [0, 1],
        [0, 2],
        [1, 2],
      ] as const) {
        const points = Array.from({ length: 65 }, (_, i) => {
          const point = [...ellipsoid.centre];
          point[plane[0]]! +=
            ellipsoid.radii[plane[0]]! * Math.cos((i / 64) * Math.PI * 2);
          point[plane[1]]! +=
            ellipsoid.radii[plane[1]]! * Math.sin((i / 64) * Math.PI * 2);
          return new T.Vector3(point[0], point[1], point[2]);
        });
        const geometry = new T.BufferGeometry().setFromPoints(points);
        const lineMaterial = new T.LineBasicMaterial({
          color: index === 0 ? 0x6f6c64 : 0x38565c,
        });
        scene.add(new T.Line(geometry, lineMaterial));
        disposables.push(geometry, lineMaterial);
      }
    });
    const sourceGeometry = new T.SphereGeometry(7, 16, 12),
      sourceMaterial = new T.MeshBasicMaterial({ color: 0x984632 });
    const source = new T.Mesh(sourceGeometry, sourceMaterial);
    scene.add(source);
    disposables.push(sourceGeometry, sourceMaterial);
    const spotGeometry = new T.CircleGeometry(1, 48),
      spotMaterial = new T.MeshBasicMaterial({
        color: 0x984632,
        side: T.DoubleSide,
      });
    const spotDisc = new T.Mesh(spotGeometry, spotMaterial);
    scene.add(spotDisc);
    disposables.push(spotGeometry, spotMaterial);
    const texture = new T.CanvasTexture(imageCanvas);
    texture.colorSpace = T.SRGBColorSpace;
    texture.magFilter = T.NearestFilter;
    texture.minFilter = T.NearestFilter;
    const detectorGeometry = new T.PlaneGeometry(
        SANDBOX_DETECTOR_SIZE,
        SANDBOX_DETECTOR_SIZE,
      ),
      detectorMaterial = new T.MeshBasicMaterial({
        map: texture,
        side: T.DoubleSide,
      });
    const detector = new T.Mesh(detectorGeometry, detectorMaterial);
    scene.add(detector);
    disposables.push(texture, detectorGeometry, detectorMaterial);
    const rayCoordinates = new Float32Array(30);
    const rayGeometry = new T.BufferGeometry();
    rayGeometry.setAttribute(
      "position",
      new T.BufferAttribute(rayCoordinates, 3),
    );
    const rayMaterial = new T.LineBasicMaterial({
      color: 0x827d70,
      transparent: true,
      opacity: 0.45,
    });
    scene.add(new T.LineSegments(rayGeometry, rayMaterial));
    disposables.push(rayGeometry, rayMaterial);
    const makeLabel = (text: string) => {
      const labelCanvas = document.createElement("canvas");
      labelCanvas.width = 256;
      labelCanvas.height = 48;
      const labelContext = labelCanvas.getContext("2d");
      if (!labelContext) throw new Error("Label canvas unavailable");
      labelContext.font = "26px sans-serif";
      labelContext.textAlign = "center";
      labelContext.textBaseline = "middle";
      labelContext.fillStyle = "#262722";
      labelContext.fillText(text, 128, 24);
      const map = new T.CanvasTexture(labelCanvas);
      map.colorSpace = T.SRGBColorSpace;
      const material = new T.SpriteMaterial({
        map,
        transparent: true,
        depthTest: false,
        sizeAttenuation: false,
      });
      const sprite = new T.Sprite(material);
      sprite.scale.set((0.032 * 256) / 48, 0.032, 1);
      sprite.renderOrder = 10;
      scene.add(sprite);
      disposables.push(map, material);
      return sprite;
    };
    const sourceLabel = makeLabel("Source"),
      phantomLabel = makeLabel("Phantom"),
      detectorLabel = makeLabel("Detector");
    phantomLabel.position.set(0, 110, 0);
    const resetView = () => {
      const geometry = currentGeometry();
      const middle = geometry.sdd / 2 - geometry.sod;
      camera.position.set(
        -geometry.sdd * 0.85,
        geometry.sdd * 0.55,
        middle + geometry.sdd * 0.95,
      );
      controls.target.set(0, 0, middle);
      controls.update();
      requestPaint();
    };
    sceneUpdate = () => {
      const geometry = currentGeometry();
      source.position.set(0, 0, -geometry.sod);
      source.visible = geometry.spotDiameter === 0;
      spotDisc.position.copy(source.position);
      spotDisc.scale.setScalar(geometry.spotDiameter / 2);
      spotDisc.visible = geometry.spotDiameter > 0;
      sourceLabel.position.set(0, 40, -geometry.sod);
      detector.position.z = geometry.sdd - geometry.sod;
      detectorLabel.position.set(0, 195, detector.position.z);
      const half = SANDBOX_DETECTOR_SIZE / 2;
      const ends = [
        [-half, -half],
        [-half, half],
        [half, half],
        [half, -half],
        [0, 0],
      ];
      ends.forEach((end, index) => {
        rayCoordinates.set(
          [0, 0, -geometry.sod, end[0]!, end[1]!, detector.position.z],
          index * 6,
        );
      });
      rayGeometry.attributes.position!.needsUpdate = true;
      rayGeometry.computeBoundingSphere();
      texture.needsUpdate = true;
      phantomMeshes.forEach((mesh, index) => {
        const material = mesh.material as import("three").MeshLambertMaterial;
        material.opacity = mode === "surface" ? 1 : index === 0 ? 0.15 : 0.8;
        material.depthWrite = mode === "surface";
      });
      const newMiddle = geometry.sdd / 2 - geometry.sod;
      camera.position.z += newMiddle - controls.target.z;
      controls.target.z = newMiddle;
      controls.update();
    };
    scenePaint = () => renderer.render(scene, camera);
    controls.addEventListener("change", requestPaint);
    resize.observe(sceneCanvas.parentElement!);
    const orbit = (azimuth: number, polar: number, zoom = 1) => {
      const sphere = new T.Spherical().setFromVector3(
        camera.position.clone().sub(controls.target),
      );
      sphere.theta += azimuth;
      sphere.phi = Math.min(Math.PI - 0.1, Math.max(0.1, sphere.phi + polar));
      sphere.radius = Math.min(4000, Math.max(180, sphere.radius * zoom));
      camera.position
        .copy(controls.target)
        .add(new T.Vector3().setFromSpherical(sphere));
      controls.update();
      requestPaint();
    };
    sceneCanvas.addEventListener(
      "keydown",
      (event) => {
        const moves: Record<string, [number, number, number]> = {
          ArrowLeft: [-0.12, 0, 1],
          ArrowRight: [0.12, 0, 1],
          ArrowUp: [0, -0.12, 1],
          ArrowDown: [0, 0.12, 1],
          "+": [0, 0, 0.85],
          "=": [0, 0, 0.85],
          "-": [0, 0, 1 / 0.85],
        };
        if (event.key === "Home") {
          event.preventDefault();
          resetView();
        } else if (moves[event.key]) {
          event.preventDefault();
          orbit(...moves[event.key]!);
        }
      },
      { signal: events.signal },
    );
    for (const [selector, action] of [
      ["[data-reset-view]", resetView],
      ["[data-zoom-in]", () => orbit(0, 0, 0.85)],
      ["[data-zoom-out]", () => orbit(0, 0, 1 / 0.85)],
    ] as const) {
      const button = root.querySelector<HTMLButtonElement>(selector);
      if (button) {
        button.disabled = false;
        button.addEventListener("click", action, { signal: events.signal });
      }
    }
    sceneCanvas.addEventListener(
      "webglcontextlost",
      (event) => {
        event.preventDefault();
        sceneCanvas.hidden = true;
        sceneFallback?.removeAttribute("hidden");
        for (const control of root.querySelectorAll<HTMLButtonElement>(
          "[data-reset-view], [data-zoom-in], [data-zoom-out]",
        ))
          control.disabled = true;
        setText(
          "[data-sandbox-status]",
          "Static acquisition view. Detector controls remain available.",
        );
      },
      { signal: events.signal },
    );
    sceneCanvas.addEventListener(
      "webglcontextrestored",
      () => {
        sceneCanvas.hidden = false;
        sceneFallback?.setAttribute("hidden", "");
        for (const control of root.querySelectorAll<HTMLButtonElement>(
          "[data-reset-view], [data-zoom-in], [data-zoom-out]",
        ))
          control.disabled = false;
        requestPaint();
        setText(
          "[data-sandbox-status]",
          "Orbit with the pointer or arrow keys. +/− zoom. Home resets the view.",
        );
      },
      { signal: events.signal },
    );
    sceneCanvas.hidden = false;
    sceneFallback?.setAttribute("hidden", "");
    sceneUpdate();
    resetView();
    requestPaint();
    setText(
      "[data-sandbox-status]",
      "Orbit with the pointer or arrow keys. +/− zoom. Home resets the view.",
    );
  } catch {
    sceneDispose?.();
    sceneDispose = undefined;
    scenePaint = undefined;
    sceneUpdate = undefined;
    sceneCanvas.hidden = true;
    sceneFallback?.removeAttribute("hidden");
    setText(
      "[data-sandbox-status]",
      "Static acquisition view. Detector controls remain available.",
    );
  }
  return teardown;
}
