import * as T from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { conditionalLines } from "./conditional-lines";
import {
  ACQUISITION_DEFAULT_ANGLE,
  ACQUISITION_DOMAIN_LIMIT,
  acquisitionDetectorArc,
  acquisitionGeometry,
  acquisitionPoint,
  acquisitionState,
} from "./acquisition-geometry";
import type {
  AcquisitionGeometry,
  AcquisitionMetadata,
  AcquisitionPoint,
} from "./acquisition-geometry";

interface RigMetadata extends AcquisitionMetadata {
  components: {
    name: string;
    link: string;
    edge_offset_bytes: number;
    edge_count: number;
  }[];
}

/** Reparent the supplied orbital link without changing any mesh coordinates. */
export function attachAcquisitionOrbit(
  parent: T.Object3D,
  rig: T.Object3D,
  geometry: AcquisitionGeometry,
) {
  const assembly = rig.getObjectByName("orbital");
  if (!assembly) throw new Error("The supplied C-arm has no orbital assembly");
  const pivot = new T.Group();
  pivot.name = "acquisition-orbital-pivot";
  pivot.position.set(...geometry.origin);
  parent.add(rig, pivot);
  parent.updateMatrixWorld(true);
  pivot.attach(assembly);
  const source = new T.Object3D();
  const detector = new T.Object3D();
  source.name = "acquisition-source-landmark";
  detector.name = "acquisition-detector-landmark";
  source.position.set(0, -geometry.sourceRadius, 0);
  detector.position.set(0, geometry.detectorRadius, 0);
  pivot.add(source, detector);
  return { pivot, source, detector };
}

export function mountAcquisitionCArm(root: HTMLElement): () => void {
  const get = <E extends Element>(name: string): E => {
    const element = root.querySelector<E>(`[data-acq="${name}"]`);
    if (!element) throw new Error(`Missing acquisition element ${name}`);
    return element;
  };
  const viewport = get<HTMLElement>("viewport");
  const canvas = get<HTMLCanvasElement>("canvas");
  const overlay = get<SVGSVGElement>("overlay");
  const status = get<HTMLElement>("status");
  const angleInput = get<HTMLInputElement>("angle");
  const controls = get<HTMLElement>("controls");
  const labelsRoot = get<SVGGElement>("labels");
  const leadersRoot = get<SVGGElement>("leaders");
  const markersRoot = get<SVGGElement>("markers");
  const svgNS = "http://www.w3.org/2000/svg";
  const listeners = new AbortController();
  const resources = new Set<{ dispose(): void }>();
  const keep = <V extends { dispose(): void }>(value: V): V => {
    resources.add(value);
    return value;
  };
  const register = (object: T.Object3D): void => {
    object.traverse((child) => {
      if (child instanceof T.Mesh || child instanceof T.LineSegments) {
        keep(child.geometry);
        for (const material of Array.isArray(child.material)
          ? child.material
          : [child.material])
          keep(material);
      }
    });
  };
  const renderer = new T.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
  });
  // Same opaque white toon material and conditional contours as Appendix A;
  // the background follows the current book paper colour.
  const paper = getComputedStyle(root).getPropertyValue("--paper").trim();
  renderer.setClearColor(new T.Color(paper || "#f7f4ec"));
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new T.Scene();
  const camera = new T.PerspectiveCamera(32, 1, 0.03, 40);
  const toonRamp = keep(
    new T.DataTexture(new Uint8Array([96, 176, 255]), 3, 1, T.RedFormat),
  );
  toonRamp.minFilter = T.NearestFilter;
  toonRamp.magFilter = T.NearestFilter;
  toonRamp.generateMipmaps = false;
  toonRamp.needsUpdate = true;
  const white = keep(
    new T.MeshToonMaterial({
      color: 0xffffff,
      gradientMap: toonRamp,
      side: T.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    }),
  );
  white.defines = { ...white.defines, FLAT_SHADED: "" };
  scene.add(new T.AmbientLight(0xffffff, 0.16));
  const light = new T.DirectionalLight(0xffffff, Math.PI * 0.95);
  light.position.set(-3, 6, 4);
  scene.add(light);
  const orbit = new OrbitControls(camera, canvas);
  orbit.enableDamping = false;
  orbit.enablePan = false;
  orbit.maxPolarAngle = Math.PI * 0.93;
  orbit.minPolarAngle = Math.PI * 0.07;
  orbit.minDistance = 1;
  orbit.maxDistance = 18;
  let stopped = false;
  let ready = false;
  let visible = true;
  let frame = 0;
  let width = 1;
  let height = 1;
  let alpha = ACQUISITION_DEFAULT_ANGLE;
  let geometry: AcquisitionGeometry;
  let articulated: ReturnType<typeof attachAcquisitionOrbit>;
  let apparatus: T.Group;
  const framingBounds = new T.Box3();
  const projected = new T.Vector3();
  const source = new T.Vector3();
  const detector = new T.Vector3();
  const origin = new T.Vector3();
  const clearanceCentre = new T.Vector3();
  const pathElements = new Map<string, SVGPathElement>();
  overlay
    .querySelectorAll<SVGPathElement>("[data-acq-path]")
    .forEach((path) => {
      pathElements.set(path.dataset.acqPath!, path);
    });
  type Label = {
    element: SVGTextElement;
    leader: SVGPathElement;
    point: T.Vector3;
    offset: [number, number];
  };
  const labels = new Map<string, Label>();
  labelsRoot.replaceChildren();
  leadersRoot.replaceChildren();
  markersRoot.replaceChildren();
  const markerElements = new Map<string, SVGElement>();
  for (const name of [
    "source",
    "detector",
    "origin",
    "centre",
    "boundary-negative",
    "boundary-positive",
    "domain-negative",
    "domain-positive",
  ]) {
    const isDetector = name === "detector";
    const marker = document.createElementNS(
      svgNS,
      isDetector ? "rect" : "circle",
    );
    if (isDetector) {
      marker.setAttribute("width", "8");
      marker.setAttribute("height", "8");
    } else marker.setAttribute("r", name === "source" ? "4" : "2.5");
    marker.setAttribute(
      "fill",
      name === "centre" ? "#a4442d" : name === "origin" ? "#27241f" : "#356f73",
    );
    marker.setAttribute("stroke", paper || "#f7f4ec");
    marker.setAttribute("stroke-width", "1.5");
    markersRoot.append(marker);
    markerElements.set(name, marker);
  }
  const project = (point: T.Vector3): [number, number] => {
    projected.copy(point).project(camera);
    return [((projected.x + 1) * width) / 2, ((1 - projected.y) * height) / 2];
  };
  const vector = (point: AcquisitionPoint) => new T.Vector3(...point);
  const world = (radius: number, angle: number) =>
    vector(acquisitionPoint(geometry, radius, angle));
  const path = (name: string, points: T.Vector3[], close = false): void => {
    pathElements.get(name)!.setAttribute(
      "d",
      points
        .map((point, i) => {
          const [x, y] = project(point);
          return `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
        })
        .join(" ") + (close ? " Z" : ""),
    );
  };
  const arc = (
    name: string,
    radius: number,
    start: number,
    end: number,
  ): void => {
    const count = Math.max(2, Math.ceil(Math.abs(end - start) * 36));
    path(
      name,
      Array.from({ length: count + 1 }, (_, i) =>
        world(radius, start + ((end - start) * i) / count),
      ),
    );
  };
  const marker = (name: string, point: T.Vector3): void => {
    const [x, y] = project(point);
    const element = markerElements.get(name)!;
    element.setAttribute(
      name === "detector" ? "x" : "cx",
      String(x - (name === "detector" ? 4 : 0)),
    );
    element.setAttribute(
      name === "detector" ? "y" : "cy",
      String(y - (name === "detector" ? 4 : 0)),
    );
  };
  const label = (
    name: string,
    text: string,
    point: T.Vector3,
    offset: [number, number],
    className = "quantity-label",
  ): void => {
    let item = labels.get(name);
    if (!item) {
      const element = document.createElementNS(svgNS, "text");
      element.setAttribute("text-anchor", "middle");
      element.setAttribute("dominant-baseline", "central");
      element.setAttribute("class", className);
      const leader = document.createElementNS(svgNS, "path");
      labelsRoot.append(element);
      leadersRoot.append(leader);
      item = { element, leader, point: new T.Vector3(), offset };
      labels.set(name, item);
    }
    if (/^R_[sd]/.test(text)) {
      const subscript = document.createElementNS(svgNS, "tspan");
      subscript.setAttribute("baseline-shift", "sub");
      subscript.setAttribute("font-size", "0.75em");
      subscript.textContent = text[2]!;
      item.element.replaceChildren(
        document.createTextNode("R"),
        subscript,
        document.createTextNode(text.slice(3)),
      );
    } else item.element.textContent = text;
    item.point.copy(point);
    item.offset = offset;
  };
  const placeLabels = (): void => {
    const boxes: {
      left: number;
      right: number;
      top: number;
      bottom: number;
    }[] = [];
    for (const { element, leader, point, offset } of labels.values()) {
      const [ax, ay] = project(point);
      const half = Math.max(5, element.getComputedTextLength() / 2);
      let best = { x: 0, y: 0, penalty: Infinity };
      // Each label follows a world anchor. Select a nearby clear screen-space
      // position, retaining a leader when the text must move away from it.
      for (const [dx, dy] of [
        offset,
        [offset[0], -offset[1]],
        ...[22, 40, 62, 86].flatMap((radius) =>
          Array.from({ length: 8 }, (_, i) => [
            Math.cos((i * Math.PI) / 4) * radius,
            Math.sin((i * Math.PI) / 4) * radius,
          ]),
        ),
      ]) {
        const x = Math.max(half + 8, Math.min(width - half - 8, ax + dx!));
        const y = Math.max(14, Math.min(height - 30, ay + dy!));
        const box = {
          left: x - half - 4,
          right: x + half + 4,
          top: y - 10,
          bottom: y + 10,
        };
        const overlap = boxes.reduce(
          (sum, other) =>
            sum +
            Math.max(
              0,
              Math.min(box.right, other.right) - Math.max(box.left, other.left),
            ) *
              Math.max(
                0,
                Math.min(box.bottom, other.bottom) -
                  Math.max(box.top, other.top),
              ),
          0,
        );
        const penalty =
          overlap * 1000 + Math.hypot(x - ax - offset[0], y - ay - offset[1]);
        if (penalty < best.penalty) best = { x, y, penalty };
      }
      element.setAttribute("x", String(best.x));
      element.setAttribute("y", String(best.y));
      const endX = Math.max(best.x - half, Math.min(best.x + half, ax));
      const endY = best.y + (ay < best.y ? -8 : 8);
      leader.setAttribute(
        "d",
        Math.hypot(endX - ax, endY - ay) > 16
          ? `M${ax},${ay} L${endX},${endY}`
          : "",
      );
      boxes.push({
        left: best.x - half - 4,
        right: best.x + half + 4,
        top: best.y - 10,
        bottom: best.y + 10,
      });
    }
  };
  const formatMm = (value: number) => (value * 1000).toFixed(2);
  const formatAngle = () =>
    `${alpha >= 0 ? "+" : "−"}${Math.abs((alpha * 180) / Math.PI)
      .toFixed(1)
      .replace(/\.0$/, "")}°`;
  const annotate = (): void => {
    articulated.source.getWorldPosition(source);
    articulated.detector.getWorldPosition(detector);
    const state = acquisitionState(geometry, alpha);
    const Rs = geometry.sourceRadius;
    const boundary = geometry.boundaryAngle;
    const disk = Array.from({ length: 97 }, (_, i) => {
      const angle = (i * 2 * Math.PI) / 96;
      return clearanceCentre
        .clone()
        .add(
          new T.Vector3(
            Math.cos(angle) * geometry.clearanceRadius,
            Math.sin(angle) * geometry.clearanceRadius,
            0,
          ),
        );
    });
    path("disk", disk, true);
    // Colour detector configurations by the unchanged source-clearance test.
    // The orbit is at -Rd*u, so it passes through the actual detector landmark.
    for (const [name, start, end] of [
      ["outside", Math.PI / 2, (3 * Math.PI) / 2],
      ["allowed-positive", boundary, ACQUISITION_DOMAIN_LIMIT],
      ["allowed-negative", -ACQUISITION_DOMAIN_LIMIT, -boundary],
      ["excluded", -boundary, boundary],
    ] as const)
      path(
        name,
        acquisitionDetectorArc(geometry, start, end).map(({ point }) =>
          vector(point),
        ),
      );
    path("reference", [origin, clearanceCentre]);
    const basisEnd = world(Rs * 0.28, Math.PI / 2);
    path("basis", [origin, basisEnd]);
    path("source-radius", [origin, source]);
    path("detector-radius", [origin, detector]);
    path("clearance", [clearanceCentre, source]);
    const diskBottom = clearanceCentre
      .clone()
      .add(new T.Vector3(0, -geometry.clearanceRadius, 0));
    path("disk-radius", [clearanceCentre, diskBottom]);
    arc("angle", Rs * 0.28, 0, alpha);
    marker("source", source);
    marker("detector", detector);
    marker("origin", origin);
    marker("centre", clearanceCentre);
    for (const [name, angle] of [
      ["boundary-negative", -boundary],
      ["boundary-positive", boundary],
      ["domain-negative", -ACQUISITION_DOMAIN_LIMIT],
      ["domain-positive", ACQUISITION_DOMAIN_LIMIT],
    ] as const)
      marker(name, world(-geometry.detectorRadius, angle));
    for (const name of ["source", "detector"])
      markerElements
        .get(name)!
        .setAttribute(
          "fill",
          state.classification === "allowed" ? "#356f73" : "#a4442d",
        );
    label("source", "Source s(α)", source, [40, 24], "endpoint-label");
    label("detector", "Detector d(α)", detector, [-24, -25], "endpoint-label");
    label("origin", "O", origin, [-14, -13], "endpoint-label");
    label("centre", "c", clearanceCentre, [-15, 7], "endpoint-label");
    label(
      "source-radius",
      `R_s = ${formatMm(Rs)} mm`,
      origin.clone().lerp(source, 0.62),
      [38, -18],
    );
    label(
      "detector-radius",
      `R_d = ${formatMm(geometry.detectorRadius)} mm`,
      origin.clone().lerp(detector, 0.65),
      [-34, -17],
    );
    label(
      "clearance",
      `‖s − c‖ = ${formatMm(state.clearance)} mm`,
      clearanceCentre.clone().lerp(source, 0.6),
      [50, 18],
    );
    label(
      "disk-radius",
      `R_s/2 = ${formatMm(geometry.clearanceRadius)} mm`,
      diskBottom,
      [0, 19],
    );
    label(
      "angle",
      `α = ${formatAngle()}`,
      world(Rs * 0.3, alpha / 2),
      [22, 17],
      "quantity-label angle-label",
    );
    label("e0", "e₀", world(Rs * 0.68, 0), [-13, -2]);
    label("e1", "e₁", basisEnd, [13, -9]);
    placeLabels();
    // Read-only evidence for the static capture and browser acceptance pass.
    root.dataset.angleDegrees = String((alpha * 180) / Math.PI);
    root.dataset.classification = state.classification;
    root.dataset.sourceWorld = JSON.stringify(source.toArray());
    root.dataset.detectorWorld = JSON.stringify(detector.toArray());
    root.dataset.cameraPosition = JSON.stringify(camera.position.toArray());
    root.dataset.cameraTarget = JSON.stringify(orbit.target.toArray());
  };
  const render = (): void => {
    frame = 0;
    if (stopped || !ready || !visible || document.hidden) return;
    scene.updateMatrixWorld(true);
    camera.updateMatrixWorld();
    renderer.render(scene, camera);
    annotate();
  };
  const requestRender = (): void => {
    if (!frame && ready && visible && !document.hidden && !stopped)
      frame = requestAnimationFrame(render);
  };
  const frameApparatus = (): void => {
    const centre = framingBounds.getCenter(new T.Vector3());
    const size = framingBounds.getSize(new T.Vector3());
    const halfFov = (camera.fov * Math.PI) / 360;
    const distance =
      Math.max(
        size.y / 2 / Math.tan(halfFov),
        size.x / 2 / (Math.tan(halfFov) * camera.aspect),
      ) *
        1.26 +
      size.z / 2;
    orbit.target.copy(centre);
    camera.position
      .copy(centre)
      .add(new T.Vector3(0.14, 0.1, 1).normalize().multiplyScalar(distance));
    orbit.minDistance = distance * 0.45;
    orbit.maxDistance = distance * 3.5;
    orbit.update();
  };
  const resize = (): void => {
    width = Math.max(1, viewport.clientWidth);
    height = Math.max(1, viewport.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    overlay.setAttribute("viewBox", `0 0 ${width} ${height}`);
    if (ready) frameApparatus();
    requestRender();
  };
  const setAngle = (next: number): void => {
    alpha = Math.max(
      -ACQUISITION_DOMAIN_LIMIT,
      Math.min(ACQUISITION_DOMAIN_LIMIT, next),
    );
    articulated.pivot.rotation.z = alpha;
    angleInput.value = String((alpha * 180) / Math.PI);
    get<HTMLOutputElement>("angle-output").value = formatAngle();
    angleInput.setAttribute("aria-valuetext", formatAngle());
    const state = acquisitionState(geometry, alpha);
    get<HTMLElement>("readout").textContent =
      `At ${formatAngle()}: source clearance ${formatMm(state.clearance)} mm, detector configuration ${state.classification === "allowed" ? "allowed" : "excluded by source clearance"}.`;
    requestRender();
  };
  orbit.addEventListener("change", requestRender);
  angleInput.addEventListener(
    "input",
    () => setAngle((Number(angleInput.value) * Math.PI) / 180),
    { signal: listeners.signal },
  );
  get<HTMLButtonElement>("reset").addEventListener(
    "click",
    () => {
      setAngle(ACQUISITION_DEFAULT_ANGLE);
      frameApparatus();
      requestRender();
    },
    { signal: listeners.signal },
  );
  canvas.addEventListener(
    "keydown",
    (event) => {
      if (!ready) return;
      const displacement = camera.position.clone().sub(orbit.target);
      const spherical = new T.Spherical().setFromVector3(displacement);
      if (event.key === "ArrowLeft") spherical.theta -= 0.09;
      else if (event.key === "ArrowRight") spherical.theta += 0.09;
      else if (event.key === "ArrowUp") spherical.phi -= 0.09;
      else if (event.key === "ArrowDown") spherical.phi += 0.09;
      else if (event.key === "+" || event.key === "=") spherical.radius *= 0.9;
      else if (event.key === "-" || event.key === "_") spherical.radius /= 0.9;
      else if (event.key === "Home") {
        frameApparatus();
        requestRender();
        event.preventDefault();
        return;
      } else return;
      spherical.phi = Math.max(
        orbit.minPolarAngle,
        Math.min(orbit.maxPolarAngle, spherical.phi),
      );
      spherical.radius = Math.max(
        orbit.minDistance,
        Math.min(orbit.maxDistance, spherical.radius),
      );
      camera.position
        .copy(orbit.target)
        .add(new T.Vector3().setFromSpherical(spherical));
      orbit.update();
      requestRender();
      event.preventDefault();
    },
    { signal: listeners.signal },
  );
  canvas.addEventListener(
    "webglcontextlost",
    (event) => {
      event.preventDefault();
      ready = false;
      root.dataset.state = "error";
      canvas.tabIndex = -1;
      canvas.setAttribute("aria-hidden", "true");
      controls.hidden = true;
      status.hidden = false;
      status.textContent =
        "The graphics context was interrupted. The static drawing remains available. Reload to restore interaction.";
    },
    { signal: listeners.signal },
  );
  document.addEventListener("visibilitychange", requestRender, {
    signal: listeners.signal,
  });
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(viewport);
  const visibilityObserver = new IntersectionObserver((entries) => {
    visible = entries.some((entry) => entry.isIntersecting);
    if (visible) requestRender();
    else if (frame) {
      cancelAnimationFrame(frame);
      frame = 0;
    }
  });
  visibilityObserver.observe(viewport);
  async function load(): Promise<void> {
    status.hidden = false;
    status.textContent = "Loading the annotated C-arm…";
    const fetchAsset = async (path: string): Promise<Response> => {
      const response = await fetch(path, { signal: listeners.signal });
      if (!response.ok)
        throw new Error(`Could not load ${path}: ${response.status}`);
      return response;
    };
    const [metadata, rigBytes, edgesBytes] = await Promise.all([
      fetchAsset("/assets/carm/c-arm-rig.json").then(
        (r) => r.json() as Promise<RigMetadata>,
      ),
      fetchAsset("/assets/carm/c-arm-rig.glb").then((r) => r.arrayBuffer()),
      fetchAsset("/assets/carm/c-arm-rig.edges.bin").then((r) =>
        r.arrayBuffer(),
      ),
    ]);
    geometry = acquisitionGeometry(metadata);
    const rig = await new GLTFLoader().parseAsync(rigBytes, "");
    register(rig.scene);
    if (stopped) {
      for (const resource of resources) resource.dispose();
      return;
    }
    apparatus = new T.Group();
    scene.add(apparatus);
    articulated = attachAcquisitionOrbit(apparatus, rig.scene, geometry);
    let count = 0;
    apparatus.traverse((child) => {
      if (child instanceof T.Mesh) {
        child.material = white;
        count++;
      }
    });
    if (count !== metadata.components.length)
      throw new Error("The acquisition figure did not retain every C-arm mesh");
    root.dataset.meshCount = String(count);
    const edges = new Float32Array(edgesBytes);
    for (const component of metadata.components) {
      const mesh = apparatus.getObjectByName(component.name);
      if (!mesh)
        throw new Error(`Missing C-arm contour parent ${component.name}`);
      const begin = component.edge_offset_bytes / 4;
      const end = begin + component.edge_count * 13;
      if (end > edges.length)
        throw new Error("C-arm contour data are incomplete");
      const contour = conditionalLines(edges.subarray(begin, end));
      register(contour);
      mesh.add(contour);
    }
    origin.set(...geometry.origin);
    clearanceCentre.set(...geometry.clearanceCentre);
    // A fixed camera fit spans every authorised illustration angle, including
    // the original complete apparatus and the clearance disk. Moving the
    // slider changes only the orbital group, never scales or recentres a mesh.
    for (let degrees = -90; degrees <= 90; degrees += 5) {
      articulated.pivot.rotation.z = (degrees * Math.PI) / 180;
      apparatus.updateMatrixWorld(true);
      framingBounds.union(new T.Box3().setFromObject(apparatus));
    }
    framingBounds.expandByPoint(
      clearanceCentre.clone().addScalar(geometry.clearanceRadius),
    );
    framingBounds.expandByPoint(
      clearanceCentre.clone().addScalar(-geometry.clearanceRadius),
    );
    ready = true;
    root.dataset.state = "ready";
    canvas.tabIndex = 0;
    canvas.removeAttribute("aria-hidden");
    status.hidden = true;
    controls.hidden = false;
    setAngle(ACQUISITION_DEFAULT_ANGLE);
    resize();
    requestRender();
  }
  resize();
  void load().catch((error: unknown) => {
    if (stopped) return;
    ready = false;
    root.dataset.state = "error";
    canvas.tabIndex = -1;
    canvas.setAttribute("aria-hidden", "true");
    controls.hidden = true;
    status.hidden = false;
    status.textContent =
      "The interactive view is unavailable. The static drawing and equations remain available.";
    for (const resource of resources) resource.dispose();
    resources.clear();
    console.error("Acquisition C-arm could not load", error);
  });
  return () => {
    stopped = true;
    listeners.abort();
    if (frame) cancelAnimationFrame(frame);
    resizeObserver.disconnect();
    visibilityObserver.disconnect();
    orbit.removeEventListener("change", requestRender);
    orbit.dispose();
    for (const resource of resources) resource.dispose();
    resources.clear();
    renderer.dispose();
    renderer.forceContextLoss();
    root.dataset.state = "static";
    canvas.tabIndex = -1;
    canvas.setAttribute("aria-hidden", "true");
    controls.hidden = true;
  };
}
