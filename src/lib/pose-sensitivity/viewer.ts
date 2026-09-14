import * as T from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { bodyEdges, conditionalLines } from "../carm/conditional-lines";
import {
  parameterLabel,
  type PelvisMeshes,
  type RecordedPose,
  type SensitivityData,
  type Vec3,
} from "./data";
import {
  draggedRecordedPose,
  poseParameter,
  steppedRecordedPose,
  type TransformAxis,
  type TransformMode,
} from "./interaction";

export interface AnatomyViewer {
  setPose(pose: RecordedPose): void;
  setProjection(image: HTMLImageElement | HTMLCanvasElement): void;
  dispose(): void;
}

/** Display recorded geometry; this module performs no projection or transport. */
export function createAnatomyViewer(
  root: HTMLElement,
  data: SensitivityData,
  meshes: PelvisMeshes,
  onPoseRequest: (pose: RecordedPose) => void,
): AnatomyViewer {
  const canvas = root.querySelector<HTMLCanvasElement>("[data-pose-canvas]")!;
  const viewport = root.querySelector<HTMLElement>("[data-pose-viewport]")!;
  const status = root.querySelector<HTMLElement>("[data-scene-status]")!;
  const resources = new Set<{ dispose(): void }>();
  const keep = <V extends { dispose(): void }>(value: V): V => {
    resources.add(value);
    return value;
  };
  const listeners = new AbortController();
  const renderer = new T.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
  });
  renderer.setClearColor(0xeeebe3);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = T.SRGBColorSpace;
  const scene = new T.Scene();
  const apparatus = new T.Group();
  scene.add(apparatus);
  const camera = new T.PerspectiveCamera(34, 1, 1, 12000);
  camera.up.set(0, 0, 1);
  const cameraTarget = new T.Vector3();
  const anatomy = new T.Group();
  anatomy.matrixAutoUpdate = false;
  scene.add(anatomy);
  const ramp = keep(
    new T.DataTexture(new Uint8Array([96, 176, 255]), 3, 1, T.RedFormat),
  );
  ramp.minFilter = T.NearestFilter;
  ramp.magFilter = T.NearestFilter;
  ramp.generateMipmaps = false;
  ramp.needsUpdate = true;
  scene.add(new T.AmbientLight(0xffffff, 0.2));
  const key = new T.DirectionalLight(0xffffff, Math.PI * 0.95);
  key.position.set(-650, 900, 1000);
  scene.add(key);
  for (const item of meshes.meshes) {
    const geometry = keep(new T.BufferGeometry());
    geometry.setAttribute(
      "position",
      new T.Float32BufferAttribute(item.positions, 3),
    );
    geometry.setIndex(item.indices);
    geometry.computeVertexNormals();
    const material = keep(
      new T.MeshToonMaterial({
        color: 0xffffff,
        gradientMap: ramp,
        side: T.DoubleSide,
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      }),
    );
    const mesh = new T.Mesh(geometry, material);
    mesh.name = item.label;
    const contour = conditionalLines(bodyEdges(geometry), 0x425d68);
    keep(contour.geometry);
    if (Array.isArray(contour.material)) contour.material.forEach(keep);
    else keep(contour.material);
    mesh.add(contour);
    anatomy.add(mesh);
  }
  const vector = (value: Vec3) => new T.Vector3(...value);
  const source = vector(data.detector.source_mm);
  const centre = vector(data.detector.centre_mm);
  const u = vector(data.detector.u_axis),
    v = vector(data.detector.v_axis);
  const detectorWidth = data.width * data.detector.pixel_spacing_mm[0];
  const detectorHeight = data.height * data.detector.pixel_spacing_mm[1];
  const corner = (x: number, y: number) =>
    centre
      .clone()
      .addScaledVector(u, (x * detectorWidth) / 2)
      .addScaledVector(v, (y * detectorHeight) / 2);
  const corners = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
  const beamPoints = corners.flatMap((point) => [source, point]);
  const beam = new T.LineSegments(
    keep(new T.BufferGeometry().setFromPoints(beamPoints)),
    keep(
      new T.LineBasicMaterial({
        color: 0xa7775c,
        transparent: true,
        opacity: 0.23,
        depthWrite: false,
      }),
    ),
  );
  apparatus.add(beam);
  const outline = new T.LineLoop(
    keep(new T.BufferGeometry().setFromPoints(corners)),
    keep(new T.LineBasicMaterial({ color: 0x7d8b8d })),
  );
  apparatus.add(outline);
  const centralRay = new T.Line(
    keep(new T.BufferGeometry().setFromPoints([source, centre])),
    keep(
      new T.LineDashedMaterial({
        color: 0xa7775c,
        dashSize: 10,
        gapSize: 8,
        transparent: true,
        opacity: 0.6,
      }),
    ),
  );
  centralRay.computeLineDistances();
  apparatus.add(centralRay);
  const texture = keep(new T.Texture());
  texture.colorSpace = T.SRGBColorSpace;
  texture.minFilter = T.NearestFilter;
  texture.magFilter = T.NearestFilter;
  texture.generateMipmaps = false;
  const detectorMaterial = keep(
    new T.MeshBasicMaterial({
      color: 0xffffff,
      side: T.DoubleSide,
      transparent: true,
      opacity: 0.75,
    }),
  );
  const detector = new T.Mesh(
    keep(new T.PlaneGeometry(detectorWidth, detectorHeight)),
    detectorMaterial,
  );
  const up = v.clone().negate();
  detector.setRotationFromMatrix(
    new T.Matrix4().makeBasis(u, up, u.clone().cross(up)),
  );
  detector.position.copy(centre);
  apparatus.add(detector);
  const focalSpot = new T.Mesh(
    keep(new T.SphereGeometry(9, 16, 12)),
    keep(new T.MeshToonMaterial({ color: 0x9b5b42, gradientMap: ramp })),
  );
  focalSpot.position.copy(source);
  apparatus.add(focalSpot);
  const pivot = new T.Mesh(
    keep(new T.SphereGeometry(3, 12, 8)),
    keep(new T.MeshBasicMaterial({ color: 0xa7775c, depthTest: false })),
  );
  pivot.renderOrder = 4;
  anatomy.add(pivot);

  const label = (text: string, position: T.Vector3) => {
    const labelCanvas = document.createElement("canvas");
    labelCanvas.width = 512;
    labelCanvas.height = 80;
    const context = labelCanvas.getContext("2d");
    if (!context) return;
    context.font = "28px system-ui, sans-serif";
    context.textAlign = "center";
    context.fillStyle = "#425d68";
    context.fillText(text, 256, 46);
    const map = keep(new T.CanvasTexture(labelCanvas));
    const sprite = new T.Sprite(
      keep(new T.SpriteMaterial({ map, depthTest: false, transparent: true })),
    );
    sprite.position.copy(position);
    sprite.scale.set(180, 28, 1);
    apparatus.add(sprite);
  };
  label("Anterior source", source.clone().add(new T.Vector3(0, 0, 35)));
  label(
    "Posterior detector",
    centre.clone().add(new T.Vector3(0, 0, detectorHeight / 2 + 30)),
  );
  // TransformControls manipulates an invisible request proxy. The anatomy uses
  // only a successfully loaded recorded matrix supplied through setPose().
  const proxy = new T.Group();
  scene.add(proxy);
  const transform = new TransformControls(camera, canvas);
  transform.setSpace("world");
  transform.setSize(1.7);
  transform.setColors(0xad573b, 0x477951, 0x376e9d, 0xe3a63d);
  transform.showXY = transform.showYZ = transform.showXZ = false;
  transform.showE = transform.showXYZE = false;
  transform.attach(proxy);
  scene.add(transform.getHelper());
  const axisLabels = (["x", "y", "z"] as const).map((name, index) => {
    const surface = document.createElement("canvas");
    surface.width = surface.height = 64;
    const context = surface.getContext("2d")!;
    context.font = "600 42px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = ["#ad573b", "#477951", "#376e9d"][index]!;
    context.fillText(name, 32, 32);
    const sprite = new T.Sprite(
      keep(
        new T.SpriteMaterial({
          map: keep(new T.CanvasTexture(surface)),
          depthTest: false,
        }),
      ),
    );
    sprite.renderOrder = 1000;
    scene.add(sprite);
    return { name, sprite };
  });
  const reference = data.poses.find((pose) => pose.parameter === "reference")!;
  let committed = reference,
    requested = reference;
  let mode: TransformMode =
    root.dataset.transformMode === "rotate" ? "rotate" : "translate";
  let axis: TransformAxis = ["x", "y", "z"].includes(
    root.dataset.transformAxis ?? "",
  )
    ? (root.dataset.transformAxis as TransformAxis)
    : "x";
  let drag:
    | {
        parameter: ReturnType<typeof poseParameter>;
        start: RecordedPose;
        position: T.Vector3;
        quaternion: T.Quaternion;
        worldSpan: number;
      }
    | undefined;
  const restoreProxy = () => {
    anatomy.matrix.decompose(proxy.position, proxy.quaternion, proxy.scale);
    proxy.updateMatrixWorld(true);
  };
  const modeButtons = root.querySelectorAll<HTMLButtonElement>(
    "[data-transform-mode]",
  );
  const publishControl = () => {
    root.dataset.transformMode = mode;
    root.dataset.transformAxis = axis;
    for (const button of modeButtons)
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.transformMode === mode),
      );
    const help = root.querySelector<HTMLElement>("[data-transform-status]");
    if (help)
      help.textContent = `${parameterLabel(poseParameter(mode, axis))} selected. Arrow keys select recorded steps. Home returns to the reference pose.`;
  };
  const request = (pose: RecordedPose) => {
    if (pose.id === requested.id) return;
    requested = pose;
    onPoseRequest(pose);
  };
  const select = (nextMode: TransformMode, nextAxis: TransformAxis) => {
    const changed = nextMode !== mode || nextAxis !== axis;
    mode = nextMode;
    axis = nextAxis;
    transform.setMode(mode);
    publishControl();
    if (changed) request(reference);
  };
  let disposed = false,
    contextLost = false;
  const render = () => {
    if (!disposed && !contextLost && !document.hidden) {
      const span =
        (camera.position.distanceTo(proxy.position) *
          Math.min(1.9 * Math.tan(T.MathUtils.degToRad(camera.fov / 2)), 7) *
          transform.size) /
        4;
      for (const { name, sprite } of axisLabels) {
        sprite.position.copy(proxy.position);
        sprite.position[name] += span * 0.73;
        sprite.scale.setScalar(span * 0.2);
      }
      renderer.render(scene, camera);
    }
  };
  const resize = () => {
    const width = Math.max(viewport.clientWidth, 1),
      height = Math.max(viewport.clientHeight, 1);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    render();
  };
  const viewDirection = new T.Vector3(0.35, 1, 0.25).normalize();
  const fit = (fullBeam: boolean) => {
    apparatus.visible = fullBeam;
    const bounds = fullBeam
      ? new T.Box3().setFromPoints([source, ...corners])
      : new T.Box3().setFromObject(anatomy);
    const sphere = bounds.getBoundingSphere(new T.Sphere());
    cameraTarget.copy(fullBeam ? sphere.center : new T.Vector3());
    const angle = Math.atan(
      Math.tan(T.MathUtils.degToRad(camera.fov / 2)) *
        Math.min(1, camera.aspect),
    );
    const distance = Math.max(350, (sphere.radius / Math.sin(angle)) * 1.14);
    camera.position.copy(cameraTarget).addScaledVector(viewDirection, distance);
    camera.lookAt(cameraTarget);
    camera.updateMatrixWorld();
    render();
  };
  // A fixed camera keeps a screen gesture about the model, never about the view.
  // Controls emits change before objectChange; skip that provisional transform.
  transform.addEventListener("change", () => {
    if (!transform.dragging) render();
  });
  transform.addEventListener("mouseDown", () => {
    if (!["X", "Y", "Z"].includes(transform.axis ?? "")) return;
    const nextAxis = transform.axis!.toLowerCase() as TransformAxis;
    drag = {
      parameter: poseParameter(mode, nextAxis),
      start: committed,
      position: proxy.position.clone(),
      quaternion: proxy.quaternion.clone(),
      worldSpan:
        camera.position.distanceTo(proxy.position) *
        Math.tan(T.MathUtils.degToRad(camera.fov / 2)) *
        0.5,
    };
    select(mode, nextAxis);
    canvas.focus({ preventScroll: true });
    root.dataset.transformDragging = "true";
  });
  transform.addEventListener("objectChange", () => {
    if (!drag) {
      restoreProxy();
      return;
    }
    const component = drag.parameter[1] as TransformAxis;
    let fraction: number;
    if (drag.parameter.startsWith("t")) {
      fraction =
        (proxy.position[component] - drag.position[component]) / drag.worldSpan;
    } else {
      const relative = proxy.quaternion
        .clone()
        .multiply(drag.quaternion.clone().invert());
      const angle = 2 * Math.atan2(relative[component], relative.w);
      fraction =
        T.MathUtils.euclideanModulo(angle + Math.PI, 2 * Math.PI) /
          (Math.PI / 2) -
        2;
    }
    // Reset BEFORE rendering or requesting: no intermediate continuous or
    // mixed transform can be shown alongside an unmatched recorded projection.
    restoreProxy();
    request(draggedRecordedPose(data, drag.parameter, drag.start, fraction));
    render();
  });
  transform.addEventListener("mouseUp", () => {
    drag = undefined;
    delete root.dataset.transformDragging;
    restoreProxy();
    render();
  });
  canvas.addEventListener(
    "pointercancel",
    () => {
      transform.pointerUp(null);
      drag = undefined;
      delete root.dataset.transformDragging;
      restoreProxy();
      render();
    },
    { signal: listeners.signal },
  );
  canvas.addEventListener(
    "keydown",
    (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey || transform.dragging)
        return;
      const key = event.key.toLowerCase();
      if (key === "t" || key === "r")
        select(key === "t" ? "translate" : "rotate", axis);
      else if (["x", "y", "z"].includes(key)) {
        select(mode, key as TransformAxis);
        transform.axis = key.toUpperCase() as "X" | "Y" | "Z";
      } else if (key === "home") request(reference);
      else if (
        ["arrowleft", "arrowright", "arrowup", "arrowdown"].includes(key)
      ) {
        const direction = key === "arrowright" || key === "arrowup" ? 1 : -1;
        request(
          steppedRecordedPose(
            data,
            poseParameter(mode, axis),
            requested,
            direction,
          ),
        );
      } else return;
      event.preventDefault();
      render();
    },
    { signal: listeners.signal },
  );
  for (const button of modeButtons) {
    button.disabled = false;
    button.addEventListener(
      "click",
      () => {
        if (!transform.dragging)
          select(
            button.dataset.transformMode === "rotate" ? "rotate" : "translate",
            axis,
          );
      },
      { signal: listeners.signal },
    );
  }
  // Framing is an explicit display action; it never changes the physical pose.
  for (const button of root.querySelectorAll<HTMLButtonElement>(
    "[data-view-action]",
  )) {
    button.disabled = false;
    button.addEventListener(
      "click",
      () => fit(button.dataset.viewAction === "beam"),
      { signal: listeners.signal },
    );
  }
  transform.setMode(mode);
  publishControl();
  const observer = new ResizeObserver(resize);
  observer.observe(viewport);
  canvas.addEventListener(
    "webglcontextlost",
    (event) => {
      event.preventDefault();
      contextLost = true;
      transform.enabled = false;
      delete root.dataset.sceneReady;
      status.hidden = false;
      status.textContent =
        "3D rendering was interrupted. The recorded images remain available.";
      canvas.hidden = true;
      for (const button of root.querySelectorAll<HTMLButtonElement>(
        "[data-view-action], [data-transform-mode]",
      ))
        button.disabled = true;
    },
    { signal: listeners.signal },
  );
  resize();
  fit(false);
  canvas.hidden = false;
  status.hidden = true;
  root.dataset.sceneReady = "true";
  return {
    setPose(pose) {
      // Recorded matrices are row-major and already act about the sacral pivot.
      anatomy.matrix.fromArray(pose.matrix).transpose();
      anatomy.matrixWorldNeedsUpdate = true;
      committed = requested = pose;
      restoreProxy();
      if (!drag && pose.parameter !== "reference") {
        mode = pose.parameter.startsWith("t") ? "translate" : "rotate";
        axis = pose.parameter[1] as TransformAxis;
        transform.setMode(mode);
        publishControl();
      }
      render();
    },
    setProjection(image) {
      texture.image = image;
      texture.needsUpdate = true;
      detectorMaterial.map = texture;
      detectorMaterial.needsUpdate = true;
      render();
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      listeners.abort();
      observer.disconnect();
      transform.detach();
      transform.dispose();
      delete root.dataset.transformDragging;
      for (const resource of resources) resource.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      canvas.hidden = true;
      // A lost WebGL context belongs to its canvas. A fresh surface prevents a
      // delayed contextlost event from disabling the next offscreen resume.
      const replacement = canvas.cloneNode(false) as HTMLCanvasElement;
      replacement.hidden = true;
      canvas.replaceWith(replacement);
      for (const button of root.querySelectorAll<HTMLButtonElement>(
        "[data-view-action], [data-transform-mode]",
      ))
        button.disabled = true;
      delete root.dataset.sceneReady;
    },
  };
}
