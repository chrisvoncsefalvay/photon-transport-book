import * as T from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { conditionalLines, bodyEdges } from "./conditional-lines";
import { createCableDeformer } from "./cable-deformation";
import {
  DEFAULT_JOINTS,
  labelsByMode,
  projectionAngles,
  worldEulerDegrees,
} from "./kinematics";
import type { JointSettings, TerminologyMode } from "./kinematics";

import { createGuides, detectorUV } from "./guides";

export function mountCArm(root: HTMLElement): () => void {
  type Vec3 = [number, number, number];
  interface RigMetadata {
    joints: { orbital: { pivot: Vec3 }; angulation: { pivot: Vec3 } };
    landmarks: { source: Vec3; detector: Vec3 };
    edge_format: { file: string };
    components: {
      name: string;
      link: string;
      edge_offset_bytes: number;
      edge_count: number;
    }[];
    cables: {
      link: string;
      endpoint0: Vec3;
      endpoint1: Vec3;
      parent0: string;
      parent1: string;
    }[];
  }
  const $ = <E extends HTMLElement = HTMLElement>(id: string) =>
    root.querySelector(`[data-carm-id="${id}"]`) as E;
  const canvas = $<HTMLCanvasElement>("scene"),
    viewport = $("viewport"),
    status = $("status");
  $("controls").replaceChildren();
  delete viewport.dataset.ready;
  delete viewport.dataset.error;
  status.hidden = false;
  status.textContent = "Loading C-arm and body…";
  const settings: JointSettings = { ...DEFAULT_JOINTS };
  let mode: TerminologyMode = "common",
    hoveredJoint: keyof JointSettings | null = null,
    stopped = false,
    ready = false,
    selected: keyof JointSettings = "orbitalDeg";
  const specs: {
    key: keyof JointSettings;
    min: number;
    max: number;
    step: number;
    unit: string;
  }[] = [
    { key: "insertionMm", min: -150, max: 150, step: 5, unit: "mm" },
    { key: "longitudinalMm", min: -500, max: 500, step: 5, unit: "mm" },
    { key: "liftMm", min: -80, max: 300, step: 5, unit: "mm" },
    { key: "angulationDeg", min: -35, max: 35, step: 1, unit: "°" },
    { key: "orbitalDeg", min: -60, max: 60, step: 1, unit: "°" },
  ];
  const rows = new Map<keyof JointSettings, HTMLElement>();
  const listeners = new AbortController(),
    resources = new Set<{ dispose(): void }>();
  const keep = <V extends { dispose(): void }>(v: V): V => {
    resources.add(v);
    return v;
  };
  const renderer = new T.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
  });
  renderer.setClearColor(0xeeeae2);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new T.Scene(),
    camera = new T.PerspectiveCamera(35, 1, 0.03, 50);
  // A three-byte lighting ramp, not an image texture on either model.
  const toonRamp = keep(
    new T.DataTexture(new Uint8Array([96, 176, 255]), 3, 1, T.RedFormat),
  );
  toonRamp.minFilter = T.NearestFilter;
  toonRamp.magFilter = T.NearestFilter;
  toonRamp.generateMipmaps = false;
  toonRamp.needsUpdate = true;
  scene.add(new T.AmbientLight(0xffffff, 0.16));
  const keyLight = new T.DirectionalLight(0xffffff, Math.PI * 0.95);
  keyLight.position.set(-3, 6, 4);
  scene.add(keyLight);
  const orbit = new OrbitControls(camera, canvas);
  orbit.enableDamping = false;
  orbit.minDistance = 1.4;
  orbit.maxDistance = 16;
  orbit.maxPolarAngle = Math.PI * 0.91;
  const guides = createGuides(scene);
  let planeVisible = true,
    axesVisible = false;
  const hoverMeshes: T.Mesh[] = [];
  let patientHeightMetres = 0;
  const machine = new T.Group(),
    lift = new T.Group(),
    insertion = new T.Group(),
    angulation = new T.Group(),
    orbital = new T.Group();
  scene.add(machine);
  machine.add(lift);
  lift.add(insertion);
  insertion.add(angulation);
  angulation.add(orbital);
  const patient = new T.Group();
  scene.add(patient);
  const source = new T.Object3D(),
    detector = new T.Object3D(),
    isocentre = new T.Object3D();
  const patientTarget = new T.Vector3();
  const chassisRest = new T.Vector3();
  const pickables: T.Mesh[] = [];
  const cables: {
    mesh: T.Mesh;
    anchors: [T.Object3D, T.Object3D];
    endpoints: [T.Vector3, T.Vector3];
    nearest: [number, number];
    deform: ReturnType<typeof createCableDeformer>;
  }[] = [];
  function updateCables(): void {
    scene.updateMatrixWorld(true);
    for (const cable of cables) {
      const deltas = cable.anchors.map((anchor, i) => {
        const delta = cable.mesh
          .worldToLocal(anchor.getWorldPosition(new T.Vector3()))
          .sub(cable.endpoints[i]!);
        if (delta.lengthSq() < 1e-24) delta.set(0, 0, 0);
        return delta;
      });
      cable.deform.update(deltas[0]!, deltas[1]!);
    }
  }
  let metadata: RigMetadata;
  const raycaster = new T.Raycaster();
  const centralRay = new T.Line(
    keep(
      new T.BufferGeometry().setFromPoints([new T.Vector3(), new T.Vector3()]),
    ),
    keep(
      new T.LineDashedMaterial({
        color: 0xa4442d,
        dashSize: 0.018,
        gapSize: 0.014,
        transparent: true,
        opacity: 0.65,
        depthTest: false,
        depthWrite: false,
      }),
    ),
  );
  centralRay.renderOrder = 5;
  scene.add(centralRay);
  const targetCross = new T.Group();
  const crossMaterial = keep(
    new T.LineBasicMaterial({
      color: 0xa4442d,
      depthTest: false,
      transparent: true,
      opacity: 0.85,
    }),
  );
  for (const axis of [
    new T.Vector3(1, 0, 0),
    new T.Vector3(0, 1, 0),
    new T.Vector3(0, 0, 1),
  ]) {
    const line = new T.Line(
      keep(
        new T.BufferGeometry().setFromPoints([
          axis.clone().multiplyScalar(-0.025),
          axis.clone().multiplyScalar(0.025),
        ]),
      ),
      crossMaterial,
    );
    line.renderOrder = 6;
    targetCross.add(line);
  }
  scene.add(targetCross);

  function label(text: string, position: T.Vector3, scale = 0.15): T.Sprite {
    const tile = document.createElement("canvas");
    tile.width = 512;
    tile.height = 96;
    const context = tile.getContext("2d")!;
    context.font = "36px Arial";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = "#50646b";
    context.fillText(text, 256, 48);
    const texture = keep(new T.CanvasTexture(tile));
    texture.colorSpace = T.SRGBColorSpace;
    const sprite = new T.Sprite(
      keep(
        new T.SpriteMaterial({
          map: texture,
          transparent: true,
          depthWrite: false,
          depthTest: false,
        }),
      ),
    );
    sprite.position.copy(position);
    sprite.scale.set((scale * 512) / 96, scale, 1);
    scene.add(sprite);
    return sprite;
  }
  function register(object: T.Object3D): void {
    object.traverse((o) => {
      if (o instanceof T.Mesh || o instanceof T.Line) {
        resources.add(o.geometry);
        for (const m of Array.isArray(o.material) ? o.material : [o.material])
          resources.add(m);
      }
    });
  }
  function select(key: keyof JointSettings): void {
    selected = key;
    for (const [k, row] of rows) row.dataset.selected = String(k === key);
    const hint = $("movement-hint");
    if (hint)
      hint.textContent = `${labelsByMode[mode][key].name}. Values describe the joint. Projection angles are measured in the patient frame.`;
  }
  function updateLabels(): void {
    for (const spec of specs) {
      const row = rows.get(spec.key)!,
        labels = labelsByMode[mode][spec.key];
      const parts = labels.name.split("/");
      row
        .querySelector("label")!
        .replaceChildren(
          parts[0]!,
          "/",
          document.createElement("wbr"),
          parts.slice(1).join("/"),
        );
      for (const input of row.querySelectorAll("input"))
        input.setAttribute(
          "aria-label",
          `${labels.name}, ${spec.unit === "°" ? "degrees" : "millimetres"}`,
        );
      row.querySelector('[data-direction="-1"]')!.textContent =
        `← ${labels.negative}`;
      row.querySelector('[data-direction="1"]')!.textContent =
        `${labels.positive} →`;
    }
    const cardiac = mode === "cardiac";
    $("heart").setAttribute("aria-pressed", String(cardiac));
    $("heart").setAttribute(
      "aria-label",
      cardiac ? "Use everyday terminology" : "Use cardiac terminology",
    );
    $("heart").title = cardiac
      ? "Use everyday terminology"
      : "Use cardiac terminology";
    $("mode-label").textContent = cardiac ? "Cardiac" : "Everyday";
    select(selected);
  }
  function setJoint(key: keyof JointSettings, value: number): void {
    const spec = specs.find((s) => s.key === key)!;
    if (!Number.isFinite(value)) value = settings[key];
    settings[key] = T.MathUtils.clamp(value, spec.min, spec.max);
    for (const input of rows.get(key)!.querySelectorAll("input"))
      input.value = String(Number(settings[key].toFixed(1)));
    select(key);
    applyPose();
  }
  for (const spec of specs) {
    const row = document.createElement("div");
    row.className = "joint-control";
    row.dataset.joint = spec.key;
    row.innerHTML = `<div class="joint-heading"><label for="carm-${spec.key}-slider"></label><div class="joint-number"><input id="carm-${spec.key}-number" type="number" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="0" disabled><span>${spec.unit}</span></div></div><input id="carm-${spec.key}-slider" type="range" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="0" disabled><div class="joint-ends"><button type="button" data-direction="-1" disabled></button><button type="button" data-direction="1" disabled></button></div>`;
    row.addEventListener(
      "pointerenter",
      () => {
        hoveredJoint = spec.key;
        render();
      },
      { signal: listeners.signal },
    );
    row.addEventListener(
      "pointerleave",
      () => {
        if (!row.contains(document.activeElement)) {
          hoveredJoint = null;
          render();
        }
      },
      { signal: listeners.signal },
    );
    row.addEventListener(
      "focusin",
      () => {
        hoveredJoint = spec.key;
        render();
      },
      { signal: listeners.signal },
    );
    row.addEventListener(
      "focusout",
      (event) => {
        if (!row.contains(event.relatedTarget as Node | null)) {
          hoveredJoint = null;
          render();
        }
      },
      { signal: listeners.signal },
    );
    rows.set(spec.key, row);
    $("controls").append(row);
    for (const input of row.querySelectorAll("input"))
      input.addEventListener(
        input.type === "range" ? "input" : "change",
        () => setJoint(spec.key, Number(input.value)),
        { signal: listeners.signal },
      );
    for (const button of row.querySelectorAll<HTMLButtonElement>("button"))
      button.addEventListener(
        "click",
        () =>
          setJoint(
            spec.key,
            settings[spec.key] + Number(button.dataset.direction) * spec.step,
          ),
        { signal: listeners.signal },
      );
  }
  updateLabels();

  const isoWorld = new T.Vector3(),
    sourceWorld = new T.Vector3(),
    detectorWorld = new T.Vector3(),
    projectionDirection = new T.Vector3();
  const framePosition = new T.Vector3(),
    frameRotation = new T.Quaternion();
  function formatted(v: number, decimals = 3): string {
    return (Math.abs(v) < 0.5 * 10 ** -decimals ? 0 : v).toFixed(decimals);
  }
  function readouts(): void {
    for (const [id, visible, name] of [
      ["beam", centralRay.visible, "central ray"],
      ["slice-plane", planeVisible, "imaging plane"],
      ["reference-axes", axesVisible, "reference axes"],
    ] as const) {
      const button = $(id);
      button.setAttribute("aria-pressed", String(visible));
      button.title = `${visible ? "Hide" : "Show"} ${name}`;
      button.setAttribute("aria-label", button.title);
    }

    isocentre.getWorldPosition(isoWorld);
    source.getWorldPosition(sourceWorld);
    detector.getWorldPosition(detectorWorld);
    projectionDirection.copy(detectorWorld).sub(sourceWorld).normalize();
    const clinical = projectionAngles(projectionDirection);
    // These are the mechanical slider settings. Derived patient-relative beam
    // angles remain in the diagnostic state and are not interchangeable with them.
    const rotationLabel =
      mode === "common"
        ? "Rotation"
        : settings.orbitalDeg > 0
          ? "LAO"
          : settings.orbitalDeg < 0
            ? "RAO"
            : "Frontal";
    const swingLabel =
      mode === "common"
        ? "Swing"
        : settings.angulationDeg > 0
          ? "Cranial"
          : settings.angulationDeg < 0
            ? "Caudal"
            : "Neutral";
    const rotationValue =
      mode === "common" ? settings.orbitalDeg : Math.abs(settings.orbitalDeg);
    const swingValue =
      mode === "common"
        ? settings.angulationDeg
        : Math.abs(settings.angulationDeg);
    $("oblique").innerHTML =
      `${rotationLabel} <span>${formatted(rotationValue, 1)}°</span>`;
    $("cranial").innerHTML =
      `${swingLabel} <span>${formatted(swingValue, 1)}°</span>`;
    const offset = isoWorld.distanceTo(patientTarget) * 1000;
    $("centering").textContent =
      offset < 0.1
        ? "Body at isocentre"
        : `Isocentre offset ${offset.toFixed(0)} mm`;
    const reference = $<HTMLSelectElement>("reference").value as
      "world" | "patient" | "detector";
    const referenceMatrix =
      reference === "detector"
        ? detector.matrixWorld.clone()
        : new T.Matrix4().makeTranslation(
            reference === "patient" ? patientTarget : new T.Vector3(),
          );
    const uv = detectorUV(patientTarget, detector.matrixWorld);
    framePosition.copy(isoWorld);
    isocentre.getWorldQuaternion(frameRotation);
    if (reference === "patient") framePosition.sub(patientTarget);
    if (reference === "detector") {
      // The visible detector frame is U=X, V=-Z, N=Y.
      const q = detector
        .getWorldQuaternion(new T.Quaternion())
        .multiply(
          new T.Quaternion().setFromAxisAngle(
            new T.Vector3(1, 0, 0),
            -Math.PI / 2,
          ),
        )
        .invert();
      framePosition.sub(detectorWorld).applyQuaternion(q);
      frameRotation.premultiply(q);
    }
    const angles = worldEulerDegrees(frameRotation);
    $("position").textContent =
      reference === "detector"
        ? `U ${formatted(uv.u)}  V ${formatted(uv.v)}`
        : `${formatted(framePosition.x)}  ${formatted(framePosition.y)}  ${formatted(framePosition.z)}`;
    if ($("position-label"))
      $("position-label").textContent =
        reference === "detector" ? "U/V" : "Position";
    $("pose-target").textContent =
      reference === "detector"
        ? "Body target·detector plane (m)"
        : "Isocentre pose·metres/degrees";
    if ($("reference-origin"))
      $("reference-origin").textContent =
        reference === "detector"
          ? "Origin: detector centre (U=0,V=0)"
          : reference === "patient"
            ? `Origin: body target (0,${formatted(patientTarget.y)},0) m`
            : "Origin: floor (0,0,0) m";
    guides.update({
      isoMatrix: isocentre.matrixWorld,
      referenceMatrix,
      reference,
      hover: hoveredJoint,
      mode,
      translationAnchor: isoWorld,
      angulationMatrix: angulation.matrixWorld,
      orbitalMatrix: orbital.matrixWorld,
    });
    $("orientation-label").textContent =
      reference === "detector" ? "UVN Euler" : "XYZ Euler";
    $("euler").textContent =
      `${formatted(angles.x, 1)}°  ${formatted(angles.y, 1)}°  ${formatted(angles.z, 1)}°`;
    $("axis-key").textContent =
      reference === "detector"
        ? "U right·V up in the detector plane"
        : reference === "patient"
          ? "X left·Y anterior·Z caudal"
          : "X left·Y up·Z caudal";
    const pos = centralRay.geometry.getAttribute(
      "position",
    ) as T.BufferAttribute;
    pos.setXYZ(0, sourceWorld.x, sourceWorld.y, sourceWorld.z);
    pos.setXYZ(1, detectorWorld.x, detectorWorld.y, detectorWorld.z);
    pos.needsUpdate = true;
    centralRay.computeLineDistances();
    targetCross.position.copy(isoWorld);
    // Read-only evidence mirrors actual scene matrices for browser verification.
    const evidence = {
      joints: { ...settings },
      chassis: machine.position.toArray(),
      mode,
      iso: isoWorld.toArray(),
      patientTarget: patientTarget.toArray(),
      source: sourceWorld.toArray(),
      detector: detectorWorld.toArray(),
      detectorDirection: projectionDirection.toArray(),
      projection: clinical,
      reference,
      framePosition: framePosition.toArray(),
      detectorUV: uv,
      referenceOrigin: new T.Vector3()
        .setFromMatrixPosition(referenceMatrix)
        .toArray(),
      referenceMatrix: referenceMatrix.toArray(),
      imagingMatrix: isocentre.matrixWorld.toArray(),
      hoveredJoint,
      planeVisible,
      axesVisible,
      patientHeightMetres,
      frameEuler: angles,
      sourceDetectorDistance: sourceWorld.distanceTo(detectorWorld),
      offsetMm: offset,
      camera: camera.position.toArray(),
      meshes: pickables.length,
      machineMaterials: [
        ...new Set(pickables.map((mesh) => (mesh.material as T.Material).type)),
      ],
      cableAttachments: cables.map((c) =>
        c.anchors.map((a, i) =>
          c.mesh
            .localToWorld(
              new T.Vector3().fromBufferAttribute(
                c.mesh.geometry.getAttribute("position"),
                c.nearest[i]!,
              ),
            )
            .distanceTo(a.getWorldPosition(new T.Vector3())),
        ),
      ),
    };
    viewport.dataset.pose = JSON.stringify(evidence);
  }
  function render(): void {
    if (!ready) return;
    scene.updateMatrixWorld(true);
    readouts();
    guides.resizeLabels(camera, viewport.clientHeight);
    renderer.render(scene, camera);
  }
  function applyPose(): void {
    if (!ready) return;
    machine.position.z = chassisRest.z + settings.longitudinalMm / 1000;
    lift.position.y = settings.liftMm / 1000;
    insertion.position.x = settings.insertionMm / 1000;
    angulation.rotation.x = -T.MathUtils.degToRad(settings.angulationDeg);
    orbital.rotation.z = -T.MathUtils.degToRad(settings.orbitalDeg);
    updateCables();
    render();
  }
  function resetCamera(): void {
    const narrow = viewport.clientWidth < 620;
    camera.position
      .copy(patientTarget)
      .add(
        new T.Vector3(
          narrow ? -3.6 : -2.3,
          narrow ? 2.7 : 1.7,
          narrow ? 4.8 : 3.0,
        ),
      );
    orbit.target.copy(patientTarget).add(new T.Vector3(0.35, -0.07, 0.1));
    // Portrait views need more horizontal room for the machine and trolley.
    if (narrow) {
      const delta = camera.position.clone().sub(orbit.target);
      camera.position
        .copy(orbit.target)
        .add(delta.multiplyScalar(Math.max(1, 0.88 / camera.aspect)));
    }
    orbit.update();
    render();
  }
  function zoom(factor: number): void {
    const offset = camera.position.clone().sub(orbit.target);
    offset.setLength(
      T.MathUtils.clamp(
        offset.length() * factor,
        orbit.minDistance,
        orbit.maxDistance,
      ),
    );
    camera.position.copy(orbit.target).add(offset);
    orbit.update();
    render();
  }
  function resize(): void {
    const w = viewport.clientWidth,
      h = viewport.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    render();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(viewport);
  orbit.addEventListener("change", render);
  $("reference").addEventListener("change", render, {
    signal: listeners.signal,
  });
  $("heart").addEventListener(
    "click",
    () => {
      mode = mode === "common" ? "cardiac" : "common";
      updateLabels();
      render();
    },
    { signal: listeners.signal },
  );
  $("beam").addEventListener(
    "click",
    () => {
      centralRay.visible = !centralRay.visible;
      targetCross.visible = centralRay.visible;
      $("beam").setAttribute("aria-pressed", String(centralRay.visible));
      $("beam").setAttribute(
        "aria-label",
        centralRay.visible ? "Hide central ray" : "Show central ray",
      );
      render();
    },
    { signal: listeners.signal },
  );
  $("slice-plane").addEventListener(
    "click",
    () => {
      planeVisible = !planeVisible;
      guides.setPlaneVisible(planeVisible);
      $("slice-plane").setAttribute("aria-pressed", String(planeVisible));
      $("slice-plane").setAttribute(
        "aria-label",
        planeVisible ? "Hide imaging plane" : "Show imaging plane",
      );
      render();
    },
    { signal: listeners.signal },
  );
  $("reference-axes").addEventListener(
    "click",
    () => {
      axesVisible = !axesVisible;
      guides.setAxesVisible(axesVisible);
      $("reference-axes").setAttribute("aria-pressed", String(axesVisible));
      $("reference-axes").setAttribute(
        "aria-label",
        axesVisible ? "Hide reference axes" : "Show reference axes",
      );
      render();
    },
    { signal: listeners.signal },
  );
  $("zoom-in").addEventListener("click", () => zoom(0.85), {
    signal: listeners.signal,
  });
  $("zoom-out").addEventListener("click", () => zoom(1 / 0.85), {
    signal: listeners.signal,
  });
  $("camera-reset").addEventListener("click", resetCamera, {
    signal: listeners.signal,
  });
  $("reset").addEventListener(
    "click",
    () => {
      Object.assign(settings, DEFAULT_JOINTS);
      machine.position.copy(chassisRest);
      for (const row of rows.values())
        for (const input of row.querySelectorAll("input")) input.value = "0";
      applyPose();
    },
    { signal: listeners.signal },
  );
  $("centre").addEventListener(
    "click",
    () => {
      scene.updateMatrixWorld(true);
      isocentre.getWorldPosition(isoWorld);
      const correction = patientTarget.clone().sub(isoWorld);
      // Keep the wheels on the floor: use the lift, not chassis height.
      setJoint("longitudinalMm", settings.longitudinalMm + correction.z * 1000);
      setJoint("insertionMm", settings.insertionMm + correction.x * 1000);
      setJoint("liftMm", settings.liftMm + correction.y * 1000);
      render();
    },
    { signal: listeners.signal },
  );
  canvas.addEventListener(
    "keydown",
    (event) => {
      const delta = camera.position.clone().sub(orbit.target),
        spherical = new T.Spherical().setFromVector3(delta);
      if (event.key === "ArrowLeft") spherical.theta -= 0.12;
      else if (event.key === "ArrowRight") spherical.theta += 0.12;
      else if (event.key === "ArrowUp") spherical.phi -= 0.1;
      else if (event.key === "ArrowDown") spherical.phi += 0.1;
      else if (event.key === "+" || event.key === "=") {
        zoom(0.9);
        event.preventDefault();
        return;
      } else if (event.key === "-") {
        zoom(1 / 0.9);
        event.preventDefault();
        return;
      } else if (event.key === "Home") {
        resetCamera();
        event.preventDefault();
        return;
      } else return;
      spherical.phi = T.MathUtils.clamp(spherical.phi, 0.08, Math.PI * 0.91);
      camera.position
        .copy(orbit.target)
        .add(new T.Vector3().setFromSpherical(spherical));
      orbit.update();
      render();
      event.preventDefault();
    },
    { signal: listeners.signal },
  );
  let pointer: { x: number; y: number } | null = null;
  canvas.addEventListener(
    "pointerdown",
    (e) => {
      pointer = { x: e.clientX, y: e.clientY };
    },
    { signal: listeners.signal },
  );
  canvas.addEventListener(
    "pointerup",
    (e) => {
      if (
        !pointer ||
        Math.hypot(e.clientX - pointer.x, e.clientY - pointer.y) > 5
      ) {
        pointer = null;
        return;
      }
      pointer = null;
      const rect = canvas.getBoundingClientRect();
      raycaster.setFromCamera(
        new T.Vector2(
          ((e.clientX - rect.left) / rect.width) * 2 - 1,
          (-(e.clientY - rect.top) / rect.height) * 2 + 1,
        ),
        camera,
      );
      const hit = raycaster.intersectObjects(hoverMeshes, false)[0];
      if (hit) {
        const key = hit.object.userData.joint as keyof JointSettings;
        if (key) {
          select(key);
          rows
            .get(key)!
            .querySelector("input[type=range]")
            ?.scrollIntoView({ block: "nearest" });
        }
      }
    },
    { signal: listeners.signal },
  );
  let lastHover = 0;
  canvas.addEventListener(
    "pointermove",
    (event) => {
      if (pointer || !ready || event.timeStamp - lastHover < 45) return;
      lastHover = event.timeStamp;
      const rect = canvas.getBoundingClientRect();
      raycaster.setFromCamera(
        new T.Vector2(
          ((event.clientX - rect.left) / rect.width) * 2 - 1,
          (-(event.clientY - rect.top) / rect.height) * 2 + 1,
        ),
        camera,
      );
      const hit = raycaster.intersectObjects(hoverMeshes, false)[0];
      const key =
        (hit?.object.userData.joint as keyof JointSettings | undefined) ?? null;
      if (key !== hoveredJoint) {
        hoveredJoint = key;
        canvas.style.cursor = key ? "pointer" : "grab";
        render();
      }
    },
    { signal: listeners.signal },
  );
  canvas.addEventListener(
    "pointerleave",
    () => {
      hoveredJoint = null;
      render();
    },
    { signal: listeners.signal },
  );
  canvas.addEventListener(
    "pointercancel",
    () => {
      pointer = null;
    },
    { signal: listeners.signal },
  );
  canvas.addEventListener(
    "webglcontextlost",
    (e) => {
      e.preventDefault();
      ready = false;
      status.hidden = false;
      status.textContent =
        "The graphics context was interrupted. Reload this view to restore it.";
    },
    { signal: listeners.signal },
  );

  async function initialise(): Promise<void> {
    const loader = new GLTFLoader();
    const contourData = fetch("/assets/carm/c-arm-rig.json").then(
      async (response) => {
        if (!response.ok) throw new Error("Rig metadata missing");
        const meta = (await response.json()) as RigMetadata;
        const edgeResponse = await fetch(
          `/assets/carm/${meta.edge_format.file}`,
        );
        if (!edgeResponse.ok) throw new Error("Contour data missing");
        return {
          meta,
          edges: new Float32Array(await edgeResponse.arrayBuffer()),
        };
      },
    );
    const [rig, body, { meta, edges }, bodyMeta] = await Promise.all([
      loader.loadAsync("/assets/carm/c-arm-rig.glb"),
      loader.loadAsync("/assets/carm/patient.glb"),
      contourData,
      fetch("/assets/carm/patient.json").then((r) => {
        if (!r.ok) throw new Error("Body metadata missing");
        return r.json();
      }),
    ]);
    metadata = meta;
    register(rig.scene);
    register(body.scene);
    if (stopped) {
      for (const resource of resources) resource.dispose();
      return;
    }
    machine.add(rig.scene);
    const iso = new T.Vector3(...metadata.joints.orbital.pivot),
      bearing = new T.Vector3(...metadata.joints.angulation.pivot);
    angulation.position.copy(bearing);
    orbital.position.copy(iso).sub(bearing);
    scene.updateMatrixWorld(true);
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
    // MeshToonMaterial has no flatShading option in Three 0.185. Enable its
    // built-in derivative-normal shader branch for the normal-free machine.
    white.defines = { ...white.defines, FLAT_SHADED: "" };
    const linkTargets: Record<string, T.Object3D> = {
      base: machine,
      lift,
      insertion,
      support: angulation,
      orbital,
      cable_base_support: machine,
      cable_support_orbital: machine,
    };
    for (const [link, target] of Object.entries(linkTargets)) {
      const group = rig.scene.getObjectByName(link);
      if (!group) continue;
      target.attach(group);
      group.traverse((obj) => {
        if (obj instanceof T.Mesh) {
          obj.material = white;
          obj.userData.joint =
            link === "orbital"
              ? "orbitalDeg"
              : link === "support"
                ? "angulationDeg"
                : link === "lift" || link === "cables"
                  ? "liftMm"
                  : link === "base"
                    ? "longitudinalMm"
                    : "insertionMm";
          pickables.push(obj);
          hoverMeshes.push(obj);
        }
      });
    }
    for (const block of metadata.components) {
      const mesh = machine.getObjectByName(block.name);
      if (!mesh) throw new Error(`Missing contour parent ${block.name}`);
      const contour = conditionalLines(
        edges.subarray(
          block.edge_offset_bytes / 4,
          block.edge_offset_bytes / 4 + block.edge_count * 13,
        ),
      );
      register(contour);
      mesh.add(contour);
      mesh.userData.contour = contour;
    }
    // Preserve each supplied cable and anchor its endpoints to their actual links.
    for (const cable of metadata.cables) {
      const block = metadata.components.find((c) => c.link === cable.link)!;
      const mesh = machine.getObjectByName(block.name) as T.Mesh;
      const endpoints = [
        new T.Vector3(...cable.endpoint0),
        new T.Vector3(...cable.endpoint1),
      ] as [T.Vector3, T.Vector3];
      const anchors = endpoints.map((point, i) => {
        const anchor = new T.Object3D();
        anchor.position.copy(point);
        machine.add(anchor);
        scene.updateMatrixWorld(true);
        machine
          .getObjectByName(i === 0 ? cable.parent0 : cable.parent1)!
          .attach(anchor);
        return anchor;
      }) as [T.Object3D, T.Object3D];
      const position = mesh.geometry.getAttribute("position");
      const nearest = endpoints.map((p) => {
        let best = 0,
          distance = Infinity;
        const v = new T.Vector3();
        for (let i = 0; i < position.count; i++) {
          v.fromBufferAttribute(position, i);
          const d = v.distanceToSquared(p);
          if (d < distance) {
            distance = d;
            best = i;
          }
        }
        return best;
      }) as [number, number];
      cables.push({
        mesh,
        anchors,
        endpoints,
        nearest,
        deform: createCableDeformer(
          mesh,
          mesh.userData.contour as T.LineSegments,
          ...endpoints,
        ),
      });
    }
    source.position.copy(new T.Vector3(...metadata.landmarks.source)).sub(iso);
    detector.position
      .copy(new T.Vector3(...metadata.landmarks.detector))
      .sub(iso);
    orbital.add(source, detector, isocentre);
    chassisRest.set(-iso.x, 0, -iso.z);
    machine.position.copy(chassisRest);
    patientTarget.set(0, iso.y, 0);
    patient.rotation.x = -Math.PI / 2;
    const torsoTarget = new T.Vector3(...(bodyMeta.torsoTarget as Vec3));
    patient.position
      .copy(patientTarget)
      .sub(torsoTarget.clone().applyQuaternion(patient.quaternion));
    patient.add(body.scene);
    body.scene.traverse((obj) => {
      if (obj instanceof T.Mesh) {
        hoverMeshes.push(obj);
        obj.material = keep(
          new T.MeshToonMaterial({
            color: 0xffffff,
            gradientMap: toonRamp,
            side: T.DoubleSide,
            polygonOffset: true,
            polygonOffsetFactor: 1,
            polygonOffsetUnits: 1,
          }),
        );
        const lines = conditionalLines(bodyEdges(obj.geometry), 0x425d68);
        register(lines);
        obj.add(lines);
      }
    });
    const ground = new T.Mesh(
      keep(new T.PlaneGeometry(6, 6)),
      keep(new T.MeshBasicMaterial({ color: 0xe8e4db, side: T.DoubleSide })),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(0.5, -0.004, 0);
    scene.add(ground);
    const grid = new T.GridHelper(6, 12, 0xd4d0c6, 0xe0dcd2);
    register(grid);
    grid.position.set(0.5, -0.002, 0);
    scene.add(grid);
    const table = new T.Mesh(
      keep(new T.BoxGeometry(0.66, 0.025, 2.0)),
      keep(
        new T.MeshBasicMaterial({
          color: 0xddd9d0,
          transparent: true,
          opacity: 0.45,
          depthWrite: false,
        }),
      ),
    );
    const bodyBounds = new T.Box3().setFromObject(patient);
    patientHeightMetres = bodyBounds.max.z - bodyBounds.min.z;
    if (Math.abs(patientHeightMetres - 1.8) > 1e-6)
      throw new Error("Patient height does not match 1.80m scale");
    table.position.set(
      patientTarget.x,
      bodyBounds.min.y - 0.0125,
      (bodyBounds.min.z + bodyBounds.max.z) / 2,
    );
    scene.add(table);
    label(
      "Head · cranial",
      new T.Vector3(
        patientTarget.x,
        patientTarget.y + 0.07,
        bodyBounds.min.z - 0.1,
      ),
      0.085,
    );
    label(
      "Feet · caudal",
      new T.Vector3(
        patientTarget.x,
        patientTarget.y + 0.04,
        bodyBounds.max.z + 0.1,
      ),
      0.085,
    );
    for (const [object, colour] of [
      [source, 0xa4442d],
      [detector, 0x536f79],
    ] as const) {
      const dot = new T.Mesh(
        keep(new T.SphereGeometry(0.009, 12, 8)),
        keep(new T.MeshBasicMaterial({ color: colour, depthTest: false })),
      );
      dot.renderOrder = 6;
      object.add(dot);
    }
    ready = true;
    for (const input of root.querySelectorAll<
      HTMLInputElement | HTMLButtonElement
    >("input,button"))
      input.disabled = false;
    status.hidden = true;
    viewport.dataset.ready = "true";
    resize();
    resetCamera();
    applyPose();
  }
  for (const button of root.querySelectorAll<HTMLButtonElement>("button"))
    button.disabled = true;
  void initialise().catch((error) => {
    if (stopped) return;
    dispose();
    console.error(error);
    status.textContent = `Unable to load the positioning view: ${error instanceof Error ? error.message : String(error)}`;
    viewport.dataset.error = "true";
  });
  function dispose(): void {
    if (stopped) return;
    stopped = true;
    ready = false;
    guides.dispose();
    listeners.abort();
    resizeObserver.disconnect();
    orbit.dispose();
    for (const resource of resources) resource.dispose();
    renderer.dispose();
  }
  window.addEventListener(
    "pagehide",
    (event) => {
      if (!event.persisted) dispose();
    },
    { signal: listeners.signal },
  );
  return dispose;
}
