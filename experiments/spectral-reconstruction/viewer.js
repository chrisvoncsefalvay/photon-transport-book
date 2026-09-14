import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const interactive = document.getElementById("interactive");
const stage = document.getElementById("stage");
const status = document.getElementById("status");
const fallback = document.getElementById("static-plate");

function fail(message) {
  interactive.hidden = true;
  fallback.open = true;
  status.textContent = message;
  window.dptViewer = { ready: false, fallback: true };
}

function decode(base64, width, read) {
  const text = atob(base64);
  const bytes = new Uint8Array(text.length);
  for (let index = 0; index < text.length; index++)
    bytes[index] = text.charCodeAt(index);
  if (bytes.byteLength % width)
    throw new Error("Malformed recorded mesh buffer");
  const values = new Array(bytes.byteLength / width);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index++)
    values[index] = read(view, index * width);
  return values;
}

function initialise(data) {
  document.getElementById("provenance").textContent = JSON.stringify(
    data.provenance,
    null,
    2,
  );
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    powerPreference: "low-power",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1;
  renderer.domElement.setAttribute("aria-hidden", "true");
  stage.prepend(renderer.domElement);
  interactive.hidden = false;

  const size = new THREE.Vector3(...data.boundsSize);
  const radius = size.length() / 2;
  const centre = new THREE.Vector3(...data.centre);
  const scenes = data.meshes.map((record) => {
    const source = decode(record.positionsFloat64LE, 8, (view, offset) =>
      view.getFloat64(offset, true),
    );
    const indices = decode(record.trianglesUint32LE, 4, (view, offset) =>
      view.getUint32(offset, true),
    );
    if (
      source.length !== record.vertexCount * 3 ||
      indices.length !== record.triangleCount * 3
    ) {
      throw new Error(
        "Recorded geometry dimensions differ from their provenance",
      );
    }
    // Preserve topology and source coordinates; only centre the display and
    // convert positions to the ordinary FP32 WebGL vertex buffer. Lighting
    // normals are interpolated without changing, welding or smoothing vertices.
    const positions = new Float32Array(source.length);
    for (let index = 0; index < source.length; index++)
      positions[index] = source[index] - centre.getComponent(index % 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    const material = new THREE.MeshStandardMaterial({
      color: 0xe9dfc9,
      roughness: 0.62,
      metalness: 0,
      side: THREE.DoubleSide,
    });
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101c23);
    scene.add(new THREE.Mesh(geometry, material));
    scene.add(new THREE.HemisphereLight(0xdde9e8, 0x23343c, 2.2));
    for (const [colour, intensity, position] of [
      [0xfff1d6, 3.2, [-3, -4, 6]],
      [0x95c5e0, 1.2, [4, 2, 0]],
      [0xffffff, 1.6, [2, 4, 5]],
    ]) {
      const light = new THREE.DirectionalLight(colour, intensity);
      light.position.set(...position);
      scene.add(light);
    }
    return scene;
  });

  const camera = new THREE.OrthographicCamera(
    -radius,
    radius,
    radius,
    -radius,
    radius * 0.01,
    radius * 12,
  );
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, stage);
  controls.enableDamping = false;
  controls.autoRotate = false;
  controls.screenSpacePanning = true;
  controls.minZoom = 0.25;
  controls.maxZoom = 8;
  controls.zoomSpeed = 0.8;
  controls.rotateSpeed = 0.65;
  const panes = [...stage.querySelectorAll(".viewport")];
  let halfHeight = radius;
  let scheduled = false;
  let currentPreset = "oblique";
  let applyingPreset = false;

  function markPreset(name) {
    currentPreset = name;
    document.querySelectorAll("[data-view]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.view === name));
    });
  }

  function fit() {
    const pane = panes[0].getBoundingClientRect();
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);
    const right = new THREE.Vector3()
      .crossVectors(direction, camera.up)
      .normalize();
    const up = new THREE.Vector3().crossVectors(right, direction).normalize();
    const halfWidth =
      (Math.abs(right.x) * size.x +
        Math.abs(right.y) * size.y +
        Math.abs(right.z) * size.z) /
      2;
    const projectedHeight =
      (Math.abs(up.x) * size.x +
        Math.abs(up.y) * size.y +
        Math.abs(up.z) * size.z) /
      2;
    halfHeight =
      Math.max(projectedHeight, halfWidth / (pane.width / pane.height)) * 1.15;
  }

  function render() {
    scheduled = false;
    if (interactive.hidden) return;
    const bounds = stage.getBoundingClientRect();
    renderer.setSize(bounds.width, bounds.height, false);
    renderer.setScissorTest(false);
    renderer.setClearColor(0x29383d, 1);
    renderer.clear();
    renderer.setScissorTest(true);
    panes.forEach((pane, index) => {
      const rect = pane.getBoundingClientRect();
      const left = rect.left - bounds.left;
      const bottom = bounds.bottom - rect.bottom;
      camera.left = (-halfHeight * rect.width) / rect.height;
      camera.right = -camera.left;
      camera.top = halfHeight;
      camera.bottom = -halfHeight;
      camera.updateProjectionMatrix();
      renderer.setViewport(left, bottom, rect.width, rect.height);
      renderer.setScissor(left, bottom, rect.width, rect.height);
      renderer.render(scenes[index], camera);
    });
  }

  function requestRender() {
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(render);
    }
  }

  function applyPreset(name) {
    applyingPreset = true;
    const [azimuth, elevation] = {
      oblique: [-75, 12],
      coronal: [-90, 0],
      sagittal: [0, 0],
    }[name];
    const az = THREE.MathUtils.degToRad(azimuth);
    const el = THREE.MathUtils.degToRad(elevation);
    controls.target.set(0, 0, 0);
    camera.position
      .set(
        Math.cos(el) * Math.cos(az),
        Math.cos(el) * Math.sin(az),
        Math.sin(el),
      )
      .multiplyScalar(radius * 3);
    camera.zoom = 1;
    camera.lookAt(controls.target);
    controls.update();
    fit();
    markPreset(name);
    applyingPreset = false;
    requestRender();
  }

  function action(name) {
    if (name === "in" || name === "out") {
      camera.zoom = THREE.MathUtils.clamp(
        camera.zoom * (name === "in" ? 1.2 : 1 / 1.2),
        controls.minZoom,
        controls.maxZoom,
      );
      camera.updateProjectionMatrix();
    } else {
      const offset = camera.position.clone().sub(controls.target);
      const distance = offset.length();
      let azimuth = Math.atan2(offset.y, offset.x);
      let elevation = Math.asin(offset.z / distance);
      azimuth += (({ left: -1, right: 1 }[name] || 0) * Math.PI) / 18;
      elevation = THREE.MathUtils.clamp(
        elevation + (({ up: 1, down: -1 }[name] || 0) * Math.PI) / 18,
        (-Math.PI * 4) / 9,
        (Math.PI * 4) / 9,
      );
      offset
        .set(
          Math.cos(elevation) * Math.cos(azimuth),
          Math.cos(elevation) * Math.sin(azimuth),
          Math.sin(elevation),
        )
        .multiplyScalar(distance);
      camera.position.copy(controls.target).add(offset);
    }
    controls.update();
    markPreset(null);
    requestRender();
  }

  controls.addEventListener("change", () => {
    if (!applyingPreset) markPreset(null);
    requestRender();
  });
  document
    .getElementById("reset")
    .addEventListener("click", () => applyPreset("oblique"));
  document
    .querySelectorAll("[data-view]")
    .forEach((button) =>
      button.addEventListener("click", () => applyPreset(button.dataset.view)),
    );
  document
    .querySelectorAll("[data-action]")
    .forEach((button) =>
      button.addEventListener("click", () => action(button.dataset.action)),
    );
  stage.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    const name = {
      ArrowLeft: "left",
      ArrowRight: "right",
      ArrowUp: "up",
      ArrowDown: "down",
      "+": "in",
      "=": "in",
      "-": "out",
    }[event.key];
    if (name || event.key.toLowerCase() === "r") {
      event.preventDefault();
      if (name) action(name);
      else applyPreset("oblique");
    }
  });
  const observer = new ResizeObserver(() => {
    fit();
    requestRender();
  });
  observer.observe(stage);
  renderer.domElement.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    fail(
      "The interactive graphics context was lost. The verified static comparison is shown below.",
    );
  });
  applyPreset("oblique");
  fallback.open = false;
  status.textContent =
    "Both panels use one camera and one physical scale. No automatic motion.";
  window.dptViewer = {
    ready: true,
    inspect: () => ({
      cameraCount: 1,
      meshCount: scenes.length,
      autoRotate: controls.autoRotate,
      cameraPosition: camera.position.toArray(),
      target: controls.target.toArray(),
      zoom: camera.zoom,
      currentPreset,
      halfHeight,
      triangleCounts: data.meshes.map((mesh) => mesh.triangleCount),
      viewportSizes: panes.map((pane) => [pane.clientWidth, pane.clientHeight]),
      sharedPhysicalUnits: "mm",
      threeVersion: data.provenance.threeVersion,
    }),
  };
  window.addEventListener(
    "pagehide",
    (event) => {
      if (event.persisted) return;
      observer.disconnect();
      controls.dispose();
      scenes.forEach((scene) =>
        scene.traverse((object) => {
          if (object.isMesh) {
            object.geometry.dispose();
            object.material.dispose();
          }
        }),
      );
      renderer.dispose();
    },
    { once: true },
  );
}

try {
  initialise(JSON.parse(document.getElementById("viewer-data").textContent));
} catch (error) {
  console.error("Recorded mesh viewer:", error);
  fail(
    "Interactive rendering is unavailable. The verified static comparison is shown below.",
  );
}
