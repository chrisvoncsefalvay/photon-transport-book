// Optional display-only asset authoring; normal builds use the checked-in PNG.
// Requires Playwright and a Chromium executable. No transport is evaluated here.
import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const root = fileURLToPath(new URL("../../", import.meta.url));
const { values } = parseArgs({
  options: {
    playwright: { type: "string" },
    chromium: { type: "string" },
  },
});
const playwright = values.playwright
  ? await import(pathToFileURL(resolve(values.playwright)).href)
  : await import("playwright");
const meshPath = "public/generated/introduction-pose-sensitivity/meshes.json";
const meshBytes = await readFile(resolve(root, meshPath));
const meshData = JSON.parse(meshBytes);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const config = {
  width: 520,
  height: 470,
  pixelRatio: 2,
  source: [0, 750, 0],
  detectorCentre: [0, -450, 0],
  detectorWidth: 600,
  detectorHeight: 320,
  viewDirection: [-1.1, 1.5, 0.2],
  artworkBounds: [16, 155, 504, 445],
  encounterCount: 192,
  samplingSeed: 184731,
};
const threeDir = dirname(fileURLToPath(import.meta.resolve("three")));
const resources = new Map([
  ["/three.module.js", await readFile(resolve(threeDir, "three.module.js"))],
  ["/three.core.js", await readFile(resolve(threeDir, "three.core.js"))],
  ["/meshes.json", meshBytes],
]);
const server = createServer((request, response) => {
  if (request.url === "/") {
    response.setHeader("Content-Type", "text/html");
    response.end("<!doctype html><html><body></body></html>");
    return;
  }
  const body = resources.get(request.url);
  if (!body) {
    response.writeHead(404).end();
    return;
  }
  response.setHeader(
    "Content-Type",
    request.url.endsWith(".json") ? "application/json" : "text/javascript",
  );
  response.end(body);
});
await new Promise((ready) => server.listen(0, "127.0.0.1", ready));
let browser;
try {
  browser = await playwright.chromium.launch({
    ...(values.chromium ? { executablePath: resolve(values.chromium) } : {}),
    headless: true,
    args: [
      "--no-sandbox",
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
    ],
  });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result = await page.evaluate(async (settings) => {
    const T = await import("/three.module.js");
    const payload = await (await fetch("/meshes.json")).json();
    const renderer = new T.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(settings.pixelRatio);
    renderer.setSize(settings.width, settings.height);
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = T.SRGBColorSpace;
    const scene = new T.Scene();
    const viewDirection = new T.Vector3(...settings.viewDirection).normalize();
    const camera = new T.OrthographicCamera(-1, 1, 1, -1, 1, 10000);
    camera.up.set(0, 0, 1);
    camera.position.copy(viewDirection).multiplyScalar(3000);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const source = new T.Vector3(...settings.source);
    const centre = new T.Vector3(...settings.detectorCentre);
    const corners = [
      [-1, 1],
      [1, 1],
      [1, -1],
      [-1, -1],
    ].map(([u, v]) =>
      centre
        .clone()
        .add(
          new T.Vector3(
            (u * settings.detectorWidth) / 2,
            0,
            (v * settings.detectorHeight) / 2,
          ),
        ),
    );
    const allPoints = [source, ...corners];
    for (const item of payload.meshes)
      for (let index = 0; index < item.positions.length; index += 3)
        allPoints.push(new T.Vector3().fromArray(item.positions, index));
    const viewBounds = new T.Box3().setFromPoints(
      allPoints.map((point) =>
        point.clone().applyMatrix4(camera.matrixWorldInverse),
      ),
    );
    const [left, top, right, bottom] = settings.artworkBounds;
    const scale = Math.min(
      (right - left) / (viewBounds.max.x - viewBounds.min.x),
      (bottom - top) / (viewBounds.max.y - viewBounds.min.y),
    );
    const viewWidth = settings.width / scale;
    const viewHeight = settings.height / scale;
    const mid = viewBounds.getCenter(new T.Vector3());
    const cameraX = mid.x - ((left + right) / 2 - settings.width / 2) / scale;
    const cameraY = mid.y + ((top + bottom) / 2 - settings.height / 2) / scale;
    camera.left = cameraX - viewWidth / 2;
    camera.right = cameraX + viewWidth / 2;
    camera.top = cameraY + viewHeight / 2;
    camera.bottom = cameraY - viewHeight / 2;
    camera.updateProjectionMatrix();
    const ramp = new T.DataTexture(
      new Uint8Array([100, 158, 210, 255]),
      4,
      1,
      T.RedFormat,
    );
    ramp.minFilter = ramp.magFilter = T.NearestFilter;
    ramp.generateMipmaps = false;
    ramp.needsUpdate = true;
    scene.add(new T.AmbientLight(0xffffff, 0.25));
    const key = new T.DirectionalLight(0xffffff, Math.PI * 0.95);
    key.position.set(-650, 900, 1000);
    scene.add(key);
    const meshes = [];
    let triangleCount = 0;
    for (const item of payload.meshes) {
      const geometry = new T.BufferGeometry();
      geometry.setAttribute(
        "position",
        new T.Float32BufferAttribute(item.positions, 3),
      );
      geometry.setIndex(item.indices);
      geometry.computeVertexNormals();
      const material = new T.MeshToonMaterial({
        color: 0xf4f1e8,
        gradientMap: ramp,
        side: T.DoubleSide,
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      });
      const mesh = new T.Mesh(geometry, material);
      scene.add(mesh);
      meshes.push(mesh);
      triangleCount += item.indices.length / 3;

      // For a fixed orthographic camera, exact triangle adjacency determines
      // silhouette edges once. Hidden edges are rejected by the depth buffer.
      const edges = new Map();
      const vertex = (index) =>
        new T.Vector3().fromArray(item.positions, index * 3);
      for (let i = 0; i < item.indices.length; i += 3) {
        const indices = item.indices.slice(i, i + 3);
        const vertices = indices.map(vertex);
        const normal = vertices[1]
          .clone()
          .sub(vertices[0])
          .cross(vertices[2].clone().sub(vertices[0]));
        const front = normal.dot(viewDirection) > 0;
        for (let j = 0; j < 3; j++) {
          const a = indices[j],
            b = indices[(j + 1) % 3];
          const id = a < b ? `${a},${b}` : `${b},${a}`;
          const record = edges.get(id);
          if (record) record.faces.push(front);
          else edges.set(id, { a, b, faces: [front] });
        }
      }
      const lines = [];
      for (const edge of edges.values())
        if (
          edge.faces.length === 1 ||
          edge.faces.some((front) => front !== edge.faces[0])
        )
          lines.push(vertex(edge.a), vertex(edge.b));
      const contour = new T.LineSegments(
        new T.BufferGeometry().setFromPoints(lines),
        new T.LineBasicMaterial({ color: 0x425d68, depthWrite: false }),
      );
      contour.renderOrder = 2;
      scene.add(contour);
    }
    const round = (value) => Math.round(value * 10000) / 10000;
    const project = (point) => {
      const projected = point.clone().project(camera);
      return {
        x: round(((projected.x + 1) * settings.width) / 2),
        y: round(((1 - projected.y) * settings.height) / 2),
      };
    };
    const encounters = [];
    let seed = settings.samplingSeed;
    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    const raycaster = new T.Raycaster();
    scene.updateMatrixWorld(true);
    for (
      let attempt = 0;
      encounters.length < settings.encounterCount && attempt < 10000;
      attempt++
    ) {
      const mesh = meshes[Math.floor(random() * meshes.length)];
      const positions = mesh.geometry.getAttribute("position");
      const target = new T.Vector3().fromBufferAttribute(
        positions,
        Math.floor(random() * positions.count),
      );
      const direction = target.clone().sub(source);
      const t = (centre.y - source.y) / direction.y;
      const detector = source.clone().addScaledVector(direction, t);
      const u = detector.x / settings.detectorWidth + 0.5;
      const v = 0.5 - detector.z / settings.detectorHeight;
      if (u < 0 || u > 1 || v < 0 || v > 1) continue;
      raycaster.set(source, direction.normalize());
      const hit = raycaster.intersectObjects(meshes, false)[0];
      if (!hit) continue;
      encounters.push({ ...project(hit.point), u: round(u), v: round(v) });
    }
    if (encounters.length !== settings.encounterCount)
      throw new Error(
        "Insufficient mesh encounters inside the recorded detector field.",
      );
    renderer.render(scene, camera);
    return {
      png: renderer.domElement.toDataURL("image/png").split(",")[1],
      width: settings.width,
      height: settings.height,
      source: project(source),
      detectorCorners: corners.map(project),
      encounters,
      camera: {
        type: "orthographic",
        position: camera.position.toArray(),
        up: camera.up.toArray(),
        target: [0, 0, 0],
        left: camera.left,
        right: camera.right,
        top: camera.top,
        bottom: camera.bottom,
        near: camera.near,
        far: camera.far,
      },
      render: {
        threeRevision: T.REVISION,
        pixelRatio: settings.pixelRatio,
        triangleCount,
      },
    };
  }, config);
  const png = Buffer.from(result.png, "base64");
  delete result.png;
  const output = resolve(root, "public/generated/transport-hero");
  await mkdir(output, { recursive: true });
  await writeFile(resolve(output, "pelvis.png"), png);
  await writeFile(
    resolve(output, "scene.json"),
    JSON.stringify(
      {
        schemaVersion: 1,
        ...result,
        provenance: {
          kind: "display-only mesh rendering",
          meshPath,
          meshSha256: sha256(meshBytes),
          imageSha256: sha256(png),
          generatorSha256: sha256(
            await readFile(fileURLToPath(import.meta.url)),
          ),
          meshAttribution: meshData.attribution,
          browserVersion: browser.version(),
          settings: config,
          geometry:
            "Unmodified RAS-mm-pivot-centred display meshes and recorded source/detector geometry; one shared orthographic projection.",
          encounters:
            "Deterministically sampled mesh vertices define source rays; first mesh intersections are recorded. Samples illustrate paths, not interaction probabilities or a transport calculation.",
          regeneration:
            "node tools/public/generate-transport-hero-assets.mjs [--playwright /path/to/playwright/index.mjs] [--chromium /path/to/chromium]",
        },
      },
      null,
      2,
    ) + "\n",
  );
  console.log(
    JSON.stringify({
      output,
      imageBytes: png.length,
      source: result.source,
      detectorCorners: result.detectorCorners,
      render: result.render,
    }),
  );
} finally {
  if (browser) await browser.close();
  await new Promise((done) => server.close(done));
}
