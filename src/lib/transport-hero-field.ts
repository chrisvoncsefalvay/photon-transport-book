/** Live display geometry and playback of recorded canonical count observations. */
import * as T from "three";
import { bodyEdges, conditionalLines } from "./carm/conditional-lines";
import { createPhosphor } from "./transport-hero-detector";

type Block = { byteOffset: number; count: number };
interface PelvisLOD {
  buffer: string;
  bufferByteLength: number;
  meshes: {
    label: string;
    positions: Block;
    normals: Block;
    indices: Block;
    triangleCount: number;
  }[];
}
interface ImpactRecord {
  width: number;
  height: number;
  frameRate: number;
  frameCount: number;
  atlasColumns: number;
  atlas: string;
  tauSeconds: number;
  flashTauSeconds: number;
  openBeamCountsPerFrame: number;
  flights: number[][];
  source: [number, number, number];
  detectorCentre: [number, number, number];
  detectorWidth: number;
  detectorHeight: number;
}
const ASSETS = "/generated/transport-hero/";
async function json<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(ASSETS + path, { signal });
  if (!response.ok) throw new Error(`Hero asset unavailable: ${path}`);
  return response.json() as Promise<T>;
}

export async function mountTransportHeroField(
  canvas: HTMLCanvasElement,
  host: HTMLElement,
  signal: AbortSignal,
): Promise<() => void> {
  const [lod, record] = await Promise.all([
    json<PelvisLOD>("pelvis-lod.json", signal),
    json<ImpactRecord>("impact-record.json", signal),
  ]);
  const response = await fetch(ASSETS + lod.buffer, { signal });
  if (!response.ok) throw new Error("Pelvis geometry unavailable");
  const binary = await response.arrayBuffer();
  if (binary.byteLength !== lod.bufferByteLength)
    throw new Error("Incomplete pelvis geometry");
  const atlas = await new T.TextureLoader().loadAsync(ASSETS + record.atlas);
  if (signal.aborted) {
    atlas.dispose();
    return () => {};
  }
  atlas.colorSpace = T.NoColorSpace;
  atlas.flipY = false;
  atlas.minFilter = atlas.magFilter = T.NearestFilter;
  atlas.generateMipmaps = false;
  const resources = new Set<{ dispose(): void }>();
  const keep = <V extends { dispose(): void }>(v: V): V => {
    resources.add(v);
    return v;
  };
  keep(atlas);
  let renderer: T.WebGLRenderer;
  try {
    renderer = new T.WebGLRenderer({ canvas, antialias: true, alpha: true });
  } catch (error) {
    atlas.dispose();
    throw error;
  }
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.outputColorSpace = T.SRGBColorSpace;
  renderer.toneMapping = T.NoToneMapping;
  let phosphor: ReturnType<typeof createPhosphor>;
  try {
    phosphor = createPhosphor(renderer, atlas, record);
  } catch (error) {
    resources.forEach((resource) => resource.dispose());
    renderer.dispose();
    throw error;
  }
  const scene = new T.Scene();
  // Leave space around the source and the complete detector throughout the orbit.
  // Match the canvas aspect ratio so resizing preserves the acquisition geometry.
  const viewHeight = (1180 * 470) / 520;
  const camera = new T.OrthographicCamera(
    -560,
    620,
    700,
    700 - viewHeight,
    1,
    10000,
  );
  camera.up.set(0, 0, 1);
  const viewDirection = new T.Vector3(-1.1, 1.5, 0.2).normalize();
  const target = new T.Vector3(0, 0, 0);
  scene.add(new T.AmbientLight(0xffffff, 0.35));
  const light = new T.DirectionalLight(0xffffff, Math.PI * 0.95);
  light.position.set(-650, 900, 1000);
  scene.add(light);
  const ramp = keep(
    new T.DataTexture(new Uint8Array([115, 180, 225, 255]), 4, 1, T.RedFormat),
  );
  ramp.minFilter = ramp.magFilter = T.NearestFilter;
  ramp.generateMipmaps = false;
  ramp.needsUpdate = true;
  const boneMaterial = keep(
    new T.MeshToonMaterial({
      color: 0xf4f1e8,
      gradientMap: ramp,
      side: T.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    }),
  );
  for (const mesh of lod.meshes) {
    const geometry = keep(new T.BufferGeometry());
    geometry.setAttribute(
      "position",
      new T.BufferAttribute(
        new Float32Array(
          binary,
          mesh.positions.byteOffset,
          mesh.positions.count,
        ),
        3,
      ),
    );
    geometry.setAttribute(
      "normal",
      new T.BufferAttribute(
        new Float32Array(binary, mesh.normals.byteOffset, mesh.normals.count),
        3,
      ),
    );
    geometry.setIndex(
      new T.BufferAttribute(
        new Uint16Array(binary, mesh.indices.byteOffset, mesh.indices.count),
        1,
      ),
    );
    const bone = new T.Mesh(geometry, boneMaterial);
    bone.name = mesh.label;
    const contour = conditionalLines(bodyEdges(geometry), 0x52666b);
    keep(contour.geometry);
    if (Array.isArray(contour.material)) contour.material.forEach(keep);
    else keep(contour.material);
    bone.add(contour);
    scene.add(bone);
  }
  const steady =
    record.openBeamCountsPerFrame /
    (1 - Math.exp(-1 / (record.frameRate * record.tauSeconds)));
  const screenMaterial = keep(
    new T.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: T.DoubleSide,
      uniforms: {
        field: { value: phosphor.texture },
        steady: { value: steady },
      },
      vertexShader: `varying vec2 sensorUV;
      void main(){sensorUV=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
      fragmentShader: `uniform sampler2D field;uniform float steady;varying vec2 sensorUV;
      void main(){
        // An unframed light field: its support dies away before the mesh boundary.
        vec2 q=(sensorUV-0.5)*2.0;
        float edge=1.0-smoothstep(0.66,0.99,length(q));
        if(edge<0.002)discard;
        vec2 e=texture2D(field,vec2(sensorUV.x,1.0-sensorUV.y)).rg;
        float brightness=pow(clamp(e.r/steady,0.0,1.15),0.78);
        float flash=clamp(e.g/2.0,0.0,1.0);
        vec3 dark=vec3(0.045);
        vec3 phosphor=vec3(0.70);
        vec3 emission=mix(dark,phosphor,brightness)+vec3(0.27)*flash;
        gl_FragColor=vec4(emission,edge*.94);
        #include <colorspace_fragment>
      }`,
    }),
  );
  const screen = new T.Mesh(
    keep(new T.PlaneGeometry(record.detectorWidth, record.detectorHeight)),
    screenMaterial,
  );
  screen.rotation.x = Math.PI / 2;
  screen.position.set(...record.detectorCentre);
  screen.renderOrder = -1;
  scene.add(screen);

  // Every displayed flight is a sparse selection of actual nonzero recorded cells.
  // The rest of the counted photons still contribute to the phosphor texture.
  const period = record.frameCount / record.frameRate,
    flightSeconds = 0.8;
  const endpoints: number[] = [],
    arrivals: number[] = [];
  for (let frame = 0; frame < record.frameCount; frame++)
    for (const cell of record.flights[frame]!) {
      const u = ((cell % record.width) + 0.5) / record.width;
      const v = (Math.floor(cell / record.width) + 0.5) / record.height;
      // Displayed flights must land in the visible light field. All recorded
      // photons, including the tapered margin, still enter the accumulator.
      if (Math.hypot((u - 0.5) * 2, (v - 0.5) * 2) > 0.64) continue;
      endpoints.push(
        record.detectorCentre[0] + (u - 0.5) * record.detectorWidth,
        record.detectorCentre[1],
        record.detectorCentre[2] + (0.5 - v) * record.detectorHeight,
      );
      arrivals.push((frame + 1) / record.frameRate);
    }
  const flightUniforms = {
    time: { value: 0 },
    period: { value: period },
    duration: { value: flightSeconds },
    source: { value: new T.Vector3(...record.source) },
    dpr: { value: renderer.getPixelRatio() },
  };
  const flightShader = `uniform float time,period,duration,dpr;uniform vec3 source;
    attribute vec3 endpoint;attribute float arrival;varying float opacity;
    void main(){
      float until=mod(arrival-mod(time,period)+period,period);
      float phase=1.0-until/duration;
      opacity=step(0.0,phase)*smoothstep(0.0,0.12,phase);
      vec3 p=mix(source,endpoint,clamp(phase,0.0,1.0));
      gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.0);
      gl_PointSize=2.15*dpr;
    }`;
  const headsGeometry = keep(new T.BufferGeometry());
  headsGeometry.setAttribute(
    "position",
    new T.Float32BufferAttribute(endpoints, 3),
  );
  headsGeometry.setAttribute(
    "endpoint",
    new T.Float32BufferAttribute(endpoints, 3),
  );
  headsGeometry.setAttribute(
    "arrival",
    new T.Float32BufferAttribute(arrivals, 1),
  );
  const headsMaterial = keep(
    new T.ShaderMaterial({
      uniforms: flightUniforms,
      transparent: true,
      depthWrite: false,
      vertexShader: flightShader,
      fragmentShader:
        `varying float opacity;void main(){float r=length(gl_PointCoord-.5);if(opacity<.01||r>.5)discard;gl_FragColor=vec4(.57,.28,.15,opacity*(1.-smoothstep(.15,.5,r)));#include <colorspace_fragment>}`.replace(
          "#include",
          "\n#include",
        ),
    }),
  );
  const heads = new T.Points(headsGeometry, headsMaterial);
  heads.frustumCulled = false;
  scene.add(heads);
  const lineEndpoints = new Float32Array(endpoints.length * 2),
    lineArrivals = new Float32Array(arrivals.length * 2),
    tail = new Float32Array(arrivals.length * 2);
  for (let i = 0; i < arrivals.length; i++)
    for (let end = 0; end < 2; end++) {
      for (let j = 0; j < 3; j++)
        lineEndpoints[i * 6 + end * 3 + j] = endpoints[i * 3 + j]!;
      lineArrivals[i * 2 + end] = arrivals[i]!;
      tail[i * 2 + end] = end === 0 ? 0.06 : 0;
    }
  const linesGeometry = keep(new T.BufferGeometry());
  linesGeometry.setAttribute(
    "position",
    new T.BufferAttribute(lineEndpoints, 3),
  );
  linesGeometry.setAttribute(
    "endpoint",
    new T.BufferAttribute(lineEndpoints, 3),
  );
  linesGeometry.setAttribute("arrival", new T.BufferAttribute(lineArrivals, 1));
  linesGeometry.setAttribute("tail", new T.BufferAttribute(tail, 1));
  const linesMaterial = keep(
    new T.ShaderMaterial({
      uniforms: flightUniforms,
      transparent: true,
      depthWrite: false,
      vertexShader: flightShader
        .replace(
          "attribute vec3 endpoint;",
          "attribute float tail;attribute vec3 endpoint;",
        )
        .replace("clamp(phase,0.0,1.0)", "clamp(phase-tail,0.0,1.0)"),
      fragmentShader:
        "varying float opacity;void main(){if(opacity<.01)discard;gl_FragColor=vec4(.63,.44,.28,opacity*.27);\n#include <colorspace_fragment>\n}",
    }),
  );
  const rays = new T.LineSegments(linesGeometry, linesMaterial);
  rays.frustumCulled = false;
  scene.add(rays);
  const source = new T.Mesh(
    keep(new T.SphereGeometry(4.8, 12, 8)),
    keep(new T.MeshBasicMaterial({ color: 0x91412f })),
  );
  source.position.set(...record.source);
  scene.add(source);

  let disposed = false,
    lost = false,
    frame: number | undefined,
    previous: number | undefined;
  let cameraTime = 0;
  const animated = () => host.dataset.state === "animated";
  const canRun = () =>
    !disposed &&
    !lost &&
    animated() &&
    host.dataset.paused !== "true" &&
    !document.hidden;
  const render = () => {
    const angle = Math.sin(cameraTime * 0.22) * 0.07;
    camera.position
      .set(
        viewDirection.x * Math.cos(angle) - viewDirection.y * Math.sin(angle),
        viewDirection.x * Math.sin(angle) + viewDirection.y * Math.cos(angle),
        viewDirection.z + Math.sin(cameraTime * 0.17) * 0.014,
      )
      .multiplyScalar(3000);
    camera.lookAt(target);
    screenMaterial.uniforms.field!.value = phosphor.texture;
    flightUniforms.time.value = phosphor.elapsed;
    heads.visible = rays.visible = animated();
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);
  };
  const tick = (now: number) => {
    frame = undefined;
    if (!canRun()) {
      previous = undefined;
      return;
    }
    const dt =
      previous === undefined ? 0 : Math.min((now - previous) / 1000, 0.1);
    previous = now;
    cameraTime += dt;
    phosphor.advance(dt);
    render();
    frame = requestAnimationFrame(tick);
  };
  const sync = () => {
    if (disposed || lost) return;
    phosphor.setStill(!animated() && host.dataset.state === "still");
    if (!animated()) render();
    if (canRun()) {
      if (frame === undefined) frame = requestAnimationFrame(tick);
    } else {
      if (frame !== undefined) cancelAnimationFrame(frame);
      frame = undefined;
      previous = undefined;
    }
  };
  const resize = () => {
    if (disposed || lost) return;
    const rect = host
      .querySelector(".transport-hero__art")!
      .getBoundingClientRect();
    renderer.setSize(
      Math.max(1, Math.round(rect.width)),
      Math.max(1, Math.round(rect.height)),
      false,
    );
    render();
  };
  const events = new AbortController();
  canvas.addEventListener(
    "webglcontextlost",
    (event) => {
      event.preventDefault();
      lost = true;
      if (frame !== undefined) cancelAnimationFrame(frame);
      frame = undefined;
      previous = undefined;
      canvas.hidden = true;
      host.querySelector<HTMLElement>(
        ".transport-hero__fallback",
      )!.style.display = "";
      host.querySelector("button")!.hidden = true;
    },
    { signal: events.signal },
  );
  canvas.addEventListener(
    "webglcontextrestored",
    () => {
      lost = false;
      phosphor.reset();
      canvas.hidden = false;
      host.querySelector<HTMLElement>(
        ".transport-hero__fallback",
      )!.style.display = "none";
      host.querySelector("button")!.hidden = false;
      resize();
      sync();
    },
    { signal: events.signal },
  );
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(host);
  const stateObserver = new MutationObserver(sync);
  stateObserver.observe(host, {
    attributes: true,
    attributeFilter: ["data-state", "data-paused"],
  });
  document.addEventListener("visibilitychange", sync, {
    signal: events.signal,
  });
  host.dataset.triangles = String(
    lod.meshes.reduce((n, m) => n + m.triangleCount, 0),
  );
  host.dataset.renderer = "three";
  resize();
  sync();
  return () => {
    disposed = true;
    if (frame !== undefined) cancelAnimationFrame(frame);
    events.abort();
    resizeObserver.disconnect();
    stateObserver.disconnect();
    phosphor.dispose();
    resources.forEach((resource) => resource.dispose());
    renderer.dispose();
    delete host.dataset.renderer;
    delete host.dataset.triangles;
  };
}
