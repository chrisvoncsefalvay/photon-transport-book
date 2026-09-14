/**
 * Recorded photon-count frames drive display-only phosphor persistence.
 * No target image, browser sampling, attenuation or transport is evaluated here.
 */
import {
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  HalfFloatType,
  LinearFilter,
  Mesh,
  NearestFilter,
  NoBlending,
  NoColorSpace,
  OrthographicCamera,
  Scene,
  ShaderMaterial,
  Vector2,
  Vector3,
  Vector4,
  WebGLRenderTarget,
  type Texture,
  type WebGLRenderer,
} from "three";

export type Point = { x: number; y: number };
export interface PhosphorRecord {
  width: number;
  height: number;
  frameCount: number;
  frameRate: number;
  atlasColumns: number;
  /** Illustrative display memory, in seconds; not a calibrated detector response. */
  tauSeconds: number;
}
export type FrameClock = { playedFrames: number; phase: number };
export type FrameImpact = {
  frameIndex: number;
  memoryWeight: number;
  flashWeight: number;
};
export type FrameAdvance = {
  decay: readonly [number, number];
  impacts: FrameImpact[];
};
export const FLASH_TAU_SECONDS = 0.08;
const NO_ADVANCE: FrameAdvance = { decay: [1, 1], impacts: [] };
const gaussianNeighbour = Math.exp(-1 / (2 * 0.65 ** 2));
/** Separable, unit-sum display PSF at recorded pixel centres. */
export const PHOSPHOR_KERNEL = [
  gaussianNeighbour / (1 + 2 * gaussianNeighbour),
  1 / (1 + 2 * gaussianNeighbour),
  gaussianNeighbour / (1 + 2 * gaussianNeighbour),
] as const;

function validateTiming(record: PhosphorRecord) {
  if (
    !Number.isSafeInteger(record.frameCount) ||
    record.frameCount < 1 ||
    !Number.isFinite(record.frameRate) ||
    record.frameRate <= 0 ||
    !Number.isFinite(record.tauSeconds) ||
    record.tauSeconds <= 0
  )
    throw new Error("Invalid recorded-frame timing or phosphor persistence.");
  // A raw 8-bit count frame, even at every pixel on every replay, must fit FP16.
  if (
    255 / -Math.expm1(-1 / (record.frameRate * record.tauSeconds)) > 65504 ||
    255 / -Math.expm1(-1 / (record.frameRate * FLASH_TAU_SECONDS)) > 65504
  )
    throw new Error(
      "Phosphor persistence exceeds the half-float display range.",
    );
}

/**
 * First frame 0 arrives at 1 / frameRate. Sum repeated recordings analytically,
 * so catching up never needs more than frameCount passes or drops observations.
 * This is an exact schedule in real arithmetic; GPU storage remains FP16.
 */
export function advanceFrameClock(
  clock: FrameClock,
  dtSeconds: number,
  record: PhosphorRecord,
  paused = false,
): FrameAdvance {
  validateTiming(record);
  if (!Number.isFinite(dtSeconds) || dtSeconds < 0)
    throw new Error("Elapsed display time must be finite and nonnegative.");
  if (paused || dtSeconds === 0) return NO_ADVANCE;
  let totalFrames = clock.phase + dtSeconds * record.frameRate;
  const nearestFrame = Math.round(totalFrames);
  if (Math.abs(totalFrames - nearestFrame) < 1e-10) totalFrames = nearestFrame;
  const crossedFrames = Math.floor(totalFrames);
  if (!Number.isSafeInteger(clock.playedFrames + crossedFrames))
    throw new Error("Recorded-frame clock exceeds the exact integer range.");
  const phase = totalFrames - crossedFrames;
  const cycleSeconds = record.frameCount / record.frameRate;
  const impacts: FrameImpact[] = [];
  for (
    let offset = 0;
    offset < Math.min(crossedFrames, record.frameCount);
    offset++
  ) {
    const repetitions =
      1 + Math.floor((crossedFrames - 1 - offset) / record.frameCount);
    const lastOffset = offset + (repetitions - 1) * record.frameCount;
    const age = (crossedFrames - lastOffset - 1 + phase) / record.frameRate;
    const weight = (tau: number) =>
      Math.exp(-age / tau) *
      (-Math.expm1((-repetitions * cycleSeconds) / tau) /
        -Math.expm1(-cycleSeconds / tau));
    impacts.push({
      frameIndex: (clock.playedFrames + offset) % record.frameCount,
      memoryWeight: weight(record.tauSeconds),
      flashWeight: weight(FLASH_TAU_SECONDS),
    });
  }
  clock.playedFrames += crossedFrames;
  clock.phase = phase;
  return {
    decay: [
      Math.exp(-dtSeconds / record.tauSeconds),
      Math.exp(-dtSeconds / FLASH_TAU_SECONDS),
    ],
    impacts,
  };
}

const vertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = position.xy * 0.5 + 0.5;
    gl_Position = vec4(position, 1.0);
  }
`;
const fragmentShader = /* glsl */ `
  uniform sampler2D uPrevious;
  uniform sampler2D uAtlas;
  uniform vec2 uSize;
  uniform vec2 uAtlasSize;
  uniform vec2 uDecay;
  uniform vec2 uWeight;
  uniform vec3 uKernel;
  uniform float uFrame;
  uniform float uColumns;
  varying vec2 vUv;

  float recordedCount(vec2 pixel, vec2 tile) {
    // Zero outside the acquired field: never wrap, clamp or sample another tile.
    if (pixel.x < 0.0 || pixel.y < 0.0 ||
        pixel.x >= uSize.x || pixel.y >= uSize.y) return 0.0;
    vec2 atlasUv = (tile * uSize + pixel + 0.5) / uAtlasSize;
    return floor(texture2D(uAtlas, atlasUv).r * 255.0 + 0.5);
  }
  void main() {
    vec2 field = texture2D(uPrevious, vUv).rg * uDecay;
    if (uFrame >= 0.0) {
      vec2 tile = vec2(mod(uFrame, uColumns), floor(uFrame / uColumns));
      vec2 pixel = floor(vUv * uSize);
      float counts = 0.0;
      for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
          counts += recordedCount(pixel + vec2(float(x), float(y)), tile)
            * uKernel[x + 1] * uKernel[y + 1];
        }
      }
      field += counts * uWeight;
    }
    gl_FragColor = vec4(field, 0.0, 1.0);
  }
`;

/**
 * R stores persistent light; G stores the fast scintillation channel. Texture
 * UV (0,0) denotes recorded pixel (0,0), with V increasing down the source atlas.
 * The caller owns atlas and the display colour map. This helper owns no RAF loop.
 */
export function createPhosphor(
  renderer: WebGLRenderer,
  atlas: Texture,
  record: PhosphorRecord,
) {
  validateTiming(record);
  if (
    !Number.isSafeInteger(record.width) ||
    record.width < 1 ||
    !Number.isSafeInteger(record.height) ||
    record.height < 1 ||
    !Number.isSafeInteger(record.atlasColumns) ||
    record.atlasColumns < 1
  )
    throw new Error("Invalid recorded count atlas dimensions.");
  if (
    atlas.colorSpace !== NoColorSpace ||
    atlas.flipY ||
    atlas.minFilter !== NearestFilter ||
    atlas.magFilter !== NearestFilter
  )
    throw new Error(
      "Count atlas requires no colour conversion, no Y flip and nearest sampling.",
    );
  const atlasWidth = record.width * record.atlasColumns;
  const atlasHeight =
    record.height * Math.ceil(record.frameCount / record.atlasColumns);
  const atlasImage = atlas.image as
    { width?: number; height?: number } | undefined;
  if (atlasImage?.width !== atlasWidth || atlasImage?.height !== atlasHeight)
    throw new Error("Count atlas dimensions do not match its recording.");
  if (!renderer.extensions.has("EXT_color_buffer_float"))
    throw new Error("Half-float phosphor rendering is unavailable.");

  const createTarget = () =>
    new WebGLRenderTarget(record.width, record.height, {
      type: HalfFloatType,
      colorSpace: NoColorSpace,
      // Arithmetic reads exact pixel centres; fractional display samples can
      // interpolate the field without adding spatial diffusion each update.
      minFilter: LinearFilter,
      magFilter: LinearFilter,
      generateMipmaps: false,
      depthBuffer: false,
      stencilBuffer: false,
    });
  let read = createTarget(),
    write = createTarget();
  let stillTarget: WebGLRenderTarget | undefined;
  let still = false,
    disposed = false;
  const clock: FrameClock = { playedFrames: 0, phase: 0 };
  const uniforms = {
    uPrevious: { value: read.texture },
    uAtlas: { value: atlas },
    uSize: { value: new Vector2(record.width, record.height) },
    uAtlasSize: {
      value: new Vector2(atlasWidth, atlasHeight),
    },
    uDecay: { value: new Vector2(1, 1) },
    uWeight: { value: new Vector2(0, 0) },
    uKernel: { value: new Vector3(...PHOSPHOR_KERNEL) },
    uFrame: { value: -1 },
    uColumns: { value: record.atlasColumns },
  };
  const material = new ShaderMaterial({
    uniforms,
    vertexShader,
    fragmentShader,
    depthTest: false,
    depthWrite: false,
    blending: NoBlending,
    toneMapped: false,
  });
  const geometry = new BufferGeometry();
  geometry.setAttribute(
    "position",
    new Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3),
  );
  const quad = new Mesh(geometry, material);
  quad.frustumCulled = false;
  const scene = new Scene();
  scene.add(quad);
  const camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const savedViewport = new Vector4(),
    savedScissor = new Vector4();
  const savedClear = new Color();

  function withRendererState(action: () => void) {
    const target = renderer.getRenderTarget();
    const cubeFace = renderer.getActiveCubeFace();
    const mipLevel = renderer.getActiveMipmapLevel();
    renderer.getViewport(savedViewport);
    renderer.getScissor(savedScissor);
    renderer.getClearColor(savedClear);
    const clearAlpha = renderer.getClearAlpha();
    const scissorTest = renderer.getScissorTest();
    const autoClear = renderer.autoClear;
    const xrEnabled = renderer.xr.enabled;
    renderer.autoClear = false;
    renderer.xr.enabled = false;
    renderer.setScissorTest(false);
    try {
      action();
    } finally {
      renderer.setRenderTarget(target, cubeFace, mipLevel);
      renderer.setViewport(savedViewport);
      renderer.setScissor(savedScissor);
      renderer.setScissorTest(scissorTest);
      renderer.setClearColor(savedClear, clearAlpha);
      renderer.autoClear = autoClear;
      renderer.xr.enabled = xrEnabled;
    }
  }
  function clear(target: WebGLRenderTarget) {
    renderer.setRenderTarget(target);
    renderer.setClearColor(0, 0);
    renderer.clear(true, false, false);
  }
  function pass(
    previous: WebGLRenderTarget,
    next: WebGLRenderTarget,
    frame: number,
    memoryDecay: number,
    flashDecay: number,
    memoryWeight: number,
    flashWeight: number,
  ) {
    uniforms.uPrevious.value = previous.texture;
    uniforms.uFrame.value = frame;
    uniforms.uDecay.value.set(memoryDecay, flashDecay);
    uniforms.uWeight.value.set(memoryWeight, flashWeight);
    renderer.setRenderTarget(next);
    renderer.render(scene, camera);
  }
  withRendererState(() => {
    clear(read);
    clear(write);
  });

  return {
    get texture() {
      return still && stillTarget ? stillTarget.texture : read.texture;
    },
    get elapsed() {
      return (clock.playedFrames + clock.phase) / record.frameRate;
    },
    get playedFrames() {
      return clock.playedFrames;
    },
    advance(dtSeconds: number, paused = false) {
      if (disposed || still) return;
      const step = advanceFrameClock(clock, dtSeconds, record, paused);
      if (step === NO_ADVANCE) return;
      withRendererState(() => {
        if (step.impacts.length === 0) {
          pass(read, write, -1, step.decay[0], step.decay[1], 0, 0);
          [read, write] = [write, read];
        } else {
          for (let index = 0; index < step.impacts.length; index++) {
            const impact = step.impacts[index]!;
            pass(
              read,
              write,
              impact.frameIndex,
              index === 0 ? step.decay[0] : 1,
              index === 0 ? step.decay[1] : 1,
              impact.memoryWeight,
              impact.flashWeight,
            );
            [read, write] = [write, read];
          }
        }
      });
    },
    setStill(enabled: boolean) {
      if (disposed) return;
      if (enabled && !stillTarget) {
        // Stationary frame-boundary field of the actual recorded-frame mean.
        // Use separate targets: reduced-motion viewing never changes playback.
        let meanRead = createTarget(),
          meanWrite = createTarget();
        const weight =
          1 /
          (record.frameCount *
            -Math.expm1(-1 / (record.frameRate * record.tauSeconds)));
        try {
          withRendererState(() => {
            clear(meanRead);
            clear(meanWrite);
            for (let frame = 0; frame < record.frameCount; frame++) {
              pass(meanRead, meanWrite, frame, 1, 0, weight, 0);
              [meanRead, meanWrite] = [meanWrite, meanRead];
            }
          });
          stillTarget = meanRead;
        } catch (error) {
          meanRead.dispose();
          throw error;
        } finally {
          meanWrite.dispose();
        }
      }
      still = enabled;
    },
    reset() {
      if (disposed) return;
      // Context restoration invalidates cached GPU contents as well as playback.
      // The next setStill(true) must build its recorded mean on the new context.
      stillTarget?.dispose();
      stillTarget = undefined;
      withRendererState(() => {
        clear(read);
        clear(write);
      });
      clock.playedFrames = 0;
      clock.phase = 0;
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      read.dispose();
      write.dispose();
      stillTarget?.dispose();
      geometry.dispose();
      material.dispose();
    },
  };
}
