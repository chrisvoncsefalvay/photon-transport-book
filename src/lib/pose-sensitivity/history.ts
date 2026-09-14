import { PARAMETERS, type Parameter, type RecordedPose } from "./data";

export interface DisplayedProjection {
  readonly pose: RecordedPose;
  /** Decoded recorded samples; callers must keep these arrays unchanged. */
  readonly values: Float32Array;
}

export interface ProjectionDelta {
  readonly values: Float64Array;
  readonly limit: number;
}

/** Subtract recorded binary32 samples in binary64, without a display transform. */
export function projectionDelta(
  current: Float32Array,
  previous?: Float32Array,
): ProjectionDelta {
  if (previous && previous.length !== current.length)
    throw new Error("Projection history dimensions differ");
  const values = new Float64Array(current.length);
  let limit = 0;
  for (let i = 0; i < current.length; i++) {
    const a = current[i]!,
      b = previous?.[i] ?? a;
    if (!Number.isFinite(a) || !Number.isFinite(b))
      throw new Error("Projection history contains nonfinite samples");
    const difference = a - b;
    values[i] = difference;
    limit = Math.max(limit, Math.abs(difference));
  }
  return { values, limit };
}

/** Zero is neutral; negative differences are blue, positive differences rust. */
export function deltaPixels(delta: ProjectionDelta): Uint8ClampedArray {
  if (!Number.isFinite(delta.limit) || delta.limit < 0)
    throw new Error("Invalid delta colour range");
  const pixels = new Uint8ClampedArray(delta.values.length * 4);
  const neutral = [247, 244, 238],
    blue = [53, 95, 120],
    rust = [169, 87, 50];
  const denominator = Math.asinh(50);
  for (let i = 0; i < delta.values.length; i++) {
    const value = delta.values[i]!;
    if (!Number.isFinite(value) || Math.abs(value) > delta.limit)
      throw new Error("Delta sample exceeds its finite colour range");
    const weight =
      delta.limit === 0
        ? 0
        : Math.asinh((Math.abs(value) / delta.limit) * 50) / denominator;
    const colour = value < 0 ? blue : rust;
    for (let channel = 0; channel < 3; channel++)
      pixels[4 * i + channel] =
        neutral[channel]! + weight * (colour[channel]! - neutral[channel]!);
    pixels[4 * i + 3] = 255;
  }
  return pixels;
}

/** Values belong to the committed single-axis record, never to a drag preview. */
export function committedCoordinates(
  pose: RecordedPose,
): Record<Parameter, number> {
  return Object.fromEntries(
    PARAMETERS.map((axis) => [axis, pose.parameter === axis ? pose.value : 0]),
  ) as Record<Parameter, number>;
}

/** Only a successful latest request advances displayed-frame history. */
export class DisplayedProjectionHistory {
  #generation = 0;
  #pending: number | undefined;
  #current: DisplayedProjection | undefined;
  #previous: DisplayedProjection | undefined;
  #delta: ProjectionDelta | undefined;

  get current(): DisplayedProjection | undefined {
    return this.#current;
  }
  get previous(): DisplayedProjection | undefined {
    return this.#previous;
  }
  get delta(): ProjectionDelta | undefined {
    if (!this.#current) return undefined;
    return (this.#delta ??= projectionDelta(
      this.#current.values,
      this.#previous?.values,
    ));
  }
  initialise(frame: DisplayedProjection): void {
    if (this.#current)
      throw new Error("Projection history already initialised");
    this.#delta = projectionDelta(frame.values);
    this.#current = frame;
  }
  request(): number {
    this.#pending = ++this.#generation;
    return this.#pending;
  }
  isLatest(token: number): boolean {
    return token === this.#pending;
  }
  fail(token: number): boolean {
    if (!this.isLatest(token)) return false;
    this.#pending = undefined;
    return true;
  }
  commit(token: number, frame: DisplayedProjection): boolean {
    if (!this.isLatest(token)) return false;
    if (!this.#current)
      throw new Error("Projection history is not initialised");
    if (frame.pose.id === this.#current.pose.id) {
      this.#pending = undefined;
      return false;
    }
    // Validate and calculate before changing any committed history.
    const delta = projectionDelta(frame.values, this.#current.values);
    this.#previous = this.#current;
    this.#current = frame;
    this.#delta = delta;
    this.#pending = undefined;
    return true;
  }
  dispose(): void {
    this.#pending = undefined;
    ++this.#generation;
  }
}
