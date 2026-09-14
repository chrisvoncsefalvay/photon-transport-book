export type RecordedSeries = {
  label: string;
  x: number[];
  y: (number | null)[];
  tone?: "ink" | "accent" | "technical" | "muted";
  dash?: boolean;
  mode?: "line" | "step" | "markers" | "bars";
  marker?: "circle" | "square";
  legend?: boolean;
};
export type Scale = "linear" | "log";
export function extent(
  values: (number | null)[],
  scale: Scale,
): [number, number] {
  const valid = values.filter(
    (v): v is number =>
      v !== null && Number.isFinite(v) && (scale !== "log" || v > 0),
  );
  if (!valid.length)
    throw new Error("A plot needs finite values in its scale domain");
  let lo = Math.min(...valid),
    hi = Math.max(...valid);
  if (lo === hi) {
    lo = scale === "log" ? lo / 10 : lo - 1;
    hi = scale === "log" ? hi * 10 : hi + 1;
  }
  return [lo, hi];
}
export function mapValue(
  value: number,
  domain: [number, number],
  range: [number, number],
  scale: Scale,
) {
  const f = scale === "log" ? Math.log10 : (x: number) => x;
  return (
    range[0] +
    ((f(value) - f(domain[0])) / (f(domain[1]) - f(domain[0]))) *
      (range[1] - range[0])
  );
}
export function plotPath(
  series: RecordedSeries,
  xd: [number, number],
  yd: [number, number],
  xs: Scale,
  ys: Scale,
) {
  if (series.x.length !== series.y.length)
    throw new Error(`Unpaired series: ${series.label}`);
  let path = "",
    connected = false;
  for (let i = 0; i < series.x.length; i++) {
    const x = series.x[i]!,
      y = series.y[i];
    if (
      y === null ||
      y === undefined ||
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      (xs === "log" && x <= 0) ||
      (ys === "log" && y <= 0)
    ) {
      connected = false;
      continue;
    }
    const px = mapValue(x, xd, [76, 400], xs).toFixed(3),
      py = mapValue(y, yd, [244, 18], ys).toFixed(3);
    path += !connected
      ? `M${px},${py}`
      : series.mode === "step"
        ? `H${px}V${py}`
        : `L${px},${py}`;
    connected = true;
  }
  return path;
}
export function ticks(domain: [number, number], scale: Scale): number[] {
  if (scale === "log") {
    const first = Math.ceil(Math.log10(domain[0])),
      last = Math.floor(Math.log10(domain[1]));
    const stride = Math.max(1, Math.ceil((last - first) / 4));
    const values = [];
    for (let p = first; p <= last; p += stride) values.push(10 ** p);
    return values.length > 1 ? values : [...domain];
  }
  const raw = (domain[1] - domain[0]) / 4,
    base = 10 ** Math.floor(Math.log10(raw));
  const factor = [1, 2, 2.5, 5, 10].find((n) => n * base >= raw)!;
  const step = factor * base;
  const values = [];
  for (
    let i = Math.ceil(domain[0] / step);
    i * step <= domain[1] + step * 1e-9;
    i++
  )
    values.push(Number((i * step).toPrecision(12)));
  return values;
}
const superscripts: Record<string, string> = {
  "-": "⁻",
  "0": "⁰",
  "1": "¹",
  "2": "²",
  "3": "³",
  "4": "⁴",
  "5": "⁵",
  "6": "⁶",
  "7": "⁷",
  "8": "⁸",
  "9": "⁹",
};
export function formatTick(n: number) {
  if (n === 0) return "0";
  if (Number.isInteger(n) && Math.abs(n) >= 10000 && Math.abs(n) < 1000000)
    return n.toLocaleString("en-GB");
  if (Math.abs(n) >= 10000 || Math.abs(n) < 0.001) {
    const exponent = Math.floor(Math.log10(Math.abs(n)));
    const mantissa = Number((n / 10 ** exponent).toPrecision(3));
    return `${mantissa === 1 ? "" : `${mantissa}×`}10${String(exponent)
      .split("")
      .map((c) => superscripts[c])
      .join("")}`;
  }
  return String(Number(n.toPrecision(5))).replace("-", "−");
}
