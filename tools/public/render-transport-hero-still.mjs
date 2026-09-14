// Optional display authoring. Build and serve the local static site first.
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const root = fileURLToPath(new URL("../../", import.meta.url));
const { values } = parseArgs({
  options: {
    url: { type: "string" },
    playwright: { type: "string" },
    chromium: { type: "string" },
  },
});
if (!values.url)
  throw new Error("Pass --url for the freshly built local preview.");
const playwright = values.playwright
  ? await import(pathToFileURL(resolve(values.playwright)).href)
  : await import("playwright");
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const sourcePaths = [
  "tools/public/render-transport-hero-still.mjs",
  "src/lib/transport-hero-field.ts",
  "src/lib/transport-hero-detector.ts",
  "src/lib/carm/conditional-lines.ts",
  "pnpm-lock.yaml",
  "public/generated/transport-hero/pelvis-lod.json",
  "public/generated/transport-hero/pelvis-lod.bin",
  "public/generated/transport-hero/impact-record.json",
  "public/generated/transport-hero/impact-atlas.png",
];
const sources = Object.fromEntries(
  await Promise.all(
    sourcePaths.map(async (path) => [
      path,
      sha256(await readFile(resolve(root, path))),
    ]),
  ),
);
const browser = await playwright.chromium.launch({
  ...(values.chromium ? { executablePath: resolve(values.chromium) } : {}),
  headless: true,
  args: ["--no-sandbox", "--enable-unsafe-swiftshader"],
});
try {
  const page = await browser.newPage({
    viewport: { width: 884, height: 1100 },
    deviceScaleFactor: 2,
    reducedMotion: "reduce",
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.goto(values.url);
  await page.locator("dpt-transport-hero").scrollIntoViewIfNeeded();
  const canvas = page.locator(".transport-hero__field");
  await canvas.waitFor({ state: "visible" });
  await page.addStyleTag({
    content: `html,body{background:transparent!important}
      body{visibility:hidden!important}
      dpt-transport-hero{visibility:visible!important;position:fixed!important;left:0!important;top:0!important;margin:0!important;width:520px!important;max-width:520px!important}
      .transport-hero__equation,.transport-hero__equation *,.transport-hero__control,astro-dev-toolbar{visibility:hidden!important}`,
  });
  await page.waitForTimeout(500);
  const size = await canvas.boundingBox();
  if (size.x !== 0 || size.y !== 0 || size.width !== 520 || size.height !== 470)
    throw new Error("Unexpected canvas bounds.");
  const first = await canvas.screenshot({ omitBackground: true });
  const second = await canvas.screenshot({ omitBackground: true });
  if (!first.equals(second))
    throw new Error("Still pixels changed without an advance.");
  if (errors.length) throw new Error(errors.join("\n"));
  for (const [path, expected] of Object.entries(sources))
    if (sha256(await readFile(resolve(root, path))) !== expected)
      throw new Error(`Source changed during capture: ${path}`);
  const destination = resolve(root, "public/generated/transport-hero");
  await writeFile(resolve(destination, "fluoro-still.png"), first);
  await writeFile(
    resolve(destination, "fluoro-still.json"),
    JSON.stringify(
      {
        schemaVersion: 1,
        method:
          "Static Three.js at reference camera with stationary mean of recorded photon-count frames; transparent capture omits equation and controls.",
        logicalSize: [520, 470],
        rasterSize: [1040, 940],
        browser: browser.version(),
        outputSha256: sha256(first),
        sources,
      },
      null,
      2,
    ) + "\n",
  );
  console.log(
    JSON.stringify({
      bytes: first.length,
      sha256: sha256(first),
      repeatIdentical: true,
    }),
  );
} finally {
  await browser.close();
}
