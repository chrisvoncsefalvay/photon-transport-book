import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { READY_IDS } from "./import-figure-rollouts.mjs";
const root = new URL("../../", import.meta.url);
const read = async (name) =>
  JSON.parse(
    await readFile(
      new URL(`public/generated/figure-rollouts/${name}`, root),
      "utf8",
    ),
  );
const hash = (b) => createHash("sha256").update(b).digest("hex");
test("all 19 curated figures have complete, hash-verified data and manifests", async () => {
  const provenance = await read("provenance.json");
  assert.deepEqual(
    Object.keys(provenance.figures).sort(),
    [...READY_IDS].sort(),
  );
  const owned = new Set();
  for (const group of ["analytic", "cuda", "noise", "transport"]) {
    const m = await read(`${group}/artifact.manifest.json`);
    for (const [file, digest] of Object.entries(m.output_digests)) {
      assert(!owned.has(file));
      owned.add(file);
      assert.equal(hash(await readFile(new URL(file, root))), digest, file);
    }
  }
  for (const names of Object.values(provenance.figures))
    for (const name of names)
      assert(owned.has(`public/generated/figure-rollouts/${name}`), name);
  assert(!JSON.stringify(provenance).includes("/home/"));
});
test("spectral chapters retain distinct coefficients, mixtures and decreasing effective slope", async () => {
  for (const [chapter, coefficients] of [
    [7, [0.01, 0.03]],
    [8, [0.02, 0.04]],
  ]) {
    const d = await read(`spectral-chapter-${chapter}.json`);
    assert.deepEqual(d.declared_coefficients_mm_inverse, coefficients);
    assert.equal(d.thickness_mm.length, 401);
    for (let i = 0; i < 401; i++) {
      const components = d.transmitted_component_means_cuda.map(
        (row) => row[i],
      );
      assert(
        Math.abs(components[0] + components[1] - d.transmission_cuda[i]) < 1e-7,
      );
    }
    assert(
      d.effective_slope_per_mm_from_stored_values[0] >
        d.effective_slope_per_mm_from_stored_values.at(-1),
    );
  }
});
test("uncertainty curves use launched histories and the exact Bernoulli variance", async () => {
  const d = await read("absorption-uncertainty.json");
  assert.equal(d.rows.length, 12);
  for (const r of d.rows) {
    assert(Math.abs(r.transmission - Math.exp(-r.optical_depth)) < 1e-14);
    assert(
      Math.abs(
        r.relative_standard_error -
          Math.sqrt((1 - r.transmission) / (r.histories * r.transmission)),
      ) < 1e-12,
    );
    assert.equal(r.replicates.length, 16);
  }
});
test("aperture uncertainty preserves the exact coupled strip law", async () => {
  const d = await read("figure-10-4.json");
  for (const s of d.series)
    for (const r of s.rows) {
      const q = (2 * r.half_step_mm) / d.parameters.width_mm;
      assert(
        Math.abs(
          r.relative_standard_error - Math.sqrt((1 - q) / (s.histories * q)),
        ) < 1e-10,
      );
      assert(
        Math.abs(r.zero_contribution_probability - (1 - q) ** s.histories) <
          1e-10,
      );
    }
});
test("recorded plots, radiographs and conceptual figures retain separate provenance and anchors", async () => {
  const chapters = [
    "volumes-line-integrals",
    "differentiating-projection",
    "recovering-pose",
    "same-pose-different-x-ray",
    "spectral-transport-detector",
    "scattering-stochastic-transport",
    "differentiating-transport",
  ];
  const found = [];
  const conceptual = [];
  const radiographs = [];
  let remaining = 0;
  for (const chapter of chapters) {
    const text = await readFile(
      new URL(`src/pages/chapters/${chapter}.mdx`, root),
      "utf8",
    );
    found.push(
      ...[...text.matchAll(/<RecordedFigure\s+id="([^"]+)"/g)].map((m) => m[1]),
    );
    conceptual.push(
      ...[...text.matchAll(/<ConceptualFigures\s+id="([^"]+)"/g)].map(
        (m) => m[1],
      ),
    );
    radiographs.push(
      ...[...text.matchAll(/<SpectralRadiographs\s+id="([^"]+)"/g)].map(
        (m) => m[1],
      ),
    );
    remaining += [...text.matchAll(/<FigurePlaceholder\s/g)].length;
  }
  assert.deepEqual(found.sort(), [...READY_IDS].sort());
  assert.deepEqual(conceptual.sort(), [
    "figure-10-1",
    "figure-10-3",
    "figure-5-1",
    "figure-9-1",
    "figure-9-2",
    "figure-9-4",
  ]);
  assert.deepEqual(radiographs, ["figure-8-2"]);
  assert.equal(remaining, 0);
});
