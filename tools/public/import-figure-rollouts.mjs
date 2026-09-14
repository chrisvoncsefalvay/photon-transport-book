import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import {
  readRegularFile,
  canonicalPath,
  parseUniqueJson,
} from "./lib/files.mjs";

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const bytes = (value) => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
const target = "public/generated/figure-rollouts";
const generator = "tools/public/import-figure-rollouts.mjs";
export const READY_IDS = [
  "4-1",
  "4-2",
  "4-3",
  "4-4",
  "5-2",
  "5-3",
  "5-4",
  "6-1",
  "6-2",
  "6-3",
  "7-1",
  "7-2",
  "7-3",
  "7-4",
  "8-1",
  "8-3",
  "9-3",
  "10-2",
  "10-4",
].map((id) => `figure-${id}`);
const analyticIds = [
  "4-1",
  "4-2",
  "4-3",
  "4-4",
  "5-2",
  "5-3",
  "5-4",
  "6-1",
  "7-4",
  "8-3",
  "10-2",
  "10-4",
];
const inputs = [
  ...analyticIds.map((id) => ({
    source: `analytic-payloads-final/figure-${id}.json`,
    name: `figure-${id}.json`,
    group: "analytic",
  })),
  ...[
    "geometry",
    "projection-sensitivity",
    "spectral-chapter-7",
    "spectral-chapter-8",
    "blur-displacement",
    "detector-noise",
  ].map((name) => ({
    source: `book-payloads-final/${name}.json`,
    name: `${name}.json`,
    group: name === "detector-noise" ? "noise" : "cuda",
  })),
  ...[0.1, 1, 3, 6].flatMap((depth) =>
    [4096, 16384, 65536].map((histories) => ({
      source: `absorption-depth-${depth}-n-${histories}/validation.json`,
      name: `absorption-${depth}-${histories}.json`,
      group: "transport",
      depth,
      histories,
    })),
  ),
];

export async function prepareFigureRollouts({ runDir, root = process.cwd() }) {
  if (!path.isAbsolute(runDir))
    throw new Error("runDir must be an absolute archive directory");
  const coverageBytes = await readRegularFile(runDir, "figure-coverage.json");
  const coverage = parseUniqueJson(
    coverageBytes.toString(),
    "figure-coverage.json",
  );
  const ready = coverage.figures.filter((f) => f.status === "ready");
  if (
    JSON.stringify(ready.map((f) => f.id).sort()) !==
    JSON.stringify([...READY_IDS].sort())
  )
    throw new Error(
      "The prepared figure set has changed; review the import allowlist",
    );
  const known = new Map(
    ready.flatMap((f) => f.artifacts.map((a) => [a.path, a.sha256])),
  );
  const runs = new Map();
  const outputs = new Map();
  const records = [];
  for (const input of inputs) {
    canonicalPath(input.source);
    const raw = await readRegularFile(runDir, input.source);
    if (known.get(input.source) !== sha(raw))
      throw new Error(`Coverage hash mismatch: ${input.source}`);
    const dir = path.posix.dirname(input.source);
    if (!runs.has(dir)) {
      const recordBytes = await readRegularFile(runDir, `${dir}/run.json`);
      const record = parseUniqueJson(recordBytes.toString(), `${dir}/run.json`);
      if (
        record.status !== "complete" ||
        !record.sources_unchanged ||
        !record.recorded_files_unchanged
      )
        throw new Error(`Incomplete or changed run: ${dir}`);
      for (const [file, digest] of Object.entries(record.source_sha256)) {
        canonicalPath(file);
        if (
          sha(await readRegularFile(runDir, `${dir}/sources/${file}`)) !==
          digest
        )
          throw new Error(`Archived source hash mismatch: ${dir}/${file}`);
      }
      for (const [file, digest] of Object.entries(record.output_sha256)) {
        canonicalPath(file);
        if (sha(await readRegularFile(runDir, `${dir}/${file}`)) !== digest)
          throw new Error(`Recorded output hash mismatch: ${dir}/${file}`);
      }
      runs.set(dir, { record, sha256: sha(recordBytes) });
    }
    const { record } = runs.get(dir);
    if (record.output_sha256[path.posix.basename(input.source)] !== sha(raw))
      throw new Error(`Run hash mismatch: ${input.source}`);
    const value = parseUniqueJson(raw.toString(), input.source);
    if (/\/home\/|\/tmp\/|[A-Z]:\\\\/.test(raw.toString()))
      throw new Error(`Private path in payload: ${input.source}`);
    outputs.set(input.name, raw);
    records.push({ ...input, sha256: sha(raw), value });
  }
  const absorption = records
    .filter((r) => r.group === "transport")
    .map((r) => ({
      optical_depth: r.depth,
      histories: r.histories,
      transmission: r.value.reference,
      relative_standard_error:
        Math.sqrt(r.value.independent_reference_variance_of_mean) /
        r.value.reference,
      replicates: r.value.replicates.map((v) => ({
        mean: v.mean,
        variance_of_mean: v.variance_of_mean,
      })),
      source: r.name,
    }));
  outputs.set(
    "absorption-uncertainty.json",
    bytes({
      derivation:
        "Relative standard error = sqrt(recorded independent variance of mean) / recorded exact transmission; replicates are retained separately.",
      independence_scope:
        "Each case has 16 disjoint history batches. Different optical-depth and history-count cases reuse seed 73129 and overlapping history identifiers; cases are coupled, not independent replications of one another.",
      rows: absorption,
    }),
  );
  const figureSources = Object.fromEntries(
    ready.map((f) => [
      f.id,
      f.artifacts
        .map((a) => inputs.find((i) => i.source === a.path)?.name)
        .filter(Boolean),
    ]),
  );
  const provenance = {
    schema_version: 1,
    claim_scope:
      "Archived mathematical fixtures and canonical executions from 9 September 2026; not measurements or claims about later source revisions.",
    coverage_sha256: sha(coverageBytes),
    figures: figureSources,
    runs: Object.fromEntries(
      [...runs].map(([dir, { record, sha256 }]) => [
        dir,
        {
          record_sha256: sha256,
          started_utc: record.started_utc,
          finished_utc: record.finished_utc,
          configuration: record.configuration,
          configuration_sha256: record.configuration_sha256,
          source_sha256: record.source_sha256,
          metadata: record.metadata,
          environment: record.environment,
        },
      ]),
    ),
  };
  const provenanceBytes = bytes(provenance);
  if (/\/home\/|\/tmp\//.test(provenanceBytes.toString()))
    throw new Error("Private path in provenance");
  outputs.set("provenance.json", provenanceBytes);
  const sourceCommit = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: root,
    encoding: "utf8",
  }).trim();
  const sourceDigests = {
    [generator]: sha(await readFile(path.join(root, generator))),
  };
  const manifests = new Map();
  for (const group of ["analytic", "cuda", "noise", "transport"]) {
    const names = inputs.filter((i) => i.group === group).map((i) => i.name);
    if (group === "analytic") names.push("provenance.json");
    if (group === "transport") names.push("absorption-uncertainty.json");
    const stochastic = ["noise", "transport"].includes(group);
    manifests.set(
      `${group}/artifact.manifest.json`,
      bytes({
        schema_version: 1,
        id: `book-figures-${group}`,
        chapter: "chapters-4-10",
        experiment: "archived-figure-rollouts",
        source_commit: sourceCommit,
        generator,
        mode: stochastic
          ? "stochastic-independent-realisations"
          : "deterministic-precomputed-sweep",
        stochastic,
        seeds: stochastic ? [group === "noise" ? 193717 : 73129] : null,
        created_at: "2026-09-11T10:47:42Z",
        parameters: {
          group,
          provenance: "/generated/figure-rollouts/provenance.json",
          ...(group === "transport"
            ? {
                independence_scope:
                  "Independent batches within each case; shared seed and overlapping history identifiers couple different cases.",
              }
            : {}),
        },
        notes:
          "source_commit identifies the curation checkout, not the archived producer. Producer source digests, configurations and actual run times are preserved in provenance.json. Input payloads are byte-identical to the verified archive.",
        outputs: names.map((n) => `${target}/${n}`),
        output_digests: Object.fromEntries(
          names.map((n) => [`${target}/${n}`, sha(outputs.get(n))]),
        ),
        source_digests: sourceDigests,
      }),
    );
  }
  return { outputs, manifests, provenance };
}

export async function importFigureRollouts(options) {
  const root = options.root ?? process.cwd();
  const prepared = await prepareFigureRollouts({ ...options, root });
  // Validate every source before touching the curated destination.
  for (const [name, content] of [...prepared.outputs, ...prepared.manifests]) {
    const dest = path.join(root, target, name);
    await mkdir(path.dirname(dest), { recursive: true });
    await writeFile(dest, content);
  }
  return {
    figures: READY_IDS.length,
    outputs: prepared.outputs.size,
    manifests: prepared.manifests.size,
  };
}
if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const index = process.argv.indexOf("--run-dir");
  if (index < 0 || !process.argv[index + 1])
    throw new Error(
      "Usage: node tools/public/import-figure-rollouts.mjs --run-dir /absolute/archive",
    );
  console.log(
    JSON.stringify(
      await importFigureRollouts({ runDir: process.argv[index + 1] }),
    ),
  );
}
