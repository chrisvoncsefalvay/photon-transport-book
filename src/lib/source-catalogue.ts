import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

export const SOURCE_CATALOGUE_SCHEMA_VERSION = 1;
export const DEFAULT_SOURCE_CATALOGUE = "public/generated/source-regions.json";

const SOURCE_LANGUAGES = [
  "c",
  "cpp",
  "cuda",
  "javascript",
  "python",
  "typescript",
  "tsx",
  "text",
] as const;

export type SourceLanguage = (typeof SOURCE_LANGUAGES)[number];

export interface SourceRegion {
  file: string;
  region: string;
  language: SourceLanguage;
  code: string;
  start_line: number;
  end_line: number;
  url: string;
}

export interface SourceCatalogue {
  schema_version: typeof SOURCE_CATALOGUE_SCHEMA_VERSION;
  source_commit: string;
  repository_url: string;
  regions: SourceRegion[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseRegion(value: unknown, index: number): SourceRegion {
  if (!isRecord(value)) {
    throw new Error(`Source catalogue region ${index} must be an object.`);
  }

  const stringFields = ["file", "region", "code", "url"] as const;
  for (const field of stringFields) {
    if (typeof value[field] !== "string" || value[field].length === 0) {
      throw new Error(
        `Source catalogue region ${index} has an invalid ${field}.`,
      );
    }
  }

  if (!SOURCE_LANGUAGES.includes(value.language as SourceLanguage)) {
    throw new Error(
      `Source catalogue region ${index} has an invalid language.`,
    );
  }

  if (
    !Number.isInteger(value.start_line) ||
    !Number.isInteger(value.end_line)
  ) {
    throw new Error(
      `Source catalogue region ${index} must contain integer line numbers.`,
    );
  }

  const startLine = value.start_line as number;
  const endLine = value.end_line as number;
  if (startLine < 1 || endLine < startLine) {
    throw new Error(
      `Source catalogue region ${index} has an invalid line range.`,
    );
  }

  return {
    file: value.file as string,
    region: value.region as string,
    language: value.language as SourceLanguage,
    code: value.code as string,
    start_line: startLine,
    end_line: endLine,
    url: value.url as string,
  };
}

export function parseSourceCatalogue(value: unknown): SourceCatalogue {
  if (!isRecord(value)) throw new Error("Source catalogue must be an object.");
  if (value.schema_version !== SOURCE_CATALOGUE_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported source catalogue schema ${String(value.schema_version)}; expected ${SOURCE_CATALOGUE_SCHEMA_VERSION}.`,
    );
  }
  if (
    typeof value.source_commit !== "string" ||
    value.source_commit.length === 0
  ) {
    throw new Error("Source catalogue must record a source_commit.");
  }
  if (
    typeof value.repository_url !== "string" ||
    value.repository_url.length === 0
  ) {
    throw new Error("Source catalogue must record a repository_url.");
  }
  if (!Array.isArray(value.regions)) {
    throw new Error("Source catalogue regions must be an array.");
  }

  const regions = value.regions.map(parseRegion);
  const identities = new Set<string>();
  for (const entry of regions) {
    const identity = `${entry.file}\0${entry.region}`;
    if (identities.has(identity)) {
      throw new Error(`Duplicate source region ${entry.file}#${entry.region}.`);
    }
    identities.add(identity);
  }

  return {
    schema_version: SOURCE_CATALOGUE_SCHEMA_VERSION,
    source_commit: value.source_commit,
    repository_url: value.repository_url,
    regions,
  };
}

export async function loadSourceCatalogue(
  path = process.env.DPT_SOURCE_CATALOGUE ?? DEFAULT_SOURCE_CATALOGUE,
): Promise<SourceCatalogue> {
  const absolutePath = resolve(process.cwd(), path);
  let serialised: string;

  try {
    serialised = await readFile(absolutePath, "utf8");
  } catch (error) {
    throw new Error(
      `Source catalogue is missing at ${path}. Run the public source extractor before Astro checks or builds.`,
      { cause: error },
    );
  }

  try {
    return parseSourceCatalogue(JSON.parse(serialised) as unknown);
  } catch (error) {
    throw new Error(`Source catalogue at ${path} is invalid.`, {
      cause: error,
    });
  }
}

export function findSourceRegion(
  catalogue: SourceCatalogue,
  file: string,
  region: string,
): SourceRegion {
  const matches = catalogue.regions.filter(
    (entry) => entry.file === file && entry.region === region,
  );

  if (matches.length !== 1) {
    throw new Error(
      matches.length === 0
        ? `Source region ${file}#${region} is not present in the generated catalogue.`
        : `Source region ${file}#${region} is ambiguous in the generated catalogue.`,
    );
  }

  return matches[0]!;
}

export async function getSourceRegion(
  file: string,
  region: string,
): Promise<SourceRegion> {
  return findSourceRegion(await loadSourceCatalogue(), file, region);
}
