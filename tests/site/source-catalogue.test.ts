import { describe, expect, it } from "vitest";

import {
  findSourceRegion,
  parseSourceCatalogue,
  type SourceCatalogue,
} from "../../src/lib/source-catalogue";

const validCatalogue: SourceCatalogue = {
  schema_version: 1,
  source_commit: "0123456789abcdef",
  repository_url: "https://github.com/example/public-book",
  regions: [
    {
      file: "python/dpt/bootstrap.py",
      region: "bootstrap-package-identity",
      language: "python",
      code: 'def package_identity() -> str:\n    return "dpt"',
      start_line: 4,
      end_line: 5,
      url: "https://github.com/example/public-book/blob/0123456789abcdef/python/dpt/bootstrap.py#L4-L5",
    },
  ],
};

describe("generated source catalogue", () => {
  it("parses and resolves an exact file and region pair", () => {
    const catalogue = parseSourceCatalogue(validCatalogue);
    expect(
      findSourceRegion(
        catalogue,
        "python/dpt/bootstrap.py",
        "bootstrap-package-identity",
      ),
    ).toMatchObject({ start_line: 4, end_line: 5, language: "python" });
  });

  it("rejects duplicate identities", () => {
    expect(() =>
      parseSourceCatalogue({
        ...validCatalogue,
        regions: [validCatalogue.regions[0], validCatalogue.regions[0]],
      }),
    ).toThrow(/Duplicate source region/);
  });

  it("fails when a requested source is absent", () => {
    expect(() =>
      findSourceRegion(validCatalogue, "python/dpt/missing.py", "missing"),
    ).toThrow(/is not present/);
  });

  it("rejects malformed line ranges", () => {
    expect(() =>
      parseSourceCatalogue({
        ...validCatalogue,
        regions: [{ ...validCatalogue.regions[0], start_line: 0 }],
      }),
    ).toThrow(/invalid line range/);
  });
});
