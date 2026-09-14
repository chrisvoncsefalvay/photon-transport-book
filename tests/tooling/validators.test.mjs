import assert from "node:assert/strict";
import test from "node:test";

import { citationKeysInSource } from "../../tools/public/validate-citations.mjs";
import { validateSchema } from "../../tools/public/lib/schema.mjs";

test("citation-key discovery ignores fenced examples and supports the MDX component", () => {
  const source = [
    '<Citation keys={["siddon1985"]} />',
    "A compact form [@second] and {cite:third}.",
    "```mdx",
    '<Citation keys={["not-live"]} />',
    "```",
  ].join("\n");
  assert.deepEqual(citationKeysInSource(source), [
    "second",
    "third",
    "siddon1985",
  ]);
});

test("the schema validator reports nested failures", () => {
  const errors = validateSchema(
    { records: [{ id: "INVALID ID", extra: true }] },
    {
      type: "object",
      required: ["records"],
      properties: {
        records: {
          type: "array",
          items: {
            type: "object",
            additionalProperties: false,
            required: ["id"],
            properties: { id: { type: "string", pattern: "^[a-z-]+$" } },
          },
        },
      },
    },
  );
  assert.ok(errors.some((error) => error.includes("$.records[0].id")));
  assert.ok(errors.some((error) => error.includes("$.records[0].extra")));
});
