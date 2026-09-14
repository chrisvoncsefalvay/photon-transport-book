/**
 * Variants are literal JSON in MDX, never executable expressions. Every entry
 * points to canonical source; the manifest extractor resolves the book region.
 * @param {string | undefined} value
 * @param {string} context
 * @returns {{ label: string, file: string, region: string }[]}
 */
export function parseSourceVariants(value, context = "Source") {
  if (value === undefined) return [];
  let entries;
  try {
    entries = JSON.parse(value);
  } catch {
    throw new Error(`${context}: variants must be a literal JSON array.`);
  }
  if (!Array.isArray(entries))
    throw new Error(`${context}: variants must be an array.`);
  const labels = new Set();
  const identities = new Set();
  return entries.map((entry) => {
    if (
      !entry ||
      typeof entry !== "object" ||
      ["label", "file", "region"].some(
        (key) => typeof entry[key] !== "string" || !entry[key].trim(),
      )
    ) {
      throw new Error(
        `${context}: each source variant requires label, file and region strings.`,
      );
    }
    const label = entry.label.trim();
    const identity = `${entry.file}\0${entry.region}`;
    if (labels.has(label) || identities.has(identity))
      throw new Error(`${context}: duplicate source variant.`);
    labels.add(label);
    identities.add(identity);
    return { label, file: entry.file, region: entry.region };
  });
}
