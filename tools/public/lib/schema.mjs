const typeChecks = {
  array: Array.isArray,
  boolean: (value) => typeof value === "boolean",
  integer: (value) => Number.isInteger(value),
  null: (value) => value === null,
  number: (value) => typeof value === "number" && Number.isFinite(value),
  object: (value) =>
    value !== null && typeof value === "object" && !Array.isArray(value),
  string: (value) => typeof value === "string",
};

function matchesType(value, expectedType) {
  const alternatives = Array.isArray(expectedType)
    ? expectedType
    : [expectedType];
  return alternatives.some((type) => typeChecks[type]?.(value));
}

export function validateSchema(value, schema, location = "$") {
  const errors = [];

  if (Object.hasOwn(schema, "const") && !Object.is(value, schema.const)) {
    errors.push(`${location} must equal ${JSON.stringify(schema.const)}`);
  }

  if (schema.oneOf) {
    const alternatives = schema.oneOf.map((candidate) =>
      validateSchema(value, candidate, location),
    );
    if (
      alternatives.filter((candidate) => candidate.length === 0).length !== 1
    ) {
      errors.push(`${location} must match exactly one allowed shape`);
    }
    return errors;
  }

  if (schema.type && !matchesType(value, schema.type)) {
    errors.push(`${location} must be ${[].concat(schema.type).join(" or ")}`);
    return errors;
  }

  if (
    schema.enum &&
    !schema.enum.some((candidate) => Object.is(candidate, value))
  ) {
    errors.push(`${location} must be one of: ${schema.enum.join(", ")}`);
  }

  if (typeof value === "string") {
    if (schema.minLength !== undefined && value.length < schema.minLength) {
      errors.push(
        `${location} must contain at least ${schema.minLength} character(s)`,
      );
    }
    if (schema.pattern && !new RegExp(schema.pattern).test(value)) {
      errors.push(`${location} does not match ${schema.pattern}`);
    }
    if (schema.format === "date-time" && Number.isNaN(Date.parse(value))) {
      errors.push(`${location} must be an ISO date-time`);
    }
    if (schema.format === "uri") {
      try {
        new URL(value);
      } catch {
        errors.push(`${location} must be a URI`);
      }
    }
  }

  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) {
      errors.push(
        `${location} must contain at least ${schema.minItems} item(s)`,
      );
    }
    if (schema.uniqueItems) {
      const serialised = value.map((item) => JSON.stringify(item));
      if (new Set(serialised).size !== serialised.length) {
        errors.push(`${location} must not contain duplicate items`);
      }
    }
    if (schema.items) {
      value.forEach((item, index) => {
        errors.push(
          ...validateSchema(item, schema.items, `${location}[${index}]`),
        );
      });
    }
  }

  if (
    typeof value === "number" &&
    schema.minimum !== undefined &&
    value < schema.minimum
  ) {
    errors.push(`${location} must be at least ${schema.minimum}`);
  }

  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    for (const required of schema.required ?? []) {
      if (!Object.hasOwn(value, required)) {
        errors.push(`${location}.${required} is required`);
      }
    }
    for (const [key, child] of Object.entries(value)) {
      if (schema.properties?.[key]) {
        errors.push(
          ...validateSchema(
            child,
            schema.properties[key],
            `${location}.${key}`,
          ),
        );
      } else if (schema.additionalProperties === false) {
        errors.push(`${location}.${key} is not allowed`);
      } else if (typeof schema.additionalProperties === "object") {
        errors.push(
          ...validateSchema(
            child,
            schema.additionalProperties,
            `${location}.${key}`,
          ),
        );
      }
    }
  }

  return errors;
}

export function assertSchema(value, schema, label) {
  const errors = validateSchema(value, schema);
  if (errors.length > 0) {
    throw new Error(
      `${label} failed schema validation:\n- ${errors.join("\n- ")}`,
    );
  }
}
