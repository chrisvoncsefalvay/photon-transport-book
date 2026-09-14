/** Public display derivatives of an independently admitted, frozen study. */
export interface ComparisonImage {
  file: string;
  sha256: string;
  width: number;
  height: number;
  physical_width_mm: number;
  physical_height_mm: number;
  window: [number, number];
  below_window_pixels: number;
  above_window_pixels: number;
  alt: string;
}

export interface ComparisonOutcome {
  id: string;
  case: number;
  replicate: number;
  regime: "dense" | "sparse" | "limited";
  method: string;
  status: "completed" | "failed" | "not_started";
  evaluation_status: string;
  hard_correctness_passed: boolean | null;
  termination: string;
  accepted_updates: number | null;
  solve_seconds: number | null;
  stationarity_passed: boolean | null;
  water_rmse: number | null;
  bone_rmse: number | null;
  attenuation_rmse_mm_inverse: number | null;
  withheld_deviance_per_sample: number | null;
}

export interface ComparisonRegime {
  id: "dense" | "sparse" | "limited";
  label: string;
  outcome_id: string;
  views: number;
  incident_photons_per_view: number;
  angle_pairs_yaw_tilt_degrees: [number, number][];
}

export interface ComparisonSlice {
  key: string;
  quantity: string;
  quantity_label: string;
  plane: "axial" | "coronal" | "sagittal";
  plane_label: string;
  units: string;
  reference: ComparisonImage;
  recovered: {
    outcome_id: string;
    image: ComparisonImage | null;
    signed_error: ComparisonImage | null;
  }[];
}

export interface ComparisonDisplay {
  method: string;
  label: string;
  model: "scalar" | "spectral";
  regimes: ComparisonRegime[];
  slices: ComparisonSlice[];
  radiograph: {
    label: string;
    view_index: number;
    yaw_degrees: number;
    tilt_degrees: number;
    channel_label: string;
    observation: ComparisonImage;
    recovered: {
      outcome_id: string;
      prediction: ComparisonImage | null;
      residual: ComparisonImage | null;
    }[];
  };
}

export interface ReconstructionComparisonStudy {
  schema_version: 1;
  scope: string;
  default_method: "spectral_metric";
  representative: { case: 2; replicate: 0 };
  displays: ComparisonDisplay[];
  outcomes: ComparisonOutcome[];
  source_sha256: Record<string, string>;
  rights: string;
  limitations: string[];
}

/** Fail the build on missing outcomes, mismatched scales or substituted examples. */
export function validateComparisonStudy(
  study: ReconstructionComparisonStudy,
): void {
  const ensure = (condition: boolean, message: string) => {
    if (!condition) throw new Error(`Reconstruction comparison: ${message}`);
  };
  const methods = [
    "scalar_poisson",
    "scalar_pwls",
    "spectral_metric",
    "spectral_euclidean",
  ];
  const regimes = ["dense", "sparse", "limited"];
  const planes = ["axial", "coronal", "sagittal"];
  const same = (left: unknown, right: unknown) =>
    JSON.stringify(left) === JSON.stringify(right);
  const digest = (value: string) => /^[a-f0-9]{64}$/.test(value);
  const finite = (value: number) => Number.isFinite(value);
  const nullableNumber = (value: number | null) =>
    value === null || (finite(value) && value >= 0);
  const expectedIDs = new Set<string>();
  for (const caseID of [2, 0]) {
    for (const regime of regimes) {
      for (const replicate of [0, 1]) {
        for (const method of methods) {
          if (method === "spectral_euclidean" && regime !== "dense") continue;
          expectedIDs.add(`case${caseID}_${regime}_r${replicate}_${method}`);
        }
      }
    }
  }
  ensure(study.schema_version === 1, "unsupported schema");
  ensure(study.default_method === "spectral_metric", "default method changed");
  ensure(
    study.representative.case === 2 && study.representative.replicate === 0,
    "prescribed representative changed",
  );
  ensure(study.outcomes.length === 40, "all forty outcomes are required");
  const outcomes = new Map(study.outcomes.map((row) => [row.id, row]));
  ensure(
    outcomes.size === 40 && [...expectedIDs].every((key) => outcomes.has(key)),
    "missing or duplicate outcome identities",
  );
  for (const row of study.outcomes) {
    ensure(
      row.id ===
        `case${row.case}_${row.regime}_r${row.replicate}_${row.method}`,
      "outcome identity disagrees with its metadata",
    );
    ensure(
      ["completed", "failed", "not_started"].includes(row.status),
      "unknown fitting status",
    );
    ensure(
      row.stationarity_passed === null ||
        typeof row.stationarity_passed === "boolean",
      "missing stationarity state",
    );
    ensure(
      row.hard_correctness_passed === null ||
        typeof row.hard_correctness_passed === "boolean",
      "missing numerical correctness state",
    );
    ensure(
      [
        row.solve_seconds,
        row.water_rmse,
        row.bone_rmse,
        row.attenuation_rmse_mm_inverse,
        row.withheld_deviance_per_sample,
      ].every(nullableNumber),
      "nonfinite or negative metric",
    );
    ensure(
      row.accepted_updates === null ||
        (Number.isInteger(row.accepted_updates) && row.accepted_updates >= 0),
      "invalid accepted-update count",
    );
    if (row.evaluation_status !== "complete") {
      ensure(
        [
          row.water_rmse,
          row.bone_rmse,
          row.attenuation_rmse_mm_inverse,
          row.withheld_deviance_per_sample,
        ].every((value) => value === null),
        "unavailable evaluation has numerical results",
      );
    } else {
      ensure(row.status === "completed", "evaluation requires a completed fit");
      ensure(
        typeof row.hard_correctness_passed === "boolean",
        "completed evaluation requires its numerical correctness state",
      );
    }
  }
  ensure(
    study.displays.length === methods.length &&
      new Set(study.displays.map((display) => display.method)).size ===
        methods.length,
    "four distinct method displays are required",
  );
  const validateImage = (item: ComparisonImage, window: [number, number]) => {
    ensure(
      /^[a-zA-Z0-9_./-]+$/.test(item.file) &&
        !item.file.startsWith("/") &&
        !item.file.split("/").includes(".."),
      "unsafe image path",
    );
    ensure(digest(item.sha256), "missing image digest");
    ensure(
      [item.width, item.height].every((n) => Number.isInteger(n) && n > 0),
      "invalid native raster size",
    );
    ensure(
      [item.physical_width_mm, item.physical_height_mm].every(
        (n) => finite(n) && n > 0,
      ),
      "invalid physical image extent",
    );
    ensure(same(item.window, window), "display window changed");
    ensure(
      [item.below_window_pixels, item.above_window_pixels].every(
        (n) => Number.isInteger(n) && n >= 0,
      ) &&
        item.below_window_pixels + item.above_window_pixels <=
          item.width * item.height,
      "invalid clipping counts",
    );
    ensure(
      typeof item.alt === "string" && item.alt.length > 0,
      "missing image description",
    );
  };
  const sameFrame = (left: ComparisonImage, right: ComparisonImage) => {
    ensure(
      same(
        [
          left.width,
          left.height,
          left.physical_width_mm,
          left.physical_height_mm,
        ],
        [
          right.width,
          right.height,
          right.physical_width_mm,
          right.physical_height_mm,
        ],
      ),
      "image samples or physical scales differ",
    );
  };
  for (const display of study.displays) {
    ensure(methods.includes(display.method), "unknown displayed method");
    ensure(
      display.model ===
        (display.method.startsWith("spectral_") ? "spectral" : "scalar"),
      "observation model differs from its method",
    );
    const requiredRegimes =
      display.method === "spectral_euclidean" ? ["dense"] : regimes;
    ensure(
      same(
        display.regimes.map((regime) => regime.id),
        requiredRegimes,
      ),
      "displayed acquisition regimes changed",
    );
    const displayIDs = display.regimes.map((regime) => regime.outcome_id);
    for (const regime of display.regimes) {
      ensure(
        regime.outcome_id === `case2_${regime.id}_r0_${display.method}`,
        "displayed outcome is not the prescribed representative",
      );
      ensure(
        regime.views === (regime.id === "dense" ? 144 : 36),
        "view count changed",
      );
      ensure(
        regime.incident_photons_per_view ===
          (regime.id === "dense" ? 200000 : 800000),
        "incident photon population changed",
      );
      ensure(
        regime.angle_pairs_yaw_tilt_degrees.length === regime.views &&
          regime.angle_pairs_yaw_tilt_degrees.every(
            (pair) => pair.length === 2 && pair.every(finite),
          ),
        "angle record is incomplete",
      );
    }
    const quantities =
      display.model === "spectral"
        ? ["bone", "water", "attenuation"]
        : ["attenuation"];
    ensure(
      display.slices.length === quantities.length * planes.length,
      "recorded centre-plane coverage differs",
    );
    const sliceIDs = new Set<string>();
    for (const slice of display.slices) {
      ensure(
        quantities.includes(slice.quantity) && planes.includes(slice.plane),
        "unknown recorded field or plane",
      );
      const key = `${slice.quantity}-${slice.plane}`;
      ensure(!sliceIDs.has(key), "duplicate centre plane");
      sliceIDs.add(key);
      const window: [number, number] =
        slice.quantity === "attenuation" ? [0, 0.05] : [0, 1];
      const errorWindow: [number, number] =
        slice.quantity === "attenuation" ? [-0.01, 0.01] : [-0.25, 0.25];
      validateImage(slice.reference, window);
      ensure(
        same(
          slice.recovered.map((item) => item.outcome_id),
          displayIDs,
        ),
        "slice outcome coverage differs",
      );
      for (const entry of slice.recovered) {
        const evaluated = outcomes.get(entry.outcome_id)!;
        const available =
          evaluated.evaluation_status === "complete" &&
          evaluated.hard_correctness_passed === true;
        ensure(
          (entry.image !== null) === available &&
            (entry.signed_error !== null) === available,
          "slice availability differs from the evaluation",
        );
        if (entry.image && entry.signed_error) {
          validateImage(entry.image, window);
          validateImage(entry.signed_error, errorWindow);
          sameFrame(slice.reference, entry.image);
          sameFrame(slice.reference, entry.signed_error);
        }
      }
    }
    ensure(
      display.radiograph.view_index === 0,
      "whole withheld display view changed",
    );
    validateImage(display.radiograph.observation, [0, 8]);
    ensure(
      same(
        display.radiograph.recovered.map((item) => item.outcome_id),
        displayIDs,
      ),
      "radiograph outcome coverage differs",
    );
    for (const entry of display.radiograph.recovered) {
      const evaluated = outcomes.get(entry.outcome_id)!;
      const available =
        evaluated.evaluation_status === "complete" &&
        evaluated.hard_correctness_passed === true;
      ensure(
        (entry.prediction !== null) === available &&
          (entry.residual !== null) === available,
        "radiograph availability differs from the evaluation",
      );
      if (entry.prediction && entry.residual) {
        validateImage(entry.prediction, [0, 8]);
        validateImage(entry.residual, [-5, 5]);
        sameFrame(display.radiograph.observation, entry.prediction);
        sameFrame(display.radiograph.observation, entry.residual);
      }
    }
  }
  ensure(
    Object.keys(study.source_sha256).length > 0 &&
      Object.values(study.source_sha256).every(digest),
    "source digests are missing",
  );
  ensure(
    typeof study.rights === "string" && study.rights.length > 0,
    "source rights are missing",
  );
  ensure(study.limitations.length > 0, "interpretation is missing");
}

export function comparisonNumber(value: number | null, digits = 4): string {
  return value === null ? "Unavailable" : value.toFixed(digits);
}

export function comparisonMethod(method: string): string {
  return (
    {
      scalar_poisson: "Scalar Poisson",
      scalar_pwls: "Scalar log-WLS",
      spectral_metric: "Spectral fixed metric",
      spectral_euclidean: "Spectral Euclidean",
    }[method] ?? method
  );
}

export function comparisonStop(reason: string): string {
  return (
    {
      wall_time_budget: "Time limit",
      iteration_budget: "Update limit",
      max_iterations: "Update limit",
      line_search_failed: "Line search failed",
      projected_gradient: "Stationarity criterion",
      mapping_tolerance: "Stationarity criterion",
      not_started: "Not started",
    }[reason] ?? reason.replaceAll("_", " ")
  );
}
