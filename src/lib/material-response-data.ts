/** Recorded finite-perturbation display contract. No fitting or projection. */
export type Material = "water" | "bone";
export type Matrix = [[number, number], [number, number]];
export interface ResponseFile {
  file: string;
  sha256: string;
  bytes: number;
}
export interface ResponseClipping {
  window: [number, number];
  samples: number;
  below: number;
  above: number;
  negative_infinity: number;
  positive_infinity: number;
  values_modified: false;
}
export interface ResponsePanel extends ResponseFile {
  injected: Material;
  recovered: Material;
  width: number;
  height: number;
  clipping: ResponseClipping;
  sampling: {
    normal_indices: [number, number];
    upper_weight: number;
    physical_coordinate_mm: 0;
    interpolation: string;
  };
  samples: ResponseFile;
}
export interface ResponsePlane {
  plane: "coronal" | "axial" | "sagittal";
  normal: "x" | "y" | "z";
  horizontal: "x" | "y";
  vertical: "y" | "z";
  extent_mm: [number, number, number, number];
  panels: ResponsePanel[];
}
export interface ResponseArray extends ResponseFile {
  role: "signed" | "absolute";
  shape: number[];
  dtype: "<f8";
  clipping_jm: ResponseClipping[][];
}
export interface ResponseOutcome {
  id: string;
  replicate: number;
  scenario: "control" | Material;
  fit_status: "completed" | "failed" | "not_started";
  evaluation_status: "complete" | "not_started" | "unavailable_fit" | "failed";
  hard_correctness_passed: boolean | null;
  accepted_steps: number | null;
  termination: string | null;
  stationarity_passed: boolean | null;
  final_mapping: number | null;
  absolute_threshold: number | null;
  solve_seconds: number | null;
  fields: (ResponseFile & { role: string; shape: number[]; dtype: string })[];
  metrics: ResponseFile | null;
}
export interface ResponseReplicate {
  replicate: number;
  status: "complete" | "failed" | "not_started" | "unavailable";
  values_available: boolean;
  matrix: Matrix | null;
  maps: ResponsePlane[];
  arrays: ResponseArray[];
  outside: ResponseFile | null;
  outside_metrics: {
    outside_whole_cells: {
      definition: string;
      cell_count: number;
      volume_mm3: number;
      signed_integral_jm_mm3: Matrix;
      absolute_integral_jm_mm3: Matrix;
      empty_region: boolean;
    };
    outside_fraction_weighted: {
      definition: string;
      volume_mm3: number;
      signed_integral_jm_mm3: Matrix;
      absolute_integral_jm_mm3: Matrix;
    };
    pattern_inner_product_mm3: number;
    cell_volume_mm3: number;
  } | null;
}
export interface MaterialResponseData {
  schema_version: 1;
  kind: "recorded-material-response";
  review_status: "accepted" | "retained_failed";
  evaluation_status: "complete" | "partial_budget" | "failed";
  source: Record<string, string>;
  grid: {
    shape: [32, 32, 32];
    spacing_mm: [number, number, number];
    origin_mm: [number, number, number];
    orientation: number[];
  };
  perturbation_amplitude: 0.025;
  pattern_centre_mm_xyz: [25, 0, 0];
  pattern_radius_mm: 15;
  dilution: 0.9;
  signed_window: [-0.05, 0.05];
  absolute_window: [0, 0.05];
  default_plane: "coronal";
  outcomes: ResponseOutcome[];
  replicates: ResponseReplicate[];
  summary: {
    available: boolean;
    mean: Matrix | null;
    minimum: Matrix | null;
    maximum: Matrix | null;
  };
  limitations: string[];
  source_url: string;
  licence: string;
}

const materials = ["water", "bone"] as const;
const scenarios = ["control", ...materials] as const;
const planes = [
  ["coronal", "y", "x", "z", 148.875, 192],
  ["axial", "z", "x", "y", 148.875, 148.875],
  ["sagittal", "x", "y", "z", 148.875, 192],
] as const;

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Expected recorded response metadata.");
  return value as Record<string, unknown>;
}
function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
function equal(actual: unknown, expected: unknown): void {
  if (JSON.stringify(actual) !== JSON.stringify(expected))
    throw new Error("Recorded response contract differs.");
}
function choice(value: unknown, allowed: readonly unknown[]): void {
  if (!allowed.includes(value)) throw new Error("Invalid recorded status.");
}
function list(value: unknown, length?: number): unknown[] {
  if (
    !Array.isArray(value) ||
    (length !== undefined && value.length !== length)
  )
    throw new Error("Recorded response coverage differs.");
  return value;
}
function matrix(value: unknown): void {
  for (const row of list(value, 2))
    if (!list(row, 2).every(finite))
      throw new Error("Expected a finite 2×2 response matrix.");
}
function text(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    !/[\u0000-\u001f]/.test(value) &&
    !/\/(?:home|tmp|fshare|Users|root)\/|file:\/\//.test(value)
  );
}
function file(value: unknown): void {
  const record = object(value);
  if (
    typeof record.file !== "string" ||
    !/^[a-z0-9_./-]+$/.test(record.file) ||
    record.file.split("/").some((part) => ["", ".", ".."].includes(part)) ||
    typeof record.sha256 !== "string" ||
    !/^[0-9a-f]{64}$/.test(record.sha256) ||
    !Number.isSafeInteger(record.bytes) ||
    (record.bytes as number) <= 0 ||
    (record.bytes as number) > 25 * 1024 * 1024
  )
    throw new Error("Unsafe recorded response asset.");
}
function clipping(value: unknown, window: number[], samples: number): void {
  const record = object(value);
  equal(record.window, window);
  equal(record.samples, samples);
  equal(record.values_modified, false);
  equal(record.negative_infinity, 0);
  equal(record.positive_infinity, 0);
  for (const key of ["below", "above"])
    if (
      !Number.isSafeInteger(record[key]) ||
      (record[key] as number) < 0 ||
      (record[key] as number) > samples
    )
      throw new Error("Invalid display clipping count.");
  if ((record.below as number) + (record.above as number) > samples)
    throw new Error("Display clipping counts overlap.");
}

export function parseMaterialResponse(value: unknown): MaterialResponseData {
  const data = object(value);
  equal(data.schema_version, 1);
  equal(data.kind, "recorded-material-response");
  choice(data.review_status, ["accepted", "retained_failed"]);
  choice(data.evaluation_status, ["complete", "partial_budget", "failed"]);
  if (
    data.evaluation_status === "failed" &&
    data.review_status !== "retained_failed"
  )
    throw new Error(
      "Failed evaluations need an explicit retained-failure record.",
    );
  equal(data.perturbation_amplitude, 0.025);
  equal(data.pattern_centre_mm_xyz, [25, 0, 0]);
  equal(data.pattern_radius_mm, 15);
  equal(data.dilution, 0.9);
  equal(data.signed_window, [-0.05, 0.05]);
  equal(data.absolute_window, [0, 0.05]);
  equal(data.default_plane, "coronal");
  const grid = object(data.grid);
  equal(grid.shape, [32, 32, 32]);
  equal(grid.spacing_mm, [4.65234375, 4.65234375, 6]);
  equal(grid.origin_mm, [-72.111328125, -72.111328125, -93]);
  equal(grid.orientation, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const source = object(data.source);
  for (const key of [
    "review_sha256",
    "source_freeze_sha256",
    "acquisition_run_sha256",
    "fits_run_sha256",
    "evaluation_run_sha256",
    "six_fit_freeze_sha256",
  ])
    if (typeof source[key] !== "string" || !/^[0-9a-f]{64}$/.test(source[key]))
      throw new Error("Missing response source identity.");
  if (
    typeof source.source_commit !== "string" ||
    !/^[0-9a-f]{40}$/.test(source.source_commit)
  )
    throw new Error("Missing immutable source revision.");
  const outcomes = list(data.outcomes, 6).map((value, index) => {
    const row = object(value),
      replicate = Math.floor(index / 3),
      scenario = scenarios[index % 3];
    equal(row.id, `case2_probe32_rep${replicate}_${scenario}_spectral_metric`);
    equal(row.replicate, replicate);
    equal(row.scenario, scenario);
    choice(row.fit_status, ["completed", "failed", "not_started"]);
    choice(row.evaluation_status, [
      "complete",
      "not_started",
      "unavailable_fit",
      "failed",
    ]);
    choice(row.hard_correctness_passed, [true, false, null]);
    choice(row.stationarity_passed, [true, false, null]);
    for (const key of [
      "accepted_steps",
      "final_mapping",
      "absolute_threshold",
      "solve_seconds",
    ])
      if (row[key] !== null && (!finite(row[key]) || row[key] < 0))
        throw new Error("Invalid recorded solver value.");
    if (
      row.accepted_steps !== null &&
      (!Number.isSafeInteger(row.accepted_steps) ||
        (row.accepted_steps as number) > 100)
    )
      throw new Error("Recorded update count exceeds its declared limit.");
    if (
      row.termination !== null &&
      (!text(row.termination) || !/^[a-z][a-z0-9_ -]*$/.test(row.termination))
    )
      throw new Error("Invalid recorded stop label.");
    const fields = list(row.fields);
    if (
      fields.length &&
      (data.review_status !== "accepted" ||
        row.evaluation_status !== "complete" ||
        row.hard_correctness_passed !== true)
    )
      throw new Error("Unadmitted fields cannot be displayed.");
    const fieldRoles = [
      "fields",
      "reference",
      "signed-error",
      "absolute-error",
    ];
    if (
      data.review_status === "accepted" &&
      row.evaluation_status === "complete"
    ) {
      equal(row.fit_status, "completed");
      equal(row.hard_correctness_passed, true);
      equal(fields.length, 4);
      file(row.metrics);
      equal(object(row.metrics).file, `records/${row.id}.json`);
    } else {
      equal(fields, []);
      equal(row.metrics, null);
    }
    fields.forEach((field, index) => {
      file(field);
      const item = object(field);
      equal(item.role, fieldRoles[index]);
      equal(item.shape, [2, 32, 32, 32]);
      equal(item.dtype, index < 2 ? "<f4" : "<f8");
      equal(item.file, `fields/${row.id}/${fieldRoles[index]}.npy`);
    });
    if (row.fit_status === "not_started") {
      for (const key of [
        "accepted_steps",
        "termination",
        "stationarity_passed",
        "final_mapping",
        "absolute_threshold",
        "solve_seconds",
      ])
        equal(row[key], null);
      equal(row.evaluation_status, "unavailable_fit");
    }
    if (row.stationarity_passed !== null) {
      if (!finite(row.final_mapping) || !finite(row.absolute_threshold))
        throw new Error(
          "A stationarity result needs both recorded mapping and threshold.",
        );
      equal(
        row.stationarity_passed,
        row.final_mapping <= row.absolute_threshold,
      );
    }
    return row;
  });
  const replicas = list(data.replicates, 2).map((value, index) => {
    const item = object(value);
    equal(item.replicate, index);
    choice(item.status, ["complete", "failed", "not_started", "unavailable"]);
    choice(item.values_available, [true, false]);
    const maps = list(item.maps),
      arrays = list(item.arrays);
    if (item.values_available) {
      if (
        data.review_status !== "accepted" ||
        item.status !== "complete" ||
        outcomes
          .slice(index * 3, index * 3 + 3)
          .some(
            (row) =>
              row.fit_status !== "completed" ||
              row.evaluation_status !== "complete" ||
              row.hard_correctness_passed !== true,
          )
      )
        throw new Error("A gain matrix needs all three accepted evaluations.");
      matrix(item.matrix);
      file(item.outside);
      equal(
        object(item.outside).file,
        `records/rep${index}-outside-response.json`,
      );
      const outside = object(item.outside_metrics);
      for (const key of ["pattern_inner_product_mm3", "cell_volume_mm3"])
        if (!finite(outside[key]) || outside[key] <= 0)
          throw new Error("Invalid recorded response normalisation volume.");
      for (const key of ["outside_whole_cells", "outside_fraction_weighted"]) {
        const region = object(outside[key]);
        if (
          !text(region.definition) ||
          !finite(region.volume_mm3) ||
          region.volume_mm3 < 0
        )
          throw new Error("Invalid recorded outside-pattern region.");
        matrix(region.signed_integral_jm_mm3);
        matrix(region.absolute_integral_jm_mm3);
        if (
          (region.absolute_integral_jm_mm3 as Matrix)
            .flat()
            .some((value) => value < 0)
        )
          throw new Error(
            "Absolute outside-pattern integrals cannot be negative.",
          );
        if (key === "outside_whole_cells") {
          if (
            !Number.isSafeInteger(region.cell_count) ||
            (region.cell_count as number) < 0 ||
            (region.cell_count as number) > 32768
          )
            throw new Error("Invalid whole-cell outside-pattern count.");
          equal(region.empty_region, region.cell_count === 0);
          equal(
            region.volume_mm3,
            (region.cell_count as number) * (outside.cell_volume_mm3 as number),
          );
        }
      }
      equal(maps.length, 3);
      equal(arrays.length, 2);
      arrays.forEach((value, a) => {
        file(value);
        const array = object(value);
        equal(array.role, a === 0 ? "signed" : "absolute");
        equal(
          array.file,
          `responses/rep${index}/${array.role}_component_difference_maps_jm.npy`,
        );
        equal(array.shape, [2, 2, 32, 32, 32]);
        equal(array.dtype, "<f8");
        for (const row of list(array.clipping_jm, 2))
          for (const cell of list(row, 2))
            clipping(cell, a === 0 ? [-0.05, 0.05] : [0, 0.05], 32768);
      });
      maps.forEach((value, p) => {
        const plane = object(value),
          expected = planes[p]!;
        ["plane", "normal", "horizontal", "vertical"].forEach((key, axis) =>
          equal(plane[key], expected[axis]),
        );
        equal(plane.extent_mm, [
          -expected[4] / 2,
          expected[4] / 2,
          -expected[5] / 2,
          expected[5] / 2,
        ]);
        list(plane.panels, 4).forEach((value, panelIndex) => {
          const panel = object(value);
          file(panel);
          file(panel.samples);
          equal(panel.injected, materials[Math.floor(panelIndex / 2)]);
          equal(panel.recovered, materials[panelIndex % 2]);
          const stem = `maps/rep${index}/${plane.plane}-${panel.injected}-${panel.recovered}`;
          equal(panel.file, `${stem}.png`);
          equal(object(panel.samples).file, `${stem}.npy`);
          equal(panel.width, 32);
          equal(panel.height, 32);
          clipping(panel.clipping, [-0.05, 0.05], 1024);
          const sampling = object(panel.sampling);
          equal(sampling.normal_indices, [15, 16]);
          equal(sampling.upper_weight, 0.5);
          equal(sampling.physical_coordinate_mm, 0);
          if (!text(sampling.interpolation))
            throw new Error("Missing plane interpolation record.");
        });
      });
    } else {
      equal(item.matrix, null);
      equal(maps, []);
      equal(arrays, []);
      equal(item.outside, null);
      equal(item.outside_metrics, null);
    }
    return item;
  });
  const summary = object(data.summary);
  equal(
    summary.available,
    replicas.every((item) => item.values_available === true),
  );
  if (summary.available) {
    ["mean", "minimum", "maximum"].forEach((key) => matrix(summary[key]));
    const first = replicas[0]!.matrix as Matrix,
      second = replicas[1]!.matrix as Matrix;
    for (let m = 0; m < 2; m++)
      for (let j = 0; j < 2; j++) {
        const a = first[m]![j]!,
          b = second[m]![j]!;
        equal((summary.mean as Matrix)[m]![j], (a + b) / 2);
        equal((summary.minimum as Matrix)[m]![j], Math.min(a, b));
        equal((summary.maximum as Matrix)[m]![j], Math.max(a, b));
      }
  } else
    ["mean", "minimum", "maximum"].forEach((key) => equal(summary[key], null));
  if (
    !list(data.limitations).every(text) ||
    (data.limitations as string[]).length < 6 ||
    !text(data.licence)
  )
    throw new Error("Response qualifications are missing.");
  equal(data.source_url, "https://doi.org/10.7937/tcia.2019.tt7f4v7o");
  return data as unknown as MaterialResponseData;
}

export function responseNumber(value: number | null): string {
  return value === null
    ? "—"
    : Number(value.toPrecision(4)).toString().replace("-", "−");
}
export function responseStatus(value: string | null): string {
  if (value === null) return "Not recorded";
  return (
    (
      {
        completed: "Completed",
        complete: "Complete",
        failed: "Failed",
        not_started: "Not started",
        unavailable: "Unavailable",
        unavailable_fit: "Fit unavailable",
        partial_budget: "Partial: evaluation budget",
        retained_failed: "Failed record retained",
      } as Record<string, string>
    )[value] ?? value.replaceAll("_", " ")
  );
}
