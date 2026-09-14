import { describe, expect, it } from "vitest";
import {
  parseMaterialResponse,
  responseNumber,
  responseStatus,
} from "../../src/lib/material-response-data";

// Explicit metadata-only software fixtures. No image/result assets are created.
const materials = ["water", "bone"];
const file = (name: string) => ({
  file: name,
  sha256: "0".repeat(64),
  bytes: 1024,
});
const clipping = (window: number[], samples: number) => ({
  window,
  samples,
  below: 0,
  above: 0,
  negative_infinity: 0,
  positive_infinity: 0,
  values_modified: false,
});

function fixture(available = false) {
  const outcomes = [0, 1].flatMap((replicate) =>
    ["control", ...materials].map((scenario) => {
      const id = `case2_probe32_rep${replicate}_${scenario}_spectral_metric`;
      return {
        id,
        replicate,
        scenario,
        fit_status: available ? "completed" : "not_started",
        evaluation_status: available ? "complete" : "unavailable_fit",
        hard_correctness_passed: available ? true : null,
        accepted_steps: available ? 20 : null,
        termination: available ? "max_steps" : null,
        stationarity_passed: available ? false : null,
        final_mapping: available ? 2 : null,
        absolute_threshold: available ? 0.1 : null,
        solve_seconds: available ? 120 : null,
        metrics: available ? file(`records/${id}.json`) : null,
        fields: available
          ? ["fields", "reference", "signed-error", "absolute-error"].map(
              (role, index) => ({
                ...file(`fields/${id}/${role}.npy`),
                role,
                shape: [2, 32, 32, 32],
                dtype: index < 2 ? "<f4" : "<f8",
              }),
            )
          : [],
      };
    }),
  );
  const replicates = [0, 1].map((replicate) => ({
    replicate,
    status: available ? "complete" : "unavailable",
    values_available: available,
    matrix: available
      ? [
          [1 + replicate, -0.5],
          [0.25, 2 + replicate],
        ]
      : null,
    outside: available
      ? file(`records/rep${replicate}-outside-response.json`)
      : null,
    outside_metrics: available
      ? {
          outside_whole_cells: {
            definition: "Metadata-only whole-cell software fixture",
            cell_count: 10,
            volume_mm3: 10,
            signed_integral_jm_mm3: [
              [1, -2],
              [3, 4],
            ],
            absolute_integral_jm_mm3: [
              [1.5, 2.5],
              [3.5, 4.5],
            ],
            empty_region: false,
          },
          outside_fraction_weighted: {
            definition: "Metadata-only fraction-weighted software fixture",
            volume_mm3: 12,
            signed_integral_jm_mm3: [
              [1.1, -2.1],
              [3.1, 4.1],
            ],
            absolute_integral_jm_mm3: [
              [1.6, 2.6],
              [3.6, 4.6],
            ],
          },
          pattern_inner_product_mm3: 1,
          cell_volume_mm3: 1,
        }
      : null,
    arrays: available
      ? ["signed", "absolute"].map((role, index) => ({
          ...file(
            `responses/rep${replicate}/${role}_component_difference_maps_jm.npy`,
          ),
          role,
          shape: [2, 2, 32, 32, 32],
          dtype: "<f8",
          clipping_jm: materials.map(() =>
            materials.map(() =>
              clipping(index === 0 ? [-0.05, 0.05] : [0, 0.05], 32768),
            ),
          ),
        }))
      : [],
    maps: available
      ? [
          ["coronal", "y", "x", "z", 148.875, 192],
          ["axial", "z", "x", "y", 148.875, 148.875],
          ["sagittal", "x", "y", "z", 148.875, 192],
        ].map(([plane, normal, horizontal, vertical, width, height]) => ({
          plane,
          normal,
          horizontal,
          vertical,
          extent_mm: [
            -(width as number) / 2,
            (width as number) / 2,
            -(height as number) / 2,
            (height as number) / 2,
          ],
          panels: materials.flatMap((injected) =>
            materials.map((recovered) => ({
              injected,
              recovered,
              width: 32,
              height: 32,
              ...file(
                `maps/rep${replicate}/${plane}-${injected}-${recovered}.png`,
              ),
              samples: file(
                `maps/rep${replicate}/${plane}-${injected}-${recovered}.npy`,
              ),
              clipping: clipping([-0.05, 0.05], 1024),
              sampling: {
                normal_indices: [15, 16],
                upper_weight: 0.5,
                physical_coordinate_mm: 0,
                interpolation: "software metadata fixture",
              },
            })),
          ),
        }))
      : [],
  }));
  return {
    schema_version: 1,
    kind: "recorded-material-response",
    review_status: "accepted",
    evaluation_status: "complete",
    source: Object.fromEntries([
      ...[
        "review_sha256",
        "source_freeze_sha256",
        "acquisition_run_sha256",
        "fits_run_sha256",
        "evaluation_run_sha256",
        "six_fit_freeze_sha256",
      ].map((name) => [name, "0".repeat(64)]),
      ["source_commit", "0".repeat(40)],
    ]),
    grid: {
      shape: [32, 32, 32],
      spacing_mm: [4.65234375, 4.65234375, 6],
      origin_mm: [-72.111328125, -72.111328125, -93],
      orientation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    },
    perturbation_amplitude: 0.025,
    pattern_centre_mm_xyz: [25, 0, 0],
    pattern_radius_mm: 15,
    dilution: 0.9,
    signed_window: [-0.05, 0.05],
    absolute_window: [0, 0.05],
    default_plane: "coronal",
    outcomes,
    replicates,
    summary: available
      ? {
          available: true,
          mean: [
            [1.5, -0.5],
            [0.25, 2.5],
          ],
          minimum: [
            [1, -0.5],
            [0.25, 2],
          ],
          maximum: [
            [2, -0.5],
            [0.25, 3],
          ],
        }
      : { available: false, mean: null, minimum: null, maximum: null },
    limitations: Array.from(
      { length: 6 },
      () => "Metadata-only software fixture, without scientific evidence.",
    ),
    source_url: "https://doi.org/10.7937/tcia.2019.tt7f4v7o",
    licence: "CC BY 3.0 software contract fixture",
  };
}

describe("material response recorded-data admission", () => {
  it("retains all six unstarted outcomes without a fabricated matrix", () => {
    const data = parseMaterialResponse(fixture());
    expect(data.outcomes).toHaveLength(6);
    expect(data.replicates.map((item) => item.matrix)).toEqual([null, null]);
    expect(data.summary.available).toBe(false);
  });
  it("preserves mixed matrix row/column order, both replicates and the descriptive range", () => {
    const data = parseMaterialResponse(fixture(true));
    expect(data.replicates[0]!.matrix).toEqual([
      [1, -0.5],
      [0.25, 2],
    ]);
    expect(data.replicates[1]!.matrix).toEqual([
      [2, -0.5],
      [0.25, 3],
    ]);
    expect(data.summary.mean).toEqual([
      [1.5, -0.5],
      [0.25, 2.5],
    ]);
    expect(data.replicates[0]!.maps.map((p) => p.plane)).toEqual([
      "coronal",
      "axial",
      "sagittal",
    ]);
    expect(
      data.replicates[0]!.outside_metrics!.outside_whole_cells
        .signed_integral_jm_mm3,
    ).toEqual([
      [1, -2],
      [3, 4],
    ]);
  });
  it.each([
    "missing_outcome",
    "swap_replicates",
    "false_pass",
    "wrong_window",
    "wrong_plane",
    "wrong_shape",
    "unsafe_file",
    "wrong_map_role",
    "bad_clipping",
    "range_omits_replicate",
    "missing_metrics",
    "negative_absolute_integral",
    "wrong_outside_volume",
    "failed_values",
  ])("rejects %s", (change) => {
    const data = fixture(true);
    switch (change) {
      case "missing_outcome":
        data.outcomes.pop();
        break;
      case "swap_replicates":
        data.replicates.reverse();
        break;
      case "false_pass":
        data.outcomes[0]!.stationarity_passed = true;
        break;
      case "wrong_window":
        data.signed_window = [-0.5, 0.5];
        break;
      case "wrong_plane":
        data.replicates[0]!.maps[0]!.panels[0]!.sampling.upper_weight = 0;
        break;
      case "wrong_shape":
        data.grid.shape[0] = 64;
        break;
      case "unsafe_file":
        data.replicates[0]!.maps[0]!.panels[0]!.file = "../private.png";
        break;
      case "wrong_map_role":
        data.replicates[0]!.maps[0]!.panels[0]!.injected = "bone";
        break;
      case "bad_clipping":
        data.replicates[0]!.maps[0]!.panels[0]!.clipping.below = 1025;
        break;
      case "range_omits_replicate":
        data.summary.maximum![0]![0] = 1;
        break;
      case "missing_metrics":
        data.outcomes[0]!.metrics = null;
        break;
      case "negative_absolute_integral":
        data.replicates[0]!.outside_metrics!.outside_whole_cells.absolute_integral_jm_mm3[0]![1] =
          -1;
        break;
      case "wrong_outside_volume":
        data.replicates[0]!.outside_metrics!.outside_whole_cells.volume_mm3 = 11;
        break;
      case "failed_values":
        data.review_status = "retained_failed";
        data.evaluation_status = "failed";
        break;
    }
    expect(() => parseMaterialResponse(data)).toThrow();
  });
  it("does not round a small retained response to an invented zero", () => {
    expect(responseNumber(1.23e-9)).toBe("1.23e−9");
    expect(responseNumber(-0.2)).toBe("−0.2");
    expect(responseNumber(null)).toBe("—");
    expect(responseStatus("not_started")).toBe("Not started");
    expect(responseStatus("wall_time_budget")).toBe("wall time budget");
  });
});
