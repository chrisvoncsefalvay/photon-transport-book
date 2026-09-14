import { describe, expect, it } from "vitest";
import {
  validateComparisonStudy,
  type ComparisonDisplay,
  type ComparisonImage,
  type ComparisonOutcome,
  type ReconstructionComparisonStudy,
} from "../../src/lib/reconstruction-comparison";

// Host display-contract fixtures. These are never generated assets or results.
function fixture(): ReconstructionComparisonStudy {
  const methods = [
    "scalar_poisson",
    "scalar_pwls",
    "spectral_metric",
    "spectral_euclidean",
  ];
  const regimes = ["dense", "sparse", "limited"] as const;
  const outcomes: ComparisonOutcome[] = [];
  for (const caseID of [2, 0])
    for (const regime of regimes)
      for (const replicate of [0, 1])
        for (const method of methods) {
          if (method === "spectral_euclidean" && regime !== "dense") continue;
          outcomes.push({
            id: `case${caseID}_${regime}_r${replicate}_${method}`,
            case: caseID,
            regime,
            replicate,
            method,
            status: "not_started",
            evaluation_status: "unavailable_fit",
            hard_correctness_passed: null,
            termination: "not_started",
            accepted_updates: null,
            solve_seconds: null,
            stationarity_passed: null,
            water_rmse: null,
            bone_rmse: null,
            attenuation_rmse_mm_inverse: null,
            withheld_deviance_per_sample: null,
          });
        }
  const image = (window: [number, number]): ComparisonImage => ({
    file: "contract-only.png",
    sha256: "a".repeat(64),
    width: 48,
    height: 48,
    physical_width_mm: 148.875,
    physical_height_mm: 192,
    window,
    below_window_pixels: 0,
    above_window_pixels: 0,
    alt: "Unrendered display-contract fixture",
  });
  const displays: ComparisonDisplay[] = methods.map((method) => {
    const model = method.startsWith("spectral") ? "spectral" : "scalar";
    const selectedRegimes =
      method === "spectral_euclidean" ? [regimes[0]] : regimes;
    const groups = selectedRegimes.map((regime) => ({
      id: regime,
      label: regime,
      outcome_id: `case2_${regime}_r0_${method}`,
      views: regime === "dense" ? 144 : 36,
      incident_photons_per_view: regime === "dense" ? 200000 : 800000,
      angle_pairs_yaw_tilt_degrees: [-15, 0, 15].flatMap((tilt) =>
        Array.from(
          { length: regime === "dense" ? 48 : 12 },
          (_, i): [number, number] => [
            regime === "limited"
              ? -60 + (120 * i) / 11
              : (360 * i) / (regime === "dense" ? 48 : 12),
            tilt,
          ],
        ),
      ),
    }));
    return {
      method,
      model,
      label: method,
      regimes: groups,
      slices: (model === "spectral"
        ? ["bone", "water", "attenuation"]
        : ["attenuation"]
      ).flatMap((quantity) =>
        (["axial", "coronal", "sagittal"] as const).map((plane) => ({
          key: `${quantity}-${plane}`,
          quantity,
          quantity_label: quantity,
          plane,
          plane_label: plane,
          units: quantity === "attenuation" ? "mm⁻¹" : "fraction",
          reference: image(quantity === "attenuation" ? [0, 0.05] : [0, 1]),
          recovered: groups.map((group) => ({
            outcome_id: group.outcome_id,
            image: null,
            signed_error: null,
          })),
        })),
      ),
      radiograph: {
        label: "Withheld view 1",
        view_index: 0,
        yaw_degrees: 3.75,
        tilt_degrees: -15,
        channel_label: model === "scalar" ? "80 keV" : "20–55 keV",
        observation: image([0, 8]),
        recovered: groups.map((group) => ({
          outcome_id: group.outcome_id,
          prediction: null,
          residual: null,
        })),
      },
    };
  });
  return {
    schema_version: 1,
    scope: "CPU display-contract fixture",
    default_method: "spectral_metric",
    representative: { case: 2, replicate: 0 },
    displays,
    outcomes,
    source_sha256: { "contract-fixture": "b".repeat(64) },
    rights: "Unrendered contract fixture",
    limitations: ["Not a scientific result"],
  };
}

describe("recorded reconstruction display admission", () => {
  it("retains the full prescribed schedule when every fit is unavailable", () => {
    expect(() => validateComparisonStudy(fixture())).not.toThrow();
  });

  it("rejects a missing or duplicated outcome, including an unstarted fit", () => {
    const missing = fixture();
    missing.outcomes.pop();
    expect(() => validateComparisonStudy(missing)).toThrow("all forty");
    const duplicate = fixture();
    duplicate.outcomes[39] = structuredClone(duplicate.outcomes[0]!);
    expect(() => validateComparisonStudy(duplicate)).toThrow(
      "duplicate outcome",
    );
  });

  it("rejects changing the fixed representative and unavailable numerical results", () => {
    const changed = fixture();
    Object.assign(changed.representative, { case: 0 });
    expect(() => validateComparisonStudy(changed)).toThrow("representative");
    const unavailable = fixture();
    unavailable.outcomes[0]!.water_rmse = 0;
    expect(() => validateComparisonStudy(unavailable)).toThrow(
      "unavailable evaluation",
    );
  });

  it("rejects different photon populations and an invented baseline regime", () => {
    const exposure = fixture();
    exposure.displays[0]!.regimes[1]!.incident_photons_per_view = 200000;
    expect(() => validateComparisonStudy(exposure)).toThrow(
      "photon population",
    );
    const baseline = fixture();
    baseline.displays[3]!.regimes.push(
      structuredClone(baseline.displays[0]!.regimes[1]!),
    );
    expect(() => validateComparisonStudy(baseline)).toThrow(
      "acquisition regimes",
    );
  });

  it("retains measured metrics after a failed numerical check without showing its images", () => {
    const failedCheck = fixture();
    Object.assign(failedCheck.outcomes[0]!, {
      status: "completed",
      evaluation_status: "complete",
      hard_correctness_passed: false,
      attenuation_rmse_mm_inverse: 0.001,
      withheld_deviance_per_sample: 1.2,
    });
    expect(() => validateComparisonStudy(failedCheck)).not.toThrow();
    const slice = failedCheck.displays[0]!.slices[0]!;
    slice.recovered[0]!.image = structuredClone(slice.reference);
    expect(() => validateComparisonStudy(failedCheck)).toThrow();
  });

  it("rejects a claimed evaluation of an uncompleted fit", () => {
    const unfinished = fixture();
    Object.assign(unfinished.outcomes[0]!, {
      evaluation_status: "complete",
      hard_correctness_passed: false,
    });
    expect(() => validateComparisonStudy(unfinished)).toThrow(
      "evaluation requires a completed fit",
    );
  });

  it("rejects changed display windows, clipping counts and omitted planes", () => {
    const window = fixture();
    window.displays[2]!.slices[0]!.reference.window = [0, 0.65];
    expect(() => validateComparisonStudy(window)).toThrow("display window");
    const clipping = fixture();
    clipping.displays[0]!.radiograph.observation.above_window_pixels = 100000;
    expect(() => validateComparisonStudy(clipping)).toThrow("clipping counts");
    const plane = fixture();
    plane.displays[0]!.slices.pop();
    expect(() => validateComparisonStudy(plane)).toThrow(
      "centre-plane coverage",
    );
  });

  it("rejects substituted withheld views and private image paths", () => {
    const view = fixture();
    view.displays[0]!.radiograph.view_index = 3;
    expect(() => validateComparisonStudy(view)).toThrow(
      "withheld display view",
    );
    const path = fixture();
    path.displays[0]!.slices[0]!.reference.file = "../../private.png";
    expect(() => validateComparisonStudy(path)).toThrow("unsafe image path");
  });
});
