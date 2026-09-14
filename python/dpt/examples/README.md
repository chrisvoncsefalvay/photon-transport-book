# Application examples

These drivers accompany Chapters 11–13. They use supplied acquisition data and the canonical `dpt` operators. Recorded numerical studies and their input preparation live under `experiments/`; the drivers do not bundle patient volumes or assume scanner calibration.

| Module                            | Application                                   | Implemented scope                                                                                                  |
| --------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `dpt.examples.registration`       | Fit a known volume to one or more radiographs | One rigid pose in a shared chart; independent primary counts; a scalar open beam and binary mask per view          |
| `dpt.examples.reconstruction`     | Estimate a voxel attenuation field            | Fixed calibrated geometry, primary counts, positivity and a quadratic spatial penalty                              |
| `dpt.examples.acquisition_design` | Select one further acquisition                | Rank supplied feasible candidates by local target uncertainty, using a known volume and a supplied prior precision |

Chapter 12 includes joint spectral material reconstruction. The example reconstruction driver here estimates a single attenuation field. The separate `dpt.material_reconstruction` module supplies the joint material solver; its executed TCIA-derived study and physical-input contract are documented in [the spectral reconstruction experiment](../../../experiments/spectral-reconstruction/README.md). Chapter 13 explains continuous acquisition design, while its driver uses finite candidates because the required geometry and higher derivatives are outside the current operator contract.

## Execution evidence and commands

The [registration and acquisition study](../../../experiments/application-study/README.md) records the prescribed pose fits and selected/fixed/random view comparisons. The [reconstruction study](../../../experiments/reconstruction-study/README.md) records scalar Poisson, post-log PWLS and joint material development runs. Read each report's stopping status, source identity and acquisition assumptions before using its results. Static checks and a completed run are distinct from numerical or physical acceptance.

Use a source checkout with Python 3.12+ and the pinned `gpu` dependency group installed. These commands illustrate the supplied-case interface; use the linked experiment recipes for the recorded study inputs and settings.

```bash
PYTHONPATH=python .venv/bin/python -m dpt.examples.registration \
  --case /data/registration/case.json --output /data/runs/registration-001

PYTHONPATH=python .venv/bin/python -m dpt.examples.reconstruction \
  --case /data/reconstruction/case.json --output /data/runs/reconstruction-001

PYTHONPATH=python .venv/bin/python -m dpt.examples.acquisition_design \
  --case /data/design/case.json --output /data/runs/design-001
```

Each command accepts `--device cuda:0`. The output must be a new directory outside the repository; existing runs are preserved. Source-checkout execution lets `RunRecorder` retain the package, driver, configuration and dependency-lock identities.

## Shared case and array format

A case is UTF-8 JSON with `schema_version` equal to the integer `1`, an `arrays` object and an application section named `registration`, `reconstruction` or `acquisition_design`. Repeated JSON keys are rejected. Application settings refer to array names rather than opening unrecorded files themselves.

Each entry in `arrays` needs all of the following fields:

| Field    | Meaning                                                                 |
| -------- | ----------------------------------------------------------------------- |
| `path`   | Path to an existing `.npy` file, absolute or relative to the case JSON  |
| `sha256` | The file's actual 64-character hexadecimal SHA-256 digest               |
| `units`  | Exact physical-unit string required by the application role             |
| `source` | Dataset/acquisition identity and the preparation or calibration applied |
| `rights` | The applicable use and redistribution terms                             |

Array names must start with a letter or digit, contain at most 128 characters, and use only letters, digits, dots, underscores or hyphens. Numeric files must be native-endian and C-contiguous; object arrays, NPZ archives, implicit casts, nonfinite values and symlink files are rejected. Drivers require exact dtype and shape, then check physical domains. If the file contains Hounsfield units, a conversion to attenuation must be justified, performed and recorded before it is supplied as `mm^-1`. Changing its unit label does not perform that conversion.

The helper checks each digest before recording and again against the bytes used to load an array. `RunRecorder` snapshots every declared input, including optional evaluation references, and checks source integrity at completion. The drivers keep references out of fitting and candidate selection. File provenance establishes identity; it does not establish that the acquisition was calibrated correctly.

The case format is a supplied-data boundary, so no generated observations, downloaded anatomy or substitute datasets are included. Array dimensions and numerical settings belong to the actual acquisition. The fields below describe the contract without fabricating a sample scan.

## Geometry and field conventions

| Object             | JSON fields and convention                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `GridSpec`         | `shape: [nz, ny, nx]`, `spacing_mm: [hx, hy, hz]`, `origin_mm: [x, y, z]`, optional `orientation`: nine row-major entries |
| `DetectorGeometry` | `source_mm`, `origin_mm`, `u`, `v`, `spacing_mm: [column, row]`, `shape: [height, width]`                                 |
| `RigidTransform`   | `rotation`: nine row-major entries; `translation_mm`: three entries; maps object coordinates into world coordinates       |

Grid origin is the first sample centre; detector origin is the first pixel centre. Orientations and detector axes must meet the canonical orthonormality and handedness requirements. Voxel arrays have shape `(nz, ny, nx)` with x contiguous; image arrays have shape `(height, width)`. Flattening changes storage shape only, not physical orientation. Pose coordinates are ordered translation x/y/z followed by rotation x/y/z, in millimetres and radians.

## Registration configuration

Put these settings under `registration`:

| Setting           | Required value                                                                                                    |
| ----------------- | ----------------------------------------------------------------------------------------------------------------- |
| `grid`            | Canonical grid fields above                                                                                       |
| `attenuation`     | Array name: FP32 `grid.shape`, units `mm^-1`, nonnegative                                                         |
| `chart`           | `anchor`: rigid-transform fields; `scales`: six positive physical scales; optional `rotation_radius_radians`      |
| `samples_per_ray` | Positive integer quadrature count                                                                                 |
| `precision`       | Optional `float32` or `float64`; default `float64` retains primary intermediates and count predictions in FP64    |
| `integration`     | Optional `midpoint` (default) or `cell_gauss`; the latter integrates the trilinear field without a sampling count |
| `policy`          | Keyword settings for the canonical `RecoveryPolicy`; unspecified members retain its documented defaults           |
| `views`           | Nonempty list of the view records below                                                                           |

Each view contains a unique `id` (letters, digits, underscores and hyphens), `geometry`, `observation`, `mask` and `open_beam_counts`. Observation arrays are FP32 images with units `counts`; this example accepts integer values from zero through `2**24` so its storage can represent every allowed integer. Masks are FP32 images with units `dimensionless`, contain only zero or one, and include at least one pixel. Set excluded observations to zero in a recorded preprocessing step and retain the raw source: the objective checks its numerical domain before applying the mask. `open_beam_counts` is a positive scalar expected primary count per pixel in the absence of the object. A spatially varying flat field needs a different composition; the driver does not silently replace it with an average.

Optional `evaluation` contains `reference_pose`, `landmarks` (an FP64 `(N, 3)` array in `mm`), and nonempty `source` and `uncertainty` descriptions. Landmark coordinates are in the object frame. These values are read for geometric scoring after the accepted pose is fixed. They must be independently supplied; reusing the fitted pose as its own reference would measure nothing.

## Reconstruction configuration

Put these settings under `reconstruction`:

| Setting           | Required value                                                              |
| ----------------- | --------------------------------------------------------------------------- |
| `grid`            | Canonical grid fields                                                       |
| `fixed_pose`      | Object-to-world rigid-transform fields                                      |
| `initial_volume`  | Array name: FP32 `grid.shape`, units `mm^-1`, nonnegative                   |
| `samples_per_ray` | Positive integer quadrature count                                           |
| `views`           | Nonempty list of `geometry`, `counts` array name and `open_beam` array name |
| `solver`          | All seven controls in the table below; none has an implicit default         |

Both count and open-beam arrays are FP32 images with units `counts`. Observations are nonnegative integer detected counts no greater than `2**24`; open-beam values are strictly positive calibrated expected counts and need not be integers. The known geometry remains fixed throughout the solve.

| `solver` field                    | Domain and meaning                                           |
| --------------------------------- | ------------------------------------------------------------ |
| `iterations`                      | Integer at least one; maximum accepted-update iterations     |
| `initial_step_mm_inverse_squared` | Finite positive trial step in `mm^-2`                        |
| `backtracking_factor`             | Strictly between zero and one; step multiplier on rejection  |
| `armijo`                          | Strictly between zero and one; required-decrease coefficient |
| `maximum_backtracks`              | Integer at least one; trial budget per iteration             |
| `gradient_mapping_tolerance_mm`   | Finite nonnegative stationarity threshold in `mm`            |
| `regularisation_mm`               | Finite nonnegative spatial-penalty coefficient in `mm`       |

The optional `mapping_step_mm_inverse_squared` is a finite positive step in `mm^-2` for the fixed Euclidean stationarity diagnostic. When omitted or `None`, it uses `initial_step_mm_inverse_squared`, preserving the original behaviour; the seven controls above remain required. Supplying it separately lets the initial Armijo trial change without redefining the convergence diagnostic.

The spatial penalty uses neighbouring voxel differences divided by physical spacing, with voxel-volume weights and no neighbour beyond the array boundary. Its coefficient has units of millimetres. The Euclidean volume gradient has units of millimetres for the dimensionless count objective; therefore the projected-gradient step has units `mm^-2`. Choose the controls on declared development data and freeze them before final evaluation.

The example uses Torch CUDA arrays for the constrained update and regulariser while the canonical Warp operators provide image formation and explicit adjoints. Shared device views, stream ordering and all retained buffers require runtime verification. No inference about memory use or throughput follows from having written that composition.

## Acquisition-design configuration

Put these settings under `acquisition_design`:

| Setting                           | Required value                                                                                                          |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `grid`, `pose`                    | Canonical grid and object-to-world transform                                                                            |
| `attenuation`                     | FP32 `grid.shape` array, units `mm^-1`, nonnegative                                                                     |
| `parameter_scales`                | Six positive physical scales for the local right pose coordinates                                                       |
| `prior_precision`                 | FP64 `(6, 6)` array, units `dimensionless`, symmetric positive definite in the scaled coordinates                       |
| `targets_object_mm`               | FP64 `(N, 3)` array, units `mm`, with 1–256 targets selected from known anatomy or the declared task                    |
| `current_information_description` | Explanation of the prior and already acquired information represented by the precision                                  |
| `samples_per_ray`                 | Positive integer quadrature count                                                                                       |
| `precision`                       | Optional `float32` (default) or `float64`; depth and per-ray cotangents share this storage precision                    |
| `integration`                     | Optional `midpoint` (default) or `cell_gauss`; use the declared fitting integration when forming its Fisher information |
| `max_candidate_pixels`            | Explicit bound on the retained per-pixel six-column Jacobian                                                            |
| `maximum_cost`, `cost_unit`       | Acquisition-cost limit and calibrated unit: `mAs`, `s` or `expected_detector_photons`                                   |
| `candidates`                      | Nonempty list of named acquisition alternatives                                                                         |
| `maximum_precision_condition`     | Optional finite condition-number limit, at least one; default `1e12`                                                    |
| `rank_relative_tolerance`         | Optional relative eigenvalue threshold strictly between zero and one; default `1e-10`                                   |

Each candidate contains a unique `name`, `geometry`, an `open_beam` array name (nonnegative FP32 image, units `photons/pixel`, with at least one positive pixel), a finite positive `cost` and a `feasibility_description`. At least one candidate must fit the positive `maximum_cost` budget. The candidate set and its feasibility assessment are supplied inputs. The driver does not command a C-arm or verify mechanical clearance. Different calibrated exposure settings can be represented as different alternatives.

`expected_detector_photons` means the sum of the calibrated open-beam image, in the absence of the object. Each candidate's stated cost must match that sum to relative tolerance `1e-6`. This budget does not represent the transmitted count sum or absorbed dose. With `mAs` or `s`, the supplied cost records the acquisition setting or duration; the driver cannot infer it from the image.

The driver bounds candidate pixel counts and rejects precision matrices beyond the configured condition limit. It also rejects illuminated means below the normal FP64 range and weighted derivatives outside its stated product-accumulation range. Zero-beam pixels contribute zero information. Numerical rejection requires reviewing the input scales or reduction method; no count floor is added to force a ranking.

Prior precision uses the declared scaled local pose chart; targets remain unscaled object-frame coordinates in millimetres. Include previous observations in the current precision once. The targets express the known task; they are not withheld evaluation landmarks obtained by looking at future measurements. The driver reports predicted local uncertainty under its ideal-count model. It does not report achieved registration accuracy, reconstruct a new unknown volume or estimate absorbed dose.

## Outputs and acceptance checks

Every run writes the effective case configuration, source/input snapshots and `run.json`. The recorder distinguishes running, failed and completed execution; completion does not establish scientific correctness. Registration additionally records its accepted transform, optimisation history, final per-view predictions and any independent geometric score. Reconstruction records its accepted volume and solver history. Acquisition design records candidate diagnostics and the selected feasible alternative under the stated criterion.

| Driver             | Application outputs inside the recorded run                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------ |
| Registration       | `optimisation.json`, `fit-report.json`, `prediction-{view-id}.npy`                                           |
| Reconstruction     | `attenuation.npy`, `solver.json`, `predictions/view-NNNN.npy`                                                |
| Acquisition design | `design.json`, `candidate_information.npy`, `candidate_local_covariance.npy`, `current_local_covariance.npy` |

Before using those outputs as evidence, independently check the example input handling, objective/gradient composition, pose/chart conventions, constrained update acceptance, stream interoperation and range handling. Reconstruction also needs adjoint/directional references, regularisation checks across voxel spacings and representative memory measurements. Design needs independently calculated small information-matrix cases, rank/conditioning cases, budget checks and held-out comparison with fixed and random feasible policies. Reference data must remain outside tuning and selection. Preserve failed runs when evaluating reliability.
