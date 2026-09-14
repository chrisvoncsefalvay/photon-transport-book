# Spectral projection and recovery

This driver loads supplied material fields and observations, then executes the canonical CUDA composition. It records the scalar objective, pose/calibration gradient, source snapshots, actual input digest and device identity. Set `mode` to `recover` to use the same safeguarded optimiser as primary pose recovery. There is no bundled anatomy, source spectrum or detector measurement, and the runner invents none when inputs are missing.

The 9 September 2026 GB10 acceptance pass executed the CUDA value/VJP, observation, multiview and profiling checks. This driver was exercised with an explicitly analytic vacuum/calibration fixture, in which pose is unidentifiable. Physical spectral studies still require supplied material, spectrum and detector inputs with provenance.

## Run with supplied inputs

From the repository root, using the Python 3.12 environment with the `gpu` dependency group:

```bash
PYTHONPATH=python .venv/bin/python experiments/spectral-projection/run.py \
  --inputs /path/to/supplied-data.npz \
  --model /path/to/physical-model.json \
  --output /path/to/new-private-run
```

The output directory must not exist and must be outside the repository. `--export-predictions` explicitly downloads the final detector arrays after evaluation and writes numerical JSON records; it creates no figures. Ordinary evaluation and optimisation transfer only pinned pose, calibration, loss, gradient and status buffers. The driver never promotes records into the book.

## Supplied input contract

`physical-model.json` has these top-level fields:

| Field               | Meaning                                                                                                                 |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `schema_version`    | Exactly `1`.                                                                                                            |
| `grid`              | `GridSpec` fields: `shape` in `(nz,ny,nx)` order, `spacing_mm`, and optional `origin_mm`/row-major `orientation`.       |
| `materials`         | Number of fixed-density material fractions, between 1 and 32.                                                           |
| `fields`            | NPZ member containing non-negative material-major fractions that sum to at most one per voxel; the remainder is vacuum. |
| `fields_provenance` | Provenance of the actual field input.                                                                                   |
| `calibration`       | Named `CalibrationBlock` definitions.                                                                                   |
| `views`             | One or more view descriptions below.                                                                                    |
| `pose_chart`        | Optional `PoseChart` arguments: `anchor`, six `scales` in mm/radians, and `rotation_radius_radians`.                    |

Every provenance object has non-empty `source`, `rights`, `description`, and a lowercase 64-character `sha256` identifying its original input bytes. Analytic fixtures must identify their defining formula and must not masquerade as physical measurements. The runner separately hashes the supplied NPZ archive; its bytes are not copied into source snapshots.

Each view supplies:

- Unique `name`; `geometry` containing source position, first pixel centre, orthonormal `u`/`v`, spacings and detector `(height,width)`; and an existing `calibration_group` name.
- Strictly increasing positive `energies_kev`; `coefficients`, the NPZ member containing `(material,energy)` total primary attenuation; `coefficients_unit` exactly `mm^-1`; and one `coefficients_provenance` object per material. Prepare other units and absorption-edge interpolation explicitly with `MaterialTable` before creating the archive.
- `weights`, naming **bin-integrated incident photon means** at the detector; `spectrum_provenance`; and `shared_weights` (default `true`). A density must first receive its stated energy quadrature weights, using `Spectrum.bin_fluence`.
- `response`, naming expected output per incident photon; `response_provenance`; `shared_response` (default `true`); `output_unit`; and `input_description` recording the fixed-density mixing and detector-acceptance assumptions.
- `observation`, naming the actual supplied observation array; `observation_provenance`; `objective` containing the `ObjectiveSpec` fields; optional `objective_weights` and a positive scalar `objective_weight` for this view.
- Optional `spatial_response`: odd `kernel_height`/`kernel_width`, non-negative `weights` summing to at most one, `provenance`, and `boundary: "zero"`. Detector dimensions come from the view geometry.

All NPZ arrays must already be native, contiguous binary32. Object arrays and pickle loading are forbidden. Supported shapes are flat arrays or their declared layouts: fractions `(M,nz,ny,nx)`/`(M,V)`, coefficients `(M,K)`, shared spectral inputs `(K,)`, varying spectral inputs `(K,height,width)`/`(K,P)`, and observations/weights `(height,width)`. The preparation boundary flattens these arrays explicitly and uploads once. It never guesses units or changes precision.

A calibration block has `name`, positive `gain` and `exposure`, finite `offset`, `fit_scale` (`none`, `gain` or `exposure`), and `fit_offset`. An active scale uses `reference * exp(scale_step * z)`; an active offset uses `reference + offset_step * z`. Exactly one multiplicative scale can be active. All views sharing a group must have the same output unit. Optional `scale_prior_precision` and `offset_prior_precision` apply Gaussian penalties to the dimensionless chart coordinates once per group.

Calibration values are rounded to binary32 for device evaluation. Gradients follow the smooth physical chart at those represented values; they do not differentiate the discontinuous rounding map. Shared scalar cotangents remain binary64 through the chart multiplication, because an early binary32 store can erase a small physical partial before a large logarithmic scale rescues it. The small fixed blur stencil also uses binary64: a positive tail can contribute representable signal when multiplied by a bright pixel, even if the coefficient alone would vanish in binary32.

## Meaning and limits

The forward order is material paths → spectral mean → optional spatial spread → shared electronic calibration → objective. Reverse execution follows the exact opposite order. It recomputes spectral depths and ray samples; no pixel-by-energy or ray-by-sample tape is retained. Multiple views share one object-to-world pose and sum gradients in that same chart.

A Poisson objective requires `output_unit: "counts"`, unit gain, zero offset, no active gain/offset and no spatial blur; its response must be a detection probability. Positive exposure can be fitted. General calibrated or energy-integrating signals use an explicitly chosen signal-domain squared-error objective. This restriction prevents an integrator's mean or a spatially correlated electronic image from silently acquiring a Poisson likelihood.

The adapter fits pose and shared scalar nuisance parameters. It holds material fields, spectrum, energy nodes, coefficients, response and blur fixed. It does not fit arbitrary pixel correction fields, differentiate visibility boundaries, supply a stochastic transport likelihood, or prove that a low residual identifies the true pose.

Extend the recorded acceptance checks to the supplied regime before making claims about a new physical study. Check directional derivatives away from non-smooth boundaries, recovery under the declared model mismatch and the relevant memory/launch workload. Input-table uncertainty and clinical validation remain separate from numerical correctness. The book includes the canonical source regions; this driver produces numerical records without rendering figures.
