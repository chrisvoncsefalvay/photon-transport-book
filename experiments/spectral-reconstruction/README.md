# Spectral material reconstruction

This experiment builds a water/reference-bone attenuation model, simulates
energy-resolved radiographs from an acquired TCIA CT, and jointly reconstructs
two material-fraction volumes with the canonical CUDA operators. The source CT
supplies anatomy for an **assigned material phantom**; it is not a measurement
of the patient's water/bone composition.

## Recorded vertebral result

The recorded study retains its original private inputs and full execution records.
After 1,000 accepted steps, global
water/bone fraction RMSE is 0.03917/0.03326, versus 0.21403/0.21403 initially.
Bone Dice is 0.97990 at the fixed 0.325 fraction threshold. Both recorded inverse
sampling checks pass, and the exported material volumes reproduce the solver's
arrays and source-RAS coordinates exactly at file-header precision.

This is a finite-budget assigned-phantom recovery demonstration. The projected
gradient tolerance is unmet, and held-out discrepancy remains above the recorded
generating-expectation noise baseline. Coarse 64³ geometry, cropped bone/ribs,
residual errors and the ideal primary-only detector model remain explicit.
The complete inputs, metrics, independent reviews, device checks and exact file
scope remain in the authoring archive; they are not bundled with this public
execution recipe. The open-input recipe below prepares a separate physical
model; it does not reproduce or replace these historical numerical results.

## Inputs and acquisition

`prepare_physics.py` retains original NIST water and ICRU-44 cortical-bone tables,
source hashes, densities, rights records and an exact preparation snapshot.
The conversion is `mu[mm^-1] = (mu/rho)[cm^2/g] * rho[g/cm^3] / 10`.
The canonical edge-aware log-log interpolator supplies 100 energy centres from
20.5 to 119.5 keV. SpekPy 2.5.4 supplies a tungsten 120 kVp spectrum with a
12-degree anode and 2.5 mm aluminium filtration. Energy-bin weights are integrated
once and normalised to 200,000 incident photons per detector pixel per exposure.

The response has three disjoint ideal photon-count channels: 20–55, 55–80 and
80–120 keV, with unit quantum efficiency in each interval. The source weights
already describe the open beam at each detector pixel. This model assumes
spatially uniform fluence; it does not infer inverse-square corrections, heel
effects or proprietary scanner calibration. Scatter, blur, pileup, charge
sharing and electronic read noise are absent. Disjoint bins use separate
Poisson random-number domains. VMI CT images are not treated as photon channels.

The principal controlled example uses `prepare_vertebrae.py`: a native
192³ crop of CT-ORG case 2 containing the vertebral region and adjacent rib
segments. The archived bone label assigns a bone fraction of 0.65, with water
filling the remaining fraction throughout a known rectangular container.
The crop is 148.875 × 148.875 × 192 mm; native voxel spacing is
0.775390625 × 0.775390625 × 1 mm. Conservative 3 × 3 × 3 cell averaging gives
a 64³ inverse grid at 2.326171875 × 2.326171875 × 3 mm. The lateral ribs and
superior/inferior bone can intersect the finite box faces; this is a controlled
anatomical phantom, not an isolated-vertebra segmentation or a complete patient.
Native CT samples and the bone mask are preserved separately for provenance
and evaluation. Both reconstructed material fields remain free.

The earlier coarse pelvis diagnostic uses `prepare_anatomy.py`. It verifies
the archived CT-ORG case 2 CT and label hashes,
preserves the originals and uses the inferior 320 mm crop. Header-led LAS-to-RAS
reorientation preserves samples; it cannot independently establish source
laterality. A recorded CT foreground rule removes the disconnected table.
Unlabelled voxels are not assumed to be air. The assigned bone fraction is 0.65
inside the body-intersected bone label, with water fraction 0.35 there; elsewhere
the body water fraction is `clip(1 + HU/1000, 0, 1)`. Remaining fraction is vacuum.
These are fixed modelling choices, not material calibration. Unknown high-HU
content outside the bone label is capped at water; iodine is not inferred.

Aligned native-cell averaging produces a `128 x 128 x 160` reference grid and a
`64 x 64 x 80` first inverse grid, with the same voxel-face support. The inverse
spacing is `6.203125 x 6.203125 x 4 mm`. This is a coarse numerical case, not the
final surface-resolution target for the book. The finite crop has vacuum
outside its boundary.

The vertebral acquisition uses three tilted circular orbits at -15, 0 and
15 degrees, 48 views per orbit, source-isocentre distance 1000 mm and source-detector
distance 1500 mm. Detector coverage is derived from all support corners at all
views. Eighteen complete views are withheld; 126 views fit all three channels
on a 96 × 96 detector. No continuous cone-beam completeness claim is made.
The forward projector uses 1,024 samples per intersecting ray and the inverse
study uses 512. The earlier pelvis diagnostic used 72 views and a 96 × 80 detector.

## Solver

`dpt.material_reconstruction.MaterialReconstruction` composes material-path
projection, the spectral expectation and the Poisson half-deviance. Every
voxel satisfies `f_m >= 0` and `sum_m f_m <= 1` through simplex projection
in the selected update metric, subject to FP32 rounding. Fraction fields are
dimensionless. The spatial penalty is

`beta * voxel_volume / 2 * sum_neighbour_pairs ((f[j]-f[i])/spacing_axis)^2`,

with `beta` in inverse millimetres and no differences across the volume boundary.
The explicit spectral adjoint is summed across channels in FP64 before the
canonical volume backprojection. Volume-gradient scatter uses the canonical
FP32 atomics; floating-point scheduling is not bitwise deterministic.

The solver binds accepted and trial projection buffers separately, allocates
image/volume workspaces during preparation, and uses projected Armijo steps.
`initial_step` starts the search; `maximum_step` separately bounds growth.
Its report distinguishes projected-gradient convergence, exhausted iteration
budget and a failed line search. A completed run record means that computation
and output verification finished; it does not certify convergence.

The solver accepts fitting views and fixed physical inputs. It accepts no
reference volume or CT-derived support mask. The experiment initialises water
at 1 and bone at 0 throughout the declared water container; this uses no internal
bone anatomy. The earlier pelvis initialisation was water 0.1 and bone 0.02.
Evaluation references are
opened only after fitting, anchored to the acquisition's frozen metadata.
Step/budget development must use training quantities; withheld errors must not
select an iteration, regularisation strength or checkpoint.

The optional two-material metric uses the noise-weighted spectral Jacobian at
prescribed water/bone paths of 200/10 mm and normalises its Fisher matrix to
trace two. It depends only on frozen physical inputs. Its positive-definite
quadratic projection solves the constrained two-variable update exactly;
it is not an unconstrained inverse-gradient step followed by Euclidean clipping.
Stopping always uses the Euclidean projected-gradient mapping. The vertebral
run fixes a mapping step of `1e-6` and a maximum budget of 1,000 accepted updates.
The run report, rather than the existence of images, establishes its termination.

## Open input preparation

`provision_inputs.py` pins the original CT-ORG CT and label by compressed
byte length and SHA-256, records CC BY 3.0 attribution before intake, validates
both gzip CRCs and their matching NIfTI geometry, and preserves the originals.
Case 2 is the default; `--case 0` selects the independently verified second pair.
The pins come from the verified official TCIA transfer; they are not advertised
as upstream-published checksums. CT-ORG currently uses TCIA's public Aspera
package. The script prints the official collection link and exact filenames;
it does not install a transfer client or assert an unverified HTTPS mirror.
Download only `volume-2.nii.gz` and `labels-2.nii.gz` through that link, then
pass their directory with `--cache`. Reusing a cache requires the same exact
bytes. Failed partial outputs and completed directories are never overwritten.

For case 0, request its instructions with `--case 0 --instructions` and obtain
`volume-0.nii.gz` and `labels-0.nii.gz` instead. These are separate original
volumes, not alternate names for the case-2 inputs.

`prepare_open_physics.py` defines the separate model
`xraylib-icrp-spekpy120-al25-v1`. It requires xraylib 4.3.0 and SpekPy 2.5.4,
retains their pinned source/distribution archives and original BSD/MIT notices,
and verifies installed distribution files against their wheel records. Water
uses xraylib's `Water, Liquid` compound at 1.0 g/cm³; bone uses
`Bone, Cortical (ICRP)` at 1.85 g/cm³. The built-in compound metadata carries
xraylib's BSD notice. This bone composition and density differ from the old
NIST ICRU-44 model at 1.92 g/cm³. No separately downloaded NIST SRD tables enter
the new preparation.

The new source uses the same 120 kVp tungsten, 12-degree anode and native
1 keV centres, with SpekPy's `casim` model and `pene` attenuation-data setting.
Aluminium filtration is applied explicitly as
`exp(-xraylib.CS_Total(13, E) * 2.699 / 10 * 2.5)` for nominal 2.5 mm Al.
The script preserves the unfiltered spectrum and filtration factors. It retains
20–120 keV, normalises integrated bins once to 200,000 photons per pixel per
exposure, and uses the fixed 20–55/55–80/80–120 keV ideal channels. No channel
search or measured scanner calibration is performed. Compound cross sections
are checked against independently mass-weighted elemental totals, with fixed
80 keV reference values, a non-unit energy-bin-width check and finite-difference
checks of the two-path spectral sensitivity. These establish input arithmetic
and local path rank; they do not establish tomographic uniqueness or convergence.

Use a new external directory and a separate Python 3.12 environment. The input
environment is CPU-only; the later acquisition and reconstruction commands use
the project's separately validated GPU environment. An optional repeatable
`--package-cache` directory can supply either pinned archive; missing package
archives are fetched from the exact official PyPI URLs after recording rights.
Raw CT and label files are never copied into the checkout or public assets.

```bash
python3.12 -m venv "$DPT_STUDY_OUTPUT/input-env"
"$DPT_STUDY_OUTPUT/input-env/bin/python" -m pip install \
  numpy==2.5.3 scipy==1.18.1 matplotlib==3.10.7 nibabel==5.4.2 \
  xraylib==4.3.0 spekpy==2.5.4

"$DPT_STUDY_OUTPUT/input-env/bin/python" \
  experiments/spectral-reconstruction/provision_inputs.py \
  --instructions --output "$DPT_STUDY_OUTPUT/download-instructions"

# Follow the official TCIA link above and download the two named files.
# DPT_CT_ORG_DOWNLOAD contains those original compressed files.
"$DPT_STUDY_OUTPUT/input-env/bin/python" \
  experiments/spectral-reconstruction/provision_inputs.py \
  --cache "$DPT_CT_ORG_DOWNLOAD" --output "$DPT_STUDY_OUTPUT/CT-ORG-case2"

PYTHONPATH=python "$DPT_STUDY_OUTPUT/input-env/bin/python" \
  experiments/spectral-reconstruction/prepare_open_physics.py \
  --output "$DPT_STUDY_OUTPUT/physics-open-v1"

export DPT_CT_ORG_SOURCE="$DPT_STUDY_OUTPUT/CT-ORG-case2"
export DPT_PHYSICS_INPUTS="$DPT_STUDY_OUTPUT/physics-open-v1"
```

## Preparing the second morphology

After obtaining the two case-0 files through the recorded TCIA link, run:

```bash
"$DPT_STUDY_OUTPUT/input-env/bin/python" \
  experiments/spectral-reconstruction/provision_inputs.py \
  --case 0 --cache "$DPT_CT_ORG_DOWNLOAD" \
  --output "$DPT_STUDY_OUTPUT/CT-ORG-case0"

"$DPT_STUDY_OUTPUT/input-env/bin/python" \
  experiments/spectral-reconstruction/prepare_case0.py \
  --source "$DPT_STUDY_OUTPUT/CT-ORG-case0" \
  --output "$DPT_STUDY_OUTPUT/anatomy-case0-new"
```

The fixed header/label rule selects native XYZ `[160:352,99:291,21:53]`.
The crop contains lumbar vertebrae and partial sacral/iliac bone, with its cut
faces retained. Its 192 × 192 × 32 samples have spacing
0.703125 × 0.703125 × 5 mm. Exact 4 × 4 × 1 cell averaging gives a
48 × 48 × 32 inverse grid in the same 135 × 135 × 160 mm box;
binary32 rounding and its material-volume bound are reported separately.
No interpolation adds axial resolution to the native 5 mm slices.

As in the primary case, the known box is assigned water and bone from the
archived bone label; it is not a decomposition of measured patient composition.
CT, labels and reference fractions remain external generator/evaluator inputs,
excluded from fitting. This command prepares anatomy only. The separate
[comparison study](../reconstruction-study/README.md) controls acquisition,
noise, methods, stopping and evaluation.

## Running a separate study

Keep data and outputs outside the checkout. The physics metadata records its
pinned package environment. The remaining
commands use the checkout's Python 3.12 GPU environment with NumPy, nibabel and
SciPy for offline preparation; rendering additionally needs Matplotlib and
scikit-image. All output directories must be new. Set `DPT_CT_ORG_SOURCE` to
the verified source collection, `DPT_PHYSICS_INPUTS` to the prepared physical
inputs and `DPT_STUDY_OUTPUT` to an external study directory. These commands
require the separately provisioned inputs above. Outputs from the new open
model must keep its model identity and be evaluated as a new study. The old
NIST table redistribution question remains unresolved for the old inputs.

```bash
.venv/bin/python experiments/spectral-reconstruction/prepare_vertebrae.py \
  --source "$DPT_CT_ORG_SOURCE" \
  --output "$DPT_STUDY_OUTPUT/anatomy-vertebrae-new"

.venv/bin/python experiments/spectral-reconstruction/run.py acquire \
  --physics "$DPT_PHYSICS_INPUTS" \
  --anatomy "$DPT_STUDY_OUTPUT/anatomy-vertebrae-new" \
  --output "$DPT_STUDY_OUTPUT/acquisition-new" \
  --samples 1024 --views-per-ring 48 --height 96 --width 96

.venv/bin/python experiments/spectral-reconstruction/run.py solve \
  --physics "$DPT_PHYSICS_INPUTS" \
  --acquisition "$DPT_STUDY_OUTPUT/acquisition-new" \
  --output "$DPT_STUDY_OUTPUT/reconstruction-new" \
  --samples 512 --iterations 1000 --initial-step 0.00001 --regularisation 0.1 \
  --metric spectral --initial-water 1 --initial-bone 0 --mapping-step 0.000001

.venv/bin/python experiments/spectral-reconstruction/render.py \
  --run "$DPT_STUDY_OUTPUT/reconstruction-new" \
  --output "$DPT_STUDY_OUTPUT/figures-new"

.venv/bin/python experiments/spectral-reconstruction/build_viewer.py \
  --figures "$DPT_STUDY_OUTPUT/figures-new"
```

`RunRecorder` retains the complete Python package, driver, inputs, configuration
and output hashes, and rejects source changes during execution. Freeze source
edits while a run is active. Acquisitions check representative vacuum, bone,
water and boundary rays against independent FP64 piecewise-Gauss integration.
Doubling forward sampling must change the checked images by less than 0.1
Poisson standard deviation per pixel. Reconstruction exports fractions,
checkpoints, fitted and withheld predictions, training history and evaluation.
Figures use actual arrays, equal physical scales and a fixed bone isosurface;
they do not smooth or substitute reference geometry for recovered geometry.
The optional viewer embeds the verified surface meshes and static comparison
in one private `bone-viewer.html`. Both panes use one orthographic camera and
the same physical scale. It bundles the existing Three.js dependency with its
licence, requires no network access and retains the static image when WebGL or
JavaScript is unavailable. Browser rendering changes lighting normals, not
surface coordinates or topology; its small FP32 positioning roundoff is recorded.

## Provenance and scope

- Anatomy: [CT-ORG](https://doi.org/10.7937/tcia.2019.tt7f4v7o), Rister et al.,
  TCIA, CC BY 3.0. Source CT and masks remain unchanged in the external source collection.
- Attenuation: [NIST X-ray mass attenuation tables](https://physics.nist.gov/PhysRefData/XrayMassCoef/tab4.html),
  water and ICRU-44 cortical bone. Original tables and derivatives remain private;
  the recorded NIST SRD rights question must be resolved before table redistribution.
- New open model: [xraylib 4.3.0](https://pypi.org/project/xraylib/4.3.0/),
  BSD-3-Clause, with original notices retained for software and built-in compound
  metadata; [Schoonjans et al.](https://doi.org/10.1016/j.sab.2011.09.011).
  Generated ICRP water/bone coefficients belong to this new recipe and are not
  relabelled historical NIST inputs.
- Spectrum: [SpekPy 2.5.4](https://pypi.org/project/spekpy/2.5.4/), MIT; the package
  licence, original wheel and generator settings are retained with the inputs.

The historical recorded study establishes a numerical reconstruction path under
its declared simulated physics. The new open model has passed CPU input checks,
a separate 144-view sampling qualification and a bounded four-method GPU pilot
in the [comparison study](../reconstruction-study/README.md). Those short fits
reach their time budgets without stationarity; they are development evidence,
not the final reconstruction comparison. Neither study validates a clinical
scanner response, recovered patient composition or resolution at the native CT scale.
Acquired spectral projections with their own calibration are a separate required
validation study. Ordinary CT volumes and vendor VMI reconstructions cannot
substitute for those measurements.
