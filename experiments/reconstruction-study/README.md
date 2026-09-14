# Reconstruction comparison study

This directory contains executed reconstruction development studies and a
prepared primary 64³ illustration. The primary illustration and final comparison
matrix have not yet run. Raw observations, fields, source snapshots and profiler
output must be written to new external folders.

The scalar comparison uses the same primary monochromatic count arrays,
known open beam, geometry, initial water box, nonnegative attenuation constraint
and physical quadratic penalty. One method minimises the Poisson half-deviance.
The other minimises post-log penalised weighted least squares:
`0.5 * sum(k * (A*mu - log(b/k))**2) + R(mu)` on strictly positive observed counts.
Zero counts have a fixed exclusion mask and are reported; no pseudocount is
inserted. Observations above the open-beam expectation give negative optical
depth data and are retained. The weights are fixed from observed counts and
represent a first-order variance approximation, not exact Gaussian data.

The spectral experiment uses separately generated polychromatic, disjoint-channel
Poisson observations and jointly reconstructs two free material fractions. A
fixed physical two-material metric is compared with Euclidean projection using
identical data and objective. That comparison is an optimiser ablation; it is
not an independent spectral physical model. Equal incident photon number does
not make the scalar and spectral experiments equivalent observations or doses.

All image formation and explicit adjoints use the canonical CUDA operators.
The scalar WLS adapter replaces only the objective composition and reuses the
canonical scalar projected solve and regulariser. Accepted callbacks record
actual immutable checkpoints; rejected trials cannot become result volumes.
Known container support is permitted. CT values, internal labels, reference
fields and reserved projections are absent from the fitting bundles and are not
loaded by the fitting API or stopping rule. This is an explicit data-access
contract, not filesystem-enforced secrecy.

The development configuration, independent checks and measured budget must be
reviewed before a final study is generated. Historical NIST material runs and
acquired calibration diagnostics retain their original physical identities.

`probe_acquisition.py` measures three native anatomical views before larger
acquisition. `acquire.py` requires that completed probe, then records all 144
development views at both native sample counts (1,024/2,048) and inverse-grid
sample counts (512/1,024). It retains every mean image and independent 24-ray
oracle used by the sampling gates. All gates precede count generation. Separate
PCG64 namespaces identify scalar and spectral physical models, views and channels.

`pilot.py` reads only the resulting count arrays, fixed physical inputs and
declared 32³ support. It performs the four prescribed fits with limits of ten
accepted updates and 60 seconds per method. The wall limit is soft: it is checked
after recording an accepted callback, so a long iteration can overrun it. Setup,
profiling, checkpoint output and explicit finalisation times are recorded
separately. A dedicated budget exception retains the actual accepted state;
other exceptions leave a failed run. All numerical trial calls are recorded.
Time or update exhaustion does not establish convergence.

The scalar driver uses one fixed object pose. Its varying source/detector
geometries are the original rays transformed by the inverse object pose; the
spectral driver uses the equivalent fixed detector with varying object poses.
This host coordinate change does not introduce another projector. Both spectral
methods report the same Euclidean stationarity diagnostic. The new fixed metric
is a determinant-normalised physical Fisher template, distinct from historical
trace-normalised templates; its condition number describes the Fisher/metric,
not the unsquared sensitivity matrix.

The separately configured actual-resolution stage uses `resolution-config-v1.json`:
four dense 48³ methods, one dense 64³ spectral-metric fit, and one 48³
spectral-metric fit for each sparse and limited-angle regime. Each has a maximum
of 20 accepted updates and a 120-second soft fit cap. The seven caps sum to
840 seconds; the stage reservation is 1,800 seconds including preparation and
measurement. This stage completed all seven outcomes with 123 accepted updates
and 686.30 seconds of total solve time. All 5,184 sampling gates and the field,
history and independent final-prediction checks passed. Every outcome remained
nonstationary; these short fits establish measured development behaviour, not
converged material recovery.

`resolution_data.py` verifies the original acquisition and the explicit reviewed
source amendment, qualifies the two new inverse grids, then prepares four
standalone fitting folders. All 5,184 sampling summaries precede new count draws.
Dense counts are reused exactly. Sparse and limited observations use separately
keyed development streams and fourfold integrated incident weights; responses
and material coefficients remain fixed. The source and destination folders must
be external paths supplied on the command line.

`resolution_run.py` reads only those fitting folders. The scalar trial step is
`3.125e-10 mm^-2`, with the independently fixed `1e-8 mm^-2` mapping diagnostic.
The optional canonical mapping setting defaults to the historical initial step
when omitted. Both spectral methods retain the same Euclidean diagnostic and
the original physical regularisation. All seven outcomes remain in the record,
including failures and runs prevented by the stage budget.

`resolution_evaluate.py` freezes every prescribed outcome before opening assigned
references or generating means. It uses the explicit-array `evaluate_fields.py`
helper, verifies final-field predictions with independent CPU trilinear ray
integrals, and reports training-view errors and source identities. This stage
does not generate or evaluate reserved views. `profile_resolution.py` captures
one completed scalar or spectral gradient/value pair through the canonical
operators, with separate ordinary timings and visible-buffer accounting.

The historical resolution-stage commands accept `--config`, `--freeze` and new external output paths
where applicable. The source freeze is a JSON object containing `files`, a
mapping from repository-relative paths to SHA-256 values, covering every
canonical Python source. The recorded host boot and monotonic clock bind
preparation and fitting to one soft stage reservation. Reusing input data in a
new study requires its own recorded configuration and source review; historical
snapshots are preserved.

## Primary 64³ illustration: prepared, not yet executed

`illustration.py` provides separate `acquire`, `fit` and `evaluate` commands for
one longer joint water/bone fraction reconstruction. It reuses the tested
canonical material solver and the existing accepted-checkpoint recorder. The
source-bound `primary64-config-v1.json` fixes the input identities and scientific
settings; changing it defines a new study requiring its own review.

The original CT-ORG case-2 CT and label produce a 192³ vertebral crop inside a
known water box. Its bone-label assignment is a simulated composition, not
measured patient material truth. Exact 3³ cell means define the 64³ evaluation
reference in the same 148.875 × 148.875 × 192 mm support. Both unknown fields start
from uniform water 1/bone 0 and remain independently free within the nonnegative
voxel simplex. The physical model is the separately prepared open
`xraylib-icrp-spekpy120-al25-v1` model; its actual FP32 array bytes are pinned.
Historical NIST-based results retain their original identities.

There are 144 fitting and 24 whole withheld views on three tilted orbits, a
96² detector with 4 mm pitch and 200,000 integrated incident photons per ray
across three disjoint channels. The illustration uses 504 separately keyed
Poisson streams, distinct from prior development streams. This seed is reserved
for the illustration and must be excluded from any future final-comparison
namespace. It is not an additional acquired dataset or a dose comparison.

Acquisition first records every native 1,024/2,048 and inverse 512/1,024 sampling
comparison and independent 24-ray oracle for all 168 views. All 4,032 strict
sampling summaries must pass before any count draw; failed qualification remains
recorded and does not silently increase the sample count. The reused checker
also qualifies scalar 80 keV projections, but this illustration generates no
scalar observations and performs no scalar fit. Fitting and withheld count
bundles are separate. The fitting command checks only its count/physical inputs
and hash-bound qualification JSON, without opening reference or generating-mean
arrays. The evaluator verifies and freezes completed fit outputs before opening
references or whole withheld views.

The fixed determinant-one material metric, physical quadratic penalty
`beta = 0.1 mm^-1`, initial trial step `1e-6`, and independent Euclidean mapping
step `1e-6` match the measured development contract. Both original stationarity
criteria remain required: relative mapping `1e-4` and mapped fraction
displacement `1e-5`. The maximum is 1,000 accepted updates or a 7,200-second soft
solve limit checked after accepted callbacks. Checkpoint/history costs are
charged to that solve time. Setup, paired gradient/value measurements and
explicit finalisation are reported separately. The 7,560-second total is a
planning reservation, not a hard deadline or a convergence prediction. An
interruption is preserved; there is no implicit restart or refunded work.

Accepted fields are saved at 0, 1, 5, 10, 20, 50, every 100 updates and the final
state. Evaluation reports material/attenuation errors, fixed ROI statistics,
component volumes, the fixed 0.325 bone threshold, physical profiles, domain and
history checks, stationarity, and fitting/withheld projection errors with an
independent final-field ray check. All stop reasons remain visible. The
inverse-reference residual describes that assigned coarse field; it is not a
proven attainable minimum.

Run the complete study from a freshly checked public source export. Set
`DPT_INPUT_PYTHON` to the pinned input-preparation environment described in the
[open-input instructions](../spectral-reconstruction/README.md#open-input-preparation),
`DPT_GPU_PYTHON` to a compatible Python 3.12 CUDA environment, and
`DPT_CT_ORG_DOWNLOAD` to the original licensed CT-ORG cache. Set
`DPT_PRIMARY_INPUTS` and `DPT_PRIMARY_OUTPUT` to new locations outside the export.
Run all commands from that export's root with its `python` directory first on
`PYTHONPATH`; the illustration rejects loaded DPT/study modules from a sibling
source tree. The source export and independently reproduced physical/input bytes
must pass their review before acquisition or fitting begins.

```bash
"$DPT_INPUT_PYTHON" experiments/spectral-reconstruction/provision_inputs.py \
  --case 2 --cache "$DPT_CT_ORG_DOWNLOAD" \
  --output "$DPT_PRIMARY_INPUTS/CT-ORG-case2"

"$DPT_INPUT_PYTHON" experiments/spectral-reconstruction/prepare_vertebrae.py \
  --source "$DPT_PRIMARY_INPUTS/CT-ORG-case2" \
  --output "$DPT_PRIMARY_INPUTS/anatomy"

PYTHONPATH=python "$DPT_INPUT_PYTHON" \
  experiments/spectral-reconstruction/prepare_open_physics.py \
  --output "$DPT_PRIMARY_INPUTS/physics"

PYTHONPATH=python "$DPT_GPU_PYTHON" \
  experiments/reconstruction-study/illustration.py acquire \
  --anatomy "$DPT_PRIMARY_INPUTS/anatomy" --physics "$DPT_PRIMARY_INPUTS/physics" \
  --config experiments/reconstruction-study/primary64-config-v1.json \
  --output "$DPT_PRIMARY_OUTPUT/acquisition" --device cuda:0

PYTHONPATH=python "$DPT_GPU_PYTHON" \
  experiments/reconstruction-study/illustration.py fit \
  --observations "$DPT_PRIMARY_OUTPUT/acquisition/fitting" \
  --output "$DPT_PRIMARY_OUTPUT/fit" --device cuda:0

# Only after the completed fit and its output/source identities are verified:
PYTHONPATH=python "$DPT_GPU_PYTHON" \
  experiments/reconstruction-study/illustration.py evaluate \
  --acquisition "$DPT_PRIMARY_OUTPUT/acquisition" --fit "$DPT_PRIMARY_OUTPUT/fit" \
  --output "$DPT_PRIMARY_OUTPUT/evaluation" --device cuda:0
```

Ordinary repeated-evaluation timings are workload-specific. The retained
resolution-stage traces have collection-completeness warnings: launch, copy and
synchronisation counts describe observed records, and no recorded allocation
event does not prove exhaustive absence of allocation. Visible DPT buffer spans
and shared-device snapshots are reported separately; they are not a measured
process peak. The longer illustration remains distinct from the unexecuted
40-fit final comparison matrix.
