# Registration and a useful next view

This worked example starts with a known CT-derived attenuation volume, generates
calibrated simulated radiographs, recovers the volume's pose, chooses an additional
view and checks the resulting recovery independently. It uses the canonical CUDA
projector, primary-count objective, pose optimiser and acquisition-design entrypoint.
The driver supplies data and connects those operations; it contains no replacement
physics or optimiser.

The input is CT-ORG case 2, with the previously recorded water/bone assignment at
80 keV, cropped and averaged into a 64 × 128 × 128 field. These are assigned
attenuation coefficients, not calibrated measurements of patient composition.
CT-ORG is distributed under CC BY 3.0; preserve its source attribution and the
input manifest. The source NIfTI hashes, affine, crop, averaging, material
coefficients and xraylib metadata are retained in `known.json`.

## Run the complete example

Set up the maintained [GPU environment](../../python/dpt/README.md). The repository
includes the admitted 4 MB input, with its source and preparation record:

```sh
PYTHONPATH=python .venv/bin/python experiments/worked-applications/run.py \
  --known public/generated/worked-examples/inputs/registration \
  --output /path/outside/repository/new-worked-example
```

The output directory must be new. The script verifies the input array hash,
checks quadrature against an independently implemented CPU integration rule,
generates new observations, performs all 25 fits and writes the independent
assessment. Its final exit succeeds only when the declared final acceptance
checks pass. Inspect `summary.json` as well as the process status.

To reproduce the input preparation from the original CT-ORG `volume-2.nii.gz`
and `labels-2.nii.gz`, use the existing preparation driver and its recorded
xraylib package:

```sh
PYTHONPATH=python .venv/bin/python experiments/application-study/prepare.py \
  --source /path/to/CT-ORG-files --case 2 \
  --protocol experiments/application-study/protocol-v1.json \
  --xraylib-package /path/to/recorded-xraylib-package \
  --output /path/outside/repository/new-prepared-volume
```

The [preparation recipe](../application-study/README.md) specifies the source,
conversion and geometry assumptions. An alternative crop or attenuation
conversion is a different input and is intentionally rejected by the worked
example's configuration.

## What the reader should see

First, two orthogonal radiographs constrain a shared six-parameter pose. The
initial transform is the identity; the simulated generating transform is kept
out of the supplied fitting case. The optimiser must reach the unchanged
scaled-gradient threshold of 0.001. Probe errors and two reserved views then
test the accepted transform independently of the fitted images.

Next, each of eight prescribed noise replicates starts from one base view.
The accepted base fit supplies its pose and likelihood Fisher information once.
The selector ranks 5°, 30°, 60° and 90° candidate views at equal photon budgets.
Only after its decision is recorded does the driver draw that candidate's noisy
image. The selected pair and the near-parallel 5° pair are refitted with the same
solver. Shared observations use the same random stream; different acquisitions
have separately keyed streams.

The baseline is explicitly a nearly repeated view. This demonstrates the value
of observing a poorly constrained direction. It does not claim that selection
beats a well-chosen fixed orthogonal view or that every noise replicate improves.
The same eight ±40-mm object-frame targets define both predicted variance and
measured error.

## Acceptance and retained evidence

The fitting projector retains FP64 depths and integrates each trilinear cell
with two Gauss nodes. This removes the gradient jumps caused by fixed midpoint
samples crossing interpolation planes. Observations are independently generated
with 4,096 midpoint samples per ray in FP64; both operators are checked against
an independent exact CPU integral before any Poisson observations are drawn.
The full-image discrepancy is also required to be small relative to Poisson noise.

`config.json` fixes the physical model, generating transform, starts, seeds,
quadrature, solver limits, targets and acceptance criteria before observations
are generated. Every prescribed outcome is retained. Success requires all main
solves to reach stationarity, satisfy the geometric gates and pass the reserved
mean-projection check. The selected policy must also improve the mean squared
task error over the near-parallel baseline under the declared paired comparison.
The eight-pair t interval is descriptive for this teaching anatomy; it does not
establish a population or clinical effect.

The records include:

- `sampling.json`: independent ray and full-image quadrature checks for all angles.
- `fits/*/`: exact cases, initial and fitted predictions, accepted trajectories,
  rejected trials, solver termination and source snapshots.
- `decision-*/`: supplied design inputs, ranked candidates and saved decision.
- `evaluation.json`: geometric errors on the exact task points and reserved-view
  mean errors, evaluated after the fits.
- `summary.json`: the final acceptance checks and paired result.
- `run.json` and `child-records.json`: source, input and output hashes and actual
  execution metadata. Interrupted or changed-source runs cannot claim completion.

`--development` uses a separate prescribed seed and one pair. It is useful for
diagnosing the implementation, but its report can never be a passing final
result. Historical study outcomes remain unchanged in their original records.
