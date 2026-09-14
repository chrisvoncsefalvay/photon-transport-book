# CT-derived material reconstruction worked example

This executable example recovers two material-fraction fields from spectral
Poisson counts with the canonical CUDA material projector, spectral model,
explicit adjoints and constrained solver. CT-ORG case 2 supplies acquired
anatomical shape; its assigned water/bone fractions are simulated composition,
not measurements of patient materials.

The supplied 192³ assignment was conservatively averaged to 16³. This coarse
trilinear field generates the observations and defines the unknowns. The
matched representation makes the example a test of instructional correctness;
it does not establish recovery of unmodelled anatomical detail, scatter, a
real scanner or clinical material accuracy. Independent exact CPU ray
integration and composed objective/adjoint checks are still required.

## Run the complete example

From the repository root, use the maintained Python 3.12 environment and a
CUDA device. The small admitted anatomy and physical-input bundle is included;
this command downloads no anatomy or physics data.

```bash
uv sync --locked --group experiment
PYTHONPATH=python .venv/bin/dpt-check-gpu
PYTHONPATH=python .venv/bin/python experiments/worked-reconstruction/run.py \
  --output ../dpt-runs/worked-reconstruction-v3
```

Choose a new output directory outside the repository. The one invocation
qualifies the acquisition, reproduces both prescribed noise replicates and
verifies their four original count-array hashes before fitting either. It
then fits both and evaluates both frozen results. It exits unsuccessfully if either
final reconstruction fails scientific acceptance. A completed `run.json` alone
is not success: inspect `summary.json`, whose `accepted` field must be true.

The inputs under
`public/generated/worked-examples/inputs/reconstruction/` include the coarse
assigned field, its original CT/label/crop provenance, the exact physical
arrays, preparation sources and licence notices. The protocol pins both the
coarse array and metadata hashes alongside the original source identities.
The original large CT and label arrays are not required to execute this recipe.
The independent evaluator opens the reference only after both fits have
finished. Fitting receives observations, fixed calibration and grid metadata;
it starts from uniform water and never reads the supplied reference fractions.

## What the protocol fixes

`protocol-v3.json` fixes 48 fitting views on three tilted full-azimuth orbits,
12 disjoint withheld views, a 48² detector covering the complete box, and two
independent noise replicates. It preserves the anatomy, geometry, noise roots,
physical penalty and acceptance thresholds declared in `protocol-v1.json`.
Development uses noise phase 1; the final two replicates use phase 2.

Version 3 repairs intermediate precision after version 2's first replicate
failed its line search at a Euclidean mapping of 77.962, above the unchanged
threshold of 10; the second replicate passed at 5.530. An independent CPU
calculation and exact CUDA replay identified the cause: a trial decreased the
FP64 objective by 0.00441149 but increased the objective computed from stored
FP32 paths and means by 0.000218335. The original outcomes remain retained.
Fitting now keeps material paths, spectral means and their cotangents in FP64;
fields, voxel gradients, calibration and observations remain FP32. Acquisition
generation stays explicitly FP32, and version 3 pins the exact original count
arrays. Neither a new noise draw nor a looser stopping rule repairs that defect.

Before drawing any counts, all expected pixels are checked at 512 versus
1,024 ray samples. Fixed rays in every view are also checked against exact
CPU integration of the declared trilinear field. Fitting uses 512 samples;
observation generation and final prediction assessment use 1,024. Independent
material-path derivatives and complete-objective finite differences are
checked before optimisation.

The objective is the summed Poisson half-deviance plus a physical quadratic
fraction-gradient penalty with coefficient 0.1 mm⁻¹. Per-voxel nonnegativity
and the water-plus-bone sum cap are enforced by exact two-material metric
projection. The worked protocol explicitly selects metric Barzilai–Borwein
step proposals and safeguarded inertia. True-gradient Armijo decrease still
accepts every update. The library's ordinary geometric, non-inertial defaults
are unchanged.

A spatial Fisher scale is prepared from the uniform field and refreshed once
after 100 accepted updates. This uses expected signal sensitivities at the
current fitted field and fixed calibration, together with a regulariser bound
and declared positive floor. BB and inertial histories restart at that
boundary. The reference volume and withheld views enter neither preparation.
The scale is a proposal metric, not a claim that the observed nonlinear
Hessian equals the Fisher matrix.

Both final criteria must pass at the original fixed Euclidean diagnostic step
1e−6: mapping relative to the uniform-start value ≤1e−4, and mapped fraction
displacement ≤1e−5. The combined absolute mapping threshold is 10 for this
protocol. Neither the material metric nor its spatial scaling changes the
stopping diagnostic. Each fit has a maximum of 6,000 accepted updates and a
soft 1,200-second solve budget, shared across its two stages. Exhausting a
budget is an unfinished reconstruction, even if its images look reasonable.

After fitting, an independent CPU simplex projection reproduces the final
mapping from the frozen field and gradient. Each material's whole-box RMSE
must be ≤0.05. Prediction of the generating means over all 12 withheld views
must have RMS error ≤1 generating Poisson standard deviation. This is an
ideal-model prediction diagnostic; it is not a patient-dose or clinical
performance claim. All prescribed replicates are retained.

## Inspect the records

`acquisition/` records source identities, qualification, the four-count identity
receipt and separate fitting and evaluation arrays. Each fit checks that
hash-bound receipt and its own fitting-count hash without opening withheld
values. `fit-rep0/` and `fit-rep1/` retain accepted fields,
refreshed predictions, derivatives, the original stopping gate, every accepted
update, fixed checkpoints and both metric preparations. `stages/` preserves
the individual solver returns; the top-level history has global update numbers
and one shared time budget. `evaluation-rep*/evaluation.json` records the
independent checks. `summary.json` binds all five child execution records.

Run from a fixed source revision. Each `RunRecorder` stores the executed
scientific source and input hashes; concurrent edits during execution would
make that provenance ambiguous. The driver also exposes `acquire`, `fit` and
`evaluate` commands for diagnosis, but the complete invocation above enforces
both fits before either reference evaluation.

To render accepted results into a new staging directory:

```bash
PYTHONPATH=python .venv/bin/python experiments/worked-reconstruction/curate.py \
  --source ../dpt-runs/worked-reconstruction-v3 \
  --output ../dpt-runs/worked-reconstruction-v3-figures
```

Curation rejects failed or development outcomes. It exports PNG/PDF figures,
fixed-plane material arrays, numerical curves and provenance; it does not
publish them. A separate `--development` run uses the retained phase-1 noise
and shorter development limits, and can never be marked accepted as a final
study. Historical time-limited studies remain separate records.
