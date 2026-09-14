# CT-derived registration and finite-view selection

This study supplies the missing anatomical execution evidence for
`dpt.examples.registration` and `dpt.examples.acquisition_design`. It reuses the
canonical primary projector, pose derivatives, objectives and safeguarded
L-BFGS. The development pilot and independent derivative/precision checks have
run. `protocol-v2.json` is the independently accepted final protocol; final
observations and comparisons are executed only after the source/role checks
pass. `protocol-v1.json` retains the earlier proposal unchanged. Nothing here
commands imaging hardware.

## Data and physical interpretation

Use original TCIA CT-ORG CT/label pairs, DOI
[10.7937/tcia.2019.tt7f4v7o](https://doi.org/10.7937/tcia.2019.tt7f4v7o), CC BY 3.0,
with retained source hashes and attribution. Case 2 is development because it
has already been inspected extensively. Cases 0 and 1 supply new final
observation sets and two additional morphologies. Their source scans have
5-mm axial spacing; this experiment must not imply finer source anatomy or
clinical population generalisation from two scans. Prior source-plane QA is
recorded; no claim that the source CTs themselves were never inspected.

Prepare new inputs from the original NIfTIs. Do not repurpose earlier arrays
whose recorded role is reconstruction evaluation only. Take the inferior
320-mm source slab after verifying header orientation; reverse X losslessly
from the verified LAS header. Keep full native X/Y coverage. Native Z crops
are 0:320 for case 2 and 0:64 for cases 0/1. Conservatively average native cells
by XYZ factors (4,4,5) and (4,4,1), respectively, into a 64×128×128 known field.
The object frame is centred on the crop and retains its physical spacing.
Finite crop faces and header-only laterality remain explicit.

Use the existing spectral-anatomy preparation rule: largest 6-connected
`HU > -500` component, axial hole filling, water fraction
`clip(1+HU/1000,0,1)` inside that body, and water/bone fractions .35/.65 on
body-intersected label 5. Record the retained body, excluded components and
label hashes; inspect the _development_ preparation for errors. This is an
assigned phantom, not HU calibration or measured patient composition.
At 80 keV, xraylib 4.3.0 `CS_Total_CP` gives water μ=.018365660412653837
mm⁻¹ and cortical-bone μ=.04108010415565857 mm⁻¹. The exact built-in compounds
are “Water, Liquid” at 1.0 g/cm³ and “Bone, Cortical (ICRP)” at 1.85 g/cm³;
the latter differs from the ICRU-44 composition used in the earlier spectral
study. Each run records all element mass fractions, densities, package and
source hashes, and confirms the compound result against its elemental sum.
The preparation uses the retained BSD-licensed xraylib source/package and
its compound metadata; it does not redistribute separately fetched NIST
attenuation tables. See `XRAYLIB-NOTICE.txt` for the retained compound-header
notice. Multiply the assigned fractions by these total attenuation
coefficients and sum. Do not use mass energy-absorption coefficients. No polychromatic, scatter, blur, acquired-count or absorbed-dose
claim is made.

The full prepared CT-derived attenuation field is legitimately known to the
registration and design algorithms. The hidden generating pose, future
candidate counts, reserved projection counts and evaluation landmarks are
separate inputs. Known anatomy is not the unknown field in this application.

## Numerical and observation contract

All image formation uses canonical CUDA operators. Final observation means use
4,096 midpoint samples through the known trilinear field. This is the same known
continuous field with distinct quadrature, not an independent physical
acquisition. For every generating angle of each final field, independently
integrate the 24 fixed detector rays with the host piecewise-cubic oracle and
compare the complete images at 512/1,024/4,096 samples. The fixed checks include
edge and near-empty rays. At 20,000 incident photons/pixel, require error
≤0.05 Poisson standard deviations at P95 and ≤0.2 maximum. Generation at 4,096
must pass the independent check. Select only the first passing fitting count
from the prespecified 512/1,024 list, record it before counts or fits, and use it
for every fit/design comparison on that field. Failure of these declared
choices blocks the protocol; final outcomes cannot choose new settings.

Generate integer Poisson counts with separate recorded NumPy PCG64 streams for
study, case, pose, replicate, view and role. This models independent primary
counts. FP32 files must exactly represent their nonnegative integers, be below
2²⁴ and include an explicit binary mask. Scalar open beam is constant within
each view. Any excluded count is recorded as zero for the supplied driver,
with the original full count array retained separately.

Use a 128×128 detector with 8-mm pitch, source-isocentre 800 mm and
source-detector 1,200 mm. The large field covers the finite phantom; this is
synthetic geometry, not an asserted C-arm model. Verify endpoints and projected
box coverage for every declared pose/view before generation. All poses are
object-to-world, with a fixed right SE(3) chart and physical scales
(1,1,1 mm,.01,.01,.01 rad). Fitting never recentres a chart mid-solve.

## Registration: recovery and weak information

The main comparison uses orthogonal 0°/90° views versus near-parallel 0°/5°
views, at equal total incident expectation (20,000 photons/pixel summed over
views). Each prescribed initial perturbation starts a fresh solver. Four
fixed perturbations and two independent noise replicates per final morphology
give 16 paired comparisons. These are repeated numerical experiments on two
known phantoms, not 32 independent patients. Initial transforms and all starts
are recorded before final generation; a poor start is never replaced after
viewing its geometric error.

A separate deliberately restricted observation uses one 0° view, only the
central eight columns, and 200 photons/pixel. Its narrow aperture/low count
assumptions are labelled; it is not dose matched to the main comparison.
Report all four starts for one frozen replicate per morphology. The experiment may expose
weak information, stagnation or a wrong basin. If it does not, report that
outcome; do not relabel a successful solve as an ambiguity demonstration or
manufacture a failure. The final report records whether this prescribed stress case actually exposes
weak information; it cannot change the case after opening a final reference.

Record target, initial and accepted final predictions, signed residuals,
optimisation trajectory, termination, scaled gradient, objective and physical
translation/rotation/probe errors. The reference is the hidden simulated
transform; its exactness is within the declared model, not patient truth.
Geometric success is RMS probe error ≤1 mm, maximum ≤2 mm, and rotation error
≤0.5°. Numerical stationarity is reported separately. The prescribed solver
budget is 150 accepted iterations/600 evaluations, with a 120-second soft
per-solve limit; development timing must justify this before launch.

## Acquisition: choose, observe, refit, evaluate

1. Fit the base 0° observation from the prescribed nominal initial pose.
2. At its accepted estimate, recompute base-view Fisher information in the
   same local right chart as candidate information. Supply this _once_ as the
   driver's `prior_precision`. It represents already acquired data, not an
   invented Gaussian prior. Require positive definiteness and condition ≤10¹⁰;
   otherwise retain a rejected design result. Do not add jitter or reuse the
   fixed original-chart Fisher at a different pose.
3. Rank the six equal-cost alternatives −90°, −60°, −30°, 30°, 60°, 90° using
   1,000 expected open-beam photons/pixel each. Base exposure is also 1,000.
   Known task points are the eight ±40-mm cube corners. They are distinct from
   reserved geometric evaluation probes.
4. Freeze the selected-view record before reading any future candidate count
   array. Compare it with the fixed +90° policy and a uniform random policy
   whose candidate IDs were drawn independently before generation. If two
   policies choose the same view, reuse that view's count array and report the
   resulting paired identity; do not pretend the observations are independent.
5. Refit base plus newly obtained candidate counts from the same accepted
   base pose, with a fresh chart/curvature history and identical settings for
   all policies. Current base observations enter the likelihood only once.
6. Only after all policy outputs are fixed, evaluate physical probe error and
   log-likelihood/deviance on complete reserved 45°/135° views with independent
   noise. Final selection, stopping and tuning cannot read these views.

Run eight independent noise replicates per final morphology at a fixed hidden
pose. Report every paired selected-versus-fixed and selected-versus-random
error, median/mean differences and within-phantom bootstrap intervals over the
eight replicate IDs. Do not pool detector pixels or the two phantoms as
independent clinical samples, claim calibrated coverage from Fisher, or
require a selected-policy win as an acceptance gate.

A constrained candidate set −15°/15°/30° repeats four replicas per morphology
with fixed +15° and an independently drawn uniform random choice. This is a
supplied angular-availability constraint, not verified mechanical clearance.
The study reports the achieved downstream consequence, including no benefit
or failure. One separate development budget check supplies an over-budget
alternative and verifies that it cannot be selected.

Reserved geometric probes are a fixed 5³ lattice over [−60,60] mm, selected
before final generation. Report their rigid-transform errors as mathematical
object probes, not manually verified clinical landmarks. Registration/design
cannot use those arrays. Complete 45°/135° evaluation projections are distinct
from final candidate angles and are never used for tuning or selection.

## Independent checks and resource gate

Add scoped tests for the supplied drivers: heterogeneous multiview Poisson
loss/chart directional derivative at a nonzero pose; shared-chart additivity;
mask-before-composition preprocessing; accepted-state final predictions;
input/hash/reference separation; independently computed target Jacobians;
FP64 Fisher versus explicit small CPU outer-product sums and an independent
finite-difference image Jacobian; zero illumination, rank/condition rejection,
budget exclusion and deterministic ties. Test sum and chart scaling directly,
not just the underlying single-view least-squares fixture.

Before final studies, time four warmed paired one/two-view gradient evaluations
and all six candidate Fisher evaluations on the development volume. Record source identities and a representative CUDA timeline for actual
allocation/launch/synchronisation/transfer evidence. The completed 120-second
development budget used 16.99 seconds: four fits took 1.50–3.01 seconds each
and six-candidate ranking, including its cold process, took 2.72 seconds.
This supports a provisional 7–12-minute estimate for all 136 final fits and
24 rankings; actual elapsed time and any soft-limit overshoot are retained.
Shared free-memory observations are not peak or exclusive allocation claims.

`prepare.py` prepares known inputs. `generate.py` and `pilot.py` retain the
development execution. `diagnose_precision.py` reproduces every development
trial and compares local arithmetic and an independent actual-field derivative.
`generate_final.py` creates the final fitting and reserved observations in
separate manifests. `batch.py` writes prescribed/random choices before count
access, records every trial and freezes all compared fits. `evaluate_final.py`
requires both final batches and all their output/source hashes before opening
reserved references. `profile_case.py` captures two warmed training-input
evaluations without fitting. `run.py` uses the supplied canonical optimiser;
there is no parallel GPU physics or solver path.

The ordinary-JSON immutable-vector boundaries in the supplied registration and
reconstruction drivers, and fresh-process Warp initialisation in reconstruction
and acquisition design, were repaired and checked with actual cold CLI runs.
Scientific defaults and previous execution records remain unchanged.

Development geometry was accurate: orthogonal-pair RMS probe error 0.01473 mm;
selected-view downstream RMS 0.01179 mm versus fixed-view 0.02262 mm. All four
fits nonetheless reported `line_search_failed` above the unchanged 0.001
scaled-gradient gate. Exact replays and the directional precision sweep support
a rounding limit in the final searches. The independent 48-ray actual-field
CPU derivative agreed within 3.52×10⁻⁶ relative error; that bounded check does
not certify the whole-image stationarity criterion. The final study preserves
this distinction between geometric recovery and numerical convergence.

All numerical outputs are new private directories. Preserve original source
bytes, development failures, final policy failures and every producer/config
hash. The final gate accepts reproducible honest comparisons; a successful
execution flag alone establishes neither registration success nor a better
acquisition policy.

## Registration arithmetic replay

`replay_registration.py` preserves the supplied case, observation hashes and previous optimisation record in a new `RunRecorder` directory. Use `--case`, `--previous` and `--output` to name those paths. `--precision float32 --integration midpoint` requests the earlier stored-depth/count arithmetic; `--precision float64` retains primary depth, prediction and cotangent precision. `--integration cell_gauss` integrates the same trilinear field between its interpolation planes. The diagnostic always records the chosen declarations; a changed integration is a new objective evaluation, not an exact historical replay.

`--stationarity-only` disables relative-loss and step stagnation exits while retaining the case's gradient threshold, iteration/evaluation budgets and line-search policy. `--independent` checks a fixed 24 rays per view using the independent CPU reference for the declared integration. This diagnostic's fixed ray set requires the historical 128×128 detector configuration; it is not a general reference selector. Old records remain untouched.

The separate prospective `experiments/worked-applications/` protocol records its integration, precision and completion policy before its final examples. Its successful executions must be assessed on their own records rather than changing the classification of these older sampled-objective results.
