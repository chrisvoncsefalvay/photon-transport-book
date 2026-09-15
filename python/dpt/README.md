# Scientific library

`dpt` contains the deterministic and bounded stochastic operators for Chapters 2–10, recovery compositions, independent references and supplied-data application drivers. The public source includes the library and its tests. Study orchestration and raw execution records are maintained separately in the private authoring repository.

`dpt.material_reconstruction.MaterialReconstruction` fits supplied spectral measurements with explicit adjoints and material constraints. Its default uses photon counts and volume fractions; signal-WLS and nonnegative equivalent-basis modes are described [below](#joint-material-volume-reconstruction). [Chapter 12](https://photontransport.com/chapters/reconstruction/) develops the reconstruction loop and shows recorded results from a CT-derived assigned-material example.

## Public entry points

| Responsibility                         | Modules and prepared interfaces                                                                                                                    |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Frames and fields                      | `geometry.RigidTransform`, `DetectorGeometry`, `volumes.GridSpec`, right-composed `compose_pose`                                                   |
| Primary projection                     | `projection.prepare_projection`, `project_optical_depth`, `projection_vjp`                                                                         |
| Derivative lifetime                    | `autodiff.FirstOrderPass`, explicit seeds, manual VJPs and outstanding-pass ownership                                                              |
| Deterministic objectives               | `objectives.prepare_objective`, `evaluate_objective`, `evaluate_primary_objective`; squared error or Poisson half-deviance in a declared domain    |
| Pose recovery                          | `recovery.PrimaryPoseEvaluator`, `recover_pose`; `registration.recover_parameters` supplies the shared safeguarded L-BFGS driver                   |
| Spectral model                         | `materials.MaterialTable` / provenance, `spectral.Spectrum`, `material_projection`, `spectral_signal` and `spectral_vjp`                           |
| Detector and acquisition               | `detector` calibration, finite spatial response/transpose, separately identified count/compound-Poisson/read-noise observations                    |
| Multiple views and nuisance parameters | `spectral_recovery.SpectralPoseEvaluator`, named shared calibration groups, fixed pose charts and `recover_spectral_pose`                          |
| Joint material volumes                 | `material_reconstruction.MaterialReconstruction`, `MaterialReconstructionView`, `MaterialReconstructionSignalView`                                 |
| Stochastic model                       | `transport` analogue histories, fixed-grid material density derivatives, original-history moments, replay and independent expected-signal products |
| Stochastic inverse problem             | `transport.inverse.prepare_transport_inverse` with `stochastic_recovery.recover_expected_signal`                                                   |
| Independent reports                    | `validation` field/gradient references, geometric recovery metrics, bounded scaled-sensitivity SVD and replicate uncertainty                       |
| Reproducible runs                      | `experiments.RunRecorder`: source/configuration snapshots, actual environment metadata, output hashes and explicit failed/incomplete states        |

Host configuration imports remain CPU-safe. CUDA wrappers load Warp only when used; the optional `gpu` environment supplies it. Bounded SVD diagnostics additionally load that environment's NumPy on demand. There is no CPU substitute for the production scientific operators and no automatic array upload, precision conversion or image download at an operator boundary.

## Representation and composition

The deterministic geometry uses right-handed coordinates, column vectors and `T_WO` from object to world. The fixed optimisation chart applies a local right SE(3) increment ordered translation then rotation. Distances are millimetres, angles radians. Detector origin is its first pixel centre and shape is `(height, width)`. Scalar/material fields use `(nz, ny, nx)`, x fastest, sample-centre origin and explicit anisotropic spacing/orientation. The sampled field extends constantly from the last sample centre to its half-cell support face, then becomes zero.

The projector integrates that exact declared discrete field along finite source-to-detector segments. Its manual first-order reverse pass includes active entry/exit faces and recomputes samples, retaining no ray-by-sample tensor. Pose reductions use FP64; volume-gradient scatter uses FP32 atomics and therefore does not promise bitwise reproducibility. Material-fraction projection reuses the same device functions and validates fractions without renormalisation. Fraction sums are accumulated in FP64 and permit an excess of at most $2^{-24}$, the binary32 unit-roundoff allowance for independently rounded nonnegative partitions. Individual values must still lie in `[0, 1]`.

`ProjectionSpec.integration="midpoint"` retains the fixed-count midpoint rule. Explicit `integration="cell_gauss"` traverses interpolation planes, including the constant half-cell edge slabs, and applies two-node Gauss integration to each cubic ray segment. This removes midpoint sampling error for the declared trilinear field, up to arithmetic error; it does not change voxel resolution or promise differentiability at support-topology changes. `samples_per_ray` remains a positive configuration field but is unused in cell-Gauss mode. Its pose VJP integrates spatial gradients and the parameter-weighted gradients and retains the external support endpoint terms. Internal moving-cell terms cancel because the field is continuous across interpolation planes. Forward, volume VJP, tape and per-ray sensitivities use the same declared integration mode.

`ProjectionSpec.precision` selects depth and depth-cotangent storage (`"float32"` by default, or `"float64"`); field storage remains FP32 and pose storage/reductions remain FP64. `PrimaryPoseProblem` defaults to FP64 intermediates, fusing primary transmission and the objective through `evaluate_primary_objective`. For Poisson counts its depth cotangent is `N - lambda`, with the usual weighting and reduction. Near agreement the shared cancellation-safe half-deviance is retained; extreme optical depths combine the beam and transmission exponents before storing a representable count. Positive observations with an unrepresentable zero count mean still fail. Explicit primary `precision="float32"` retains the earlier composed storage path for replay.

For bounded offline figures, `projection.projection_pose_sensitivities` exports the existing per-ray pose calculation into caller-owned FP64 storage of shape `(6*pixels,)`, pixel first and translation then rotation. Each row is multiplied by a depth cotangent matching `ProjectionSpec.precision` (FP32 by default, optionally FP64): use ones for optical-depth sensitivities or `transmission_vjp` cotangents for count sensitivities. The diagnostic performs one ray traversal per pixel and skips reduction; ordinary VJPs retain their existing reduction and storage. It allocates no device buffers, supports only first-order local right pose derivatives and rejects ambient tapes or outstanding projection passes. Checked calls synchronise for numerical status, and unchecked calls require a later status check.

Workspaces bind one CUDA stream and allocate persistent scratch before repeated evaluation. Caller buffers and fixed inputs retain their declared lifetime. Checked low-level calls validate device values; unchecked calls require valid current inputs and explicit numerical-status checkpoints. A successful check never certifies subsequently mutated data. Warm required specialisations before any future capture or timing. Manual VJPs reject ambient tapes unless the interface explicitly accepts a tape; higher derivatives are unsupported.

Recovery adapters retain detector-sized buffers on CUDA. Only pose/nuisance control values, scalar losses, small gradients and status words cross the host boundary. Their fixed input arrays stay immutable throughout a solve. Pinning alone does not settle lifetime: exception paths drain the owning stream before control staging can be reused. The solver returns the last accepted state; scratch predictions may still describe a rejected trial and must be recomputed explicitly before export.

The shared optimiser uses a fixed chart, finite trial checks and limited curvature memory. Its line search seeks strong Wolfe conditions; at the configured maximum step it also accepts strict decrease satisfying Armijo. The first descent direction is scaled, and a failed search can retry conservative steepest descent even before curvature pairs exist. Stagnation and iteration budgets are separate from stationarity. Independent geometric metrics remain necessary even when loss decreases. Calibration groups fix one of gain/exposure and permit an explicitly shared positive scale and affine offset; arbitrary fitted per-pixel correction fields are excluded. The spectral adapter accepts multiple views with shared pose and named acquisition groups, and keeps the supplied material basis, spectrum and response fixed.

## Joint material-volume reconstruction

`MaterialReconstruction` composes the canonical material projector, spectral model
and objective across supplied fitting views. The defaults remain
`observation_model="poisson"` and `field_domain="fractions"` with
`MaterialReconstructionView`: supplied integer-valued FP32 counts and nonnegative
volume fractions whose voxel sums are at most one.

For fixed-weight signal least squares, set `observation_model="signal_wls"` and
supply `MaterialReconstructionSignalView(geometry, pose, observations, weights,
response, objective_weights, valid)`. Arrays must be contiguous native NumPy
arrays. `observations` and `objective_weights` are FP32 `(C, H, W)`; `valid` is
uint8 with that same shape and values zero or one. Objective weights are finite,
nonnegative, deterministic scale factors; they imply statistical precision only
when separately justified. The data objective sums
`0.5 * objective_weights * (prediction - observation)**2` over valid lanes and
all fitting views. It does not assert an independent likelihood for correlated
channels or infer original photon counts from averaged signals.

Masked objective lanes branch before reading observations or evaluating residuals,
and contribute exactly zero loss and cotangent. Their observations may contain
NaN or infinity; active observations must be finite. Zero objective weight alone
does not replace an explicit validity mask. The upstream spectral model must still
satisfy its numerical contract. Construction makes fixed GPU copies of supplied
observations, masks and objective weights; changing their host arrays afterwards
does not change the prepared fit.

Spectral FP32 `weights` have shape `(C, E)` with `SpectralSpec.shared_weights=True`, or
`(C, E, H, W)` with `shared_weights=False` in signal-WLS mode. The shared FP32
`response` remains `(C, E)`, with values in `[0, 1]` and `shared_response=True`.
Weights represent integrated energy-bin contributions in the explicitly declared
output unit. A calibrated effective signal kernel can use a unit response and
include spatial open-beam intensity in its weights exactly once; this does not
identify source spectrum and detector response separately. Reusing the same
weight/response array objects across views deduplicates their GPU uploads.

`SpectralSpec.precision="float64"` makes the material solver retain FP64 paths,
spectral predictions, objective seeds and path cotangents through its composed
forward/reverse passes. Fields, voxel gradients, observations and calibration
remain FP32. The default `"float32"` mode retains the earlier intermediate
storage. Near agreement, rounding a path or predicted count before computing
the loss can obscure a decreasing step even when the derivative is accurate;
FP64 loss reduction alone cannot recover those discarded bits. The material
projector and Fisher preparation follow the same selected precision.

Set `field_domain="nonnegative"` for dimensionless equivalent-basis coefficients
without a sum cap. Their normalisation belongs in the supplied `(M, E)` attenuation
coefficients in mm⁻¹; integrated basis paths are in mm. These fields are not
automatically volume fractions, mass densities or concentrations. The legacy names
`initial_fractions`, `fractions` and `fractions_numpy()` refer to fields in the
declared domain. The low-level `prepare_material_projection` accepts the same
explicit `field_domain` choice.

`evaluate(gradient=True)` computes the fitting objective and resident gradient;
`evaluate(gradient=False, trial=True)` preserves the accepted gradient. The solver
uses projected Armijo steps, with an optional fixed SPD two-material metric and
the exact projection for the declared domain. Stationarity uses the Euclidean
projected mapping. Physical-spacing quadratic regularisation is applied once per
objective, with `regularisation_mm_inverse` in mm⁻¹; changing view count changes
the balance with the summed data objective unless its weight is adjusted.
`solve()` reports the actual termination, accepted steps and numerical rejections.
Refresh accepted-state predictions before calling `predictions_numpy()`; field
and prediction exports are explicit synchronised host transfers. An iteration
budget or a small training loss does not establish reconstruction accuracy.

`MaterialReconstructionSettings(step_selection="bb", acceleration="inertial")`
optionally uses metric Barzilai–Borwein step proposals and safeguarded momentum.
The secant proposal is `(s.T @ H @ s) / (s.T @ y)` for consecutive accepted
field and Euclidean-gradient differences; invalid curvature falls back to the
ordinary geometric proposal. Inertia extrapolates from the preceding accepted
field before the same constrained projection. A non-descent, nonfinite or
Armijo-rejected inertial proposal restarts without momentum at the same step.
Acceptance always uses the true gradient and actual rounded displacement;
the Euclidean stopping diagnostic and its fixed threshold are unchanged.
Neither option implies a convergence-rate guarantee for spectral reconstruction.
The defaults remain `step_selection="geometric", acceleration="none"`.
For N material coefficients, the optional secant workspace adds 16N+16 CUDA
bytes and a reusable 20-byte pinned host control buffer; inertia adds one
resident 4N-byte preceding field. No field or gradient vector is copied to the
host to select these proposals. Explicit checkpoints remain caller-controlled.

`material_preconditioning.prepare_fisher_scaling(solver)` prepares an optional
fixed positive voxel scale for a two-material Poisson model with a supplied SPD
material metric. Canonical spectral derivatives and projection adjoints form a
local expected-Fisher row bound, with a regulariser row bound and a declared
floor for weak or unobserved voxels. The result is normalised and stays on CUDA;
`solver.set_voxel_metric_scale(result.scale)` makes an owned copy for the next
solve. Each update uses the block metric `D_i H`; the BB numerator includes the
same D. This changes the proposal, never the objective or Euclidean stopping
test. Expected Fisher curvature is not the complete observed nonlinear Hessian,
and flooring a weak voxel does not supply missing measurement information.
Prepare the scale between solves, from the supplied accepted field and fixed
calibration. The helper reads neither observed counts nor reference composition,
and reports its floor, normalisation and curvature range explicitly.

## Physical scope and provenance

The worked-example bundles contain attributed CT-derived assigned fields and the recorded fixed spectral inputs needed to reproduce their simulated acquisitions. They do not supply measured patient composition or scanner calibration. Their manifests retain source identity, units, preparation, checksums and redistribution rights. Other acquisitions must supply those same records. Analytic fixtures name their defining formula and their limited validation purpose. Total primary attenuation is distinct from a transport model's partial interaction coefficients.

The stochastic engine deliberately uses a different field model: axis-aligned piecewise-constant cells with lower-face origin. Its detector currently uses an axis-aligned plane and explicit pixel faces. See [transport/README.md](transport/README.md) for physical omissions, layout, counter namespaces, per-history failure codes, means/variance, score-function derivatives and the bounded Compton model. Moving-boundary, active angular-law and general source-distribution derivatives are rejected until their missing estimator terms are derived and independently accepted. Original analogue histories are the uncertainty unit; replay is not another observation.

## Commands and acceptance

The [application examples](examples/README.md) accompany Chapters 11–13: rigid registration, fixed-geometry attenuation reconstruction and selection of a further registration view. They accept supplied arrays with recorded provenance; the guide describes their case format and executable commands. The scalar attenuation example is distinct from the joint material solver documented above.

[`dpt.experiments.RunRecorder`](experiments.py) records source/configuration snapshots and output hashes for these applications. It writes schema-version-2 `run.json` files with `running`, `complete` or `failed` status and a `finished_utc` timestamp when execution ends. Completion records execution and source integrity; evaluate numerical and physical accuracy separately. Output directories must be new and outside the source checkout.

Run the CPU suite with `PYTHONPATH=python .venv/bin/python -m pytest tests/python`; add `--run-gpu` to require real CUDA, including the independent operator and composed-gradient references. Ruff and strict host Pyright check the Python boundary; Warp's executable annotations are excluded at the DSL-module boundary in `pyproject.toml`; strict checks still cover the host wrappers and tests. CPU CI is never CUDA evidence.

When profiling the library, prepare workspaces and warm required specialisations before measuring repeated operators. Report state traffic and synchronisation alongside elapsed time. CUDA event intervals can include host submission gaps, and concentrated detector tallies can behave differently from dispersed ones.

The transmission API below is used by the composed primary operator.

## CUDA transmission

`dpt.transmission` maps caller-supplied optical depths to deterministic primary transmission, expected counts, log transmission and removed-primary fractions. It also supplies an inverse-decrement helper and explicit first-order vector–Jacobian products. It does not trace rays, sample photon counts or infer attenuation coefficients.

Install Python 3.12+ with the pinned `gpu` dependency group. Importing the public module is CPU-safe; preparing a workspace requires CUDA. There is one production implementation, in `kernels/transmission.py`. `validation/transmission.py` is a separate high-precision test oracle and is never used by the operator.

### Storage and execution

Create `TransmissionSpec` with `beam="none"`, `"scalar"`, `"device-scalar"` or `"per-pixel"`. The last two accept CUDA arrays; a fixed Python scalar is explicitly rounded to binary32. An active illumination parameter must use array storage. `prepare_transmission(spec, max_pixels=..., device="cuda:0", stream=...)` allocates persistent diagnostics and any scalar-reduction scratch. Preparation does not allocate image outputs or compile every specialisation.

All arrays are contiguous one-dimensional `warp.float32` on the same CUDA device. Flatten detector images in the caller's documented pixel order. Inputs, seeds and outputs have identical lengths, except a device-scalar beam or its gradient, which has length one. Empty images are supported. Input/output overlap, overlapping destinations, strided layouts and capacity overruns raise `TransmissionError`.

Call `transmit` with independently selected `out_T`, `out_counts`, `out_log_T` and `out_removed` destinations. At least one output is required; counts require explicit illumination. Each non-empty evaluation uses one specialised pointwise kernel. The caller owns all destinations and their lifetime. Warm each required specialisation before capturing a CUDA graph or measuring it.

Checked calls scan the current device inputs and synchronise before writing. They reject nonfinite/negative depths or illumination, and inverse decrements outside `[0, 1)`. `validate=False` is an explicit promise that the **current** device inputs meet these contracts; it avoids those scans and their host synchronisation. Host shape, alias and stream checks remain. An earlier successful scan does not certify an array that has subsequently changed.

A workspace has one owning stream. Use that stream explicitly or make it current with `warp.ScopedStream`; backward must run with it current. Event/wait handoffs are the caller's responsibility. Separate concurrent executions need separate workspaces. `scratch_bytes` reports persistent device diagnostics and reduction storage, excluding caller-owned buffers.

### Numerical policy

Storage is binary32. The ordinary exponential path remains binary32 up to optical depth 64; the tail exponential, count scaling, VJP products and sums use binary64 registers. Every final destination is rounded once. No full-image binary64 temporary is allocated. Counts and derivatives are formed before a possibly underflowing transmission is stored. The log output is exact unary negation, including its zero sign. Removed fractions and their inverse use native `expm1f` and `log1pf`; physical zero outputs are positive zero. Fast maths is disabled; fused multiply-add is enabled explicitly.

True output underflow is allowed; clipping and gradient floors are absent. A nonfinite final gradient sets a device status flag. Checked VJPs raise `GradientRangeError`; unchecked VJPs and tape callbacks require `workspace.check_status()` before accepting gradients. After an error, discard the reverse pass, zero the tape and explicitly call `workspace.clear_status()` before starting an unchecked independent pass. Error reporting identifies an invalid input index only on the exceptional path.

### Differentiation

`transmission_vjp` accepts any combination of `seed_T`, `seed_counts`, `seed_log_T` and `seed_removed`; missing seeds mean zero. Requested `out_grad_L` and `out_grad_n0` must correspond to active inputs. Standalone calls overwrite gradients and preserve seeds. A scalar beam uses a fixed 256-lane binary64 reduction tree with `O(P/256)` scratch. No per-pixel global scalar atomic is used.

For Warp tape composition, allocate participating arrays with `requires_grad=True` and pass `tape=` explicitly. The adapter accumulates input cotangents and consumes ordinary output cotangents after all local consumers finish; `retain_grad=True` preserves an output cotangent. `tape.zero()` separates independent reverse passes. Incoming external seed arrays remain caller-owned and are copied into the already allocated output gradients by Warp. `FirstOrderPass.backward` checks seed shape, dtype, CUDA device and contiguity before that copy.

Original inputs must stay unchanged until backward completes. Recomputing from those inputs preserves derivatives that would vanish if reconstructed from rounded outputs. Debug `warp.config.verify_autograd_array_access` registers read dependencies, including fixed inputs and view parents, and supports Warp's recorded-write diagnostics. It is not comprehensive mutation protection: raw writes and Warp's `fill_`/`zero_` can bypass that diagnostic. The immutability precondition still applies.

Only first-order derivatives are supported. Recording a VJP on an ambient tape raises instead of silently producing a zero higher derivative. Omitting the explicit tape while recording a forward operator also raises. The ambient-tape check is a small compatibility boundary pinned to Warp 1.17's internal runtime accessor; the derivative itself uses its public custom-callback API.

### Reproduction

Run `PYTHONPATH=python uv run --group gpu pytest --run-gpu tests/python` from the source-checkout root. Omitting `--run-gpu` keeps GPU modules out of CPU collection; requesting it requires real CUDA and does not silently skip an unavailable device. The tests exercise [the transmission implementation](transmission.py) against [independent mathematical references](validation/transmission.py). The book renders excerpts from the same library source.
