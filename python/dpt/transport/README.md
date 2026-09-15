# Fixed-grid photon transport

This package supplies CUDA analogue and continuous-absorption history estimators with expected-score derivatives for the parameter sets described below. Public library tests cover numerical contracts and independent analytic references. Supplied interaction coefficients and acquisition models require their own physical justification.

## Physical model and representations

`TransportSpec` combines an axis-aligned, piecewise-constant material grid, supplied partial interaction coefficients, a source energy and a planar detector above the grid. Material-grid storage shape is `(nz, ny, nx)`, x fastest, matching the deterministic volume convention. `origin_mm` denotes the **lower cell face**, whereas deterministic sampled volumes use sample centres. This is a different field model, not an implicit conversion from trilinear attenuation or CT values. `shape_xyz` supplies geometric axis extents to kernels. Detector shape is `(width, height)` and flat pixel index is `row * width + column`; its xy origin also denotes lower faces.

The supplied material-major coefficient arrays have energy fastest, binary64 values in inverse millimetres, and an explicit provenance identifier. Absorption plus scattering defines total extinction. Linear interpolation between declared energy nodes is part of the model. There is no coefficient extrapolation or invented material database. An escaped history contains both primary and scattered contributions; do not add a deterministic primary image again.

Two conditional scattering laws are available:

- `isotropic-elastic` preserves photon energy. This is an explicitly idealised scattering law, useful for controlled models and transport validation.
- `free-electron-compton` samples unpolarised Klein-Nishina scattering from stationary free electrons and applies Compton energy conservation. Coefficients must describe that supplied macroscopic model. Bound-electron, coherent, polarisation, fluorescence and pair-production corrections are absent. Its energy table must cover every evaluated history energy. Leaving that support invalidates the estimate; no low-energy history is silently absorbed.

The detector records either photon-count weight or incident energy in keV times that weight. The latter is an ideal energy score, not a calibrated electronics output or a Poisson count observation. The final escaped straight ray intersects the detector at most once because the grid is convex and the detector lies above it. Pixel support is half-open. Source points lie below the detector; source directions are binary64 unit vectors and source importance weights are supplied explicitly.

## Execution and failure ownership

`prepare_transport` binds caller-owned int32 material IDs and binary64 coefficient arrays, uploads the small declared energy-node vector and prepares four bytes of diagnostic state. Model arrays remain immutable for the workspace lifetime. Source arrays, density scales and output destinations are caller-owned; dynamic arrays remain unchanged until execution completes. All arrays must be contiguous and reside on the owning CUDA device and stream. Upload or cross-stream handoff happens explicitly before a call.

`trace_histories` writes pixel, score, final energy, event count and terminal status per original history. `out_pixel=-1` denotes a miss or absorption. Every launched original history belongs in the denominator, including zeros. Checked calls scan dynamic inputs and synchronise at declared boundaries. `validate=False` promises valid current inputs and permits asynchronous execution; `check_status()` is mandatory before accepting any estimate. Warm kernel specialisations before graph capture. Status persists across unchecked launches until the caller explicitly discards a failed run and clears it.

One thread owns one complete history. A sampled optical depth persists across material faces; crossing a face changes the extinction rate, not the random free-flight sample. Tied faces are crossed together, and exact face coordinates avoid arbitrary epsilon shifts. No history-by-event tape, ray-by-step array, roulette or branching allocation is created. Event, crossing and angular-rejection limits are resource guards: reaching any limit invalidates the entire estimate. Arithmetic failures raise the shared `NumericalError`; invalid physical inputs and energy-support failures raise `TransportError`, a compatibility subtype of `ContractError`. Resource guards remain `IncompleteHistoryError`. A contract failure cannot become a numerical trial rejection merely because arithmetic also produced a nonfinite value.

Energy scores combine weight, amplitude and energy through mantissa/exponent factorisation, so a compensating small amplitude can preserve a finite score even when weight times energy alone would overflow. The engine uses binary64 geometric arithmetic, coefficient interpolation, path weights and likelihood scores. This retains distinguishable cell faces and the declared score range; the tally optimisation below preserves that precision contract. Forward history state is thread-local; output storage costs 28 bytes per history. Derivative replay adds only its caller-owned pixel/status/derivative outputs and recomputes trajectories for one selected material at a time. Prepared moment scratch costs 20 bytes per detector pixel, excluding caller outputs. Exceptional histories update a status atomic. Every live tally lane first reaches a converged ballot that records whether it has a valid contribution; misses participate in that ballot with a false predicate. Moment accumulation groups the selected warp lanes with the same detector address, reduces their normalised FP64 scores through a shuffle tree, and lets the group leader update the detector. Scale maxima and centred deviations use the same grouping. Sparse masks, detector misses and the final partial warp retain their original-history meaning. This reduces contention without a replicated detector histogram, extra device scratch or a change of precision. CUDA targets below SM 70 retain individual atomics. Grouping adds some work when contributions are already dispersed; performance depends on the detector occupancy, not just the history count.

## Random identity and replay

The shared `dpt.kernels.random.random4` implements Philox4x32-10. A key is the 64-bit seed; a counter includes 64-bit original-history identity, event and domain. Domain zero collision draws use lanes for optical depth, interaction type, polar cosine and azimuth. Compton rejection uses domains with the high bit set, one per trial. Source sampling uses event `0x80000000` and domain zero, disjoint from collision events. Detector-observation channels use their separately reserved namespace. Neither voxel crossings nor launch chunk boundaries change random identity.

Each open-interval uniform contains 32 random bits. That finite resolution limits extreme-tail probability experiments and must be included in their numerical analysis. Replay promises the same integer counter schedule and, on the same device/build with unchanged inputs, the same paths. It does not promise bitwise reproducible floating-point atomic sums across scheduling, architectures or compilers. `HistoryBatch.source_namespace` records independently sampled source inputs; disjoint transport counters alone cannot make a reused random source draw independent.

## Derivatives and inverse estimators

`derivative_histories` supports a selected log material-density scale and a fixed-distribution source amplitude. Under analogue sampling, the log-density score for material `m` is its collision count minus the total integrated extinction accumulated in that material. Each collision contributes one because density scales both partial coefficients together; their branching probabilities and conditional scattering laws do not depend on that scale. Every survival segment contributes, including final escape. This is a likelihood-ratio expected-score derivative under the declared probability law, subject to the recorded floating-point/RNG approximation. Replaying a trace through an ordinary tape would not supply those sampling-density terms.

The source-amplitude derivative returns the base score directly, including at zero amplitude. The positive log-amplitude parameter returns amplitude times the base score before reduction, retaining the chain factor without dividing by amplitude. Material-density derivatives multiply the original source weight, amplitude, energy factor and likelihood score directly through mantissa/exponent factorisation; they never reconstruct a derivative from an underflowed forward score. The capability table rejects moving geometry, hard detector-edge motion, active angular/energy laws and changing source distributions. These require additional density/pathwise/boundary terms. A failed or truncated history invalidates derivatives just as it invalidates the forward estimate.

`history_mean` and `history_moments` share validation, ownership checks, scratch initialisation and the scaled mean launch sequence. `history_moments` first establishes a per-pixel magnitude scale, then uses a centred pass, including zero-score misses, to estimate marginal mean and variance of that mean at the independent original-history level. Its sampling-law precondition is IID source sampling; a repeated deterministic ray also qualifies. The caller must establish that law because score arrays alone cannot verify it. There is no redundant Boolean acknowledgement argument. Deterministically stratified sources need independent complete-batch replicates instead. Detector pixels are correlated, so summing marginal variances is not a valid scalar loss error bar. Any future split descendants must be summed within their original history before computing moments.

For a fixed observation, `independent_squared_gradient` combines an unbiased mean estimate from batch A with an unbiased derivative estimate from independent batch B. Their product estimates the gradient of squared error of the **expected** measurement. Reusing a batch adds a covariance term and is rejected. `independent_squared_loss` uses the product of two independent residual means; individual unbiased estimates can be negative. Candidate and incumbent may share random numbers within each factor, while factors remain independent. `summarise_replicates` supplies scalar mean/standard-error checkpoints from independent complete replicates; it does not claim a distribution-free confidence interval.

## Concrete inverse composition

`prepare_transport_inverse` builds `TransportSquaredOracle`, implementing the shared `IndependentSquaredOracle` protocol. Select `log-material-density` and optionally `log-source-amplitude` parameters. The chart uses absolute logarithms of positive density/amplitude; fixed materials retain the supplied base scales. A `ParallelBeam` specifies a fixed pencil or uniformly sampled rectangle. Its source RNG uses the original history identity, separately from flight and scattering draws. No user callback is needed for this supported acquisition.

Preparation owns reusable source/history images, detector means/components, a reduction workspace and a pinned small parameter buffer. Each new chart value uploads only that vector. Gradient replicates reuse one independent mean image while replaying the derivative batch per selected parameter. Change replicates reuse the two independent factors between incumbent and candidate. The objective reduction stays on CUDA; four-byte status words and scalar losses/gradient components cross to the controller at explicit checkpoints. Scalar results use prepared pinned host staging, and exception exits drain the owning stream before staging can be reused. Immutable observations and weights are scanned during preparation. Actual replayed histories and parameter/scalar transfer totals are recorded separately from the controller's unique-history budget.

Compose these operations through [`prepare_transport_inverse`](inverse.py) and [`recover_expected_signal`](../stochastic_recovery.py). For independent checks, use the [transport reference functions](../validation/transport.py) and the [library test suite](../../../tests/python/). The optional [`RunRecorder`](../experiments.py) preserves the supplied configuration and source identity when recording a solve.

## Validation

Authored tests cover geometry/energy domains, 64-bit identity ranges, source-stream independence, layer survival, density-score signs, original-history moments, chunk/replay identity, zero extinction, explicit budget failure, zero-amplitude derivatives, independent squared-product algebra and Compton angular/energy laws. The angular reference uses independent quadrature and the integrated Klein-Nishina cross section; it is not a histogram generated by another invocation of the sampler.

Run `PYTHONPATH=python uv run --group gpu pytest --run-gpu tests/python` from the source-checkout root. The suite includes sparse detector addresses, signed derivatives, misses and partial warps with independent mean and centred-variance checks. Extend those checks for new grid, source or detector regimes. Energy-table convergence and the validity of supplied physical coefficients remain separate questions from numerical implementation.

## Source support

Warp ballot membership follows [NVIDIA's warp-level primitive guidance](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/): determine the participating mask at a point reached by every live lane, then use that mask for matching and the peer masks for shuffles. Empty ballots and the final partial warp contribute no invented histories.

Counter-based identity follows [Salmon et al., _Parallel Random Numbers: As Easy as 1, 2, 3_](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf). Conditional Compton scattering follows [Klein and Nishina (1929)](https://doi.org/10.1007/BF01366453); the electron rest-energy constant uses the [NIST CODATA value](https://physics.nist.gov/cuu/Constants/Table/allascii.txt). These sources support algorithms/constants; they do not provide the user's material coefficients. Warp API behaviour is pinned to the repository's Warp 1.17 dependency.

[Chapter 9](https://photontransport.com/chapters/scattering-stochastic-transport/) derives the forward estimator; [Chapter 10](https://photontransport.com/chapters/differentiating-transport/) derives its likelihood contributions, replay and independent inverse-problem products. Their source listings use the implementation files in this package.

Compare detector-tally strategies at matched history counts and detector occupancy. Grouping contributions can reduce contention for concentrated tallies while adding overhead when destinations are dispersed. Report warmed operator timings separately from complete transport throughput.

## Continuous absorption and bounded inverse models

`TransportSpec(estimator="continuous-absorption")` samples scattering free flights
at `rho * mu_s` and integrates `rho * mu_a` along every realised segment, including
escape. Coefficient arrays remain density-independent. Forward and derivative
scores combine the logarithmic absorption weight with original factor exponents;
`exp(-tau)` is never an independently rounded factor that can erase a rescued
finite output. No roulette, splitting or weight cutoff is added. Guard exhaustion
still invalidates the complete estimate. The analogue default is unchanged.

For active log density, the weighted derivative is the complete detector score
multiplied by the scattering count minus integrated total extinction in that
material. Its absorption term differentiates the explicit weight; its scattering
terms differentiate the path likelihood. This requires positive densities, fixed
geometry/source/angular laws and coefficient zeros, and sufficient integrability.
Moving boundaries and active angular/source laws remain unsupported.

`prepare_transport_inverse(..., local_model=True)` prepares a dense Jacobian and
fixed FP64 Gram reduction tree for at most 16 parameters. Its default additional
memory budget is 256 MiB (`local_model_max_bytes`); preparation checks both the
budget and signed-32-bit indexing. `model_replicate` retains the independent
residual/derivative gradient product and returns `J_hat.T @ W @ J_hat` as a PSD
proposal metric. It is generally a biased estimate of expected-signal curvature.
Images remain resident; the returned matrix has only parameter-squared entries.
The oracle's byte counters count parameter/scalar payloads, excluding the existing
status-word transfers. `histories_traced` counts successfully submitted ray lanes,
including replay and lanes in a subsequently invalidated batch; it does not imply
every guarded history reached physical termination.

`StochasticPolicy(proposal="quadratic")` enables the bounded Euclidean log-chart
solve, model reuse at unchanged incumbents and independently sampled acceptance.
Damping does not change reported pre-damping rank. The final-validation sample size is fixed and its budget
is reserved before another model consumes that budget; its history identities are
assigned when the final checkpoint executes. Failed final checks return
unresolved. Replicate-SE bands remain heuristic, including their vector use and
repeated consultation; no confidence-sequence guarantee is claimed.

The read-only `deterministic_sampling` property is established at preparation only
for continuous weighting, a fixed pencil source and globally zero immutable
scattering coefficients. Mutable input arrays must honour the same lifetime
contract as the other prepared static data. Random source positions remain
stochastic even in pure absorption. `numerical_gradient_allowance` is an explicit
arithmetic allowance, not a substitute for sampling uncertainty.

Check density recovery against analytic attenuation in the deterministic absorption limit, then assess scattering cases with independent histories. Strong attenuation can make the signal weakly sensitive to density, so inspect parameter error separately from the absolute gradient tolerance.

Ordinary sampling and trust-radius exits use the reserved final checkpoint; invalid histories and numerical failures still raise. `final_validation_trigger` records why the checkpoint was reached. The final gradient check uses independent empirical uncertainty, whose interpretation follows the assumptions described above.
