# Stochastic density recovery

This driver composes the concrete `TransportSquaredOracle` with the independent-batch acceptance controller. It fits the log density of one analytic absorbing layer, with all other densities and the source fixed. The target is the independently calculated expectation at the declared reference density. It is explicitly a mathematical identifiability case, not sampled acquisition data or anatomy.

From the repository root:

```sh
PYTHONPATH=python .venv/bin/python experiments/transport-recovery/run.py --output /path/outside/repository/new-recovery-run
```

`recovery.json` records accepted/rejected trial decisions, every completed gradient attempt, termination reason and subreason, target/recovered density, parameter error, unique history usage, actual repeated transport work and small host-transfer totals. Each gradient attempt includes its parameter vector, iteration, batch size, original-history range, gradient components and standard errors, their norms, and the controller decision. Acceptance attempts that remain inconclusive are not included in the step list. Stochastic uncertainty may require larger batches or end in an unresolved/budget outcome. Process completion is not automatically labelled successful recovery. The run directory must be new and outside the authoring/export tree; sources, configuration and actual device metadata are recorded by the shared run recorder.

The 9 September 2026 GB10 acceptance and integration runs completed with `sampling_unresolved`, which is retained as the stopping reason. Parameter proximity does not replace the declared stochastic acceptance criterion. The run records numerical trajectories without producing visualisations.

## Larger-batch diagnostic runs

Two additional configurations keep the original seed, eight replicates and all convergence/acceptance tolerances:

```sh
PYTHONPATH=python .venv/bin/python experiments/transport-recovery/run.py --config experiments/transport-recovery/config-larger-batch.json --output /path/outside/repository/new-larger-batch-run
PYTHONPATH=python .venv/bin/python experiments/transport-recovery/run.py --config experiments/transport-recovery/config-larger-budget.json --output /path/outside/repository/new-larger-budget-run
```

The first raises the per-batch ceiling from 262,144 to 4,194,304 histories. The second also raises the total unique-history budget from 100 million to one billion. These are separate controls: with eight independent pairs, one gradient attempt needs sixteen times its batch size in original histories. The controller reserves complete replicate sets and stops before exceeding the budget.

On the 9 September GB10 diagnostic runs, the original configuration reproduced all prior numerical fields exactly and identified `gradient_uncertainty_at_maximum_batch`. The larger ceiling alone stopped at `history_budget` after 67,108,864 unique histories; its largest completed gradient batch was 1,048,576. The larger-budget configuration reached 4,194,304, completed three additional rejected trials and stopped at `gradient_uncertainty_at_maximum_batch` after 570,425,344 unique histories. All three retained density 1.2523227162 against the reference 1.25. These runs establish the reason for stopping; they do not establish convergence.

In the final larger-budget run, the estimated log-density gradient was 4.3343e-6 with standard error 6.4308e-6. Its gradient-band upper value was 1.7196e-5, above the 1e-5 tolerance, and its relative standard error exceeded 0.5. The two-standard-error policy is an estimated uncertainty rule, not a distribution-free confidence guarantee. Increasing sampling does not force every fresh estimate to improve monotonically.

## Recovery repair matrix

`repair.py --mode original|family|ablation|reference|stochastic --output /outside/repo`
executes the production library with source-snapshotted private results. The
original configuration files above remain historical baselines. The matrix lives
in `dpt.transport.recovery_experiments`; no optimiser or physics is copied here.

- `original`: original observation, target, initial point and `1e-5` tolerance;
  continuous weighting and quadratic proposals, 256 histories per factor.
- `family`: 36 combinations of base depth (0.1, 1, 3, 6), target density
  (0.5, 1.25, 2) and initial density (0.6, 1, 1.8). The independently computed
  gradient and density error are separate outputs.
- `ablation`: two estimators by two proposal policies, with common initial batch,
  sampling ceilings and fixed final pool. The large final pool deliberately
  remains identical across stochastic and deterministic variants; use `original`
  to assess the separately declared deterministic sampling schedule.
- `reference`: 256 independent 65,536-history forward batches in a declared
  isotropic-scattering cube, with a separate seed. The seven-SE reference interval
  is heuristic. The forward implementation has separate quadrature/limit tests.
- `stochastic --reference /path/to/reference.json`: 32 predeclared independent
  log-amplitude fits to observation 0.08. The reference is used only to assess
  answers. Proposal and acceptance use the production oracle without reference
  means or target parameters. Preserve unresolved outcomes and the complete final
  gradient records; do not select seeds after seeing results.

The outputs contain original-history reservations, actual candidate steps,
replicate statistics and fixed held-out checks. Count false successful stops and
runs with a false accepted step against the independent reference interval.
An interval-straddling comparison is indeterminate. Adaptive steps are not IID
binomial trials; report uncertainty across complete independent runs. Even zero
failures among 32 runs cannot certify a failure probability below 1%.

The first 32 seeds (`--seed-base 9031001`) were retained as development after they
revealed unused final-checkpoint budget at sampling exits. The amended rule was
evaluated on 32 fresh seeds (`--seed-base 2026090901`, stride104729), with the same
physical model, sample sizes, budget and tolerances. All results remain recorded;
the independent assessment checks each seed against the prior registration.

## Scaling sampling for the log-amplitude walkthrough

`repair.py --sampling-multiplier 16` increases the scattering workspace capacity,
initial/maximum/final batch sizes, unique-history budget and independent reference
batch by sixteen. It leaves the physical model, initial amplitude, observation,
eight replicate pairs, proposal rule, acceptance rule and `1e-5` gradient tolerance
unchanged. The multiplier defaults to one, preserving the historical sampling
design, and is rejected for modes other than `reference` and `stochastic`.

At multiplier sixteen the initial per-factor batch is 65,536, maximum/final batch
is 1,048,576, and unique-history budget is 256 million. The reference uses 256
batches of 1,048,576. Its seven-standard-error interval remains heuristic. Larger
independent batches reduce sampling uncertainty; they do not guarantee a passing
fit or make an unresolved candidate acceptable.

Use a new output directory for each command:

```sh
PYTHONPATH=python .venv/bin/python experiments/transport-recovery/repair.py --mode reference --sampling-multiplier 16 --output /outside/repo/new-reference
PYTHONPATH=python .venv/bin/python experiments/transport-recovery/repair.py --mode stochastic --sampling-multiplier 16 --seed-base 2026091501 --repetitions 32 --reference /outside/repo/new-reference/scattering-reference/reference.json --output /outside/repo/new-evaluation
```

The root seed above belongs to the prospectively reserved second walkthrough
evaluation. Do not substitute a seed after seeing its result. `--repetitions`
defaults to 32; a value of one supports development on an already retained seed.
It counts complete recovery runs, not the eight independent pairs inside an
estimate. The final evaluation keeps all 32 runs and designates its first one
before execution. The stride is 104729.

New reference and recovery configurations record the complete transport grid,
detector, source, coefficients, active input, observation, effective policy and
workspace capacity. Recovery verifies the reference's completed run receipt,
file hash, sampling work, empirical statistics and matching physical/sampling
configuration before allocating GPU work. Historical references without the new
physical descriptor are accepted only at multiplier one. A reference remains an
assessment input; its mean never chooses optimisation steps.

Freeze an immutable source checkout before execution and retain the reference,
all recovery records and independent assessment. The 14 September v1 designated
seed was unresolved at the original budget. Testing the larger design on that
same retained seed is development; it is not the fresh v2 evaluation. The private
execution ledger records both outcomes and the new predeclared sequence.

For the complete Chapter 10 run, install the pinned environment with
`uv sync --group check --group gpu`, then invoke:

```sh
bash experiments/transport-recovery/worked-example.sh /absolute/path/to/a/new-directory
```

The wrapper performs CUDA preflight, generates the independent reference, then
runs all 32 declared recovery seeds. Its final standard-library-only verifier
exits nonzero if any prescribed run is incomplete, unresolved, changed or fails
the recorded independent reference check. A completed recovery process alone
does not make the wrapper succeed. `worked-example.json` freezes the complete
physical/source/material descriptor, sampling budget and controller settings,
as well as the seed sequence; internally consistent records for another model
or policy are rejected. `verify_walkthrough.py` checks receipt hashes and the
existing final-band/reference decisions, without estimating new statistics.
It can also be run on completed output without CUDA:

```sh
python3 experiments/transport-recovery/verify_walkthrough.py --input /absolute/path/to/output/recovery --reference /absolute/path/to/output/reference/scattering-reference/reference.json
```

The first prescribed v2 run takes four
accepted updates; all 32 recorded v2 runs pass both the controller and independent
reference stationarity checks. The smaller-budget v1 designated run remains
unresolved and is retained as the diagnostic motivating the sampling change.
The private execution ledger holds the frozen source, complete records and audit.

`curate_walkthrough.py` validates those completed records before producing the
small Chapter 10 data and Matplotlib SVG/PDF/PNG figure. It requires the full
recovery root, reference JSON, independent assessment and pre-execution freeze;
run `--help` for its arguments. Curated files contain mathematical inputs and
recorded outputs, without raw paths or environment inventories. Existing
provenance binds their bytes, so changes to curated files require regeneration.
