# Acquisition mismatch with held-out observations

This driver compares a pose fit with fixed calibration against the same fit with declared shared nuisance parameters. Both use `SpectralPoseEvaluator` and the canonical optimiser. Supplied held-out views are evaluated only after fitting and never enter the fitting objective, gradients or line search. It creates numerical records, without figures or synthetic substitute observations.

The 9 September 2026 GB10 acceptance pass executed the driver with an analytic vacuum/calibration fixture. Its shared gain fit removed the held-out deterministic calibration residual; vacuum makes pose unidentifiable. This checks ingestion, fitting/held-out separation and composition, not recovery from physical acquisitions.

## Inputs and invocation

Use the NPZ/JSON input contract described in [the spectral experiment](../spectral-projection/README.md). At least one fitting calibration group must declare an active scale or offset. Held-out group names must correspond to fitting groups so their estimated calibration can be carried forward without fitting held-out measurements. Both models start from the same pose chart and calibration references.

```bash
PYTHONPATH=python .venv/bin/python experiments/acquisition-mismatch/run.py \
  --inputs /path/to/supplied-data.npz \
  --model /path/to/physical-model.json \
  --fit-views acquisition-1 acquisition-2 \
  --heldout-views independent-acquisition-3 \
  --output /path/to/new-private-comparison
```

Names must be distinct, exist in the model and form disjoint fitting/held-out sets. Reusing the same observation buffer across the two sets is rejected. The caller remains responsible for actual acquisition independence, patient-level separation where required, and provenance; different member names do not establish statistical independence.

The output must be a new directory outside the repository. Model metadata and executable sources are snapshotted there before input preparation.

Each model records its accepted pose, calibration, termination reason, optimisation trajectory, held-out loss and individual held-out view losses. Explicit chart priors apply to the joint fit only when requested. The run manifest records source snapshots, configuration, device identity and the supplied archive's digest. Input changes during execution fail the run. Records remain private until separately reviewed.

## What a comparison establishes

A held-out residual change establishes performance under the chosen observation model. It does not establish true pose recovery, remove spectrum/scatter mismatch, or demonstrate that a nuisance parameter is physically calibrated. Independent pose/reference measurements are needed to measure bias; crossed controlled acquisitions and repeated observations are needed to assess confounding and uncertainty. This driver accepts those observations but does not manufacture them.

The recorded acceptance pass covers operator/VJP and shared-group gradient checks. Extend pose/nuisance finite-difference sweeps and independently specified recovery/mismatch studies to each supplied acquisition regime. Record numerical range, conditioning, retained GPU memory, actual launch structure and profiling separately. The chapter listings expose the parameter chart, view aggregation and held-out boundary from their canonical package code.
