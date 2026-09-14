# Complete transport loss-gradient checks

The target is half-squared error of the expected photon signal from declared analytic absorption layers. This driver compares the complete independent-batch gradient with an independent analytic reference and records a common-random-number finite-difference step sweep. Mean and derivative factors use disjoint original histories; the finite-difference product factors are also independent. The results are mathematical validation evidence, not a recovered physical specimen.

From the repository root:

```sh
PYTHONPATH=python .venv/bin/python experiments/transport-gradients/run.py --output /path/outside/repository/new-gradient-run
```

The output directory must be new and outside the authoring/export tree. Source/configuration snapshots and actual-device metadata accompany `gradients.json`, which records scalar replicate uncertainty, finite-difference steps, actual replay work and unique history count. An unresolved zero error bar or failed seven-standard-error analytic check marks the run failed. Inspect the full step-size sweep as well; a statistical band is not a general proof of differentiability.

The 9 September 2026 GB10 acceptance and integration runs completed this analytic gradient protocol. Their independent replicate statistics and finite-difference sweeps remain private numerical records; no plots or public artefacts are generated.
