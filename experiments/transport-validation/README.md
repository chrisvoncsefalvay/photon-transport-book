# Layered transport validation

This driver executes `dpt.transport` on explicitly mathematical 1 mm pure-absorption layers and records actual CUDA means and original-history uncertainty. The independent Decimal survival reference supplies the expected value and Bernoulli variance. It does not supply material measurements or anatomy.

Run from the repository root with Python 3.12 and the pinned GPU dependencies:

```sh
PYTHONPATH=python .venv/bin/python experiments/transport-validation/run.py --output /path/outside/repository/new-validation-run
```

The destination must be new and outside the authoring/export tree. `run.json` records source snapshots, hashes, configuration, installed versions, actual device and completion state; `validation.json` records all independent batch means and the declared seven-standard-error acceptance check. No figures or public artefacts are generated. An incomplete history, failed check or changed source marks the run failed.

The 9 September 2026 GB10 acceptance and integration passes executed this protocol. The integration sweep records four optical depths, three history counts and sixteen independent replicates per case. The absorption check does not establish Compton accuracy or performance; those require their own recorded sampler, domain and execution evidence.
