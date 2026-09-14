# Analytic book rollouts

This driver records the exact mathematical examples described by figure placeholders in Chapters 4–10. It produces JSON arrays and source/configuration hashes, with no plots, browser rendering, photon histories or GPU execution. The reference functions live in `python/dpt/validation/book_illustrations.py`; the driver only selects inputs and records their outputs.

Figure 4.1 uses the worked grid coordinates and spacings from section 4.2 through the canonical `GridSpec` conversion. Figure 4.4 uses the stated 100 mm quadratic field and includes its prescribed sample counts. Other parameters are explicitly illustrative choices in `config.json`, not anatomical dimensions, calibrated detector data or measured material coefficients.

The outputs cover:

- 4.1–4.4: physical-to-grid coordinates, three boundary extensions, finite slab intervals and quadratic midpoint error.
- 5.2–5.4: a fixed-node staircase, binary64 transmission differences with a 100-digit reference, and the exact sphere chord near tangency.
- 6.1: infinite-domain Gaussian feature overlap, not empirical registration capture range.
- 7.4: remaining information over the full derivative-cosine range, including nearly parallel directions.
- 8.3: stipulated shared-charge and exclusive-assignment event laws with equal means and different exact compound-Poisson covariances.
- 10.2 and 10.4: the swept uniform aperture interval and exact paired-estimator standard error. These do not imply that the production transport API supports moving-boundary derivatives.

An absent sphere derivative at tangency is labelled `inward-divergence`. Slab intervals unconstrained by a parallel coordinate and undefined staircase derivatives also carry explicit states. JSON never contains Infinity or NaN. Smooth numerical comparisons and singular endpoints remain separate records.

Run from the repository root:

```sh
PYTHONPATH=python .venv/bin/python experiments/book-analytic-rollouts/run.py --output /absolute/private/new-analytic-run
```

The destination must be new and outside the repository. `run.json` records completion only after verifying every declared source, configuration snapshot and output hash. The dependency versions it records are installed package metadata; they do not indicate GPU execution. Tests compare the quadratic sum with exact rational arithmetic, aperture uncertainty with a Bernoulli law, covariance with explicit event moments and difficult chord/difference values with independent high-precision calculations.
