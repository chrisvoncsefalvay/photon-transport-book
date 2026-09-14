# Numerical book rollouts

This driver exports numerical data for later book figures. It calls the canonical CUDA spectral, detector, projection and transmission operators and their adjoints. It creates no SVG, PNG, chart component or browser physics implementation. All fields, rates and coefficients are explicitly mathematical fixtures; no anatomical volume, tissue table or measured acquisition is supplied.

From the repository root with the pinned Python 3.12 GPU environment:

```bash
PYTHONPATH=python .venv/bin/python experiments/book-rollouts/run.py \
  --output /absolute/private/new-book-rollout
```

The destination must be new and outside the repository. `run.json` records the exact configuration, installed versions, actual CUDA device, source snapshots and output hashes. A failed reference check or changed source leaves a failed record. `config.json` fixes the sampling protocol before execution; passing a run means its stated checks passed, not that the mathematical fixture is a physical validation study.

| Output                        | Recorded quantities and intended use                                                                                                                                                                                                     |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spectral-chapter-7.json`     | The chapter's equal-weight mixture with coefficients 0.01 and 0.03 mm⁻¹: 401 thickness samples, transmission, each transmitted component and the path derivative. Figure 7.2.                                                            |
| `spectral-chapter-8.json`     | The separate Chapter 8 mixture with coefficients 0.02 and 0.04 mm⁻¹ and the same payload structure. Figure 8.1.                                                                                                                          |
| `detector-noise.json`         | Baseline, doubled exposure and doubled gain, each with 131,072 independent counter-identified draws. Exact histograms and empirical/reference moments are retained. Figure 7.1.                                                          |
| `blur-displacement.json`      | A compact polynomial feature and a passive finite stencil. Moving the feature one pixel and shifting the stencil by its corresponding signed offset produce bitwise-equal filtered arrays. Figure 7.3.                                   |
| `projection-sensitivity.json` | Actual 24×24 optical depths/counts, the uploaded field, six-coordinate depth/count Jacobians, local SVDs, pose–gain angles and the linearised decomposition of a declared gain mismatch. Numerical support for Figures 5.1, 6.3 and 7.4. |
| `geometry.json`               | Anisotropic sample/grid coordinates, half-cell support and perspective projections of points on one source ray. Figures 4.1 and 6.2.                                                                                                     |

The two spectral tables deliberately preserve both ideal decimal coefficients and the binary32 values supplied to CUDA. Independent 60-digit Decimal calculations test the represented inputs with a relative budget of 2×10⁻⁷. Derived log signals and component fractions identify that they were calculated from stored outputs; they are not silently substituted for higher-precision quantities.

The observation check uses the exact Poisson mean and fourth central moment to assess the sample mean and unbiased sample variance. Its seven-standard-error thresholds are fixed in the driver. Histograms count actual canonical samples. Different condition identities denote independent draws; the shared seed does not mean the same random variables were reused.

The projection Jacobian is a bounded offline diagnostic assembled through basis VJPs. The production optimiser retains no such matrix and uses one contracted reverse product. Count derivatives pass through the canonical transmission and projection adjoints, including their binary32 cotangent boundary. A separate contraction checks consistency of the exported depth rows, and three image locations are checked against the independent discrete reference. This consistency check supplements the independent directional sweeps in `projection-gradients/`; it cannot replace them.

Local sensitivity uses declared parameter scales and identity precision weights. The mismatch decomposition records the least-squares cutoff, retained rank and normal-equation residual. It does not claim a calibrated noise covariance, nonlinear recovery or global identifiability. Grid round trips and source-ray projection invariance are checked before writing the geometry payload.

Use the existing `projection-convergence/`, `projection-gradients/`, `pose-recovery/` and transport experiment drivers for their larger numerical sweeps and trajectories. Conceptual drawings, unsupported moving-boundary derivatives, null-collision methods and event-queue schedules are not manufactured by this runner. Numerical records remain private until a separate reviewed export; local execution does not publish the book.
