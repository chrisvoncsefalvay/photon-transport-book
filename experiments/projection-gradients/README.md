# Projection gradients

This experiment contracts the canonical CUDA projection with a specified signed detector cotangent and records all six local right-SE(3) pose derivatives. Translation coordinates are in mm, rotations in radians. The perturbed pose is `anchor @ exp(delta^)`; the anchor remains fixed throughout each directional sweep.

`gradients.json` compares the CUDA VJP with two finite-difference sequences: actual CUDA forward outputs and an independent binary64 evaluation of the same fixed-count quadrature over the exact uploaded field values. The two references have different roundoff floors. Neither classifies interpolation knots, intersection ties or grazing rays as smooth; those cases have separate test contracts.

Run from the repository root with the optional GPU dependencies installed:

```sh
PYTHONPATH=python python3 experiments/projection-gradients/run.py --output /absolute/private/new-run
```

The output directory must be new. Run status, source hashes, configuration and device metadata accompany the actual records. The field is a labelled analytic quadratic, not anatomy. There are no checked-in numerical results or figures. Review convergence over several step sizes, units, branch stability and the standalone tape/volume-adjoint tests before using these records as Chapter 5 evidence.

The 9 September 2026 GB10 acceptance and integration passes executed this protocol with the recorded source versions. The numerical sweep is available for later book figures; rendering and public export are separate steps.
