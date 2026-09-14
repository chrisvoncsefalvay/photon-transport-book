# Classified directional checks

`run.py` records the canonical CUDA projection, transmission and pose VJP for one ray through a prescribed 3×3×3 trilinear tent. This is an exact software fixture, not anatomy. The object-space ray crosses an interior interpolation knot when the x translation changes sign.

At translation 0.25 mm, steps below 0.25 mm remain on one smooth branch. At zero, the ordinary derivative does not exist: the floor-selected cell derivative and the symmetric difference answer different questions. Both one-sided differences are retained. The figure must distinguish steps crossing the knot from the smooth-step regime.

The field is constant along y and z at its sample centres, with x weights (1,2,1). The half-cell support with boundary-sample clamping makes the integrated z weight 3 mm. For |t|<1 mm, the exact depth is (3 mm) μ [2−|t|/(1 mm)]; this supplies an independent smooth derivative. Scalar FP64 evaluation of the stored field independently checks the forward calculation. Actual FP32 subtraction errors are retained, including exact-zero symmetric differences.

Run on an available CUDA device, into a new private directory:

```bash
PYTHONPATH=python .venv/bin/python experiments/directional-checks/run.py --output /absolute/private/new-run
```

The experiment snapshots its sources and hashes outputs using `RunRecorder`. It does not change scientific kernels or claim physical validation or performance.
