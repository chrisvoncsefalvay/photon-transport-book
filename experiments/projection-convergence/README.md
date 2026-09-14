# Projection convergence

This experiment sweeps quadrature count at fixed voxel resolution and voxel resolution at fixed quadrature count. Both retain the same 20 mm cubic support. The attenuation field is an explicitly mathematical positive quadratic, with coefficients in mm⁻¹ and mm⁻³; it represents no anatomical or measured material data.

The CUDA operator is `dpt.projection.project_optical_depth`. An independent piecewise Gauss rule integrates its sampled trilinear field exactly up to binary64 arithmetic; a closed polynomial integral supplies the continuous-field reference. `convergence.json` separates quadrature error, grid error and their combined effect. The uploaded binary32 values, rather than unrounded host inputs, define the sampled-field reference.

Run from the repository root with the optional GPU dependencies installed:

```sh
PYTHONPATH=python python3 experiments/projection-convergence/run.py --output /absolute/private/new-run
```

The output directory must be new. A running/failed/complete run record includes source hashes, configuration and device metadata. These files record actual execution only; no output is checked in and no visualisation is generated. A completed run is not automatically an accuracy acceptance result: review error trends, boundary cases and the independent GPU test suite before integrating the evidence into Chapter 4.

The 9 September 2026 GB10 acceptance and integration passes executed this protocol with the recorded source versions. The numerical sweep is available for later book figures; rendering and public export are separate steps.
