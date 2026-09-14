# Pelvic pose sensitivities

This offline experiment supplies Figure 1.1 in introduction §1.3. The canonical
Warp projection and transmission operators produce the expected primary count
image and six local right-SE(3) sensitivities, in counts/mm and counts/rad.
The browser loads recorded results and transforms display meshes; it performs no
photon-transport calculation.

## Inputs and model

`config.json` binds one archived 256³ synthetic MAISI pelvic CT and its paired
conditioning mask by SHA-256. `prepare.py` verifies those identities and their
shared native RAS affine, derives the fixed sacral centroid, and transposes XYZ
storage into contiguous ZYX without interpolation. The actual 1.5177865028 mm
spacing is retained; the generation request's nominal 1.5 mm is not substituted.
The full CT supplies attenuation. Labels never mask the field.

The illustrative conversion is `mu = 0.01837 * max(0, 1 + HU/1000)` in mm⁻¹.
The water value comes from the total mass-attenuation coefficient at **80 keV**,
0.1837 cm²/g at a stated density of 1 g/cm³ in the
[NIST water table](https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html).
This is an uncalibrated water-equivalent approximation, not an 80 kVp spectrum
or a material-specific model. No scatter, spectral response or noise is modelled.

The AP source is 750 mm anterior to the fixed pivot; the detector is 450 mm
posterior, with a 600×320 mm field at 256×480 pixels. Columns increase towards
model right; rows increase inferiorly. The display does not apply a radiological
left/right mirror. The cropped inferior hip is outside the selected central and
superior pelvic projection; this is not a complete pelvic-ring study.

`mesh.py` extracts labels 2–4 from the augmented TotalSegmentator conditioning
mask. Display smoothing and stride-2 marching cubes change only these surfaces.
The credit, CC BY 4.0 licence and modifications travel with `meshes.json` and the
figure. See `ATTRIBUTIONS.md`; these are not fresh segmentations of the synthetic CT.

## Execution

Use Python 3.12, the pinned `experiment` dependency group and the additional
NIfTI/meshing dependencies in `requirements-volume.txt`. No raw volume is in the
repository. Each output directory must be new and outside the source tree.

```bash
uv pip install --python .venv/bin/python -r experiments/pose-sensitivity/requirements-volume.txt
PYTHONPATH=python .venv/bin/python experiments/pose-sensitivity/prepare.py \
  --image /absolute/private/image.nii.gz --labels /absolute/private/label.nii.gz \
  --output /absolute/private/prepared
PYTHONPATH=python .venv/bin/python experiments/pose-sensitivity/mesh.py \
  --labels /absolute/private/label.nii.gz --output /absolute/private/meshed
PYTHONPATH=python .venv/bin/python experiments/pose-sensitivity/run.py \
  --prepared /absolute/private/prepared --meshed /absolute/private/meshed \
  --output /absolute/private/recorded
node tools/public/import-pose-sensitivity-artifacts.mjs --run-dir /absolute/private/recorded
```

A version-2 run record snapshots source/configuration bytes and hashes every
output. Failed acceptance leaves a failed record and cannot be imported. The
importer admits only the named figure payloads, verifies their dimensions,
values, units, matrices and hashes, and writes the generated-artifact manifest.
It does not publish the book.

## Acceptance and limits

The 256–8192 sample pilot separated image and derivative convergence. The final
run repeats 1024, 2048, 4096 and 8192 samples per ray; the last two refinements
must pass the frozen `config.json` budgets for counts **and** all six derivative
maps. All six coordinates receive seven signed finite-difference step pairs at
the final detector resolution. Independent scalar FP64 reference rays include
fixed detector positions and each coordinate's strongest pixel. Their discrete
midpoint differences check the derivative of the same numerical model; their
piecewise-Gauss values separately test quadrature error. A mixed-direction Taylor
sweep and an arbitrary signed VJP contraction supplement the per-pixel checks.

Finite-difference disagreement at a pixel can reflect interpolation knots or
FP32 roundoff. Its per-axis counts and the private mask remain in the record;
no global smoothness or clinical-validity claim follows from acceptance.
The six maps are derivatives **at the reference pose**. Selecting one of the 55
recorded poses changes the displayed projection and mesh transform, while those
reference sensitivities remain fixed. Each translation spans −2 to +2 mm in
0.5 mm steps; each rotation spans −0.05 to +0.05 rad in 0.01 rad steps.
Every displayed pose is additionally checked at 4096 and 8192 samples per ray
against the same count max/p99 budgets. Pose payload names include the frozen
configuration digest so an old page cannot fetch a different pose at an old URL.

The input volume, work buffers and six-column Jacobian remain on the GPU during
each operator call. The diagnostic specialisation shares the canonical VJP's
per-ray arithmetic and writes six FP64 values per pixel before reduction. It
uses one tiled launch, no volume-gradient atomics, no sample tape, and no
per-pixel allocations. Explicit record downloads happen in this offline driver.
The complete detector Jacobian occupies 5.625 MiB. No performance comparison is
claimed on the shared GB10.
