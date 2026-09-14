# MAISI radiograph study

This offline study prepares the three archived synthetic MAISI CT candidates and
records primary radiographs with canonical CUDA projection/transmission operators.
It also records a 31-frame pelvic yaw sweep and four canonical Poisson observations
of one declared low-contrast region in the same pelvic reference radiograph.

The native RAS affine and HU values are preserved. Each volume is centred at the
physical centre of its full native grid, rather than the sacral pivot used by the
introduction's separate pose-sensitivity study. Attenuation is the same uncalibrated
80 keV water-equivalent approximation, `0.01837 * max(0, 1 + HU/1000)` mm⁻¹.
All tissue in the synthetic CT contributes; no conditioning-mask labels, meshes,
anatomical completion, spectrum or scatter model are introduced.

The fixed source is at (0, 750, 0) mm and detector centre at (0, −450, 0) mm.
At identity this is an AP beam. Rotating the object by π about its superior axis
gives PA; π/2 gives a right-to-left lateral beam. Every image records the exact
body-to-acquisition matrix and source direction in body coordinates. Columns and
rows remain fixed in acquisition coordinates; these are not mirrored clinical
viewing presets. Field size is 600×480 mm at 480×256 pixels: preserve the physical
5:4 aspect ratio when displaying these anisotropic detector pixels.

All 39 distinct views are independently recorded at 4096 and 8192 samples per ray.
Acceptance requires maximum/p99 count differences no greater than 0.25/0.05 at
an open-beam mean of 1000. Three independent piecewise-polynomial FP64 reference
rays per view use `dpt.validation.projection.integrate_sampled_field`; each count
must agree within 0.05. The full synthetic field remains resident during each
volume's CUDA calls. Host downloads and PNG encoding occur only in this offline
driver. The reused `Scene` retains the field, pose and output buffers; projection
workspace is recreated whenever the sampling level changes.
No new scientific kernels or browser physics are added. No performance claim is
made on the shared GPU.

Each CT display is a native central axial, coronal or sagittal slice with explicit
orientation and physical dimensions. Radiograph PNGs linearly map optical depth
0–8 to black–white, so greater attenuation is bright. These fixed display windows
clip only PNG intensities; raw HU, attenuation, counts and optical depth retain
their recorded values. The native finite CT boundary remains visible where
intersected. Projection views are of the available cropped field, not complete
patient acquisitions.

The noise region is selected deterministically from 32×32 windows in the central
60% of the pelvic AP detector, at stride eight. Eligible windows have mean
transmission between 0.01 and 0.2; the window with lowest coefficient of variation
is retained only when that value is below 0.08. Its four-pixel border defines the
background mean. Expected counts are scaled to background levels 10, 100, 1000
and 10000; canonical Poisson draws retain distinct observation identities and
uint64 observations. All four displays use normalised counts 0.8–1.2 with inverse
greyscale. Total-count and Pearson statistics use predeclared seven-standard-error
checks, and identity-preserving replay must reproduce each observation exactly.
The patch is actual synthetic-CT projection texture, not a clinical lesion or a
generated shape presented as anatomy.

Set `DPT_INPUT_ROOT` to the prepared MAISI inputs and `DPT_OUTPUT_ROOT` to a new
external output directory. Run with the existing experiment environment:

```bash
PYTHONPATH=python .venv/bin/python experiments/radiograph-study/run.py \
  --input-root "$DPT_INPUT_ROOT" \
  --output "$DPT_OUTPUT_ROOT" \
  --device cuda:0
```

`run.json` records source, configuration and output hashes. `study.json` contains
display metadata, native geometry, per-view matrices, refinement/reference
checks and observation statistics. Original volumes and full float/uint64 output
arrays remain private. A separate site importer may curate only reviewed PNGs
and metadata after the run completes and every acceptance check passes.
