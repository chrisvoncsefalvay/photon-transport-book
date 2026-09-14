# Recorded hero observations

This display experiment replaces the painted image reveal with recorded
Poisson photon-count observations. It preserves the introduction's original
synthetic attenuation field and conditioning mask, and constructs a **separate
isolated-pelvis phantom** by retaining attenuation only at mask labels 2–4.
These labels conditioned the synthetic CT; they are not new CT segmentations.
The existing water-equivalent 80 keV approximation is retained. There is no
scatter, spectrum, clinical detector calibration or dose claim.

The canonical CUDA projection and transmission operators evaluate a wider
720×600 mm field; independent midpoint refinement checks its integration.
`dpt.detector.sample_poisson_counts` records 128 independently identified
count images. Integer counts are packed losslessly into an 8-bit PNG atlas
only after checking their maximum. The mean for each count bin uses one central
pencil ray, without detector-area quadrature. Playback places its recorded count
at the pixel centre and time-bin endpoint. It does not claim subpixel or
continuous arrival-time resolution.

The browser applies a normalised small spatial kernel, additive light and
exponential persistence to these observations. Exposure, persistence and flash
times are authored display parameters. The finite recording repeats; this
replay is not a new independent observation stream. The receiver has no border
and its light tapers into the surrounding page. The cropped native mask is
preserved; no missing pelvic anatomy is completed or invented.

Install the optional authoring dependencies into the Python experiment environment:

```bash
uv sync --group experiment
uv pip install --python .venv/bin/python -r experiments/hero-fluoro/requirements-authoring.txt
```

Run with explicit private inputs:

```bash
PYTHONPATH=python .venv/bin/python experiments/hero-fluoro/run.py \
  --prepared /absolute/private/prepared-final \
  --labels /absolute/private/label.nii.gz \
  --output /absolute/private/new-hero-record
```

Only the atlas, selected actual-hit cells, statistics and provenance are copied
to `public/generated/transport-hero/`; raw CT, masks and arrays remain private.

The screen uses neutral greyscale. After building and serving the local static
site, regenerate its transparent, reduced-motion fallback with:

```bash
node tools/public/render-transport-hero-still.mjs \
  --url http://127.0.0.1:4174/ \
  --playwright /absolute/path/to/playwright/index.mjs \
  --chromium /absolute/path/to/chromium
```

The optional authoring tool records the renderer, source and output hashes in
`fluoro-still.json`; normal site builds use the checked-in image. Build again
after regeneration to copy the new fallback into the preview output.
