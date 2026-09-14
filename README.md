# Differentiable Photon Transport — Volume I: Foundations

This repository is the published, inspectable edition of an online-first
technical book. It develops differentiable photon transport from canonical
production-grade NVIDIA Warp/CUDA implementations, using X-ray and C-arm
imaging as the initial domain.

The browser site is static. Expensive interactions use checked-in results
generated from the same source files shown in the book; no production GPU
service or second browser physics implementation is involved.

## Local site

Install Node.js and pnpm, then run:

```sh
pnpm install --frozen-lockfile
pnpm run dev
```

Use `pnpm run build` for the static production build. Source listings are
extracted at build time from marked regions in the executable Python sources.

## Computational reproduction

Python dependencies are managed with `uv`. Structural checks can run without a
GPU, but reproducing Warp/CUDA experiments requires a supported NVIDIA GPU and
CUDA driver. Published artefact manifests record the generator, source commit,
device, precision, parameters, and whether a result is exact, precomputed,
recorded, or stochastic.

## Provenance and licensing

The private authoring repository exports this edition through a deterministic
allowlist. Release metadata identifies the private source commit without
exposing private notes, research material, review agents, or promotion code.

Code is licensed under Apache License 2.0; book text and editorial content are
licensed under CC BY-NC 4.0. Third-party assets retain their own licences and
must be recorded in `ATTRIBUTIONS.md` before inclusion.
