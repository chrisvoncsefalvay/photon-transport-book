# Differentiable Photon Transport — Volume I: Foundations

[Read the book at photontransport.com](https://photontransport.com/).

This repository contains the book's text and executable NVIDIA Warp/CUDA
examples. We develop differentiable photon transport using X-ray and C-arm
imaging.

The site is static. Figures that need expensive calculations use recorded
simulation results, with downloadable numerical data and provenance. Those
simulations run outside the browser.

## Local site

Install Node.js 24 and the pnpm version pinned in `package.json`, then run:

```sh
pnpm install --frozen-lockfile
pnpm run dev
```

Use `pnpm run build:vercel` to build the published snapshot, as described in
[the deployment guide](DEPLOYMENT.md). Source listings are extracted at build
time from marked regions in the executable Python sources.

## Use the computational code

The `python/dpt` package contains the renderer, derivatives and inverse solvers
developed in the book. The [application examples](python/dpt/examples/README.md)
provide runnable interfaces for registration, scalar attenuation reconstruction
and acquisition selection with supplied data.

Python dependencies are managed with `uv`. Structural checks can run without a
GPU; executing the Warp/CUDA operators requires a supported NVIDIA GPU and CUDA
driver. Figure downloads preserve recorded numerical results and their
provenance, including the model, parameters and computational precision.

## Provenance and licensing

An explicit file allowlist controls what is exported from the authoring
repository. The site's [snapshot record](https://photontransport.com/website-snapshot.json)
identifies the source commit used for its build. Private notes, research
material, review agents and promotion code stay in the authoring repository.

Code is licensed under Apache License 2.0; book text and editorial content are
licensed under CC BY-NC 4.0. Third-party assets retain their own licences and
must be recorded in `ATTRIBUTIONS.md` before inclusion.
