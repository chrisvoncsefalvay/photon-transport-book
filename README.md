# Differentiable Photon Transport — Volume I: Foundations

**Build a differentiable X-ray renderer from scratch, then use it to solve inverse problems.**

By Chris von Csefalvay · [Read the book](https://photontransport.com/)

From zero to actual differentiable photon transport in CUDA/Warp without losing
your sanity to physics, your livelihood to GPU rental and your will to live to
maths.

An X-ray image records what happened to photons on their way through an object.
A differentiable renderer lets us work backwards: how should we change the
object's pose or its material properties to explain that image? This book develops the physics, mathematics and GPU code needed to answer
that question.

## What you will build

We begin with attenuation along a ray and build towards a renderer with a
calibrated source, a three-dimensional volume and a detector. We derive its
reverse calculation, use image differences to recover pose, and extend the
model to include energy-dependent interactions and scattering. The later
chapters use the resulting derivatives for registration, material reconstruction
and choosing an additional view.

The implementation grows with the argument. You will write the operations that
turn a volume into an image and an image discrepancy into a parameter update,
with NVIDIA Warp and CUDA doing the computational work. Along the way, we
examine why a modelling choice is useful, where an approximation breaks down,
and how the methods relate to established work in rendering and inverse imaging.

## Who it is for

The book is for researchers, engineers and programmers who want to understand
and build differentiable imaging methods. Familiarity with Python, linear
algebra and calculus will help. The X-ray physics and transport mathematics are
developed as they become necessary.

## Read and work along

Start with [Chapter 1](https://photontransport.com/chapters/introduction/), which
sets out the problem and the route through the book. Each chapter connects its
derivations to the accompanying [Python implementation](python/dpt/README.md).
Use the supplied source to compare your own implementation at each stage.

The [application examples](python/dpt/examples/README.md) cover registration,
attenuation reconstruction and acquisition selection. Running the GPU code
requires a supported NVIDIA GPU and CUDA driver; the online book can be read
without either.

## Citation and licences

See [CITATION.cff](CITATION.cff) for citation details and the archived edition.
Code is licensed under [Apache License 2.0](LICENSE-CODE); book text and editorial
content are licensed under [CC BY-NC 4.0](LICENSE-CONTENT). Third-party material
is credited in [ATTRIBUTIONS.md](ATTRIBUTIONS.md).
