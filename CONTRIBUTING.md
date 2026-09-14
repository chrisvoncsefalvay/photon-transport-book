# Contributing

Contributions should preserve the single-source executable model, static publication architecture, and reviewed release boundary.

## Scope

- Put executable scientific code in `python/dpt/`, with focused tests and explicit `book:` regions only where the book refers to it.
- Put prose and page composition in `src/content/`; never paste a second implementation into MDX.
- Put publication-safe build utilities in `tools/public/`. Maintainer-only release and authoring machinery is outside the public contribution surface.
- Do not commit raw medical data, unlicensed models, local profiles, W&B run directories, secrets, or fabricated scientific output.

## Licensing boundary

By contributing, you agree that code, CSS, configuration, tests, and executable examples are provided under Apache-2.0, while book prose, editorial content, and original editorial figures are provided under CC BY-NC 4.0 unless a file records another compatible licence. Third-party work must retain its original terms and receive an entry in `ATTRIBUTIONS.md`.

## Checks

Run the relevant site and Python commands from `README.md`. Code shown in the book must pass source-region validation. A generated artefact must pass its schema and include truthful provenance. GPU correctness or performance claims require actual NVIDIA hardware evidence; CPU CI does not substitute for it.

Release publication is maintainer-operated. Contributors should run the public validators but should not add deployment credentials or automation that bypasses review.
