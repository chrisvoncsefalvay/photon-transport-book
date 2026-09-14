# Experiments

The [spectral material reconstruction study](spectral-reconstruction/README.md)
provides physical-input preparation, CT-derived spectral observations, a joint
CUDA material-volume solver and recorded numerical evaluation. Its response is
an explicitly ideal photon-count model; its material reference is assigned from
TCIA anatomy, not measured patient composition.

`transmission-contract/` executes the canonical CUDA library, validates deterministic sweeps against an independent high-precision reference, and renders the figures used in sections 2.6–2.7. Its README records the dependency groups and runner commands.

Raw run records, source snapshots, timings and profiler captures belong in an explicit private output directory outside the entire repository. A reviewed projection of real figure outputs belongs in `public/generated/`, with a manifest that verifies the current source and output hashes. Generating a local figure does not publish or promote the book.

The GPU preflight never logs into W&B. These experiments require no service credentials or network logging. The pelvic figure uses separately attributed synthetic CT and conditioning-mask inputs. Numerical stress workloads are explicitly synthetic and are not presented as physical measurements.

`pose-sensitivity/` prepares the archived synthetic pelvic CT, validates its canonical CUDA projection and six pose derivatives, and records the 55 poses used by the sensitivity figure in Chapter 5. Its README distinguishes the synthetic CT from the attributed conditioning-mask surfaces.

`radiograph-study/` records three native MAISI CT volumes as AP/PA/lateral primary projections, a wider pelvic rotation sequence, CT slices and a four-exposure Poisson patch. Its geometry, display windows, model limits and validation are explicit; private raw records are curated into display assets without exporting CT volumes.

`directional-checks/` supplies two classified canonical CUDA checks for the introduction: a smooth translated trilinear fixture and a pose exactly at an interior interpolation knot. Both one-sided differences and the selected VJP are retained, with independent analytic and scalar references.

## Remaining library experiments

All eight drivers were executed on the GB10 during the 9 September 2026 acceptance pass. Their raw records remain private. Analytic software fixtures establish implementation checks; supplied material data and physical model validity require separate evidence. A completed run records what happened, including an unresolved recovery outcome, rather than declaring scientific success.

| Directory                 | Actual library composition and record                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------- |
| `projection-convergence/` | Separate grid and quadrature sweeps against exact sampled-field and continuous-field references    |
| `projection-gradients/`   | Local pose VJP versus independent directional step sweeps                                          |
| `pose-recovery/`          | Prescribed starts, canonical deterministic inverse loop and independent geometric metrics          |
| `spectral-projection/`    | Supplied material fields, energy weights and detector response through the spectral composition    |
| `acquisition-mismatch/`   | Named acquisition groups, fixed/shared nuisance constraints and multiple-view recovery             |
| `transport-validation/`   | Analytic flight/survival limits and complete original-history moments                              |
| `transport-gradients/`    | Supported likelihood-score derivatives, replay and independent finite-difference/replicate reports |
| `transport-recovery/`     | Concrete CUDA expected-signal oracle with independent product estimators and stochastic acceptance |

`dpt.experiments.RunRecorder` captures source/configuration bytes and installed versions before a run, writes actual driver-supplied device metadata and values, and verifies source/output hashes at completion. Failed runs remain failed. It does not create a public manifest or render plots. Each README distinguishes analytic protocol fixtures from supplied physical data and lists the acceptance checks. `library-profile/` provides a separate warmed timing harness for actual library operators; its analytic workloads, NVTX ranges, source snapshots and contention records make before/after comparisons reproducible.

## Run record format

The eight library drivers above use `dpt.experiments` for root discovery, CLI configuration, a new private output directory and complete package source snapshots. Their new `run.json` files use schema version 2: `status` is `running`, `complete` or `failed`; `started_utc` and `finished_utc` identify execution times. `configuration`, `environment` and `metadata` hold their respective records, while `source_sha256` and `output_sha256` identify recorded bytes. The exact effective configuration bytes, including CLI overrides, are also saved as `configuration.json` and bound by `configuration_sha256`. A complete run must also report unchanged sources and recorded files. Completion alone does not imply successful recovery or physical validation. Other study entrypoints document their input, preparation and run records in their own README.

Version 1 historically had two writers: general runs used `complete`/`finished_utc`, and transmission used `passed`/`completed_utc`. Historical private files remain intact. The transmission importer reads its historical version-1 form and the common version-2 form, then writes a sanitised version-2 projection. Transmission `validation.json` remains a separate version-1 validation report with `status: passed`; the plot reader consumes this validation payload, not `run.json`.
