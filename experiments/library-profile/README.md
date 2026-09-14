# Canonical library profiling

This driver measures actual library operators with explicitly analytic inputs. It contains no replacement transport or projection implementation. The fixtures describe reproducible workloads; they do not represent measured anatomy, spectra or clinical performance.

Run from the isolated authoring worktree with its pinned CUDA environment and the Python `nvtx` package. Select a new output directory outside the repository:

```bash
PYTHONPATH=python .venv/bin/python experiments/library-profile/run.py \
  --case projection --side 256 --grid 64 --samples 128 \
  --iterations 10 --repeats 7 --output /absolute/private/runs/projection-baseline

PYTHONPATH=python .venv/bin/python experiments/library-profile/run.py \
  --case spectral --side 512 --materials 2 --energies 32 \
  --output /absolute/private/runs/spectral-baseline

PYTHONPATH=python .venv/bin/python experiments/library-profile/run.py \
  --case transport --side 256 --grid 64 --histories 65536 \
  --output /absolute/private/runs/transport-baseline
```

Each case allocates all device inputs, destinations and workspace scratch on one explicit stream before timing. A checked invocation validates every operation, followed by three unchecked warmup cycles by default. Inputs then remain immutable. Timing uses unchecked calls with persistent error status; mandatory status checks occur before and after the measured region. Kernel compilation, allocation and diagnostic downloads are excluded. Device outputs are not copied back during timing.

| Case         | Named timed operations                                                       | Declared fixture                                                                                                                                                                                                    |
| ------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `projection` | `projection_forward`, `projection_pose_vjp`                                  | Quadratic scalar attenuation on a 20 mm cube, finite source/detector rays and a fixed oblique pose. FP32 sampled field; FP64 pose gradient.                                                                         |
| `spectral`   | `spectral_forward`, `detector_forward`, `detector_vjp`, `spectral_paths_vjp` | Positive analytic material coefficients, constant 10 mm paths, uniform energy-bin populations and unit response. Calibration uses fixed exposure and active shared gain/offset.                                     |
| `transport`  | `transport_forward`, `transport_density_replay`, `transport_moments`         | Homogeneous absorption 0.01/mm and isotropic elastic scattering 0.02/mm. Repeated deterministic central source ray with independent history RNG identities. Direct hits intentionally concentrate tally contention. |

`--operation NAME` restricts measured operations while warming the full case to establish dependencies. `--side`, `--grid`, `--samples`, `--histories`, `--materials` and `--energies` set workload dimensions. `--iterations`, `--repeats`, `--warmup`, `--seed` and `--device` make comparisons reproducible. The stochastic timing loop deliberately replays the same batch; these repetitions measure performance and are **not independent statistical replicates**.

The current harness does not measure volume-scatter VJPs, shared spectral coefficient/weight derivatives, blur/noise sampling, energy-changing Compton histories, complete recovery iterations or graph replay. Those require separately named workloads before making corresponding performance claims.

## Timing and interpretation

`timings.json` contains every repeat plus the median and inclusive interquartile range in milliseconds per invocation. CUDA events bracket a batch of `iterations` calls on the owning stream. Event elapsed time can include GPU idle gaps caused by host submission; it is not a sum of pure kernel durations. Host wall time includes event recording and waiting for completion. NVTX annotations enclose each measured batch; CUDA profiler start/stop encloses the measured operation groups. Synchronisation between repeats is explicit.

`run.json` and source snapshots identify the exact implementation, configuration, Python/package environment, Warp version, CUDA driver, device architecture and workspace scratch. Scratch excludes caller-owned input/output arrays. `contention-before.json` and `contention-after.json` preserve timestamped `nvidia-smi` output, including failures to query it. These are endpoint snapshots and do not prove that the GPU stayed uncontended. Never terminate another user's workload to improve the timings.

Use matched dimensions, numerical policy, seed and device for before/after comparisons. Repeat clean timing runs outside profilers; Nsight replay/instrumentation changes timing. Run independent correctness tests and sanitizers before interpreting a faster implementation as an improvement.

## Nsight and cold compilation

```bash
PYTHONPATH=python nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
  -o /absolute/private/runs/projection-timeline \
  .venv/bin/python experiments/library-profile/run.py \
  --case projection --iterations 10 --repeats 3 \
  --output /absolute/private/runs/projection-nsys

PYTHONPATH=python ncu --profile-from-start off --set full \
  --launch-count 1 -o /absolute/private/runs/projection-kernel \
  .venv/bin/python experiments/library-profile/run.py \
  --case projection --operation projection_forward --iterations 1 --repeats 2 \
  --output /absolute/private/runs/projection-ncu
```

For cold-cache evidence, invoke this command through NVIDIA's `warp_compile_probe`, supplying the same argument vector and a fresh output path. The probe owns cache isolation and compile-time reporting; this driver deliberately implements no competing cache deletion/reset mechanism. Its timing records always describe warmed operations, even when the containing process began with a cold cache. Preserve the probe report beside the run. If the capability is unavailable, report cold compilation as unmeasured rather than treating initial process wall time as kernel execution time.
