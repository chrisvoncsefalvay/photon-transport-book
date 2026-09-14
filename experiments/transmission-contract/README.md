# Transmission contract experiment

This runner executes the canonical `dpt.transmission` CUDA operator. Its mathematical stress
inputs are deterministic depth/decrement sweeps and analytic path cases, not anatomy or
observations. Every plotted canonical forward/inverse value passes the independent 100-digit Decimal
oracle's frozen acceptance budgets before any figure is written. Log transmission is checked
bit for bit, including signed zero. The labelled subtraction diagnostic illustrates loss of
information and is never accepted as a replacement operator. The path cases check the
transmission operator at independently supplied optical depths; they do not validate a path
integrator that has not yet been implemented.

From the repository root, use the pinned GPU and experiment dependencies (including
`matplotlib==3.10.7`):

```bash
uv sync --group gpu --group experiment
PYTHONPATH=python uv run --no-sync python experiments/transmission-contract/run.py \
  --output-dir /tmp/dpt-transmission-validation
```

The explicit `PYTHONPATH` is useful in linked worktrees when an existing editable installation
points to another checkout. A CUDA device is mandatory; unavailable CUDA fails the run.
`--device cuda:0` is the default. `--output-dir` must be new or empty and must resolve outside
`public/` and `experiments/`, including through symlinks. The command never overwrites an
existing run or promotes any result into the book's public artefacts.

## Outputs and provenance

- `run.json`: completion/failure status, exact command, effective configuration, source commit
  and dirty state, SHA256 hashes, runtime/device/compiler details, numerical policy and output hashes.
- `sources/`: exact copies of the relevant implementation, oracle, runner, configuration and
  dependency files. A source change during the run marks the run failed and requires rerunning.
- `validation.json`: actual CUDA outputs, high-precision references and per-value ULP checks;
  plus the analytic case definitions and inverse-decrement errors.
- `transmission-sweep.svg`: transmission, expected counts and log transmission across depth.
- `transmission-weak-attenuation.svg`: canonical removed fraction and inverse accuracy, with
  explicitly labelled FP32 subtraction of stored library transmission as a cancellation diagnostic.
- `transmission-tail-rescue.svg`: separately rounded transmission/counts, with the first sampled
  stored zero explicitly annotated. Zeros are omitted from logarithmic axes, not floored.
- Corresponding PNG previews support visual inspection. SVG text remains text; no third-party
  font binary is embedded. Matplotlib uses a fixed SVG hash salt and suppresses the creation-date
  field, so the same inputs/style can produce byte-stable SVG files.

The SVGs plot recorded library results; the runner contains no replacement transport formula.
Reference points and ULP errors come from `dpt.validation.transmission`. Sampling geometry,
open-beam means and mathematical domains are recorded in the run files. Runtime environment
metadata and raw benchmark results remain private unless separately reviewed for export.

## Warmed execution measurements

```bash
PYTHONPATH=python uv run --no-sync python experiments/transmission-contract/run.py \
  --output-dir /tmp/dpt-transmission-benchmark --benchmark
```

`config.json` selects five pixel counts from 1,024 to 16,777,216, pointwise block sizes 128 and
256, and ordinary, tail and alternating numerical regimes. The reference pattern repeats 256
specified binary32 depths. Inputs and outputs are prepared before timing. Forward outputs and
both depth/beam gradients are checked before that workload's timing begins. These bounded
checks supplement the full test suite; they do not replace it.

Each case measures fused four-output forward execution, four separate library launches,
individual outputs, count-seeded VJPs, forward-plus-VJP execution and captured-graph replay for
a fixed scalar, per-pixel active beam and active broadcast beam. It also measures a matched
device-copy baseline and checked forward/VJP wall time through completed output. The reduction always uses 256-thread tiles regardless of pointwise block size.
The file `benchmark.json` is updated after each size/block/regime group so completed timing
records survive a subsequent failure; `run.json` decides whether the whole run passed.

Five independent batches use timing-enabled CUDA events on the workspace stream. Reported
samples divide batch event duration by its iteration count; median, minimum and maximum are
retained. Device timestamps can include host submission gaps between launches, especially for
small inputs. Compilation, upload, allocation, warmup and checked-call content validation are
excluded from event timings. Checked wall-clock rows explicitly include host metadata checks,
device value scans, status readback and stream completion. These event timings therefore
describe warmed asynchronous library submission, not a
claim of isolated arithmetic throughput. Logical byte counts are estimates of array accesses,
not measured DRAM traffic; status checks, cache effects and scalar-reduction scratch are stated
separately. Timing records label CUDA-event versus checked-wall measurements. Nsight collection
is a separate run. Scalar-gradient prechecks use the canonical sum of per-element budgets,
a gamma bound with twelve additions per reduction level, and final binary32 rounding; the
actual bound and addition depth are retained in each applicable benchmark row.

For a short CLI/integration smoke, override the workload without modifying the canonical config:

```bash
PYTHONPATH=python uv run --no-sync python experiments/transmission-contract/run.py \
  --output-dir /tmp/dpt-transmission-smoke --benchmark --sizes 1024 --iterations 2
```

Use a fresh output directory for each run. Do not run other GPU tests or profiling while taking
representative timings. The runner records `nvidia-smi` output, but this snapshot cannot prove
exclusive GPU use throughout the experiment; the operator must coordinate that externally.

The twelve-addition bound follows the pinned [Warp 1.17 tile reduction source](https://github.com/NVIDIA/warp/blob/v1.17.0/warp/native/tile_reduce.h#L258):
five shuffle additions combine each 32-lane warp, then thread zero serially adds the eight
warp totals with seven more additions. A 256-element tile therefore has a longest path of
twelve additions, even though a balanced 256-element tree would have depth eight. The bound
is applied at every recursive level, including partially filled tiles.

The numerical sum uses no global floating-point atomics. Warp's tile machinery does use
shared-memory integer atomic bookkeeping to track participating warps; this is distinct from
accumulating per-pixel contributions into a contended global scalar. Profiler records distinguish
shared bookkeeping, global atomic transactions and the actual FP64 numerical reduction.

The driver shares the version-2 `RunRecorder` envelope documented in [the experiments overview](../README.md#run-record-format), including failure handling and source/output verification. `--output-dir` is an alias for the common `--output` argument; both require a new directory outside the entire repository. Figures and validation are recorded before completion, and every Python package dependency is snapshotted. The separate `validation.json` payload retains `status: passed` after the independent checks succeed.
