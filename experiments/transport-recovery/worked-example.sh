#!/usr/bin/env bash
set -euo pipefail

script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$script_root"

# region book:transport-scattering-worked-commands
DPT_EXAMPLE_OUTPUT=${1:?Pass a new absolute output directory outside the checkout}
if [[ "$DPT_EXAMPLE_OUTPUT" != /* || -e "$DPT_EXAMPLE_OUTPUT" ]]; then
  echo "The output directory must be absolute and new." >&2
  exit 2
fi
export PYTHONPATH=python
.venv/bin/dpt-check-gpu
.venv/bin/python experiments/transport-recovery/repair.py \
  --mode reference --sampling-multiplier 16 \
  --output "$DPT_EXAMPLE_OUTPUT/reference"
.venv/bin/python experiments/transport-recovery/repair.py \
  --mode stochastic --sampling-multiplier 16 --repetitions 32 \
  --seed-base 2026091501 \
  --reference "$DPT_EXAMPLE_OUTPUT/reference/scattering-reference/reference.json" \
  --output "$DPT_EXAMPLE_OUTPUT/recovery"
.venv/bin/python experiments/transport-recovery/verify_walkthrough.py \
  --input "$DPT_EXAMPLE_OUTPUT/recovery" \
  --reference "$DPT_EXAMPLE_OUTPUT/reference/scattering-reference/reference.json"
# endregion book:transport-scattering-worked-commands
