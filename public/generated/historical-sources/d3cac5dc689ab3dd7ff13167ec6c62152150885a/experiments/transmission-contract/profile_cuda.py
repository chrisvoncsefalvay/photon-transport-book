"""Bounded warmed CUDA capture for Nsight or Compute Sanitizer.

The CUDA profiler range contains only library forwards/VJPs (or their graph
replays). Input construction, module compilation and diagnostic transfers are
outside the capture. This script defines no transmission physics.
"""

from __future__ import annotations

import argparse
import importlib
import json
from typing import Any

from dpt.transmission import TransmissionSpec, prepare_transmission, transmission_vjp, transmit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=1048576)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--regime", choices=("ordinary", "tail", "mixed"), default="ordinary")
    parser.add_argument(
        "--beam", choices=("scalar", "device-scalar", "per-pixel"), default="scalar"
    )
    parser.add_argument("--mask", type=int, choices=range(1, 16), default=15)
    parser.add_argument("--block", type=int, choices=(128, 256), default=256)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if args.pixels <= 0 or args.iterations <= 0:
        parser.error("pixels and iterations must be positive")
    wp: Any = importlib.import_module("warp")
    np: Any = importlib.import_module("numpy")
    wp.init()
    device = wp.get_device("cuda:0")
    pattern = np.array(
        [1.0, 2.0] if args.regime == "ordinary" else [110.0, 200.0], dtype=np.float32
    )
    if args.regime == "mixed":
        pattern[0] = 1.0
    depth = wp.array(np.resize(pattern, args.pixels), dtype=wp.float32, device=device)
    workspace = prepare_transmission(
        TransmissionSpec(beam=args.beam, active_beam=args.beam != "scalar", block_dim=args.block),
        max_pixels=args.pixels,
    )
    beam: Any = (
        1000.0
        if args.beam == "scalar"
        else wp.full(
            1 if args.beam == "device-scalar" else args.pixels,
            1000.0,
            dtype=wp.float32,
            device=device,
        )
    )
    destinations = {
        name: wp.empty(args.pixels, dtype=wp.float32, device=device)
        for index, name in enumerate(("out_T", "out_counts", "out_log_T", "out_removed"))
        if args.mask & (1 << index)
    }
    seed = wp.ones(args.pixels, dtype=wp.float32, device=device)
    gradient = wp.empty(args.pixels, dtype=wp.float32, device=device)
    beam_gradient = None if args.beam == "scalar" else wp.empty_like(beam)

    def operation() -> None:
        transmit(depth, beam, workspace=workspace, validate=False, **destinations)
        transmission_vjp(
            depth,
            beam,
            seed_counts=seed,
            out_grad_L=gradient,
            out_grad_n0=beam_gradient,
            workspace=workspace,
            validate=False,
        )

    transmit(depth, beam, workspace=workspace, **destinations)
    for _ in range(3):
        operation()
    workspace.check_status()
    graph = None
    if args.graph:
        with wp.ScopedCapture(device=device) as capture:
            operation()
        graph = capture.graph
        wp.capture_launch(graph)  # Instantiate/upload before the measured range.
    wp.synchronize_device(device)
    wp.cuda_profiler_start(device)
    for _ in range(args.iterations):
        if graph is None:
            operation()
        else:
            wp.capture_launch(graph)
    # Drain only at the capture boundary so the last queued kernels are recorded.
    # This explicit drain is excluded from claims about synchronisation inside operation().
    wp.synchronize_stream(workspace.stream)
    wp.cuda_profiler_stop(device)
    workspace.check_status()
    print(
        json.dumps(
            {
                "status": "passed",
                "parameters": vars(args),
                "device": device.name,
                "warp": wp.__version__,
                "scratch_bytes": workspace.scratch_bytes,
                "input_bytes": int(depth.size) * 4
                + (0 if args.beam == "scalar" else int(beam.size) * 4),
                "output_bytes": sum(int(value.size) * 4 for value in destinations.values()),
                "seed_and_gradient_bytes": 8 * args.pixels
                + (0 if beam_gradient is None else 4 * int(beam_gradient.size)),
                "capture": "cudaProfilerStart/Stop; excludes setup, compilation and diagnostics",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
