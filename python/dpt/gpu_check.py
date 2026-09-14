"""Honest local preflight for the optional CUDA development environment."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from collections.abc import Sequence
from types import ModuleType
from typing import Any


class PreflightError(RuntimeError):
    """A clear, expected GPU preflight failure."""


def _load_dependency(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name:
            raise PreflightError(
                f"optional dependency {name!r} is not installed; run `uv sync --group gpu` first"
            ) from error
        raise PreflightError(
            f"optional dependency {name!r} has a missing transitive dependency: {error.name!r}"
        ) from error
    except Exception as error:
        raise PreflightError(f"could not import optional dependency {name!r}: {error}") from error


def _check_torch(torch: Any) -> str:
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        raise PreflightError("PyTorch was imported but does not expose the torch.cuda API")

    try:
        is_available = bool(cuda.is_available())
        device_count = int(cuda.device_count())
    except Exception as error:
        raise PreflightError(f"PyTorch could not query CUDA devices: {error}") from error
    if not is_available:
        build_cuda = getattr(getattr(torch, "version", None), "cuda", None)
        raise PreflightError(
            "PyTorch cannot access CUDA "
            f"(torch.version.cuda={build_cuda!r}, visible_device_count={device_count})"
        )

    try:
        device = torch.device("cuda:0")
        left = torch.tensor((1.0, 2.0), dtype=torch.float32, device=device)
        right = torch.tensor((3.0, 4.0), dtype=torch.float32, device=device)
        observed = float(torch.dot(left, right).item())
    except Exception as error:
        raise PreflightError(f"PyTorch CUDA operation failed: {error}") from error
    if observed != 11.0:
        raise PreflightError(f"PyTorch CUDA dot product returned {observed!r}, expected 11.0")

    build_cuda = getattr(getattr(torch, "version", None), "cuda", "unknown")
    return (
        f"PyTorch {getattr(torch, '__version__', 'unknown')}: PASS — "
        f"{cuda.get_device_name(device)}; build CUDA {build_cuda}; CUDA dot=11.0"
    )


def _check_warp(wp: Any) -> str:
    try:
        wp.init()
        device_count = int(wp.get_cuda_device_count())
    except Exception as error:
        raise PreflightError(f"Warp could not initialise CUDA: {error}") from error
    if device_count < 1:
        raise PreflightError("Warp reports no CUDA devices")

    try:
        kernel_module = importlib.import_module("dpt._gpu_preflight_kernel")
        device = wp.get_device("cuda:0")
        values = wp.array((1, 2, 3), dtype=wp.int32, device=device)
        wp.launch(kernel=kernel_module.bootstrap_add_one, dim=3, inputs=[values], device=device)
        wp.synchronize_device(device)
        observed = values.numpy().tolist()
    except Exception as error:
        raise PreflightError(f"Warp CUDA kernel execution failed: {error}") from error
    if observed != [2, 3, 4]:
        raise PreflightError(f"Warp CUDA kernel returned {observed!r}, expected [2, 3, 4]")

    device_name = getattr(device, "name", "cuda:0")
    architecture = getattr(device, "arch", "unknown architecture")
    return (
        f"Warp {getattr(wp, '__version__', 'unknown')}: PASS — "
        f"{device_name}; {architecture}; CUDA kernel result=[2, 3, 4]"
    )


def _check_wandb(wandb: Any) -> str:
    return (
        f"Weights & Biases {getattr(wandb, '__version__', 'unknown')}: PASS — import only; "
        "authentication and network access intentionally not checked"
    )


def run_preflight() -> tuple[str, ...]:
    """Run import checks and real, minimal CUDA operations for both GPU runtimes."""
    torch = _load_dependency("torch")
    wp = _load_dependency("warp")
    wandb = _load_dependency("wandb")
    return (_check_torch(torch), _check_warp(wp), _check_wandb(wandb))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local GPU preflight and return a process exit status."""
    parser = argparse.ArgumentParser(
        description="Verify DPT GPU dependencies with real tiny CUDA operations."
    )
    parser.parse_args(argv)

    print(f"DPT GPU preflight — Python {platform.python_version()}", flush=True)
    try:
        results = run_preflight()
    except PreflightError as error:
        print(f"GPU preflight FAILED: {error}", file=sys.stderr)
        return 1

    for result in results:
        print(result)
    print("GPU preflight PASSED — both CUDA operations executed and returned expected values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
