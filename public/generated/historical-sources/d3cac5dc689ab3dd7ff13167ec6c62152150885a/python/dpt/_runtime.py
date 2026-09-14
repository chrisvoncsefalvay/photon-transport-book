"""Narrow Warp boundary for new operators; importing this file is CPU-safe.

Every workspace binds to one stream. Context methods validate metadata only;
device-content checks and their synchronisation belong to the calling operator.
There are no device allocations or implicit copies in these helpers.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from dpt.contracts import ContractError


def load_warp() -> Any:
    try:
        return importlib.import_module("warp")
    except ModuleNotFoundError as error:
        raise RuntimeError("This operator requires the optional gpu dependency group") from error


def load_kernels(module: str) -> Any:
    return importlib.import_module(module)


@dataclass(frozen=True, slots=True)
class DeviceContext:
    """CUDA device and stream identity retained for a workspace's entire lifetime."""

    wp: Any = field(repr=False)
    device: Any
    stream: Any
    contract_error: type[ContractError] = field(default=ContractError, repr=False)

    def assert_stream(self, stream: Any = None) -> Any:
        selected = self.wp.get_stream(self.device) if stream is None else stream
        if selected.device != self.device or selected.cuda_stream != self.stream.cuda_stream:
            raise self.contract_error("use the workspace's owning stream on its CUDA device")
        return selected

    def scope(self) -> Any:
        """Enter the owning stream without inserting a cross-stream dependency."""
        return self.wp.ScopedStream(self.stream, sync_enter=False)

    def array(
        self,
        value: Any,
        name: str,
        *,
        dtype: Any,
        shape: tuple[int, ...] | None = None,
        ndim: int = 1,
    ) -> int:
        if not isinstance(value, self.wp.array):
            raise self.contract_error(f"{name} must be a Warp CUDA array")
        if value.device != self.device or value.dtype != dtype:
            precision = "binary32" if dtype == self.wp.float32 else str(dtype)
            raise self.contract_error(
                f"{name} has the wrong device or dtype; expected {precision} on {self.device}"
            )
        if not value.is_contiguous or value.ndim != ndim:
            dimensions = "one-dimensional" if ndim == 1 else f"{ndim}-dimensional"
            raise self.contract_error(f"{name} must be contiguous and {dimensions}")
        if shape is not None and tuple(value.shape) != shape:
            raise self.contract_error(f"{name} has shape {value.shape}; expected {shape}")
        return int(value.size)

    def disjoint(self, reads: list[tuple[str, Any]], writes: list[tuple[str, Any]]) -> None:
        """Reject byte-range overlap, including views and unequal element widths."""

        def bounds(array: Any) -> tuple[int, int]:
            start = int(array.ptr)
            return start, start + int(array.size) * self.wp.types.type_size_in_bytes(array.dtype)

        for index, (name, destination) in enumerate(writes):
            if not destination.size:
                continue
            lo, hi = bounds(destination)
            for other, source in reads + writes[:index]:
                if not source.size:
                    continue
                source_lo, source_hi = bounds(source)
                if lo < source_hi and source_lo < hi:
                    raise self.contract_error(f"{name} overlaps {other}")


def prepare_context(
    *,
    device: Any = "cuda:0",
    stream: Any = None,
    contract_error: type[ContractError] = ContractError,
) -> DeviceContext:
    """Resolve CUDA ownership during preparation, before repeated evaluation."""
    wp = load_warp()
    dev = wp.get_device(device)
    if not dev.is_cuda:
        raise contract_error("scientific production operators require a CUDA device")
    selected = wp.get_stream(dev) if stream is None else stream
    if selected.device != dev:
        raise contract_error("stream and device differ")
    return DeviceContext(wp, dev, selected, contract_error)


def active_tape() -> Any:
    # Warp 1.17 has no public ambient-tape accessor. Keep this compatibility
    # dependency isolated; explicit tape identity prevents omitted derivatives.
    runtime = importlib.import_module("warp._src.context").runtime
    return None if runtime is None else runtime.tape


def require_no_tape(*, contract_error: type[ContractError] = ContractError) -> None:
    if active_tape() is not None:
        raise contract_error(
            "this operation does not support recording on an ambient tape "
            "(including higher-order recording)"
        )


def ensure_tape(tape: Any, *, contract_error: type[ContractError] = ContractError) -> None:
    active = active_tape()
    if active is not None and active is not tape:
        raise contract_error("pass the active Warp tape explicitly")
