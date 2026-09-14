"""CUDA transmission with explicit storage, numerical and differentiation contracts.

Importing this module does not initialise CUDA or import Warp. Checked calls
validate device contents; ``validate=False`` is the asynchronous integration
boundary for callers that guarantee the current inputs satisfy these contracts.
No output arrays are allocated by an evaluation. See ``TransmissionWorkspace``
for stream ownership and checked completion of custom tape gradients.
"""

from __future__ import annotations

# Scientific API names mirror the chapter symbols; private scratch is module-owned.
# ruff: noqa: N803, N815
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from dpt._reductions import ReductionTree, prepare_reduction
from dpt._runtime import DeviceContext, ensure_tape, load_kernels, prepare_context, require_no_tape
from dpt.contracts import ContractError, NumericalError, binary32_scalar

BeamMode = Literal["none", "scalar", "device-scalar", "per-pixel"]
_BEAM_MODES: dict[str, int] = {"none": 0, "scalar": 1, "device-scalar": 2, "per-pixel": 3}
_LIMIT = 2**31 - 1
_TILE = 256


class TransmissionError(ContractError):
    """Input, layout, aliasing or ownership violates the operator contract."""


class GradientRangeError(NumericalError):
    """A requested gradient cannot be represented in its binary32 destination."""


@dataclass(frozen=True, slots=True)
class TransmissionSpec:
    """Compile-independent input policy; the beam scalar's value remains dynamic."""

    beam: BeamMode = "none"
    active_L: bool = True
    active_beam: bool = False
    block_dim: int = 256

    def __post_init__(self) -> None:
        if self.beam not in _BEAM_MODES:
            raise TransmissionError(f"unsupported beam representation: {self.beam!r}")
        if self.active_beam and self.beam not in ("device-scalar", "per-pixel"):
            raise TransmissionError("an active beam must be a CUDA array, not a Python scalar")
        if self.block_dim not in (128, 256):
            raise TransmissionError("block_dim must be 128 or 256")


@dataclass(slots=True)
class TransmissionWorkspace:
    """Prepared scratch on one CUDA stream; never concurrently share a workspace.

    Inputs must stay unchanged until backward completes. Custom tape callbacks
    enqueue range checks asynchronously; call ``check_status()`` after backward
    before accepting the gradients. Use ``tape.zero()`` between independent
    reverse passes. Second derivatives through the custom callbacks are not
    supported. Workspace memory is O(P / 256) only for an active scalar beam.
    """

    spec: TransmissionSpec
    max_pixels: int
    context: DeviceContext
    _kernels: Any = field(repr=False)
    _empty: Any = field(repr=False)
    _status: Any = field(repr=False)
    _reduction: ReductionTree | None = field(repr=False)

    @property
    def device(self) -> Any:
        return self.context.device

    @property
    def stream(self) -> Any:
        return self.context.stream

    @property
    def _wp(self) -> Any:
        return self.context.wp

    @property
    def scratch_bytes(self) -> int:
        """Device scratch excluding caller-owned inputs, outputs and their adjoints."""
        return 4 + (self._reduction.scratch_bytes if self._reduction is not None else 0)

    def clear_status(self) -> None:
        """Clear accumulated diagnostics explicitly, on the owning stream."""
        with self._wp.ScopedStream(self.stream, sync_enter=False):
            self._status.zero_()

    def check_status(self) -> None:
        """Synchronise the owning stream and reject any accumulated numerical error."""
        self._wp.synchronize_stream(self.stream)
        code = int(self._status.numpy()[0])
        if code & 2:
            raise GradientRangeError("gradient overflow: discard this reverse pass")
        if code:
            raise TransmissionError("non-finite or out-of-domain device input")


def prepare_transmission(
    spec: TransmissionSpec | None = None,
    *,
    device: str = "cuda:0",
    max_pixels: int,
    stream: Any = None,
) -> TransmissionWorkspace:
    """Allocate persistent diagnostics and reduction scratch, never image outputs."""
    if isinstance(max_pixels, bool) or type(max_pixels) is not int:
        raise TransmissionError("max_pixels must be an integer")
    if not 0 <= max_pixels <= _LIMIT:
        raise TransmissionError(f"max_pixels must lie in [0, {_LIMIT}]")
    context = prepare_context(device=device, stream=stream, contract_error=TransmissionError)
    wp, kernels = context.wp, load_kernels("dpt.kernels.transmission")
    spec = spec or TransmissionSpec()
    reduction = None
    with context.scope():
        empty = wp.empty(0, dtype=wp.float32, device=context.device)
        status = wp.zeros(1, dtype=wp.int32, device=context.device)
        if spec.active_beam and spec.beam == "device-scalar":
            reduction = prepare_reduction(context, max_pixels, tile=_TILE)
    return TransmissionWorkspace(spec, max_pixels, context, kernels, empty, status, reduction)


def _array(
    value: Any, name: str, workspace: TransmissionWorkspace, length: int | None = None
) -> int:
    size = workspace.context.array(
        value,
        name,
        dtype=workspace._wp.float32,
        shape=(length,) if length is not None else None,
    )
    if size > workspace.max_pixels and length != 1:
        raise TransmissionError(f"{name} exceeds workspace capacity")
    return size


def _beam(n0: Any, count: int, workspace: TransmissionWorkspace) -> tuple[Any, float]:
    mode = workspace.spec.beam
    if mode == "none":
        if n0 is not None:
            raise TransmissionError("this workspace declares no open-beam input")
        return workspace._empty, 0.0
    if mode == "scalar":
        try:
            return workspace._empty, binary32_scalar(n0, "n0", minimum=0.0)
        except ContractError as error:
            raise TransmissionError(str(error)) from error
    _array(n0, "n0", workspace, 1 if mode == "device-scalar" else count)
    return n0, 0.0


def _values(workspace: TransmissionWorkspace, arrays: list[tuple[str, Any, str]]) -> None:
    wp, kernels = workspace._wp, workspace._kernels
    workspace.clear_status()
    for _, array, kind in arrays:
        if array.size:
            wp.launch(
                getattr(kernels, kind),
                dim=array.size,
                inputs=[array],
                outputs=[workspace._status],
                stream=workspace.stream,
                record_tape=False,
            )
    try:
        workspace.check_status()
    except TransmissionError as error:
        for name, array, kind in arrays:
            for index, item in enumerate(array.numpy()):
                value = float(item)
                valid = math.isfinite(value)
                if kind != "finite_seed":
                    valid = valid and value >= 0
                if kind == "decrement_domain":
                    valid = valid and value < 1
                if not valid:
                    raise TransmissionError(f"{name}[{index}]={value!r} violates {kind}") from error
        raise


# region book:transmission-contract
def transmit(
    L: Any,
    n0: Any = None,
    *,
    out_T: Any = None,
    out_counts: Any = None,
    out_log_T: Any = None,
    out_removed: Any = None,
    workspace: TransmissionWorkspace,
    stream: Any = None,
    tape: Any = None,
    validate: bool = True,
) -> None:
    """Evaluate selected deterministic outputs in caller-owned binary32 buffers.

    L is finite, non-negative optical depth. n0 is the declared open-beam
    expectation at the detector. No sampling, clipping or implicit transfer is
    performed. The default checks device contents before writing outputs.
    With validate=False the caller guarantees valid *current* device inputs;
    launches remain asynchronous. Pass tape explicitly to record the custom
    first-order adjoint, and retain original inputs until backward completes.
    """
    # endregion book:transmission-contract
    ensure_tape(tape, contract_error=TransmissionError)
    wp, kernels = workspace._wp, workspace._kernels
    selected = workspace.context.assert_stream(stream)
    count = _array(L, "L", workspace)
    beam, scalar = _beam(n0, count, workspace)
    outputs = (out_T, out_counts, out_log_T, out_removed)
    mask = sum(1 << i for i, value in enumerate(outputs) if value is not None)
    if mask == 0:
        raise TransmissionError("request at least one output")
    if out_counts is not None and workspace.spec.beam == "none":
        raise TransmissionError("counts require an explicit open-beam input")
    writes: list[tuple[str, Any]] = []
    for name, value in zip(
        ("out_T", "out_counts", "out_log_T", "out_removed"), outputs, strict=True
    ):
        if value is not None:
            _array(value, name, workspace, count)
            writes.append((name, value))
    workspace.context.disjoint([("L", L), ("n0", beam)], writes)
    arrays = _tape_arrays(L, beam, outputs, workspace) if tape is not None else []
    # Stream ownership was checked above; callers supply cross-stream ordering.
    with wp.ScopedStream(selected, sync_enter=False):
        if validate:
            _values(workspace, [("L", L, "finite_nonnegative"), ("n0", beam, "finite_nonnegative")])
        if count:
            wp.launch(
                kernels.get_forward_kernel(mask, _BEAM_MODES[workspace.spec.beam]),
                dim=count,
                inputs=[L, beam, scalar],
                outputs=[value if value is not None else workspace._empty for value in outputs],
                stream=selected,
                block_dim=workspace.spec.block_dim,
                record_tape=False,
            )
        if tape is not None:
            _record_transmission(tape, L, n0, beam, outputs, workspace, arrays)


def _tape_arrays(
    L: Any, beam: Any, outputs: tuple[Any, ...], workspace: TransmissionWorkspace
) -> list[Any]:
    arrays = [value for value in outputs if value is not None]
    if workspace.spec.active_L:
        arrays.append(L)
    if workspace.spec.active_beam:
        arrays.append(beam)
    if not workspace.spec.active_L and not workspace.spec.active_beam:
        raise TransmissionError("recording requires at least one active input")
    for array in arrays:
        if array.grad is None:
            raise TransmissionError("allocate participating tape arrays with requires_grad=True")
    workspace.context.disjoint(
        [("forward input", L), ("beam", beam)]
        + [("forward output", a) for a in outputs if a is not None],
        [("gradient", a.grad) for a in arrays],
    )
    return arrays


def _dependencies(
    tape: Any, reads: tuple[Any, Any], outputs: tuple[Any, ...], workspace: TransmissionWorkspace
) -> None:
    if not workspace._wp.config.verify_autograd_array_access:
        return
    # Public access markers diagnose recorded writes before backward. A zero-dim
    # record gives Tape.reset()/backward() ownership of fixed inputs and view
    # parents too, without allocating gradients or launching a marker thread.
    for value in outputs:
        if value is not None:
            value.mark_write()
    for value in reads:
        value.mark_read()
    padded = [value if value is not None else workspace._empty for value in outputs]
    padded += [workspace._empty] * (4 - len(padded))
    tape.record_launch(
        workspace._kernels.dependency_marker,
        dim=0,
        max_blocks=0,
        inputs=list(reads),
        outputs=padded,
        device=workspace.device,
        block_dim=workspace.spec.block_dim,
    )


def _record_transmission(
    tape: Any,
    L: Any,
    n0: Any,
    beam: Any,
    outputs: tuple[Any, ...],
    workspace: TransmissionWorkspace,
    arrays: list[Any],
) -> None:
    wp = workspace._wp
    _dependencies(tape, (L, beam), outputs, workspace)

    def backward() -> None:
        workspace.context.assert_stream()
        transmission_vjp(
            L,
            n0,
            seed_T=outputs[0].grad if outputs[0] is not None else None,
            seed_counts=outputs[1].grad if outputs[1] is not None else None,
            seed_log_T=outputs[2].grad if outputs[2] is not None else None,
            seed_removed=outputs[3].grad if outputs[3] is not None else None,
            out_grad_L=L.grad if workspace.spec.active_L else None,
            out_grad_n0=beam.grad if workspace.spec.active_beam else None,
            workspace=workspace,
            validate=False,
            _accumulate=True,
        )
        with wp.ScopedStream(workspace.stream, sync_enter=False):
            for value in outputs:
                if value is not None and not value.retain_grad:
                    value.grad.zero_()

    tape.record_func(backward, arrays)


def transmission_vjp(
    L: Any,
    n0: Any = None,
    *,
    seed_T: Any = None,
    seed_counts: Any = None,
    seed_log_T: Any = None,
    seed_removed: Any = None,
    out_grad_L: Any = None,
    out_grad_n0: Any = None,
    workspace: TransmissionWorkspace,
    stream: Any = None,
    validate: bool = True,
    _accumulate: bool = False,
) -> None:
    """Overwrite requested first-order VJPs; caller cotangents are preserved.

    Missing seeds are zero. Active scalar illumination is reduced in a fixed
    FP64 tree. validate=False defers range-error reporting to check_status().
    _accumulate is reserved for the tape adapter, not the standalone API.
    """
    require_no_tape(contract_error=TransmissionError)
    wp, kernels = workspace._wp, workspace._kernels
    selected = workspace.context.assert_stream(stream)
    count = _array(L, "L", workspace)
    beam, scalar = _beam(n0, count, workspace)
    seeds = (seed_T, seed_counts, seed_log_T, seed_removed)
    mask = sum(1 << i for i, value in enumerate(seeds) if value is not None)
    if seed_counts is not None and workspace.spec.beam == "none":
        raise TransmissionError("a count cotangent requires an open-beam input")
    reads = [("L", L), ("n0", beam)]
    values = [("L", L, "finite_nonnegative"), ("n0", beam, "finite_nonnegative")]
    for name, seed in zip(
        ("seed_T", "seed_counts", "seed_log_T", "seed_removed"), seeds, strict=True
    ):
        if seed is not None:
            _array(seed, name, workspace, count)
            reads.append((name, seed))
            values.append((name, seed, "finite_seed"))
    writes: list[tuple[str, Any]] = []
    if out_grad_L is not None:
        if not workspace.spec.active_L:
            raise TransmissionError("L is declared fixed")
        _array(out_grad_L, "out_grad_L", workspace, count)
        writes.append(("out_grad_L", out_grad_L))
    if out_grad_n0 is not None:
        if not workspace.spec.active_beam:
            raise TransmissionError("n0 is declared fixed")
        _array(
            out_grad_n0,
            "out_grad_n0",
            workspace,
            1 if workspace.spec.beam == "device-scalar" else count,
        )
        writes.append(("out_grad_n0", out_grad_n0))
    if not writes:
        raise TransmissionError("request at least one active input gradient")
    workspace.context.disjoint(reads, writes)
    # Stream ownership was checked above; callers supply cross-stream ordering.
    with wp.ScopedStream(selected, sync_enter=False):
        if validate:
            _values(workspace, values)
        per_pixel_beam = out_grad_n0 is not None and workspace.spec.beam == "per-pixel"
        if count and (out_grad_L is not None or per_pixel_beam):
            wp.launch(
                kernels.get_vjp_kernel(
                    mask,
                    _BEAM_MODES[workspace.spec.beam],
                    out_grad_L is not None,
                    per_pixel_beam,
                    _accumulate,
                ),
                dim=count,
                inputs=[L, beam, scalar]
                + [v if v is not None else workspace._empty for v in seeds],
                outputs=[
                    out_grad_L if out_grad_L is not None else workspace._empty,
                    out_grad_n0 if per_pixel_beam else workspace._empty,
                    workspace._status,
                ],
                stream=selected,
                block_dim=workspace.spec.block_dim,
                record_tape=False,
            )
        if out_grad_n0 is not None and workspace.spec.beam == "device-scalar":
            if count and seed_counts is not None:
                reduction = workspace._reduction
                assert reduction is not None
                blocks = (count + _TILE - 1) // _TILE
                wp.launch_tiled(
                    kernels.get_beam_partial_kernel(),
                    dim=blocks,
                    inputs=[L, seed_counts],
                    outputs=[reduction.partials[0]],
                    block_dim=_TILE,
                    stream=selected,
                    record_tape=False,
                )
                reduction.finish(
                    blocks,
                    out_grad_n0,
                    workspace._status,
                    accumulate=_accumulate,
                )
            elif not _accumulate:
                out_grad_n0.zero_()
        if validate:
            workspace.check_status()


def optical_depth_from_removed(
    delta: Any,
    *,
    out_L: Any,
    workspace: TransmissionWorkspace,
    stream: Any = None,
    tape: Any = None,
    validate: bool = True,
) -> None:
    """Evaluate -log1p(-delta), for finite 0 <= delta < 1, without cancellation."""
    ensure_tape(tape, contract_error=TransmissionError)
    wp, kernels = workspace._wp, workspace._kernels
    selected = workspace.context.assert_stream(stream)
    count = _array(delta, "delta", workspace)
    _array(out_L, "out_L", workspace, count)
    workspace.context.disjoint([("delta", delta)], [("out_L", out_L)])
    if tape is not None:
        if delta.grad is None or out_L.grad is None:
            raise TransmissionError("inverse tape arrays require preallocated gradients")
        workspace.context.disjoint(
            [("delta", delta), ("out_L", out_L)],
            [("delta.grad", delta.grad), ("out_L.grad", out_L.grad)],
        )
    # Stream ownership was checked above; callers supply cross-stream ordering.
    with wp.ScopedStream(selected, sync_enter=False):
        if validate:
            _values(workspace, [("delta", delta, "decrement_domain")])
        if count:
            wp.launch(
                kernels.get_inverse_kernel(),
                dim=count,
                inputs=[delta],
                outputs=[out_L],
                stream=selected,
                block_dim=workspace.spec.block_dim,
                record_tape=False,
            )
        if tape is not None:
            _dependencies(tape, (delta, workspace._empty), (out_L,), workspace)

            def backward() -> None:
                workspace.context.assert_stream()
                optical_depth_from_removed_vjp(
                    delta,
                    out_L.grad,
                    out_grad_delta=delta.grad,
                    workspace=workspace,
                    validate=False,
                    _accumulate=True,
                )
                if not out_L.retain_grad:
                    out_L.grad.zero_()

            tape.record_func(backward, [delta, out_L])


def optical_depth_from_removed_vjp(
    delta: Any,
    seed_L: Any,
    *,
    out_grad_delta: Any,
    workspace: TransmissionWorkspace,
    stream: Any = None,
    validate: bool = True,
    _accumulate: bool = False,
) -> None:
    """First-order inverse-decrement VJP; preserve seeds and overwrite the result."""
    require_no_tape(contract_error=TransmissionError)
    wp, kernels = workspace._wp, workspace._kernels
    selected = workspace.context.assert_stream(stream)
    count = _array(delta, "delta", workspace)
    _array(seed_L, "seed_L", workspace, count)
    _array(out_grad_delta, "out_grad_delta", workspace, count)
    workspace.context.disjoint(
        [("delta", delta), ("seed_L", seed_L)], [("out_grad_delta", out_grad_delta)]
    )
    # Stream ownership was checked above; callers supply cross-stream ordering.
    with wp.ScopedStream(selected, sync_enter=False):
        if validate:
            _values(
                workspace, [("delta", delta, "decrement_domain"), ("seed_L", seed_L, "finite_seed")]
            )
        if count:
            wp.launch(
                kernels.get_inverse_vjp_kernel(_accumulate),
                dim=count,
                inputs=[delta, seed_L],
                outputs=[out_grad_delta, workspace._status],
                stream=selected,
                block_dim=workspace.spec.block_dim,
                record_tape=False,
            )
        if validate:
            workspace.check_status()
