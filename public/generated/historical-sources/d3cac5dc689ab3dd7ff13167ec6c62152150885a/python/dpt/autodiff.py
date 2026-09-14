"""First-order tape ownership for composed deterministic operators.

A tape is an execution lifetime, not an image Jacobian. This small owner makes
output seeding, error checkpoints and discard/reset explicit. Scientific
operators still receive the underlying tape through their tape= argument.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from dpt._runtime import load_warp, prepare_context, require_no_tape
from dpt.contracts import ContractError


class CheckedWorkspace(Protocol):
    def clear_status(self) -> None: ...
    def check_status(self) -> None: ...


# region book:autodiff-first-order-lifetime
@dataclass(slots=True)
class FirstOrderPass:
    """Own one forward/reverse evaluation and its explicit diagnostic checkpoint.

    Use as a context manager; pass .tape into canonical forward operators.
    Call backward after exiting the context. Keep original inputs immutable
    until close(). A new iteration requires a fresh context entry after close.
    """

    workspaces: Sequence[CheckedWorkspace]
    tape: Any = field(default=None, init=False, repr=False)
    _state: str = field(default="new", init=False)

    def __enter__(self) -> FirstOrderPass:
        if self._state != "new":
            raise ContractError("a FirstOrderPass owns exactly one evaluation")
        require_no_tape()
        for workspace in self.workspaces:
            workspace.clear_status()
        self.tape = load_warp().Tape()
        self.tape.__enter__()
        self._state = "recording"
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.tape.__exit__(exc_type, exc_value, traceback)
        self._state = "recorded"
        if exc_type is not None:
            self.close()

    def backward(self, output: Any, seed: Any) -> None:
        """Seed a caller-owned output cotangent and complete numerical checks."""
        if self._state != "recorded":
            raise ContractError("backward requires one completed, unconsumed forward pass")
        try:
            wp = load_warp()
            if not isinstance(output, wp.array) or output.grad is None:
                raise ContractError("output must be a Warp CUDA array with a preallocated gradient")
            ctx = prepare_context(device=output.device)
            ctx.array(output, "output", dtype=output.dtype, ndim=output.ndim)
            ctx.array(seed, "seed", dtype=output.dtype, shape=tuple(output.shape), ndim=output.ndim)
            self.tape.backward(grads={output: seed})
            for workspace in self.workspaces:
                workspace.check_status()
            self._state = "complete"
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Clear accumulated cotangents and release every abandoned recording."""
        if self._state == "recording":
            raise ContractError("exit the recording context before closing it")
        if self.tape is not None and self._state != "closed":
            self.tape.zero()
            self.tape.reset()
            for workspace in self.workspaces:
                discard = getattr(workspace, "discard_recording", None)
                if discard is not None:
                    discard(self.tape)
        self._state = "closed"


# endregion book:autodiff-first-order-lifetime
