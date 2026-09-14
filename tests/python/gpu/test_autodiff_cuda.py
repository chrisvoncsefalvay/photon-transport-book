"""Actual CUDA seed-boundary regressions; invalid seeds never reach Warp's copy."""

import importlib
from typing import Any

import pytest

from dpt.autodiff import FirstOrderPass
from dpt.contracts import ContractError
from dpt.transmission import prepare_transmission, transmit

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("kind", ["short", "int32", "float64", "cpu", "strided", "rank"])
def test_invalid_seed_closes_the_pass_without_a_partial_cotangent(kind: str) -> None:
    workspace = prepare_transmission(max_pixels=2)
    depth = wp.ones(2, dtype=wp.float32, device="cuda:0", requires_grad=True)
    output = wp.empty(2, dtype=wp.float32, device="cuda:0", requires_grad=True)
    with FirstOrderPass([workspace]) as reverse:
        transmit(depth, out_log_T=output, workspace=workspace, tape=reverse.tape)
    if kind == "strided":
        seed = wp.ones(4, dtype=wp.float32, device="cuda:0")[::2]
    else:
        shape = (1, 2) if kind == "rank" else (1 if kind == "short" else 2,)
        dtype = wp.int32 if kind == "int32" else wp.float64 if kind == "float64" else wp.float32
        seed = wp.ones(shape, dtype=dtype, device="cpu" if kind == "cpu" else "cuda:0")
    with pytest.raises(ContractError):
        reverse.backward(output, seed)
    assert depth.grad.numpy().tolist() == [0.0, 0.0]
    with pytest.raises(ContractError, match="unconsumed"):
        reverse.backward(output, wp.ones(2, dtype=wp.float32, device="cuda:0"))
    reverse.close()


def test_valid_seed_preserves_caller_values_and_completes_backward() -> None:
    workspace = prepare_transmission(max_pixels=2)
    depth = wp.ones(2, dtype=wp.float32, device="cuda:0", requires_grad=True)
    output = wp.empty(2, dtype=wp.float32, device="cuda:0", requires_grad=True)
    seed = wp.array([2.0, -3.0], dtype=wp.float32, device="cuda:0")
    with FirstOrderPass([workspace]) as reverse:
        transmit(depth, out_log_T=output, workspace=workspace, tape=reverse.tape)
    reverse.backward(output, seed)
    assert depth.grad.numpy().tolist() == [-2.0, 3.0]
    assert seed.numpy().tolist() == [2.0, -3.0]
    reverse.close()
