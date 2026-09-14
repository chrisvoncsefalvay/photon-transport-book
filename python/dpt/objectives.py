"""Image objectives with an explicit measurement domain and image-space seed.

Poisson half-deviance accepts nonnegative integer observations and nonnegative
count means; a zero mean with a positive count is rejected. Squared error accepts
finite values in the declared domain. Weights are fixed nonnegative per-pixel
coefficients. A mean reduction divides by pixel count, not by the sum of weights.
An explicit uint8 validity mask skips invalid observation lanes before reading
their target or forming residuals, with exact zero loss/seed. It is fixed data,
not a residual-dependent weight. Unmasked predictions/observations remain checked.
"""

# Workspace internals are owned by functions in this module.
# pyright: reportPrivateUsage=false
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from dpt._reductions import ReductionTree, prepare_reduction
from dpt._runtime import DeviceContext, load_kernels, prepare_context, require_no_tape
from dpt.contracts import ContractError, NumericalError, finite_scalar, integer


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    kind: Literal["squared_error", "poisson"] = "squared_error"
    domain: Literal["counts", "signal", "log_transmission"] = "signal"
    reduction: Literal["sum", "mean"] = "sum"
    weighted: bool = False
    masked: bool = False
    precision: Literal["float32", "float64"] = "float32"

    def __post_init__(self) -> None:
        if self.kind not in ("squared_error", "poisson"):
            raise ContractError("unsupported objective")
        if self.precision not in ("float32", "float64"):
            raise ContractError("objective precision must be float32 or float64")
        if self.domain not in ("counts", "signal", "log_transmission"):
            raise ContractError("declare the objective's measurement domain")
        if self.kind == "poisson" and self.domain != "counts":
            raise ContractError("a Poisson objective requires count-domain observations")
        if (
            self.reduction not in ("sum", "mean")
            or type(self.weighted) is not bool
            or type(self.masked) is not bool
        ):
            raise ContractError("invalid objective reduction or weight policy")


@dataclass(slots=True)
class ObjectiveWorkspace:
    spec: ObjectiveSpec
    max_pixels: int
    context: DeviceContext
    _kernels: Any = field(repr=False)
    _empty: Any = field(repr=False)
    _empty_seed: Any = field(repr=False)
    _empty_valid: Any = field(repr=False)
    _status: Any = field(repr=False)
    _reduction: ReductionTree = field(repr=False)

    @property
    def scratch_bytes(self) -> int:
        return 4 + self._reduction.scratch_bytes

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()

    def check_status(self) -> None:
        """Synchronise at an explicit acceptance boundary; failed outputs are unusable."""
        self.context.wp.synchronize_stream(self.context.stream)
        code = int(self._status.numpy()[0])
        if code & 1:
            raise ContractError("invalid prediction, observation or weight")
        if code & 2:
            raise NumericalError("objective or image-space seed is not representable")

    def validate_observation(
        self, observation: Any, *, weights: Any = None, valid: Any = None, stream: Any = None
    ) -> None:
        """Check fixed observations and optional weights without a prediction."""
        require_no_tape()
        ctx = self.context
        ctx.assert_stream(stream)
        size = ctx.array(observation, "observation", dtype=ctx.wp.float32)
        if size > self.max_pixels or (not size and self.spec.reduction == "mean"):
            raise ContractError("observation size is outside the objective capacity/domain")
        if self.spec.weighted:
            ctx.array(weights, "weights", dtype=ctx.wp.float32, shape=(size,))
        elif weights is not None:
            raise ContractError("weights require a weighted objective specification")
        mask = _validity(self, valid, size)
        with ctx.scope():
            self.clear_status()
            if size:
                ctx.wp.launch(
                    self._kernels.observation_validation_kernel(
                        self.spec.kind == "poisson", self.spec.weighted, self.spec.masked
                    ),
                    dim=size,
                    inputs=[observation, weights if self.spec.weighted else self._empty, mask],
                    outputs=[self._status],
                    stream=ctx.stream,
                    record_tape=False,
                )
            self.check_status()


def prepare_objective(
    spec: ObjectiveSpec,
    *,
    max_pixels: int,
    device: str = "cuda:0",
    stream: Any = None,
) -> ObjectiveWorkspace:
    capacity = integer(max_pixels, "max_pixels")
    context = prepare_context(device=device, stream=stream)
    wp = context.wp
    kernels = load_kernels("dpt.kernels.objectives")
    reduction = prepare_reduction(context, capacity)
    with context.scope():
        empty = wp.empty(0, dtype=wp.float32, device=context.device)
        empty_seed = wp.empty(
            0,
            dtype=wp.float64 if spec.precision == "float64" else wp.float32,
            device=context.device,
        )
        empty_valid = wp.empty(0, dtype=wp.uint8, device=context.device)
        status = wp.zeros(1, dtype=wp.int32, device=context.device)
    return ObjectiveWorkspace(
        spec, capacity, context, kernels, empty, empty_seed, empty_valid, status, reduction
    )


def _validity(workspace: ObjectiveWorkspace, valid: Any, size: int) -> Any:
    if workspace.spec.masked:
        workspace.context.array(valid, "valid", dtype=workspace.context.wp.uint8, shape=(size,))
        return valid
    if valid is not None:
        raise ContractError("valid requires a masked objective specification")
    return workspace._empty_valid


# region book:objective-public-contract
def evaluate_objective(
    prediction: Any,
    observation: Any,
    *,
    out_loss: Any,
    out_seed: Any = None,
    weights: Any = None,
    valid: Any = None,
    workspace: ObjectiveWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Write an FP64 scalar loss and optional derivative w.r.t. prediction.

    ``spec.precision`` selects prediction/seed storage; observations and fixed
    weights remain FP32. Arithmetic and loss reductions retain FP64. The
    separate fused primary interface always retains its FP64 depth boundary.

    Use the seed with the renderer's explicit VJP. This combined loss/derivative
    interface deliberately rejects ambient tapes instead of silently omitting
    the reduction's gradient. The caller owns every destination and must keep
    participating buffers alive until the stream completes. Unchecked calls are
    asynchronous; clear/check persistent status at explicit batch boundaries.
    Warm each selected output specialisation before CUDA graph capture.
    """
    ctx, spec, kernels = workspace.context, workspace.spec, workspace._kernels
    ctx.assert_stream(stream)
    require_no_tape()
    wp = ctx.wp
    value_dtype = wp.float64 if spec.precision == "float64" else wp.float32
    size = ctx.array(prediction, "prediction", dtype=value_dtype)
    if size > workspace.max_pixels:
        raise ContractError("prediction exceeds workspace capacity")
    if size == 0 and spec.reduction == "mean":
        raise ContractError("a mean objective requires at least one pixel")
    ctx.array(observation, "observation", dtype=wp.float32, shape=(size,))
    ctx.array(out_loss, "out_loss", dtype=wp.float64, shape=(1,))
    reads = [("prediction", prediction), ("observation", observation)]
    writes = [("out_loss", out_loss)]
    if spec.weighted:
        ctx.array(weights, "weights", dtype=wp.float32, shape=(size,))
        reads.append(("weights", weights))
    elif weights is not None:
        raise ContractError("weights require a weighted objective specification")
    if out_seed is not None:
        ctx.array(out_seed, "out_seed", dtype=value_dtype, shape=(size,))
        writes.append(("out_seed", out_seed))
    ctx.disjoint(reads, writes)
    weight_array = weights if spec.weighted else workspace._empty
    validity = _validity(workspace, valid, size)
    if spec.masked:
        reads.append(("valid", valid))
        ctx.disjoint(reads, writes)
    seed = out_seed if out_seed is not None else workspace._empty_seed
    poisson = spec.kind == "poisson"
    with ctx.scope():
        if validate:
            workspace._status.zero_()
            if size:
                wp.launch(
                    kernels.validation_kernel(poisson, spec.weighted, spec.masked, spec.precision),
                    dim=size,
                    inputs=[prediction, observation, weight_array, validity, workspace._status],
                    device=ctx.device,
                    stream=ctx.stream,
                    record_tape=False,
                )
            workspace.check_status()
        count = (size + 255) // 256
        if size:
            normalisation = 1.0 / size if spec.reduction == "mean" else 1.0
            wp.launch_tiled(
                kernels.evaluation_kernel(
                    poisson, spec.weighted, out_seed is not None, spec.masked, spec.precision
                ),
                dim=count,
                block_dim=256,
                inputs=[
                    prediction,
                    observation,
                    weight_array,
                    validity,
                    size,
                    normalisation,
                    seed,
                    workspace._reduction.partials[0],
                    workspace._status,
                ],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        workspace._reduction.finish(count, out_loss, workspace._status)
        if validate:
            workspace.check_status()


# endregion book:objective-public-contract


def evaluate_primary_objective(
    optical_depth: Any,
    observation: Any,
    *,
    open_beam: float = 1.0,
    out_prediction: Any,
    out_loss: Any,
    out_depth_seed: Any,
    weights: Any = None,
    valid: Any = None,
    workspace: ObjectiveWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Compose primary transmission and its objective without FP32 intermediates.

    Depth, prediction and depth cotangent use caller-owned FP64 arrays. Fixed
    observations/weights remain FP32. The field discretisation, objective and
    reduction are unchanged; the explicit depth cotangent is N-lambda for a
    Poisson term. This is a first-order composition, not a new measurement model.
    A positive observation with a numerically zero mean still fails explicitly.
    """
    require_no_tape()
    ctx, spec, wp = workspace.context, workspace.spec, workspace.context.wp
    ctx.assert_stream(stream)
    if spec.domain not in ("counts", "log_transmission"):
        raise ContractError("primary composition requires counts or log transmission")
    beam = finite_scalar(open_beam, "open_beam", minimum=0.0)
    size = ctx.array(optical_depth, "optical_depth", dtype=wp.float64)
    if size > workspace.max_pixels or (not size and spec.reduction == "mean"):
        raise ContractError("primary image is outside the objective capacity/domain")
    ctx.array(observation, "observation", dtype=wp.float32, shape=(size,))
    ctx.array(out_prediction, "out_prediction", dtype=wp.float64, shape=(size,))
    ctx.array(out_depth_seed, "out_depth_seed", dtype=wp.float64, shape=(size,))
    ctx.array(out_loss, "out_loss", dtype=wp.float64, shape=(1,))
    reads = [("optical_depth", optical_depth), ("observation", observation)]
    if spec.weighted:
        ctx.array(weights, "weights", dtype=wp.float32, shape=(size,))
        reads.append(("weights", weights))
    elif weights is not None:
        raise ContractError("weights require a weighted objective specification")
    mask = _validity(workspace, valid, size)
    if spec.masked:
        reads.append(("valid", valid))
    ctx.disjoint(
        reads, [("prediction", out_prediction), ("depth seed", out_depth_seed), ("loss", out_loss)]
    )
    with ctx.scope():
        if validate:
            workspace.validate_observation(observation, weights=weights, valid=valid, stream=stream)
        count = (size + 255) // 256
        if size:
            wp.launch_tiled(
                workspace._kernels.primary_evaluation_kernel(
                    spec.kind == "poisson", spec.domain == "counts", spec.weighted, spec.masked
                ),
                dim=count,
                block_dim=256,
                inputs=[
                    optical_depth,
                    beam,
                    observation,
                    weights if spec.weighted else workspace._empty,
                    mask,
                    size,
                    1.0 / size if spec.reduction == "mean" else 1.0,
                ],
                outputs=[
                    out_prediction,
                    out_depth_seed,
                    workspace._reduction.partials[0],
                    workspace._status,
                ],
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
        workspace._reduction.finish(count, out_loss, workspace._status)
        if validate:
            workspace.check_status()


def reduce_objective_components(
    components: Any,
    *,
    out_loss: Any,
    workspace: ObjectiveWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Sum caller-produced FP64 components without re-evaluating a loss.

    This lets independently constructed stochastic objective/gradient estimators
    use the same scalar reduction storage. Components must already include any
    weights or normalisation. Warm this reduction before CUDA graph capture.
    An empty sum is zero; no per-component storage is allocated here.
    """
    ctx, wp = workspace.context, workspace.context.wp
    ctx.assert_stream(stream)
    require_no_tape()
    if workspace.spec.reduction != "sum" or workspace.spec.weighted or workspace.spec.masked:
        raise ContractError("component reduction requires an unweighted sum workspace")
    size = ctx.array(components, "components", dtype=wp.float64)
    if size > workspace.max_pixels:
        raise ContractError("components exceed reduction capacity")
    ctx.array(out_loss, "out_loss", dtype=wp.float64, shape=(1,))
    ctx.disjoint([("components", components)], [("out_loss", out_loss)])
    with ctx.scope():
        if validate:
            workspace.clear_status()
        workspace._reduction.finish(size, out_loss, workspace._status, source=components)
        if validate:
            workspace.check_status()
