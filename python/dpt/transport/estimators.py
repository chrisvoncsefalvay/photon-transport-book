"""Original-history uncertainty and independent products for stochastic inversion.

Detector misses count as zero histories. Photon branches are not independent
samples, and plug-in squared sample means are not unbiased squared expectations.
The product estimators below explicitly require disjoint random-stream identities.
"""

from __future__ import annotations

# Workspace internals are shared only within this package.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, field
from typing import Any

from dpt._runtime import require_no_tape
from dpt.statistics import mean_standard_error

from .forward import TransportWorkspace
from .model import TransportError
from .rng import HistoryBatch, require_independent

_NO_VARIANCE = object()


@dataclass(slots=True)
class EstimatorWorkspace:
    """Reusable detector moments; no per-call allocations or hidden downloads."""

    transport: TransportWorkspace
    _sum: Any = field(repr=False)
    _hits: Any = field(repr=False)
    _scale: Any = field(repr=False)

    @property
    def scratch_bytes(self) -> int:
        return 20 * self.transport.spec.detector.pixels


def prepare_estimators(workspace: TransportWorkspace) -> EstimatorWorkspace:
    """Allocate binary64 sum/scale images and int32 hit counts on the owning stream."""
    require_no_tape()
    context = workspace.context
    with context.scope():
        sums = context.wp.empty(
            workspace.spec.detector.pixels, dtype=context.wp.float64, device=context.device
        )
        hits = context.wp.empty(
            workspace.spec.detector.pixels, dtype=context.wp.int32, device=context.device
        )
        scale = context.wp.empty_like(sums)
    return EstimatorWorkspace(workspace, sums, hits, scale)


def _accumulate_history_mean(
    pixel: Any,
    score: Any,
    history_status: Any,
    *,
    batch: HistoryBatch,
    workspace: EstimatorWorkspace,
    out_mean: Any,
    out_variance_of_mean: Any = _NO_VARIANCE,
    stream: Any,
    validate: bool,
) -> None:
    """Validate shared ownership, clear scratch and launch scaled original-history means."""
    transport = workspace.transport
    context = transport.context
    context.assert_stream(stream)
    require_no_tape()
    wp = context.wp
    pixels = transport.spec.detector.pixels
    for name, value, dtype in (
        ("pixel", pixel, wp.int32),
        ("score", score, wp.float64),
        ("history_status", history_status, wp.int32),
    ):
        context.array(value, name, dtype=dtype, shape=(batch.count,))
    outputs = [("out_mean", out_mean)]
    scratch = [("moment_sum", workspace._sum), ("moment_scale", workspace._scale)]
    if out_variance_of_mean is not _NO_VARIANCE:
        outputs.append(("out_variance_of_mean", out_variance_of_mean))
        scratch.append(("moment_hits", workspace._hits))
    for name, value in outputs:
        context.array(value, name, dtype=wp.float64, shape=(pixels,))
    context.disjoint(
        [
            *transport._reads(),
            ("pixel", pixel),
            ("score", score),
            ("history_status", history_status),
        ],
        outputs + scratch,
    )
    if validate:
        transport.check_status()
    with context.scope():
        workspace._sum.zero_()
        workspace._scale.zero_()
        if out_variance_of_mean is not _NO_VARIANCE:
            workspace._hits.zero_()
            out_variance_of_mean.zero_()
    transport._launch(
        transport._kernels.find_tally_scale,
        batch.count,
        [pixel, score, history_status, pixels, workspace._scale, transport._status],
    )
    transport._launch(
        transport._kernels.tally_sum,
        batch.count,
        [pixel, score, history_status, pixels, workspace._scale, workspace._sum, transport._status],
    )
    transport._launch(
        transport._kernels.finish_mean,
        pixels,
        [workspace._sum, workspace._scale, batch.count, out_mean, transport._status],
    )


# region book:transport-original-history-moments
def history_moments(
    pixel: Any,
    score: Any,
    history_status: Any,
    *,
    batch: HistoryBatch,
    workspace: EstimatorWorkspace,
    out_mean: Any,
    out_variance_of_mean: Any,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Mean and unbiased variance of that mean, for IID original-history scores.

    Source histories must be independently identically distributed; a repeated
    deterministic ray also qualifies. This sampling-law precondition cannot be
    verified from the realised score arrays.
    Deterministically stratified sources require variance across independent
    *replicated complete batches*, not this within-history formula. The two
    output images contain marginal variances; they do not claim independent
    detector pixels or supply a covariance-free loss-error bar.

    FP64 scale, mean and centred-deviation passes preserve the result range. Their
    floating summation order is not bitwise reproducible. This analogue engine
    produces at most one contribution per original history; descendant splitting
    must first combine contributions before using these moment operations.
    """
    if batch.count < 2:
        raise TransportError("variance of a mean requires at least two original histories")
    _accumulate_history_mean(
        pixel,
        score,
        history_status,
        batch=batch,
        workspace=workspace,
        out_mean=out_mean,
        out_variance_of_mean=out_variance_of_mean,
        stream=stream,
        validate=validate,
    )
    transport = workspace.transport
    pixels = transport.spec.detector.pixels
    transport._launch(
        transport._kernels.centred_moments,
        batch.count,
        [
            pixel,
            score,
            out_mean,
            workspace._scale,
            out_variance_of_mean,
            workspace._hits,
            transport._status,
        ],
    )
    transport._launch(
        transport._kernels.finish_variance,
        pixels,
        [
            out_mean,
            workspace._scale,
            workspace._hits,
            batch.count,
            out_variance_of_mean,
            transport._status,
        ],
    )
    if validate:
        transport.check_status()


# endregion book:transport-original-history-moments


def _product(
    mean_a: Any,
    other_b: Any,
    observed: Any,
    weights: Any,
    *,
    batches: tuple[HistoryBatch, HistoryBatch],
    workspace: TransportWorkspace,
    out_components: Any,
    loss: bool,
    stream: Any,
    validate: bool,
) -> None:
    require_independent(*batches)
    require_no_tape()
    context = workspace.context
    context.assert_stream(stream)
    arrays = [
        ("mean_a", mean_a),
        ("other_b", other_b),
        ("observed", observed),
        ("weights", weights),
    ]
    shape = (workspace.spec.detector.pixels,)
    for name, value in [*arrays, ("out_components", out_components)]:
        context.array(value, name, dtype=context.wp.float64, shape=shape)
    context.disjoint(workspace._reads() + arrays, [("out_components", out_components)])
    if validate:
        workspace.check_status()
    workspace._launch(
        workspace._kernels.independent_product,
        shape[0],
        [mean_a, other_b, observed, weights, int(loss), out_components, workspace._status],
    )
    if validate:
        workspace.check_status()


# region book:transport-independent-loss-gradient
def independent_squared_gradient(
    mean_a: Any,
    derivative_mean_b: Any,
    observed: Any,
    weights: Any,
    *,
    batches: tuple[HistoryBatch, HistoryBatch],
    workspace: TransportWorkspace,
    out_components: Any,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Write unbiased per-pixel contributions to the loss-of-expected-score gradient.

    For fixed finite observation y and nonnegative fixed pixel weight w, the
    target is 1/2 sum_p w_p (E[S_p]-y_p)^2. Independent unbiased estimates A of
    E[S] and B of its derivative give E[w(A-y)B] = w(E[S]-y)dE[S]. Using the same
    histories in both factors generally adds their covariance and is rejected.
    Sum components on device using the shared reduction operator. Estimate the
    scalar gradient's uncertainty across independent product replicates, because
    pixel covariances generally do not vanish.
    """
    _product(
        mean_a,
        derivative_mean_b,
        observed,
        weights,
        batches=batches,
        workspace=workspace,
        out_components=out_components,
        loss=False,
        stream=stream,
        validate=validate,
    )


# endregion book:transport-independent-loss-gradient


def independent_squared_loss(
    mean_a: Any,
    mean_b: Any,
    observed: Any,
    weights: Any,
    *,
    batches: tuple[HistoryBatch, HistoryBatch],
    workspace: TransportWorkspace,
    out_components: Any,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Write unbiased loss components from two independent score estimates.

    Individual estimates may be negative even though the expected squared-error
    loss is nonnegative. Clamping a noisy estimate to zero would introduce bias.
    A candidate-minus-incumbent difference may use common random numbers between
    the two parameter values, provided the two product factors remain independent.
    Independent replicate differences provide the acceptance-policy uncertainty.
    """
    _product(
        mean_a,
        mean_b,
        observed,
        weights,
        batches=batches,
        workspace=workspace,
        out_components=out_components,
        loss=True,
        stream=stream,
        validate=validate,
    )


@dataclass(frozen=True, slots=True)
class ReplicateEstimate:
    """A scalar checkpoint, after device reduction and explicit host transfer."""

    value: float
    batches: tuple[HistoryBatch, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or not self.batches:
            raise TransportError(
                "replicate estimates need a finite value and their batch identities"
            )
        require_independent(*self.batches)


@dataclass(frozen=True, slots=True)
class MeanUncertainty:
    mean: float
    standard_error: float
    replicates: int


def summarise_replicates(estimates: tuple[ReplicateEstimate, ...]) -> MeanUncertainty:
    """Estimate uncertainty across genuinely independent complete scalar replicates.

    The caller may pass paired candidate-minus-incumbent objective changes as
    values. Within each pair, common random numbers are permitted; record each
    *distinct* batch once. Across replicate pairs no streams may overlap.
    """
    if len(estimates) < 2:
        raise TransportError("uncertainty requires at least two independent replicates")
    for index, estimate in enumerate(estimates):
        for other in estimates[index + 1 :]:
            require_independent(*estimate.batches, *other.batches)
    count = len(estimates)
    mean, standard_error = mean_standard_error(tuple(estimate.value for estimate in estimates))
    return MeanUncertainty(mean, standard_error, count)


def history_mean(
    pixel: Any,
    score: Any,
    history_status: Any,
    *,
    batch: HistoryBatch,
    workspace: EstimatorWorkspace,
    out_mean: Any,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Accumulate a mean without inventing a within-batch uncertainty estimate.

    This supports single-history batches and independent replicated batch designs.
    It avoids both the extra squared-score atomic and avoidable score-square
    overflow when only the mean is requested by a nonlinear product estimator.
    """
    _accumulate_history_mean(
        pixel,
        score,
        history_status,
        batch=batch,
        workspace=workspace,
        out_mean=out_mean,
        stream=stream,
        validate=validate,
    )
    if validate:
        workspace.transport.check_status()
