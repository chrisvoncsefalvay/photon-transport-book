"""Deterministic detector operators and explicitly separate observation models.

Expected response, spatial spreading, electronic calibration and random draws
are distinct operations with caller-owned buffers. An energy-integrating mean
is not passed to a Poisson count sampler by the library. Imports are CPU-safe;
all numerical image work uses the single CUDA implementation.
"""

from __future__ import annotations

# Public Python boundaries validate runtime inputs; workspace scratch stays module-owned.
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from dpt._runtime import DeviceContext, load_kernels, prepare_context, require_no_tape
from dpt.contracts import ContractError, NumericalError, finite_scalar, finite_tuple, integer
from dpt.materials import Provenance


@dataclass(slots=True)
class DetectorWorkspace:
    """Reusable diagnostics and scalar-reduction scratch on one CUDA stream."""

    max_pixels: int
    context: DeviceContext
    _kernels: Any = field(repr=False)
    _checks: Any = field(repr=False)
    _status: Any = field(repr=False)
    _partials: Any = field(repr=False)

    @property
    def reduction_groups(self) -> int:
        """Read-only launch bound derived from the allocated scratch capacity."""
        return int(self._partials.size)

    @property
    def scratch_bytes(self) -> int:
        return 4 + 8 * int(self._partials.size)

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()

    def check_status(self) -> None:
        """Synchronise and reject a numerical failure; no failed draw is an observation."""
        self.context.wp.synchronize_stream(self.context.stream)
        code = int(self._status.numpy()[0])
        if code & 1:
            raise ContractError("detector device inputs violate their finite/domain/rate contract")
        if code & 2:
            raise NumericalError("detector output or gradient overflows its declared precision")
        if code & 4:
            raise NumericalError("Poisson draw budget exhausted; discard the entire realisation")

    def _values(self, arrays: list[tuple[Any, str]]) -> None:
        self.clear_status()
        for array, domain in arrays:
            if not array.size:
                continue
            kernel = {
                "finite": self._checks.check_finite,
                "nonnegative": self._checks.check_nonnegative,
                "positive": self._kernels.check_positive,
                "rate": self._kernels.check_rates,
            }[domain]
            self.context.wp.launch(
                kernel,
                dim=array.size,
                inputs=[array],
                outputs=[self._status],
                stream=self.context.stream,
                record_tape=False,
            )
        self.check_status()

    def _pixels(self, values: Any, name: str) -> int:
        count = self.context.array(values, name, dtype=self.context.wp.float32)
        if count > self.max_pixels:
            raise ContractError(f"{name} exceeds workspace capacity")
        return count


def prepare_detector(
    *,
    max_pixels: int,
    device: str = "cuda:0",
    stream: Any = None,
    reduction_groups: int = 64,
) -> DetectorWorkspace:
    """Prepare O(groups) device scratch; do not allocate any detector image."""
    integer(max_pixels, "max_pixels")
    integer(reduction_groups, "reduction_groups", minimum=1, maximum=256)
    context = prepare_context(device=device, stream=stream)
    kernels = load_kernels("dpt.kernels.detector")
    checks = load_kernels("dpt.kernels.spectral")
    with context.scope():
        status = context.wp.zeros(1, dtype=context.wp.int32, device=context.device)
        partials = context.wp.empty(
            reduction_groups, dtype=context.wp.float64, device=context.device
        )
    return DetectorWorkspace(max_pixels, context, kernels, checks, status, partials)


# region book:calibration-identifiability
@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    """y = gain * exposure * mean + offset, in the declared output unit.

    Gain is strictly positive; exposure is non-negative; offset is any finite
    electronic baseline. A fixed flat-field may vary by pixel. Optimised gain
    and offset are shared scalars: unconstrained correction images are excluded.
    Exposure and gain cannot both be active because their scale is unidentifiable.
    """

    output_unit: str
    shared_gain: bool = True
    shared_offset: bool = True
    active_mean: bool = True
    active_gain: bool = False
    active_exposure: bool = False
    active_offset: bool = False

    def __post_init__(self) -> None:
        if not self.output_unit.strip():
            raise ContractError("calibration output unit is required")
        for name in (
            "shared_gain",
            "shared_offset",
            "active_mean",
            "active_gain",
            "active_exposure",
            "active_offset",
        ):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f"{name} must be a boolean")
        if self.active_gain and self.active_exposure:
            raise ContractError("fix gain or exposure before fitting the other scale")
        if self.active_gain and not self.shared_gain:
            raise ContractError("a fitted gain must be a shared acquisition parameter")
        if self.active_offset and not self.shared_offset:
            raise ContractError("a fitted offset must be a shared acquisition parameter")


# endregion book:calibration-identifiability


def _calibration_inputs(
    mean: Any,
    gain: Any,
    exposure: Any,
    offset: Any,
    spec: CalibrationSpec,
    workspace: DetectorWorkspace,
) -> tuple[int, list[tuple[str, Any]]]:
    ctx = workspace.context
    pixels = workspace._pixels(mean, "mean")
    for name, array, size in (
        ("gain", gain, 1 if spec.shared_gain else pixels),
        ("exposure", exposure, 1),
        ("offset", offset, 1 if spec.shared_offset else pixels),
    ):
        ctx.array(array, name, dtype=ctx.wp.float32, shape=(size,))
    return pixels, [("mean", mean), ("gain", gain), ("exposure", exposure), ("offset", offset)]


def calibrate(
    mean: Any,
    gain: Any,
    exposure: Any,
    offset: Any,
    *,
    out_signal: Any,
    spec: CalibrationSpec,
    workspace: DetectorWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Apply deterministic calibration with one fused pixel launch and FP64 intermediates."""
    require_no_tape()
    ctx = workspace.context
    ctx.assert_stream(stream)
    pixels, reads = _calibration_inputs(mean, gain, exposure, offset, spec, workspace)
    ctx.array(out_signal, "out_signal", dtype=ctx.wp.float32, shape=(pixels,))
    ctx.disjoint(reads, [("out_signal", out_signal)])
    with ctx.scope():
        if validate:
            workspace._values(
                [
                    (mean, "nonnegative"),
                    (gain, "positive"),
                    (exposure, "nonnegative"),
                    (offset, "finite"),
                ]
            )
        if pixels:
            ctx.wp.launch(
                workspace._kernels.get_calibration(spec.shared_gain, spec.shared_offset),
                dim=pixels,
                inputs=[mean, gain, exposure, offset],
                outputs=[out_signal, workspace._status],
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.check_status()


def calibration_vjp(
    mean: Any,
    gain: Any,
    exposure: Any,
    offset: Any,
    *,
    seed: Any,
    out_grad_mean: Any = None,
    out_grad_gain: Any = None,
    out_grad_exposure: Any = None,
    out_grad_offset: Any = None,
    spec: CalibrationSpec,
    workspace: DetectorWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Overwrite first-order products; shared nuisance reductions use FP64 trees.

    Per-image cotangents use binary32. Each requested shared scalar destination
    may be binary32 or binary64. Use binary64 before a logarithmic parameter
    chart: its scale factor can recover a derivative too small for binary32.
    """
    require_no_tape()
    ctx = workspace.context
    ctx.assert_stream(stream)
    pixels, reads = _calibration_inputs(mean, gain, exposure, offset, spec, workspace)
    ctx.array(seed, "seed", dtype=ctx.wp.float32, shape=(pixels,))
    writes: list[tuple[str, Any]] = []
    for name, output, active, size in (
        ("mean", out_grad_mean, spec.active_mean, pixels),
        ("gain", out_grad_gain, spec.active_gain, 1),
        ("exposure", out_grad_exposure, spec.active_exposure, 1),
        ("offset", out_grad_offset, spec.active_offset, 1),
    ):
        if output is not None:
            if not active:
                raise ContractError(f"calibration {name} was declared fixed")
            dtype = ctx.wp.float32
            if name != "mean":
                dtype = getattr(output, "dtype", None)
                if dtype not in (ctx.wp.float32, ctx.wp.float64):
                    raise ContractError("shared calibration gradients must be binary32 or binary64")
            ctx.array(output, f"out_grad_{name}", dtype=dtype, shape=(size,))
            writes.append((f"out_grad_{name}", output))
    if not writes:
        raise ContractError("request at least one active calibration gradient")
    ctx.disjoint([*reads, ("seed", seed)], writes)
    with ctx.scope():
        if validate:
            workspace._values(
                [
                    (mean, "nonnegative"),
                    (gain, "positive"),
                    (exposure, "nonnegative"),
                    (offset, "finite"),
                    (seed, "finite"),
                ]
            )
        if pixels and out_grad_mean is not None:
            ctx.wp.launch(
                workspace._kernels.get_calibration_pixel_vjp(spec.shared_gain, True),
                dim=pixels,
                inputs=[gain, exposure, seed],
                outputs=[out_grad_mean, workspace._status],
                stream=ctx.stream,
                record_tape=False,
            )
        groups = min(workspace.reduction_groups, max(1, (pixels + 255) // 256))
        for kind, output in enumerate((out_grad_gain, out_grad_exposure, out_grad_offset)):
            if output is None:
                continue
            if not pixels:
                output.zero_()
                continue
            ctx.wp.launch_tiled(
                workspace._kernels.get_calibration_partials(spec.shared_gain, kind),
                dim=groups,
                block_dim=256,
                inputs=[mean, gain, exposure, seed, pixels, groups],
                outputs=[workspace._partials],
                stream=ctx.stream,
                record_tape=False,
            )
            ctx.wp.launch(
                workspace._checks.get_finish_shared(output.dtype == ctx.wp.float64),
                dim=1,
                inputs=[workspace._partials, groups],
                outputs=[output, workspace._status],
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.check_status()


@dataclass(frozen=True, slots=True)
class BlurSpec:
    """Finite detector response: Bx[r,c] = sum_ij h[i,j] x[r+i-a,c+j-b].

    Samples outside the detector are zero. The non-negative odd-sized kernel
    sums to at most one, representing a passive spread with optional loss.
    The edge is never renormalised; transpose therefore reverses the offsets.
    """

    height: int
    width: int
    kernel_height: int
    kernel_width: int
    weights: tuple[float, ...]
    provenance: Provenance
    boundary: Literal["zero"] = "zero"

    def __post_init__(self) -> None:
        for name in ("height", "width", "kernel_height", "kernel_width"):
            integer(getattr(self, name), name, minimum=1)
        if self.height * self.width > 2**31 - 1:
            raise ContractError("detector dimensions exceed the flat index range")
        if self.kernel_height % 2 != 1 or self.kernel_width % 2 != 1:
            raise ContractError("spatial kernel dimensions must be odd, with a unique centre")
        finite_tuple(self.weights, "blur weights")
        if len(self.weights) != self.kernel_height * self.kernel_width:
            raise ContractError("blur weights must match the declared kernel footprint")
        if math.fsum(self.weights) > 1.0:
            raise ContractError("passive blur weights must sum to at most one")
        if self.boundary != "zero":
            raise ContractError("only explicit zero-extension boundaries are supported")
        if not isinstance(self.provenance, Provenance):
            raise ContractError("spatial-response provenance is required")


@dataclass(slots=True)
class BlurWorkspace:
    spec: BlurSpec
    detector: DetectorWorkspace
    _weights: Any = field(repr=False)

    @property
    def scratch_bytes(self) -> int:
        return self.detector.scratch_bytes + 8 * int(self._weights.size)

    def clear_status(self) -> None:
        self.detector.clear_status()

    def check_status(self) -> None:
        self.detector.check_status()


def prepare_blur(spec: BlurSpec, *, device: str = "cuda:0", stream: Any = None) -> BlurWorkspace:
    """Upload the fixed stencil in binary64, preserving its declared host coefficients.

    Stencils are small immutable calibration data. Narrowing their coefficients
    to binary32 can erase a positive tail whose product with a bright input is
    representable, and can change passive total mass. Images still use binary32.
    """
    detector = prepare_detector(max_pixels=spec.height * spec.width, device=device, stream=stream)
    ctx = detector.context
    with ctx.scope():
        weights = ctx.wp.array(spec.weights, dtype=ctx.wp.float64, device=ctx.device)
    return BlurWorkspace(spec, detector, weights)


def _apply_blur(
    source: Any,
    output: Any,
    workspace: BlurWorkspace,
    transpose: bool,
    stream: Any,
    validate: bool,
) -> None:
    require_no_tape()
    spec, detector = workspace.spec, workspace.detector
    ctx = detector.context
    ctx.assert_stream(stream)
    size = spec.height * spec.width
    ctx.array(source, "source", dtype=ctx.wp.float32, shape=(size,))
    ctx.array(output, "output", dtype=ctx.wp.float32, shape=(size,))
    ctx.disjoint([("source", source), ("weights", workspace._weights)], [("output", output)])
    with ctx.scope():
        if validate:
            detector._values([(source, "finite")])
        ctx.wp.launch(
            detector._kernels.get_blur(
                spec.height, spec.width, spec.kernel_height, spec.kernel_width, transpose
            ),
            dim=size,
            inputs=[source, workspace._weights],
            outputs=[output, detector._status],
            stream=ctx.stream,
            record_tape=False,
        )
        if validate:
            detector.check_status()


def blur(
    source: Any,
    *,
    out_signal: Any,
    workspace: BlurWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Apply the declared finite spatial response; inputs may be signed real signals."""
    _apply_blur(source, out_signal, workspace, False, stream, validate)


def blur_transpose(
    seed: Any,
    *,
    out_grad_signal: Any,
    workspace: BlurWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Apply the exact algebraic transpose of the same zero-extended discrete matrix."""
    _apply_blur(seed, out_grad_signal, workspace, True, stream, validate)


# region book:observation-identity
@dataclass(frozen=True, slots=True)
class ObservationIdentity:
    """A reproducible random experiment independent of launch tiling.

    Counter identity = (observation_id << 32) | global_pixel. Poisson bin k has
    domain 1 + energy_offset + k; electronic read noise has domain 0x7fffffff.
    Transport owns domain 0 and domains >= 0x80000000. Reusing an identity is
    common random numbers, not an independent replication. Keep bin identities
    stable when batching, and assign a fresh observation_id for a fresh image.
    ``draw_budget`` bounds rejection proposals, never the sampled count. Below
    rate 10 an inverse-CDF draw uses one proposal with a fixed internal work
    bound; higher rates use at most ``draw_budget`` transformed proposals.
    """

    seed: int
    observation_id: int
    pixel_offset: int = 0
    energy_offset: int = 0
    draw_budget: int = 256

    def __post_init__(self) -> None:
        integer(self.seed, "seed", maximum=2**64 - 1)
        integer(self.observation_id, "observation_id", maximum=2**32 - 1)
        integer(self.pixel_offset, "pixel_offset", maximum=2**32 - 1)
        integer(self.energy_offset, "energy_offset", maximum=2**31 - 3)
        integer(self.draw_budget, "draw_budget", minimum=1)

    def validate_extent(self, pixels: int, energies: int = 1) -> None:
        integer(pixels, "pixels")
        integer(energies, "energies", minimum=1)
        if self.pixel_offset + pixels > 2**32:
            raise ContractError("pixel identities would wrap")
        if self.energy_offset + energies > 2**31 - 2:
            raise ContractError("Poisson domains would overlap the read-noise/transport namespace")


# endregion book:observation-identity


def sample_poisson_counts(
    expected_counts: Any,
    *,
    out_counts: Any,
    identity: ObservationIdentity,
    workspace: DetectorWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Draw uint64 independent counts from count-domain means in [0, 10⁹].

    This is an observation generator, not a differentiable forward operator.
    Its input must already be a photon-counting mean. A zero mean returns zero;
    rejection-budget exhaustion invalidates the realisation. No noise or
    non-negativity clipping is applied to calibrated electronic signals here.
    The declared upper rate is an implementation boundary, not an executed
    acceptance result; large-rate mass accuracy and sampling remain CUDA gates.
    """
    require_no_tape()
    ctx = workspace.context
    ctx.assert_stream(stream)
    pixels = workspace._pixels(expected_counts, "expected_counts")
    identity.validate_extent(pixels)
    ctx.array(out_counts, "out_counts", dtype=ctx.wp.uint64, shape=(pixels,))
    ctx.disjoint([("expected_counts", expected_counts)], [("out_counts", out_counts)])
    with ctx.scope():
        if validate:
            workspace._values([(expected_counts, "rate")])
        if pixels:
            ctx.wp.launch(
                workspace._kernels.poisson_counts,
                dim=pixels,
                inputs=[
                    expected_counts,
                    ctx.wp.uint64(identity.seed),
                    ctx.wp.uint32(identity.observation_id),
                    ctx.wp.uint32(identity.pixel_offset),
                    ctx.wp.uint32(identity.energy_offset),
                    identity.draw_budget,
                ],
                outputs=[out_counts, workspace._status],
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.check_status()


def sample_compound_poisson(
    detected_bin_means: Any,
    photon_scores: Any,
    *,
    energies: int,
    shared_scores: bool,
    out_signal: Any,
    identity: ObservationIdentity,
    workspace: DetectorWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Draw sum_k score[k,p] N[k,p], N[k,p] ~ Poisson(detected_bin_means[k,p]).

    The independent rate array is energy-major (K,P). Scores are deterministic
    output per detected photon; an arbitrary mean response does not specify this
    law. The variance before spatial spread is sum_k rate[k,p] score[k,p]².
    Apply spatial spread to this realisation, then add independent electronic
    noise if that order describes the actual detector. Energy-response variance
    within a bin requires a different, explicitly supplied observation law.
    """
    require_no_tape()
    integer(energies, "energies", minimum=1, maximum=65536)
    if type(shared_scores) is not bool:
        raise ContractError("shared_scores must be a boolean")
    ctx = workspace.context
    ctx.assert_stream(stream)
    size = ctx.array(detected_bin_means, "detected_bin_means", dtype=ctx.wp.float32)
    if size % energies:
        raise ContractError("detected bin means must have energy-major shape (K,P)")
    pixels = size // energies
    if pixels > workspace.max_pixels or size > 2**31 - 1:
        raise ContractError("detected bin means exceed the declared pixel/index capacity")
    identity.validate_extent(pixels, energies)
    ctx.array(
        photon_scores,
        "photon_scores",
        dtype=ctx.wp.float32,
        shape=(energies * (1 if shared_scores else pixels),),
    )
    ctx.array(out_signal, "out_signal", dtype=ctx.wp.float32, shape=(pixels,))
    ctx.disjoint(
        [("detected_bin_means", detected_bin_means), ("photon_scores", photon_scores)],
        [("out_signal", out_signal)],
    )
    with ctx.scope():
        if validate:
            workspace._values([(detected_bin_means, "rate"), (photon_scores, "nonnegative")])
        if pixels:
            ctx.wp.launch(
                workspace._kernels.get_compound_poisson(energies, shared_scores),
                dim=pixels,
                inputs=[
                    detected_bin_means,
                    photon_scores,
                    pixels,
                    ctx.wp.uint64(identity.seed),
                    ctx.wp.uint32(identity.observation_id),
                    ctx.wp.uint32(identity.pixel_offset),
                    ctx.wp.uint32(identity.energy_offset),
                    identity.draw_budget,
                ],
                outputs=[out_signal, workspace._status],
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.check_status()


def add_gaussian_read_noise(
    signal: Any,
    *,
    standard_deviation: float,
    out_signal: Any,
    identity: ObservationIdentity,
    workspace: DetectorWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Add independent electronic Gaussian noise in the signal's units, without clipping."""
    require_no_tape()
    sigma = finite_scalar(standard_deviation, "standard_deviation", minimum=0)
    ctx = workspace.context
    ctx.assert_stream(stream)
    pixels = workspace._pixels(signal, "signal")
    identity.validate_extent(pixels)
    ctx.array(out_signal, "out_signal", dtype=ctx.wp.float32, shape=(pixels,))
    ctx.disjoint([("signal", signal)], [("out_signal", out_signal)])
    with ctx.scope():
        if validate:
            workspace._values([(signal, "finite")])
        if pixels:
            ctx.wp.launch(
                workspace._kernels.gaussian_read_noise,
                dim=pixels,
                inputs=[
                    signal,
                    sigma,
                    ctx.wp.uint64(identity.seed),
                    ctx.wp.uint32(identity.observation_id),
                    ctx.wp.uint32(identity.pixel_offset),
                ],
                outputs=[out_signal, workspace._status],
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.check_status()
