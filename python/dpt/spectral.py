"""GPU-resident polychromatic primary transmission from material paths.

Importing this module is CPU-safe. Explicit host data preparation precedes CUDA
uploads; paths and signals have an explicit retained precision. Reverse products
recompute energy depths and preserve original inputs, with no P-by-K tape.
The API intentionally requires explicit VJP composition and rejects ambient tape
recording. It does not differentiate material interpolation or energy nodes.
"""

from __future__ import annotations

# Public Python boundaries validate runtime inputs; workspace scratch stays module-owned.
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from dpt._runtime import DeviceContext, load_kernels, prepare_context, require_no_tape
from dpt.contracts import ContractError, NumericalError, finite_tuple, integer
from dpt.materials import Provenance


# region book:spectrum-units
@dataclass(frozen=True, slots=True)
class Spectrum:
    """A supplied open-beam spectrum at the detector, before response losses.

    Density values are photons/keV and require positive quadrature weights in
    keV. Bin values are already integrated expected photon populations and must
    not receive a second energy weight. No normalisation or inverse-square
    correction is guessed. Discrete lines may be included as integrated bins.
    """

    energies_kev: tuple[float, ...]
    values: tuple[float, ...]
    representation: Literal["density", "bin-fluence"]
    provenance: Provenance
    quadrature_weights_kev: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        finite_tuple(self.energies_kev, "energies_kev", positive=True)
        finite_tuple(self.values, "spectrum")
        if len(self.values) != len(self.energies_kev):
            raise ContractError("spectrum values and energies must have equal length")
        if any(a >= b for a, b in zip(self.energies_kev, self.energies_kev[1:], strict=False)):
            raise ContractError("spectrum energy nodes must be strictly increasing")
        if not isinstance(self.provenance, Provenance):
            raise ContractError("spectrum provenance is required")
        if self.representation == "density":
            if self.quadrature_weights_kev is None:
                raise ContractError("a density needs explicit energy quadrature weights")
            finite_tuple(self.quadrature_weights_kev, "quadrature weights", positive=True)
            if len(self.quadrature_weights_kev) != len(self.values):
                raise ContractError("one quadrature weight is required for each density value")
        elif self.representation == "bin-fluence":
            if self.quadrature_weights_kev is not None:
                raise ContractError("bin-integrated fluence must not be weighted twice")
        else:
            raise ContractError("spectrum representation must be density or bin-fluence")

    @property
    def bin_fluence(self) -> tuple[float, ...]:
        """Materialise energy-integrated host values at the explicit ingestion boundary."""
        if self.quadrature_weights_kev is None:
            return self.values
        result = tuple(
            value * weight
            for value, weight in zip(self.values, self.quadrature_weights_kev, strict=True)
        )
        if any(not math.isfinite(value) for value in result):
            raise ContractError("energy integration overflowed")
        if any(
            source > 0 and target == 0 for source, target in zip(self.values, result, strict=True)
        ):
            raise ContractError("energy integration underflowed a positive bin population")
        return result


# endregion book:spectrum-units


@dataclass(frozen=True, slots=True)
class SpectralSpec:
    """Physical array layout and parameter capabilities, fixed at preparation.

    The coefficients are total primary attenuation in mm⁻¹, with material paths
    in mm. ``input_description`` records the basis mixing/density assumption and
    detector acceptance used to turn source output into per-pixel fluence.
    Active weights are bin populations, not unconstrained normalised fractions.
    ``precision`` selects paths, means, image seeds and path gradients. Supplied
    coefficients, weights, responses and their gradients remain binary32.
    Retained binary64 rejects underflow of a nonzero exponential or intermediate
    product: later multiplication could otherwise conceal a representable tail.
    """

    materials: int
    energies: int
    coefficients_provenance: tuple[Provenance, ...]
    spectrum_provenance: Provenance
    response_provenance: Provenance
    output_unit: str
    input_description: str
    shared_weights: bool = True
    shared_response: bool = True
    active_paths: bool = True
    active_weights: bool = False
    active_response: bool = False
    active_coefficients: bool = False
    reduction_groups: int = 64
    precision: Literal["float32", "float64"] = "float32"

    def __post_init__(self) -> None:
        # Reverse keeps two M-element FP64 vectors per pixel. This bound is an
        # explicit implementation envelope, not an empirical performance claim.
        if self.precision not in ("float32", "float64"):
            raise ContractError("spectral precision must be float32 or float64")
        integer(self.materials, "materials", minimum=1, maximum=32)
        integer(self.energies, "energies", minimum=1, maximum=65536)
        integer(self.reduction_groups, "reduction_groups", minimum=1, maximum=256)
        if (
            not isinstance(self.coefficients_provenance, tuple)
            or len(self.coefficients_provenance) != self.materials
            or any(not isinstance(p, Provenance) for p in self.coefficients_provenance)
        ):
            raise ContractError("one immutable provenance record is required per material")
        if not isinstance(self.spectrum_provenance, Provenance):
            raise ContractError("spectrum provenance is required")
        if not isinstance(self.response_provenance, Provenance):
            raise ContractError("response provenance is required")
        if not self.output_unit.strip() or not self.input_description.strip():
            raise ContractError("spectral output units and input assumptions must be recorded")
        for name in (
            "shared_weights",
            "shared_response",
            "active_paths",
            "active_weights",
            "active_response",
            "active_coefficients",
        ):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f"{name} must be a boolean")


@dataclass(slots=True)
class SpectralWorkspace:
    """One stream, persistent diagnostics and bounded shared-parameter reduction."""

    spec: SpectralSpec
    max_pixels: int
    context: DeviceContext
    _kernels: Any = field(repr=False)
    _status: Any = field(repr=False)
    _empty: Any = field(repr=False)
    _empty_signal: Any = field(repr=False)
    _partials: Any = field(repr=False)

    @property
    def dtype(self) -> Any:
        return (
            self.context.wp.float64 if self.spec.precision == "float64" else self.context.wp.float32
        )

    @property
    def scratch_bytes(self) -> int:
        return 4 + 8 * int(self._partials.size)

    def clear_status(self) -> None:
        with self.context.scope():
            self._status.zero_()

    def check_status(self) -> None:
        """Explicit completion point; output/gradient overflow invalidates the pass."""
        self.context.wp.synchronize_stream(self.context.stream)
        code = int(self._status.numpy()[0])
        if code & 1:
            raise ContractError("spectral inputs contain nonfinite or negative values")
        if code & 4:
            raise NumericalError(
                "spectral intermediate underflow: unsupported exponent/product range"
            )
        if code & 2:
            raise NumericalError(
                "spectral output or gradient cannot be represented in its destination precision"
            )

    def _check_values(self, arrays: list[tuple[Any, bool]]) -> None:
        self.clear_status()
        for values, nonnegative in arrays:
            if values.size:
                self.context.wp.launch(
                    self._kernels.get_value_check(
                        values.dtype == self.context.wp.float64, nonnegative
                    ),
                    dim=values.size,
                    inputs=[values],
                    outputs=[self._status],
                    stream=self.context.stream,
                    record_tape=False,
                )
        self.check_status()

    def validate_inputs(
        self,
        coefficients: Any,
        weights: Any,
        response: Any,
        *,
        pixels: int,
        paths: Any = None,
        probability_response: bool = False,
        stream: Any = None,
    ) -> None:
        """Validate current supplied arrays without evaluating a spectral image.

        A count observation requires probabilities; a general first-moment
        response only requires nonnegative values and its declared output unit.
        Omit paths when preparing fixed inputs before the material projection.
        """
        require_no_tape()
        ctx, spec = self.context, self.spec
        ctx.assert_stream(stream)
        integer(pixels, "pixels", maximum=self.max_pixels)
        if type(probability_response) is not bool:
            raise ContractError("probability_response must be a boolean")
        arrays = [
            (values, True)
            for _, values in _fixed_inputs(coefficients, weights, response, pixels, self)
        ]
        if paths is not None:
            ctx.array(paths, "paths", dtype=self.dtype, shape=(spec.materials * pixels,))
            arrays.append((paths, True))
        with ctx.scope():
            self._check_values(arrays)
            if probability_response and response.size:
                ctx.wp.launch(
                    self._kernels.check_probability,
                    dim=response.size,
                    inputs=[response],
                    outputs=[self._status],
                    stream=ctx.stream,
                    record_tape=False,
                )
                self.check_status()


def prepare_spectral(
    spec: SpectralSpec,
    *,
    max_pixels: int,
    device: str = "cuda:0",
    stream: Any = None,
) -> SpectralWorkspace:
    """Allocate scratch only; callers own source coefficients, weights and destinations."""
    integer(max_pixels, "max_pixels", maximum=(2**31 - 1) // max(spec.materials, spec.energies))
    context = prepare_context(device=device, stream=stream)
    kernels = load_kernels("dpt.kernels.spectral")
    parameters = 0
    if (spec.active_weights and spec.shared_weights) or (
        spec.active_response and spec.shared_response
    ):
        parameters = spec.energies
    if spec.active_coefficients:
        parameters = spec.materials * spec.energies
    with context.scope():
        status = context.wp.zeros(1, dtype=context.wp.int32, device=context.device)
        empty = context.wp.empty(0, dtype=context.wp.float32, device=context.device)
        empty_signal = context.wp.empty(
            0,
            dtype=context.wp.float64 if spec.precision == "float64" else context.wp.float32,
            device=context.device,
        )
        partials = context.wp.empty(
            parameters * spec.reduction_groups,
            dtype=context.wp.float64,
            device=context.device,
        )
    return SpectralWorkspace(
        spec, max_pixels, context, kernels, status, empty, empty_signal, partials
    )


def _inputs(
    paths: Any,
    coefficients: Any,
    weights: Any,
    response: Any,
    workspace: SpectralWorkspace,
) -> tuple[int, list[tuple[str, Any]]]:
    ctx, spec = workspace.context, workspace.spec
    size = ctx.array(paths, "paths", dtype=workspace.dtype)
    if size % spec.materials:
        raise ContractError("material-major paths must contain M*P values")
    pixels = size // spec.materials
    if pixels > workspace.max_pixels:
        raise ContractError("material paths exceed workspace pixel capacity")
    return pixels, [
        ("paths", paths),
        *_fixed_inputs(coefficients, weights, response, pixels, workspace),
    ]


def _fixed_inputs(
    coefficients: Any,
    weights: Any,
    response: Any,
    pixels: int,
    workspace: SpectralWorkspace,
) -> list[tuple[str, Any]]:
    ctx, spec = workspace.context, workspace.spec
    ctx.array(
        coefficients, "coefficients", dtype=ctx.wp.float32, shape=(spec.materials * spec.energies,)
    )
    ctx.array(
        weights,
        "weights",
        dtype=ctx.wp.float32,
        shape=(spec.energies * (1 if spec.shared_weights else pixels),),
    )
    ctx.array(
        response,
        "response",
        dtype=ctx.wp.float32,
        shape=(spec.energies * (1 if spec.shared_response else pixels),),
    )
    return [
        ("coefficients", coefficients),
        ("weights", weights),
        ("response", response),
    ]


# region book:spectral-call-contract
def spectral_signal(
    paths: Any,
    coefficients: Any,
    weights: Any,
    response: Any,
    *,
    out_mean: Any,
    workspace: SpectralWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Compute sum_k weights[k,p]*response[k,p]*exp(-sum_m mu[m,k]*A[m,p]).

    Arrays are flat contiguous CUDA buffers with the prepared precision. ``weights`` are
    integrated expected photons per bin, never a spectral density. Checked calls
    scan current inputs before writing and synchronise for final range status.
    ``validate=False`` promises valid current inputs and defers range status to
    ``workspace.check_status()``. No input may alias a destination.
    """
    require_no_tape()
    ctx, spec = workspace.context, workspace.spec
    ctx.assert_stream(stream)
    pixels, reads = _inputs(paths, coefficients, weights, response, workspace)
    ctx.array(out_mean, "out_mean", dtype=workspace.dtype, shape=(pixels,))
    ctx.disjoint(reads, [("out_mean", out_mean)])
    with ctx.scope():
        if validate:
            workspace.validate_inputs(
                coefficients, weights, response, pixels=pixels, paths=paths, stream=ctx.stream
            )
        if pixels:
            ctx.wp.launch(
                workspace._kernels.get_forward(
                    spec.materials,
                    spec.energies,
                    spec.shared_weights,
                    spec.shared_response,
                    spec.precision == "float64",
                ),
                dim=pixels,
                # Smaller blocks pack the register-heavy spectral work more evenly.
                # The reduction kernels retain their separate 256-lane layout.
                block_dim=128,
                inputs=[paths, coefficients, weights, response, pixels],
                outputs=[out_mean, workspace._status],
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.check_status()


# endregion book:spectral-call-contract


def spectral_vjp(
    paths: Any,
    coefficients: Any,
    weights: Any,
    response: Any,
    *,
    seed: Any,
    out_grad_paths: Any = None,
    out_grad_weights: Any = None,
    out_grad_response: Any = None,
    out_grad_coefficients: Any = None,
    workspace: SpectralWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Overwrite selected first-order products, recomputing from unchanged inputs.

    Zero source/response values retain their one-sided algebraic derivatives;
    no derivative divides by a weight or by the rounded output. Shared parameter
    sums have a fixed split/tree order, and never use floating-point atomics.
    Caller-supplied output seeds remain unchanged. Higher derivatives and ambient
    tape recording raise rather than returning an incomplete product.
    """
    require_no_tape()
    ctx, spec = workspace.context, workspace.spec
    ctx.assert_stream(stream)
    pixels, reads = _inputs(paths, coefficients, weights, response, workspace)
    ctx.array(seed, "seed", dtype=workspace.dtype, shape=(pixels,))
    destinations = (
        ("paths", out_grad_paths, spec.active_paths, spec.materials * pixels),
        (
            "weights",
            out_grad_weights,
            spec.active_weights,
            spec.energies * (1 if spec.shared_weights else pixels),
        ),
        (
            "response",
            out_grad_response,
            spec.active_response,
            spec.energies * (1 if spec.shared_response else pixels),
        ),
        (
            "coefficients",
            out_grad_coefficients,
            spec.active_coefficients,
            spec.materials * spec.energies,
        ),
    )
    writes: list[tuple[str, Any]] = []
    for name, output, active, length in destinations:
        if output is not None:
            if not active:
                raise ContractError(f"{name} was declared fixed")
            ctx.array(
                output,
                f"out_grad_{name}",
                dtype=workspace.dtype if name == "paths" else ctx.wp.float32,
                shape=(length,),
            )
            writes.append((f"out_grad_{name}", output))
    if not writes:
        raise ContractError("request at least one active spectral gradient")
    ctx.disjoint([*reads, ("seed", seed)], writes)
    with ctx.scope():
        if validate:
            workspace._check_values([*((value, True) for _, value in reads), (seed, False)])
        pixel_weights = out_grad_weights is not None and not spec.shared_weights
        pixel_response = out_grad_response is not None and not spec.shared_response
        if pixels and (out_grad_paths is not None or pixel_weights or pixel_response):
            ctx.wp.launch(
                workspace._kernels.get_pixel_vjp(
                    spec.materials,
                    spec.energies,
                    spec.shared_weights,
                    spec.shared_response,
                    out_grad_paths is not None,
                    pixel_weights,
                    pixel_response,
                    spec.precision == "float64",
                ),
                dim=pixels,
                block_dim=128,
                inputs=[paths, coefficients, weights, response, seed, pixels],
                outputs=[
                    out_grad_paths if out_grad_paths is not None else workspace._empty_signal,
                    out_grad_weights if pixel_weights else workspace._empty,
                    out_grad_response if pixel_response else workspace._empty,
                    workspace._status,
                ],
                stream=ctx.stream,
                record_tape=False,
            )
        groups = min(spec.reduction_groups, max(1, (pixels + 255) // 256))
        for kind, output, shared, parameters in (
            (0, out_grad_weights, spec.shared_weights, spec.energies),
            (1, out_grad_response, spec.shared_response, spec.energies),
            (2, out_grad_coefficients, True, spec.materials * spec.energies),
        ):
            if output is None or not shared:
                continue
            if pixels:
                partial_kernel = (
                    workspace._kernels.get_coefficient_partials(
                        spec.materials,
                        spec.energies,
                        spec.shared_weights,
                        spec.shared_response,
                        spec.precision == "float64",
                    )
                    if kind == 2
                    else workspace._kernels.get_shared_partials(
                        spec.materials,
                        spec.energies,
                        spec.shared_weights,
                        spec.shared_response,
                        kind,
                        spec.precision == "float64",
                    )
                )
                ctx.wp.launch_tiled(
                    partial_kernel,
                    dim=(groups, spec.energies if kind == 2 else parameters),
                    block_dim=256,
                    inputs=[paths, coefficients, weights, response, seed, pixels, groups],
                    outputs=[workspace._partials, workspace._status],
                    stream=ctx.stream,
                    record_tape=False,
                )
                ctx.wp.launch(
                    workspace._kernels.finish_shared,
                    dim=parameters,
                    inputs=[workspace._partials, groups],
                    outputs=[output, workspace._status],
                    stream=ctx.stream,
                    record_tape=False,
                )
            else:
                output.zero_()
        if validate:
            workspace.check_status()


def spectral_bin_counts(
    paths: Any,
    coefficients: Any,
    weights: Any,
    detection_probability: Any,
    *,
    out_bin_counts: Any,
    workspace: SpectralWorkspace,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Materialise detected count means in explicit caller-owned (K,P) storage.

    This observation-preparation boundary uses probabilities in [0,1], rather
    than an integrator's first-moment response. Feed these independent Poisson
    rates and deterministic per-photon scores to ``sample_compound_poisson``.
    It deliberately allocates no P*K tensor unless the caller requests that
    observation representation. This helper exposes no derivative or tape rule;
    differentiate the expected image with ``spectral_vjp`` instead.
    """
    require_no_tape()
    ctx, spec = workspace.context, workspace.spec
    ctx.assert_stream(stream)
    pixels, reads = _inputs(paths, coefficients, weights, detection_probability, workspace)
    ctx.array(
        out_bin_counts, "out_bin_counts", dtype=workspace.dtype, shape=(spec.energies * pixels,)
    )
    ctx.disjoint(reads, [("out_bin_counts", out_bin_counts)])
    with ctx.scope():
        if validate:
            workspace.validate_inputs(
                coefficients,
                weights,
                detection_probability,
                pixels=pixels,
                paths=paths,
                probability_response=True,
                stream=ctx.stream,
            )
        if pixels:
            ctx.wp.launch(
                workspace._kernels.get_bin_counts(
                    spec.materials,
                    spec.energies,
                    spec.shared_weights,
                    spec.shared_response,
                    spec.precision == "float64",
                ),
                dim=(spec.energies, pixels),
                inputs=[paths, coefficients, weights, detection_probability, pixels],
                outputs=[out_bin_counts, workspace._status],
                stream=ctx.stream,
                record_tape=False,
            )
        if validate:
            workspace.check_status()
