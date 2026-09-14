"""Prepared CUDA transport, sparse history scoring and mandatory completion checks.

Device arrays are caller-owned. Model buffers remain immutable for the workspace
lifetime; dynamic density/source arrays remain unchanged until the owning stream
finishes. Preparation uploads only the small declared energy grid. Repeated calls
allocate no device arrays. Explicitly unchecked calls require ``check_status``
before accepting outputs; incomplete histories never become absorbed histories.
"""

from __future__ import annotations

# Workspace internals are shared only within this package.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass, field
from typing import Any

from dpt._runtime import DeviceContext, load_kernels, prepare_context, require_no_tape
from dpt.contracts import NumericalError

from .model import IncompleteHistoryError, TransportError, TransportSpec, _physical
from .rng import HistoryBatch


@dataclass(slots=True)
class TransportWorkspace:
    """One stream, immutable material tables, small diagnostics and no history tape."""

    spec: TransportSpec
    max_histories: int
    context: DeviceContext
    material_ids: Any
    absorption: Any
    scattering: Any
    _energies: Any = field(repr=False)
    _parameters: Any = field(repr=False)
    _status: Any = field(repr=False)
    _kernels: Any = field(repr=False)

    @property
    def scratch_bytes(self) -> int:
        return 4 + 8 * len(self.spec.energy_nodes)

    def clear_status(self) -> None:
        """Reset at a declared independent run boundary, after discarding failures."""
        with self.context.scope():
            self._status.zero_()

    def check_status(self) -> None:
        """Synchronise this stream and transfer only the four-byte status word."""
        self.context.wp.synchronize_stream(self.context.stream)
        code = int(self._status.numpy()[0])
        if code & (4 | 8 | 64):
            raise IncompleteHistoryError(
                "event, voxel-crossing or angular-rejection budget exhausted; discard the estimate"
            )
        if code & 32:
            raise TransportError("a history left the supplied coefficient energy support")
        if code & ~(4 | 8 | 16 | 32 | 64):
            raise TransportError("invalid device transport inputs")
        if code & 16:
            raise NumericalError("nonfinite transport arithmetic or an invalid moment")

    def _launch(self, kernel: Any, dim: int, inputs: list[Any]) -> None:
        self.context.wp.launch(
            kernel,
            dim=dim,
            inputs=inputs,
            device=self.context.device,
            stream=self.context.stream,
            block_dim=self.spec.block_dim,
            record_tape=False,
        )

    def _model_inputs(self) -> list[Any]:
        return [
            self.material_ids,
            self._energies,
            self.absorption,
            self.scattering,
            self._parameters,
        ]

    def _reads(self) -> list[tuple[str, Any]]:
        return [
            ("material_ids", self.material_ids),
            ("absorption", self.absorption),
            ("scattering", self.scattering),
            ("energy_nodes", self._energies),
            ("workspace_status", self._status),
        ]


def prepare_transport(
    spec: TransportSpec,
    *,
    material_ids: Any,
    absorption: Any,
    scattering: Any,
    max_histories: int,
    device: str = "cuda:0",
    stream: Any = None,
    validate: bool = True,
) -> TransportWorkspace:
    """Bind CUDA buffers and validate supplied immutable coefficients explicitly.

    Material IDs are int32, flattened x fastest. Coefficients are binary64,
    material-major with energy fastest. Their partial sum is total extinction;
    no inconsistent independent total coefficient is accepted. The tables are
    already linear coefficients in inverse mm. For Compton mode they describe
    free-electron scattering in the user's declared model, not bound-electron
    corrections inferred by this package.
    """
    if type(max_histories) is not int or not 0 < max_histories < 2**31:
        raise TransportError("max_histories must be a positive signed 32-bit integer")
    context = prepare_context(device=device, stream=stream)
    require_no_tape()
    wp = context.wp
    context.array(material_ids, "material_ids", dtype=wp.int32, shape=(math.prod(spec.grid.shape),))
    table_size = spec.grid.materials * len(spec.energy_nodes)
    if table_size >= 2**31:
        raise TransportError("material-energy table exceeds signed 32-bit indexing")
    for name, value in (("absorption", absorption), ("scattering", scattering)):
        context.array(value, name, dtype=wp.float64, shape=(table_size,))
    kernels = load_kernels("dpt.transport.kernels")
    with context.scope():
        energies = wp.array(list(spec.energy_nodes), dtype=wp.float64, device=context.device)
        status = wp.zeros(1, dtype=wp.int32, device=context.device)
    parameters = kernels.Parameters()
    parameters.origin = wp.vec3d(*spec.grid.origin_mm)
    parameters.spacing = wp.vec3d(*spec.grid.spacing_mm)
    parameters.shape = wp.vec3i(*spec.grid.shape_xyz)
    parameters.detector_lower = wp.vec2d(*spec.detector.lower_xy_mm)
    parameters.detector_spacing = wp.vec2d(*spec.detector.spacing_xy_mm)
    parameters.detector_shape = wp.vec2i(*spec.detector.shape)
    parameters.detector_z = spec.detector.z_mm
    parameters.source_energy = spec.energy_kev
    parameters.energy_bins = len(spec.energy_nodes)
    parameters.max_events = spec.max_events
    parameters.max_crossings = spec.max_crossings
    parameters.max_angle_trials = spec.max_angle_trials
    parameters.compton = int(spec.scattering_law == "free-electron-compton")
    parameters.energy_score = int(spec.scoring == "energy-kev")
    parameters.continuous_absorption = int(spec.estimator == "continuous-absorption")
    workspace = TransportWorkspace(
        spec,
        max_histories,
        context,
        material_ids,
        absorption,
        scattering,
        energies,
        parameters,
        status,
        kernels,
    )
    if validate:
        workspace._launch(
            kernels.validate_model,
            max(int(material_ids.size), table_size),
            [material_ids, absorption, scattering, spec.grid.materials, status],
        )
        workspace.check_status()
    return workspace


def _validate_call(
    workspace: TransportWorkspace,
    batch: HistoryBatch,
    positions: Any,
    directions: Any,
    weights: Any,
    density: Any,
    writes: list[tuple[str, Any, Any]],
    source_amplitude: float,
    stream: Any,
    validate: bool,
) -> None:
    require_no_tape()
    context = workspace.context
    context.assert_stream(stream)
    wp = context.wp
    if batch.count > workspace.max_histories:
        raise TransportError("history batch exceeds workspace capacity")
    _physical(source_amplitude, "source amplitude", nonnegative=True)
    source_arrays = [("positions", positions), ("directions", directions), ("weights", weights)]
    for name, value in source_arrays:
        context.array(
            value,
            name,
            dtype=wp.float64 if name == "weights" else wp.vec3d,
            shape=(batch.count,),
        )
    context.array(density, "density", dtype=wp.float64, shape=(workspace.spec.grid.materials,))
    for name, value, dtype in writes:
        context.array(value, name, dtype=dtype, shape=(batch.count,))
    context.disjoint(
        workspace._reads() + source_arrays + [("density", density)],
        [(name, value) for name, value, _ in writes],
    )
    # Unchecked calls retain prior failure status. A successful later launch
    # cannot launder an incomplete earlier estimate into an accepted one.
    if validate:
        workspace.check_status()
        workspace._launch(
            workspace._kernels.validate_sources,
            max(batch.count, workspace.spec.grid.materials),
            [
                positions,
                directions,
                weights,
                density,
                workspace.spec.detector.z_mm,
                workspace._status,
            ],
        )
        workspace.check_status()


# region book:transport-history-execution
def trace_histories(
    positions: Any,
    directions: Any,
    weights: Any,
    density: Any,
    *,
    batch: HistoryBatch,
    workspace: TransportWorkspace,
    out_pixel: Any,
    out_score: Any,
    out_energy: Any,
    out_events: Any,
    out_status: Any,
    source_amplitude: float = 1.0,
    stream: Any = None,
    validate: bool = True,
) -> None:
    """Write one sparse detector score per independent original history.

    Binary64 positions are in mm; directions must be unit vectors to 1e-12 in
    squared norm. Source positions lie below the detector plane. Nonnegative
    base weights encode the caller's fixed source importance sampling, while
    ``source_amplitude`` scales the expected measurement. ``density`` contains
    strictly positive dimensionless scales multiplying both partial coefficients.

    The score is per launched history; averaging includes misses and absorption
    as zeros. It is not normalised by detected photons. `out_pixel=-1` means no
    detector hit. Score, final energy (keV), collision count and status are always
    written, including on failure; invalid outputs must be discarded as a batch.
    Continuous absorption samples scattering flights and weights each detector
    hit by absorption along its entire path; its event count counts scatterings.
    The analogue default retains sampled absorption and total-extinction flights.
    """
    wp = workspace.context.wp
    _validate_call(
        workspace,
        batch,
        positions,
        directions,
        weights,
        density,
        [
            ("out_pixel", out_pixel, wp.int32),
            ("out_score", out_score, wp.float64),
            ("out_energy", out_energy, wp.float64),
            ("out_events", out_events, wp.int32),
            ("out_status", out_status, wp.int32),
        ],
        source_amplitude,
        stream,
        validate,
    )
    workspace._launch(
        workspace._kernels.trace_histories,
        batch.count,
        [
            positions,
            directions,
            weights,
            density,
            *workspace._model_inputs(),
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            source_amplitude,
            out_pixel,
            out_score,
            out_energy,
            out_events,
            out_status,
            workspace._status,
        ],
    )
    if validate:
        workspace.check_status()


# endregion book:transport-history-execution
