"""Fixed-distribution parallel sources keyed by original history identity."""

# Internal workspace launch plumbing is shared within the transport package.
# pyright: reportPrivateUsage=false
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from dpt._runtime import require_no_tape

from .forward import TransportWorkspace
from .model import TransportError, _freeze_fields, _physical
from .rng import HistoryBatch


@dataclass(frozen=True, slots=True)
class ParallelBeam:
    """Uniform rectangular source in an xy plane, with one fixed unit direction.

    Extent zero in both axes specifies a deterministic pencil. `weight` is the
    expected source score per sampled ray before the separate source amplitude;
    it is not silently multiplied by area, exposure or inverse-square factors.
    The caller sets that physical normalisation. The distribution stays fixed
    throughout recovery and samples original histories independently by identity.
    """

    lower_mm: tuple[float, float, float]
    extent_xy_mm: tuple[float, float]
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    weight: float = 1.0

    def __post_init__(self) -> None:
        _freeze_fields(self, ("lower_mm", "extent_xy_mm", "direction"))
        for value in (*self.lower_mm, *self.extent_xy_mm, *self.direction):
            _physical(value, "source geometry")
        if len(self.lower_mm) != 3 or len(self.direction) != 3 or len(self.extent_xy_mm) != 2:
            raise TransportError("source vectors have incompatible dimensions")
        if any(value < 0 for value in self.extent_xy_mm):
            raise TransportError("source extents must be nonnegative")
        if abs(math.fsum(value * value for value in self.direction) - 1) > 1e-12:
            raise TransportError("source direction must be a unit vector")
        _physical(self.weight, "source weight", nonnegative=True)
        for lower, extent in zip(self.lower_mm[:2], self.extent_xy_mm, strict=True):
            if not math.isfinite(lower + extent) or (extent > 0 and lower + extent == lower):
                raise TransportError("source extent overflows its coordinate representation")


# region book:transport-source-sampling
def sample_parallel_beam(
    source: ParallelBeam,
    *,
    batch: HistoryBatch,
    workspace: TransportWorkspace,
    out_position: Any,
    out_direction: Any,
    out_weight: Any,
    stream: Any = None,
) -> None:
    """Fill caller-owned source arrays with independently keyed rectangle samples.

    This operation does not allocate, download or synchronise. Its constants are
    checked by ParallelBeam and each original identity chooses its own source
    sample. Event bit 31 reserves source draws separately from collision events.
    The random source law is fixed when differentiating material densities.
    """
    require_no_tape()
    context = workspace.context
    context.assert_stream(stream)
    wp = context.wp
    if batch.count > workspace.max_histories or source.lower_mm[2] >= workspace.spec.detector.z_mm:
        raise TransportError("source batch capacity or source-detector geometry is invalid")
    for name, value, dtype in (
        ("out_position", out_position, wp.vec3d),
        ("out_direction", out_direction, wp.vec3d),
        ("out_weight", out_weight, wp.float64),
    ):
        context.array(value, name, dtype=dtype, shape=(batch.count,))
    context.disjoint(
        workspace._reads(),
        [
            ("out_position", out_position),
            ("out_direction", out_direction),
            ("out_weight", out_weight),
        ],
    )
    workspace._launch(
        workspace._kernels.sample_parallel_source,
        batch.count,
        [
            wp.vec3d(*source.lower_mm),
            wp.vec2d(*source.extent_xy_mm),
            wp.vec3d(*source.direction),
            source.weight,
            wp.uint64(batch.seed),
            wp.uint64(batch.first_history),
            out_position,
            out_direction,
            out_weight,
        ],
    )


# endregion book:transport-source-sampling
