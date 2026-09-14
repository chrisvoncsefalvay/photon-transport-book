"""Host-only contracts for fixed-grid photon transport estimators.

Supplied linear coefficients in inverse millimetres define absorption plus either
isotropic elastic or free-electron Klein-Nishina/Compton scattering. The latter
changes energy and uses an explicit coefficient energy table; it does not include
bound-electron, polarisation or coherent-scattering corrections. The material
grid is piecewise constant, unlike the deterministic projector's trilinear field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from itertools import pairwise
from typing import Any, Literal, cast

from dpt.contracts import ContractError, finite_scalar


class TransportError(ContractError):
    """Compatibility subtype for transport physical, storage and execution contracts."""


class IncompleteHistoryError(RuntimeError):
    """A resource guard stopped a history; the entire estimate is invalid."""


class HistoryStatus(IntEnum):
    """Each launched history writes exactly one terminal status."""

    ESCAPED = 0
    ABSORBED = 1
    EVENT_BUDGET = 2
    CROSSING_BUDGET = 3
    NUMERICAL_FAILURE = 4
    ENERGY_SUPPORT = 5
    ANGLE_BUDGET = 6


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or not 0 < value < 2**31:
        raise TransportError(f"{name} must be a positive signed 32-bit integer")


def _physical(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        scalar = finite_scalar(value, name, minimum=0.0 if positive or nonnegative else None)
    except ContractError as error:
        raise TransportError(str(error)) from error
    if positive and scalar == 0:
        raise TransportError(f"{name} must be positive")
    return scalar


def _freeze_fields(instance: Any, names: tuple[str, ...]) -> None:
    for name in names:
        object.__setattr__(instance, name, tuple(getattr(instance, name)))


@dataclass(frozen=True, slots=True)
class MaterialGrid:
    """Axis-aligned cell faces; shape is (nz, ny, nx), device indices x fastest."""

    origin_mm: tuple[float, float, float]
    spacing_mm: tuple[float, float, float]
    shape: tuple[int, int, int]
    materials: int

    def __post_init__(self) -> None:
        _freeze_fields(self, ("origin_mm", "spacing_mm", "shape"))
        if len(self.origin_mm) != 3 or len(self.spacing_mm) != 3 or len(self.shape) != 3:
            raise TransportError("origin, spacing and shape must each have three components")
        _positive_integer(self.materials, "materials")
        for count, step, origin in zip(
            self.shape_xyz, self.spacing_mm, self.origin_mm, strict=True
        ):
            _positive_integer(count, "grid extent")
            _physical(step, "cell spacing", positive=True)
            _physical(origin, "cell origin")
            endpoint = origin + count * step
            if (
                not math.isfinite(endpoint)
                or origin + step == origin
                or endpoint - step == endpoint
            ):
                raise TransportError("cell faces must remain distinguishable in binary64")
        if math.prod(self.shape) >= 2**31:
            raise TransportError("the flattened material grid exceeds signed 32-bit indexing")

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        """Geometric extents for device vectors; storage shape remains (z,y,x)."""
        return self.shape[2], self.shape[1], self.shape[0]

    @property
    def upper_mm(self) -> tuple[float, float, float]:
        return cast(
            tuple[float, float, float],
            tuple(
                start + count * step
                for start, count, step in zip(
                    self.origin_mm, self.shape_xyz, self.spacing_mm, strict=True
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class PlanarDetector:
    """An outward-facing z plane, pixel lower faces (x,y), and count score.

    Pixel support is half-open. The detector must be above the entire material
    grid; an escaped straight ray can therefore hit it at most once. A recorded
    score includes both uncollided and scattered photons. Do not add a separate
    deterministic primary image to this estimate.
    """

    lower_xy_mm: tuple[float, float]
    spacing_xy_mm: tuple[float, float]
    shape: tuple[int, int]  # width, height
    z_mm: float

    def __post_init__(self) -> None:
        _freeze_fields(self, ("lower_xy_mm", "spacing_xy_mm", "shape"))
        _physical(self.z_mm, "detector z")
        if any(len(value) != 2 for value in (self.lower_xy_mm, self.spacing_xy_mm, self.shape)):
            raise TransportError("detector xy fields must have two components")
        for count, step, lower in zip(
            self.shape, self.spacing_xy_mm, self.lower_xy_mm, strict=True
        ):
            _positive_integer(count, "detector extent")
            _physical(step, "pixel spacing", positive=True)
            _physical(lower, "detector origin")
            end = lower + count * step
            if not math.isfinite(end):
                raise TransportError("detector endpoint must be finite")
            if lower + step == lower or end - step == end:
                raise TransportError("detector pixels must be distinguishable in binary64")
        if self.pixels >= 2**31:
            raise TransportError("the flattened detector exceeds signed 32-bit indexing")

    @property
    def pixels(self) -> int:
        return math.prod(self.shape)


# region book:transport-model-contract
@dataclass(frozen=True, slots=True)
class TransportSpec:
    """A complete physical scope and finite launch-resource policy.

    Source rays and nonnegative importance weights are supplied by the caller.
    Their sampling law must be fixed with respect to active material parameters;
    the library does not silently invent an emission distribution. Provenance
    identifies the user's coefficient source, not a fabricated material asset.
    """

    grid: MaterialGrid
    detector: PlanarDetector
    energy_kev: float
    coefficient_provenance: str
    coefficient_energies_kev: tuple[float, ...] = ()
    scattering_law: Literal["isotropic-elastic", "free-electron-compton"] = "isotropic-elastic"
    scoring: Literal["photon-count", "energy-kev"] = "photon-count"
    max_angle_trials: int = 4096
    max_events: int = 4096
    max_crossings: int = 65536
    block_dim: Literal[64, 128, 256] = 128
    estimator: Literal["analogue", "continuous-absorption"] = "analogue"

    def __post_init__(self) -> None:
        _freeze_fields(self, ("coefficient_energies_kev",))
        _physical(self.energy_kev, "source energy", positive=True)
        for node in self.coefficient_energies_kev:
            _physical(node, "coefficient energy", nonnegative=True)
        if type(self.coefficient_provenance) is not str or not self.coefficient_provenance.strip():
            raise TransportError("the supplied coefficients require a provenance identifier")
        if self.detector.z_mm <= self.grid.upper_mm[2]:
            raise TransportError("the detector plane must lie above the entire material grid")
        nodes = self.energy_nodes
        if any(right <= left for left, right in pairwise(nodes)):
            raise TransportError("coefficient energies must be strictly increasing")
        if not nodes[0] <= self.energy_kev <= nodes[-1]:
            raise TransportError("coefficient energies must contain the source energy")
        if self.scattering_law not in ("isotropic-elastic", "free-electron-compton"):
            raise TransportError("unsupported scattering law")
        if self.scattering_law == "free-electron-compton" and len(nodes) < 2:
            raise TransportError("Compton scattering needs energy-dependent coefficient tables")
        if self.scoring not in ("photon-count", "energy-kev"):
            raise TransportError("unsupported detector score")
        if self.estimator not in ("analogue", "continuous-absorption"):
            raise TransportError("unsupported transport estimator")
        _positive_integer(self.max_angle_trials, "max_angle_trials")
        _positive_integer(self.max_events, "max_events")
        _positive_integer(self.max_crossings, "max_crossings")
        if self.block_dim not in (64, 128, 256):
            raise TransportError("block_dim must be 64, 128 or 256")

    @property
    def energy_nodes(self) -> tuple[float, ...]:
        """An omitted grid declares coefficients at the one elastic source energy."""
        return self.coefficient_energies_kev or (self.energy_kev,)


# endregion book:transport-model-contract
