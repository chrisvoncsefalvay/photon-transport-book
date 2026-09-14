"""Explicit, CPU-safe ingestion of primary attenuation coefficients.

There are no bundled physical tables. A caller supplies the values, identifies
where they came from and records the rights under which they may be used.
Interpolation is confined to the supplied support; repeated energies encode the
two one-sided values at an absorption edge rather than an interval to smooth.
"""

from __future__ import annotations

# Public Python boundaries validate runtime inputs; workspace scratch stays module-owned.
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false
import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from dpt.contracts import ContractError, finite_scalar, finite_tuple


@dataclass(frozen=True, slots=True)
class Provenance:
    """Identity of the actual input, distinct from a bibliographic citation.

    ``sha256`` identifies the original source bytes, before unit conversion.
    Analytic fixtures must say so in ``description`` and cite their defining
    formula in ``source``; a real material name is not fixture provenance.
    """

    source: str
    sha256: str
    rights: str
    description: str

    def __post_init__(self) -> None:
        for name in ("source", "rights", "description"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"provenance {name} must be non-empty")
        if not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ContractError("provenance sha256 must contain 64 lowercase hexadecimal digits")


def validate_energy_grid(energies_kev: tuple[float, ...]) -> None:
    """Validate distinct, positive keV nodes shared by spectral inputs and response."""
    finite_tuple(energies_kev, "energies_kev", positive=True)
    if any(a >= b for a, b in pairwise(energies_kev)):
        raise ContractError("energy nodes must be strictly increasing")


def _log_ratio(high: float, low: float) -> float:
    """Retain adjacent large energies without overflowing a widely separated ratio."""
    relative = (high - low) / low
    return math.log1p(relative) if math.isfinite(relative) else math.log(high) - math.log(low)


# region book:material-table-contract
@dataclass(frozen=True, slots=True)
class MaterialTable:
    """Total primary attenuation at one declared material composition/density.

    Arrays are immutable host metadata, never an alternative production backend.
    Energies are keV. Duplicate adjacent energies encode below/above edge values,
    in that order; at most two entries may share an energy. The density attached
    to a mass table is in g/cm³. Linear coefficients already include density.
    """

    name: str
    energies_kev: tuple[float, ...]
    coefficients: tuple[float, ...]
    units: Literal["mm^-1", "cm^-1", "cm^2/g"]
    provenance: Provenance
    density_g_cm3: float | None = None
    interpolation: Literal["linear", "log-log"] = "log-log"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ContractError("a material table needs an unambiguous material/composition name")
        if not isinstance(self.provenance, Provenance):
            raise ContractError("material-table provenance is required")
        finite_tuple(self.energies_kev, "energies_kev", positive=True)
        finite_tuple(self.coefficients, "coefficients")
        if len(self.energies_kev) != len(self.coefficients):
            raise ContractError("energies and coefficients must have equal lengths")
        if self.units not in ("mm^-1", "cm^-1", "cm^2/g"):
            raise ContractError("unsupported coefficient units")
        if self.interpolation not in ("linear", "log-log"):
            raise ContractError("interpolation must be linear or log-log")
        if self.interpolation == "log-log" and any(c <= 0 for c in self.coefficients):
            raise ContractError("log-log interpolation requires strictly positive coefficients")
        for index, energy in enumerate(self.energies_kev[1:], 1):
            if energy < self.energies_kev[index - 1]:
                raise ContractError("energy support must be non-decreasing")
            if index > 1 and energy == self.energies_kev[index - 2]:
                raise ContractError("an absorption edge has exactly two one-sided entries")
        if self.units == "cm^2/g":
            density = self.density_g_cm3
            density = finite_scalar(density, "reference density in g/cm³", minimum=0)
            if density <= 0:
                raise ContractError("reference density must be strictly positive")
        elif self.density_g_cm3 is not None:
            raise ContractError(
                "linear coefficients already include density; do not apply it twice"
            )

    @property
    def linear_mm_inverse(self) -> tuple[float, ...]:
        """Convert source values once, without inventing a different mixture model."""
        scale = 1.0
        if self.units == "cm^-1":
            scale = 0.1
        elif self.units == "cm^2/g":
            assert self.density_g_cm3 is not None
            scale = self.density_g_cm3 / 10.0
        converted = tuple(c * scale for c in self.coefficients)
        if any(not math.isfinite(c) for c in converted):
            raise ContractError("unit conversion overflowed; coefficient table is unusable")
        if any(
            source > 0 and target == 0
            for source, target in zip(self.coefficients, converted, strict=True)
        ):
            raise ContractError("unit conversion underflowed a positive attenuation coefficient")
        return converted

    def at_energies(
        self,
        energies_kev: tuple[float, ...],
        *,
        edge_side: Literal["below", "above"],
    ) -> tuple[float, ...]:
        """Interpolate in mm⁻¹, requiring the one-sided convention at exact edges."""
        if edge_side not in ("below", "above"):
            raise ContractError("edge_side must be explicitly below or above")
        finite_tuple(energies_kev, "query energies", positive=True)
        grid = self.energies_kev
        coefficients = self.linear_mm_inverse
        result: list[float] = []
        for energy in energies_kev:
            if energy < grid[0] or energy > grid[-1]:
                raise ContractError(f"{self.name}: {energy} keV lies outside supplied support")
            left, right = bisect_left(grid, energy), bisect_right(grid, energy)
            if left != right:
                result.append(coefficients[left if edge_side == "below" else right - 1])
                continue
            low, high = left - 1, left
            if self.interpolation == "log-log":
                fraction = _log_ratio(energy, grid[low]) / _log_ratio(grid[high], grid[low])
                value = math.exp(
                    (1.0 - fraction) * math.log(coefficients[low])
                    + fraction * math.log(coefficients[high])
                )
            else:
                fraction = (energy - grid[low]) / (grid[high] - grid[low])
                value = (1.0 - fraction) * coefficients[low] + fraction * coefficients[high]
            result.append(value)
        return tuple(result)


# endregion book:material-table-contract


@dataclass(frozen=True, slots=True)
class MaterialBasis:
    """Fixed-density, non-negative dimensionless basis fields integrated in mm.

    ``mixing_assumption`` states whether fields are volume fractions, material
    indicators or relative concentrations. There is no implicit HU conversion.
    Spatially varying mass density must be included in the declared basis field.
    """

    tables: tuple[MaterialTable, ...]
    mixing_assumption: str

    def __post_init__(self) -> None:
        if not isinstance(self.tables, tuple) or not self.tables:
            raise ContractError("a material basis requires an immutable tuple of tables")
        if any(not isinstance(table, MaterialTable) for table in self.tables):
            raise ContractError("material basis entries must be MaterialTable instances")
        if len({table.name for table in self.tables}) != len(self.tables):
            raise ContractError("material basis names must be unique")
        if not self.mixing_assumption.strip():
            raise ContractError("the material-basis mixing assumption must be recorded")

    def coefficients_at(
        self, energies_kev: tuple[float, ...], *, edge_side: Literal["below", "above"]
    ) -> tuple[float, ...]:
        """Return material-major (material, energy) coefficients for explicit upload."""
        return tuple(
            value
            for table in self.tables
            for value in table.at_energies(energies_kev, edge_side=edge_side)
        )
