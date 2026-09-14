"""Fixed-grid analogue transport with explicit statistical and derivative contracts.

Importing this package is CPU-safe. CUDA is loaded only by workspace preparation;
no material coefficients, sampled sources, or result images are bundled.
"""

from .model import (
    HistoryStatus,
    IncompleteHistoryError,
    MaterialGrid,
    PlanarDetector,
    TransportError,
    TransportSpec,
)
from .rng import HistoryBatch

__all__ = [
    "HistoryBatch",
    "HistoryStatus",
    "IncompleteHistoryError",
    "MaterialGrid",
    "PlanarDetector",
    "TransportError",
    "TransportSpec",
]
