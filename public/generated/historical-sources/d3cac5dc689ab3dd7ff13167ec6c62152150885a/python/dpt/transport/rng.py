"""History identity and independence declarations; no GPU import or RNG execution.

Philox4x32-10 uses (history low, history high, event, 0) as its counter and the
64-bit run seed as its key. Draw lanes 0,1,2,3 are respectively free-flight optical
depth, interaction type, polar cosine and azimuth. Crossing a material face does
not advance the event or discard residual optical depth. Integer counter
identity is independent of launch ordering, chunking, and derivative replay.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import TransportError


@dataclass(frozen=True, slots=True)
class HistoryBatch:
    """A disjoint range of original histories, with an explicit source-draw namespace.

    `source_namespace` must identify the caller's source random stream as well as
    any random importance weights. Disjoint transport histories do not make a
    reused random source draw independent. Use distinct namespaces for independent
    source batches, or the literal ``deterministic`` for a deterministic source.
    """

    seed: int
    first_history: int
    count: int
    source_namespace: str = "deterministic"

    def __post_init__(self) -> None:
        for name, value in (("seed", self.seed), ("first_history", self.first_history)):
            if type(value) is not int or not 0 <= value < 2**64:
                raise TransportError(f"{name} must be an unsigned 64-bit integer")
        if type(self.count) is not int or not 0 < self.count < 2**31:
            raise TransportError("history count must be a positive signed 32-bit integer")
        if self.first_history + self.count > 2**64:
            raise TransportError("the history range would wrap its 64-bit identity")
        if not self.source_namespace.strip():
            raise TransportError("declare the source random-stream namespace")

    def independent_of(self, other: HistoryBatch) -> bool:
        """Check declared stream separation, not empirical statistical independence."""
        transport_separate = self.seed != other.seed or (
            self.first_history + self.count <= other.first_history
            or other.first_history + other.count <= self.first_history
        )
        source_separate = (
            self.source_namespace == other.source_namespace == "deterministic"
            or self.source_namespace != other.source_namespace
        )
        return transport_separate and source_separate


def require_independent(*batches: HistoryBatch) -> None:
    """Reject overlapping random histories before combining nonlinear estimators."""
    for index, left in enumerate(batches):
        for right in batches[index + 1 :]:
            if not left.independent_of(right):
                raise TransportError(
                    "nonlinear estimator requires disjoint transport/source streams"
                )
