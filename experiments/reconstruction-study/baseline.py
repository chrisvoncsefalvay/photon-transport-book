"""Post-log PWLS using canonical projection, volume VJP and projected solve.

This is an approximate-likelihood scalar comparator, not a new GPU backend.
The supplied count-domain driver remains the Poisson implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from dpt.contracts import ContractError, NumericalError
from dpt.examples.reconstruction import Reconstruction
from dpt.objectives import ObjectiveSpec, evaluate_objective, prepare_objective
from dpt.projection import project_optical_depth, projection_vjp
from dpt.transmission import transmit


@dataclass(frozen=True, slots=True)
class PostLogData:
    optical_depth: Any
    weights: Any
    valid: Any
    excluded_zero_counts: int
    negative_optical_depths: int


def post_log_data(counts: Any, beam: Any) -> PostLogData:
    """Construct fixed log(b/k), weight k and an explicit zero-count mask.

    Only positive counts enter logarithms. Counts greater than the known beam
    yield negative optical-depth data and are retained without clipping.
    """
    for name, value in (("counts", counts), ("beam", beam)):
        if not isinstance(value, np.ndarray):
            raise ContractError(f"{name} must be a finite contiguous native FP32 image")
        candidate = cast(Any, value)
        if (
            candidate.dtype != np.float32
            or not candidate.dtype.isnative
            or not candidate.flags.c_contiguous
            or candidate.ndim != 2
            or not np.isfinite(candidate).all()
        ):
            raise ContractError(f"{name} must be a finite contiguous native FP32 image")
    if counts.shape != beam.shape or counts.size == 0:
        raise ContractError("counts and beam must have the same nonempty image shape")
    if (counts < 0).any() or (counts % 1 != 0).any() or (counts > 2**24).any():
        raise ContractError("counts must be exactly represented nonnegative FP32 integers")
    if (beam <= 0).any():
        raise ContractError("beam must be strictly positive")
    valid = np.ascontiguousarray(counts > 0, dtype=np.uint8)
    active = valid.astype(bool)
    log_data = np.zeros(counts.shape, dtype=np.float32)
    # Subtract FP64 logs instead of taking a potentially extreme FP32 ratio.
    log_data[active] = np.log(beam[active].astype(np.float64)) - np.log(
        counts[active].astype(np.float64)
    )
    if not np.isfinite(log_data).all():
        raise ContractError("post-log data are not representable in FP32")
    weights = counts.copy()
    return PostLogData(
        optical_depth=log_data,
        weights=weights,
        valid=valid,
        excluded_zero_counts=int((~active).sum()),
        negative_optical_depths=int((log_data[active] < 0).sum()),
    )


class PostLogReconstruction(Reconstruction):
    """Change only the data objective; inherit the scalar solve and regulariser.

    The inherited constructor retains count observations and prediction buffers
    alongside fixed WLS arrays. WLS evaluation does not evaluate exponentials;
    call ``predictions_numpy`` for refreshed physical count predictions.
    """

    def __init__(self, case: Any, *, device: str, torch: Any, wp: Any) -> None:
        super().__init__(case, device=device, torch=torch, wp=wp)
        self.preprocessing: list[dict[str, int]] = []
        entries = case.config["reconstruction"]["views"]
        for entry, view in zip(entries, self.views, strict=True):
            counts = case.array(entry["counts"], shape=view["shape"], units="counts")
            beam = case.array(entry["open_beam"], shape=view["shape"], units="counts")
            data = post_log_data(counts, beam)
            view["objective"] = prepare_objective(
                ObjectiveSpec(kind="squared_error", domain="signal", weighted=True, masked=True),
                max_pixels=counts.size,
                device=device,
                stream=self.stream,
            )
            with wp.ScopedStream(self.stream, sync_enter=False):
                for name, values, dtype in (
                    ("log_observed", data.optical_depth, wp.float32),
                    ("wls_weights", data.weights, wp.float32),
                    ("valid", data.valid, wp.uint8),
                ):
                    view[name] = wp.array(values.reshape(-1), dtype=dtype, device=device)
            view["objective"].validate_observation(
                view["log_observed"],
                weights=view["wls_weights"],
                valid=view["valid"],
                stream=self.stream,
            )
            self.preprocessing.append(
                {
                    "pixels": counts.size,
                    "excluded_zero_counts": data.excluded_zero_counts,
                    "negative_optical_depths": data.negative_optical_depths,
                }
            )
        # Fixed arrays are uploaded once and never changed during line search.
        wp.synchronize_stream(self.stream)
        if all(row["pixels"] == row["excluded_zero_counts"] for row in self.preprocessing):
            raise ContractError("post-log WLS requires at least one positive observed count")

    def evaluate(self, volume: Any, volume_wp: Any, *, gradient: bool) -> float:
        if gradient:
            self.gradient.zero_()
        loss = 0.0
        for view in self.views:
            project_optical_depth(
                volume_wp,
                self.pose,
                workspace=view["projection"],
                out_L=view["depth"],
                stream=self.stream,
            )
            evaluate_objective(
                view["depth"],
                view["log_observed"],
                weights=view["wls_weights"],
                valid=view["valid"],
                workspace=view["objective"],
                out_loss=view["loss"],
                out_seed=view["depth_seed"] if gradient else None,
                stream=self.stream,
            )
            if gradient:
                projection_vjp(
                    volume_wp,
                    self.pose,
                    adj_L=view["depth_seed"],
                    workspace=view["projection"],
                    out_mu=self.gradient_wp,
                    accumulate=True,
                    stream=self.stream,
                )
            loss += float(view["loss"].numpy()[0])
        loss += self.regularise(volume, gradient=gradient)
        if not math.isfinite(loss):
            raise NumericalError("post-log WLS objective is nonfinite")
        if gradient and not bool(self.torch.isfinite(self.gradient).all().item()):
            raise NumericalError("post-log WLS volume gradient is nonfinite")
        return loss

    def predictions_numpy(self) -> tuple[Any, ...]:
        """Explicitly refresh and export mean counts at the accepted volume."""
        results: list[Any] = []
        for view in self.views:
            project_optical_depth(
                self.mu_wp,
                self.pose,
                workspace=view["projection"],
                out_L=view["depth"],
                stream=self.stream,
            )
            transmit(
                view["depth"],
                view["beam"],
                workspace=view["transmission"],
                out_counts=view["prediction"],
                stream=self.stream,
            )
            results.append(view["prediction"].numpy().reshape(view["shape"]))
        return tuple(results)
