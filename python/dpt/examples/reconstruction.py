"""Fixed-geometry primary-count reconstruction; supplied inputs only.

Uses the canonical Warp projector and first-order volume VJP. Torch supplies
CUDA vector arithmetic with shared storage on the same stream. The examples
README links the recorded studies and their separate numerical acceptance.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dpt.contracts import ContractError, NumericalError, finite_scalar, integer
from dpt.examples._common import example_parser, load_case, write_array
from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.objectives import ObjectiveSpec, evaluate_objective, prepare_objective
from dpt.projection import ProjectionSpec, prepare_projection, project_optical_depth, projection_vjp
from dpt.transmission import TransmissionSpec, prepare_transmission, transmission_vjp, transmit
from dpt.volumes import GridSpec


@dataclass(frozen=True, slots=True)
class SolverSettings:
    iterations: int
    initial_step_mm_inverse_squared: float
    backtracking_factor: float
    armijo: float
    maximum_backtracks: int
    gradient_mapping_tolerance_mm: float
    regularisation_mm: float
    mapping_step_mm_inverse_squared: float | None = None

    @classmethod
    def read(cls, settings: dict[str, Any]) -> SolverSettings:
        result = cls(
            iterations=integer(settings["iterations"], "iterations", minimum=1),
            initial_step_mm_inverse_squared=finite_scalar(
                settings["initial_step_mm_inverse_squared"], "initial step", minimum=0
            ),
            backtracking_factor=finite_scalar(settings["backtracking_factor"], "backtracking"),
            armijo=finite_scalar(settings["armijo"], "Armijo coefficient"),
            maximum_backtracks=integer(
                settings["maximum_backtracks"], "maximum_backtracks", minimum=1
            ),
            gradient_mapping_tolerance_mm=finite_scalar(
                settings["gradient_mapping_tolerance_mm"], "gradient tolerance", minimum=0
            ),
            regularisation_mm=finite_scalar(
                settings["regularisation_mm"], "regularisation coefficient", minimum=0
            ),
            mapping_step_mm_inverse_squared=(
                finite_scalar(
                    settings["mapping_step_mm_inverse_squared"], "mapping step", minimum=0
                )
                if settings.get("mapping_step_mm_inverse_squared") is not None
                else None
            ),
        )
        if result.initial_step_mm_inverse_squared <= 0:
            raise ContractError("Set initial_step_mm_inverse_squared to a positive value.")
        if (
            result.mapping_step_mm_inverse_squared is not None
            and result.mapping_step_mm_inverse_squared <= 0
        ):
            raise ContractError("Set mapping_step_mm_inverse_squared to a positive value.")
        if not 0 < result.backtracking_factor < 1 or not 0 < result.armijo < 1:
            raise ContractError("Set backtracking_factor and armijo strictly between zero and one.")
        return result


class Reconstruction:
    """Serial full-data evaluations with persistent CUDA buffers and checked VJPs."""

    def __init__(self, case: Any, *, device: str, torch: Any, wp: Any) -> None:
        self.torch, self.wp, self.device = torch, wp, device
        settings = case.config["reconstruction"]
        self.policy = SolverSettings.read(settings["solver"])
        self.grid = GridSpec(**{key: tuple(value) for key, value in settings["grid"].items()})
        pose = RigidTransform(
            **{key: tuple(value) for key, value in settings["fixed_pose"].items()}
        )
        samples = integer(settings["samples_per_ray"], "samples_per_ray", minimum=1)
        initial = case.array(settings["initial_volume"], shape=self.grid.shape, units="mm^-1")
        if (initial < 0).any():
            raise ContractError(
                "initial_volume contains negative attenuation; supply values in mm^-1."
            )
        if not settings["views"]:
            raise ContractError("Supply at least one calibrated view in reconstruction.views.")
        self.torch_stream = torch.cuda.current_stream(device=device)
        wp.init()
        self.stream = wp.stream_from_torch(self.torch_stream)
        self.mu = torch.as_tensor(initial, device=device).flatten().clone()
        self.trial = torch.empty_like(self.mu)
        self.gradient = torch.empty_like(self.mu)
        self.delta = torch.empty_like(self.mu)
        self.edge32 = torch.empty_like(self.mu)
        self.volume64 = torch.empty(self.grid.voxels, dtype=torch.float64, device=device)
        self.edge64 = torch.empty_like(self.volume64)
        self.scalar = torch.empty((), dtype=torch.float64, device=device)
        self.pose = wp.array(pose.packed(), dtype=wp.float64, device=device)
        self.mu_wp = wp.from_torch(self.mu)
        self.trial_wp = wp.from_torch(self.trial)
        self.gradient_wp = wp.from_torch(self.gradient)
        self.views: list[dict[str, Any]] = []
        for entry in settings["views"]:
            geometry = DetectorGeometry(
                **{key: tuple(value) for key, value in entry["geometry"].items()}
            )
            observed = case.array(entry["counts"], shape=geometry.shape, units="counts")
            beam = case.array(entry["open_beam"], shape=geometry.shape, units="counts")
            if (observed < 0).any() or (observed % 1 != 0).any() or (observed > 2**24).any():
                raise ContractError(
                    "counts must be nonnegative integers no greater than 2^24; "
                    "supply unprocessed counts exactly representable in float32."
                )
            if (beam <= 0).any():
                raise ContractError("open_beam must contain positive calibrated mean counts.")
            projection = prepare_projection(
                self.grid,
                geometry,
                ProjectionSpec(samples, active_pose=False, active_volume=True),
                device=device,
                stream=self.stream,
            )
            view: dict[str, Any] = {
                "projection": projection,
                "transmission": prepare_transmission(
                    TransmissionSpec(beam="per-pixel"),
                    max_pixels=geometry.pixels,
                    device=device,
                    stream=self.stream,
                ),
                "objective": prepare_objective(
                    ObjectiveSpec(kind="poisson", domain="counts", reduction="sum"),
                    max_pixels=geometry.pixels,
                    device=device,
                    stream=self.stream,
                ),
                "observed": wp.array(observed.reshape(-1), dtype=wp.float32, device=device),
                "beam": wp.array(beam.reshape(-1), dtype=wp.float32, device=device),
                "shape": geometry.shape,
            }
            for name in ("depth", "prediction", "image_seed", "depth_seed"):
                view[name] = wp.empty(geometry.pixels, dtype=wp.float32, device=device)
            view["loss"] = wp.empty(1, dtype=wp.float64, device=device)
            self.views.append(view)
        # All uploads above are complete before either framework consumes them.
        # Subsequent Torch/Warp work shares exactly one stream.
        wp.synchronize_device(device)
        self.edges: list[tuple[Any, Any, Any, Any, Any, float]] = []
        shaped64 = self.volume64.view(self.grid.shape)
        shaped_gradient = self.gradient.view(self.grid.shape)
        cell_volume = math.prod(self.grid.spacing_mm)
        for axis, spacing in enumerate(reversed(self.grid.spacing_mm)):
            if self.grid.shape[axis] == 1:
                continue
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis], high[axis] = slice(None, -1), slice(1, None)
            lower, upper = tuple(low), tuple(high)
            shape = shaped64[lower].shape
            count = shaped64[lower].numel()
            coefficient = self.policy.regularisation_mm * cell_volume / spacing**2
            if not math.isfinite(coefficient):
                raise ContractError(
                    "Grid spacing and regularisation overflow; rescale their units."
                )
            self.edges.append(
                (
                    shaped64[lower],
                    shaped64[upper],
                    shaped_gradient[lower],
                    shaped_gradient[upper],
                    (self.edge64[:count].view(shape), self.edge32[:count].view(shape)),
                    coefficient,
                )
            )

    # region book:example-reconstruction-volume-gradient
    def evaluate(self, volume: Any, volume_wp: Any, *, gradient: bool) -> float:
        """Evaluate every supplied view before an update; add the prior once."""
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
            transmit(
                view["depth"],
                view["beam"],
                workspace=view["transmission"],
                out_counts=view["prediction"],
                stream=self.stream,
            )
            evaluate_objective(
                view["prediction"],
                view["observed"],
                workspace=view["objective"],
                out_loss=view["loss"],
                out_seed=view["image_seed"] if gradient else None,
                stream=self.stream,
            )
            if gradient:
                transmission_vjp(
                    view["depth"],
                    view["beam"],
                    seed_counts=view["image_seed"],
                    workspace=view["transmission"],
                    out_grad_L=view["depth_seed"],
                    stream=self.stream,
                )
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
            raise NumericalError("The objective is non-finite; inspect counts and model range.")
        if gradient and not bool(self.torch.isfinite(self.gradient).all().item()):
            raise NumericalError("The volume gradient overflowed; discard this evaluation.")
        return loss

    # endregion book:example-reconstruction-volume-gradient

    # region book:example-reconstruction-regularisation
    def regularise(self, volume: Any, *, gradient: bool) -> float:
        """Quadratic physical-space differences; omit edges beyond the grid."""
        self.volume64.copy_(volume)
        penalty = 0.0
        for low, high, grad_low, grad_high, scratch, coefficient in self.edges:
            difference, derivative = scratch
            self.torch.sub(high, low, out=difference)
            if gradient:
                # Scale in FP64 before the FP32 accumulation boundary.
                difference.mul_(coefficient)
                derivative.copy_(difference)
                grad_low.sub_(derivative)
                grad_high.add_(derivative)
                # Restore unscaled differences for the objective.
                self.torch.sub(high, low, out=difference)
            difference.square_()
            self.torch.sum(difference, dim=(0, 1, 2), out=self.scalar)
            penalty += 0.5 * coefficient * float(self.scalar.item())
        return penalty

    # endregion book:example-reconstruction-regularisation

    def inner_product(self, left: Any, right: Any) -> float:
        self.volume64.copy_(left)
        self.edge64.copy_(right)
        self.volume64.mul_(self.edge64)
        self.torch.sum(self.volume64, dim=0, out=self.scalar)
        return float(self.scalar.item())

    def displacement(self, step: float) -> tuple[float, float]:
        # min(g, mu / step) evaluates the projected-gradient mapping
        # without cancellation between almost identical accepted/trial voxels.
        self.volume64.copy_(self.gradient)
        self.edge64.copy_(self.mu).div_(step)
        self.torch.minimum(self.volume64, self.edge64, out=self.volume64)
        self.torch.abs(self.volume64, out=self.edge64)
        mapping = float(self.edge64.max().item())
        self.volume64.mul_(-step).add_(self.mu).clamp_(min=0)
        self.trial.copy_(self.volume64)
        self.torch.sub(self.trial, self.mu, out=self.delta)
        slope = self.inner_product(self.gradient, self.delta)
        return mapping, slope

    # region book:example-reconstruction-projected-armijo
    def solve(
        self, callback: Callable[[int, dict[str, Any]], None] | None = None
    ) -> dict[str, Any]:
        """Solve with optional observation of accepted updates only.

        The callback receives the stable one-based accepted iteration and a
        detached history dictionary after ``mu`` has been updated. It may read
        or export the accepted fields but must not mutate solver state. Callback
        exceptions propagate with the last accepted volume intact.

        An optional fixed mapping step separates the stationarity diagnostic
        from the initial Armijo trial. Omitting it preserves the original
        diagnostic at ``initial_step_mm_inverse_squared``.
        """
        policy = self.policy
        mapping_step = (
            policy.initial_step_mm_inverse_squared
            if policy.mapping_step_mm_inverse_squared is None
            else policy.mapping_step_mm_inverse_squared
        )
        history: list[dict[str, Any]] = []
        reason = "iteration_budget"
        for iteration in range(policy.iterations):
            loss = self.evaluate(self.mu, self.mu_wp, gradient=True)
            step = policy.initial_step_mm_inverse_squared
            mapping, _ = self.displacement(mapping_step)
            if not math.isfinite(mapping):
                raise NumericalError("The trial update overflowed; reduce the initial step.")
            if mapping <= policy.gradient_mapping_tolerance_mm:
                reason = "projected_gradient_tolerance"
                break
            accepted = False
            for attempt in range(policy.maximum_backtracks):
                if step == 0:
                    break
                _, slope = self.displacement(step)
                if not math.isfinite(slope) or slope >= 0:
                    step *= policy.backtracking_factor
                    continue
                # Trial value evaluation leaves the accepted gradient unchanged.
                # Domain/range errors abort with the last accepted volume intact.
                trial_loss = self.evaluate(self.trial, self.trial_wp, gradient=False)
                if trial_loss < loss and trial_loss <= loss + policy.armijo * slope:
                    self.mu.copy_(self.trial)
                    history.append(
                        {
                            "iteration": iteration + 1,
                            "loss_before": loss,
                            "loss_after": trial_loss,
                            "step_mm_inverse_squared": step,
                            "backtracks": attempt,
                            "mapping_mm_before": mapping,
                        }
                    )
                    accepted = True
                    if callback is not None:
                        callback(iteration + 1, dict(history[-1]))
                    break
                step *= policy.backtracking_factor
            if not accepted:
                reason = "line_search_failed"
                break
        # Trial buffers may describe a rejected volume; refresh accepted outputs.
        final_loss = self.evaluate(self.mu, self.mu_wp, gradient=True)
        final_mapping, _ = self.displacement(mapping_step)
        if final_mapping <= policy.gradient_mapping_tolerance_mm:
            reason = "projected_gradient_tolerance"
        return {
            "termination": reason,
            "accepted_steps": len(history),
            "final_objective": final_loss,
            "final_gradient_mapping_mm": final_mapping,
            "history": history,
        }

    # endregion book:example-reconstruction-projected-armijo


def main() -> None:
    args = example_parser(
        "Reconstruct attenuation from supplied calibrated primary counts."
    ).parse_args()
    case = load_case(args.case)
    if not str(args.device).startswith("cuda:"):
        raise ContractError("Select a CUDA device, for example --device cuda:0.")
    torch: Any = importlib.import_module("torch")
    wp: Any = importlib.import_module("warp")
    with case.record(
        args.output,
        entrypoint=__file__,
        metadata={
            "application": "reconstruction",
            "execution_acceptance": "requires separate numerical and physical assessment",
        },
    ) as run:
        # Never build a Torch autograd graph around the manually supplied VJP.
        with torch.cuda.device(args.device), torch.no_grad():
            try:
                solver = Reconstruction(case, device=args.device, torch=torch, wp=wp)
                run.set_metadata(
                    device=args.device,
                    gpu_name=torch.cuda.get_device_name(args.device),
                    torch_version=torch.__version__,
                    volume_units="mm^-1",
                    objective="summed Poisson half-deviance plus quadratic regularisation",
                )
                report = solver.solve()
                wp.synchronize_stream(solver.stream)
                write_array(
                    run, "attenuation.npy", solver.mu.cpu().numpy().reshape(solver.grid.shape)
                )
                for index, view in enumerate(solver.views):
                    write_array(
                        run,
                        f"predictions/view-{index:04d}.npy",
                        view["prediction"].numpy().reshape(view["shape"]),
                    )
                run.write_json("solver.json", report)
            finally:
                # Drain pending work before exception paths release shared owners.
                torch.cuda.synchronize(args.device)


if __name__ == "__main__":
    main()
