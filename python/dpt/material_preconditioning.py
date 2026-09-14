"""Fixed voxel scaling from fitting-model Fisher information, prepared on CUDA.

This module estimates a metric; it does not change a loss, gradient, constraint
or stationarity test. It reads the accepted field and fixed acquisition inputs,
never observed counts or reference composition. All image/voxel arrays remain
resident. Preparation is deliberately outside the repeated solver loop.
"""

from __future__ import annotations

# This canonical companion deliberately shares solver-owned preparation scratch.
# It does not make those internal buffers part of the public caller interface.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dpt._reductions import prepare_reduction
from dpt._runtime import load_kernels, require_no_tape
from dpt.contracts import ContractError, NumericalError, finite_scalar
from dpt.material_projection import project_material_paths
from dpt.projection import project_optical_depth, projection_vjp
from dpt.spectral import spectral_signal, spectral_vjp

if TYPE_CHECKING:
    from dpt.material_reconstruction import MaterialReconstruction


@dataclass(frozen=True, slots=True)
class FisherScaling:
    """Resident positive FP32 voxel metric and scalar preparation diagnostics."""

    scale: Any
    diagnostics: dict[str, Any]


def prepare_fisher_scaling(
    solver: MaterialReconstruction, *, minimum_relative_curvature: float = 1.0e-6
) -> FisherScaling:
    """Prepare ``D`` for the existing two-material metric ``D[v] H``.

    At each ray, canonical spectral VJPs form ``B=sum_c j_c j_c.T/lambda_c``.
    Let ``kappa=lambda_max(H^-1/2 B H^-1/2)`` and let ``A`` be the existing
    nonnegative sampled-field line integral. The scalar volume VJP computes
    ``A.T @ (kappa * (A @ 1))``. Jensen's inequality makes its block diagonal
    product with H a majoriser of the local Fisher matrix in exact arithmetic.
    This is not a bound on the complete observed nonlinear Poisson Hessian.

    The regulariser contributes twice each voxel's weighted neighbour degree
    divided by lambda_min(H), a row majorant for its componentwise Laplacian.
    A declared floor relative to the largest total curvature covers unobserved
    or weak voxels without claiming information there. All-zero total curvature
    is rejected. The floored array is divided by its arithmetic mean, retaining
    dimensionless scales of order one; Armijo still decides every actual step.

    Fractions and their gradient are unchanged. Prepared image/path scratch is
    reused, so prediction exports require a fresh evaluation afterwards. The
    returned array owns its storage; ``set_voxel_metric_scale`` can copy it once.
    Path, mean and intermediate cotangent precision follows the spectral
    specification. Fields, the volume adjoint and the final scale remain FP32.
    Unsupported models fail explicitly rather than substituting another metric.
    """
    relative_floor = finite_scalar(
        minimum_relative_curvature, "minimum_relative_curvature", minimum=0
    )
    if not 0 < relative_floor <= 1:
        raise ContractError("minimum_relative_curvature must lie in (0,1]")
    if solver.materials != 2 or solver.material_metric is None:
        raise ContractError("Fisher scaling requires two materials and a fixed positive metric")
    if solver.observation_model != "poisson":
        raise ContractError("Fisher scaling currently requires the independent Poisson model")
    require_no_tape()
    ctx, wp, np = solver.context, solver.wp, solver.np
    ctx.assert_stream(solver.stream)
    kernels = load_kernels("dpt.kernels.material_preconditioning")
    a, b, c = solver.material_metric
    matrix = np.asarray(((a, b), (b, c)), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    minimum_eigenvalue = float(eigenvalues[0])
    inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    if minimum_eigenvalue <= 0 or not np.isfinite(inverse_root).all():
        raise ContractError("material metric inverse square root is not representable")
    whiten = wp.vec3d(inverse_root[0, 0], inverse_root[0, 1], inverse_root[1, 1])
    precision = solver._spectral.spec.precision
    double = precision == "float64"
    value_type = wp.float64 if double else wp.float32
    voxels = solver.grid.voxels
    maximum_pixels = max(view["pixels"] for view in solver.views)
    reduction = prepare_reduction(ctx, voxels)

    def check_status() -> None:
        wp.synchronize_stream(ctx.stream)
        if int(status.numpy()[0]):
            raise NumericalError("Fisher voxel scaling is not finite and representable")

    with ctx.scope():
        ones = wp.full(voxels, 1.0, dtype=wp.float32, device=ctx.device)
        projected = wp.zeros(voxels, dtype=wp.float32, device=ctx.device)
        curvature = wp.empty(voxels, dtype=wp.float64, device=ctx.device)
        scale = wp.empty(voxels, dtype=wp.float32, device=ctx.device)
        ray_ones = wp.full(maximum_pixels, 1.0, dtype=value_type, device=ctx.device)
        mean = wp.empty(maximum_pixels, dtype=value_type, device=ctx.device)
        derivative = wp.empty(2 * maximum_pixels, dtype=value_type, device=ctx.device)
        length = wp.empty(maximum_pixels, dtype=value_type, device=ctx.device)
        seed = wp.empty(maximum_pixels, dtype=value_type, device=ctx.device)
        fisher = wp.empty(maximum_pixels, dtype=wp.vec3d, device=ctx.device)
        extrema = wp.array((math.inf, 0.0), dtype=wp.float64, device=ctx.device)
        counts = wp.zeros(2, dtype=wp.int32, device=ctx.device)
        status = wp.zeros(1, dtype=wp.int32, device=ctx.device)
        total = wp.empty(1, dtype=wp.float64, device=ctx.device)
        solver._prediction_state = "uninitialised"
        solver._spectral.clear_status()
        solver.views[0]["accepted"].validate_inputs(stream=ctx.stream)
        for view in solver.views:
            pixels = view["pixels"]
            projection = view["accepted"].projection
            current_mean, current_derivative = mean[:pixels], derivative[: 2 * pixels]
            current_length, current_seed = length[:pixels], seed[:pixels]
            current_fisher, current_ones = fisher[:pixels], ray_ones[:pixels]
            current_fisher.zero_()
            project_material_paths(view["pose"], workspace=view["accepted"], stream=ctx.stream)
            for channel in view["channels"]:
                spectral_signal(
                    view["paths"],
                    solver._coefficients,
                    channel["weights"],
                    channel["response"],
                    out_mean=current_mean,
                    workspace=solver._spectral,
                    stream=ctx.stream,
                )
                spectral_vjp(
                    view["paths"],
                    solver._coefficients,
                    channel["weights"],
                    channel["response"],
                    seed=current_ones,
                    out_grad_paths=current_derivative,
                    workspace=solver._spectral,
                    stream=ctx.stream,
                )
                wp.launch(
                    kernels.get_add_fisher(double),
                    dim=pixels,
                    inputs=[current_mean, current_derivative, pixels, current_fisher, status],
                    stream=ctx.stream,
                    record_tape=False,
                )
            project_optical_depth(
                ones,
                view["pose"],
                out_L=current_length,
                workspace=projection,
                stream=ctx.stream,
            )
            wp.launch(
                kernels.get_fisher_row_seed(double),
                dim=pixels,
                inputs=[current_fisher, current_length, whiten, current_seed, status],
                stream=ctx.stream,
                record_tape=False,
            )
            check_status()
            projection_vjp(
                ones,
                view["pose"],
                adj_L=current_seed,
                out_mu=projected,
                workspace=projection,
                stream=ctx.stream,
                accumulate=True,
            )
        wp.launch(
            kernels.add_regulariser,
            dim=voxels,
            inputs=[
                projected,
                *solver.grid.shape[::-1],
                wp.vec3d(*solver._penalty_coefficients),
                minimum_eigenvalue,
                curvature,
                extrema,
                counts,
                status,
            ],
            stream=ctx.stream,
            record_tape=False,
        )
        check_status()
        low, high = map(float, extrema.numpy())
        if high <= 0:
            raise ContractError("no positive Fisher or regularisation curvature; scaling undefined")
        floor = high * relative_floor
        if not math.isfinite(floor) or floor <= 0:
            raise NumericalError("the declared relative curvature floor is not representable")
        wp.launch(
            kernels.floor_curvature,
            dim=voxels,
            inputs=[curvature, floor, counts],
            stream=ctx.stream,
            record_tape=False,
        )
        reduction.finish(voxels, total, status, source=curvature)
        check_status()
        normaliser = float(total.numpy()[0]) / voxels
        wp.launch(
            kernels.normalise_curvature,
            dim=voxels,
            inputs=[curvature, normaliser, scale, status],
            stream=ctx.stream,
            record_tape=False,
        )
        check_status()
        zero_data, floored = map(int, counts.numpy())
    return FisherScaling(
        scale,
        {
            "method": "local Poisson Fisher row majorant plus regulariser row majorant",
            "field_state": "accepted field at preparation; fixed for the following solve",
            "material_metric": solver.material_metric,
            "minimum_material_metric_eigenvalue": minimum_eigenvalue,
            "normalisation": "arithmetic mean of floored curvature",
            "normaliser": normaliser,
            "minimum_raw_curvature": low,
            "maximum_raw_curvature": high,
            "minimum_relative_curvature": relative_floor,
            "curvature_floor": floor,
            "minimum_scale": max(low, floor) / normaliser,
            "maximum_scale": high / normaliser,
            "zero_data_curvature_voxels": zero_data,
            "floored_voxels": floored,
            "voxels": voxels,
            "views": len(solver.views),
            "scale_bytes": voxels * 4,
            "path_signal_precision": precision,
            "accumulation": "canonical FP32 volume VJP; FP64 Fisher algebra and sum reduction",
            "scope": "positive algorithmic metric; not observed-Hessian or identifiability proof",
        },
    )
