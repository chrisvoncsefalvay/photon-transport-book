"""Export numerical book payloads from the canonical library, without rendering.

Analytic fixture definitions and independent checks are recorded beside actual
CUDA results. CPU work is limited to ingestion, small offline diagnostics and
references. No field, spectral, detector or derivative kernel is reimplemented.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from typing import Any

from dpt.contracts import ContractError, finite_scalar, integer
from dpt.detector import (
    BlurSpec,
    ObservationIdentity,
    blur,
    prepare_blur,
    prepare_detector,
    sample_compound_poisson,
)
from dpt.experiments import (
    RunRecorder,
    experiment_parser,
    experiment_sources,
    private_output,
    repository_root,
)
from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
from dpt.materials import Provenance
from dpt.projection import (
    ProjectionSpec,
    prepare_projection,
    project_optical_depth,
    projection_vjp,
)
from dpt.spectral import (
    SpectralSpec,
    prepare_spectral,
    spectral_bin_counts,
    spectral_signal,
    spectral_vjp,
)
from dpt.transmission import (
    TransmissionSpec,
    prepare_transmission,
    transmission_vjp,
    transmit,
)
from dpt.validation.identifiability import sensitivity_spectrum
from dpt.validation.projection import integrate_sampled_field, quadratic_sample_values
from dpt.validation.recovery import detector_coordinate
from dpt.volumes import GridSpec

ROOT = repository_root(__file__)


@dataclass(frozen=True, slots=True)
class Configuration:
    schema_version: int
    thickness_max_mm: float
    thickness_samples: int
    noise_samples_per_condition: int
    noise_seed: int
    projection_width: int
    projection_samples_per_ray: int

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported book-rollout schema")
        if finite_scalar(self.thickness_max_mm, "maximum thickness", minimum=0) == 0:
            raise ContractError("maximum thickness must be positive")
        integer(self.thickness_samples, "thickness samples", minimum=2, maximum=10001)
        integer(self.noise_samples_per_condition, "noise samples", minimum=1024, maximum=2**22)
        integer(self.noise_seed, "noise seed", maximum=2**64 - 1)
        integer(self.projection_width, "diagnostic image width", minimum=8, maximum=64)
        integer(self.projection_samples_per_ray, "projection samples", minimum=1, maximum=4096)


def _provenance(definition: str) -> Provenance:
    return Provenance(
        source="experiments/book-rollouts/run.py: " + definition,
        sha256=hashlib.sha256(definition.encode()).hexdigest(),
        rights="Author-defined mathematical fixture; no third-party data",
        description=definition + "; not physical measurements or anatomical material",
    )


def _spectral(config: Configuration, run: RunRecorder, wp: Any, np: Any, device: Any) -> None:
    thickness = np.linspace(0, config.thickness_max_mm, config.thickness_samples, dtype=np.float32)
    paths = wp.array(thickness, dtype=wp.float32, device=device)
    weights = wp.array([0.5, 0.5], dtype=wp.float32, device=device)
    response = wp.ones(2, dtype=wp.float32, device=device)
    mean = wp.empty(config.thickness_samples, dtype=wp.float32, device=device)
    derivative = wp.empty_like(mean)
    bins = wp.empty(2 * config.thickness_samples, dtype=wp.float32, device=device)
    seed = wp.ones(config.thickness_samples, dtype=wp.float32, device=device)
    for chapter, coefficients in ((7, (0.01, 0.03)), (8, (0.02, 0.04))):
        definition = f"Chapter {chapter}: two components mu={coefficients} mm^-1, weights=(0.5,0.5)"
        provenance = _provenance(definition)
        spec = SpectralSpec(1, 2, (provenance,), provenance, provenance, "counts", definition)
        workspace = prepare_spectral(spec, max_pixels=config.thickness_samples, device=str(device))
        mu = wp.array(coefficients, dtype=wp.float32, device=device)
        spectral_signal(paths, mu, weights, response, out_mean=mean, workspace=workspace)
        spectral_bin_counts(paths, mu, weights, response, out_bin_counts=bins, workspace=workspace)
        spectral_vjp(
            paths, mu, weights, response, seed=seed, out_grad_paths=derivative, workspace=workspace
        )
        actual, grad = mean.numpy().astype(np.float64), derivative.numpy().astype(np.float64)
        stored_mu = mu.numpy().tolist()
        exact_input: list[float] = []
        exact_grad: list[float] = []
        ideal_decimal: list[float] = []
        with localcontext() as context:
            context.prec = 60
            for distance in thickness:
                d = Decimal.from_float(float(distance))
                terms = [(-Decimal.from_float(m) * d).exp() / 2 for m in stored_mu]
                exact_input.append(float(sum(terms)))
                exact_grad.append(
                    float(
                        -sum(
                            Decimal.from_float(m) * t for m, t in zip(stored_mu, terms, strict=True)
                        )
                    )
                )
                ideal_decimal.append(
                    float(sum((-Decimal(str(m)) * d).exp() / 2 for m in coefficients))
                )
        value_relative = float(np.max(np.abs(actual - exact_input) / np.asarray(exact_input)))
        grad_relative = float(np.max(np.abs(grad - exact_grad) / np.abs(exact_grad)))
        if max(value_relative, grad_relative) > 2e-7:
            raise RuntimeError("spectral payload failed its declared exact-input relative budget")
        contributions = bins.numpy().reshape(2, -1).astype(np.float64)
        run.write_json(
            f"spectral-chapter-{chapter}.json",
            {
                "figure_ids": [f"figure-{chapter}-2" if chapter == 7 else "figure-8-1"],
                "fixture": definition,
                "provenance": asdict(provenance),
                "thickness_mm": thickness.tolist(),
                "declared_coefficients_mm_inverse": coefficients,
                "represented_coefficients_mm_inverse": stored_mu,
                "incident_count_weights": [0.5, 0.5],
                "transmission_cuda": actual.tolist(),
                "transmission_derivative_per_mm_cuda": grad.tolist(),
                "transmitted_component_means_cuda": contributions.tolist(),
                "effective_optical_depth_from_stored_mean": (-np.log(actual)).tolist(),
                "effective_slope_per_mm_from_stored_values": (-grad / actual).tolist(),
                "transmitted_component_fractions_from_stored_means": (
                    contributions / contributions.sum(axis=0)
                ).tolist(),
                "independent_decimal_exact_input_transmission": exact_input,
                "independent_decimal_exact_input_derivative": exact_grad,
                "independent_ideal_decimal_coefficient_transmission": ideal_decimal,
                "max_relative_value_error": value_relative,
                "max_relative_derivative_error": grad_relative,
                "relative_acceptance_budget": 2e-7,
                "rounding_note": (
                    "Derived log/slope/fractions use downloaded rounded outputs; "
                    "Decimal references separately identify ideal decimal and actual "
                    "binary32 inputs."
                ),
            },
        )


def _noise(config: Configuration, run: RunRecorder, wp: Any, np: Any, device: Any) -> None:
    samples = config.noise_samples_per_condition
    workspace = prepare_detector(max_pixels=samples, device=str(device))
    out = wp.empty(samples, dtype=wp.float32, device=device)
    rows: list[dict[str, object]] = []
    for condition, (name, rate, gain) in enumerate(
        (("baseline", 20.0, 1.0), ("double-exposure", 40.0, 1.0), ("double-gain", 20.0, 2.0))
    ):
        rates = wp.full(samples, rate, dtype=wp.float32, device=device)
        scores = wp.array([gain], dtype=wp.float32, device=device)
        identity = ObservationIdentity(config.noise_seed, condition)
        sample_compound_poisson(
            rates,
            scores,
            energies=1,
            shared_scores=True,
            out_signal=out,
            identity=identity,
            workspace=workspace,
        )
        values = out.numpy().astype(np.float64)
        mean, variance = float(values.mean()), float(values.var(ddof=1))
        reference_mean, reference_variance = rate * gain, rate * gain**2
        mean_se = math.sqrt(reference_variance / samples)
        # Exact variance of unbiased sample variance for gain*Poisson(rate).
        variance_se = gain**2 * math.sqrt((rate + 2 * rate**2 * samples / (samples - 1)) / samples)
        mean_z = (mean - reference_mean) / mean_se
        variance_z = (variance - reference_variance) / variance_se
        bins, counts = np.unique(values, return_counts=True)
        rows.append(
            {
                "condition": name,
                "detected_photon_mean": rate,
                "signal_per_photon": gain,
                "identity": asdict(identity),
                "independent_pixel_draws": samples,
                "sample_mean": mean,
                "unbiased_sample_variance": variance,
                "reference_mean": reference_mean,
                "reference_variance": reference_variance,
                "mean_error_in_standard_errors": mean_z,
                "variance_error_in_standard_errors": variance_z,
                "signal_histogram_values": bins.tolist(),
                "signal_histogram_counts": counts.tolist(),
            }
        )
        if max(abs(mean_z), abs(variance_z)) > 7:
            raise RuntimeError(f"{name} observation moments failed the declared seven-sigma check")
    run.write_json(
        "detector-noise.json",
        {
            "figure_ids": ["figure-7-1"],
            "fixture": (
                "Uniform analytic photon rate; zero electronic offset/read noise; "
                "independent draws across pixel counters and condition identities"
            ),
            "signal_unit": "arbitrary signal units",
            "sampling_operator": "dpt.detector.sample_compound_poisson",
            "acceptance": (
                "Mean and unbiased variance within seven exact standard errors, fixed "
                "before execution"
            ),
            "conditions": rows,
        },
    )


def _blur(run: RunRecorder, wp: Any, np: Any, device: Any) -> None:
    width = 129
    coordinate = np.arange(width) - width // 2
    source_host = np.where(np.abs(coordinate) <= 12, (1 - (coordinate / 13) ** 2) ** 2, 0).astype(
        np.float32
    )
    source = wp.array(source_host, dtype=wp.float32, device=device)
    shifted_source = wp.array(np.roll(source_host, 1), dtype=wp.float32, device=device)
    base_kernel = (0.0, 0.125, 0.75, 0.125, 0.0)
    # The canonical operator indexes source[p+offset]: decreasing the kernel
    # offset by one shifts its response towards increasing pixel coordinates.
    shifted_kernel = (0.125, 0.75, 0.125, 0.0, 0.0)
    provenance = _provenance(
        "Compact polynomial signal; passive weights=(0,1/8,3/4,1/8,0); one-pixel translations"
    )
    base = prepare_blur(BlurSpec(1, width, 1, 5, base_kernel, provenance), device=str(device))
    shifted = prepare_blur(BlurSpec(1, width, 1, 5, shifted_kernel, provenance), device=str(device))
    out_a = wp.empty(width, dtype=wp.float32, device=device)
    out_b = wp.empty_like(out_a)
    blur(shifted_source, out_signal=out_a, workspace=base)
    blur(source, out_signal=out_b, workspace=shifted)
    a, b = out_a.numpy(), out_b.numpy()
    if not np.array_equal(a, b):
        raise RuntimeError("finite-support blur translation identity failed")
    run.write_json(
        "blur-displacement.json",
        {
            "figure_ids": ["figure-7-3"],
            "pixel_coordinate": coordinate.tolist(),
            "provenance": asdict(provenance),
            "input_signal": source_host.tolist(),
            "shifted_input_signal": np.roll(source_host, 1).tolist(),
            "base_stencil": base_kernel,
            "shifted_stencil": shifted_kernel,
            "shifted_feature_filtered_cuda": a.tolist(),
            "shifted_kernel_filtered_cuda": b.tolist(),
            "bitwise_equal": True,
            "boundary": "zero; compact input support stays clear of all detector edges",
            "index_convention": (
                "B x[p] = sum_j h[j] x[p+j-centre]; kernel array shift -1 yields "
                "output displacement +1"
            ),
        },
    )


def _projection(config: Configuration, run: RunRecorder, wp: Any, np: Any, device: Any) -> None:
    width = config.projection_width
    grid = GridSpec((17, 19, 21), (0.7, 0.9, 1.1), (-6.8, -8.2, -8.3))
    geometry = DetectorGeometry(
        (-80, 0.37, -0.21),
        (80, -20, -20),
        (0, 1, 0),
        (0, 0, 1),
        (40 / (width - 1), 40 / (width - 1)),
        (width, width),
    )
    anchor = compose_pose(RigidTransform(), (0.23, -0.14, 0.17, 0.03, -0.02, 0.01))
    field = wp.array(
        quadratic_sample_values(grid, intercept=0.01, curvature=(0.0003, 0.0005, 0.0002)),
        dtype=wp.float32,
        device=device,
    )
    pose = wp.array(anchor.packed(), dtype=wp.float64, device=device)
    depth = wp.empty(geometry.pixels, dtype=wp.float32, device=device)
    counts = wp.empty_like(depth)
    projector = prepare_projection(
        grid, geometry, ProjectionSpec(config.projection_samples_per_ray), device=str(device)
    )
    transmission = prepare_transmission(
        TransmissionSpec(beam="scalar"),
        max_pixels=geometry.pixels,
        device=str(device),
        stream=projector.stream,
    )
    project_optical_depth(field, pose, out_L=depth, workspace=projector)
    transmit(depth, 1000.0, out_counts=counts, workspace=transmission)
    # A deliberately bounded diagnostic extracts Jacobian rows via basis VJPs.
    # Production recovery uses one contracted VJP and never stores this matrix.
    seed = wp.empty(geometry.pixels, dtype=wp.float32, device=device)
    pinned = wp.zeros(geometry.pixels, dtype=wp.float32, device="cpu", pinned=True)
    host_seed = pinned.numpy()
    gradient = wp.empty(6, dtype=wp.float64, device=device)
    count_depth_seed = wp.empty_like(seed)
    jacobian = np.empty((geometry.pixels, 6), dtype=np.float64)
    count_jacobian = np.empty_like(jacobian)
    for pixel in range(geometry.pixels):
        if pixel:
            host_seed[pixel - 1] = 0
        host_seed[pixel] = 1
        wp.copy(seed, pinned, stream=projector.stream)
        projection_vjp(field, pose, adj_L=seed, out_pose=gradient, workspace=projector)
        jacobian[pixel] = gradient.numpy()
        transmission_vjp(
            depth,
            1000.0,
            seed_counts=seed,
            out_grad_L=count_depth_seed,
            workspace=transmission,
        )
        projection_vjp(field, pose, adj_L=count_depth_seed, out_pose=gradient, workspace=projector)
        count_jacobian[pixel] = gradient.numpy()
    # Check the exported diagnostic against a separately contracted VJP.
    host_seed[:] = [(-1.0) ** p * (0.5 + (p % 7) / 8) for p in range(geometry.pixels)]
    wp.copy(seed, pinned, stream=projector.stream)
    projection_vjp(field, pose, adj_L=seed, out_pose=gradient, workspace=projector)
    aggregate = gradient.numpy()
    expected = np.asarray(
        [
            math.fsum(jacobian[p, axis] * float(host_seed[p]) for p in range(geometry.pixels))
            for axis in range(6)
        ]
    )
    if not np.allclose(aggregate, expected, rtol=2e-12, atol=2e-12):
        raise RuntimeError("exported projection Jacobian failed aggregate VJP consistency")
    stored = field.numpy().tolist()
    checks: list[dict[str, object]] = []
    for row, column in ((0, 0), (width // 2, width // 2), (width - 1, width - 1)):
        reference = integrate_sampled_field(
            stored,
            grid,
            geometry,
            anchor,
            row,
            column,
            midpoint_samples=config.projection_samples_per_ray,
        )
        actual = float(depth.numpy()[row * width + column])
        if abs(actual - reference) > 2e-7 * max(1.0, abs(reference)):
            raise RuntimeError("projection image failed independent discrete reference")
        checks.append(
            {"row": row, "column": column, "cuda": actual, "independent_reference": reference}
        )
    depth_host = depth.numpy().astype(np.float64)
    # At unit electronic gain the log-gain image derivative is the input mean.
    # Count pose derivatives above pass through both canonical adjoints and
    # their declared binary32 intermediate cotangent boundary.
    gain_derivative = counts.numpy().astype(np.float64)
    scales = (1.0, 1.0, 1.0, 0.01, 0.01, 0.01)
    extended = np.column_stack((count_jacobian, gain_derivative))
    cosines: list[dict[str, object]] = []
    for axis in range(6):
        column = count_jacobian[:, axis]
        cosine = float(
            np.dot(column, gain_derivative)
            / (np.linalg.norm(column) * np.linalg.norm(gain_derivative))
        )
        cosines.append(
            {
                "axis": axis,
                "cosine_with_log_gain": cosine,
                "information_fraction_after_one_log_gain": max(0.0, 1 - cosine**2),
            }
        )
    mismatch = 0.03 * gain_derivative
    scaled_j = count_jacobian * np.asarray(scales)
    bias, _, retained_rank, singular_values = np.linalg.lstsq(scaled_j, mismatch, rcond=1e-8)
    explained = scaled_j @ bias
    residual = mismatch - explained
    orthogonality_norm = float(np.linalg.norm(scaled_j.T @ residual))
    run.write_json(
        "projection-sensitivity.json",
        {
            "figure_ids": ["figure-5-1", "figure-6-3", "figure-7-4"],
            "fixture": (
                "Declared positive quadratic attenuation, not anatomy; same-model "
                "local analytic diagnostic"
            ),
            "grid": asdict(grid),
            "geometry": asdict(geometry),
            "anchor": asdict(anchor),
            "field_values_mm_inverse": stored,
            "samples_per_ray": config.projection_samples_per_ray,
            "optical_depth_cuda": depth_host.tolist(),
            "expected_counts_cuda": counts.numpy().tolist(),
            "local_depth_jacobian_cuda": jacobian.tolist(),
            "local_count_jacobian_chain": count_jacobian.tolist(),
            "count_jacobian_definition": (
                "Canonical transmission_vjp then projection_vjp; "
                "binary32 depth cotangents and binary64 pose cotangents"
            ),
            "coordinate_order": ["tx_mm", "ty_mm", "tz_mm", "rx_rad", "ry_rad", "rz_rad"],
            "chart": "local right SE3 around recorded anchor",
            "parameter_scales": scales,
            "precision_weights": "identity; no noise covariance or statistical information claim",
            "pose_sensitivity": asdict(sensitivity_spectrum(count_jacobian.tolist(), scales)),
            "pose_and_log_gain_sensitivity": asdict(
                sensitivity_spectrum(extended.tolist(), (*scales, 1.0))
            ),
            "pose_gain_angles": cosines,
            "mismatch_decomposition": {
                "mean_mismatch": mismatch.tolist(),
                "pose_explainable": explained.tolist(),
                "orthogonal_residual": residual.tolist(),
                "local_scaled_pose_bias": bias.tolist(),
                "lstsq_relative_cutoff": 1e-8,
                "lstsq_retained_rank": int(retained_rank),
                "lstsq_singular_values": singular_values.tolist(),
                "normal_equation_residual_norm": orthogonality_norm,
                "orthogonality_scope": (
                    "Projection onto the retained singular subspace; "
                    "full-column orthogonality requires full retained rank"
                ),
                "claim_scope": (
                    "linearised least squares at fixed pose; no nonlinear recovery claim"
                ),
            },
            "independent_value_checks": checks,
            "aggregate_vjp_max_absolute_discrepancy": float(np.max(np.abs(aggregate - expected))),
            "diagnostic_storage": (
                "Bounded offline Jacobian, never allocated or retained by the production optimiser"
            ),
        },
    )
    ray_points = [
        (float(x), 0.37 + (x + 80) * 0.02, -0.21 + (x + 80) * 0.01) for x in (-40, -20, 0, 20, 40)
    ]
    locations = [detector_coordinate(geometry, point) for point in ray_points]
    if max(math.dist(locations[0], location) for location in locations) > 1e-10:
        raise RuntimeError("canonical perspective ray invariance failed")
    indices = [(0.0, 0.0, 0.0), (3.0, 4.0, 5.0), (3.25, 4.5, 5.75)]
    positions = [grid.grid_to_object(point) for point in indices]
    if (
        max(
            math.dist(index, grid.object_to_grid(position))
            for index, position in zip(indices, positions, strict=True)
        )
        > 1e-12
    ):
        raise RuntimeError("anisotropic grid coordinate round trip failed")
    run.write_json(
        "geometry.json",
        {
            "figure_ids": ["figure-4-1", "figure-6-2"],
            "grid": asdict(grid),
            "indices_xyz": indices,
            "positions_object_mm": positions,
            "support_grid_coordinates": grid.support,
            "detector_geometry": asdict(geometry),
            "points_on_one_source_ray_mm": ray_points,
            "continuous_detector_coordinates_column_row": locations,
            "claim_scope": (
                "canonical coordinate transforms and perspective point invariance; "
                "conceptual arrow/layout design deferred"
            ),
        },
    )


def main() -> None:
    parser = experiment_parser(__file__, __doc__)
    args = parser.parse_args()
    config = Configuration(**json.loads(args.config.read_text()))
    output = private_output(args.output, ROOT)
    sources = experiment_sources(__file__, args.config)
    with RunRecorder(output, configuration=asdict(config), sources=sources) as run:
        wp: Any = importlib.import_module("warp")
        np: Any = importlib.import_module("numpy")
        device = wp.get_device(args.device)
        if not device.is_cuda:
            raise ContractError("book operator rollouts require actual CUDA")
        run.set_metadata(
            device=str(device),
            device_name=device.name,
            architecture=device.arch,
            cuda_driver=device.runtime.driver_version,
            fixture="analytic software cases, never measured physical inputs",
            visualisation="none",
        )
        _spectral(config, run, wp, np, device)
        _noise(config, run, wp, np, device)
        _blur(run, wp, np, device)
        _projection(config, run, wp, np, device)


if __name__ == "__main__":
    main()
