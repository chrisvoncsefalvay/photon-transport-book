"""Record formula-based figure payloads; no plotting or photon-history execution."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import cast

from dpt.contracts import ContractError, finite_scalar, integer
from dpt.experiments import (
    RunRecorder,
    experiment_parser,
    experiment_sources,
    private_output,
    repository_root,
)
from dpt.validation.book_illustrations import (
    aperture_difference,
    charge_assignment,
    gaussian_overlap,
    lattice_values,
    moving_interval,
    nuisance_information_fraction,
    quadratic_midpoint,
    slab_interval,
    sphere_chord,
    transmission_difference,
)
from dpt.volumes import GridSpec

ROOT = repository_root(__file__)


def section(configuration: dict[str, object], name: str) -> dict[str, object]:
    value = configuration[name]
    if not isinstance(value, dict):
        raise ContractError(f"configuration {name} must be an object")
    return cast(dict[str, object], value)


def numbers(record: dict[str, object], name: str) -> tuple[float, ...]:
    value = record[name]
    if not isinstance(value, list) or not value:
        raise ContractError(f"configuration {name} must be a nonempty array")
    return tuple(finite_scalar(x, name) for x in cast(list[object], value))


def scalar(record: dict[str, object], name: str) -> float:
    return finite_scalar(record[name], name)


def vector(record: dict[str, object], name: str) -> tuple[float, float, float]:
    value = numbers(record, name)
    if len(value) != 3:
        raise ContractError(f"{name} requires three coordinates")
    return (value[0], value[1], value[2])


def counts(record: dict[str, object], name: str) -> tuple[int, ...]:
    value = record[name]
    if not isinstance(value, list) or not value:
        raise ContractError(f"configuration {name} must be a nonempty integer array")
    return tuple(integer(x, name, minimum=1) for x in cast(list[object], value))


def payloads(config: dict[str, object]) -> dict[str, object]:
    expected = {
        "schema_version",
        "parameter_status",
        "grid",
        "lattice",
        "slabs",
        "quadratic",
        "moving_interval",
        "transmission",
        "sphere",
        "gaussian",
        "nuisance",
        "charge",
        "aperture",
    }
    if set(config) != expected or type(config["schema_version"]) is not int:
        raise ContractError("unsupported analytic-rollout configuration")
    if config["schema_version"] != 1:
        raise ContractError("unsupported analytic-rollout configuration")
    if not isinstance(config["parameter_status"], str) or not config["parameter_status"]:
        raise ContractError("declare which parameters come from prose and which are illustrative")
    result: dict[str, object] = {}
    grid = section(config, "grid")
    shape = counts(grid, "shape")
    if len(shape) != 3:
        raise ContractError("grid shape needs three dimensions")
    specification = GridSpec(
        (shape[0], shape[1], shape[2]), vector(grid, "spacing_mm"), vector(grid, "origin_mm")
    )
    positions = (vector(grid, "first_point_mm"), vector(grid, "second_point_mm"))
    result["4.1"] = {
        "grid": asdict(specification),
        "points": [
            {"object_mm": point, "grid": specification.object_to_grid(point)} for point in positions
        ],
        "formula": "4.3",
        "parameter_status": "worked coordinates and spacings from section4.2; shape illustrative",
    }
    lattice = section(config, "lattice")
    result["4.2"] = {
        "formula": "4.2,4.5",
        "parameters": lattice,
        "rows": [
            asdict(lattice_values(numbers(lattice, "coefficients_mm_inverse"), point))
            for point in numbers(lattice, "grid_coordinates")
        ],
    }
    slabs = section(config, "slabs")
    ray_names = ("crossing", "parallel_miss", "tangent")
    result["4.3"] = {
        "formula": "4.9-4.11",
        "parameters": slabs,
        "rays": {
            name: asdict(
                slab_interval(
                    vector(section(slabs, name), "source_mm"),
                    vector(section(slabs, name), "endpoint_mm"),
                    vector(slabs, "lower_mm"),
                    vector(slabs, "upper_mm"),
                )
            )
            for name in ray_names
        },
    }
    quadratic = section(config, "quadratic")
    result["4.4"] = {
        "formula": "4.15-4.16",
        "parameters": quadratic,
        "rows": [
            asdict(
                quadratic_midpoint(
                    scalar(quadratic, "length_mm"),
                    scalar(quadratic, "intercept_mm_inverse"),
                    scalar(quadratic, "linear_mm_inverse_squared"),
                    scalar(quadratic, "curvature_mm_inverse_cubed"),
                    count,
                )
            )
            for count in counts(quadratic, "sample_counts")
        ],
    }
    interval = section(config, "moving_interval")
    result["5.2"] = {
        "formula": "5.14",
        "parameters": interval,
        "series": [
            {
                "samples": count,
                "rows": [
                    asdict(
                        moving_interval(
                            entry,
                            scalar(interval, "exit_mm"),
                            scalar(interval, "ray_length_mm"),
                            scalar(interval, "attenuation_mm_inverse"),
                            count,
                        )
                    )
                    for entry in numbers(interval, "entry_mm")
                ],
            }
            for count in counts(interval, "sample_counts")
        ],
    }
    transmission = section(config, "transmission")
    result["5.3"] = {
        "formula": "5.16-5.18 with Phi(z)=exp(-(L+s*z))",
        "parameters": transmission,
        "arithmetic": (
            "CPU binary64 exp and subtraction; 100-digit Decimal reference, no CUDA claim"
        ),
        "rows": [
            asdict(
                transmission_difference(
                    scalar(transmission, "optical_depth"), scalar(transmission, "depth_scale"), h
                )
            )
            for h in numbers(transmission, "steps")
        ],
    }
    sphere = section(config, "sphere")
    result["5.4"] = {
        "formula": "5.21",
        "parameters": sphere,
        "rows": [
            asdict(
                sphere_chord(
                    scalar(sphere, "radius_mm"), scalar(sphere, "attenuation_mm_inverse"), b
                )
            )
            for b in numbers(sphere, "impact_mm")
        ],
    }
    gaussian = section(config, "gaussian")
    result["6.1"] = {
        "formula": "6.10-6.11",
        "parameters": gaussian,
        "claim_scope": "analytic infinite-domain overlap, not measured capture range or recovery",
        "series": [
            {
                "width_pixels": width,
                "rows": [
                    asdict(gaussian_overlap(width, shift))
                    for shift in numbers(gaussian, "shifts_pixels")
                ],
            }
            for width in numbers(gaussian, "widths_pixels")
        ],
    }
    nuisance = section(config, "nuisance")
    result["7.4"] = {
        "formula": "7.19",
        "parameters": nuisance,
        "rows": [
            {"cosine": c, "information_fraction": nuisance_information_fraction(c)}
            for c in numbers(nuisance, "cosines")
        ],
    }
    charge = section(config, "charge")
    result["8.3"] = {
        "formula": "8.12",
        "parameters": charge,
        "claim_scope": (
            "stipulated equal-mean event laws; arbitrary unit-event signal, not calibrated detector"
        ),
        "moments": asdict(
            charge_assignment(scalar(charge, "arrival_mean"), numbers(charge, "fractions"))
        ),
    }
    aperture = section(config, "aperture")
    rows = [
        {
            "histories": count,
            "rows": [
                asdict(
                    aperture_difference(
                        scalar(aperture, "width_mm"),
                        scalar(aperture, "edge_mm"),
                        h,
                        scalar(aperture, "source_population"),
                        count,
                    )
                )
                for h in numbers(aperture, "half_steps_mm")
            ],
        }
        for count in counts(aperture, "history_counts")
    ]
    for figure in ("10.2", "10.4"):
        result[figure] = {
            "formula": "10.11,10.17",
            "parameters": aperture,
            "series": rows,
            "claim_scope": (
                "exact uniform-strip moments, not a production moving-boundary estimator"
            ),
        }
    return result


def main() -> None:
    parser = experiment_parser(__file__, __doc__, cuda=False)
    arguments = parser.parse_args()
    output = private_output(arguments.output, ROOT)
    configuration = json.loads(arguments.config.read_text())
    if not isinstance(configuration, dict):
        raise ContractError("configuration must be an object")
    configuration = cast(dict[str, object], configuration)
    names = [
        "python/dpt/validation/book_illustrations.py",
        "python/dpt/contracts.py",
        "python/dpt/experiments.py",
        "python/dpt/geometry.py",
        "python/dpt/volumes.py",
        "experiments/book-analytic-rollouts/run.py",
        "pyproject.toml",
        "uv.lock",
    ]
    names.extend(
        f"src/pages/chapters/{chapter}.mdx"
        for chapter in (
            "volumes-line-integrals",
            "differentiating-projection",
            "recovering-pose",
            "same-pose-different-x-ray",
            "spectral-transport-detector",
            "differentiating-transport",
        )
    )
    sources = experiment_sources(
        __file__, arguments.config, extra={name: ROOT / name for name in names}
    )
    with RunRecorder(
        output,
        configuration=configuration,
        sources=sources,
        metadata={
            "execution": "CPU closed-form/reference evaluation",
            "claim_scope": (
                "mathematical illustrations; no photon histories, anatomy, measurements "
                "or GPU performance"
            ),
            "visualisations": False,
        },
    ) as run:
        for figure, values in payloads(configuration).items():
            run.write_json(f"figure-{figure.replace('.', '-')}.json", values)


if __name__ == "__main__":
    main()
