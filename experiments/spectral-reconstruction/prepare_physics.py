"""Freeze traceable water/bone coefficients and an explicit simulated acquisition.

CPU preparation only; requires NumPy and exactly SpekPy 2.5.4. Source HTML,
licensing evidence and package records are supplied from a private directory.
No scanner measurements, material labels or patient VMI values enter this model.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import html
import importlib.metadata
import json
import platform
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import spekpy

from dpt.materials import MaterialTable, Provenance
from dpt.spectral import SpectralSpec, Spectrum

SPEKPY_VERSION = "2.5.4"
NIST_RIGHTS = (
    "NIST SRD 126; dataset metadata declares https://www.nist.gov/open/license. "
    "The policy reserves SRD copyright. Private research preparation with "
    "attribution; public redistribution of source or derived tables is not "
    "cleared by this preparation. See retained policy and dataset metadata."
)
TABLES = (
    ("water", "Water, Liquid", "water.html", 1.0),
    ("cortical_bone", "Bone, Cortical (ICRU-44)", "bone.html", 1.92),
)
PATH_PAIRS_MM = (
    (0.0, 0.0),
    (100.0, 5.0),
    (200.0, 10.0),
    (300.0, 20.0),
    (400.0, 40.0),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_sources(source_root: Path) -> dict[str, dict[str, Any]]:
    """Check retained original bytes; never accept an unrecorded source table."""
    intent = source_root / "source-license-manifest.json"
    if not intent.is_file():
        raise ValueError("source/licence intent must be recorded before source intake")
    records = json.loads((source_root / "source-records.json").read_text())
    result = {}
    for record in records:
        path = (source_root / record["path"]).resolve()
        if not path.is_relative_to(source_root.resolve()):
            raise ValueError("source record escapes its supplied source root")
        if digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"source digest/size mismatch: {path}")
        result[path.name] = record
    required = {
        "water.html",
        "bone.html",
        "materials-tab2.html",
        "nist-rights.html",
        "nist-disclaimer.html",
        "nist-srd126-metadata.json",
        "spekpy-license.txt",
        "spekpy-pypi-2.5.4.json",
    }
    if required - result.keys():
        raise ValueError(f"missing source/licence records: {required - result.keys()}")
    package = json.loads((source_root / "package-record.json").read_text())
    wheel = source_root / "packages" / package["record"]["filename"]
    if digest(wheel) != package["record"]["digests"]["sha256"]:
        raise ValueError("retained SpekPy wheel does not match official PyPI SHA-256")
    if importlib.metadata.version("spekpy") != SPEKPY_VERSION:
        raise ValueError(f"this preparation requires SpekPy {SPEKPY_VERSION}")
    return result


def read_table(
    source_root: Path, record: dict[str, Any], name: str, density: float
) -> MaterialTable:
    """Read the ASCII table once, preserving both sides of absorption edges."""
    text = (source_root / record["path"]).read_text()
    blocks = re.findall(r"<PRE>(.*?)</PRE>", text, flags=re.I | re.S)
    if len(blocks) != 1:
        raise ValueError("expected exactly one NIST ASCII table")
    number = r"[0-9]+\.[0-9]+E[+-][0-9]+"
    pattern = rf"^\s*(?:\d+\s+K\s+)?({number})\s+({number})\s+({number})\s*$"
    rows = re.findall(pattern, html.unescape(blocks[0]), flags=re.M)
    if len(rows) < 30:
        raise ValueError("incomplete NIST coefficient table")
    return MaterialTable(
        name=name,
        energies_kev=tuple(float(row[0]) * 1000 for row in rows),
        coefficients=tuple(float(row[1]) for row in rows),
        units="cm^2/g",
        density_g_cm3=density,
        interpolation="log-log",
        provenance=Provenance(
            source=record["url"],
            sha256=record["sha256"],
            rights=NIST_RIGHTS,
            description=(
                f"NIST SRD 126 total mass attenuation, first coefficient column; "
                f"{name}, reference density {density} g/cm^3 from NIST table 2. "
                "Energy-absorption coefficients are not used."
            ),
        ),
    )


def reference_materials(
    source_root: Path, records: dict[str, dict[str, Any]]
) -> tuple[list[MaterialTable], list[dict[str, Any]], dict[str, Any]]:
    composition_html = (source_root / "sources/materials-tab2.html").read_text()
    tables, descriptions = [], []
    for name, label, filename, density in TABLES:
        match = re.search(
            rf"<TR[^>]*>\s*<TD[^>]*>{re.escape(label)}</TD>(.*?)</TR>",
            composition_html,
            flags=re.I | re.S,
        )
        if match is None:
            raise ValueError(f"NIST table 2 lacks {label}")
        cells = re.findall(r"<TD[^>]*>(.*?)</TD>", match[1], flags=re.I | re.S)
        if len(cells) != 4 or float(cells[2].strip()) != density:
            raise ValueError(f"unexpected reference density for {label}")
        composition = {
            int(z): float(fraction) for z, fraction in re.findall(r"(\d+):\s*([\d.]+)", cells[3])
        }
        np.testing.assert_allclose(sum(composition.values()), 1, atol=1e-12)
        table = read_table(source_root, records[filename], name, density)
        tables.append(table)
        descriptions.append(
            {
                "name": name,
                "reference_name": label,
                "density_g_cm3": density,
                "element_mass_fractions_by_atomic_number": composition,
                "density_composition_source": records["materials-tab2.html"],
                "table_rows": len(table.energies_kev),
                "duplicate_edge_pairs": len(table.energies_kev) - len(set(table.energies_kev)),
            }
        )
    # Independently transcribed reference rows check column selection and units.
    energies = (20.0, 40.0, 80.0, 100.0)
    mass_reference = np.array(
        [
            [0.8096, 0.2683, 0.1837, 0.1707],
            [4.001, 0.6655, 0.2229, 0.1855],
        ]
    )
    expected = mass_reference * np.array([1.0, 1.92])[:, None] / 10
    actual = np.array([table.at_energies(energies, edge_side="above") for table in tables])
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    # The calcium K-edge remains a jump; ingestion must not merge its two rows.
    bone = tables[1]
    edge = next(e for e, f in pairwise(bone.energies_kev) if e == f and e > 4)
    below = bone.at_energies((edge,), edge_side="below")[0]
    above = bone.at_energies((edge,), edge_side="above")[0]
    np.testing.assert_allclose([below, above], np.array([129.6, 333.2]) * 1.92 / 10)
    return (
        tables,
        descriptions,
        {
            "reference_energies_kev": energies,
            "reference_total_mass_attenuation_cm2_g": mass_reference.tolist(),
            "reference_linear_attenuation_mm_inv": actual.tolist(),
            "calcium_edge_kev": edge,
            "calcium_edge_below_above_mm_inv": [below, above],
            "status": "passed",
        },
    )


def channel_response(energies: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    return np.asarray(
        [(energies >= low) & (energies < high) for low, high in pairwise(thresholds)],
        dtype=np.float64,
    )


def sensitivity(
    mu: np.ndarray,
    weights: np.ndarray,
    response: np.ndarray,
) -> list[dict[str, Any]]:
    """Poisson Fisher square root with derivatives in counts per path millimetre."""
    records = []
    for pair in PATH_PAIRS_MM:
        path = np.asarray(pair)
        transmitted = response * weights * np.exp(-path @ mu)
        means = transmitted.sum(axis=1)
        jacobian = -transmitted @ mu.T
        whitened = jacobian / np.sqrt(means[:, None])
        singular = np.linalg.svd(whitened, compute_uv=False)
        rank = int(np.linalg.matrix_rank(whitened))
        # Independent finite differences of the mean, with no analytic Jacobian.
        step = 1e-3
        differences = []
        for axis in np.eye(2):
            plus = response @ (weights * np.exp(-(path + step * axis) @ mu))
            minus = response @ (weights * np.exp(-(path - step * axis) @ mu))
            differences.append((plus - minus) / (2 * step))
        finite_difference = np.asarray(differences).T
        np.testing.assert_allclose(jacobian, finite_difference, rtol=1e-7, atol=1e-10)
        if rank != 2 or np.any(means <= 0) or np.any(means >= 2**24):
            raise ValueError("count range or two-material local identifiability failed")
        records.append(
            {
                "water_bone_paths_mm": pair,
                "expected_counts": means.tolist(),
                "poisson_whitened_path_jacobian": whitened.tolist(),
                "singular_values_per_mm": singular.tolist(),
                "rank": rank,
                "condition_number": float(singular[0] / singular[-1]),
                "local_unbiased_path_crlb_sd_mm": np.sqrt(
                    np.diag(np.linalg.inv(whitened.T @ whitened))
                ).tolist(),
                "finite_difference_check": "passed; central difference step 0.001 mm",
            }
        )
    return records


def prepare(source_root: Path, output: Path, photons: float) -> None:
    source_root, output = source_root.resolve(), output.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo):
        raise ValueError("physical inputs must remain outside the repository")
    if not np.isfinite(photons) or not 0 < photons < 2**24:
        raise ValueError("photon budget must be finite, positive and below 2^24")
    records = verify_sources(source_root)
    tables, materials, reference_checks = reference_materials(source_root, records)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True, exist_ok=False)
    configuration = {
        "kvp": 120.0,
        "th": 12.0,
        "dk": 1.0,
        "physics": "casim",
        "mu_data_source": "nist",
        "x": 0.0,
        "y": 0.0,
        "z": 100.0,
        "mas": 1.0,
        "brem": True,
        "char": True,
        "obli": True,
        "targ": "W",
        "shift": 0.0,
        "trans": False,
    }
    spectrum = spekpy.Spek(**configuration).filter("Al", 2.5)
    native_energy, native_density = spectrum.get_spectrum(edges=False, flu=True, diff=True)
    energy_again, native_bins = spectrum.get_spectrum(edges=False, flu=True, diff=False)
    np.testing.assert_array_equal(native_energy, energy_again)
    np.testing.assert_allclose(native_bins, native_density * configuration["dk"], rtol=1e-14)
    np.testing.assert_array_equal(native_energy, np.arange(1.5, 120.0, 1.0))
    # A non-unit width exposes an accidental second energy-width multiplication.
    wide = spekpy.Spek(**(configuration | {"dk": 2.0})).filter("Al", 2.5)
    _, wide_density = wide.get_spectrum(edges=False, flu=True, diff=True)
    _, wide_bins = wide.get_spectrum(edges=False, flu=True, diff=False)
    np.testing.assert_allclose(wide_bins, 2.0 * wide_density, rtol=1e-14)
    np.savetxt(
        stage / "spekpy-native-spectrum.csv",
        np.column_stack([native_energy, native_density, native_bins]),
        delimiter=",",
        header="energy_kev,density_photons_cm2_kev,bin_photons_cm2",
        comments="",
        fmt="%.17g",
    )
    retained = native_energy >= 20
    energies = native_energy[retained]
    incident = native_bins[retained] * (photons / native_bins[retained].sum())
    spectrum_provenance = Provenance(
        source="spekpy-native-spectrum.csv; https://pypi.org/project/spekpy/2.5.4/",
        sha256=digest(stage / "spekpy-native-spectrum.csv"),
        rights=(
            "Generated research spectrum; SpekPy 2.5.4 MIT licence and original notice retained."
        ),
        description=(
            "SpekPy casim W reflection target, 120 kVp, 12 degrees, 2.5 mm Al; "
            f"20-120 keV bins normalised to {photons:g} incident photons per ray. "
            "Tube mAs and reference distance define the source shape, "
            "not a scanner exposure calibration."
        ),
    )
    supplied = Spectrum(
        tuple(energies),
        tuple(incident),
        "bin-fluence",
        spectrum_provenance,
    )
    np.testing.assert_array_equal(supplied.bin_fluence, incident)
    np.testing.assert_allclose(incident.sum(), photons, rtol=1e-14)
    mu = np.array([table.at_energies(tuple(energies), edge_side="above") for table in tables])
    candidates = []
    # This modest, declared design comparison is not a claim of global optimality.
    for middle in ((45, 70), (50, 75), (50, 80), (55, 80), (60, 85)):
        thresholds = (20.0, *middle, 120.0)
        response = channel_response(energies, thresholds)
        checks = sensitivity(mu, incident, response)
        candidates.append(
            {
                "thresholds_kev": thresholds,
                "path_checks": checks,
                "worst_condition_number": max(c["condition_number"] for c in checks),
            }
        )
    chosen = min(candidates, key=lambda c: c["worst_condition_number"])
    response = channel_response(energies, chosen["thresholds_kev"])
    np.testing.assert_array_equal(response.sum(axis=0), np.ones(len(energies)))
    acquisition = {
        "generator": "SpekPy",
        "generator_version": SPEKPY_VERSION,
        "generator_constructor": configuration,
        "filtration": [{"material": "Al", "thickness_mm": 2.5}],
        "reference_source_distance_cm": 100.0,
        "native_density_unit": "photons cm^-2 keV^-1 at the stated reference mAs and distance",
        "native_energy_centres_kev": native_energy.tolist(),
        "retained_energy_interval_kev": [20.0, 120.0],
        "energy_bin_width_kev": 1.0,
        "excluded_below_20kev_native_fluence_fraction": float(
            native_bins[~retained].sum() / native_bins.sum()
        ),
        "incident_photons_per_ray_per_exposure_in_retained_interval": photons,
        "detector_acceptance": (
            "Uniform per-pixel incident budget prescribed at the detector; "
            "no inverse-square or pixel-area factor is applied again."
        ),
        "exposure": 1.0,
        "gain": 1.0,
        "offset": 0.0,
        "thresholds_kev": chosen["thresholds_kev"],
        "threshold_convention": (
            "lower inclusive, upper exclusive; thresholds coincide with native bin edges"
        ),
        "response": (
            "Ideal photon counting: unit detection probability in exactly one disjoint "
            "energy bin, zero elsewhere; no measured scanner calibration."
        ),
        "channel_open_beam_expected_counts": (response @ incident).tolist(),
        "channel_selection": (
            "Smallest worst unscaled Poisson-whitened two-path Jacobian condition number "
            "across five predefined path pairs, among five declared threshold pairs."
        ),
    }
    write_json(stage / "acquisition-definition.json", acquisition)
    response_provenance = Provenance(
        source="acquisition-definition.json",
        sha256=digest(stage / "acquisition-definition.json"),
        rights=(
            "Project-authored explicit ideal detector model; no third-party scanner calibration."
        ),
        description=acquisition["response"],
    )
    spec = SpectralSpec(
        materials=2,
        energies=len(energies),
        coefficients_provenance=tuple(table.provenance for table in tables),
        spectrum_provenance=spectrum_provenance,
        response_provenance=response_provenance,
        output_unit="counts",
        shared_weights=True,
        shared_response=True,
        active_paths=True,
        active_weights=False,
        active_response=False,
        active_coefficients=False,
        input_description=(
            "Fixed-density water (1 g/cm^3) and cortical bone (1.92 g/cm^3) "
            "volume fractions; nonnegative sum at most one, remainder vacuum. "
            "Material path units mm. Ideal disjoint photon-count channels and "
            "uniform prescribed detector-plane photon budget."
        ),
    )
    arrays = {
        "energies_kev": energies,
        "mu_mm_inv": mu,
        "incident_weights": incident,
        "weights": np.broadcast_to(incident, response.shape),
        "response": response,
    }
    arrays = {key: np.ascontiguousarray(value, dtype=np.float32) for key, value in arrays.items()}
    quantised_checks = sensitivity(
        arrays["mu_mm_inv"].astype(np.float64),
        arrays["incident_weights"].astype(np.float64),
        arrays["response"].astype(np.float64),
    )
    np.testing.assert_allclose(arrays["incident_weights"].sum(dtype=np.float64), photons, rtol=1e-7)
    np.savez(stage / "physics.npz", **arrays)
    metadata = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "kind": "simulated_spectral_acquisition_with_reference_material_coefficients",
        "materials": materials,
        "acquisition": acquisition,
        "coefficients_provenance": [dataclasses.asdict(table.provenance) for table in tables],
        "spectrum_provenance": dataclasses.asdict(spectrum_provenance),
        "response_provenance": dataclasses.asdict(response_provenance),
        "spectral_spec": dataclasses.asdict(spec),
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "physics_npz_sha256": digest(stage / "physics.npz"),
        "source_root": str(source_root),
        "sources": list(records.values()),
        "source_license_manifest_sha256": digest(source_root / "source-license-manifest.json"),
        "spekpy_package": json.loads((source_root / "package-record.json").read_text()),
        "interpolation": (
            "dpt.materials.MaterialTable log-log interpolation, edge_side='above'; "
            "all duplicate edge rows preserved, no extrapolation"
        ),
        "unit_conversion": "mu_mm_inv = total_mu_over_rho_cm2_g * reference_density_g_cm3 / 10",
        "energy_integration": (
            "SpekPy get_spectrum(flu=True,diff=False) already integrates each energy bin; "
            "no second width factor"
        ),
        "validation": {
            "reference_material_checks": reference_checks,
            "channel_design_comparison": candidates,
            "float32_export_path_checks": quantised_checks,
            "channel_partition": "exactly one channel per retained energy bin",
            "nonunit_energy_width_check": (
                "Separate SpekPy dk=2 keV spectrum: diff=False equals density times 2, "
                "confirming bin-fluence units independently of the production dk=1 grid"
            ),
            "float32_total_incident_photons": float(
                arrays["incident_weights"].sum(dtype=np.float64)
            ),
            "interpretation": (
                "Local two-path identifiability in this acquisition model; this does not "
                "establish global tomographic uniqueness or patient composition accuracy."
            ),
        },
        "limitations": [
            "Reference fixed-density materials assigned to anatomy are simulation truth, "
            "not measured patient material composition.",
            "Spectrum is generated, not acquired or scanner calibrated; "
            "patient VMI images are not spectral count channels.",
            "Sub-20 keV spectrum is excluded and retained fluence is explicitly "
            "renormalised to the prescribed photon budget.",
            "Ideal quantum efficiency, exact energy bins; no charge sharing, pulse pile-up, "
            "spectral tails, electronic noise, dead time or cross-talk.",
            "Primary attenuation only; scattered photons removed from the primary beam "
            "are not transported back to the detector.",
            "No bow-tie, heel-effect variation, focal-spot blur, spatial blur or "
            "detector-plane inverse-square variation.",
            "NIST interpolation and reference densities carry source/model uncertainty; "
            "no patient-specific calibration or uncertainty propagation.",
            "Public redistribution of NIST source/derived tables is not cleared "
            "by this private preparation.",
        ],
        "environment": {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "spekpy")},
        },
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
        "canonical_source_sha256": {
            str(path.relative_to(repo)): digest(path)
            for path in (
                repo / "python/dpt/materials.py",
                repo / "python/dpt/spectral.py",
                repo / "python/dpt/contracts.py",
            )
        },
    }
    (stage / "prepare_physics.py").write_bytes(Path(__file__).read_bytes())
    write_json(stage / "metadata.json", metadata)
    stage.rename(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "physics_npz_sha256": metadata["physics_npz_sha256"],
                "thresholds_kev": acquisition["thresholds_kev"],
                "channel_open_beam_expected_counts": acquisition[
                    "channel_open_beam_expected_counts"
                ],
                "worst_condition_number": chosen["worst_condition_number"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new private output directory")
    parser.add_argument("--photons", type=float, default=200_000)
    args = parser.parse_args()
    prepare(args.source_root, args.output, args.photons)


if __name__ == "__main__":
    main()
