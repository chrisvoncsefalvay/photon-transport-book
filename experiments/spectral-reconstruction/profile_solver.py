"""Profile the real anatomical workload, with explicit source and input identity."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import dpt
from dpt.experiments import RunRecorder, private_output, repository_root
from dpt.material_reconstruction import (
    MaterialReconstruction,
    MaterialReconstructionSettings,
    MaterialReconstructionView,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("euclidean", "spectral"), default="euclidean")
    args = parser.parse_args()
    entrypoint = Path(__file__).with_name("run.py")
    module_spec = importlib.util.spec_from_file_location("spectral_experiment", entrypoint)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("experiment driver is unavailable")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    acquisition = json.loads((args.acquisition / "run.json").read_text())
    if acquisition["status"] != "complete":
        raise ValueError("profiling requires a complete acquisition record")
    config = acquisition["configuration"]
    for name, expected in acquisition["output_sha256"].items():
        if module.digest(args.acquisition / name) != expected:
            raise ValueError("acquisition bytes changed")
    _, physics, spectral = module.load_physics(args.physics)
    for name, key in [
        ("physics.npz", "physics_npz_sha256"),
        ("metadata.json", "physics_metadata_sha256"),
    ]:
        if module.digest(args.physics / name) != config[key]:
            raise ValueError("profiling physics differs from the frozen acquisition")
    grid = module.grid_from(config["inverse_grid"])
    geometry = module.DetectorGeometry(**{k: tuple(v) for k, v in config["geometry"].items()})
    poses = [module.RigidTransform(**{k: tuple(v) for k, v in p.items()}) for p in config["poses"]]
    counts = np.load(args.acquisition / "observed-counts.npy", allow_pickle=False)
    initial = np.empty((2, *grid.shape), dtype=np.float32)
    initial[0].fill(0.1)
    initial[1].fill(0.02)
    views = tuple(
        MaterialReconstructionView(
            geometry, poses[i], counts[i], physics["weights"], physics["response"]
        )
        for i in config["fit_views"]
    )
    optional = {}
    if args.metric == "spectral":
        optional["material_metric"] = module.material_metric(physics)[0]
    solver = MaterialReconstruction(
        grid=grid,
        views=views,
        coefficients=physics["mu_mm_inv"],
        spectral_spec=spectral,
        initial_fractions=initial,
        settings=MaterialReconstructionSettings(iterations=1, regularisation_mm_inverse=0.1),
        samples_per_ray=256,
        **optional,
    )
    # Capture the package actually imported, including a frozen baseline package
    # when PYTHONPATH deliberately selects an earlier completed run's sources.
    package_root = Path(dpt.__file__).resolve().parent
    sources = {
        "python/dpt/" + p.relative_to(package_root).as_posix(): p
        for p in package_root.rglob("*.py")
    }
    sources.update(
        {
            "driver/profile.py": Path(__file__),
            "driver/run.py": entrypoint,
            "inputs/acquisition.json": args.acquisition / "run.json",
            "inputs/counts.npy": args.acquisition / "observed-counts.npy",
            "inputs/physics.npz": args.physics / "physics.npz",
            "inputs/physics-metadata.json": args.physics / "metadata.json",
        }
    )
    configuration = {
        "grid": config["inverse_grid"],
        "views": len(views),
        "channels": 3,
        "detector": config["geometry"],
        "energies": spectral.energies,
        "samples_per_ray": 256,
        "metric": args.metric,
        "package_root": str(package_root),
        "repeats": 3,
    }
    with RunRecorder(
        private_output(args.output, repository_root(__file__)),
        configuration=configuration,
        sources=sources,
    ) as run:
        solver.evaluate(gradient=True)
        solver.projected_trial(1e-7)
        solver.evaluate(gradient=False, trial=True)
        rows = []
        wp.cuda_profiler_start(solver.device)
        try:
            for index in range(3):
                start = time.perf_counter()
                value = solver.evaluate(gradient=True)
                elapsed = time.perf_counter() - start
                rows.append(
                    {
                        "operation": "full gradient evaluation",
                        "repeat": index,
                        "host_wall_seconds": elapsed,
                        "objective": value,
                    }
                )
        finally:
            wp.cuda_profiler_stop(solver.device)
        run.write_json("timings.json", rows)
        run.set_metadata(
            device=str(solver.device),
            interpretation=(
                "includes validation, launches, device work and host synchronisation; shared GPU"
            ),
        )
        print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
