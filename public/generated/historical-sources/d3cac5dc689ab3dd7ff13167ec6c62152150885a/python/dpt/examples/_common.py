"""Input provenance and private recording for the application examples.

Array preparation is a host-side boundary. Scientific operators receive explicit
device arrays from the drivers; this module neither imports CUDA nor launches it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dpt.experiments import RunRecorder, experiment_sources, private_output, repository_root


class ExampleInputError(ValueError):
    """A supplied application case does not satisfy its documented contract."""


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExampleInputError(f"{label} must be a JSON object with named fields")
    if not all(isinstance(key, str) for key in cast(dict[object, object], value)):
        raise ExampleInputError(f"{label} must be a JSON object with named fields")
    return cast(dict[str, Any], value)


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExampleInputError(f"JSON field {key!r} is repeated; keep one explicit value")
        result[key] = value
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExampleInputError(f"{label} must be a non-empty string")
    return value.strip()


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_file():
        raise ExampleInputError(f"{label} must identify an existing regular, non-symlink file")
    return expanded.resolve()


@dataclass(frozen=True)
class _ArrayInput:
    path: Path
    sha256: str
    units: str
    source: str
    rights: str


@dataclass
class SuppliedCase:
    """One versioned case file and its explicitly declared numeric inputs."""

    path: Path
    config: dict[str, Any]
    _case_sha256: str
    _arrays: dict[str, _ArrayInput]

    # region book:example-supplied-array
    def array(
        self,
        name: str,
        *,
        dtype: str = "float32",
        shape: tuple[int, ...] | None = None,
        units: str,
    ) -> Any:
        """Read the checked bytes without casting, reshaping or inventing values."""
        if name not in self._arrays:
            raise ExampleInputError(f"arrays.{name} is missing; declare its file and provenance")
        record = self._arrays[name]
        if record.units != units:
            raise ExampleInputError(
                f"arrays.{name}.units must be {units!r}; convert and document the input first"
            )
        payload = record.path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise ExampleInputError(f"arrays.{name} changed or does not match its sha256")
        np: Any = importlib.import_module("numpy")
        try:
            data: Any = np.load(io.BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ExampleInputError(f"arrays.{name} must contain one numeric .npy array") from error
        if not isinstance(data, np.ndarray):
            raise ExampleInputError(f"arrays.{name} must contain one .npy array, not an archive")
        if data.dtype != np.dtype(dtype) or not data.dtype.isnative:
            raise ExampleInputError(
                f"arrays.{name} needs native-endian {dtype}; its stored dtype is {data.dtype}"
            )
        if not data.flags.c_contiguous:
            raise ExampleInputError(f"arrays.{name} must be saved in contiguous C order")
        if shape is not None and tuple(data.shape) != shape:
            raise ExampleInputError(
                f"arrays.{name} needs shape {shape}; its stored shape is {tuple(data.shape)}"
            )
        if data.dtype.kind not in "fiu" or not bool(np.isfinite(data).all()):
            raise ExampleInputError(f"arrays.{name} must contain only finite real numeric values")
        return data

    # endregion book:example-supplied-array

    def record(
        self,
        output: Path,
        *,
        entrypoint: str | Path,
        metadata: dict[str, object] | None = None,
    ) -> RunRecorder:
        """Prepare a new record after confirming that loaded inputs are unchanged.

        All declared arrays, including optional evaluation references, are
        snapshotted for provenance. Drivers decide when to read each role and
        must keep withheld references out of their optimisation decisions.
        """
        if _digest(self.path) != self._case_sha256:
            raise ExampleInputError("The case JSON changed after loading; start with a stable file")
        extra: dict[str, Path] = {}
        for name, record in self._arrays.items():
            if record.path.is_symlink() or _digest(record.path) != record.sha256:
                raise ExampleInputError(f"arrays.{name} changed before recording could start")
            extra[f"inputs/{name}.npy"] = record.path
        root = repository_root(entrypoint)
        destination = private_output(output, root)
        provenance = {
            name: {
                "sha256": record.sha256,
                "units": record.units,
                "source": record.source,
                "rights": record.rights,
            }
            for name, record in self._arrays.items()
        }
        return RunRecorder(
            destination,
            configuration=self.config,
            sources=experiment_sources(entrypoint, self.path, extra=extra),
            metadata={
                **(metadata or {}),
                "input_provenance": provenance,
                "scientific_acceptance": "requires independent numerical validation",
            },
        )


def load_case(path: Path) -> SuppliedCase:
    """Validate the manifest and file identities without loading CUDA."""
    path = _regular_file(path, "--case")
    payload = path.read_bytes()
    try:
        config = _mapping(
            json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_fields), "case"
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExampleInputError("--case must contain valid UTF-8 JSON") from error
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ExampleInputError("case.schema_version must be the integer 1")
    arrays = _mapping(config.get("arrays"), "case.arrays")
    if not arrays:
        raise ExampleInputError("case.arrays must declare the supplied .npy inputs")
    inputs: dict[str, _ArrayInput] = {}
    for name, value in arrays.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
            raise ExampleInputError(
                "Array names must use letters, digits, dots, underscores or hyphens"
            )
        fields = _mapping(value, f"arrays.{name}")
        raw_path = Path(_text(fields.get("path"), f"arrays.{name}.path")).expanduser()
        array_path = _regular_file(
            raw_path if raw_path.is_absolute() else path.parent / raw_path,
            f"arrays.{name}.path",
        )
        if array_path.suffix != ".npy":
            raise ExampleInputError(f"arrays.{name}.path must end in .npy")
        expected = _text(fields.get("sha256"), f"arrays.{name}.sha256").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ExampleInputError(f"arrays.{name}.sha256 must contain 64 hexadecimal characters")
        if _digest(array_path) != expected:
            raise ExampleInputError(f"arrays.{name} does not match its declared sha256")
        inputs[name] = _ArrayInput(
            array_path,
            expected,
            _text(fields.get("units"), f"arrays.{name}.units"),
            _text(fields.get("source"), f"arrays.{name}.source"),
            _text(fields.get("rights"), f"arrays.{name}.rights"),
        )
    return SuppliedCase(path, config, hashlib.sha256(payload).hexdigest(), inputs)


def example_parser(description: str) -> argparse.ArgumentParser:
    """Expose the same preparation and output arguments in every application."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--case", type=Path, required=True, help="Supplied case JSON with input hashes"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New private output directory outside the repository",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device used by the application")
    return parser


def write_array(run: RunRecorder, name: str, array: Any) -> None:
    """Record one final host array; drivers explicitly choose when to download."""
    if Path(name).suffix != ".npy":
        raise ExampleInputError("Array output names must end in .npy")
    np: Any = importlib.import_module("numpy")
    if not isinstance(array, np.ndarray) or array.dtype.kind not in "fiu":
        raise ExampleInputError("Array output must be a real numeric NumPy array")
    if not bool(np.isfinite(array).all()):
        raise ExampleInputError("Array output contains nonfinite values; inspect the failed solve")
    payload = io.BytesIO()
    np.save(payload, array, allow_pickle=False)
    run.write_bytes(name, payload.getvalue())
