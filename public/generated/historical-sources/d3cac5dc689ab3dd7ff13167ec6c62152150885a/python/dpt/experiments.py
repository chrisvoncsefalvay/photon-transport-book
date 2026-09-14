"""Private run records with source snapshots and explicit completion state.

This module records executions when a driver is run. It neither launches an
experiment nor promotes an artefact into the book. Snapshots describe the files
at entry and exit; they are not an attestation of a compiler's loaded modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any

from dpt.contracts import ContractError


def repository_root(entrypoint: str | Path) -> Path:
    """Find the authoring/export root from an actual driver or package file."""
    for parent in Path(entrypoint).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "python/dpt").is_dir():
            return parent
    raise ContractError("experiment entrypoint is outside a dpt source tree")


def experiment_parser(
    entrypoint: str | Path,
    description: str | None,
    *,
    configuration: bool = True,
    cuda: bool = True,
) -> argparse.ArgumentParser:
    """Common CLI; --output-dir remains an alias for older transmission commands."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", "--output-dir", type=Path, required=True)
    if configuration:
        parser.add_argument(
            "--config", type=Path, default=Path(entrypoint).with_name("config.json")
        )
    if cuda:
        parser.add_argument("--device", default="cuda:0")
    return parser


def private_output(path: Path, root: Path) -> Path:
    """Require a new private directory outside the entire source tree."""
    output = path.expanduser().resolve()
    if output.is_relative_to(root.resolve()):
        raise ContractError("raw experiment output must be outside the authoring repository")
    if output.exists():
        raise ContractError("output directory must be new; existing records are preserved")
    return output


def experiment_sources(
    entrypoint: str | Path,
    configuration: Path | None = None,
    *,
    extra: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    """Snapshot the package, driver, lock and exact configuration consistently.

    Capturing the complete package avoids silently omitting newly factored kernel
    or runtime dependencies from an experiment's provenance.
    """
    root = repository_root(entrypoint)
    paths = sorted((root / "python/dpt").rglob("*.py"))
    paths.extend((Path(entrypoint).resolve(), root / "pyproject.toml", root / "uv.lock"))
    sources = {path.relative_to(root).as_posix(): path for path in paths}
    if configuration is not None:
        sources["configuration.json"] = configuration
    sources.update(extra or {})
    return sources


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _relative(name: str) -> Path:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in name.split("/"))
        or "\\" in name
        or "\x00" in name
    ):
        raise ContractError("record paths must be nonempty relative POSIX paths without traversal")
    return Path(*path.parts)


def _bytes(payload: object, *, sort_keys: bool = True) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=sort_keys, allow_nan=False) + "\n").encode()


def _atomic(path: Path, payload: bytes) -> None:
    """Replace one record only after its complete bytes have reached the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


# region book:experiment-run-provenance
class RunRecorder:
    """Own one new output directory and a running/complete/failed JSON manifest.

    Drivers supply all scientific configuration, input hashes and actual device
    metadata. Installed distribution versions are read without importing CUDA.
    A run cannot become complete if a declared source changed during execution.
    An interrupted process leaves ``running`` behind, never a success record.
    """

    def __init__(
        self,
        output: Path,
        *,
        configuration: dict[str, object],
        sources: Mapping[str, Path],
        metadata: dict[str, object] | None = None,
    ) -> None:
        if not sources:
            raise ContractError("a reproducible run needs at least one declared source")
        self.output = Path(output)
        self.sources = {
            str(_relative(name).as_posix()): Path(path) for name, path in sources.items()
        }
        # Round-trip once so later mutation of caller dictionaries cannot alter a record.
        self._configuration: Any = json.loads(_bytes(configuration))
        self._metadata: dict[str, Any] = json.loads(_bytes(metadata or {}))
        self._record: dict[str, Any] = {}
        self._hashes: dict[str, str] = {}
        self._outputs: dict[str, str] = {}
        self._entered = False
        self._closed = False

    def __enter__(self) -> RunRecorder:
        if self._entered:
            raise ContractError("a recorder is single-use")
        # Snapshot every declared source before creating an output directory.
        snapshots: dict[str, bytes] = {}
        for name, path in self.sources.items():
            if path.is_symlink() or not path.is_file():
                raise ContractError(f"source must be a regular non-symlink file: {name}")
            snapshots[name] = path.read_bytes()
        self.output.mkdir(parents=True, exist_ok=False)
        self._entered = True
        self._hashes = {
            name: hashlib.sha256(payload).hexdigest() for name, payload in snapshots.items()
        }
        self._record = {
            "schema_version": 2,
            "status": "running",
            "started_utc": _timestamp(),
            "configuration": self._configuration,
            "configuration_sha256": hashlib.sha256(_bytes(self._configuration)).hexdigest(),
            "metadata": self._metadata,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "distributions": {
                    name: _version(name)
                    for name in ("differentiable-photon-transport", "warp-lang", "numpy")
                },
            },
            "source_sha256": self._hashes,
            "output_sha256": self._outputs,
        }
        try:
            self._flush()
            # Preserve the exact serialised effective configuration, including
            # CLI overrides, so readers can verify its hash without reproducing
            # Python's floating-point/Unicode JSON formatting in another language.
            self.write_json("configuration.json", self._configuration)
            for name, payload in snapshots.items():
                _atomic(self.output / "sources" / _relative(name), payload)
        except BaseException as error:
            self._record.update(status="failed", finished_utc=_timestamp(), error=repr(error))
            self._closed = True
            self._flush()
            raise
        return self

    def _flush(self) -> None:
        _atomic(self.output / "run.json", _bytes(self._record))

    def _require_open(self) -> None:
        if not self._entered or self._closed:
            raise ContractError("run output is writable only inside an active recorder context")

    def set_metadata(self, **fields: object) -> None:
        self._require_open()
        self._metadata.update(json.loads(_bytes(fields)))
        self._flush()

    def write_json(self, name: str, payload: object) -> None:
        if _relative(name).suffix != ".json":
            raise ContractError("JSON output needs a .json path")
        # Preserve the declared column/field order in scientific payloads.
        # Effective configuration has already been round-tripped in sorted order.
        self.write_bytes(name, _bytes(payload, sort_keys=False))

    def write_bytes(self, name: str, data: bytes) -> None:
        """Record a rendered artefact with the same ownership and hash checks as JSON."""
        self._require_open()
        relative = _relative(name)
        if (
            relative.parts[0] == "sources"
            or relative.as_posix() == "run.json"
            or name in self._outputs
        ):
            raise ContractError("output needs a new path outside the reserved manifest/sources")
        path = self.output / relative
        if path.exists() or any(parent.is_symlink() for parent in path.parents):
            raise ContractError(
                "refusing to overwrite output or follow an output-directory symlink"
            )
        _atomic(path, data)
        self._outputs[name] = hashlib.sha256(data).hexdigest()
        self._flush()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._require_open()
        changed: list[str] = []
        for name, path in self.sources.items():
            try:
                matches = (
                    not path.is_symlink()
                    and hashlib.sha256(path.read_bytes()).hexdigest() == self._hashes[name]
                )
            except OSError:
                matches = False
            if not matches:
                changed.append(name)
        changed_outputs: list[str] = []
        recorded_files = {
            **self._outputs,
            **{f"sources/{name}": digest for name, digest in self._hashes.items()},
        }
        for name, expected in recorded_files.items():
            path = self.output / _relative(name)
            try:
                matches = (
                    not path.is_symlink()
                    and hashlib.sha256(path.read_bytes()).hexdigest() == expected
                )
            except OSError:
                matches = False
            if not matches:
                changed_outputs.append(name)
        failure = exception
        if changed and failure is None:
            failure = RuntimeError(
                f"declared sources changed during execution: {', '.join(changed)}"
            )
        if changed_outputs and failure is None:
            failure = RuntimeError(
                f"recorded files changed during execution: {', '.join(changed_outputs)}"
            )
        self._record.update(
            status="complete" if failure is None else "failed",
            finished_utc=_timestamp(),
            sources_unchanged=not changed,
            changed_sources=changed,
            recorded_files_unchanged=not changed_outputs,
            changed_recorded_files=changed_outputs,
        )
        if failure is not None:
            self._record["error"] = repr(failure)
        self._closed = True
        try:
            self._flush()
        except OSError as error:
            if exception is None:
                raise
            exception.add_note(f"could not finalise the failed run record: {error}")
        if exception is None and failure is not None:
            raise failure
        return False


# endregion book:experiment-run-provenance
