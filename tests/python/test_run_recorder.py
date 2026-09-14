"""Filesystem provenance failure cases; no scientific execution is involved."""

import hashlib
import json
from pathlib import Path

import pytest

from dpt.contracts import ContractError
from dpt.experiments import RunRecorder


def test_snapshots_hashes_and_complete_state(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text("# declared source\n")
    output = tmp_path / "run"
    with RunRecorder(
        output, configuration={"seed": 17}, sources={"python/model.py": source}
    ) as run:
        assert json.loads((output / "run.json").read_text())["status"] == "running"
        run.set_metadata(fixture="analytic contract")
        run.write_json("values.json", {"observed": [1.0, 0.0]})
    record = json.loads((output / "run.json").read_text())
    assert record["schema_version"] == 2
    assert "finished_utc" in record and "completed_utc" not in record
    assert record["status"] == "complete"
    assert record["sources_unchanged"] is True
    assert (output / "sources/python/model.py").read_bytes() == source.read_bytes()
    assert (
        record["output_sha256"]["values.json"]
        == hashlib.sha256((output / "values.json").read_bytes()).hexdigest()
    )


def test_changed_source_never_claims_completion(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text("# before\n")
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="sources changed"):
        with RunRecorder(output, configuration={}, sources={"model.py": source}):
            source.write_text("# after\n")
    record = json.loads((output / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["changed_sources"] == ["model.py"]
    assert (output / "sources/model.py").read_text() == "# before\n"


def test_execution_error_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.touch()
    output = tmp_path / "run"
    with pytest.raises(ArithmeticError, match="independent failure"):
        with RunRecorder(output, configuration={}, sources={"model.py": source}):
            raise ArithmeticError("independent failure")
    assert json.loads((output / "run.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("name", ["values.json", "sources/model.py"])
def test_output_and_snapshot_tampering_invalidates_completion(tmp_path: Path, name: str) -> None:
    source = tmp_path / "model.py"
    source.write_text("# original\n")
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="recorded files changed"):
        with RunRecorder(output, configuration={}, sources={"model.py": source}) as run:
            run.write_json("values.json", {"value": 1})
            (output / name).write_text("altered")
    record = json.loads((output / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["changed_recorded_files"] == [name]


def test_no_overwrites_reserved_paths_or_nonfinite_results(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.touch()
    output = tmp_path / "run"
    with RunRecorder(output, configuration={}, sources={"model.py": source}) as run:
        for path in ("../escape.json", "/escape.json", "run.json", "sources/code.json"):
            with pytest.raises(ContractError):
                run.write_json(path, {})
        with pytest.raises(ValueError):
            run.write_json("bad.json", {"value": float("nan")})
        assert not (output / "bad.json").exists()
        run.write_json("first.json", {})
        with pytest.raises(ContractError):
            run.write_json("first.json", {})
    with pytest.raises(FileExistsError):
        with RunRecorder(output, configuration={}, sources={"model.py": source}):
            pass


def test_private_output_and_cli_aliases(tmp_path: Path) -> None:
    from dpt.experiments import experiment_parser, private_output

    root = tmp_path / "repo"
    root.mkdir()
    for path in (root, root / "notes/run", root / "public/run"):
        with pytest.raises(ContractError, match="outside"):
            private_output(path, root)
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ContractError, match="outside"):
        private_output(link / "run", root)
    output = tmp_path / "run"
    assert private_output(output, root) == output
    parser = experiment_parser(root / "run.py", "test")
    assert parser.parse_args(["--output-dir", str(output)]).output == output
    assert parser.parse_args(["--output", str(output)]).output == output
    output.mkdir()
    with pytest.raises(ContractError, match="new"):
        private_output(output, root)


def test_binary_outputs_share_recorder_integrity_checks(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("# fixture")
    with RunRecorder(tmp_path / "run", configuration={}, sources={"source.py": source}) as run:
        run.write_bytes("figure.svg", b"<svg/>")
        with pytest.raises(ContractError):
            run.write_bytes("figure.svg", b"changed")
    record = json.loads((tmp_path / "run/run.json").read_text())
    assert record["output_sha256"]["figure.svg"] == hashlib.sha256(b"<svg/>").hexdigest()
