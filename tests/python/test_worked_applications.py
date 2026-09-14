"""CPU checks for the worked study's comparison and observation contracts."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from dpt.experiments import RunRecorder

np: Any = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "experiments/worked-applications/run.py"


@pytest.fixture
def driver() -> Any:
    # The private drivers use sibling module names; leave no imports or path
    # entries that could replace another experiment's helpers in this process.
    names = ("_study", "_final", "batch", "run")
    previous = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    path = sys.path.copy()
    try:
        spec = importlib.util.spec_from_file_location("worked_application_contract", DRIVER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = path
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


def protocol() -> dict[str, Any]:
    return json.loads(DRIVER.with_name("config.json").read_text())


def test_frozen_comparison_contract_is_accepted(driver: Any) -> None:
    driver.validate_protocol(protocol())


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (None, "task_coordinates_mm", [-30.0, 30.0], "task corners"),
        ("acquisition", "base_view_degrees", 5.0, "zero-degree"),
        ("acquisition", "candidate_angles_degrees", [5.0, 5.0], "distinct integral"),
        ("acquisition", "candidate_angles_degrees", [5.0, 5.1], "distinct integral"),
        ("acquisition", "candidate_angles_degrees", [5.0, float("nan")], "distinct integral"),
        ("acquisition", "fixed_view_degrees", 75.0, "including the baseline"),
        ("randomness", "replicates", 7, "eight independent replicates"),
        ("acceptance", "gradient_tolerance", 0.01, "thresholds must agree"),
        (None, "held_out_views_degrees", [45.0, 90.0], "disjoint"),
    ],
)
def test_changed_protocol_cannot_silently_change_comparison(
    driver: Any, section: str | None, key: str, value: Any, message: str
) -> None:
    config = protocol()
    destination = config if section is None else config[section]
    destination[key] = value
    with pytest.raises(ValueError, match=message):
        driver.validate_protocol(config)


def test_same_selected_and_fixed_angle_reuses_the_observation(driver: Any, tmp_path: Path) -> None:
    config = protocol()
    public: dict[str, Any] = {"views": {}}
    angle = config["acquisition"]["fixed_view_degrees"]
    means = {angle: np.full(config["geometry"]["detector_shape_hw"], 0.5)}
    with RunRecorder(
        tmp_path / "observations", configuration=config, sources={"driver.py": DRIVER}
    ) as run:
        first = driver._observation(run, public, means, config, angle, 1000, 3, 0, 2026091401)
        source = run.output / public["views"][first]["observation"]["file"]
        original = source.read_bytes()
        repeated = driver._observation(run, public, means, config, angle, 1000, 3, 0, 2026091401)
        assert repeated == first
        assert source.read_bytes() == original
        assert len(public["views"]) == 1
        independent = driver._observation(run, public, means, config, angle, 1000, 3, 1, 2026091401)
        assert independent != first
        other = run.output / public["views"][independent]["observation"]["file"]
        assert not np.array_equal(np.load(source), np.load(other))
