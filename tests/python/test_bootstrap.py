from pathlib import Path

import dpt


def test_package_identity() -> None:
    assert dpt.package_identity() == "differentiable-photon-transport"


def test_bootstrap_region_is_unique_and_exact() -> None:
    source_path = Path(__file__).parents[2] / "python" / "dpt" / "bootstrap.py"
    source = source_path.read_text(encoding="utf-8")

    assert source.count("# region book:bootstrap-package-identity") == 1
    assert source.count("# endregion book:bootstrap-package-identity") == 1
