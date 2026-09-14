"""Differentiable Photon Transport scientific package."""

from importlib.metadata import PackageNotFoundError, version

from dpt.bootstrap import package_identity

try:
    __version__ = version("differentiable-photon-transport")
except PackageNotFoundError:  # Direct source-tree imports are useful to editors and type checkers.
    __version__ = "0+unknown"

__all__ = ["__version__", "package_identity"]
