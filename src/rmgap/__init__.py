import importlib.metadata

try:
    __version__ = importlib.metadata.version("rmgap")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
