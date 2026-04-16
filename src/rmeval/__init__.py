import importlib.metadata

from .task_runner import TaskRunner
from .rm import make_rm

try:
    __version__ = importlib.metadata.version("rmeval")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = [
    "TaskRunner",
    "make_rm",
]
