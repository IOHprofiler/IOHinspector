from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("iohinspector")
except PackageNotFoundError:
    __version__ = "unknown"

from . import figures, indicators, metrics, plots  # noqa: F401  (iohinspector.figures)
from ._upgrade import removed_names
from .align import align_data
from .indicators import *
from .manager import DataManager
from .metrics import *
from .plots import *

__all__ = [  # noqa: PLE0604  (the subpackages' names)
    "DataManager",
    "align_data",
    *indicators.__all__,
    *metrics.__all__,
    *plots.__all__,
]

__getattr__ = removed_names(__name__)
