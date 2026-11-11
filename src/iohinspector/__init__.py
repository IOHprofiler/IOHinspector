from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("iohinspector")
except PackageNotFoundError:
    __version__ = "unknown"

from . import indicators, metrics, plots
from ._upgrade import removed_names
from .align import align_data, turbo_align
from .indicators import *
from .manager import DataManager
from .metrics import *
from .plots import *

__all__ = [
    "DataManager",
    "align_data",
    "turbo_align",
    *indicators.__all__,
    *metrics.__all__,
    *plots.__all__,
]

__getattr__ = removed_names(__name__)
