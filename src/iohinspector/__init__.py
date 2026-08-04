from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("iohinspector")
except PackageNotFoundError:
    __version__ = "unknown"

from .align import *
from .data import *
from .manager import *
from .indicators import *
from .metrics import *
from .plots import *
