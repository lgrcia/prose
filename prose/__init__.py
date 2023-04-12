import warnings

from astropy.wcs import FITSFixedWarning

warnings.simplefilter("ignore", FITSFixedWarning)

from . import config

CONFIG = config.ConfigManager()
CONFIG.check_builtins_changes()

from pkg_resources import get_distribution

from . import visualization as viz
from .core import Block, FITSImage, Image, Sequence, source
from .fluxes import Fluxes
from .io.fitsmanager import FitsManager
from .simulations import example_image
from .telescope import Telescope

__version__ = get_distribution("prose").version

# TODO: update Telescope "names" fields
# TODO: document custom Image using _get_data_header
