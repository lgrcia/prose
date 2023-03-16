import warnings
from astropy.wcs import FITSFixedWarning

warnings.simplefilter("ignore", FITSFixedWarning)

from . import config

CONFIG = config.ConfigManager()
CONFIG.check_builtins_changes()

from . import visualization as viz

from .io.fitsmanager import FitsManager
from .telescope import Telescope
from .core import Block, Sequence, Image, FITSImage, source
from .fluxes import Fluxes
from .simulations import example_image

from pkg_resources import get_distribution

__version__ = get_distribution("prose").version

# TODO: update Telescope "names" fields
# TODO: document custom Image using _get_data_header
