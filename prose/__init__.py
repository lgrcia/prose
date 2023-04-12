import warnings

from astropy.wcs import FITSFixedWarning

warnings.simplefilter("ignore", FITSFixedWarning)

from prose import config

CONFIG = config.ConfigManager()
CONFIG.check_builtins_changes()

from pkg_resources import get_distribution

from prose import visualization as viz
from prose.core import Block, FITSImage, Image, Sequence, source
from prose.fluxes import Fluxes
from prose.io.fitsmanager import FitsManager
from prose.simulations import example_image
from prose.telescope import Telescope

__version__ = get_distribution("prose").version

# TODO: update Telescope "names" fields
# TODO: document custom Image using _get_data_header
