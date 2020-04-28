__version__ = "0.0.1"

import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter("ignore", FITSFixedWarning)

from prose import config

CONFIG = config.ConfigManager()

from .products.photometry import Photometry
from prose.lightcurves import LightCurve, LightCurves
from prose.io import FitsManager
from prose.telescope import Telescope
