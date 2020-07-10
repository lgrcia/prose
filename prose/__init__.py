__version__ = "0.0.1"

import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter("ignore", FITSFixedWarning)

from prose import config

CONFIG = config.ConfigManager()

from prose.io import FitsManager
from prose.lightcurves import LightCurve, LightCurves
from prose.telescope import Telescope
from prose.pipeline.base import Block, Unit, Image
from prose.pipeline.units import Reduction
from prose.photproducts import PhotProducts
