__version__ = "0.0.1"

import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter("ignore", FITSFixedWarning)

from prose import config

CONFIG = config.ConfigManager()


import prose.visualisation as viz

from prose.io import FitsManager
from prose.fluxes import Fluxes, LightCurves
from prose.telescope import Telescope
from prose.blocks.base import Block, Unit, Image
from prose.blocks.units import Reduction, AperturePhotometry
from prose.observation import Observation


def load(photfile):
    return Observation(photfile)
