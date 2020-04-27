__version__ = "0.0.1"

from prose import config

CONFIG = config.ConfigManager()

from .products.photometry import Photometry
