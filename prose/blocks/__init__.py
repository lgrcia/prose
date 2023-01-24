from ..core.block import Block
from .detection import *
from .geometry import *
from .centroids import *
from .psf import *
from .alignment import *
from .catalogs import *
from .photometry import *
from .utils import *

class DataBlock(Block):

    def __init__(self, name=None):
        super().__init__(name)