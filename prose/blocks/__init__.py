from .registration import XYShift, Twirl
from .alignment import AffineTransform, Cutout2D
from .detection import SegmentedPeaks, DAOFindStars, SEDetection, Peaks
from .psf import MedianPSF, FastGaussian, KeepGoodStars, FWHM, Cutouts
from .photometry import PhotutilsAperturePhotometry, SEAperturePhotometry, PhotutilsPSFPhotometry
from .shepard import  Shepard
from . import catalogs
from .vizualisation import RawVideo
from .utils import *
from .centroids import *

# import prose
# from prose import Block
# import sys, inspect
# import importlib
#
#
# def blocks_in_module(module):
#     classes = []
#     module = importlib.import_module(module.__name__)
#     for name, obj in inspect.getmembers(sys.modules[module.__name__]):
#         if inspect.isclass(obj) and issubclass(obj, Block) and obj != Block:
#             classes.append(obj)
#     return classes
#
#
# def get_submodules(module):
#     return [_module for _, _module in inspect.getmembers(module, inspect.ismodule)]
#
#
# def flatten(S):
#     if S == []:
#         return S
#     if isinstance(S[0], list):
#         return flatten(S[0]) + flatten(S[1:])
#     return S[:1] + flatten(S[1:])
#
#
# def submodules_blocks(module):
#     submodules = get_submodules(module)
#     return flatten([blocks_in_module(mo) for mo in submodules if len(blocks_in_module(mo))>0])
#
#
# __all__ = flatten([
#     submodules_blocks(prose._blocks),
#     # submodules_blocks(prose._diagnostics),
#     # submodules_blocks(prose.neural),
# ])