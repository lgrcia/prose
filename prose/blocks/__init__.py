from prose.blocks.registration import XYShift, AstroAlignShift
from prose.blocks.alignment import Align
from prose.blocks.detection import SegmentedPeaks, DAOFindStars, SEDetection
from prose.blocks.calibration import Calibration, Trim
from prose.blocks.psf import Gaussian2D, Moffat2D
from prose.blocks.base import Unit, Block
from prose.blocks.photometry import PhotutilsAperturePhotometry, SEAperturePhotometry, PSFPhotometry
from prose.blocks.imutils import *
from prose.blocks.io import SavePhot

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