from tqdm import tqdm
from astropy.io import fits
from prose.console_utils import TQDM_BAR_FORMAT
from astropy.wcs import WCS
from prose import visualisation as viz
from collections import OrderedDict
from tabulate import tabulate
import numpy as np
from time import time


class Sequence:
    # TODO: add index self.i in image within unit loop

    def __init__(self, blocks, files, name="default", telescope=None, **kwargs):
        self.name = name
        self.files_or_images = files if not isinstance(files, str) else [files]
        self.blocks = blocks

        self.data = {}
        self._telescope = None
        self.telescope = telescope

    def __getattr__(self, item):
        return self.blocks_dict[item]

    @property
    def telescope(self):
        return self._telescope

    @telescope.setter
    def telescope(self, telescope):
        self._telescope = telescope
        for block in self.blocks:
            block.set_telescope(telescope)

    @property
    def blocks(self):
        return list(self.blocks_dict.values())

    @blocks.setter
    def blocks(self, blocks):
        self.blocks_dict = OrderedDict({
            block.name if block.name is not None else "block{}".format(i): block
            for i, block in enumerate(blocks)
        })

    def run(self, show_progress=True):
        if show_progress:
            progress = lambda x: tqdm(
                x,
                desc=self.name,
                unit="images",
                ncols=80,
                bar_format=TQDM_BAR_FORMAT,
            )

        else:
            progress = lambda x: x

        if isinstance(self.files_or_images, list):
            if len(self.files_or_images) == 0:
                raise ValueError("No images to process")
        elif self.files_or_images is None:
            raise ValueError("No images to process")

        # initialization
        for block in self.blocks:
            block.set_telescope(self.telescope)
            block.set_unit_data(self.data)
            block.initialize()

        # run
        for i, file_or_image in enumerate(progress(self.files_or_images)):
            if isinstance(file_or_image, str):
                image = Image(file_or_image)
            else:
                image = file_or_image
            image.i = i
            discard_message = False
            for block in self.blocks:
                # This allows to discard image in any blocks
                if not image.discarded:
                    block._run(image)
                elif not discard_message:
                    discard_message = True
                    if isinstance(file_or_image, str):
                        print(f"Warning: image {i} (...{file_or_image[i]}) discarded in {type(block).__name__}")
                    else:
                        print(f"Warning: image {i} discarded in {type(block).__name__}")

            del image

        # terminate
        for block in self.blocks:
            block.terminate()

    def __str__(self):
        rows = [[block.name, block.__class__.__name__, f"{block.processing_time:.4f} s"] for block in self.blocks]
        headers = ["name", "type", "processing"]

        return tabulate(rows, headers, tablefmt="fancy_grid")

    def citations(self):
        citations = [block.citations() for block in self.blocks if block.citations() is not None]
        return citations if len(citations) > 0 else None

    def insert_before(self, before, block):
        pass

    @property
    def processing_time(self):
        return np.sum([block.processing_time for block in self.blocks])


class Image:

    def __init__(self, fitspath=None, data=None, header=None, **kwargs):
        if fitspath is not None:
            self.data = fits.getdata(fitspath).astype(float)
            self.header = fits.getheader(fitspath)
            self.path = fitspath
        else:
            self.data = data
            self.header = header if header is not None else {}
            self.path = None

        self.discarded = False

        self.__dict__.update(kwargs)

    def copy(self, data=True):
        new_self = Image()
        new_self.__dict__.update(self.__dict__)
        if not data:
            del new_self.__dict__["data"]

        return new_self

    @property
    def wcs(self):
        return WCS(self.header)


class Block:

    def __init__(self, name=None):
        self.name = name
        self.unit_data = None
        self.telescope = None
        self.processing_time = 0
        self.runs = 0

    def initialize(self, *args):
        pass

    def set_unit_data(self, unit_data):
        self.unit_data = unit_data

    def set_telescope(self, telescope):
        self.telescope = telescope

    def _run(self, *args, **kwargs):
        t0 = time()
        self.run(*args, **kwargs)
        self.processing_time += time() - t0
        self.runs += 1

    def run(self, image, **kwargs):
        raise NotImplementedError()

    def terminate(self):
        pass

    def stack_method(self, image):
        pass

    def show_image(self, image):
        viz.show_stars(image)

    @staticmethod
    def citations():
        return None

    @staticmethod
    def doc():
        return ""


class Pipeline:

    def __init__(self, units, name="default", **kwargs):
        self.name = name
        self.units = units
        self.data = {}
        self._telescope = None

    @property
    def telescope(self):
        return self._telescope

    @telescope.setter
    def telescope(self, telescope):
        self._telescope = telescope
        for unit in self.units:
            unit.set_telescope(telescope)

    @property
    def blocks(self):
        return self.units_dict.values()

    @blocks.setter
    def blocks(self, units):
        self.units_dict = OrderedDict({
            unit.name if unit.name is not None else "block{}".format(i): unit
            for i, unit in enumerate(units)
        })

    def run(self):
        # run
        for unit in self.units:
            unit.run()

