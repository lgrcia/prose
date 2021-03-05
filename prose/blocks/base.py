from tqdm import tqdm
from astropy.io import fits
from ..console_utils import TQDM_BAR_FORMAT
from astropy.wcs import WCS
from .. import visualisation as viz
from collections import OrderedDict
from tabulate import tabulate
from time import time


class Sequence:
    # TODO: add index self.i in image within unit loop

    def __init__(self, blocks, files, name="default", telescope=None, **kwargs):
        self.name = name
        self.files = files if not isinstance(files, str) else [files]
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
        return self.blocks_dict.values()

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
                unit="files",
                ncols=80,
                bar_format=TQDM_BAR_FORMAT,
            )

        else:
            progress = lambda x: x

        if isinstance(self.files, list):
            if len(self.files) == 0:
                raise ValueError("No files to process")
        elif self.files is None:
            raise ValueError("No files to process")

        # initialization
        for block in self.blocks:
            block.set_telescope(self.telescope)
            block.set_unit_data(self.data)
            block.initialize()

        # run
        for i, file_path in enumerate(progress(self.files)):
            image = Image(file_path)
            discard_message = False
            for block in self.blocks:
                # This allows to discard image in any blocks
                if not image.discarded:
                    block._run(image)
                elif not discard_message:
                    discard_message = True
                    print(f"Warning: image {i} (...{file_path[-12::]}) discarded in {type(block).__name__}")

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


class Image:

    def __init__(self, fitspath=None, data=None, header=None, **kwargs):
        if fitspath is not None:
            self.data = fits.getdata(fitspath).astype(float)
            self.header = fits.getheader(fitspath)
            self.path = fitspath
        else:
            assert data is not None, "If FITS path is not provided, data kwarg should be set"
            self.data = data
            self.header = header if header is not None else {}
            self.path = None

        self.discarded = False

        self.__dict__.update(kwargs)

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

    def citations(self, image):
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

