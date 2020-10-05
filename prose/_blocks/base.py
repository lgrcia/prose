from tqdm import tqdm
from astropy.io import fits
from prose.console_utils import TQDM_BAR_FORMAT
from astropy.wcs import WCS
import prose.visualisation as viz
from collections import OrderedDict
from tabulate import tabulate


class Unit:
    # TODO: add index self.i in image within unit loop

    def __init__(self, blocks, files, name="default", show_progress=True, telescope=None, **kwargs):
        self.name = name
        self.files = files if not isinstance(files, str) else [files]
        self.blocks = blocks

        if show_progress:
            self.progress = lambda x: tqdm(
            x,
            desc=self.name,
            unit="files",
            ncols=80,
            bar_format=TQDM_BAR_FORMAT,
        )

        else:
            self.progress = lambda x: x

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

    def run(self):
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
        for file_path in self.progress(self.files):
            image = Image(file_path)
            for block in self.blocks:
                block.run(image)

        # terminate
        for block in self.blocks:
            block.terminate()

        return self

    def summary(self):
        rows = [[block.name, block.__class__.__name__] for block in self.blocks]
        headers = ["name", "type"]
        print("{} unit blocks summary:\n{}".format(self.__class__.__name__, tabulate(
            rows, headers, tablefmt="fancy_grid"
        )))

    def citations(self):
        citations = [block.citations() for block in self.blocks if block.citations() is not None]
        return citations if len(citations) > 0 else None

    def insert_before(self, before, block):
        pass




class Image:

    def __init__(self, file_path, **kwargs):
        self.data = fits.getdata(file_path)
        self.header = fits.getheader(file_path)
        self.wcs = WCS(self.header)
        self.path = file_path
        self.__dict__.update(kwargs)

    def get_other_data(self, image):
        for key, value in image.__dict__.items():
            if key not in self.__dict__:
                self.__dict__[key] = value.copy()


class Block:

    def __init__(self, name=None):
        self.name = name
        self.unit_data = None
        self.telescope = None

    def initialize(self, *args):
        pass

    def set_unit_data(self, unit_data):
        self.unit_data = unit_data

    def set_telescope(self, telescope):
        self.telescope = telescope

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

