from tqdm import tqdm
from astropy.io import fits
from prose.console_utils import TQDM_BAR_FORMAT
from astropy.wcs import WCS
import prose.visualisation as viz
from collections import OrderedDict
from tabulate import tabulate


class Unit:
    # TODO: add index self.i in image within unit loop

    def __init__(self, blocks, fits_manager, name="default", files="light", show_progress=True, n_images=None, **kwargs):
        self.name = name
        self.fits_manager = fits_manager
        self.blocks = blocks

        self.retrieve_files(files, n_images=n_images)

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

        if self.fits_manager.has_stack():
            self.stack_image = Image(self.fits_manager.get("stack")[0])

    @property
    def blocks(self):
        return self.blocks_dict.values()

    @blocks.setter
    def blocks(self, blocks):
        self.blocks_dict = OrderedDict({
            block.name if block.name is not None else "block{}".format(i): block
            for i, block in enumerate(blocks)
        })

    def retrieve_files(self, keyword, n_images=None):
        self.fits_manager.files = self.fits_manager.get(keyword, n_images=n_images)
        self.files = self.fits_manager.files

    def get_data_header(self, file_path):
        return fits.getdata(file_path), fits.getheader(file_path)

    def run(self):
        if isinstance(self.files, list):
            if len(self.files) == 0:
                raise ValueError("No files to process")
        elif self.files is None:
            raise ValueError("No files to process")

        for block in self.blocks:
            block.initialize(self.fits_manager)

        stack_blocks = [block for block in self.blocks if block.stack]
        blocks = [block for block in self.blocks if not block.stack]
        has_stack_block = len(stack_blocks) > 0

        for block in stack_blocks:
            block.run(self.stack_image)
            block.stack_method(self.stack_image)

        for file_path in self.progress(self.files):
            image = Image(file_path)
            if has_stack_block:
                image.get_other_data(self.stack_image)
            for block in blocks:
                block.run(image)

        for block in self.blocks:
            block.terminate()

    def summary(self):
        rows = [[block.name, block.__class__.__name__] for block in self.blocks]
        headers = ["name", "type"]
        print("{} unit blocks summary:\n{}".format(self.__class__.__name__, tabulate(
            rows, headers, tablefmt="fancy_grid"
        )))

    def citations(self):
        citations = [block.citations() for block in self.blocks if block.citations() is not None]
        return citations if len(citations) > 0 else None


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

    def __init__(self, stack=False, name=None):
        self.stack = stack
        self.name = name

    def initialize(self, *args):
        pass

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


class PrintDim(Block):

    def __init__(self):
        pass

    def initialize(self, *args):
        print("I am a block")

    def run(self, image):
        pass


class Reduction(Unit):

    def __init__(self, ):
        pass


