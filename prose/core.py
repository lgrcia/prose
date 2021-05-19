from tqdm import tqdm
from astropy.io import fits
from .console_utils import TQDM_BAR_FORMAT
from astropy.wcs import WCS
from . import viz
from . import Telescope
from collections import OrderedDict
from tabulate import tabulate
import numpy as np
from time import time
from pathlib import Path
from astropy.time import Time


class Image:

    def __init__(self, fitspath=None, data=None, header=None, **kwargs):
        if fitspath is not None:
            self.path = fitspath
            self.get_data_header()
        else:
            self.data = data
            self.header = header if header is not None else {}
            self.path = None

        self.telescope = None
        self.discarded = False
        self.__dict__.update(kwargs)
        self.check_telescope()

    def get_data_header(self):
        self.data = fits.getdata(self.path).astype(float)
        self.header = fits.getheader(self.path)

    def copy(self, data=True):
        new_self = self.__class__(**self.__dict__)
        if not data:
            del new_self.__dict__["data"]

        return new_self

    def check_telescope(self):
        if self.header:
            self.telescope = Telescope.from_name(self.header["TELESCOP"])

    def get(self, keyword, default=None):
        return self.header.get(keyword, default)

    @property
    def wcs(self):
        return WCS(self.header)

    @property
    def exposure(self):
        return self.get(self.telescope.keyword_exposure_time, None)

    @property
    def jd_utc(self):
        # if jd keyword not in header compute jd from date
        if self.telescope.keyword_jd in self.header:
            jd = self.get(self.telescope.keyword_jd, None) + self.telescope.mjd
        else:
            jd = Time(self.date, scale="utc").to_value('jd') + self.telescope.mjd

        return Time(
            jd,
            format="jd",
            scale=self.telescope.jd_scale,
            location=self.telescope.earth_location).utc.value

    @property
    def date(self):
        return self.telescope.date(self.header)

    @property
    def bjd_tdb(self):
        jd_bjd = self.get(self.telescope.keyword_bjd, None)
        if jd_bjd is not None:
            jd_bjd += self.telescope.mjd

            if self.telescope.keyword_jd in self.header:
                time_format = "bjd"
            else:
                time_format = "jd"

            return Time(jd_bjd,
                        format=time_format,
                        scale=self.telescope.jd_scale,
                        location=self.telescope.earth_location).tdb.value

        else:
            return None

    @property
    def seeing(self):
        return self.get(self.telescope.keyword_seeing, None)

    @property
    def ra(self):
        return self.get(self.telescope.keyword_ra, None)

    @property
    def dec(self):
        return self.get(self.telescope.keyword_dec, None)

    @property
    def flip(self):
        return self.get(self.telescope.keyword_flip, None)

    @property
    def airmass(self):
        return self.get(self.telescope.keyword_airmass, None)

    @property
    def shape(self):
        return np.array(self.data.shape)


class Block:

    def __init__(self, name=None):
        self.name = name
        self.unit_data = None
        self.processing_time = 0
        self.runs = 0

    def initialize(self, *args):
        pass

    def set_unit_data(self, unit_data):
        self.unit_data = unit_data

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

  
class Sequence:
    # TODO: add index self.i in image within unit loop

    def __init__(self, blocks, files, name="default", loader=Image, **kwargs):
        self.name = name
        self.files_or_images = files if not isinstance(files, (str, Path)) else [files]
        self.blocks = blocks
        self.loader = loader

        self.data = {}

    def __getattr__(self, item):
        return self.blocks_dict[item]

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
            block.set_unit_data(self.data)
            block.initialize()

        # run
        for i, file_or_image in enumerate(progress(self.files_or_images)):
            if isinstance(file_or_image, (str, Path)):
                image = self.loader(file_or_image)
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
