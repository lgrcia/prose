from tqdm import tqdm
from astropy.io import fits
from .console_utils import TQDM_BAR_FORMAT
from astropy.wcs import WCS
from . import viz, utils, Telescope
from collections import OrderedDict
from tabulate import tabulate
import numpy as np
from time import time
from pathlib import Path
from astropy.time import Time
from functools import partial
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from .utils import register_args
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle
from dateutil import parser as dparser

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
        self.discard = False
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
           self.telescope = Telescope.from_names(self.header.get("INSTRUME", ""), self.header.get("TELESCOP", ""))

    def get(self, keyword, default=None):
        return self.header.get(keyword, default)

    @property
    def wcs(self):
        return WCS(self.header)

    @wcs.setter
    def wcs(self, new_wcs):
        self.header.update(new_wcs.to_header())

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
        _ra = self.get(self.telescope.keyword_ra, None)
        if _ra is not None:
            _ra = Angle(_ra, self.telescope.ra_unit).to(u.deg)
        return _ra

    @property
    def dec(self):
        _dec = self.get(self.telescope.keyword_dec, None)
        if _dec is not None:
            _dec = Angle(_dec, self.telescope.dec_unit).to(u.deg)
        return _dec

    @property
    def flip(self):
        return self.get(self.telescope.keyword_flip, None)

    @property
    def airmass(self):
        return self.get(self.telescope.keyword_airmass, None)

    @property
    def shape(self):
        return np.array(self.data.shape)

    @property
    def date(self):
        dparser.parse(self.header[self.telescope.keyword_observation_date])

    def show(self, 
        cmap="Greys_r", 
        ax=None, 
        figsize=(10,10), 
        stars=None, 
        stars_labels=True, 
        vmin=True, 
        vmax=None, 
        scale=1.5,
        frame=False
        ):
        if ax is None:
            if not isinstance(figsize, (list, tuple)):
                if isinstance(figsize, (float, int)):
                    figsize = (figsize, figsize)
                else:
                    raise TypeError("figsize must be tuple or list or float or int")
            fig = plt.figure(figsize=figsize)
            if frame:
                ax = fig.add_subplot(111, projection=self.wcs)
            else:
                ax = fig.add_subplot(111)

        if vmin is True or vmax is True:
            med = np.nanmedian(self.data)
            vmin = med
            vmax = scale*np.nanstd(self.data) + med
            _ = ax.imshow(self.data, cmap=cmap, origin="lower",vmin=vmin,vmax=vmax)
        elif all([vmin, vmax]) is False:
            _ = ax.imshow(utils.z_scale(self.data, 0.05*scale), cmap=cmap, origin="lower")
        else:
            _ = ax.imshow(self.data, cmap=cmap, origin="lower",vmin=vmin,vmax=vmax)
        
        if stars is None:
            stars = "stars_coords" in self.__dict__
        
        if stars:
            label = np.arange(len(self.stars_coords)) if stars_labels else None
            viz.plot_marks(*self.stars_coords.T, label=label, ax=ax)

        if frame:
            overlay = ax.get_coords_overlay(self.wcs)
            overlay.grid(color='white', ls='dotted')
            overlay[0].set_axislabel('Right Ascension (J2000)')
            overlay[1].set_axislabel('Declination (J2000)')

    def show_cutout(self, star=None, size=200, marks=True, **kwargs):
        """
        Show a zoomed cutout around a detected star or coordinates

        Parameters
        ----------
        star : [type], optional
            detected star id or (x, y) coordinate, by default None
        size : int, optional
            side size of square cutout in pixel, by default 200
        """

        if star is None:
            x, y = self.stars_coords[self.target]
        elif isinstance(star, int):
            x, y = self.stars_coords[star]
        elif isinstance(star, (tuple, list, np.ndarray)):
            x, y = star
        else:
            raise ValueError("star type not understood")

        self.show(**kwargs)
        plt.xlim(np.array([-size / 2, size / 2]) + x)
        plt.ylim(np.array([-size / 2, size / 2]) + y)
        if marks and hasattr(self, "stars_coords"):
            idxs = np.argwhere(np.max(np.abs(self.stars_coords - [x, y]), axis=1) < size).squeeze()
            viz.plot_marks(*self.stars_coords[idxs].T, label=idxs)

    @property
    def skycoord(self):
        """astropy SkyCoord object based on header RAn, DEC
        """
        return SkyCoord(self.ra, self.dec, frame='icrs')


    @property
    def fov(self):
        return np.array(self.shape) * self.pixel_scale.to(u.deg)

    @property
    def pixel_scale(self):
        return self.telescope.pixel_scale.to(u.arcsec)

class Block:
    """A ``Block`` is a single unit of processing acting on the ``Image`` object, reading, processing and writing its attributes. When placed in a sequence, it goes through three steps:

        1. :py:meth:`~prose.Block.initialize` method is called before the sequence is run
        2. *Images* go succesively and sequentially through its :py:meth:`~prose.run` methods
        3. :py:meth:`~prose.Block.terminate` method is called after the sequence is terminated

        Parameters
        ----------
        name : [type], optional
            [description], by default None
    """

    def __init__(self, name=None):
        """[summary]

        Parameters
        ----------
        name : [type], optional
            [description], by default None
        """
        self.name = name
        self.unit_data = None
        self.processing_time = 0
        self.runs = 0

        # recording args and kwargs for reproducibility
        # when subclassing, use @utils.register_args decorator (see docs)

    def initialize(self, *args):
        pass
    
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

    def concat(self, block):
        return self

    def __call__(self, image):
        image_copy = image.copy()
        self.run(image_copy)
        return image_copy

  
class Sequence:
    # TODO: add index self.i in image within unit loop

    def __init__(self, blocks, name="", loader=Image, **kwargs):
        self.name = name
        self.files_or_images = []
        self.blocks = blocks
        self.loader = loader

        self.data = {}
        self.n_processed_images = None

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

    def run(self, images, show_progress=True):

        self.files_or_images = images if not isinstance(images, (str, Path, Image)) else [images]

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

        self.n_processed_images = 0

        # run
        for i, file_or_image in enumerate(progress(self.files_or_images)):
            if isinstance(file_or_image, (str, Path)):
                image = self.loader(file_or_image)
            else:
                image = file_or_image
            image.i = i
            self._last_image = image
            discard_message = False

            last_block = None

            for b, block in enumerate(self.blocks):
                # This allows to discard image in any Block
                if not image.discard:
                    block._run(image)
                    # except:
                    #     # TODO
                    #     if not last_block is None:
                    #         print(f"{type(last_block).__name__} failed")
                elif not discard_message:
                    last_block = self.blocks[b-1]
                    discard_message = True
                    print(f"Warning: image {i} discarded in {type(last_block).__name__}")

            del image
            self.n_processed_images += 1

        # terminate
        for block in self.blocks:
            block.terminate()

    def __str__(self):
        rows = [[
            block.name, block.__class__.__name__, f"{block.processing_time:.3f} s ({(block.processing_time/self.processing_time)*100:.0f}%)"] 
            for block in self.blocks
            ]
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

    def __getitem__(self, item):
        return self.blocks[item]

    # import/export properties
    # ------------------------

    @staticmethod
    def from_dicts(blocks_dicts):
        blocks = []
        for block_dict in blocks_dicts:
            block = block_dict["block"](*block_dict["args"], **block_dict["kwargs"])
            block.name = block_dict["name"]
            blocks.append(block)
            
        return Sequence(blocks)

    @property
    def as_dicts(self):
        blocks = []
        for block in self.blocks:
            blocks.append(dict(
                block=block.__class__,
                name=block.name,
                args=block.args,
                kwargs=block.kwargs
            ))

        return blocks


class MultiProcessSequence(Sequence):
    
    def run(self, show_progress=True):
        if show_progress:
            def progress(x, **kwargs): 
                return tqdm(
                    x,
                    desc=self.name,
                    unit="images",
                    ncols=80,
                    bar_format=TQDM_BAR_FORMAT,
                    **kwargs
                )

        else:
            def progress(x, **kwargs): return x

        if isinstance(self.files_or_images, list):
            if len(self.files_or_images) == 0:
                raise ValueError("No images to process")
        elif self.files_or_images is None:
            raise ValueError("No images to process")

        self.n_processed_images = 0
        
        processed_blocks = mp.Manager().list(self.blocks)
        blocks_queue = mp.Manager().Queue()
        
        blocks_writing_process = mp.Process(
            target=partial(
                _concat_blocks,
                current_blocks=processed_blocks,
            ), args=(blocks_queue,)
        )
    
        blocks_writing_process.deamon = True
        blocks_writing_process.start()
        
        with mp.Pool() as pool:
            for _ in progress(pool.imap(partial(
                _run_blocks_on_image,
                blocks_queue=blocks_queue,
                blocks_list = self.blocks,
                loader = self.loader
            ), self.files_or_images), total=len(self.files_or_images)):
                pass
            
        blocks_queue.put("done")

        self.blocks = processed_blocks
   
        # terminate
        for block in self.blocks:
            block.terminate()


def _run_blocks_on_image(file_or_image, blocks_queue=None, blocks_list=None, loader=None):

    if isinstance(file_or_image, (str, Path)):
        image = loader(file_or_image)
    else:
        image = file_or_image

    discard_message = False
    last_block = None

    for b, block in enumerate(blocks_list):
        # This allows to discard image in any Block
        if not image.discard:
            block._run(image)
        elif not discard_message:
            last_block = blocks_list[b-1]
            discard_message = True
            print(f"Warning: image ? discarded in {type(last_block).__name__}")

    del image
    blocks_queue.put(blocks_list)
    
def _concat_blocks(blocks_queue, current_blocks=None):
    while True:
        new_blocks = blocks_queue.get()
        if new_blocks == "done":
            break
        else:
            for i, block in enumerate(new_blocks):
                block.concat(current_blocks[i])
                current_blocks[i] = block