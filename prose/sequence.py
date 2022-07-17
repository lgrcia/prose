from re import A
from tqdm import tqdm
import xarray
from .console_utils import TQDM_BAR_FORMAT, warning, error
from collections import OrderedDict
from tabulate import tabulate
import numpy as np
from time import time
from .image import Image
from pathlib import Path
from functools import partial
import multiprocessing as mp
from .blocks.utils import DataBlock
import sys
import yaml
from .utils import full_class_name

def progress(name, x, **kwargs):
    return tqdm(
        x,
        desc=name,
        unit="images",
        ncols=80,
        bar_format=TQDM_BAR_FORMAT,
        **kwargs
    )

class Sequence:
    # TODO: add index self.i in image within unit loop

    def __init__(self, blocks, name="", loader=Image):
        self.name = name
        self.images = []
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

    def run(self, images, telescope=None, terminate=True, show_progress=True):
        self.images = images if not isinstance(images, (str, Path, Image)) else [images]

        if not show_progress:
            def _p(x, **kwargs): return x
            self.progress = _p
        else:
            self.progress = partial(progress, self.name)

        if isinstance(self.images, list):
            if len(self.images) == 0:
                raise ValueError("No images to process")
        elif self.images is None:
            raise ValueError("No images to process")

        # run
        self.n_processed_images = 0
        self.discards = {}
        self._run(telescope=telescope)

        if terminate:
            self.terminate()
        
        for block_name, discarded in self.discards.items():
            warning(f"{block_name} discarded image{'s' if len(discarded)>1 else ''} {', '.join(discarded)}")

    def _run(self, telescope=None):
        for i, image in enumerate(self.progress(self.images)):
            if isinstance(image, (str, Path)):
                image = self.loader(image, telescope=telescope)

            image.i = i

            for block in self.blocks:
                # This allows to discard image in any Block
                if image.discard:
                    discard_block = image.discard_block
                    self.add_discard(discard_block, i)
                    break
                else:
                    block._run(image)
                    if image.discard:
                        image.discard_block = type(block).__name__

            del image
            self.n_processed_images += 1

    def terminate(self):
        for block in self.blocks:
            block.terminate()

    def __str__(self):
        rows = [[
            i, block.name, block.__class__.__name__, f"{block.processing_time:.3f} s ({(block.processing_time/self.processing_time)*100:.0f}%)"] 
            for i, block in enumerate(self.blocks)
            ]
        headers = ["index", "name", "type", "processing"]

        return tabulate(rows, headers, tablefmt="fancy_grid")

    def __repr__(self) -> str:
        return self.__str__()

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

    def add_discard(self, discard_block, i):
        if discard_block not in self.discards:
            self.discards[discard_block] = []
        self.discards[discard_block].append(str(i))


    def params_dict(self):
        d = {}
        for block in self.blocks:
            params = {
                'args': {k: a.tolist() if isinstance(a, np.ndarray) else a for k, a in block.args.items()}, 
                'kwargs': {k:v.tolist() if isinstance(v, np.ndarray) else v for k, v in block.kwargs.items()}
            }
            if len(block.args) == 0:
                del params['args']
            if len(block.kwargs) == 0:
                del params['kwargs']
                
            d[full_class_name(block)] = params

        return d

    @property
    def params_str(self):
        return yaml.safe_dump(self.params_dict(), sort_keys=False)

    def from_params_str(self, params_str):
        pass


class MPSequence(Sequence):

    def __init__(self, blocks, data_blocks=None, name="", loader=Image):
        super().__init__(blocks, name=name, loader=loader)
        if data_blocks is None:
            self.data = None
            self._has_data = False
        else:
            self.data = Sequence(data_blocks)
            self._has_data = True

    def check_data_blocks(self):
        bad_blocks = []
        for b in self.blocks: 
            if isinstance(b, DataBlock):
                bad_blocks.append(f"{b.__class__.__name__}")
        if len(bad_blocks) > 0:
            bad_blocks = ', '.join(list(np.unique(bad_blocks)))
            error(f"Data blocks [{bad_blocks}] cannot be used in MPSequence\n\nConsider using the data_blocks kwargs")
            sys.exit()
    
    def _run(self, telescope=None):
        self.check_data_blocks()

        self.n_processed_images = 0
        n = len(self.images)
        processed_blocks = mp.Manager().list(self.blocks)
        images_i = list(enumerate(self.images))

        with mp.Pool() as pool:
            for image in self.progress(pool.imap(partial(
                _run_all,
                blocks=processed_blocks,
                loader=self.loader
            ), images_i), total=n):
                if not image.discard:
                    if self._has_data:
                        self.data.run(image, terminate=False, show_progress=False)
                else:
                    self.add_discard(image.discard_block, image.i)

    def terminate(self):
        if self._has_data:
            self.data.terminate()

def _run_all(image_i, blocks=None, loader=None):

    i, image = image_i

    if isinstance(image, (str, Path)):
        image = loader(image)
    
    image.i = i

    for block in blocks:
        # This allows to discard image in any Block
        if image.discard:
            return image
        else:
            block._run(image)
            if image.discard:
                image.discard_block = type(block).__name__

    return image