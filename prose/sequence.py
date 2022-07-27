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

    def __init__(self, blocks, name=""):
        self.name = name
        self.images = []
        self.blocks = blocks

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

    def _set_blocks_in_sequence(self, in_sequence):
        for b in self.blocks:
            b.in_sequence = in_sequence

    def run(self, images, terminate=True, show_progress=True, loader=Image):
        self._set_blocks_in_sequence(True)
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
        self._run(loader=loader)

        if terminate:
            self.terminate()
        
        for block_name, discarded in self.discards.items():
            warning(f"{block_name} discarded image{'s' if len(discarded)>1 else ''} {', '.join(discarded)}")

    def _run(self, loader=Image):
        for i, image in enumerate(self.progress(self.images)):
            if isinstance(image, (str, Path)):
                image = loader(image)

            image.i = i

            for block in self.blocks:
                block._run(image)
                # This allows to discard image in any Block
                if image.discard:
                    self.add_discard(type(block).__name__, i)
                    break

            del image
            self.n_processed_images += 1

    def terminate(self):
        for block in self.blocks:
            block.terminate()
        self._set_blocks_in_sequence(False)

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

    # io
    # --

    def add_discard(self, discard_block, i):
        if discard_block not in self.discards:
            self.discards[discard_block] = []
        self.discards[discard_block].append(str(i))


    @property
    def args(self):
        blocks = []
        for block in self.blocks:        
            blocks.append({
                'block': full_class_name(block),
                **block.args
            })

        return blocks

    @classmethod
    def from_args(cls, args):
        import prose
        
        blocks = []
        for block_dict in args:
            block_class = block_dict["block"]
            del block_dict["block"]
            block = eval(block_class).from_args(block_dict)
            blocks.append(block)
            
        return cls(blocks)

    @property
    def params_str(self):
        return yaml.safe_dump(self.args, sort_keys=False)


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