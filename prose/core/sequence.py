import sys
from collections import OrderedDict
from functools import partial
from pathlib import Path
from time import time

import multiprocess as mp
import numpy as np
import yaml
from tabulate import tabulate
from tqdm.autonotebook import tqdm

from prose.citations import citations as default_citations
from prose.console_utils import TQDM_BAR_FORMAT, error, warning
from prose.core.image import Buffer, FITSImage, Image
from prose.utils import full_class_name


def progress(name, x, **kwargs):
    return tqdm(x, desc=name, unit="images", **kwargs)


class Sequence:
    def __init__(self, blocks, name=None):
        """A sequence of :py:class:`Block` objects to sequentially process images

        Parameters
        ----------
        blocks : list
            list of :py:class:`Block` objects
        name : str, optional
            name of the sequence, by default None
        """
        self.name = name
        self.images = []
        self.blocks_dict = None
        self.blocks = blocks

        self.data = {}
        self.n_processed_images = None
        self.last_image = None

        # initially the buffer must have a size of max(front_size) + max(back_size) in
        # order to hold all images necessary to all blocks
        buffer_size = np.max([block.size for block in self.blocks])
        self.buffer = Buffer(buffer_size)

    def __getattr__(self, item):
        return self.blocks_dict[item]

    @property
    def blocks(self):
        """list of :py:class:`Block` objects

        Returns
        -------
        _type_
            _description_
        """
        return list(self.blocks_dict.values())

    @blocks.setter
    def blocks(self, blocks):
        self.blocks_dict = OrderedDict(
            {
                block.name if block.name is not None else "block{}".format(i): block
                for i, block in enumerate(blocks)
            }
        )

    def _set_blocks_in_sequence(self, in_sequence):
        for b in self.blocks:
            b.in_sequence = in_sequence

    def run(self, images, terminate=True, show_progress=True, loader=FITSImage):
        """Run the sequence

        Parameters
        ----------
        images : list, str, :py:class:`Image`
            :py:class:`Image` object or path (single or as a list) to be processed by the sequence
        terminate : bool, optional
            whether to run :py:class:`Sequence.terminate` at the end of the sequence, by default True
        show_progress : bool, optional
            whether to show a progress bar, by default True
        loader : Image sub-class, optional
            An Image sub-class to load images path(s) of provided as inputs, by default py:class:`Image`
        """
        self._set_blocks_in_sequence(True)
        self.images = images if not isinstance(images, (str, Path, Image)) else [images]
        assert len(self.images) != 0, "Empty array or no images provided"

        if not show_progress:

            def _p(x, **kwargs):
                return x

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
            warning(
                f"{block_name} discarded image{'s' if len(discarded)>1 else ''} {', '.join(discarded)}"
            )

    def _load(self, image, loader=FITSImage):
        _image = loader(image) if isinstance(image, (str, Path)) else image
        return _image

    def _run(self, loader=FITSImage):
        self.buffer.loader = partial(self._load, loader=loader)
        self.buffer.init(self.images)

        for i, buffer in enumerate(self.progress(self.buffer, total=len(self.images))):
            buffer.current.i = i
            self.last_image = buffer.current

            for block in self.blocks:
                block._run(buffer)
                # This allows to discard image in any Block
                if buffer.current is not None:
                    if buffer.current.discard:
                        self._add_discard(type(block).__name__, buffer.current.i)
                        break

            self.n_processed_images += 1

    def terminate(self):
        """Run the :py:class:`Block.terminate` method of all blocks"""
        for block in self.blocks:
            block.terminate()
        self._set_blocks_in_sequence(False)

    def __str__(self):
        rows = [
            [
                i,
                block.name,
                block.__class__.__name__,
                f"{block.processing_time:.3f} s ({(block.processing_time/self.processing_time)*100:.0f}%)",
            ]
            for i, block in enumerate(self.blocks)
        ]
        headers = ["index", "name", "type", "processing"]

        return tabulate(rows, headers, tablefmt="fancy_grid")

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def processing_time(self):
        """Total processing time of the sequence last run"""
        return np.sum([block.processing_time for block in self.blocks])

    def __getitem__(self, item):
        return self.blocks[item]

    # io
    # --

    def _add_discard(self, discard_block, i):
        if discard_block not in self.discards:
            self.discards[discard_block] = []
        self.discards[discard_block].append(str(i))

    @property
    def args(self):
        blocks = []
        for block in self.blocks:
            blocks.append({"block": full_class_name(block), **block.args})

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

    def citations(self):
        """Return the citations of the sequence"""

        # concatenate all blocks citations
        citations = []
        for block in self.blocks:
            citations += block.citations

        # remove duplicates
        citations = list(set(citations))
        citation_dict = {}

        for name in citations:
            if name[0] == "@":
                citation_dict[name] = name
            else:
                citation_dict[name] = default_citations[name]

        tex_citep = ", ".join(
            [
                f"{name} \citep{{{name}}}"
                for name in citation_dict.keys()
                if name not in ["prose", "astropy"]
            ]
        )
        tex_citep += " and astropy \citep{astropy}"
        tex = (
            f"This research made use of \\textsf{{prose}} \citep{{prose}} and its dependencies ({tex_citep})."
            ""
        )

        return tex, "\n\n".join(citation_dict.values())


class SequenceParallel(Sequence):
    """
    A multi-process :py:class:`Sequence` of blocks to be executed in parallel.

    The data_blocks allow blocks carying large amount of data to be run sequentially
    so that they are not copied from one process to another.

    Parameters
    ----------
    blocks : list
        A list of blocks to be executed in parallel.
    data_blocks : list, optional
        A list of data blocks to be executed in parallel.
    name : str, optional
        A name for the sequence.
    """

    def __init__(self, blocks, data_blocks=None, name=""):
        super().__init__(blocks, name=name)
        if data_blocks is None:
            self.data = None
            self._has_data = False
        else:
            self.data = Sequence(data_blocks)
            self._has_data = True

    def check_data_blocks(self):
        bad_blocks = []
        for b in self.blocks:
            if b._data_block:
                bad_blocks.append(f"{b.__class__.__name__}")
        if len(bad_blocks) > 0:
            bad_blocks = ", ".join(list(np.unique(bad_blocks)))
            error(
                f"Data blocks [{bad_blocks}] cannot be used in MPSequence\n\nConsider using the data_blocks kwargs"
            )
            sys.exit()

    def _run(self, loader=FITSImage):
        self.check_data_blocks()

        self.n_processed_images = 0
        n = len(self.images)
        processed_blocks = mp.Manager().list(self.blocks)
        images_i = list(enumerate(self.images))

        with mp.Pool() as pool:
            for image in self.progress(
                pool.imap(
                    partial(_run_all, blocks=processed_blocks, loader=loader), images_i
                ),
                total=n,
            ):
                if not image.discard:
                    if self._has_data:
                        self.data.run(image, terminate=False, show_progress=False)
                else:
                    self._add_discard(image.discard_block, image.i)

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
