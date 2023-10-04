import inspect
from time import time
from typing import Union

from prose.console_utils import warning
from prose.core.image import Buffer, Image

import contextlib


@contextlib.contextmanager
def _exception_context(msg):
    try:
        yield
    except Exception as ex:
        if ex.args:
            msg = f"[{msg}] {ex.args[0]}"
        else:
            str(msg)
        ex.args = (msg,) + ex.args[1:]
        raise


class Block(object):
    """Single unit of processing acting on the :py:class:`~prose.Image` object

    Reading, processing and writing :py:class:`~prose.Image` attributes. When placed in a sequence, it goes through two steps:

        1. :py:meth:`~prose.Block.run` on each image fed to the :py:class:`~prose.Sequence`
        2. :py:meth:`~prose.Block.terminate` called after the :py:class:`~prose.Sequence` is terminated

    Parameters
    ----------
    name : str, optional
        name of the block, by default None
    size: int, optional
        number of images processed by the block, by default 1

    All prose blocks must be child of this parent class
    """

    def __init__(self, name=None, verbose=False, size=1, read=None):
        assert size % 2 == 1, "block size must be odd"
        _name = self.__class__.__name__
        _issue = f"https://github.com/lgrcia/prose/issues/new?title=Missing+doc+for+{_name}&body=Documentation+is+missing+for+block+{_name}"

        self.__doc__ = f"[**click to ask for documentation**]({_issue})"

        self.name = name
        self.unit_data = None
        self.processing_time = 0
        self.runs = 0
        self.in_sequence = False
        self.verbose = verbose

        self._data_block = False
        self.size = size
        if read is not None:
            assert isinstance(read, list), "require must be a list"
            self.read = read
        else:
            self.read = []

    @property
    def args(self):
        return self._args

    def _check_require(self, image):
        for _require in self.read:
            if _require == "sources":
                if len(image.sources) == 0:
                    raise AttributeError(f"Image must have sources (0 found)")
            elif _require == "wcs":
                if not image.plate_solved:
                    raise AttributeError(f"Image must have valid WCS")
            if not hasattr(image, _require) and not hasattr(image.computed, _require):
                raise AttributeError(
                    f"[{self.__class__.__name__}] Image must have attribute '{_require}'"
                )

    def _run(self, buffer):
        t0 = time()
        if isinstance(buffer, Buffer):
            image = buffer[0] if self.size == 1 else buffer
        elif isinstance(buffer, Image):
            image = buffer
        else:
            raise ValueError("block must be run on a Buffer or an Image")
        with _exception_context(self.__class__.__name__):
            self._check_require(image)
            self.run(image)
        self.processing_time += time() - t0
        self.runs += 1

    def run(self, image: Image):
        """Running on a image (must be overwritten when subclassed)

        Parameters
        ----------
        image : prose.Image
            image to be processed
        """
        raise NotImplementedError()

    def terminate(self):
        """Method called after block's :py:class:`~prose.Sequence` is finished (if any)"""
        pass

    @property
    def citations(self) -> list:
        """
        Returns a string listing the packages used by the block.

        Returns
        -------
        str
            A string listing the packages used by the block.
        """
        return ["astropy", "prose", "numpy", "scipy"]

    @staticmethod
    def _doc():
        return ""

    def __call__(self, image):
        image_copy = image.copy()
        self._run(image_copy)
        if image_copy.discard:
            warning(f"{self.__class__.__name__} discarded Image")
        return image_copy


# rewrite the tested function to accept Block instances as well as strings
# requires pytest and is not used in the docs for now
def is_tested(block_class: Union[Block, str]) -> bool:
    """Check if a block is tested

    Parameters
    ----------
    block_class : Block or str
        block to be tested

    Returns
    -------
    bool
        True if the block is tested, False otherwise

    Raises
    ------
    TypeError
        if block is not a Block subclass or a string
    """
    import pytest

    if isinstance(block_class, str):
        block_name = block_class
    elif issubclass(block_class, Block):
        block_name = block_class.__name__
    else:
        raise TypeError(f"block must be a Block subclass or a string")
    result = pytest.main(["-q", "-k", block_name])
    return result == 0
