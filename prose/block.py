from time import time


class Block:
    """Single unit of processing acting on the :py:class:`~prose.Image` object
    
    Reading, processing and writing :py:class:`~prose.Image` attributes. When placed in a sequence, it goes through two steps:

        1. :py:meth:`~prose.Block.run` on each image fed to the :py:class:`~prose.Sequence`
        2. :py:meth:`~prose.Block.terminate` called after the :py:class:`~prose.Sequence` is terminated


    Parameters
    ----------
    name : str, optional
        name of the block, by default None

    All prose blocks must be child of this parent class
    """

    def __init__(self, name=None):
        """Instanciation
        """
        self.name = name
        self.unit_data = None
        self.processing_time = 0
        self.runs = 0

        # recording args and kwargs for reproducibility
        # when subclassing, use @utils.register_args decorator (see docs)

    def _run(self, *args, **kwargs):
        t0 = time()
        self.run(*args, **kwargs)
        self.processing_time += time() - t0
        self.runs += 1

    def run(self, image, **kwargs):
        """Running on a image (must be overwritten when subclassed)

        Parameters
        ----------
        image : prose.Image
            image to be processed
        """
        raise NotImplementedError()

    def terminate(self):
        """Method called after block's :py:class:`~prose.Sequence` is finished (if any)
        """
        pass

    @staticmethod
    def citations():
        return None

    @staticmethod
    def _doc():
        return ""

    def concat(self, block):
        return self

    def __call__(self, image):
        image_copy = image.copy()
        self.run(image_copy)
        return image_copy

    @staticmethod
    def concatenate(blocks):
        block = blocks[0]
        for b in blocks[1::]:
            block.concat(b)
        return block