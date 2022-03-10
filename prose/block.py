from time import time


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