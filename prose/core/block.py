from time import time
import inspect
from ..console_utils import warning


class Block(object):
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
    @staticmethod
    def __new__(cls, *args, **kwargs):
        s = inspect.signature(cls.__init__)
        # TODO:
        # make copy if copy function available on each args and kwargs

        defaults = {name: value.default for name, value in s.parameters.items() if value.default != inspect._empty}
        argspecs = s.bind(None, *args, **kwargs).arguments
        defaults.update(argspecs)
        del defaults['self']
        cls._args = defaults
        cls.__new__.__signature__ = s
        return super().__new__(cls)

    def __init__(self, name=None, verbose=False):
        _name  = self.__class__.__name__
        _issue = f"https://github.com/lgrcia/prose/issues/new?title=Missing+doc+for+{_name}&body=Documentation+is+missing+for+block+{_name}"

        self.__doc__  = f"[**click to ask for documentation**]({_issue})"

        self.name = name
        self.unit_data = None
        self.processing_time = 0
        self.runs = 0
        self.in_sequence = False
        self.verbose = verbose

    @property
    def args(self):
        return self._args

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

    @property
    def citations(self):
        return None

    @staticmethod
    def _doc():
        return ""

    def __call__(self, image):
        image_copy = image.copy()
        self.run(image_copy)
        if image_copy.discard:
            warning(f"{self.__class__.__name__} discarded Image")
        return image_copy

    @classmethod
    def from_args(cls, args):
        _args, varargs, varkw, _, kwonlyargs, *_ = inspect.getfullargspec(cls.__init__)

        _args = [args[k]  for k in _args if k != 'self'] if _args is not None else []
        varargs = args[varargs] if varargs in args else []
        varkw = args[varkw] if varkw in args else {}
        kwonlyargs = {k: args[k] for k in kwonlyargs} if kwonlyargs is not None else {}

        return cls(*_args, *varargs, **varkw, **kwonlyargs)