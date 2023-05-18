import io
import shutil
import tempfile
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.transform import resize

from prose import Block, viz
from prose.visualization import corner_text

__all__ = ["VideoPlot"]


def im_to_255(image, factor=0.25):
    if factor != 1:
        return (
            resize(
                image.astype(float),
                (np.array(np.shape(image)) * factor).astype(int),
                anti_aliasing=False,
            )
            * 255
        ).astype("uint8")
    else:
        data = image.copy().astype(float)
        data = data / np.max(data)
        data = data * 255
        return data.astype("uint8")


class _Video(Block):
    """Base block to build a video"""

    def __init__(self, destination, fps=10, **kwargs):
        super().__init__(**kwargs)
        self.destination = destination
        self.images = []
        self.fps = fps
        self.checked_writer = False

    def run(self, image):
        if not self.checked_writer:
            _ = imageio.get_writer(self.destination, mode="I")
            self.checked_writer = True

    def terminate(self):
        imageio.mimsave(self.destination, self.images, fps=self.fps)

    @property
    def citations(self):
        return super().citations + ["imageio"]


class RawVideo(_Video):
    def __init__(
        self, destination, attribute="data", fps=10, function=None, scale=1, **kwargs
    ):
        super().__init__(destination, fps=fps, **kwargs)
        if function is None:

            def _donothing(data):
                return data

            function = _donothing

        self.function = function
        self.scale = scale
        self.attribute = attribute

    def run(self, image):
        super().run(image)
        data = self.function(image.__dict__[self.attribute])
        self.images.append(im_to_255(data, factor=self.scale))


class VideoPlot(_Video):
    def __init__(self, plot_function, destination, fps=10, name=None):
        """Make a video out of a plotting function

        Parameters
        ----------
        plot_function : function
            a plotting function taking an :py:class:`prose.Image` as input
        destination : str or Path
            destination of the video, including extension
        fps : int, optional
            frame per seconds, by default 10
        antialias : bool, optional
            whether pyplot antialias should be used, by default False
        """
        super().__init__(destination, fps=fps, name=name)
        self.plot_function = plot_function
        self.destination = destination
        self._temp = tempfile.mkdtemp()
        self._images = []

    def run(self, image):
        self.plot_function(image)
        buf = io.BytesIO()
        plt.savefig(buf)
        self.images.append(imageio.imread(buf))
        plt.close()

    def terminate(self):
        super().terminate()
        shutil.rmtree(self._temp)


class LivePlot(Block):
    def __init__(self, plot_function=None, sleep=0.0, size=None, **kwargs):
        super().__init__(**kwargs)
        if plot_function is None:
            plot_function = lambda im: viz.show_stars(
                im.data,
                im.stars_coords if hasattr(im, "stars_coords") else None,
                size=size,
            )

        self.plot_function = plot_function
        self.sleep = sleep
        self.display = None
        self.size = size
        self.figure_added = False

    def run(self, image):
        if not self.figure_added:
            from IPython import display as disp

            self.display = disp
            if isinstance(self.size, tuple):
                plt.figure(figsize=self.size)
            self.figure_added = True

        self.plot_function(image)
        self.display.clear_output(wait=True)
        self.display.display(plt.gcf())
        time.sleep(self.sleep)
        plt.cla()

    def terminate(self):
        plt.close()
