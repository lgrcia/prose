import numpy as np
from .. import Block, viz
import matplotlib.pyplot as plt
import imageio
from prose.visualization import corner_text
from skimage.transform import resize
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time

def im_to_255(image, factor=0.25):
    if factor !=1:
        return (
            resize(
                image.astype(float),
                (np.array(np.shape(image)) * factor).astype(int),
                anti_aliasing=False,
            ) * 255).astype("uint8")
    else:
        data = image.copy().astype(float)
        data = data/np.max(data)
        data = data * 255
        return data.astype("uint8")


class _Video(Block):
    """Base block to build a video
    """

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
        return "imageio"


class RawVideo(_Video):
    
    def __init__(self, destination, attribute="data", fps=10, function=None, scale=1, **kwargs):
        super().__init__(destination, fps=fps, **kwargs)
        if function is None:
            def _donothing(data): return data
            function = _donothing
        
        self.function = function
        self.scale = scale
        self.attribute = attribute
    
    def run(self, image):
        super().run(image)
        data = self.function(image.__dict__[self.attribute])
        self.images.append(im_to_255(data, factor=self.scale))
        
        
class PlotVideo(_Video):

    def __init__(self, plot_function, destination, fps=10, antialias=False, **kwargs):
        super().__init__(destination, fps=fps, **kwargs)
        self.plot_function = plot_function
        self._init_alias = plt.rcParams['text.antialiased']
        plt.rcParams['text.antialiased'] = antialias

    def to_rbg(self):
        fig = plt.gcf()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        width, height = fig.canvas.get_width_height()
        returned = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        plt.imshow(returned)
        plt.close()
        return returned

    def run(self, image):
        super().run(image)
        self.plot_function(image)
        self.images.append(self.to_rbg())

    def terminate(self):
        super().terminate()
        plt.rcParams['text.antialiased'] = self._init_alias


class LivePlot(Block):

    def __init__(self, plot_function=None, sleep=0., size=None, **kwargs):
        super().__init__(**kwargs)
        if plot_function is None:
            plot_function = lambda im: viz.show_stars(
                im.data, im.stars_coords if hasattr(im, "stars_coords") else None,
                size=size
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
