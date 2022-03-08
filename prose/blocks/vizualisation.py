import numpy as np
from prose import utils, Block
import matplotlib.pyplot as plt
import imageio
from prose.visualization import corner_text
from skimage.transform import resize
from matplotlib.backends.backend_agg import FigureCanvasAgg


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

    def citations(self):
        return "imageio"


class RawVideo(_Video):
    
    def __init__(self, destination, attribute="data", fps=10, function=None, **kwargs):
        super().__init__(destination, fps=fps, **kwargs)
        if function is None:
            def _donothing(data): return data
            function = _donothing
        
        self.function = function
            
        self.attribute = attribute
    
    def run(self, image):
        super().run(image)
        data = self.function(image.__dict__[self.attribute])
        self.images.append(im_to_255(data, factor=1))
        
        
class PlotVideo(_Video):

    def __init__(self, plot_function, destination, fps=10, **kwargs):
        super().__init__(destination, fps=fps, **kwargs)
        self.plot_function = plot_function
        self._init_alias = plt.rcParams['text.antialiased']
        plt.rcParams['text.antialiased'] = False

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