from prose import Block
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import prose.visualisation as viz


class ShowStars(Block):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, image):
        fig = viz.fancy_show_stars(image.data, image.stars_coords)
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image.data = np.asarray(buf)
        plt.close()
