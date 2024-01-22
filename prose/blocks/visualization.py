import io

import imageio
import matplotlib.pyplot as plt
import numpy as np

from prose.core.block import Block
from prose.utils import z_scale

__all__ = ["VideoPlot", "Video"]


def im_to_255(image):
    data = image.copy().astype(float)
    data = data / np.max(data)
    data = data * 255
    return data.astype("uint8")


class Video(Block):
    def __init__(
        self,
        destination,
        fps=10,
        compression=None,
        data_function=None,
        width=None,
        contrast=0.1,
        name=None,
    ):
        """
        A block to create a video from images data (using ffmpeg).

        Parameters
        ----------
        destination : str
            The path to save the resulting video.
        fps : int, optional
            The frames per second of the resulting video. Default is 10.
        compression : int, optional
            The compression rate of the resulting video (-cr value of fmmpeg).
            Default is None.
        data_function : callable, optional
            A function to apply to each image data before adding it to the video.
            If none, a z scale is applied to the data with a contrast given by
            :code:`contrast`.Default is None.
        width : int, optional
            The width in pixels of the resulting video.
            Default is None (i.e. original image size).
        contrast : float, optional
            The contrast of the resulting video. Default is 0.1.
            Either :code:`contrast` or :code:`data_function` must be provided.
        name : str, optional
            The name of the block. Default is None.

        Attributes
        ----------
        citations : list of str
            The citations for the block.

        Methods
        -------
        run(image)
            Adds an image to the video.
        terminate()
            Closes the video writer.

        """
        super().__init__(name=name)
        if data_function is None:

            def data_function(data):
                new_data = data.copy()
                new_data = z_scale(new_data, c=contrast)
                return new_data

        output = []
        if compression is not None:
            output += ["-crf", f"{compression}"]
        if width is not None:
            output += ["-vf", f"scale={width}:-1"]
        self.writer = imageio.get_writer(
            destination,
            mode="I",
            fps=fps,
            output_params=output if len(output) > 0 else None,
        )
        self.function = data_function

    def run(self, image):
        data = self.function(image.data)
        self.writer.append_data(im_to_255(data))

    def terminate(self):
        self.writer.close()

    @property
    def citations(self):
        return super().citations + ["imageio"]


class VideoPlot(Video):
    def __init__(
        self,
        plot_function,
        destination,
        fps=10,
        compression=None,
        width=None,
        name=None,
    ):
        """
        A block to create a video from a matploltib plot (using ffmpeg).

        Parameters
        ----------
        plot_function : callable
            A function that takes an image as input and produce a plot.
        destination : str
            The path to save the resulting video.
        fps : int, optional
            The frames per second of the resulting video. Default is 10.
        compression : int, optional
            The compression rate of the resulting video (-cr value of fmmpeg).
            Default is None.
        width : int, optional
            The width in pixels of the resulting video.
            Default is None (i.e. original image size).
        name : str, optional
            The name of the block. Default is None.

        Attributes
        ----------
        citations : list of str
            The citations for the block.

        Methods
        -------
        run(image)
            Adds a plot to the video.
        terminate()
            Closes the video writer.

        """
        super().__init__(
            destination,
            fps=fps,
            compression=compression,
            width=width,
            name=name,
        )
        self.plot_function = plot_function

    def run(self, image):
        self.plot_function(image)
        buf = io.BytesIO()
        plt.savefig(buf)
        self.writer.append_data(imageio.imread(buf))
        plt.close()

    def terminate(self):
        plt.close()
        super().terminate()
