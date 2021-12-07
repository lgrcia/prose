import numpy as np
from ..core import Block, Image
from .. import utils
import matplotlib.pyplot as plt
from .. import viz
from astropy.nddata import Cutout2D
from ..console_utils import info
from time import sleep

np.seterr(divide="ignore")


def easy_median(images):
    # To avoid memory errors, we split the median computation in 50
    images = np.array(images)
    shape_divisors = utils.divisors(images.shape[1])
    n = shape_divisors[np.argmin(np.abs(50 - shape_divisors))]
    return np.concatenate([np.median(im, axis=0) for im in np.split(images, n, axis=1)])


class Calibration(Block):
    """
    Flat, Bias and Dark calibration

    Parameters
    ----------
    darks : list
        list of dark files paths
    flats : list
        list of flat files paths
    bias : list
        list of bias files paths
    """
    def __init__(self, darks=None, flats=None, bias=None, loader=Image, **kwargs):

        super().__init__(**kwargs)
        if darks is None:
            darks = []
        if flats is None:
            flats = []
        if bias is None:
            bias = []
        self.images = {
            "dark": darks,
            "flat": flats,
            "bias": bias
        }

        self.master_dark = None
        self.master_flat = None
        self.master_bias = None

        self.loader = loader

    def calibration(self, image, exp_time):
        return (image - (self.master_dark * exp_time + self.master_bias)) / self.master_flat

    def _produce_master(self, image_type):
        _master = []
        images = self.images[image_type]

        if len(images) == 0:
            info(f"No {image_type} images set")
            if image_type == "dark":
                self.master_dark = 0
            elif image_type == "bias":
                self.master_bias = 0
            elif image_type == "flat":
                self.master_flat = 1

        for image_path in images:
            image = self.loader(image_path)
            if image_type == "dark":
                _dark = (image.data - self.master_bias) / image.exposure
                _master.append(_dark)
            elif image_type == "bias":
                _master.append(image.data)
            elif image_type == "flat":
                _flat = image.data - self.master_bias - self.master_dark*image.exposure
                _flat /= np.mean(_flat)
                _master.append(_flat)
                del image

        if len(_master) > 0:
            med = easy_median(_master)
            if image_type == "dark":
                self.master_dark = med.copy()
            elif image_type == "bias":
                self.master_bias = med.copy()
            elif image_type == "flat":
                self.master_flat = med.copy()
            del _master

    def initialize(self):
        if self.master_bias is None:
            self._produce_master("bias")
        if self.master_dark is None:
            self._produce_master("dark")
        if self.master_flat is None:
            self._produce_master("flat")
        sleep(0.1)

    def plot_masters(self):
        plt.figure(figsize=(40, 10))
        plt.subplot(131)
        plt.title("Master bias")
        im = plt.imshow(utils.z_scale(self.master_bias), cmap="Greys_r")
        viz.add_colorbar(im)
        plt.subplot(132)
        plt.title("Master dark")
        im = plt.imshow(utils.z_scale(self.master_dark), cmap="Greys_r")
        viz.add_colorbar(im)
        plt.subplot(133)
        plt.title("Master flat")
        im = plt.imshow(utils.z_scale(self.master_flat), cmap="Greys_r")
        viz.add_colorbar(im)

    def run(self, image, **kwargs):
        data = image.data
        calibrated_image = self.calibration(data, image.exposure)
        calibrated_image[calibrated_image < 0] = 0.
        calibrated_image[~np.isfinite(calibrated_image)] = -1

        image.data = calibrated_image

    def citations(self):
        return "astropy", "numpy"


class Trim(Block):
    """Image trimming. If trim is not specified, triming is taken from the telescope characteristics

    Parameters
    ----------
    skip_wcs : bool, optional
        whether to skip applying trim to WCS, by default False
    trim : tuple, optional
        (x, y) trim values, by default None
    """

    def __init__(self, skip_wcs=False, trim=None, **kwargs):

        super().__init__(**kwargs)
        self.skip_wcs = skip_wcs
        self.trim = trim

    def run(self, image, **kwargs):
        shape = image.shape
        center = shape[::-1] / 2
        trim = self.trim if self.trim is not None else image.telescope.trimming[::-1]
        dimension = shape - 2 * np.array(trim)
        trim_image = Cutout2D(image.data, center, dimension, wcs=None if self.skip_wcs else image.wcs)
        image.data = trim_image.data
        if not self.skip_wcs:
            image.header.update(trim_image.wcs.to_header())

    def __call__(self, data):
        trim_x, trim_y = self.trim
        return data[trim_x:-trim_x, trim_y:-trim_y]

