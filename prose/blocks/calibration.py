import numpy as np
from ..core import Block
from astropy.io import fits
from .. import utils
import matplotlib.pyplot as plt
from .. import visualisation as viz
from astropy.nddata import Cutout2D
from ..console_utils import info
from time import sleep


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
    dark : list
        list of dark files paths
    flat : list
        list of flat files paths
    bias : list
        list of bias files paths
    """
    def __init__(self, dark=None, flat=None, bias=None, **kwargs):

        super().__init__(**kwargs)
        if dark is None:
            dark = []
        if flat is None:
            flat = []
        if bias is None:
            bias = []
        self.images = {"dark": dark, "flat": flat, "bias": bias}

        self.master_dark = None
        self.master_flat = None
        self.master_bias = None

    def calibration(self, image, exp_time):
        return (image - (self.master_dark * exp_time + self.master_bias)) / self.master_flat

    def _produce_master(self, image_type):
        _master = []
        kw_exp_time = self.telescope.keyword_exposure_time
        images = self.images[image_type]
        if len(images) == 0:
            info(f"No {image_type} images set")
            if image_type == "dark":
                self.master_dark = 0
            elif image_type == "bias":
                self.master_bias = 0
            elif image_type == "flat":
                self.master_flat = 1

        for i, fits_path in enumerate(images):
            hdu = fits.open(fits_path)
            primary_hdu = hdu[0]
            image, header = primary_hdu.data, primary_hdu.header
            image = image.astype("float64")
            hdu.close()
            if image_type == "dark":
                _dark = (image - self.master_bias) / header[kw_exp_time]
                _master.append(_dark)
            elif image_type == "bias":
                _master.append(image)
            elif image_type == "flat":
                _flat = image - self.master_bias - self.master_dark*header[kw_exp_time]
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
        assert self.telescope is not None, "Calibration block needs telescope to be set (in Unit)"
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

    def run(self, image):
        # TODO: Investigate flip
        data = image.data
        header = image.header
        exp_time = header[self.telescope.keyword_exposure_time]
        calibrated_image = self.calibration(data, exp_time)
        calibrated_image[calibrated_image < 0] = 0.

        # if flip:
        #     calibrated_image = calibrated_image[::-1, ::-1]

        image.data = calibrated_image

    def citations(self):
        return "astropy", "numpy"


class Trim(Block):
    """Images trimming
    """

    def __init__(self, skip_wcs=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_wcs = skip_wcs

    def run(self, image, **kwargs):
        shape = np.array(image.data.shape)
        center = shape[::-1] / 2
        dimension = shape - 2 * np.array(self.telescope.trimming[::-1])
        trim_image = Cutout2D(image.data, center, dimension, wcs=None if self.skip_wcs else image.wcs)
        image.data = trim_image.data
        if not self.skip_wcs:
            image.header.update(trim_image.wcs.to_header())
