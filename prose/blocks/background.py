import numpy as np
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

from prose import Block
from prose.blocks.psf import *
from prose.utils import binn2D


class PhotutilsBackground2D(Block):
    def __init__(self, subtract=True, box_size=(50, 50), filter_size=(3, 3), name=None):
        """
        Initializes the PhotutilsBackground2D block.

        |read| :code:`Image.data`

        |write| :code:`Image.data`, :code:`Image.bkg`

        Parameters
        ----------
        subtract : bool, optional
            Whether to subtract the estimated background from the image. Default is True.
        box_size : tuple of int, optional
            The size of the box used to compute the background. Default is (50, 50).
        filter_size : tuple of int, optional
            The size of the filter used to smooth the background. Default is (3, 3).
        """
        super().__init__(name=name)
        self.sigma_clip = SigmaClip(sigma=3.0)
        self.bkg_estimator = MedianBackground()
        self.subtract = subtract
        self.box_size = box_size
        self.filter_size = filter_size

    def run(self, image):
        sigma_clip = SigmaClip(sigma=3.0)
        self.bkg = Background2D(
            image.data,
            box_size=self.box_size,
            filter_size=self.filter_size,
            sigma_clip=sigma_clip,
            bkg_estimator=self.bkg_estimator,
        ).background
        if self.subtract:
            image.bkg = self.bkg
            image.data = image.data - self.bkg

    @property
    def citations(self) -> list:
        return super().citations + ["photutils"]


class BackgroundPoly(Block):
    """[EXPERIMENTAL] Linear fit of the background with polynomials
    Notes:
    - sigma clipped and binned image (to remain fast)
    - order > 2 fail
    - image shape must be factor if `binning`
    """

    def __init__(self, ref=None, order=2, iterations=2, sigclip=2, binning=4, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        if ref is not None:
            self.X = self.design_matrix(ref.shape)
        else:
            self.X = None
        self.iterations = iterations
        self.sigclip = sigclip
        self.binning = binning

    def design_matrix(self, shape):
        x, y = np.indices(shape)
        X = np.polynomial.polynomial.polyvander2d(
            x.flatten(), y.flatten(), (self.order, self.order)
        )
        X[:, 1:] -= X.mean(0)[1:]
        X[:, 1:] -= X.std(0)[1:]
        return X

    def run(self, image):
        # First sigma clipping and binning
        data = image.data.copy()
        mask = (data - np.mean(data)) < self.sigclip * np.std(data)
        data[~mask] = np.median(data[mask])
        bin_data = binn2D(data, self.binning)
        mask = np.ones_like(bin_data).astype(bool).flatten()

        if self.X is None:
            self.X = self.design_matrix(image.shape)
        elif self.X.shape[0] != np.product(image.shape):
            self.X = self.design_matrix(image.shape)

        bin_X = np.array(
            [
                binn2D(np.reshape(x, image.shape), self.binning).flatten()
                for x in self.X.T
            ]
        ).T

        for _ in range(self.iterations):
            masked_data = bin_data.flatten()[mask]
            w = np.linalg.lstsq(bin_X[mask, :], masked_data, rcond=False)[0]
            res = bin_data.flatten() - bin_X @ w
            mask *= res < self.sigclip * np.std(res[mask])

        image.bkg = np.reshape(self.X @ w, image.shape)

    @property
    def citations(self) -> list:
        return super().citations + ["numpy", "astropy"]
