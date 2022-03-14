from prose import  Block
from prose.blocks.psf import *
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground


class PhotutilsBackground2D(Block):

    def __init__(self, subtract=True, **kwargs):
        super().__init__(**kwargs)
        self.sigma_clip = SigmaClip(sigma=3.)
        self.bkg_estimator = MedianBackground()
        self.subtract = subtract

    def run(self, image):
        sigma_clip = SigmaClip(sigma=3.)
        self.bkg = Background2D(
            image.data, (50, 50),
            filter_size=(3, 3),
            sigma_clip=sigma_clip, 
            bkg_estimator=self.bkg_estimator
        ).background
        if self.subtract:
             image.data = image.data - self.bkg