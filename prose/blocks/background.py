from prose import  Block
from prose.blocks.psf import *
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground


class PhotutilsBackground2D(Block):

    def __init__(self, subtract=True, name=None, box_size=(50,50)):
        super().__init__(name=None)
        self.sigma_clip = SigmaClip(sigma=3.)
        self.bkg_estimator = MedianBackground()
        self.subtract = subtract
        self.box_size = box_size

    def run(self, image):
        sigma_clip = SigmaClip(sigma=3.)
        self.bkg = Background2D(
            image.data, box_size=self.box_size,
            filter_size=(3, 3),
            sigma_clip=sigma_clip, 
            bkg_estimator=self.bkg_estimator
        ).background
        if self.subtract:
            image.bkg = self.bkg
            image.data = image.data - self.bkg