from prose.pipeline.base import Block
from astropy.io import fits
import numpy as np
from astropy.time import Time
from os import path
import imageio
import prose.visualisation as viz


class Stack(Block):

    def __init__(self, destination, reference=1/2, overwrite=False):
        super(Stack, self).__init__()
        self.reference = reference
        self.stack = None
        self.stack_header = None
        self.destination = destination
        self.fits_manager = None
        self.overwrite = overwrite

    def initialize(self, fits_manager):
        self.fits_manager = fits_manager
        reference_frame = int(self.reference * len(fits_manager.files))
        reference_image_path = fits_manager.files[reference_frame]
        self.stack_header = fits.getheader(reference_image_path)

    def run(self, image, *args):
        if self.stack is None:
            self.stack = image.data

        else:
            self.stack += image.data

    def terminate(self):

        self.stack /= len(self.fits_manager.files)
        stack_hdu = fits.PrimaryHDU(self.stack)
        self.stack_header[self.fits_manager.telescope.keyword_image_type] = "Stack image"
        self.stack_header["REDDATE"] = Time.now().to_value("fits")
        self.stack_header["NIMAGES"] = len(self.fits_manager.files)

        changing_flip_idxs = np.array([
            idx for idx, (i, j) in enumerate(zip(self.fits_manager.files_df["flip"],
                                                 self.fits_manager.files_df["flip"][1:]), 1) if i != j])

        if len(changing_flip_idxs) > 0:
            self.stack_header["FLIPTIME"] = self.fits_manager.files_df["jd"].iloc[changing_flip_idxs].values[0]

        stack_hdu.header = self.stack_header
        stack_hdu.writeto(self.destination, overwrite=self.overwrite)


class SaveReduced(Block):
    def __init__(self, destination, overwrite=False):
        super().__init__()
        self.destination = destination
        self.overwrite = overwrite
        self.telescope = None

    def initialize(self, fits_manager):
        self.telescope = fits_manager.telescope

    def run(self, image, *args):

        new_hdu = fits.PrimaryHDU(image.data)
        new_hdu.header = image.header

        image.header["SEEING"] = new_hdu.header.get(self.telescope.keyword_seeing, "")
        image.header["BZERO"] = 0
        image.header["REDDATE"] = Time.now().to_value("fits")
        image.header[self.telescope.keyword_image_type] = "reduced"

        fits_new_path = path.join(
            self.destination,
            path.splitext(path.basename(image.path))[0] + "_reduced.fits"
        )

        new_hdu.writeto(fits_new_path, overwrite=self.overwrite)


class Gif(Block):
    def __init__(self, destination, overwrite=True):
        super().__init__()
        self.destination = destination
        self.overwrite = overwrite
        self.images = []

    def run(self, image, *args):
        self.images.append(viz.gif_image_array(image.data))

    def terminate(self):
        imageio.mimsave(self.destination, self.images)


