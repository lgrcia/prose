from prose._blocks.base import Block
from astropy.io import fits
import numpy as np
from astropy.time import Time
from os import path
import imageio
import prose.visualisation as viz
from astropy.table import Table
import pandas as pd
from prose import utils
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from prose import io


class Stack(Block):
    """Build a FITS stack image of the observation

    Parameters
    ----------
    destination : str
        path of the stack image (must be a .fits file name)
    header : dict, optional
        header base of the stack image to be saved, default is None for fresh header
    overwrite : bool, optional
        weather to overwrite file if exists, by default False
    """

    def __init__(self, destination, header=None, overwrite=False, **kwargs):

        super(Stack, self).__init__(**kwargs)
        self.stack = None
        self.n_images = 0
        self.stack_header = header
        self.destination = destination
        self.fits_manager = None
        self.overwrite = overwrite

        self.reference_image_path = None

    def run(self, image):
        if self.stack is None:
            self.stack = image.data

        else:
            self.stack += image.data

        self.n_images += 1

    def terminate(self):

        self.stack_header[self.telescope.keyword_image_type] = "Stack image"
        self.stack_header["BZERO"] = 0
        self.stack_header["REDDATE"] = Time.now().to_value("fits")
        self.stack_header["NIMAGES"] = self.n_images

        # changing_flip_idxs = np.array([
        #     idx for idx, (i, j) in enumerate(zip(self.fits_manager.files_df["flip"],
        #                                          self.fits_manager.files_df["flip"][1:]), 1) if i != j])
        #
        # if len(changing_flip_idxs) > 0:
        #     self.stack_header["FLIPTIME"] = self.fits_manager.files_df["jd"].iloc[changing_flip_idxs].values[0]

        stack_hdu = fits.PrimaryHDU(self.stack, header=self.stack_header)
        stack_hdu.writeto(self.destination, overwrite=self.overwrite)


class StackStd(Block):

    def __init__(self, destination, overwrite=False):
        super(StackStd, self).__init__()
        self.images = []
        self.stack_header = None
        self.destination = destination
        self.fits_manager = None
        self.overwrite = overwrite
        self.stack_std = None

    def initialize(self, fits_manager):
        self.fits_manager = fits_manager
        if fits_manager.has_stack():
            self.stack_header = fits.getheader(fits_manager.get("stack")[0])

    def run(self, image, **kwargs):
        self.images.append(image.data)

    def terminate(self):
        self.images = np.array(self.images)
        # shape_divisors = utils.divisors(self.images[0].shape[1])
        # n = shape_divisors[np.argmin(np.abs(50 - shape_divisors))]
        self.stack_std = np.std(self.images, axis=0) #concatenate([np.std(im, axis=0) for im in np.split(self.images, n, axis=1)])
        stack_hdu = fits.PrimaryHDU(self.stack_std, header=self.stack_header)
        stack_hdu.header["IMTYPE"] = "std"
        stack_hdu.writeto(self.destination, overwrite=self.overwrite)


class SaveReduced(Block):
    """Save reduced FITS images.

    Parameters
    ----------
    destination : str
        folder path of the images. Orignial name is used with the addition of :code:`_reduced.fits`
    overwrite : bool, optional
        weather to overwrite file if exists, by default False
    """
    def __init__(self, destination, overwrite=False, **kwargs):

        super().__init__(**kwargs)
        self.destination = destination
        self.overwrite = overwrite

    def run(self, image, **kwargs):

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

# TODO: make ImageIOBlock block

class Video(Block):
    """Build a video of all :code:`Image.data`.

    Can be either from raw image or a :code:`int8` rgb image.

    Parameters
    ----------
    destination : str
        path of the video which format depends on the extension (e.g. :code:`.mp4`, or :code:`.gif)
    overwrite : bool, optional
        weather to overwrite file if exists, by default False
    factor : float, optional
        subsampling factor of the image, by default 0.25
    fps : int, optional
        frames per second of the video, by default 10
    from_fits : bool, optional
        Wether :code:`Image.data` is a raw fits image, by default False. If True, a z scaling is applied as well as casting to `uint8`
    """

    def __init__(self, destination, overwrite=True, factor=0.25, fps=10, from_fits=False, **kwargs):

        super().__init__(**kwargs)
        self.destination = destination
        self.overwrite = overwrite
        self.images = []
        self.factor = factor
        self.fps = fps
        self.from_fits = from_fits

    def run(self, image, **kwargs):
        if self.from_fits:
            self.images.append(viz.gif_image_array(image.data, factor=self.factor))
        else:
            self.images.append(image.data.copy())

    def terminate(self):
        imageio.mimsave(self.destination, self.images, fps=self.fps)

    def citations(self):
        return "imageio"



class SavePhots(Block):
    """Save photometric products into a FITS :code:`.phots` file. See :ref:`phots-structure` for more info

    Parameters
    ----------
    destination : str
        path of the file (must be a .phots file name)
    overwrite : bool, optional
        weather to overwrite file if exists, by default False
    """
    def __init__(self, destination, overwrite=False, header=None, **kwargs):
        super().__init__(**kwargs)
        self.destination = destination
        self.overwrite = overwrite
        self.telescope = None
        self.images = []
        self.stack_path = None
        self.telescope = None
        self.fits_manager = None
        self.header = header

    def run(self, image, **kwargs):
        self.images.append(image)

    def terminate(self):
        if self.header is not None:
            self.header["REDDATE"] = Time.now().to_value("fits")

        fluxes = np.array([im.fluxes for im in self.images])
        fluxes_errors = np.array([im.fluxes_errors for im in self.images])
        stars = self.images[0].stars_coords
        sky = np.array([im.sky for im in self.images])

        # backward compatibility
        fluxes = np.moveaxis(fluxes, 1, 0)
        fluxes_errors = np.moveaxis(fluxes_errors, 1, 0)
        fluxes = np.moveaxis(fluxes, 2, 1)
        fluxes_errors = np.moveaxis(fluxes_errors, 2, 1)

        if len(fluxes.shape) == 2:
            fluxes = np.array([fluxes])
            fluxes_errors = np.array([fluxes_errors])

        data = {}

        for keyword in [
            "sky",
            "fwhm",
            "fwhmx",
            "fwhmy",
            "psf_angle",
            "dx",
            "dy",
            "airmass",
            self.telescope.keyword_exposure_time,
            self.telescope.keyword_julian_date,
            self.telescope.keyword_seeing,
            self.telescope.keyword_ra,
            self.telescope.keyword_dec,
        ]:
            _data = []
            if keyword in self.images[0].header:
                for image in self.images:
                    _data.append(image.header[keyword])

                data[keyword.lower()] = _data

        hdu_list = [
            fits.PrimaryHDU(header=self.header),
            fits.ImageHDU(fluxes, name="photometry"),
            fits.ImageHDU(fluxes_errors, name="photometry errors"),
            fits.ImageHDU(stars, name="stars")
        ]

        data_table = Table.from_pandas(pd.DataFrame(data))
        hdu_list.append(fits.BinTableHDU(data_table, name="time series"))

        # These are other data produced by the photometry task wished to be saved in the .phot
        for key in [
            "apertures_area",
            "annulus_area"
        ]:
            if key in self.images[0].__dict__:
                _data = []
                for image in self.images:
                    _data.append(image.__dict__[key])

                hdu_list.append(fits.ImageHDU(np.array(_data), name=key.replace("_", " ")))

        hdu = fits.HDUList(hdu_list)
        hdu.writeto(self.destination, overwrite=self.overwrite)


from astropy.stats import sigma_clipped_stats


class RemoveBackground(Block):

    def __init__(self):
        super().__init__()
        self.stack_data = None

    def run(self, image, **kwargs):
        _, im_median, _ = sigma_clipped_stats(image.data, sigma=3.0)
        image.data = im_median


class CleanCosmics(Block):

    def __init__(self, threshold=2):
        super().__init__()
        self.stack_data = None
        self.threshold = threshold
        self.sigma_clip = SigmaClip(sigma=3.)
        self.bkg_estimator = MedianBackground()

    def initialize(self, fits_manager):
        if fits_manager.has_stack():
            self.stack_data = fits.getdata(fits_manager.get("stack")[0])
        self.std_stack = fits.getdata(path.join(fits_manager.folder, "test_std.fits"))

    def run(self, image, **kwargs):
        mask = image.data > (self.stack_data + self.std_stack * self.threshold)
        image.data[mask] = self.stack_data[mask]


class Pass(Block):
    """A Block that does nothing"""
    def run(self, image):
        pass


class ImageBuffer(Block):
    """Store the last Image
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image = None

    def run(self, image):
        self.image = image


class Set(Block):
    """Set specific attribute to every image

    For example to set attributes ``a`` with the value 2 on every image (i.e Image.a = 2):
    
    .. code-block:: python

        from prose import blocks

        set_block = blocks.Set(a=2)

    Parameters
    ----------
    kwargs : kwargs
        keywords argument and values to be set on every image
    """
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self.kwargs = kwargs

    def run(self, image):
        image.__dict__.update(self.kwargs)
