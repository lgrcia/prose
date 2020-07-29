from prose import Unit, blocks, io
import os
from os import path
from astropy.io import fits
from prose.console_utils import INFO_LABEL
import numpy as np


class Reduction:
    """
    Reduction unit producing a reduced FITS folder

    Parameters
    ----------
    fits_manager: prose.FitsManager
        Fits manager of the observation. Should contain a single obs

    destination: destination (
    reference
    overwrite
    n_images
    calibration
    """
    """A reduction unit producing a reduced FITS folder

    Parameters
    ----------
    fits_manager : prose.FitsManager
        Fits manager of the observation. Should contain a single obs
    destination : str, optional
        Destination of the newly created folder, by default beside the folder given to FitsManager
    reference : float, optional
        Reference image to use for alignment from 0 (first image) to 1 (last image), by default 1/2
    overwrite : bool, optional
        wether to overwrtie existing products, by default False
    n_images : int, optional
        number of images to process, by default None for all images
    calibration : bool, optional
        weather to perform calibration, by default True (if False images are still trimmed)
    """

    def __init__(
            self,
            fits_manager,
            destination=None,
            reference=1 / 2,
            overwrite=False,
            calibration=True):

        self.fits_manager = fits_manager
        self.destination = destination
        self.overwrite = overwrite
        self.calibration = calibration

        # set on prepare
        self.stack_path = None
        self.gif_path = None
        self.prepare()

        # set reference file
        reference_id = int(reference * len(self.files))
        self.reference_fits = self.files[reference_id]

    def run(self):

        reference_unit = Unit([
            blocks.Calibration(self.fits_manager.get("dark"), self.fits_manager.get("flat"), self.fits_manager.get("bias"),
                               name="calibration"),
            blocks.Trim(name="trimming"),
            blocks.SegmentedPeaks(n_stars=50, name="detection"),
            blocks.ImageBuffer(name="buffer")
        ], self.reference_fits, telescope=self.fits_manager.telescope, show_progress=False)

        reference_unit.run()

        ref_image = reference_unit.blocks_dict["buffer"].image
        calibration_block = reference_unit.blocks_dict["calibration"]

        reduction_unit = Unit([
            calibration_block if self.calibration else blocks.Pass(),
            blocks.Trim(name="trimming"),
            blocks.SegmentedPeaks(n_stars=50, name="detection"),
            blocks.XYShift(ref_image.stars_coords, name="shift"),
            blocks.Align(ref_image.data, name="alignment"),
            blocks.Gaussian2D(name="fwhm"),
            blocks.Stack(self.stack_path, header=ref_image.header, overwrite=self.overwrite, name="stack"),
            blocks.SaveReduced(self.destination, overwrite=self.overwrite, name="saving"),
            blocks.Video(self.gif_path, name="video", from_fits=True)
        ], self.files, telescope=self.fits_manager.telescope, name="Reduction")

        reduction_unit.run()

    def prepare(self):
        if len(self.fits_manager._observations) == 1:
            self.fits_manager.set_observation(
                0,
                check_calib_telescope=self.calibration,
                calibration=self.calibration,
                calibration_date_limit=0
            )

        if self.destination is None:
            self.destination = path.join(path.dirname(self.fits_manager.folder),
                                         self.fits_manager.products_denominator)
        self.stack_path = "{}{}".format(
            path.join(self.destination, self.fits_manager.products_denominator),
            "_stack.fits",
        )
        self.gif_path = "{}{}".format(
            path.join(self.destination, self.fits_manager.products_denominator),
            "_movie.gif",
        )
        if path.exists(self.stack_path) and not self.overwrite:
            raise AssertionError("stack {} already exists".format(self.stack_path))

        if not path.exists(self.destination):
            os.mkdir(self.destination)

        self.files = self.fits_manager.get("light")


class Photometry:
    """Photometric extraction unit

    Parameters
    ----------
    fits_manager : prose.FitsManager
        Fits manager of the observation. Should contain a single obs
    overwrite : bool, optional
        whether to overwrite existing products, by default False
    n_stars : int, optional
        max number of stars to take into account, by default 500
    """

    def __init__(self, fits_manager, overwrite=False, n_stars=500):

        self.fits_manager = fits_manager
        self.overwrite = overwrite
        self.n_stars = n_stars

        self.prepare()

    def run(self):

        stack_path = self.fits_manager.get("stack")[0]
        assert stack_path is not None, "No stack found"

        reference_detection_unit =  Unit([
            blocks.DAOFindStars(n_stars=self.n_stars, name="detection"),
            blocks.Moffat2D(name="fwhm"),
            blocks.ImageBuffer(name="buffer"),
            ], stack_path, telescope=self.fits_manager.telescope, show_progress=False)

        reference_detection_unit.run()
        stack_image = reference_detection_unit.blocks_dict["buffer"].image
        ref_stars = stack_image.stars_coords
        fwhm = stack_image.fwhm

        print("{} detected stars: {}".format(INFO_LABEL, len(ref_stars)))
        print("{} global psf FWHM: {:.2f} (pixels)".format(INFO_LABEL, np.mean(fwhm)))

        photometry_unit = Unit([
            blocks.Set(stars_coords=ref_stars, name="set stars"),
            blocks.Set(fwhm=fwhm, name="set fwhm"),
            blocks.SEAperturePhotometry(name="photometry"),
            blocks.SavePhots(self.phot_path, header=fits.getheader(stack_path), overwrite=self.overwrite, name="saving")
        ], self.files, telescope=self.fits_manager.telescope, name="Photometry")

        photometry_unit.run()

    def prepare(self):
        if isinstance(self.fits_manager, str):
            self.fits_manager = io.FitsManager(self.fits_manager, light_kw="reduced", verbose=False)

        self.destination = self.fits_manager.folder

        if len(self.fits_manager._observations) == 1:
            self.fits_manager.set_observation(0)
        else:
            self.fits_manager.describe("obs")

        self.phot_path = path.join(
            self.destination, "{}.phots".format(self.fits_manager.products_denominator))

        if path.exists(self.phot_path) and not self.overwrite:
            raise OSError("{}already exists".format(self.phot_path))

        self.files = self.fits_manager.get("reduced")

