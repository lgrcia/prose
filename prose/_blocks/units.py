from prose import Unit
from prose import blocks
from prose import io
import os
from os import path


class Reduction(Unit):
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
            reference=1/2,
            overwrite=False,
            n_images=None,
            calibration=True):


        self.fits_manager = fits_manager
        self.destination = destination
        self.overwrite = overwrite
        self.calibration = calibration

        self.prepare()

        default_methods = [
            blocks.Calibration(name="calibration") if calibration else blocks.Trim(name="calibration"),
            blocks.SegmentedPeaks(n_stars=50, name="detection"),
            blocks.XYShift(detection=blocks.SegmentedPeaks(n_stars=50), reference=reference, name="shift"),
            blocks.Align(reference=reference, name="alignment"),
            blocks.Gaussian2D(name="fwhm"),
            blocks.Stack(self.stack_path, overwrite=overwrite, name="stack"),
            blocks.SaveReduced(self.destination, overwrite=overwrite, name="saving"),
            blocks.Video(self.gif_path, name="video", from_fits=True)
        ]

        super().__init__(default_methods, self.fits_manager, "Reduction", files="light", show_progress=True,
                         n_images=n_images)

    def prepare(self):
        if len(self.fits_manager.observations) == 1:
            self.fits_manager.set_observation(
                0,
                check_calib_telescope=self.calibration,
                keep_closest_calibration=self.calibration,
                calibration_date_limit=0
            )

        if self.destination is None:
            self.destination = path.join(path.dirname(self.fits_manager.folder), self.fits_manager.products_denominator)

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


class Photometry(Unit):
    """Photometric extraction unit

    Parameters
    ----------
    fits_manager : prose.FitsManager
        Fits manager of the observation. Should contain a single obs
    overwrite : bool, optional
        wether to overwrtie existing products, by default False
    n_stars : int, optional
        max number of stars to take into account, by default 500
    psf : bool, optional
        weather to perform psf photomery, aperture otherwise, by default False
    n_images : int, optional
        number of images to process, by default None for all images
    """

    def __init__(self, fits_manager, overwrite=False, n_stars=500, psf=False, n_images=None):

        self.fits_manager = fits_manager
        self.overwrite = overwrite

        self.prepare()

        default_methods = [
            blocks.DAOFindStars(n_stars=n_stars, stack=True, name="detection"),
            blocks.Gaussian2D(stack=True, name="fwhm"),
            blocks.ForcedAperturePhotometry(name="photometry"),
            blocks.SavePhotometricProducts(self.phot_path, overwrite=overwrite, name="saving")
        ]

        super().__init__(
            default_methods,
            self.fits_manager,
            "Photometric extraction",
            files="reduced",
            show_progress=True,
            n_images=n_images)

    def prepare(self):
        if isinstance(self.fits_manager, str):
            self.fits_manager = io.FitsManager(self.fits_manager, light_kw="reduced", verbose=False)

        self.destination = self.fits_manager.folder

        if len(self.fits_manager.observations) == 1:
            self.fits_manager.set_observation(0)
        else:
            self.fits_manager.describe("obs")

        self.phot_path = path.join(
            self.destination, "{}.phots".format(self.fits_manager.products_denominator))

        if path.exists(self.phot_path) and not self.overwrite:
            raise OSError("{}already exists".format(self.phot_path))

