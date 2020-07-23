from prose import Unit
from prose import blocks
from prose import io
import os
from os import path


class Reduction(Unit):

    def __init__(
            self,
            fits_manager,
            destination=None,
            reference=1/2,
            overwrite=False,
            raise_exists=True,
            n_images=None,
            calibration=True):

        if len(fits_manager.observations) == 1:
            fits_manager.set_observation(
                0,
                check_calib_telescope=calibration,
                keep_closest_calibration=calibration,
                calibration_date_limit=0
            )

        if destination is None:
            destination = path.join(path.dirname(fits_manager.folder), fits_manager.products_denominator)

        self.stack_path = "{}{}".format(
            path.join(destination, fits_manager.products_denominator),
            "_stack.fits",
        )

        self.gif_path = "{}{}".format(
            path.join(destination, fits_manager.products_denominator),
            "_movie.gif",
        )

        if path.exists(self.stack_path) and not overwrite:
            if raise_exists:
                raise AssertionError("stack {} already exists".format(self.stack_path))

        default_methods = [
            blocks.Calibration(name="calibration") if calibration else blocks.Trim(name="calibration"),
            blocks.SegmentedPeaks(n_stars=50, name="detection"),
            blocks.XYShift(detection=blocks.SegmentedPeaks(n_stars=50), reference=reference, name="shift"),
            blocks.Align(reference=reference, name="alignment"),
            blocks.Gaussian2D(name="fwhm"),
            blocks.Stack(self.stack_path, overwrite=overwrite, name="stack"),
            blocks.SaveReduced(destination, overwrite=overwrite, name="saving"),
            blocks.Video(self.gif_path, name="video", from_fits=True)
        ]

        super().__init__(default_methods, fits_manager, "Reduction", files="light", show_progress=True,
                         n_images=n_images)

        if not path.exists(destination):
            os.mkdir(destination)

        self.destination = destination


class Photometry(Unit):

    def __init__(self, fits_manager, overwrite=False, n_stars=500, psf=False, n_images=None):

        if isinstance(fits_manager, str):
            fits_manager = io.FitsManager(fits_manager, light_kw="reduced", verbose=False)

        self.destination = fits_manager.folder

        if len(fits_manager.observations) == 1:
            fits_manager.set_observation(0)
        else:
            fits_manager.describe("obs")

        self.phot_path = path.join(
            self.destination, "{}.phots".format(fits_manager.products_denominator))

        if path.exists(self.phot_path) and not overwrite:
            raise OSError("{}already exists".format(self.phot_path))

        default_methods = [
            blocks.DAOFindStars(n_stars=n_stars, stack=True, name="detection"),
            blocks.Gaussian2D(stack=True, name="fwhm"),
            blocks.ForcedAperturePhotometry(name="photometry"),
            blocks.SavePhotometricProducts(self.phot_path, overwrite=overwrite, name="saving")
        ]

        super().__init__(
            default_methods,
            fits_manager,
            "Photometric extraction",
            files="reduced",
            show_progress=True,
            n_images=n_images)
