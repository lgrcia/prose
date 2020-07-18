
from prose.pipeline.registration import XYShift
from prose.pipeline.alignment import Align
from prose.pipeline.detection import SegmentedPeaks, DAOFindStars
from prose.pipeline.calibration import Calibration, Trim
from prose.pipeline.psf import Gaussian2D
from prose.pipeline.base import Unit
from prose.pipeline.photometry import FixedAperturePhotometry
from prose.pipeline.imutils import Stack, SaveReduced, Gif, SavePhotometricProducts
from prose import io
import os
from os import path
import numpy as np


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
            Calibration(name="calibration") if calibration else Trim(name="calibration"),
            SegmentedPeaks(n_stars=50, name="detection"),
            XYShift(detection=SegmentedPeaks(n_stars=50), reference=reference, name="shift"),
            Align(reference=reference, name="alignment"),
            Gaussian2D(name="fwhm"),
            Stack(self.stack_path, overwrite=overwrite, name="stack"),
            SaveReduced(destination, overwrite=overwrite, name="saving"),
            Gif(self.gif_path, name="video")
        ]

        super().__init__(default_methods, "Reduction", fits_manager, files="light", show_progress=True,
                         n_images=n_images)

        if not path.exists(destination):
            os.mkdir(destination)

        self.destination = destination


class Photometry(Unit):

    def __init__(self, fits_manager, overwrite=False, n_stars=500, psf=False):

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
            DAOFindStars(n_stars=n_stars, stack=True, name="detection"),
            Gaussian2D(stack=True, name="fwhm"),
            FixedAperturePhotometry(name="photometry"),
            SavePhotometricProducts(self.phot_path, overwrite=overwrite, name="saving")
        ]

        super().__init__(default_methods, "Photometric extraction", fits_manager, files="reduced", show_progress=True)

