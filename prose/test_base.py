
from prose.pipeline.registration import XYShift
from prose.pipeline.alignment import Align
from prose.pipeline.detection import SegmentedPeaks, DAOFindStars
from prose.pipeline.calibration import Calibration
from prose.pipeline.psf import NonLinearGaussian2D
from prose.pipeline.base import Unit
from prose.pipeline.photometry import AperturePhotometry
from prose.pipeline.imutils import Stack, SaveReduced, Gif
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
            n_images=None):

        if len(fits_manager.observations) == 1:
            fits_manager.set_observation(
                0,
                check_calib_telescope=True,
                keep_closest_calibration=True,
                calibration_date_limit=0
            )

        if destination is None:
            destination = path.join(path.dirname(fits_manager.folder), fits_manager.products_denominator)

        stack_path = "{}{}".format(
            path.join(destination, fits_manager.products_denominator),
            "_stack.fits",
        )

        gif_path = "{}{}".format(
            path.join(destination, fits_manager.products_denominator),
            "_movie.gif",
        )

        if path.exists(stack_path) and not overwrite:
            if raise_exists:
                raise AssertionError("stack {} already exists".format(stack_path))

        default_methods = [
            Calibration(),
            SegmentedPeaks(n_stars=50),
            XYShift(detection=SegmentedPeaks(n_stars=50), reference=reference),
            Align(reference=reference),
            NonLinearGaussian2D(),
            Stack(stack_path, overwrite=overwrite),
            SaveReduced(destination, overwrite=overwrite),
            Gif(gif_path)
        ]

        super().__init__(
            fits_manager,
            "Reduction",
            default_methods,
            files="light",
            show_progress=True,
            n_images=n_images
        )

        if not path.exists(destination):
            os.mkdir(destination)

        self.destination = destination


class Photometry(Unit):

    def __init__(self, fits_manager, overwrite=False):
        default_methods = [
            DAOFindStars(n_stars=500, stack=True),
            NonLinearGaussian2D(),
            # AperturePhotometry(fwhm_fit),
            # SavePhotometry(overwrite=overwrite),
        ]

        super().__init__(
            fits_manager,
            "Photometric extraction",
            default_methods,
            files="reduced",
            show_progress=True,
        )


# fm = io.FitsManager("/Users/lionelgarcia/Data/test/20190922_cambridge_server", depth=3, index=True)
# fm.set_observation(0)
# r = Reduction(fm, overwrite=True, n_images=10)
# r.run()


fm = io.FitsManager("/Users/lionelgarcia/Data/test/Io_20190922_Sp0111-4908_I+z", depth=1, index=True)
p = Photometry(fm)
p.run()





