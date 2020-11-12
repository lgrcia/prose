from prose import Unit, blocks, io, Block, Telescope
from prose.blocks import io as bio
import os
from os import path
from astropy.io import fits
from prose.console_utils import INFO_LABEL
import numpy as np
import time


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

        ref_image = reference_unit.buffer.image
        calibration_block = reference_unit.calibration

        reduction_unit = Unit([
            blocks.Pass() if not self.calibration else calibration_block,
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
        """
        This will prepare the `self.destination` containing the:

        - ``self.stack_path``
        - ``self.gif_path``

        Returns
        -------

        """
        # if len(self.fits_manager._observations) == 1:
        #     self.fits_manager.set_observation(
        #         0,
        #         check_calib_telescope=self.calibration,
        #         calibration=self.calibration,
        #         calibration_date_limit=0
        #     )

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
            raise AssertionError("stack {} already exists, consider using the 'overwrite' kwargs".format(self.stack_path))

        if not path.exists(self.destination):
            os.mkdir(self.destination)

        self.files = self.fits_manager.images


class AperturePhotometry:
    """Aperture Photometry unit

    Parameters
    ----------
    fits_manager : prose.FitsManager
         FitsManager of the observation. Should contain a single obs. One of ``fits_manager`` or ``files`` and ``stack` should  be provided
    files : list of str, optional
        List of files to process. One of ``fits_manager`` or ``files`` and ``stack` should  be provided
    stack: str, optional
        Path of the stack image. Should be specified if ``files`` is specified.
    overwrite : bool, optional
        whether to overwrite existing products, by default False
    n_stars : int, optional
        max number of stars to take into account, by default 500
    apertures : list or np.ndarray, optional
        Apertures radii to be used. If None, by default np.arange(0.1, 10, 0.25)
    r_in : int, optional
        Radius of the inner annulus to be used in pixels, by default 5
    r_out : int, optional
        Radius of the outer annulus to be used in pixels, by default 8
    fwhm_scale : bool, optional
        wheater to multiply `apertures`, `r_in` and `r_out` by the global fwhm, by default True
    sigclip : float, optional
        Sigma clipping factor used in the annulus, by default 3. No effect if `method="sextrcator"`
    method : str, optional
        Method to bue used ("photutils" or "sextractor"), by default "photutils"
    centroid : prose.Block, optional
        Centroid block to be used, by default None
    """

    def __init__(self,
                 fits_manager=None,
                 files=None,
                 stack=None,
                 overwrite=False,
                 n_stars=500,
                 apertures=None,
                 r_in=5,
                 r_out=8,
                 fwhm_scale=True,
                 sigclip=3.,
                 psf=blocks.Gaussian2D,
                 photometry=blocks.PhotutilsAperturePhotometry,
                 centroid=None):

        if apertures is None:
            apertures = np.arange(0.1, 10, 0.25)

        self.fits_manager = fits_manager
        self.overwrite = overwrite
        self.n_stars = n_stars
        self.reference_detection_unit = None
        self.photometry_unit = None
        self.destination = None

        # preparing inputs and outputs
        self.destination = None
        self.stack_path = None
        self.phot_path = None
        self.files = None
        self.telescope = None
        self.prepare(fits_manager=fits_manager, files=files, stack=stack)

        # Blocks
        assert centroid is None or issubclass(centroid, Block), "centroid must be a subclass of Block"
        self.centroid = centroid
        # ==
        assert psf is None or issubclass(psf, Block), "psf must be a subclass of Block"
        self.psf = psf
        # ==
        assert photometry is None or issubclass(photometry, Block), "photometry must be a subclass of Block"
        self.photometry = photometry(
            apertures=apertures,
            r_in=r_in,
            r_out=r_out,
            sigclip=sigclip,
            fwhm_scale=fwhm_scale,
            name="photometry"
        )

    def run_reference_detection(self):
        self.reference_detection_unit = Unit([
            blocks.DAOFindStars(n_stars=self.n_stars, name="detection"),
            self.psf(name="fwhm"),
            blocks.ImageBuffer(name="buffer"),
        ], self.stack_path, telescope=self.fits_manager.telescope, show_progress=False)

        self.reference_detection_unit.run()
        stack_image = self.reference_detection_unit.buffer.image
        ref_stars = stack_image.stars_coords
        fwhm = stack_image.fwhm

        print("{} detected stars: {}".format(INFO_LABEL, len(ref_stars)))
        print("{} global psf FWHM: {:.2f} (pixels)".format(INFO_LABEL, np.mean(fwhm)))

        time.sleep(0.5)

        self.photometry_unit = Unit([
            blocks.Set(stars_coords=ref_stars, name="set stars"),
            blocks.Set(fwhm=fwhm, name="set fwhm"),
            blocks.Pass() if not isinstance(self.centroid, Block) else self.centroid,
            self.photometry,
            bio.SavePhot(
                self.phot_path,
                header=fits.getheader(self.stack_path),
                stack=fits.getdata(self.stack_path),
                name="saving")
        ], self.files, telescope=self.telescope, name="Photometry")

    def run(self, destination=None):
        if self.phot_path is None:
            assert destination is not None, "You must provide a destination"
        if destination is not None:
            self.phot_path = destination.replace(".phot", "") + ".phot"

        self._check_phot_path()

        self.run_reference_detection()
        self.photometry_unit.run()

    def _check_phot_path(self):
        if path.exists(self.phot_path) and not self.overwrite:
            raise OSError("{} already exists".format(self.phot_path))

    def prepare(self, fits_manager=None, files=None, stack=None):
        """
        Check that stack and observation is present and set ``self.phot_path``

        """
        if fits_manager is not None:
            if isinstance(self.fits_manager, str):
                self.fits_manager = io.FitsManager(self.fits_manager, image_kw="reduced", verbose=False)
            elif isinstance(self.fits_manager, io.FitsManager):
                if self.fits_manager.image_kw != "reduced":
                    print(f"Warning: image keyword is '{self.fits_manager.image_kw}'")

            self.destination = self.fits_manager.folder

            if len(self.fits_manager._observations) == 1:
                self.fits_manager.set_observation(0)
            else:
                self.fits_manager.describe("obs")

            self.phot_path = path.join(
                self.destination, "{}.phot".format(self.fits_manager.products_denominator))

            self.files = self.fits_manager.images
            self.stack_path = self.fits_manager.stack

            self.telescope = self.fits_manager.telescope

            self._check_phot_path()

        elif files is not None:
            assert stack is not None, "'stack' should be specified is 'files' is specified"

            self.stack_path = stack
            self.files = files
            self.telescope = Telescope(self.stack_path)
