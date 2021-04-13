from .. import Sequence, blocks, io, Block, Telescope, viz
from os import path
from astropy.io import fits
from ..console_utils import info
from .. import utils
import matplotlib.pyplot as plt
import numpy as np
import time


class Photometry:
    """Base unit for Photometry

    Parameters
    ----------
    fits_manager : prose.FitsManager
         FitsManager of the observation. Should contain a single obs. One of ``fits_manager`` or ``files`` and ``stack` should  be provided
    files : list of str, optional
        List of files to process. One of ``fits_manager`` or ``files`` and ``stack`` should  be provided
    stack: str, optional
        Path of the stack image. Should be specified if ``files`` is specified.
    overwrite : bool, optional
        whether to overwrite existing products, by default False
    n_stars : int, optional
        max number of stars to take into account, by default 500
    ignore_telescope: bool, optional
        whether to load a default telescope if telescope not recognised, by default False
    """

    def __init__(self,
                 fits_manager=None,
                 files=None,
                 stack=None,
                 overwrite=False,
                 n_stars=500,
                 psf=blocks.Gaussian2D,
                 ignore_telescope=False,
                 show=False,
                 verbose=True):

        self.fits_manager = fits_manager
        self.overwrite = overwrite
        self.n_stars = n_stars
        self.reference_detection_sequence = None
        self.photometry_sequence = None
        self.destination = None
        self.verbose = verbose

        # preparing inputs and outputs
        self.destination = None
        self.stack_path = None
        self.phot_path = None
        self.files = None
        self.telescope = None
        self.prepare(fits_manager=fits_manager, files=files, stack=stack)

        if not ignore_telescope:
            assert self.fits_manager.telescope.name != "Unknown", \
                "Telescope has not been recognised, to load a default one set ignore_telescope=True (kwargs)"

        assert psf is None or issubclass(psf, Block), "psf must be a subclass of Block"
        self.psf = psf
        self.show = show

    def run_reference_detection(self):
        self.reference_detection_sequence = Sequence([
            blocks.DAOFindStars(n_stars=self.n_stars, name="detection"),
            self.psf(name="fwhm"),
            blocks.ImageBuffer(name="buffer"),
        ], self.stack_path, telescope=self.fits_manager.telescope, show_progress=False)

        self.reference_detection_sequence.run(show_progress=False)
        stack_image = self.reference_detection_sequence.buffer.image
        ref_stars = stack_image.stars_coords
        fwhm = stack_image.fwhm

        info(f"detected stars: {len(ref_stars)}")
        info(f"global psf FWHM: {np.mean(fwhm):.2f} (pixels)")

        time.sleep(0.5)

        return stack_image, ref_stars, fwhm

    def run(self, destination=None):
        if self.phot_path is None:
            assert destination is not None, "You must provide a destination"
        if destination is not None:
            self.phot_path = destination.replace(".phot", "") + ".phot"

        self._check_phot_path()
        self.run_reference_detection()
        self.photometry_sequence.run(show_progress=self.verbose)

    def _check_phot_path(self):
        if path.exists(self.phot_path) and not self.overwrite:
            raise OSError("{} already exists".format(self.phot_path))

    def prepare(self, fits_manager=None, files=None, stack=None):
        """
        Check that stack and observation is present and set ``self.phot_path``

        """
        if self.fits_manager is not None:
            if isinstance(self.fits_manager, str):
                self.fits_manager = io.FitsManager(self.fits_manager, image_kw="reduced", verbose=False)
            elif isinstance(self.fits_manager, io.FitsManager):
                if self.fits_manager.image_kw != "reduced":
                    print(f"Warning: image keyword is '{self.fits_manager.image_kw}'")

            self.destination = self.fits_manager.folder

            if not self.fits_manager.unique_obs:
                _ = self.fits_manager.observations
                raise AssertionError("Multiple observations found")
            else:
                self.fits_manager.set_observation(0, future=100000)

            self.phot_path = path.join(
                self.destination, "{}.phot".format(self.fits_manager.products_denominator))

            self.files = self.fits_manager.images
            self.stack_path = self.fits_manager.stack

            self.telescope = self.fits_manager.telescope

        elif files is not None:
            assert stack is not None, "'stack' should be specified if 'files' is specified"

            self.stack_path = stack
            self.files = files
            self.telescope = Telescope(self.stack_path)

    def __repr__(self):
        return f"{self.reference_detection_sequence}\n{self.photometry_sequence}"

    @property
    def processing_time(self):
        return self.reference_detection_sequence.processing_time + self.photometry_sequence.processing_time


class AperturePhotometry(Photometry):
    """Aperture Photometry unit

    Parameters
    ----------
    fits_manager : prose.FitsManager
         FitsManager of the observation. Should contain a single obs. One of ``fits_manager`` or ``files`` and ``stack` should  be provided
    files : list of str, optional
        List of files to process. One of ``fits_manager`` or ``files`` and ``stack`` should  be provided
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
        wheater to multiply ``apertures``, ``r_in`` and ``r_out`` by the global fwhm, by default True
    sigclip : float, optional
        Sigma clipping factor used in the annulus, by default 3. No effect if :class:`~prose.blocks.SEAperturePhotometry` is used
    psf : Block, optional
        PSF modeling Block (mainly used to estimate fwhm and scale aperture if ``fwhm_scale`` is ``True``), by default :class:`~prose.blocks.Gaussian2D`
    photometry : Block, optional
        aperture photometry Block, by default :class:`~prose.blocks.PhotutilsAperturePhotometry`
    centroid : Block, optional
        centroid computing Block, by default None to keep centroid fixed
    ignore_telescope: bool, optional
        whether to load a default telescope if telescope not recognised, by default False
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
                 centroid=None,
                 show=False,
                 ignore_telescope=False,
                 verbose=True):

        if apertures is None:
            apertures = np.arange(0.1, 10, 0.25)

        super().__init__(
            fits_manager=fits_manager,
            files=files,
            stack=stack,
            overwrite=overwrite,
            n_stars=n_stars,
            psf=psf,
            show=show,
            ignore_telescope=ignore_telescope,
            verbose=verbose
        )

        # Blocks
        assert centroid is None or issubclass(centroid, Block), "centroid must be a subclass of Block"
        if centroid is None:
            self.centroid = None
        else:
            self.centroid = centroid()
        # ==
        assert photometry is None or issubclass(photometry, Block), "photometry must be a subclass of Block"
        self.photometry = photometry(
            apertures=apertures,
            r_in=r_in,
            r_out=r_out,
            sigclip=sigclip,
            fwhm_scale=fwhm_scale,
            name="photometry",
            set_once=True
        )

        if show:
            def plot_function(im, cmap="Greys_r", color=[0.51, 0.86, 1.]):
                stars = im.stars_coords
                plt.imshow(utils.z_scale(im.data), cmap=cmap, origin="lower")
                viz.plot_marks(*stars.T, np.arange(len(stars)), color=color)

            self.show = blocks.LivePlot(plot_function, size=(10, 10))
        else:
            self.show = blocks.Pass()

    def run_reference_detection(self):
        stack_image, ref_stars, fwhm = super().run_reference_detection()

        centroid = blocks.Pass() if not isinstance(self.centroid, Block) else self.centroid

        self.photometry_sequence = Sequence([
            blocks.Set(stars_coords=ref_stars, name="set stars"),
            blocks.Set(fwhm=fwhm, name="set fwhm"),
            centroid,
            self.show,
            self.photometry,
            blocks.io.SavePhot(self.phot_path, header=stack_image.header, stack=stack_image.data, name="saving")
        ], self.files, telescope=self.telescope, name="Photometry")


class PSFPhotometry(Photometry):
    """PSF Photometry unit (not tested, use not recommended)

    Parameters
    ----------
    fits_manager : prose.FitsManager
         FitsManager of the observation. Should contain a single obs. One of ``fits_manager`` or ``files`` and ``stack` should  be provided
    files : list of str, optional
        List of files to process. One of ``fits_manager`` or ``files`` and ``stack`` should  be provided
    stack: str, optional
        Path of the stack image. Should be specified if ``files`` is specified.
    overwrite : bool, optional
        whether to overwrite existing products, by default False
    n_stars : int, optional
        max number of stars to take into account, by default 500
    psf : Block, optional
        PSF modeling Block (mainly used to estimate fwhm and scale aperture if ``fwhm_scale`` is ``True``), by default :class:`~prose.blocks.Gaussian2D`
    photometry : Block, optional
        aperture photometry Block, by default :class:`~prose.blocks.PhotutilsAperturePhotometry`
    ignore_telescope: bool, optional
        whether to load a default telescope if telescope not recognised, by default False
    """

    def __init__(self,
                 fits_manager=None,
                 files=None,
                 stack=None,
                 overwrite=False,
                 n_stars=500,
                 psf=blocks.Gaussian2D,
                 photometry=blocks.PhotutilsPSFPhotometry,
                 ignore_telescope=False):
        super().__init__(
            fits_manager=fits_manager,
            files=files,
            stack=stack,
            overwrite=overwrite,
            n_stars=n_stars,
            psf=psf,
            ignore_telescope=ignore_telescope
        )

        # Blocks
        assert photometry is None or issubclass(photometry, Block), "photometry must be a subclass of Block"
        self.photometry = photometry

    def run_reference_detection(self):
        stack_image, ref_stars, fwhm = super().run_reference_detection()

        self.photometry_unit = Sequence([
            blocks.Set(stars_coords=ref_stars, name="set stars"),
            blocks.Set(fwhm=fwhm, name="set fwhm"),
            self.photometry(fwhm),
            blocks.io.SavePhot(
                self.phot_path,
                header=fits.getheader(self.stack_path),
                stack=fits.getdata(self.stack_path),
                name="saving")
        ], self.files, telescope=self.telescope, name="Photometry")