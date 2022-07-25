from .. import Sequence, blocks, Block, viz
from os import path
from ..console_utils import info
from .. import utils
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import xarray as xr
from .. import __version__


def plot_function(im, cmap="Greys_r", color=[0.51, 0.86, 1.]):
    stars = im.stars_coords
    plt.imshow(utils.z_scale(im.data), cmap=cmap, origin="lower")
    viz.plot_marks(*stars.T, np.arange(len(stars)), color=color)


class Photometry:
    """Base class for the photometry

    Parameters
    ----------
    files : list of str, optional
        List of files to process
    stack: str, optional
        Path of the stack image
    stars : (2,n) array, optional
        stars on which apertures are placed, by default None so that stars are automatically detected
    overwrite : bool, optional
        whether to overwrite existing products, by default False
    n_stars : int, optional
        max number of stars to take into account, by default 500
    psf : Block, optional
        PSF modeling Block (mainly used to estimate fwhm and scale aperture if ``fwhm_scale`` is ``True``), by default :class:`~prose.blocks.Gaussian2D`
    photometry : Block, optional
        aperture photometry Block, by default :class:`~prose.blocks.PhotutilsAperturePhotometry`
    centroid : Block, optional
        centroid computing Block, by default None to keep centroid fixed
    show : bool, optional
        Wheter to plot the field while doing the photometry, by default False (plotting slow down the processing)
    verbose : bool, optional
        wheter to log progress og the photometric extraction, by default True
    """

    def __init__(self,
                 files=None,
                 stack=None,
                 stars=None,
                 overwrite=False,
                 n_stars=500,
                 psf=blocks.psf.Gaussian2D,
                 stack_psf=blocks.FWHM,
                 photometry=blocks.PhotutilsAperturePhotometry,
                 centroid=None,
                 show=False,
                 verbose=True,
                 twirl=False,
                 **kwargs):

        self.overwrite = overwrite
        self.n_stars = n_stars
        self.images = files
        self.stack = stack
        self.verbose = verbose
        self.stars = stars
        self.centroid = centroid
        self.twirl = twirl

        # sequences
        self.detection_s = None
        self.photometry_s = None

        # preparing inputs and outputs
        self.destination = None
        self.phot = None

        # check blocks
        assert psf is None or issubclass(psf, Block), "psf must be a subclass of Block"
        self.psf = psf
        assert stack_psf is None or issubclass(stack_psf, Block), "stack_psf must be a subclass of Block"
        self.stack_psf = stack_psf
        self.photometry = photometry(**kwargs)
        self.show = show

    def run(self, destination,**kwargs):
        self._check_phot_path(destination)

        # Reference detection
        # -------------------
        self.detection_s = Sequence([
            blocks.DAOFindStars(n_stars=self.n_stars, name="detection",**kwargs),
            blocks.Set(stars_coords=self.stars) if self.stars is not None else blocks.Pass(),
            blocks.Cutouts(size=51),
            blocks.MedianPSF(),
            self.stack_psf(name="fwhm"),
            blocks.ImageBuffer(name="buffer"),
        ])

        self.detection_s.run(self.stack, show_progress=False)
        reference = self.detection_s.buffer.image
        self.stars = reference.stars_coords

        # logging
        info(f"detected stars: {len(reference.stars_coords)}")
        info(f"global psf FWHM: {np.mean(reference.fwhm):.2f} (pixels)")
        time.sleep(0.5)

        # Photometry
        # ----------
        centroid = blocks.Pass() if not isinstance(self.centroid, Block) else self.centroid

        self.photometry_s = Sequence([
            blocks.Set(
                stars_coords=reference.stars_coords.copy(),
                fwhm=reference.fwhm.copy()
            ),
            blocks.AffineTransform(data=False, stars=True, inverse=True) if self.twirl else blocks.Pass(),
            centroid,
            self.show,
            blocks.Peaks(),
            self.photometry,
            blocks.ImageBuffer(),
            blocks.XArray(
                (("time", "apertures", "star"), "fluxes"),
                (("time", "apertures", "star"), "errors"),
                (("time", "apertures", "star"), "apertures_area"),
                (("time", "apertures", "star"), "apertures_radii"),
                (("time", "star"), "annulus_sky"),
                ("time", "sky"),
                (("time", "apertures"), "apertures_area"),
                (("time", "apertures"), "apertures_radii"),
                ("time", "annulus_rin"),
                ("time", "annulus_rout"),
                ("time", "annulus_area"),
                (("time", "star"), "peaks")
            ),
        ], name="Photometry")

        self.photometry_s.run(self.images, show_progress=self.verbose)
        self.save_xarray()

    def save_xarray(self):
        if path.exists(self.phot):
            initial_xarray = xr.load_dataset(self.phot, engine="netcdf4")
        else:
            initial_xarray = xr.Dataset()

        #  TODO: align in time? In case calib and phot are only partially overlapping
        initial_xarray = initial_xarray.drop_dims(["star", "apertures"], errors="ignore")
        initial_xarray = initial_xarray.drop_vars(["annulus_rin", "annulus_rout", "annulus_area", "sky"], errors="ignore")
        phot_xarray = self.photometry_s.xarray.xarray
        xarray = xr.merge([phot_xarray,initial_xarray], combine_attrs="no_conflicts", join='left', compat='override')
        xarray = xarray.transpose("apertures", "star", "time", ...)
        xarray = xarray.assign_coords(stars=(("star", "n"), self.stars))
        xarray.attrs["photometry"] = [b.__class__.__name__ for b in self.photometry_s.blocks]
        xarray.attrs["photometry_prose_version"] = __version__
        xarray.to_netcdf(self.phot)

    def _check_phot_path(self, destination):
        destination = Path(destination)
        if destination.is_dir():
            parent = destination
        else:
            parent = destination.parent

        self.phot = parent / (destination.stem + '.phot')

    def __repr__(self):
        return f"{self.detection_s}\n{self.photometry_s}"

    @property
    def processing_time(self):
        return self.detection_s.processing_time + self.photometry_s.processing_time


class AperturePhotometry(Photometry):
    """Aperture Photometry pipeline

    Parameters
    ----------
    files : list of str, optional
        List of files to process.
    stack: str, optional
        Path of the stack image.
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
        whether to multiply ``apertures``, ``r_in`` and ``r_out`` by the global fwhm, by default True
    sigclip : float, optional
        Sigma clipping factor used in the annulus, by default 3. No effect if :class:`~prose.blocks.SEAperturePhotometry` is used
    psf : Block, optional
        PSF modeling Block (mainly used to estimate fwhm and scale aperture if ``fwhm_scale`` is ``True``), by default :class:`~prose.blocks.Gaussian2D`
    photometry : Block, optional
        aperture photometry Block, by default :class:`~prose.blocks.PhotutilsAperturePhotometry`
    centroid : Block, optional
        centroid computing Block, by default None to keep centroid fixed
    show: bool, optional
        within a notebook, whether to show processed image during reduction, by default False 
    verbose: bool, optional
        whether to print processing info and loading bars, by default True
    twirl: bool, optional,
        whether to use the Twirl block for alignment (see blocks.registration.Twirl)
    """

    def __init__(self,
                 files=None,
                 stack=None,
                 overwrite=False,
                 n_stars=500,
                 stars=None,
                 apertures=None,
                 r_in=5,
                 r_out=8,
                 fwhm_scale=True,
                 sigclip=3.,
                 psf=blocks.psf.Gaussian2D,
                 photometry=blocks.PhotutilsAperturePhotometry,
                 centroid=blocks.BalletCentroid,
                 show=False,
                 verbose=True,
                 twirl=True):

        if apertures is None:
            apertures = np.arange(0.1, 10, 0.25)

        super().__init__(
            files=files,
            stack=stack,
            overwrite=overwrite,
            n_stars=n_stars,
            stars=stars,
            psf=psf,
            show=show,
            verbose=verbose,
            apertures=apertures,
            r_in=r_in,
            r_out=r_out,
            sigclip=sigclip,
            scale=fwhm_scale,
            name="photometry",
            twirl=twirl

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
            scale=fwhm_scale,
            name="photometry",
        )

        if show:
            self.show = blocks.LivePlot(plot_function, size=(10, 10))
        else:
            self.show = blocks.Pass()


class PSFPhotometry(Photometry):
    """PSF Photometry pipeline (not tested)

    Parameters
    ----------
    files : list of str, optional
        List of files to process
    stack: str, optional
        Path of the stack image
    overwrite : bool, optional
        whether to overwrite existing products, by default False
    n_stars : int, optional
        max number of stars to take into account, by default 500
    psf : Block, optional
        PSF modeling Block (mainly used to estimate fwhm and scale aperture if ``fwhm_scale`` is ``True``), by default :class:`~prose.blocks.Gaussian2D`
    photometry : Block, optional
        aperture photometry Block, by default :class:`~prose.blocks.PhotutilsAperturePhotometry`
    """

    def __init__(self,
                 files=None,
                 stack=None,
                 overwrite=False,
                 n_stars=500,
                 psf=blocks.psf.Gaussian2D,
                 photometry=blocks.PhotutilsPSFPhotometry):

        super().__init__(
            files=files,
            stack=stack,
            overwrite=overwrite,
            n_stars=n_stars,
            psf=psf,
            photometry=photometry
        )


class ShapeletPhotometry(Photometry):
    """PSF Photometry pipeline (not tested)

    Parameters
    ----------
    files : list of str, optional
        List of files to process
    stack: str, optional
        Path of the stack image
    overwrite : bool, optional
        whether to overwrite existing products, by default False
    n_stars : int, optional
        max number of stars to take into account, by default 500
    psf : Block, optional
        PSF modeling Block (mainly used to estimate fwhm and scale aperture if ``fwhm_scale`` is ``True``), by default :class:`~prose.blocks.Gaussian2D`
    photometry : Block, optional
        aperture photometry Block, by default :class:`~prose.blocks.PhotutilsAperturePhotometry`
    """

    def __init__(self,
                 files=None,
                 stack=None,
                 stars=None,
                 overwrite=False,
                 n_stars=500,
                 psf=blocks.psf.Gaussian2D,
                 photometry=blocks.Shepard):

        super().__init__(
            files=files,
            stack=stack,
            overwrite=overwrite,
            stars=stars,
            n_stars=n_stars,
            psf=psf,
            photometry=photometry
        )
