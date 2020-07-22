import numpy as np
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table
from photutils import aperture_photometry
from astropy.stats import sigma_clipped_stats
from photutils.background import MMMBackground
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.modeling.fitting import LevMarLSQFitter
from photutils import CircularAperture, CircularAnnulus
from prose import io, FitsManager
from photutils.psf import IntegratedGaussianPRF, DAOGroup, BasicPSFPhotometry
from prose.console_utils import TQDM_BAR_FORMAT, INFO_LABEL
from prose._blocks.psf import Gaussian2D
from prose._blocks.base import Block


# TODO: differential_vaphot
# TODO: difference imaging


class BasePhotometry(Block):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PSFPhotometry(BasePhotometry):

    def __init__(self, fits_explorer=None, fwhm=Gaussian2D, **kwargs):

        super().__init__(**kwargs)
        self.set_fits_explorer(fits_explorer)
        self.fwhm_fit = Gaussian2D()

    def set_fits_explorer(self, fits_explorer):
        if isinstance(fits_explorer, FitsManager):
            self.fits_explorer = fits_explorer
            self.files = self.fits_explorer.get("reduced")

    def run(self, stars_coords):
        stack_path = self.fits_explorer.get("stack")[0]
        stack_fwhm = np.mean(self.fwhm_fit.run(fits.getdata(stack_path), stars_coords)[0:2])

        print("{} global psf FWHM: {:.2f} (pixels)".format(INFO_LABEL, np.mean(stack_fwhm)))

        n_stars = np.shape(stars_coords)[0]
        n_images = len(self.files)

        fluxes = np.zeros((n_stars, n_images))

        pos = Table(
            names=["x_0", "y_0"], data=[stars_coords[:, 0], stars_coords[:, 1]]
        )

        daogroup = DAOGroup(2.0 * stack_fwhm * gaussian_sigma_to_fwhm)

        mmm_bkg = MMMBackground()

        psf_model = IntegratedGaussianPRF(sigma=stack_fwhm)
        psf_model.sigma.fixed = False

        sky = []

        psf_model.x_0.fixed = True
        psf_model.y_0.fixed = True

        photometry = BasicPSFPhotometry(
            group_maker=daogroup,
            bkg_estimator=mmm_bkg,
            psf_model=psf_model,
            fitter=LevMarLSQFitter(),
            fitshape=(17, 17)
        )

        for i, image in enumerate(
                tqdm(
                    self.files[0::],
                    desc="Photometry extraction",
                    unit="files",
                    ncols=80,
                    bar_format=TQDM_BAR_FORMAT,
                )
        ):
            image = fits.getdata(image)

            result_tab = photometry(image=image, init_guesses=pos)

            fluxes[:, i] = result_tab["flux_fit"]
            sky.append(1)

        return fluxes, np.ones_like(fluxes), {"sky": sky}


class ForcedAperturePhotometry(Block):
    """
    Fixed positions aperture photometry.
    For more details check https://photutils.readthedocs.io/en/stable/aperture.html

    Parameters
    ----------
    fits_explorer : FitsManager
        FitsManager containing a single observation
    apertures : ndarray or list, optional
        apertures in fraction of fwhm, by default None, i.e. np.arange(0.1, 10, 0.25)
    annulus_inner_radius : int, optional
        radius of the inner annulus in fraction of fwhm, by default 5
    annulus_outer_radius : int, optional
        radius of the outer annulus in fraction of fwhm, by default 8
    """

    def __init__(self, apertures=None, fwhm_fit=None, annulus_inner_radius=5, annulus_outer_radius=8, **kwargs):

        super().__init__(**kwargs)
        if apertures is None:
            self.apertures = np.arange(0.1, 10, 0.25)
        else:
            self.apertures = apertures

        self.annulus_inner_radius = annulus_inner_radius
        self.annulus_outer_radius = annulus_outer_radius
        self.fits_manager = None

        if fwhm_fit is None:
            self.fwhm_fit = Gaussian2D()

        self.n_apertures = len(self.apertures)
        self.n_stars = None
        self.circular_apertures = None
        self.annulus_apertures = None
        self.annulus_masks = None

        self.circular_apertures_area = None
        self.annulus_area = None

    def initialize(self, fits_manager):
        if isinstance(fits_manager, FitsManager):
            self.fits_manager = fits_manager

    def set_apertures(self, stars_coords, fwhm):

        self.annulus_apertures = CircularAnnulus(
            stars_coords,
            r_in=self.annulus_inner_radius * fwhm,
            r_out=self.annulus_outer_radius * fwhm,
        )
        if callable(self.annulus_apertures.area):
            self.annulus_area = self.annulus_apertures.area()
        else:
            self.annulus_area = self.annulus_apertures.area

        self.circular_apertures = [CircularAperture(stars_coords, r=fwhm*aperture) for aperture in self.apertures]

        # Unresolved buf; sometimes circular_apertures.area is a method, sometimes a float
        if callable(self.circular_apertures[0].area):
            self.circular_apertures_area = [ca.area() for ca in self.circular_apertures]
        else:
            self.circular_apertures_area = [ca.area for ca in self.circular_apertures]

        self.annulus_masks = self.annulus_apertures.to_mask(method="center")
        self.n_stars = len(stars_coords)

    def run(self, image, **kwargs):
        if self.circular_apertures is None:
            self.set_apertures(image.stars_coords, image.fwhm)

        bkg_median = []
        for mask in self.annulus_masks:
            annulus_data = mask.multiply(image.data)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigma_clip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigma_clip)

        bkg_median = np.array(bkg_median)

        image.apertures_area = self.circular_apertures_area
        image.sky = bkg_median
        image.fluxes = np.zeros((self.n_apertures, self.n_stars))
        image.annulus_area = self.annulus_area

        for a, ape in enumerate(self.apertures):
            photometry = aperture_photometry(image.data, self.circular_apertures[a])
            image.fluxes[a] = np.array(photometry["aperture_sum"] - (bkg_median * self.circular_apertures_area[a]))

        self.compute_error(image)
        image.header["sky"] = np.mean(image.sky)

    def compute_error(self, image):

        image.fluxes_errors = np.zeros((self.n_apertures, self.n_stars))

        for i, aperture_area in enumerate(self.circular_apertures_area):
            area = aperture_area * (1 + aperture_area / self.annulus_area)
            image.fluxes_errors[i, :] = self.fits_manager.telescope.error(
                image.fluxes[i],
                area,
                image.sky,
                image.header[self.fits_manager.telescope.keyword_exposure_time],
                airmass=image.header[self.fits_manager.telescope.keyword_airmass],
            )


class MovingAperturePhotometry(ForcedAperturePhotometry):
    """
    Aperture photometry.
    For more details check https://photutils.readthedocs.io/en/stable/aperture.html

    Parameters
    ----------
    fits_explorer : FitsManager
        FitsManager containing a single observation
    apertures : ndarray or list, optional
        apertures in fraction of fwhm, by default None, i.e. np.arange(0.1, 10, 0.25)
    annulus_inner_radius : int, optional
        radius of the inner annulus in fraction of fwhm, by default 5
    annulus_outer_radius : int, optional
        radius of the outer annulus in fraction of fwhm, by default 8
    """

    def __init__(self, apertures=None, fwhm_fit=None, annulus_inner_radius=5, annulus_outer_radius=8, **kwargs):
        super().__init__(
            apertures=apertures,
            fwhm_fit=fwhm_fit,
            annulus_inner_radius=annulus_inner_radius,
            annulus_outer_radius=annulus_outer_radius,
            **kwargs)

    def run(self, image, **kwargs):
        self.set_apertures(image.stars_coords.copy(), image.fwhm)

        bkg_median = []
        for mask in self.annulus_masks:
            annulus_data = mask.multiply(image.data)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigma_clip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigma_clip)

        bkg_median = np.array(bkg_median)

        image.apertures_area = self.circular_apertures_area
        image.sky = bkg_median
        image.fluxes = np.zeros((self.n_apertures, self.n_stars))
        image.annulus_area = self.annulus_area

        for a, ape in enumerate(self.apertures):
            photometry = aperture_photometry(image.data, self.circular_apertures[a])
            image.fluxes[a] = np.array(photometry["aperture_sum"] - (bkg_median * self.circular_apertures_area[a]))

        self.compute_error(image)
        image.header["sky"] = np.mean(image.sky)