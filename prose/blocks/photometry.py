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
from .. import io, FitsManager
from photutils.psf import IntegratedGaussianPRF, DAOGroup, BasicPSFPhotometry
from ..console_utils import TQDM_BAR_FORMAT, INFO_LABEL
from .psf import Gaussian2D
from .base import Block
try:
    import sep
except:
    raise AssertionError("Please install sep")


# TODO: differential_vaphot
# TODO: difference imaging


class PhotutilsPSFPhotometry(Block):

    def __init__(self, fwhm, **kwargs):
        super().__init__(**kwargs)

        daogroup = DAOGroup(2.0 * fwhm * gaussian_sigma_to_fwhm)
        mmm_bkg = MMMBackground()
        psf_model = IntegratedGaussianPRF(sigma=fwhm)
        psf_model.sigma.fixed = False
        psf_model.x_0.fixed = True
        psf_model.y_0.fixed = True

        self.photometry = BasicPSFPhotometry(
            group_maker=daogroup,
            bkg_estimator=mmm_bkg,
            psf_model=psf_model,
            fitter=LevMarLSQFitter(),
            fitshape=(17, 17)
        )

    def run(self, image, **kwargs):
        result_tab = self.photometry(
            image=image.data,
            init_guesses=Table(names=["x_0", "y_0"], data=[image.stars_coords[:, 0], image.stars_coords[:, 1]])
        )
        image.fluxes = result_tab["flux_fit"]
        image.fluxes_errors = result_tab['flux_unc']


class PhotutilsAperturePhotometry(Block):
    """
    Aperture photometry using :code:`photutils`.
    For more details check https://photutils.readthedocs.io/en/stable/aperture.html

    Parameters
    ----------
    apertures : ndarray or list, optional
        apertures in fraction of fwhm, by default None, i.e. np.arange(0.1, 10, 0.25)
    r_in : int, optional
        radius of the inner annulus in fraction of fwhm, by default 5
    r_out : int, optional
        radius of the outer annulus in fraction of fwhm, by default 8
    """

    def __init__(
            self,
            apertures=None,
            r_in=5,
            r_out=8,
            fwhm_scale=True,
            sigclip = 2.,
            **kwargs):

        super().__init__(**kwargs)
        if apertures is None:
            self.apertures = np.arange(0.1, 10, 0.25)
        else:
            self.apertures = apertures

        self.annulus_inner_radius = r_in
        self.annulus_outer_radius = r_out

        self.n_apertures = len(self.apertures)
        self.n_stars = None
        self.circular_apertures = None
        self.annulus_apertures = None
        self.annulus_masks = None

        self.circular_apertures_area = None
        self.annulus_area = None
        self.fwhm_scale = fwhm_scale
        self.sigclip = sigclip

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
        self.set_apertures(image.stars_coords, image.fwhm if self.fwhm_scale else 1)

        bkg_median = []
        for mask in self.annulus_masks:
            annulus_data = mask.multiply(image.data)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigma_clip, _ = sigma_clipped_stats(annulus_data_1d, sigma=self.sigclip)
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
            image.fluxes_errors[i, :] = self.telescope.error(
                image.fluxes[i],
                area,
                image.sky,
                image.header[self.telescope.keyword_exposure_time],
                airmass=image.header[self.telescope.keyword_airmass],
            )

    def citations(self):
        return "astropy", "photutils"

    @staticmethod
    def doc():
        return r"""Aperture photometry using the :code:`CircularAperture` and :code:`CircularAnnulus` of photutils_ with a wide range of apertures. By default annulus goes from 5 fwhm to 8 fwhm and apertures from 0.1 to 10 times the fwhm with 0.25 steps (leading to 40 apertures).

The error (e.g. in ADU) is then computed following:

.. math::

   \sigma = \sqrt{S + (A_p + \frac{A_p}{A_n})(b + r^2 + \frac{gain^2}{2}) + scint }


.. image:: images/aperture_phot.png
   :align: center
   :width: 110px

with :math:`S` the flux (ADU) within an aperture of area :math:`A_p`, :math:`b` the background flux (ADU) within an annulus of area :math:`A_n`, :math:`r` the read-noise (ADU) and :math:`scint` is a scintillation term expressed as:


.. math::

   scint = \frac{S_fd^{2/3} airmass^{7/4} h}{16T}

with :math:`S_f` a scintillation factor, :math:`d` the aperture diameter (m), :math:`h` the altitude (m) and :math:`T` the exposure time.

The positions of individual stars are taken from :code:`Image.stars_coords` so one of the detection block should be used, placed before this one."""


class SEAperturePhotometry(Block):
    """
    Aperture photometry using :code:`sep`.
    For more details check https://sep.readthedocs.io

    SEP is a python wrapping of the C Source Extractor code, hence being 2 times faster that Photutils version. Forced aperture photometry can be done simple by using :code:`stack=True` on the detection Block used, hence using stack sources positions along the photometric extraction.

    Parameters
    ----------
    fits_explorer : FitsManager
        FitsManager containing a single observation
    apertures : ndarray or list, optional
        apertures in fraction of fwhm, by default None, i.e. np.arange(0.1, 10, 0.25)
    r_in : int, optional
        radius of the inner annulus in fraction of fwhm, by default 5
    r_out : int, optional
        radius of the outer annulus in fraction of fwhm, by default 8
    """

    def __init__(self, apertures=None, r_in=5, r_out=8, fwhm_scale=True, **kwargs):

        super().__init__(**kwargs)
        if apertures is None:
            self.apertures = np.arange(0.1, 10, 0.25)
        else:
            self.apertures = apertures

        self.annulus_inner_radius = r_in
        self.annulus_outer_radius = r_out

        self.n_apertures = len(self.apertures)
        self.n_stars = None
        self.circular_apertures = None
        self.annulus_apertures = None
        self.annulus_masks = None

        self.circular_apertures_area = None
        self.annulus_area = None
        self.fwhm_scale = fwhm_scale

    def run(self, image):
        if self.fwhm_scale:
            r_in = self.annulus_inner_radius * image.fwhm
            r_out = self.annulus_outer_radius * image.fwhm
            r = self.apertures * image.fwhm
        else:
            r_in = self.annulus_inner_radius
            r_out = self.annulus_outer_radius
            r = self.apertures

        self.n_stars = len(image.stars_coords)
        image.fluxes = np.zeros((self.n_apertures, self.n_stars))

        data = image.data.copy().byteswap().newbyteorder()

        for i, _r in enumerate(r):
            image.fluxes[i, :], fluxerr, flag = sep.sum_circle(
                data, 
                *image.stars_coords.T, 
                _r, bkgann=(r_in, r_out), subpix=0)

        image.sky = 0
        image.apertures_area = np.pi * r**2
        image.annulus_area = np.pi * (r_out**2 - r_in**2)

        # fluxes = np.zeros((self.n_apertures, self.n_stars))
        # bkg = np.zeros((self.n_apertures, self.n_stars))

        # for i, _r in enumerate(r):
        #     fluxes[i, :], _, _ = sep.sum_circle(
        #         data, 
        #         *image.stars_coords.T, 
        #         _r, bkgann=(r_in, r_out), subpix=0)
            
        #     bkg[i, :], _, _ = sep.sum_circann(data, *image.stars_coords.T, r_in, r_out, subpix=0)

        #     image.fluxes = fluxes - bkg

        # image.sky = np.mean(bkg)
        # image.apertures_area = np.pi * r**2
        # image.annulus_area = np.pi * (r_out**2 - r_in**2)
        # image.header["sky"] = np.mean(image.sky)

        self.compute_error(image)

    def compute_error(self, image):

        image.fluxes_errors = np.zeros((self.n_apertures, self.n_stars))

        for i, aperture_area in enumerate(image.apertures_area):
            area = aperture_area * (1 + aperture_area / image.annulus_area)
            image.fluxes_errors[i, :] = self.telescope.error(
                image.fluxes[i],
                area,
                image.sky,
                image.header[self.telescope.keyword_exposure_time],
                airmass=image.header[self.telescope.keyword_airmass],
            )

    @staticmethod
    def doc():
        return r"""Aperture photometry using `sep <https://sep.readthedocs.io>`_, a python wrapper around the C Source Extractor.

The error (e.g. in ADU) is then computed following:

.. math::

   \sigma = \sqrt{S + (A_p + \frac{A_p}{A_n})(b + r^2 + \frac{gain^2}{2}) + scint }


.. image:: images/aperture_phot.png
   :align: center
   :width: 110px

with :math:`S` the flux (ADU) within an aperture of area :math:`A_p`, :math:`b` the background flux (ADU) within an annulus of area :math:`A_n`, :math:`r` the read-noise (ADU) and :math:`scint` is a scintillation term expressed as:


.. math::

   scint = \frac{S_fd^{2/3} airmass^{7/4} h}{16T}

with :math:`S_f` a scintillation factor, :math:`d` the aperture diameter (m), :math:`h` the altitude (m) and :math:`T` the exposure time.

The positions of individual stars are taken from :code:`Image.stars_coords` so one of the detection block should be used, placed before this one."""