import numpy as np
from astropy.table import Table
from photutils import aperture_photometry
from astropy.stats import sigma_clipped_stats
from photutils.background import MMMBackground
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.modeling.fitting import LevMarLSQFitter
from photutils import CircularAperture, CircularAnnulus
from .. import FitsManager
from photutils.psf import IntegratedGaussianPRF, DAOGroup, BasicPSFPhotometry
from ..core import Block
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
        image.fluxes = np.expand_dims(result_tab["flux_fit"].data, 0)
        image.fluxes_errors = np.sqrt(image.fluxes)  # result_tab['flux_unc']


class PhotutilsAperturePhotometry(Block):
    """
    Aperture photometry using :code:`photutils`.
    For more details check https://photutils.readthedocs.io/en/stable/aperture.html

    |write| 
    
    - ``Image.stars_coords``
    - ``Image.apertures_area``
    - ``Image.sky``
    - ``Image.fluxes``
    - ``Image.annulus_area``
    - ``Image.annulus_rin``
    - ``Image.annulus_rout``
    - ``Image.apertures_radii``
    - ``Image.fluxes``

    |modify| 

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
            set_once=False,
            **kwargs):

        super().__init__(**kwargs)
        if apertures is None:
            self.apertures = np.arange(0.1, 8, 0.25)
        else:
            self.apertures = apertures

        self.annulus_inner_radius = r_in
        self.annulus_outer_radius = r_out
        self.annulus_final_rin = None
        self.annulus_final_rout = None
        self.aperture_final_r = None

        self.n_apertures = len(self.apertures)
        self.n_stars = None
        self.circular_apertures = None
        self.annulus_apertures = None
        self.annulus_masks = None

        self.circular_apertures_area = None
        self.annulus_area = None
        self.fwhm_scale = fwhm_scale
        self.sigclip = sigclip
        self.set_once = set_once

    def set_apertures(self, stars_coords, fwhm):

        self.annulus_final_rin = self.annulus_inner_radius * fwhm
        self.annulus_final_rout = self.annulus_outer_radius * fwhm
        self.aperture_final_r = fwhm*self.apertures

        self.annulus_apertures = CircularAnnulus(
            stars_coords,
            r_in=self.annulus_final_rin,
            r_out=self.annulus_final_rout,
        )
        if callable(self.annulus_apertures.area):
            self.annulus_area = self.annulus_apertures.area()
        else:
            self.annulus_area = self.annulus_apertures.area

        self.circular_apertures = [CircularAperture(stars_coords, r=r) for r in self.aperture_final_r]

        # Unresolved buf; sometimes circular_apertures.area is a method, sometimes a float
        if callable(self.circular_apertures[0].area):
            self.circular_apertures_area = [ca.area() for ca in self.circular_apertures]
        else:
            self.circular_apertures_area = [ca.area for ca in self.circular_apertures]

        self.annulus_masks = self.annulus_apertures.to_mask(method="center")
        self.n_stars = len(stars_coords)

    def run(self, image, **kwargs):
        if self.circular_apertures is None and self.set_once:
            self.set_apertures(image.stars_coords, image.fwhm if self.fwhm_scale else 1)
        else:
            self.set_apertures(image.stars_coords, image.fwhm if self.fwhm_scale else 1)

        bkg_median = []
        for mask in self.annulus_masks:
            annulus_data = mask.multiply(image.data)
            if annulus_data is not None:
                annulus_data_1d = annulus_data[mask.data > 0]
                _, median_sigma_clip, _ = sigma_clipped_stats(annulus_data_1d, sigma=self.sigclip)
                bkg_median.append(median_sigma_clip)
            else:
                bkg_median.append(0.)

        bkg_median = np.array(bkg_median)

        image.apertures_area = self.circular_apertures_area
        image.sky = bkg_median
        image.fluxes = np.zeros((self.n_apertures, self.n_stars))
        image.annulus_area = self.annulus_area
        image.annulus_rin = self.annulus_final_rin
        image.annulus_rout = self.annulus_final_rout
        image.apertures_radii = self.aperture_final_r

        data = image.data.copy()
        data[data < 0] = 0

        photometry = aperture_photometry(data, self.circular_apertures)
        fluxes = np.array([
            photometry[f"aperture_sum_{a}"] - (bkg_median * self.circular_apertures_area[a])
            for a in range(len(self.apertures))
        ])

        # dummy values if negative or nan
        fluxes[np.isnan(fluxes)] = 1
        fluxes[fluxes < 0 ] = 1

        image.fluxes = fluxes

        self.compute_error(image)
        image.header["sky"] = np.mean(image.sky)

    def compute_error(self, image):

        image.errors = np.zeros((self.n_apertures, self.n_stars))

        for i, aperture_area in enumerate(self.circular_apertures_area):
            area = aperture_area * (1 + aperture_area / self.annulus_area)
            image.errors[i, :] = image.telescope.error(
                image.fluxes[i],
                area,
                image.sky,
                image.exposure,
                airmass=image.get("keyword_airmass"),
            )

    def citations(self):
        return "astropy", "photutils"


class SEAperturePhotometry(Block):
    """
    Aperture photometry using :code:`sep`.
    For more details check https://sep.readthedocs.io

    SEP is a python wrapping of the C Source Extractor code, hence being 2 times faster that Photutils version.
    Forced aperture photometry can be done simple by using :code:`stack=True` on the detection Block used, hence using
    stack sources positions along the photometric extraction.

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

        image.errors = np.zeros((self.n_apertures, self.n_stars))

        for i, aperture_area in enumerate(image.apertures_area):
            area = aperture_area * (1 + aperture_area / image.annulus_area)
            image.errors[i, :] = image.telescope.error(
                image.fluxes[i],
                area,
                image.sky,
                image.exposure,
                airmass=image.header[image.telescope.keyword_airmass],
            )
