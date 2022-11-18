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
from .. import Block
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
    r"""
    Aperture photometry using the :code:`CircularAperture` and :code:`CircularAnnulus` of photutils_ with a wide range of apertures. By default annulus goes from 5 fwhm to 8 fwhm and apertures from 0.1 to 10 times the fwhm with 0.25 steps (leading to 40 apertures).

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

    The positions of individual stars are taken from :code:`Image.stars_coords` so one of the detection block should be used, placed before this one.

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
        apertures in fraction of fwhm, by default None, i.e. np.arange(0.1, 8, 0.25)
    r_in : int, optional
        radius of the inner annulus in fraction of fwhm, by default 5
    r_out : int, optional
        radius of the outer annulus in fraction of fwhm, by default 8
    scale: bool or float:
        Multiplication factor applied to `apertures`.
        - if True: `apertures` multiplied by image.fwhm, varying for each image
        - if False: `apertures` not multiplied
        - if float: `apertures` multiplied `scale` and held fixed for all images
    """

    
    def __init__(
            self,
            apertures=None,
            r_in=5,
            r_out=8,
            scale=True,
            sigclip = 2.,
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
        self.scale = scale
        self.sigclip = sigclip

        self._has_fix_scale = not isinstance(self.scale, bool)

    def set_apertures(self, stars_coords, fwhm=1):

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

    def run(self, image):
        try:
            if self._has_fix_scale:
                self.set_apertures(image.stars_coords, self.scale)
            elif self.scale:
                self.set_apertures(image.stars_coords, image.fwhm)
            else:
                self.set_apertures(image.stars_coords)
        except ZeroDivisionError: # temporary
            image.discard=True
            return None

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
        image.annulus_sky = bkg_median
        image.sky = bkg_median.mean()
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

    @property
    def citations(self):
        return "photutils"


class SEAperturePhotometry(Block):
    
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

    def set_apertures(self, stars_coords, fwhm=1):

        self.annulus_final_rin = self.annulus_inner_radius * fwhm
        self.annulus_final_rout = self.annulus_outer_radius * fwhm
        self.aperture_final_r = fwhm*self.apertures

        self.annulus_area = np.pi * (self.annulus_final_rout - self.annulus_final_ri)**2
        self.circular_apertures_area = np.pi * self.aperture_final_r**2

        self.apertures = np.repeat(self.aperture_final_r[..., None], len(stars_coords), axis=1)
        self.n_stars = len(stars_coords)

    def run(self, image):
        try:
            if self._has_fix_scale:
                self.set_apertures(image.stars_coords, self.scale)
            elif self.scale:
                self.set_apertures(image.stars_coords, image.fwhm)
            else:
                self.set_apertures(image.stars_coords)
        except ZeroDivisionError: # temporary
            image.discard=True
            return None

        data = image.data.copy()
        data[data < 0] = 0

        x, y = image.stars_coords.T
        image.apertures_area = self.circular_apertures_area
        image.annulus_sky = sep.sum_circann(data, x, y, self.annulus_final_rin, self.annulus_final_rout, subpix=0)
        image.sky = image.annulus_sky.median()
        image.annulus_area = self.annulus_area
        image.annulus_rin = self.annulus_final_rin
        image.annulus_rout = self.annulus_final_rout
        image.apertures_radii = self.aperture_final_r

        fluxes = sep.sum_circle(data, x, y, self.apertures, subpix=0)

        # dummy values if negative or nan
        fluxes[np.isnan(fluxes)] = 1
        fluxes[fluxes < 0 ] = 1

        image.fluxes = fluxes

        self.compute_error(image)
        image.header["sky"] = np.mean(image.sky)

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
