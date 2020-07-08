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
from prose.pipeline.psf import NonLinearGaussian2D


# TODO: differential_vaphot
# TODO: difference imaging

def variable_aperture_photometry_annulus(*args, **kwargs):
    return aperture_photometry_annulus(*args, fixed_fwhm=False, **kwargs)

class BasePhotometry:

    def __init__(self):
        pass

class PSFPhotometry(BasePhotometry):

    def __init__(
        self,
        fits_explorer=None,
        fwhm=NonLinearGaussian2D):

        self.set_fits_explorer(fits_explorer)
        self.fwhm_fit = NonLinearGaussian2D()
    
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



class AperturePhotometry(BasePhotometry):
    """
    Perform aperture photometry on a list of aligned fits.
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

    def __init__(
        self,
        fits_explorer=None,       
        apertures=None,
        annulus_inner_radius=5,
        annulus_outer_radius=8
    ):
        self.set_fits_explorer(fits_explorer)

        if apertures is None:
            self.apertures = np.arange(0.1, 10, 0.25)
        else:
            self.apertures = apertures

        self.annulus_inner_radius = annulus_inner_radius
        self.annulus_outer_radius = annulus_outer_radius
        self.fwhm_fit = NonLinearGaussian2D()

    def set_fits_explorer(self, fits_explorer):
        if isinstance(fits_explorer, FitsManager):
            self.fits_explorer = fits_explorer
            self.files = self.fits_explorer.get("reduced")
            self.exposure = np.min(io.fits_keyword_values(self.files, self.fits_explorer.telescope.keyword_exposure_time))
            self.airmass = io.fits_keyword_values(self.files, self.fits_explorer.telescope.keyword_airmass)

    def run(self, stars_coords):
        stack_path = self.fits_explorer.get("stack")[0]
        stack_fwhm = np.mean(self.fwhm_fit.run(fits.getdata(stack_path), stars_coords)[0:2])

        print("{} global psf FWHM: {:.2f} (pixels)".format(INFO_LABEL, np.mean(stack_fwhm)))

        n_stars = np.shape(stars_coords)[0]
        n_images = len(self.files)
        n_apertures = len(self.apertures)

        fluxes = np.zeros((n_apertures, n_stars, n_images))
        apertures_area = np.zeros((n_apertures, n_images))
        annulus_area = np.zeros(n_images)
        sky = np.zeros((n_stars, n_images))

        for i, file_path in enumerate(
                tqdm(
                    self.files[0::],
                    desc="Photometry extraction",
                    unit="files",
                    ncols=80,
                    bar_format=TQDM_BAR_FORMAT,
                )
        ):
            _apertures_area = []

            data = fits.getdata(file_path)

            if isinstance(stack_fwhm, (np.ndarray, list)):
                _fwhm = stack_fwhm[i]
            else:
                _fwhm = stack_fwhm

            annulus_apertures = CircularAnnulus(
                stars_coords,
                r_in=self.annulus_inner_radius * _fwhm,
                r_out=self.annulus_outer_radius * _fwhm,
            )
            annulus_masks = annulus_apertures.to_mask(method="center")
            if callable(annulus_apertures.area):
                annulus_area[i] = annulus_apertures.area()
            else:
                annulus_area[i] = annulus_apertures.area

            bkg_median = []
            for mask in annulus_masks:
                annulus_data = mask.multiply(data)
                annulus_data_1d = annulus_data[mask.data > 0]
                _, median_sigma_clip, _ = sigma_clipped_stats(annulus_data_1d)
                bkg_median.append(median_sigma_clip)

            bkg_median = np.array(bkg_median)

            for a, ape in enumerate(self.apertures):
                # aperture diameter in pixel
                aperture = _fwhm * ape
                circular_apertures = CircularAperture(stars_coords, r=aperture)

                # Unresolved buf; sometimes circular_apertures.area is a method, sometimes a float
                if callable(circular_apertures.area):
                    circular_apertures_area = circular_apertures.area()
                else:
                    circular_apertures_area = circular_apertures.area

                im_phot = aperture_photometry(data, circular_apertures)
                _fluxes = im_phot["aperture_sum"] - (bkg_median * circular_apertures_area)
                fluxes[a, :, i] = np.array(_fluxes)
                apertures_area[a, i] = circular_apertures_area


            sky[:, i] = bkg_median

        fluxes_errors = np.zeros(np.shape(fluxes))

        for i, aperture_area in enumerate(apertures_area):
            area = aperture_area * (1 + aperture_area/annulus_area)
            fluxes_errors[i, :, :] = self.fits_explorer.telescope.error(
                fluxes[i, :],
                area,
                sky,
                self.exposure,
                airmass=self.airmass,
            )
        
        return fluxes, fluxes_errors, {
            "apertures_area": apertures_area,
            "annulus_area": annulus_area,
            "annulus_sky": sky
        }