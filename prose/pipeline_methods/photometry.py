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
from photutils.psf import IntegratedGaussianPRF, DAOGroup, BasicPSFPhotometry
from prose.console_utils import TQDM_BAR_FORMAT, INFO_LABEL


def aperture_photometry_annulus(
        fits_files,
        stars_positions,
        fwhm,
        apertures=None,
        fixed_fwhm=True,
        annulus_inner_radius=5,
        annulus_outer_radius=8,
):
    """
    Perform aperture photometry on a list of aligned fits.
    For more details check https://photutils.readthedocs.io/en/stable/aperture.html

    Parameters
    ----------
    fits_files : list
        list of fits files
    stars_positions : ndarray
        (x,y) stars coordinate in the aligned fits
    fwhm : float
        global fwhm in pixel
    apertures : ndarray or list, optional
        apertures in fraction of fwhm, by default np.arange(0.1, 10, 0.25)
    fixed_fwhm : bool, optional
        Wether final apertures are computed w.r.t. a fixed global fwhm or vary with each image fwhm, by default True
    annulus_inner_radius : int, optional
        radius of the inner annulus in fraction of fwhm, by default 5
    annulus_outer_radius : int, optional
        radius of the outer annulus in fraction of fwhm, by default 8

    Returns
    -------
    tuple as (ndarray, dict)
        - ndarray: fluxes with shape (apertures, stars, images)
        - dict: {
            "apertures": apertures,
            "sky": array of mean annulus flux for each images
        }
    """
    if apertures is None:
        np.arange(0.1, 10, 0.25)

    print("{} global psf FWHM: {:.2f} (pixels)".format(INFO_LABEL, np.mean(fwhm)))

    n_stars = np.shape(stars_positions)[0]
    n_images = len(fits_files)
    n_apertures = len(apertures)

    photometry_data = np.zeros((n_apertures, n_stars, n_images))
    sky = []

    for f, file_path in enumerate(
            tqdm(
                fits_files[0::],
                desc="Photometry extraction",
                unit="files",
                ncols=80,
                bar_format=TQDM_BAR_FORMAT,
            )
    ):
        phot_areas = []
        _apertures = []

        data = fits.getdata(file_path)

        if not fixed_fwhm:
            _fwhm = fwhm[f]
        else:
            _fwhm = fwhm

        annulus_apertures = CircularAnnulus(
            stars_positions,
            r_in=annulus_inner_radius * _fwhm,
            r_out=annulus_outer_radius * _fwhm,
        )
        annulus_masks = annulus_apertures.to_mask(method="center")

        bkg_median = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(data)

            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigma_clip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigma_clip)

        bkg_median = np.array(bkg_median)

        for a, ape in enumerate(apertures):
            _apertures.append(_fwhm * ape)  # aperture diameter in pixel
            circular_apertures = CircularAperture(stars_positions, r=_fwhm * ape / 2)

            # Unresolved buf; sometimes circular_apertures.area is a method, sometimes a float
            if callable(circular_apertures.area):
                circular_apertures_area = circular_apertures.area()
            else:
                circular_apertures_area = circular_apertures.area

            phot_areas.append(circular_apertures_area)

            im_phot = aperture_photometry(data, circular_apertures)
            im_phot["annulus_median"] = bkg_median
            im_phot["aper_bkg"] = bkg_median * circular_apertures_area
            im_phot["aper_sum_bkgsub"] = im_phot["aperture_sum"] - im_phot["aper_bkg"]

            photometry_data[a, :, f] = np.array(im_phot["aper_sum_bkgsub"])

        sky.append(np.mean(bkg_median))

    return photometry_data, {"apertures": _apertures, "sky": sky}


def psf_photometry_basic(fits_files, stars_positions, fwhm):
    """
    A simple and still experimental PSF photometry method
    For more details check:
    https://photutils.readthedocs.io/en/stable/psf.html

    Parameters
    ----------
    fits_files : list
        list of fits files
    stars_positions : ndarray
        (x,y) stars coordinate in the aligned fits
    fwhm : float
        global fwhm in pixel

    Returns
    -------
    tuple as (ndarray, dict)
        - ndarray: fluxes with shape (apertures, stars, images)
        - dict: {"sky": for now just ones}
    """
    print("{} global psf FWHM: {:.2f} (pixels)".format(INFO_LABEL, np.mean(fwhm)))

    n_stars = np.shape(stars_positions)[0]
    n_images = len(fits_files)

    photometry_data = np.zeros((1, n_stars, n_images))

    pos = Table(
        names=["x_0", "y_0"], data=[stars_positions[:, 0], stars_positions[:, 1]]
    )

    daogroup = DAOGroup(2.0 * fwhm * gaussian_sigma_to_fwhm)

    mmm_bkg = MMMBackground()

    psf_model = IntegratedGaussianPRF(sigma=fwhm)
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
                fits_files[0::],
                desc="Photometry extraction",
                unit="files",
                ncols=80,
                bar_format=TQDM_BAR_FORMAT,
            )
    ):
        image = fits.getdata(image)

        result_tab = photometry(image=image, init_guesses=pos)

        photometry_data[0, :, i] = result_tab["flux_fit"]
        sky.append(1)

    return photometry_data, {"sky": sky}