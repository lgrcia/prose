import astropy.wcs.utils as wcsutils
from .blocks import SegmentedPeaks
import warnings
from . import Image, twirl, utils
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from datetime import datetime
from astropy.time import Time


def gaia_stars(image, n=500, limit=-1, align=15):
    """Query gaia catalog for stars in the field
    """
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = limit

    cone_radius = np.sqrt(2) * np.max(image.shape) * image.telescope.pixel_scale / 120

    coord = SkyCoord(image.ra, image.dec, frame='icrs', unit=(image.telescope.ra_unit, image.telescope.dec_unit))
    radius = u.Quantity(cone_radius, u.arcminute)
    gaia_query = Gaia.cone_search_async(coord, radius, verbose=False, )
    gaia_data = gaia_query.get_results()
    gaia_data.sort("phot_g_mean_flux", reverse=True)

    delta_years = (utils.datetime_to_years(datetime.strptime(image.date.split("T")[0], "%Y-%m-%d")) - \
                   gaia_data["ref_epoch"].data.data) * u.year

    dra = delta_years * gaia_data["pmra"].to(u.deg / u.year)
    ddec = delta_years * gaia_data["pmdec"].to(u.deg / u.year)

    skycoords = SkyCoord(
        ra=gaia_data['ra'].quantity + dra,
        dec=gaia_data['dec'].quantity + ddec,
        pm_ra_cosdec=gaia_data['pmra'],
        pm_dec=gaia_data['pmdec'],
        radial_velocity=gaia_data['radial_velocity'],
        obstime=Time(2015.0, format='decimalyear'))

    gaias = np.array(wcsutils.skycoord_to_pixel(skycoords, image.wcs)).T
    gaias[np.any(np.isnan(gaias), 1), :] = [0, 0]
    gaia_data["x"], gaia_data["y"] = gaias.T
    inside = np.all((np.array([0, 0]) < gaias) & (gaias < np.array(image.shape)), 1)
    gaia_data = gaia_data[np.argwhere(inside).squeeze()]
    gaia_data = gaia_data.to_pandas()
    gaia_data = gaia_data.iloc[0:n]
    gaias = gaia_data[["x", "y"]].values

    if isinstance(align, int):
        stars = SegmentedPeaks(n_stars=align)(image.data)[0]
        X = twirl.find_transform(gaias[0:align], stars, n=align)
        gaias = twirl.affine_transform(X)(gaias)

    w, h = image.shape
    if np.abs(np.mean(gaia_data["x"])) > w or np.abs(np.mean(gaia_data["y"])) > h:
        warnings.warn("Catalog stars seem out of the field. Check that your stack is solved and that telescope "
                      "'ra_unit' and 'dec_unit' are well set")

    return gaias
