import twirl
from .. import Block
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import numpy as np
from astropy.time import Time
import warnings

def gaia_query(center, fov, *args, limit=10000):
    """
    https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
    """
    
    from astroquery.gaia import Gaia
    
    if isinstance(center, SkyCoord):
        ra = center.ra.to(u.deg).value
        dec = center.dec.to(u.deg).value
    
    if isinstance(fov, u.Quantity):
        if len(fov) == 2:
            ra_fov, dec_fov = fov.to(u.deg).value
        else:
            ra_fov = dec_fov = fov.to(u.deg).value

        radius = np.min([ra_fov, dec_fov])/2

    job = Gaia.launch_job(f"select top {limit} {','.join(args) if isinstance(args, (tuple, list)) else args} from gaiadr2.gaia_source where "
                          "1=CONTAINS("
                          f"POINT('ICRS', {ra}, {dec}), "
                          f"CIRCLE('ICRS',ra, dec, {radius}))"
                          "order by phot_g_mean_mag")

    return job.get_results()


def image_gaia_query(image, *args, limit=3000, correct_pm=True):

    table = gaia_query(image.skycoord, image.fov, "*", limit=limit)

    if correct_pm:
        skycoord = SkyCoord(
                ra=table['ra'].quantity, 
                dec=table['dec'].quantity,
                pm_ra_cosdec=table['pmra'].quantity,
                pm_dec=table['pmdec'].quantity, 
                distance=table['parallax'].quantity,
                obstime=Time('2015-06-01 00:00:00.0')
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skycoord = skycoord.apply_space_motion(Time(image.date))

        table["ra"] = skycoord.ra
        table["dec"] = skycoord.dec

    return table

class CatalogBlock(Block):

    def __init__(self, name, mode=None, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.catalog_name = name

    def get_catalog(self, image):
        raise NotImplementedError()
    
    def run(self, image):
        catalog = self.get_catalog(image)
        radecs = np.array([catalog["ra"].quantity.to(u.deg), catalog["dec"].quantity.to(u.deg)])
        stars_coords = np.array(SkyCoord(*radecs, unit="deg").to_pixel(image.wcs))
        catalog["x"], catalog["y"] = stars_coords
    
        if self.mode == "replace":
            image.stars_coords = stars_coords.T[np.all(np.isfinite(stars_coords), 0)]

        elif self.mode == "crossmatch":
            x, y = catalog[["x", "y"]].values.T
            pass

        image.catalogs[self.catalog_name] = catalog.to_pandas()

        
class PlateSolve(Block):
    
    def __init__(self, ref_image=None, n_gaia=50, tolerance=10, n_twirl=15, **kwargs):
        super().__init__(**kwargs)
        self.n_gaia = n_gaia
        self.ref_image = ref_image is not None
        self.n_gaia = n_gaia
        self.n_twirl = n_twirl
        self.tolerance = tolerance
        
        if ref_image:
            self.ref_image = self(ref_image)
    
    def run(self, image):
        if not self.ref_image:
            table = image_gaia_query(image).to_pandas()
            self.gaias = np.array([table.ra, table.dec]).T
            self.gaias[np.any(np.isnan(self.gaias), 1)] = 0
            
        image.wcs = twirl._compute_wcs(image.stars_coords, self.gaias, n=self.n_twirl, tolerance=self.tolerance)


class GaiaCatalog(CatalogBlock):
    
    def __init__(self, n_stars=10000, tolerance=4, correct_pm=True, **kwargs):
        CatalogBlock.__init__(self, "gaia", **kwargs)
        self.n_stars = n_stars
        self.tolerance = tolerance
        self.correct_pm = correct_pm

    def get_catalog(self, image):
        table =  image_gaia_query(image, correct_pm=self.correct_pm, limit=self.n_stars)
        table.rename_column('DESIGNATION', 'id')
        return table

    def run(self, image):
        CatalogBlock.run(self, image)