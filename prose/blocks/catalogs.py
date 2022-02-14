import twirl
from prose import Block
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import numpy as np

def gaia_query(center, fov, *args, limit=3000):
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

    job = Gaia.launch_job(f"select top {limit} {','.join(args)} from gaiadr2.gaia_source where "
                          "1=CONTAINS("
                          f"POINT('ICRS', {ra}, {dec}), "
                          f"CIRCLE('ICRS',ra, dec, {ra_fov/2}))"
                          #f"ra between {ra-ra_fov/2} and {ra+ra_fov/2} and "
                          #f"dec between {dec-dec_fov/2} and {dec+dec_fov/2}"
                          "order by phot_g_mean_mag")

    return job.get_results().to_pandas()


class GaiaBlock(Block):
    
    def __init__(self, n_stars=50, **kwargs):
        super().__init__(**kwargs)
        self.n_stars = n_stars
        self.gaia_table = None
    
    def get_table(self, image, radius, *args):
        self.gaia_table = gaia_query(image.skycoord, radius*2, *args, limit=self.n_stars)
        
class PlateSolve(GaiaBlock):
    
    def __init__(self, ref_image=None, n_stars=50, **kwargs):
        super().__init__(n_stars=n_stars, **kwargs)
        self.ref_image = ref_image is not None
        self.n_stars = n_stars
        
        if ref_image:
            self.get_radecs(ref_image)
            
    def get_radecs(self, image):
        self.get_table(image, image.fov/2, "ra", "dec")
        self.gaias = np.array([self.gaia_table.ra, self.gaia_table.dec]).T
    
    def run(self, image):
        if not self.ref_image:
            self.get_radecs(image)
            
        image.wcs = twirl._compute_wcs(image.stars_coords, self.gaias)

class GaiaCatalog(GaiaBlock):
    
    def __init__(self, n_stars=10000, tolerance=10, remove_gaias=False, **kwargs):
        super().__init__(n_stars=n_stars, **kwargs)
        self.n_stars = n_stars
        self.tolerance = tolerance
        self.remove_gaias = False #TODO
        
    def get_coords_id(self, image):
        self.get_table(image, np.sqrt(2)*image.fov/2, "ra", "dec", "designation")
        self.gaias = np.array([self.gaia_table.ra, self.gaia_table.dec]).T
        self.designation = np.array(self.gaia_table.designation.values, dtype='str')
    
    def run(self, image):
        self.get_coords_id(image)
        gaias = np.array(SkyCoord(self.gaias, unit="deg").to_pixel(image.wcs)).T
        idxs = twirl.utils.cross_match(image.stars_coords, gaias, return_ixds=True, tolerance=self.tolerance)
        image.gaia_catalog = pd.DataFrame(index=range(len(image.stars_coords)), 
                                          columns=["designation", "match_distance", "ra(deg)", "dec(deg)", "x", "y"])
        for star, gaia_star in idxs:
            image.gaia_catalog["designation"].at[star] = self.designation[gaia_star]
            image.gaia_catalog["ra(deg)"].at[star] = self.gaias[gaia_star][0]
            image.gaia_catalog["dec(deg)"].at[star] = self.gaias[gaia_star][1]
            image.gaia_catalog["match_distance"].at[star] = np.linalg.norm(gaias[gaia_star] - image.stars_coords[star])
            image.gaia_catalog["x"].at[star] = image.stars_coords[star][0]
            image.gaia_catalog["y"].at[star] = image.stars_coords[star][1]