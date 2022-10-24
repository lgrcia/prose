import twirl
from .. import Block
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import numpy as np
from astropy.time import Time
import warnings
from ..utils import gaia_query, sparsify, register_args
from twirl.utils import plot as tplot
from .registration import cross_match
from astroquery.mast import Catalogs
from . import vizualisation as viz

def image_gaia_query(image, *args, limit=3000, correct_pm=True, wcs=True, circular=True, fov=None):
    
    if wcs:
        center = image.wcs.pixel_to_world(*(np.array(image.shape)/2)[::-1])
    else:
        center = image.skycoord

    if fov is None:
        fov = image.fov

    table = gaia_query(center, fov, "*", limit=limit, circular=circular)

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

class _CatalogBlock(Block):

    
    def __init__(self, name, mode=None, limit=10000, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.catalog_name = name
        self.limit = limit

    def get_catalog(self, image):
        raise NotImplementedError()
    
    def run(self, image):
        catalog = self.get_catalog(image)
        radecs = np.array([catalog["ra"].quantity.to(u.deg), catalog["dec"].quantity.to(u.deg)])
        stars_coords = np.array(SkyCoord(*radecs, unit="deg").to_pixel(image.wcs))
        catalog["x"], catalog["y"] = stars_coords
        catalog = catalog.to_pandas()
        image.catalogs[self.catalog_name] = catalog
        stars_coords = stars_coords.T
    
        if self.mode == "replace":
            mask = np.all(stars_coords < image.shape[::-1], 1) & np.all(stars_coords > 0, 1)
            mask = mask & ~np.any(np.isnan(stars_coords), 1)
            image.stars_coords = stars_coords[mask][0:self.limit]
            catalog = catalog.iloc[np.flatnonzero(mask)].reset_index()

        elif self.mode == "crossmatch":
            xys = catalog[["x", "y"]].values
            matches = cross_match(image.stars_coords, xys, return_ixds=True)
            catalog.loc[len(xys)] = [np.nan for _ in catalog.keys()]
            matches[np.isnan(matches)] = -1
            matches = matches.astype(int)
            catalog = catalog.iloc[matches.T[1]].reset_index()

        image.catalogs[self.catalog_name] = catalog.iloc[0:self.limit]
        
class PlateSolve(Block):
    
    
    def __init__(self, reference=None, n=30, tolerance=10, radius=1, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius * u.arcmin.to("deg")
        self.n = n
        self.reference = reference
        self.tolerance = tolerance
        self.debug = debug
    
    def run(self, image):
        stars = image.stars_coords*image.pixel_scale.to("deg").value
        stars = sparsify(stars, self.radius)/image.pixel_scale.to("deg").value

        if self.reference is None:
            table = image_gaia_query(image, wcs=False, circular=True, fov=image.fov.max()).to_pandas()
            gaias = np.array([table.ra, table.dec]).T
            gaias = gaias[~np.any(np.isnan(gaias), 1)]
        else:
            gaias = self.reference.catalogs["gaia"][["ra", "dec"]].values

        gaias = sparsify(gaias, self.radius)

        image.wcs = twirl._compute_wcs(stars[0:self.n], gaias[0:self.n], n=self.n, tolerance=self.tolerance)

        if self.debug:
            image.show(stars=False)
            _gaias = np.array(image.wcs.world_to_pixel(SkyCoord(gaias[0:self.n], unit="deg"))).T
            image.plot_marks(_gaias, ms=15, label="stars coordinates based on SkyCooord")



class GaiaCatalog(_CatalogBlock):
    
    
    def __init__(self, correct_pm=True, limit=10000, **kwargs):
        _CatalogBlock.__init__(self, "gaia", limit=limit, **kwargs)
        self.correct_pm = correct_pm

    def get_catalog(self, image):
        max_fov = image.fov.max()*np.sqrt(2)
        table =  image_gaia_query(
            image, 
            correct_pm=self.correct_pm, 
            limit=500000, 
            circular=True,
            fov=max_fov
        )
        table.rename_column('DESIGNATION', 'id')
        return table

    def run(self, image):
        _CatalogBlock.run(self, image)


class TESSCatalog(_CatalogBlock):
    
    
    def __init__(self, limit=10000, **kwargs):
        _CatalogBlock.__init__(self, "tess", limit=limit, **kwargs)

    def get_catalog(self, image):
        max_fov = image.fov.max()*np.sqrt(2)/2
        table = Catalogs.query_region(image.skycoord, max_fov, "TIC", verbose=False)
        table["ra"].unit = "deg"
        table["dec"].unit = "deg"
        table.rename_column('ID', 'id')
        return table

    def run(self, image):
        _CatalogBlock.run(self, image)