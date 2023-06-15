import warnings

import astropy.units as u
import numpy as np
import pandas as pd
import twirl
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.mast import Catalogs
from twirl.geometry import sparsify

from prose import Block
from prose import visualization as viz
from prose.core.source import PointSource, Sources
from prose.utils import cross_match, gaia_query

__all__ = ["GaiaCatalog", "TESSCatalog"]


def image_gaia_query(
    image, limit=3000, correct_pm=True, wcs=True, circular=True, fov=None
):
    if wcs:
        center = image.wcs.pixel_to_world(*(np.array(image.shape) / 2)[::-1])
    else:
        center = image.skycoord

    if fov is None:
        fov = image.fov

    table = gaia_query(center, fov, "*", limit=limit, circular=circular)

    if correct_pm:
        skycoord = SkyCoord(
            ra=table["ra"].quantity,
            dec=table["dec"].quantity,
            pm_ra_cosdec=table["pmra"].quantity,
            pm_dec=table["pmdec"].quantity,
            distance=table["parallax"].quantity,
            obstime=Time("2015-06-01 00:00:00.0"),
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
        radecs = np.array(
            [catalog["ra"].quantity.to(u.deg), catalog["dec"].quantity.to(u.deg)]
        )
        stars_coords = np.array(SkyCoord(*radecs, unit="deg").to_pixel(image.wcs))
        catalog["x"], catalog["y"] = stars_coords
        catalog = catalog.to_pandas()
        image.catalogs[self.catalog_name] = catalog
        stars_coords = stars_coords.T

        if self.mode == "replace":
            mask = np.all(stars_coords < image.shape[::-1], 1) & np.all(
                stars_coords > 0, 1
            )
            mask = mask & ~np.any(np.isnan(stars_coords), 1)
            image.sources = Sources(stars_coords[mask][0 : self.limit])
            catalog = catalog.iloc[np.flatnonzero(mask)].reset_index()

        elif self.mode == "crossmatch":
            coords_1 = image.sources.coords
            coords_2 = catalog[["x", "y"]].values

            _coords_2 = dict(zip(range(len(coords_2)), coords_2))
            tolerance = 10
            matches = {}

            for i, coords in enumerate(coords_1):
                idxs, available_coords = list(zip(*_coords_2.items()))
                distances = np.linalg.norm(available_coords - coords, axis=1)
                if np.all(np.isnan(available_coords)):
                    break
                closest = np.nanargmin(distances)
                if distances[closest] < tolerance:
                    matches[i] = idxs[closest]
                    del _coords_2[idxs[closest]]
                else:
                    matches[i] = None

            new_df_dict = []
            nans = {name: np.nan for name in image.catalogs[self.catalog_name].keys()}

            for i in range(len(coords_1)):
                if matches[i] is not None:
                    new_df_dict.append(
                        dict(image.catalogs[self.catalog_name].iloc[int(matches[i])])
                    )
                    pass
                else:
                    new_df_dict.append(nans)

            catalog = pd.DataFrame(new_df_dict)

        image.catalogs[self.catalog_name] = catalog.iloc[0 : self.limit]


class PlateSolve(Block):
    """
    A block that performs plate solving on an astronomical image using a Gaia catalog.

    Parameters
    ----------
    reference : `None` or `~prose.Image`
        A reference image containing a Gaia catalog to use for plate solving.
        If `None`, a new catalog will be queried using `image_gaia_query`.
        Default is `None`.
    n : int
        The number of stars from catalog to use for plate solving.
        Default is 30.
    tolerance : float, optional
        The minimum distance between two coordinates to be considered cross-matched
        (in `pixels` units). This serves to compute the number of coordinates being
        matched between `radecs` and `pixels` for a given transform.
        By default 12.
    radius : `None` or `~astropy.units.Quantity`
        The search radius (in degrees) for the Gaia catalog.
        If `None`, the radius will be set to 1/12th of the maximum field of view of the image.
        Default is `None`.
    debug : bool
        If `True`, the image and the matched stars will be plotted for debugging purposes.
        Default is `False`.
    quads_tolerance : float, optional
        The minimum euclidean distance between two quads to be matched and tested.
        By default 0.1.
    field : float
        The field of view to use for the Gaia catalog query, in fraction of the image field of view.
        Default is 1.2.
    min_match : float, optional
        The fraction of `pixels` coordinates that must be matched to stop the search.
        I.e., if the number of matched points is `>= min_match * len(pixels)`, the
        search stops and return the found transform. By default 0.7.
    name : str
        The name of the block.
        Default is `None`.
    """

    def __init__(
        self,
        reference=None,
        n=30,
        tolerance=10,
        radius=None,
        debug=False,
        quads_tolerance=0.1,
        field=1.2,
        min_match=0.8,
        name=None,
    ):
        super().__init__(name=name)
        self.radius = radius
        self.n = n
        self.reference = reference
        self.tolerance = tolerance
        self.quads_tolerance = quads_tolerance
        self.debug = debug
        self.field = field
        self.min_match = min_match

    def run(self, image):
        radius = image.fov.max() / 12 if self.radius is None else self.radius
        stars = image.sources.coords * image.pixel_scale.to("arcmin").value
        stars = (
            sparsify(stars, radius.to("arcmin").value)
            / image.pixel_scale.to("arcmin").value
        )

        if self.reference is None:
            table = image_gaia_query(
                image, wcs=False, circular=True, fov=image.fov.max() * self.field
            ).to_pandas()
            gaias = np.array([table.ra, table.dec]).T
            gaias = gaias[~np.any(np.isnan(gaias), 1)]
        else:
            gaias = self.reference.catalogs["gaia"][["ra", "dec"]].values

        gaias = sparsify(gaias, radius.to("deg").value)

        new_wcs = twirl.compute_wcs(
            stars,
            gaias[0 : self.n],
            tolerance=self.tolerance,
            quads_tolerance=self.quads_tolerance,
            min_match=self.min_match,
        )
        image.wcs = new_wcs
        coords = np.array(image.wcs.world_to_pixel(SkyCoord(gaias, unit="deg"))).T
        idxs = cross_match(image.sources.coords, coords, return_idxs=True)
        image.computed["plat_solve_success"] = np.count_nonzero(
            ~np.isnan(idxs[:, 1])
        ) / len(gaias)

        if self.debug:
            image.show()
            coords = np.array(image.wcs.world_to_pixel(SkyCoord(gaias, unit="deg"))).T
            _gaias = Sources([PointSource(coords=c) for c in coords])
            _gaias.plot(c="y")

    @property
    def citations(self):
        return super().citations + ["twirl"]


class GaiaCatalog(_CatalogBlock):
    def __init__(self, correct_pm=True, limit=10000, mode=None):
        """Query gaia catalog

        Catalog is written in Image.catalogs as a pandas DataFrame. If mode is ""crossmatch" the index of catalog sources in the DataFrame matches with the index of sources in Image.sources

        |read| :code:`Image.sources` if mode is "crossmatch"

        |write| :code:`Image.catalogs`

        - :code:`Image.sources` if mode is "crossmatch"
        - :code:`Image.catalogs`

        Parameters
        ----------
        correct_pm : bool, optional
            whether to correct proper motion, by default True
        limit : int, optional
            limit number of stars queried, by default 10000
        mode: str, optional
            "crossmatch" to match existing Image.sources or "replace" to use queried stars as Image.sources
        """
        _CatalogBlock.__init__(self, "gaia", limit=limit, mode=mode)
        self.correct_pm = correct_pm

    def get_catalog(self, image):
        max_fov = image.fov.max() * np.sqrt(2)
        table = image_gaia_query(
            image, correct_pm=self.correct_pm, limit=500000, circular=True, fov=max_fov
        )
        table.rename_column("DESIGNATION", "id")
        return table

    def run(self, image):
        _CatalogBlock.run(self, image)

    @property
    def citations(self):
        return super().citations + ["astroquery"]


class TESSCatalog(_CatalogBlock):
    def __init__(self, limit=10000, mode=None):
        """Query TESS (TIC) catalog

        Catalog is written in Image.catalogs as a pandas DataFrame. If mode is ""crossmatch" the index of catalog sources in the DataFrame matches with the index of sources in Image.sources

        |read| :code:`Image.sources` if mode is "crossmatch"

        |write| :code:`Image.catalogs`

        - :code:`Image.sources` if mode is "crossmatch"
        - :code:`Image.catalogs`


        Parameters
        ----------
        limit : int, optional
            limit number of stars queried, by default 10000
        mode: str, optional
            "crossmatch" to match existing Image.sources or "replace" to use queried stars as Image.sources
        """
        _CatalogBlock.__init__(self, "tess", limit=limit, mode=mode)

    def get_catalog(self, image):
        max_fov = image.fov.max() * np.sqrt(2) / 2
        table = Catalogs.query_region(
            image.skycoord, radius=max_fov, catalog="TIC", verbose=False
        )
        table["ra"].unit = "deg"
        table["dec"].unit = "deg"
        table.rename_column("ID", "id")
        return table

    def run(self, image):
        _CatalogBlock.run(self, image)

    @property
    def citations(self):
        return super().citations + ["astroquery"]
