import warnings

import astropy.units as u
import numpy as np
import pandas as pd
import twirl
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.mast import Catalogs

from prose import Block
from prose import visualization as viz
from prose.core.source import PointSource, Sources
from prose.utils import cross_match, gaia_query, sparsify

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

    def _get_catalog(self, image):
        raise NotImplementedError()

    def run(self, image):
        catalog = self._get_catalog(image)
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
            image.sources = Sources(
                [
                    PointSource(coords=s, i=i)
                    for i, s in enumerate(stars_coords[mask][0 : self.limit])
                ]
            )
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


# TODO
class PlateSolve(Block):
    def __init__(
        self,
        reference=None,
        n=15,
        tolerance=10,
        radius=1.2,
        debug=False,
        name=None,
        asterism=4,
    ):
        """Plate solve an image using twirl.

        This block uses the twirl package to plate solve an image. It matches the positions
        of stars in the image to positions of stars in a reference catalog
        (either Gaia or a user-provided catalog) and computes a new WCS for the image.

        Parameters
        ----------
        reference : Image, optional
            The reference image to use for matching stars. If None, the block will use Gaia
            DR2 as the reference catalog.
        n : int, optional
            The number of stars to use for matching. Default is 30.
        tolerance : int, optional
            The tolerance (in pixels) for matching stars. Default is 10.
        radius : float, optional
            The radius (in arcminutes) around each star to search for matches. Default is 1.2.
        debug : bool, optional
            Whether to print debug information. Default is False.
        """
        super().__init__(name=name)
        self.radius = radius * u.arcmin.to("deg")
        self.n = n
        self.reference = reference
        self.tolerance = tolerance
        self.debug = debug
        self.asterism = asterism

    def run(self, image):
        stars = image.sources.coords * image.pixel_scale.to("deg").value
        stars = sparsify(stars, self.radius) / image.pixel_scale.to("deg").value

        if self.reference is None:
            table = image_gaia_query(
                image, wcs=False, circular=True, fov=image.fov.max()
            ).to_pandas()
            gaias = np.array([table.ra, table.dec]).T
            gaias = gaias[~np.any(np.isnan(gaias), 1)]
        else:
            gaias = self.reference.catalogs["gaia"][["ra", "dec"]].values

        gaias = sparsify(gaias, self.radius)

        new_wcs = twirl.compute_wcs(
            stars[0 : self.n],
            gaias[0 : self.n],
            tolerance=self.tolerance,
            asterism=self.asterism,
        )
        image.wcs = new_wcs
        coords = np.array(image.wcs.world_to_pixel(SkyCoord(gaias, unit="deg"))).T
        idxs = cross_match(image.sources.coords, coords, return_idxs=True)
        image.computed["plat_solve_success"] = np.count_nonzero(
            ~np.isnan(idxs[:, 1])
        ) / len(gaias)

        if self.debug:
            image.show()
            coords = np.array(
                image.wcs.world_to_pixel(SkyCoord(gaias[0 : self.n], unit="deg"))
            ).T
            _gaias = Sources([PointSource(coords=c) for c in coords])
            _gaias.plot(c="y")

    @property
    def citations(self) -> list:
        return super().citations + ["twirl"]


class GaiaCatalog(_CatalogBlock):
    def __init__(self, correct_pm=True, limit=10000, mode=None):
        """Query gaia catalog.

        Catalog is written in Image.catalogs as a pandas DataFrame. If mode is ""crossmatch" the index of catalog sources in the DataFrame matches with the index of sources in Image.sources

        |read| :code:`Image.sources` if mode is "crossmatch"

        |write|

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

    def _get_catalog(self, image):
        max_fov = image.fov.max() * np.sqrt(2)
        table = image_gaia_query(
            image, correct_pm=self.correct_pm, limit=500000, circular=True, fov=max_fov
        )
        table.rename_column("DESIGNATION", "id")
        return table

    def run(self, image):
        _CatalogBlock.run(self, image)

    @property
    def citations(self) -> list:
        return super().citations + ["astroquery"]


class TESSCatalog(_CatalogBlock):
    def __init__(self, limit=10000, mode=None):
        """Query TESS (TIC) catalog

        Catalog is written in Image.catalogs as a pandas DataFrame. If mode is ""crossmatch" the index of catalog sources in the DataFrame matches with the index of sources in Image.sources

        |read| :code:`Image.sources` if mode is "crossmatch"

        |write|

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

    def _get_catalog(self, image):
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
    def citations(self) -> list:
        return super().citations + ["astroquery"]
