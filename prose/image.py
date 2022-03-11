from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
#from .blocks.utils import register_args
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle
from dateutil import parser as dparser
from astropy.wcs import WCS
from . import viz, utils, Telescope
from .utils import easy_median
from .console_utils import info

from astropy.io import fits
from datetime import timedelta

class Image:

    def __init__(self, fitspath=None, data=None, header=None, **kwargs):
        if fitspath is not None:
            self.path = fitspath
            self.get_data_header()
        else:
            self.data = data
            self.header = header if header is not None else {}
            self.path = None

        self.telescope = None
        self.discard = False
        self.__dict__.update(kwargs)
        self.check_telescope()
        self.catalogs = {}

    def get_data_header(self):
        self.data = fits.getdata(self.path).astype(float)
        self.header = fits.getheader(self.path)

    def copy(self, data=True):
        new_self = self.__class__(**self.__dict__)
        if not data:
            del new_self.__dict__["data"]

        return new_self

    def check_telescope(self):
        if self.header:
           self.telescope = Telescope.from_names(self.header.get("INSTRUME", ""), self.header.get("TELESCOP", ""))

    def get(self, keyword, default=None):
        return self.header.get(keyword, default)

    @property
    def wcs(self):
        return WCS(self.header)

    @wcs.setter
    def wcs(self, new_wcs):
        self.header.update(new_wcs.to_header())

    @property
    def exposure(self):
        return self.get(self.telescope.keyword_exposure_time, None)

    @property
    def jd_utc(self):
        # if jd keyword not in header compute jd from date
        if self.telescope.keyword_jd in self.header:
            jd = self.get(self.telescope.keyword_jd, None) + self.telescope.mjd
        else:
            jd = Time(self.date, scale="utc").to_value('jd') + self.telescope.mjd

        return Time(
            jd,
            format="jd",
            scale=self.telescope.jd_scale,
            location=self.telescope.earth_location).utc.value

    @property
    def bjd_tdb(self):
        jd_bjd = self.get(self.telescope.keyword_bjd, None)
        if jd_bjd is not None:
            jd_bjd += self.telescope.mjd

            if self.telescope.keyword_jd in self.header:
                time_format = "bjd"
            else:
                time_format = "jd"

            return Time(jd_bjd,
                        format=time_format,
                        scale=self.telescope.jd_scale,
                        location=self.telescope.earth_location).tdb.value

        else:
            return None

    @property
    def seeing(self):
        return self.get(self.telescope.keyword_seeing, None)

    @property
    def ra(self):
        _ra = self.get(self.telescope.keyword_ra, None)
        if _ra is not None:
            _ra = Angle(_ra, self.telescope.ra_unit).to(u.deg)
        return _ra

    @property
    def dec(self):
        _dec = self.get(self.telescope.keyword_dec, None)
        if _dec is not None:
            _dec = Angle(_dec, self.telescope.dec_unit).to(u.deg)
        return _dec

    @property
    def flip(self):
        return self.get(self.telescope.keyword_flip, None)

    @property
    def airmass(self):
        return self.get(self.telescope.keyword_airmass, None)

    @property
    def shape(self):
        return np.array(self.data.shape)

    @property
    def date(self):
        return dparser.parse(self.header[self.telescope.keyword_observation_date])

    @property
    def night_date(self):
        return (dparser.parse(self.header[self.telescope.keyword_observation_date]) - timedelta(hours=15)).date()

    @property
    def label(self):
        return "_".join([
            self.telescope.name,
            self.night_date.strftime("%Y%m%d"),
            self.header.get(self.telescope.keyword_object, "?"),
            self.filter
        ])

    @property
    def filter(self):
        return self.header.get(self.telescope.keyword_filter, None)
    
    def show(self, 
        cmap="Greys_r", 
        ax=None, 
        figsize=(10,10), 
        stars=None, 
        stars_labels=True, 
        vmin=True, 
        vmax=None, 
        scale=1.5,
        frame=False
        ):
        if ax is None:
            if not isinstance(figsize, (list, tuple)):
                if isinstance(figsize, (float, int)):
                    figsize = (figsize, figsize)
                else:
                    raise TypeError("figsize must be tuple or list or float or int")
            fig = plt.figure(figsize=figsize)
            if frame:
                ax = fig.add_subplot(111, projection=self.wcs)
            else:
                ax = fig.add_subplot(111)

        if vmin is True or vmax is True:
            med = np.nanmedian(self.data)
            vmin = med
            vmax = scale*np.nanstd(self.data) + med
            _ = ax.imshow(self.data, cmap=cmap, origin="lower",vmin=vmin,vmax=vmax)
        elif all([vmin, vmax]) is False:
            _ = ax.imshow(utils.z_scale(self.data, 0.05*scale), cmap=cmap, origin="lower")
        else:
            _ = ax.imshow(self.data, cmap=cmap, origin="lower",vmin=vmin,vmax=vmax)
        
        if stars is None:
            stars = "stars_coords" in self.__dict__
        
        if stars:
            label = np.arange(len(self.stars_coords)) if stars_labels else None
            viz.plot_marks(*self.stars_coords.T, label=label, ax=ax)

        if frame:
            overlay = ax.get_coords_overlay(self.wcs)
            overlay.grid(color='white', ls='dotted')
            overlay[0].set_axislabel('Right Ascension (J2000)')
            overlay[1].set_axislabel('Declination (J2000)')

    def show_cutout(self, star=None, size=200, marks=True, **kwargs):
        """
        Show a zoomed cutout around a detected star or coordinates

        Parameters
        ----------
        star : [type], optional
            detected star id or (x, y) coordinate, by default None
        size : int, optional
            side size of square cutout in pixel, by default 200
        """

        if star is None:
            x, y = self.stars_coords[self.target]
        elif isinstance(star, int):
            x, y = self.stars_coords[star]
        elif isinstance(star, (tuple, list, np.ndarray)):
            x, y = star
        else:
            raise ValueError("star type not understood")

        self.show(**kwargs)
        plt.xlim(np.array([-size / 2, size / 2]) + x)
        plt.ylim(np.array([-size / 2, size / 2]) + y)
        if marks and hasattr(self, "stars_coords"):
            idxs = np.argwhere(np.max(np.abs(self.stars_coords - [x, y]), axis=1) < size).squeeze()
            viz.plot_marks(*self.stars_coords[idxs].T, label=idxs)

    @property
    def skycoord(self):
        """astropy SkyCoord object based on header RAn, DEC
        """
        return SkyCoord(self.ra, self.dec, frame='icrs')


    @property
    def fov(self):
        return np.array(self.shape) * self.pixel_scale.to(u.deg)

    @property
    def pixel_scale(self):
        return self.telescope.pixel_scale.to(u.arcsec)

    def plot_circle(self, center, arcmin=2.5):
        if isinstance(center, int):
            x, y = self.stars[center]
        elif isinstance(center, (tuple, list, np.ndarray)):
            x, y = center
        else:
            raise ValueError("center type not understood")

        search_radius = 60 * arcmin / self.telescope.pixel_scale
        circle = plt.Circle((x, y), search_radius, fill=None, ec="white", alpha=0.6)

        ax = plt.gca()
        ax.add_artist(circle)
        plt.annotate(f"radius {arcmin}'", xy=[x, y + search_radius + 15], color="white",
                     ha='center', fontsize=12, va='bottom', alpha=0.6)

    def plot_catalog(self, name, color="y", label=False, n=100000):
        assert name in self.catalogs, f"Catalog '{name}' not present, consider using ..."
        x, y = self.catalogs[name][["x", "y"]].values[0:n].T
        labels = self.catalogs[name]["id"].values if label else None
        viz.plot_marks(x, y, labels, color=color)

    @property
    def plate_solved(self):
        """Return wether the image is plate solved
        """
        return self.wcs.has_celestial