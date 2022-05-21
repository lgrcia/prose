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
    r"""Base object containing FITS image data and metadata

    When a FITS path (or header) is provided, keyword values are used to identify and instantiate a :py:class:`~prose.Telescope` object. Image attributes then use this object to retrieve specific image information such as ra, dec, untis... etc

    Parameters
    ----------
    fitspath : str or Path, optional
        file path , by default None
    data : numpy.ndarray, optional
        image data, by default None
    header : dict-like, optional
        image metadata, by default None 

    Example
    -------

    .. jupyter-execute::

        from prose.tutorials import image_sample

        # loading and showing an example image
        image = image_sample("05 38 44.851", "+04 32 47.68")
        image.show()

    .. jupyter-execute::
        
        image.header[0:10] # the 10 first lines

    Once this object is instantiated, its parameters are mapped to the ones of the telescope, detected from the header information. This exposes conveniant attributres, for example:

    .. jupyter-execute::

        print(f"pixel scale : {image.pixel_scale:.2f}\nFOV: {image.fov}\nnight: {image.night_date}\n")

    some of them being directly translated into astropy Quantity or datetime object.

    """

    def __init__(self, fitspath=None, data=None, header=None, **kwargs):
        """
        Image instanciation
        """
        if fitspath is not None:
            self.path = fitspath
            self._get_data_header()
        else:
            self.data = data
            self.header = header if header is not None else {}
            self.path = None

        self.telescope = None
        self.discard = False
        self.__dict__.update(kwargs)
        self._check_telescope()
        self.catalogs = {}

    def _get_data_header(self):
        """Retrieve data and metadata from an image
        """
        self.data = fits.getdata(self.path).astype(float)
        self.header = fits.getheader(self.path)

    def copy(self, data=True):
        """Copy of image object

        Parameters
        ----------
        data : bool, optional
            whether to copy data, by default True

        Returns
        -------
        Image
            copied object
        """
        new_self = self.__class__(**self.__dict__)
        if not data:
            del new_self.__dict__["data"]

        return new_self

    def _check_telescope(self):
        """Instantiate ``self.telescope`` from ``INSTRUME`` and or ``TELESCOP`` keywords
        """
        if self.header:
           self.telescope = Telescope.from_names(self.header.get("INSTRUME", ""), self.header.get("TELESCOP", ""))

    def get(self, key, default=None):
        """Return corresponding value from header, similar to ``dict.get``

        Parameters
        ----------
        key : str
        default : any, optional
            value to return if key not in ``Image.header``, by default None
        """
        return self.header.get(key, default)

    @property
    def wcs(self):
        """astropy.wcs.WCS object associated with the FITS ``Image.header``
        """
        return WCS(self.header)

    @wcs.setter
    def wcs(self, new_wcs):
        self.header.update(new_wcs.to_header())

    @property
    def exposure(self):
        """Image exposure time in seconds

        Returns
        -------
        astropy.units.quantity.Quantity
        """
        return self.get(self.telescope.keyword_exposure_time, None) * u.s

    @property
    def jd_utc(self):
        """JD UTC time of the observation

        Returns
        -------
        astropy.time.Time
        """
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
        """BJD TDB time of the observation

        Returns
        -------
        astropy.time.Time
        """
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
        """Seeing of the image as written is header
        """
        return self.get(self.telescope.keyword_seeing, None)

    @property
    def ra(self):
        """RA of the image as written in header

        Returns
        -------
        astropy.coordinates.angles.Angle
        """
        _ra = self.get(self.telescope.keyword_ra, None)
        if _ra is not None:
            _ra = Angle(_ra, self.telescope.ra_unit).to(u.deg)
        return _ra

    @property
    def dec(self):
        """DEC of the image as written in header

        Returns
        -------
        astropy.coordinates.angles.Angle
        """
        _dec = self.get(self.telescope.keyword_dec, None)
        if _dec is not None:
            _dec = Angle(_dec, self.telescope.dec_unit).to(u.deg)
        return _dec

    @property
    def flip(self):
        """Telescope flip as written in image header
        """
        return self.get(self.telescope.keyword_flip, None)

    @property
    def airmass(self):
        """Observation airmass as written in image header
        """
        return self.get(self.telescope.keyword_airmass, None)

    @property
    def shape(self):
        """Shape of the image data np.ndarray

        Returns
        -------
        2D tuple
        """
        return np.array(self.data.shape)

    @property
    def date(self):
        """datetime of the observation

        Returns
        -------
        datetime.datetime
        """
        return dparser.parse(self.header[self.telescope.keyword_observation_date])

    @property
    def night_date(self):
        """date of the night when night started.

        Returns
        -------
        datetime.date
        """
        # TODO: do according to last astronomical twilight?
        return (dparser.parse(self.header[self.telescope.keyword_observation_date]) - timedelta(hours=15)).date()

    @property
    def label(self):
        """A conveniant {Telescope}_{Date}_{Object}_{Filter} string

        Returns
        -------
        str
        """
        return "_".join([
            self.telescope.name,
            self.night_date.strftime("%Y%m%d"),
            self.header.get(self.telescope.keyword_object, "?"),
            self.filter
        ])

    @property
    def filter(self):
        """Observation filter as written in image header
        """
        return self.header.get(self.telescope.keyword_filter, None)
    
    def show(self, 
        cmap="Greys_r", 
        ax=None, 
        figsize=(10,10), 
        stars=None, 
        stars_labels=True, 
        zscale=True,
        frame=False,
        contrast=0.1,
        **kwargs
        ):
        """Show image data

        Parameters
        ----------
        cmap : str, optional
            matplotlib colormap, by default "Greys_r"
        ax : subplot, optional
            matplotlbib Axes in which to plot, by default None
        figsize : tuple, optional
            matplotlib figure size if ax not sepcified, by default (10,10)
        stars : bool, optional
            whether to show ``Image.stars_coords``, by default None
        stars_labels : bool, optional
            whether top show stars indexes, by default True
        zscale : bool, optional
            whether to apply a z scale to plotted image data, by default False
        frame : bool, optional
            whether to show astronomical coordinates axes, by default False
        contrast : float, optional
            image contrast used in image scaling, by default 0.1

        See also
        --------
        show_cutout :
            Show a specific star cutout
        plot_catalog :
            Plot catalog stars on an image
        plot_circle :
            Plot circle with radius in astronomical units  
        """
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

        if zscale is False:
            vmin = np.nanmedian(self.data)
            vmax = vmax = vmin*(1+contrast)/(1-contrast)
            _ = ax.imshow(self.data, cmap=cmap, origin="lower",vmin=vmin,vmax=vmax, **kwargs)
        else:
            _ = ax.imshow(utils.z_scale(self.data, contrast), cmap=cmap, origin="lower", **kwargs)
        
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

    def show_cutout(self, star=None, size=200, **kwargs):
        """Show a zoomed cutout around a detected star or coordinates

        Parameters
        ----------
        star : [type], optional
            detected star id or (x, y) coordinate, by default None
        size : int, optional
            side size of square cutout in pixel, by default 200
        **kwargs passed to self.show
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
        stars = kwargs.get("stars", False)
        if stars and hasattr(self, "stars_coords"):
            idxs = np.argwhere(np.max(np.abs(self.stars_coords - [x, y]), axis=1) < size).squeeze()
            viz.plot_marks(*self.stars_coords[idxs].T, label=idxs)

    @property
    def skycoord(self):
        """astropy SkyCoord object based on header RAn, DEC
        """
        return SkyCoord(self.ra, self.dec, frame='icrs')


    @property
    def fov(self):
        """Field of view of the image in degrees

        Returns
        -------
        astropy.units Quantity
        """
        return np.array(self.shape) * self.pixel_scale.to(u.deg)

    @property
    def pixel_scale(self):
        """Pixel scale (or plate scale) in arcseconds

        Returns
        -------
        astropy.units Quantity
        """
        return self.telescope.pixel_scale.to(u.arcsec)

    def plot_aperture(self, center, arcmin=2.5):
        """Plot an aperture with radius defined in arcmin

        must be over :py:class:`Image.show` or :py:class:`Image.show_cutout` plot)

        Parameters
        ----------
        center : int or tuple-like
            center of the aperture
            - if int: index of ``Image.stars_coords`` serving as center
        arcmin : float, optional
            radius of the aperture in arcminutes, by default 2.5
        """
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
        """Plot catalog stars 
        
        must be over :py:class:`Image.show` or :py:class:`Image.show_cutout` plot

        Parameters
        ----------
        name : str
            catalog name as stored in :py:class:Image.catalog`
        color : str, optional
            color of stars markers, by default "y"
        label : bool, optional
            whether to show stars catalogs ids, by default False
        n : int, optional
            number of brightest catalog stars to show, by default 100000
        """
        assert name in self.catalogs, f"Catalog '{name}' not present, consider using ..."
        x, y = self.catalogs[name][["x", "y"]].values[0:n].T
        labels = self.catalogs[name]["id"].values if label else None
        viz.plot_marks(x, y, labels, color=color)

    @property
    def plate_solved(self):
        """Return whether the image is plate solved
        """
        return self.wcs.has_celestial

    def writeto(self, destination):
        hdu = fits.PrimaryHDU(data=self.data, header=fits.Header(utils.clean_header(self.header)))
        hdu.writeto(destination, overwrite=True)

