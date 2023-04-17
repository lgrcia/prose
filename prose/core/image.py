import pickle
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Union

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.io.fits.hdu.base import _BaseHDU
from astropy.nddata import Cutout2D as astopy_Cutout2D
from astropy.nddata import overlap_slices
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.wcs import WCS
from dateutil import parser as dparser
from matplotlib import gridspec
from PIL import Image

from prose import utils, viz
from prose.core.source import Sources
from prose.telescope import Telescope


@dataclass
class Image:
    """
    Image object containing image data and metadata

    This is a Python Data Class, so that most attributes described below can be used as
    keyword-arguments when instantiated
    """

    data: np.ndarray = None
    """Image data"""

    metadata: dict = None
    """Image metadata"""

    catalogs: dict = None
    """Catalogs associated with the image contained in a dictionary of 
    pandas dataframes"""

    _sources: Union[Sources, dict] = None

    origin: tuple = (0, 0)
    """Image origin"""

    discard: bool = False
    """Whether image as been discarded by a block"""

    computed: dict = None
    """A dictionary containing any user and block-defined attributes"""

    _wcs = None

    def __post_init__(self):
        assert (
            isinstance(self.data, np.ndarray) or self.data is None
        ), f"data must be a np.ndarray, not {type(self.data)}"
        if self.metadata is None:
            self.metadata = {}
        if self.catalogs is None:
            self.catalogs = {}
        if self.computed is None:
            self.computed = {}
        if isinstance(self._sources, dict):
            self._sources = Sources(**self._sources)
        if self._sources is None:
            self._sources = Sources([])

    def __setattr__(self, name, value):
        if hasattr(self, name):
            super().__setattr__(name, value)
        else:
            if "computed" in self.__dict__:
                self.computed[name] = value
            else:
                super().__setattr__(name, value)

    def __getattr__(self, name):
        if "computed" not in self.__dict__:
            super.__getattr__(self, name)
        else:
            if name in self.computed:
                return self.computed[name]
            else:
                raise AttributeError(f"{name} cannot be interpreted as Image attribute")

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
        new_self = deepcopy(self)
        return new_self

    def __copy__(self):
        return self.copy()

    def show(
        self,
        cmap="Greys_r",
        ax=None,
        figsize=8,
        zscale=True,
        frame=False,
        contrast=0.1,
        sources=True,
        **kwargs,
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
        ms: int
            stars markers size
        ft: int
            stars label font size

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
            vmax = vmax = vmin * (1 + contrast) / (1 - contrast)
            _ = ax.imshow(
                self.data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, **kwargs
            )
        else:
            _ = ax.imshow(
                utils.z_scale(self.data, contrast), cmap=cmap, origin="lower", **kwargs
            )

        if frame:
            overlay = ax.get_coords_overlay(self.wcs)
            overlay.grid(color="white", ls="dotted")
            overlay[0].set_axislabel("Right Ascension (J2000)")
            overlay[1].set_axislabel("Declination (J2000)")

        if sources:
            if self.sources is not None:
                self.sources.plot()

        ax.set_xlim(0, self.shape[1] - 1)
        ax.set_ylim(0, self.shape[0] - 1)
        self._wcs = None

    def _from_metadata_with_unit(self, name):
        unit_name = f"{name}_unit"
        value = self.metadata[name]
        unit = str_to_astropy_unit(self.metadata[unit_name])
        if name in ["ra", "dec"]:
            return Angle(value, unit).to(u.deg)
        else:
            return value * unit
            ""

    @property
    def shape(self):
        return np.array(self.data.shape)

    @property
    def ra(self):
        """Right-Ascension as an astropy Quantity"""
        return self._from_metadata_with_unit("ra")

    @property
    def dec(self):
        """Declination as an astropy Quantity"""
        return self._from_metadata_with_unit("dec")

    @property
    def exposure(self):
        """Exposure time as an astropy Quantity"""
        return self._from_metadata_with_unit("exposure")

    @property
    def jd(self):
        return self.metadata["jd"]

    @property
    def pixel_scale(self):
        """Pixel scale (or plate scale) as an astropy Quantity"""
        return self._from_metadata_with_unit("pixel_scale")

    @property
    def filter(self):
        """Filter name"""
        return self.metadata["filter"]

    @property
    def fov(self):
        """RA-DEC field of view of the image in degrees

        Returns
        -------
        astropy.units Quantity
        """
        return np.array(self.shape)[::-1] * self.pixel_scale.to(u.deg)

    @property
    def date(self):
        """datetime of the observation

        Returns
        -------
        datetime.datetime
        """
        return dparser.parse(self.metadata["date"])

    @property
    def night_date(self):
        """date of the night when night started.

        Returns
        -------
        datetime.date
        """
        # TODO: do according to last astronomical twilight?
        return (self.date - timedelta(hours=15)).date()

    def set(self, name: str, value):
        """Set a computed value

        Parameters
        ----------
        name : str
            name of the computed value
        value : any
            value to set
        """
        self.computed[name] = value

    def get(self, name):
        return self.computed[name]

    @property
    def sources(self):
        """Image sources

        Returns
        -------
        prose.core.source.Sources
        """
        return self._sources

    @sources.setter
    def sources(self, new_sources):
        if isinstance(new_sources, Sources):
            self._sources = new_sources
        else:
            self._sources = Sources(np.array(new_sources))

    def cutout(self, coords, shape, wcs=True, sources=True, reset_index=True):
        """Return a list of Image cutouts from the image

        Parameters
        ----------
        coords : np.ndarray
            (N, 2) array of cutouts center coordinates
        shape : tuple or int
            The shape of the cutouts to extract. If int, shape is (shape, shape)
        wcs : bool, optional
            whether to compute and include cutouts WCS (takes more time), by default True
        reset_index: bool,
            whether to reset the sources indexes, by default True
        Returns
        -------
        list of Image
            image cutouts
        """
        if isinstance(shape, (int, float)):
            shape = (shape, shape)

        if isinstance(coords, int):
            coords = self.sources.coords[coords]

        new_image = astopy_Cutout2D(
            self.data,
            coords,
            shape,
            wcs=self.wcs if wcs else None,
            fill_value=np.nan,
            mode="partial",
        )

        # get sources
        new_sources = []
        if sources:
            if len(self._sources) > 0:
                sources_in = np.all(
                    np.abs(self.sources.coords - coords) < np.array(shape)[::-1] / 2, 1
                )
                _sources = self._sources[sources_in]

                for s in _sources:
                    _s = s.copy()
                    _s.coords = _s.coords - coords + np.array(shape)[::-1] / 2
                    new_sources.append(_s)

        image = Image(new_image.data, deepcopy(self.metadata), deepcopy(self.computed))
        image._sources = Sources(new_sources)
        image.wcs = new_image.wcs
        image.origin = tuple(np.array(new_image.bbox_original).T[0][::-1])
        image.catalogs = deepcopy(self.catalogs)
        for name, catalog in image.catalogs.items():
            image.catalogs[name][["x", "y"]] -= coords - np.array(shape) / 2
            # xy = catalog[["x", "y"]].values
            # idxs = np.all((xy - coords / 2) < np.array(shape) / 2, 1)
            # image.catalogs[name] = catalog[idxs]

        if reset_index:
            for i, s in enumerate(image.sources):
                s.i = i

        return image

    @property
    def wcs(self):
        """astropy.wcs.WCS object associated with the FITS ``Image.header``"""
        if self._wcs is None:
            self._wcs = WCS(self.metadata.get("wcs", None))
        return self._wcs

    @wcs.setter
    def wcs(self, new_wcs):
        if new_wcs is not None:
            if isinstance(new_wcs, WCS):
                self.metadata["wcs"] = new_wcs.to_header().tostring()
                self._wcs = new_wcs

    @property
    def plate_solved(self):
        """Return whether the image is plate solved"""
        return self.wcs.has_celestial

    def writeto(self, destination: Union[str, Path]):
        """Write image to FITS file

        Parameters
        ----------
        destination : Union[str, Path]
            destination path
        """
        hdu = fits.PrimaryHDU(
            data=self.data, header=fits.Header(utils.clean_header(self.header))
        )
        hdu.writeto(destination, overwrite=True)

    @property
    def skycoord(self):
        """astropy SkyCoord object based on header RAn, DEC"""
        return SkyCoord(self.ra, self.dec, frame="icrs")

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
        assert (
            name in self.catalogs
        ), f"Catalog '{name}' not present, consider using ..."
        x, y = self.catalogs[name][["x", "y"]].values.T
        labels = self.catalogs[name]["id"].values if label else None
        viz.plot_marks(x, y, labels, color=color)

    def plot_model(self, data, figsize=(5, 5), cmap=None, c="C0", contour=False):
        plt.figure(figsize=figsize)
        axes = gridspec.GridSpec(2, 2, width_ratios=[9, 2], height_ratios=[2, 9])
        axes.update(wspace=0, hspace=0)

        # axtt = plt.subplot(gs[1, 1])
        ax = plt.subplot(axes[1, 0])
        axr = plt.subplot(axes[1, 1], sharey=ax)
        axt = plt.subplot(axes[0, 0], sharex=ax)

        ax.imshow(self.data, alpha=1, cmap=cmap, origin="lower")
        if contour:
            ax.contour(data, colors="w", alpha=0.7)

        x, y = np.indices(data.shape)

        axt.plot(y[0], np.mean(self.data, axis=0), c=c, label="data")
        axt.plot(y[0], np.mean(data, axis=0), "--", c="k", label="model")
        axt.axis("off")
        axt.legend()

        axr.plot(np.mean(self.data, axis=1), y[0], c=c)
        axr.plot(np.mean(data, axis=1), y[0], "--", c="k")
        axr.axis("off")

    def asdict(self, image_dtype="int16", low_data=True):
        im_dict = asdict(self.copy())
        if low_data:
            im_dict["data"] = utils.z_scale(im_dict["data"]) * (2**7 - 1)
            image_dtype = "int8"
        im_dict["data"] = im_dict["data"].astype(image_dtype)
        return im_dict

    def save(self, filepath, image_dtype="int16", low_data=True):
        with open(filepath, "wb") as f:
            pickle.dump(self.asdict(image_dtype=image_dtype, low_data=low_data), f)

    @classmethod
    def load(cls, filepath):
        return cls(**pickle.load(open(filepath, "rb")))

    def symetric_profile(self, source, binn=1.0):
        x, y = source.coords
        Y, X = np.indices(self.shape)
        radii = (np.sqrt((X - x) ** 2 + (Y - y) ** 2)).flatten()
        d, values = self.profile(radii)
        idxs = utils.index_binning(d, binn)
        mean = lambda x: np.array([np.mean(x[i]) for i in idxs])
        return mean(d), mean(values)

    def profile(self, d):
        idxs = np.argsort(d)
        _d = d[idxs]
        pixels = self.data.flatten()
        pixels = pixels[idxs]

        return _d, pixels

    def data_cutouts(self, sources, shape):
        if isinstance(sources, Sources):
            sources = sources.coords

        cutouts = []
        for x, y in sources:
            c = np.zeros(shape)
            large, small = overlap_slices(self.shape, shape, (y, x))
            c[small] = self.data[large]
            cutouts.append(c)

        return np.array(cutouts)

    def major_profile(self, source, binn=1.0, debug=False):
        p1 = source.coords[:, None, None]
        p2 = (source.vertexes[0])[:, None, None]
        Y, X = np.indices(self.data.shape)
        p3 = np.array([X, Y])

        # projection
        # https://stackoverflow.com/questions/61341712/calculate-projected-point-location-x-y-on-given-line-startx-y-endx-y
        l2 = np.sum((p1 - p2) ** 2)
        assert l2 != 0, "p1 and p2 are the same points"
        distances = np.sum((p3 - p1) * (p2 - p1), 0) / np.sqrt(l2)
        flat_distance = distances.flatten()
        idxs = utils.index_binning(flat_distance, binn)
        distance = np.array([flat_distance[i].mean() for i in idxs])
        values = np.array([np.nanmax(self.data.flatten()[i]) for i in idxs])

        if debug:
            D = np.zeros(self.data.flatten().shape)
            for i, j in enumerate(idxs):
                D[j] = i
            plt.figure()
            plt.imshow(np.reshape(D, self.shape), origin="lower")

        return distance, values

    @property
    def label(self):
        """A conveniant {Telescope}_{Date}_{Object}_{Filter} string

        Returns
        -------
        str
        """
        return "_".join(
            [
                self.metadata["telescope"],
                self.night_date.strftime("%Y%m%d"),
                self.metadata["object"],
                self.filter,
            ]
        )


def str_to_astropy_unit(unit_string):
    return u.__dict__[unit_string]


def FITSImage(
    filepath_or_hdu: Union[str, Path, _BaseHDU],
    verbose: bool = False,
    load_units: bool = True,
    load_data: bool = True,
    telescope: Telescope = None,
) -> Image:
    """Create an image from a FITS file

    Parameters
    ----------
    filepath_or_hdu : str
        path of fits file of HDU object
    verbose : bool, optional
        whether to be verbose, by default False
    load_units : bool, optional
        whether to load metadata units, by default True
    load_data : bool, optional
        whether to load image data, by default True

    Returns
    -------
    :py:class:`~prose.Image`
    """
    if isinstance(filepath_or_hdu, (str, Path)):
        values = fits.getdata(filepath_or_hdu).astype(float) if load_data else None
        header = fits.getheader(filepath_or_hdu)
        path = filepath_or_hdu
    elif issubclass(type(filepath_or_hdu), _BaseHDU):
        values = filepath_or_hdu.data
        header = filepath_or_hdu.header
        path = None
    else:
        raise ValueError("filepath must be a str")

    if telescope is None:
        telescope = Telescope.from_names(
            header.get("INSTRUME", ""), header.get("TELESCOP", ""), verbose=verbose
        )

    metadata = {
        "telescope": telescope.name,
        "exposure": header.get(telescope.keyword_exposure_time, None),
        "ra": header.get(telescope.keyword_ra, None),
        "dec": header.get(telescope.keyword_dec, None),
        "filter": header.get(telescope.keyword_filter, None),
        "date": telescope.date(header).isoformat(),
        "jd": header.get(telescope.keyword_jd, None),
        "object": header.get(telescope.keyword_object, None),
        "pixel_scale": telescope.pixel_scale,
        "overscan": telescope.trimming[::-1],
        "path": path,
        "dimensions": (header.get("NAXIS1", 1), header.get("NAXIS2", 1)),
        "type": telescope.image_type(header),
    }

    if load_units:
        metadata.update(
            {
                "exposure_unit": "s",
                "ra_unit": telescope.ra_unit,
                "dec_unit": telescope.dec_unit,
                "jd_scale": telescope.jd_scale,
                "pixel_scale_unit": "arcsec",
            }
        )

    image = Image(values, metadata, {})
    if image.metadata["jd"] is None:
        image.metadata["jd"] = Time(image.date).jd
    image.fits_header = header
    image.wcs = WCS(header)
    image.telescope = telescope

    return image


class Buffer:
    def __init__(self, size: int, loader: callable = None):
        """Object to load and access adjacent items in a list

        Parameters
        ----------
        size : int
            number of items accessible
        loader : callable, optional
            a function that load an item in the buffer, by default None corresponding
            to lambda x: x

        Example
        -------
        .. code-block:: python

            from prose.core.image import Buffer
            import numpy as np

            # items to be loaded in the buffer
            init = np.arange(0, 10)

            # create and initialize
            buffer = Buffer(size=3)
            buffer.init(init)

            for buffer in buffer:
                print(buffer.previous, buffer.current, buffer.next)

        .. code-block:: text

            None 0 1
            0 1 2
            1 2 3
            2 3 4
            3 4 5
            4 5 6
            5 6 7
            6 7 8
            7 8 9
            8 9 None

        """
        assert size % 2 == 1, "size must be odd"
        self.mid_index = int((size - 1) // 2)
        self.items = [None] * max(size, 1)
        if loader is None:
            loader = lambda item: item
        self.loader = loader
        self.queue = None  # items to be loaded

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        """Get item by index relative to current

        Parameters
        ----------
        i : int
            index

        Returns
        -------
        Image or None
            images[current + i]
        """
        return self.items[self.mid_index + i]

    def __setitem__(self, i: int, item: Image):
        self.items[self.mid_index + i] = item

    def append(self, item):
        """Add an item to the buffer (and delete last)

        Parameters
        ----------
        item : any
            item to be loaded
        """
        last_item = self.items.pop(0)
        del last_item
        self.items.append(item)

    def init(self, items):
        """Prepare items to be loaded in the buffer.

        The first items are loaded with the :code:`Buffer.loader` function

        Parameters
        ----------
        items : list
            items to be loaded in the buffer
        """
        for item in items[: self.mid_index]:
            self.append(self.loader(item))
        self.queue = [*items[self.mid_index :], *[None] * self.mid_index]

    def __iter__(self):
        for item in self.queue:
            self.append(self.loader(item))
            yield self

    def sub(self, size, offset):
        pass

    @property
    def previous(self):
        return self[-1]

    @property
    def current(self):
        return self[0]

    @property
    def next(self):
        return self[1]
