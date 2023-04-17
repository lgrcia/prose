from functools import partial
from pathlib import Path

import numpy as np
from astropy.io import fits

from prose.console_utils import info
from prose.core import Block, FITSImage, Image, Sources
from prose.fluxes import Fluxes
from prose.utils import easy_median

__all__ = [
    "LimitSources",
    "Apply",
    "SortSources",
    "Get",
    "Calibration",
    "CleanBadPixels",
    "Del",
    "GetFluxes",
    "WriteTo",
    "SelectiveStack",
]


# TODO: document and test
class SortSources(Block):
    def __init__(self, verbose=False, key="cutout_sum", name=None):
        """Sort sources given a function
        TODO

        Parameters
        ----------
        verbose : bool, optional
            _description_, by default False
        key : str, optional
            _description_, by default "cutout_sum"
        name : _type_, optional
            _description_, by default None
        Returns
        -------
        _type_
            _description_
        """
        super().__init__(name, verbose)
        if isinstance(key, str):
            if key == "cutout_sum":

                def key(cutout):
                    return np.nansum(cutout.data)

        assert callable(key)

        self.key = key

    def run(self, image: Image):
        keys = np.array([self.key(cutout) for cutout in image.cutouts])
        idxs = np.argsort(keys)[::-1]
        sources = image._sources[idxs]
        for i, s in enumerate(sources):
            s.i = i
        image._sources = Sources(sources)


class Apply(Block):
    """Apply a function to an image

    Parameters
    ----------
    kwargs : function
        function to apply of the form f(image) -> None
    """

    def __init__(self, function, name=None):
        super().__init__(name=name)
        self.function = function

    def run(self, image):
        self.function(image)


class Get(Block):
    def __init__(self, *attributes, name: str = "get", arrays: bool = True, **getters):
        """Retrieve and store properties from an :py:class:`~prose.Image`

        If a list of paths is provided to a :py:class:`~prose.Sequence`, each image is
        created at the beginning of the sequence, and deleted at the end, so that
        computed data stored as :py:class:`prose.Image` properties are deleted at each iteration.
        Using the Get blocks provides a way to retain any daa stored in images before
        they are deleted.

        When a sequence is finished, this block has a `values` property, a dictionary
        where all retained properties are accessible by name, and consist of a list with
        a length corresponding to the number of images processed. The parameters of this
        dictionary are the args and kwargs provided to the block (see Example).

        If Image is constructed from a FITS image, header values can be retrieved using the
        syntax "keyword:KEY" (see example todo)

        Parameters
        ----------
        *attributes: str
            names of properties to retain
        name : str, optional
            name of the block, by default "get"
        arrays : bool, optional
            whether to convert each array of data as a numpy array , by default True
        **getters: function
            name and functions

        Example
        -------
        TODO

        """
        super().__init__(name=name)
        new_getters = {}

        def get_from_header(image, key=None):
            return image.fits_header[key]

        def get(image, key=None):
            return getattr(image, key)

        for attr in attributes:
            if "keyword:" in attr:
                attr = attr.split("keyword:")[-1]
                new_getters[attr.lower()] = partial(get_from_header, key=attr)
            else:
                new_getters[attr.lower()] = partial(get, key=attr)

        getters.update(new_getters)
        self.getters = getters
        self.values = {name: [] for name in getters.keys()}
        self.arrays = arrays
        self._parallel_friendly = True

    def run(self, image: Image):
        for name, get in self.getters.items():
            value = get(image)
            self.values[name].append(value)

    def terminate(self):
        if self.arrays:
            for key, value in self.values.items():
                self.values[key] = np.array(value)

    def __getitem__(self, key):
        return self.values[key]

    def __getattr__(self, key):
        if key in self.getters.keys():
            return self.values[key]
        else:
            raise AttributeError()


class Calibration(Block):
    def __init__(
        self,
        darks: list = None,
        flats: list = None,
        bias: list = None,
        loader=FITSImage,
        easy_ram: bool = True,
        verbose: bool = True,
        shared: bool = False,
        **kwargs,
    ):
        """Flat, Bias and Dark calibration

        Parameters
        ----------
        darks : list, optional
            list of dark files paths, by default None
        flats : list, optional
            list of flat files paths, by default None
        bias : list, optional
            list of bias files paths, by default None
        loader : object, optional
            loader used to load str path to :py:class:`~prose.Image`, by default :py:class:`~prose.FITSImage`
        easy_ram : bool, optional
            whether to compute the master median per chunks, going easy on the RAM, by default True
        verbose : bool, optional
            whether to log information about master calibration images building, by default True
        shared : bool, optional
            whether to allow the master calibration images to be shared, useful for multi-processing, by default False
        """

        super().__init__(**kwargs)

        self.loader = loader
        self.easy_ram = easy_ram

        self.shapes = {}

        self.master_bias = self._produce_master(bias, "bias")
        self.master_dark = self._produce_master(darks, "dark")
        self.master_flat = self._produce_master(flats, "flat")

        if shared:
            self._share()
        self.verbose = verbose

        self.calibration = self._calibration_shared if shared else self._calibration
        self._parallel_friendly = shared

    def _produce_master(self, images, image_type):
        if images is not None:
            assert isinstance(
                images, (list, np.ndarray, str)
            ), "images must be list or array or path"
            if len(images) == 0:
                images = None

        def _median(im):
            if self.easy_ram:
                return easy_median(im)
            else:
                return np.median(im, 0)

        _master = []

        if images is None:
            if self.verbose:
                info(f"No {image_type} images set")
            if image_type == "dark":
                master = np.array([0.0])
            elif image_type == "bias":
                master = np.array([0.0])
            elif image_type == "flat":
                master = np.array([1.0])
        else:
            if self.verbose:
                info(f"Building master {image_type}")

            for image_path in images:
                image = self.loader(image_path)
                if image_type == "dark":
                    _dark = (image.data - self.master_bias) / image.exposure.value
                    _master.append(_dark)
                elif image_type == "bias":
                    _master.append(image.data)
                elif image_type == "flat":
                    _flat = (
                        image.data
                        - self.master_bias
                        - self.master_dark * image.exposure.value
                    )
                    _flat /= np.mean(_flat)
                    _master.append(_flat)
                    del image

            if len(_master) > 0:
                master = _median(_master)
            else:
                master = None

        self.shapes[image_type] = master.shape

        return master

    def _calibration_shared(self, image, exp_time):
        bias = np.memmap(
            "__bias.array", dtype="float32", mode="r", shape=self.shapes["bias"]
        )
        dark = np.memmap(
            "__dark.array", dtype="float32", mode="r", shape=self.shapes["dark"]
        )
        flat = np.memmap(
            "__flat.array", dtype="float32", mode="r", shape=self.shapes["flat"]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return (image - (dark * exp_time + bias)) / flat

    def _calibration(self, image, exp_time):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                image - (self.master_dark * exp_time + self.master_bias)
            ) / self.master_flat

    def run(self, image):
        data = image.data
        calibrated_image = self.calibration(data, image.exposure.value)
        calibrated_image[calibrated_image < 0] = np.nan
        calibrated_image[~np.isfinite(calibrated_image)] = -1
        image.data = calibrated_image

    def _share(self):
        for imtype in ["bias", "dark", "flat"]:
            data = self.__dict__[f"master_{imtype}"]
            m = np.memmap(
                f"__{imtype}.array", dtype="float32", mode="w+", shape=data.shape
            )
            if data.ndim == 2:
                m[:, :] = data[:, :]
            else:
                m[:] = data[:]

            del self.__dict__[f"master_{imtype}"]

    @property
    def citations(self):
        return "astropy", "numpy"


class CleanBadPixels(Block):
    def __init__(
        self,
        bad_pixels_map=None,
        darks=None,
        flats=None,
        min_flat=0.6,
        loader=Image,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.loader = loader

        assert (
            darks is not None or bad_pixels_map is not None
        ), "bad_pixels_map or darks must be specified"
        if darks is not None:
            info("buidling bad pixels map")
            if darks is not None:
                max_dark = self.loader(darks[0]).data
                min_dark = self.loader(darks[0]).data

                for im in darks:
                    data = self.loader(im).data
                    max_dark = np.max([max_dark, data], axis=0)
                    min_dark = np.min([min_dark, data], axis=0)

                master_max_dark = self.loader(data=max_dark).data
                master_min_dark = self.loader(data=min_dark).data

                theshold = 3 * np.std(master_max_dark)
                median = np.median(master_max_dark)
                hots = np.abs(master_max_dark) - median > theshold
                deads = master_min_dark < median / 2

                self.bad_pixels = np.where(hots | deads)
                self.bad_pixels_map = np.zeros_like(master_min_dark)

            if flats is not None:
                _flats = []
                for flat in flats:
                    data = self.loader(flat).data
                    _flats.append(data / np.mean(data))
                master_flat = easy_median(_flats)
                master_flat = self.clean(master_flat)
                bad_flats = np.where(master_flat < min_flat)
                if len(bad_flats) == 2:
                    self.bad_pixels = (
                        np.hstack([self.bad_pixels[0], bad_flats[0]]),
                        np.hstack([self.bad_pixels[1], bad_flats[1]]),
                    )

            self.bad_pixels_map[self.bad_pixels] = 1

        elif bad_pixels_map is not None:
            if isinstance(bad_pixels_map, (str, Path)):
                bad_pixels_map = Image(bad_pixels_map).data
            elif isinstance(bad_pixels_map, Image):
                bad_pixels_map = bad_pixels_map.data
            else:
                bad_pixels_map = bad_pixels_map

            self.bad_pixels_map = bad_pixels_map
            self.bad_pixels = np.where(bad_pixels_map == 1)

    def clean(self, data):
        data[self.bad_pixels] = np.nan
        data[data < 0] = np.nan
        nans = np.array(np.where(np.isnan(data))).T
        padded_data = np.pad(data.copy(), (1, 1), constant_values=np.nan)

        for i, j in nans + 1:
            mean = np.nanmean(
                [
                    padded_data[i, j - 1],
                    padded_data[i, j + 1],
                    padded_data[i - 1, j],
                    padded_data[i + 1, j],
                ]
            )
            padded_data[i, j] = mean
            data[i - 1, j - 1] = mean

        return data

    def run(self, image):
        image.data = self.clean(image.data.copy())


class Del(Block):
    def __init__(self, *names, name="del"):
        """Remove a property from an Image

        In general this is use in multi-processing sequences to avoid large image properties to be copied in-between processes

        Parameters
        ----------
        *names: str
            properties to be deleted from image
        name : str, optional
            name of the block, by default "del"
        """
        super().__init__(name=name)
        self.names = names

    def run(self, image):
        for name in self.names:
            setattr(image, name, None)


class LimitSources(Block):
    def __init__(self, min: int = 4, max: int = 10000, name=None):
        """Limit number of sources. If not in between min and max sources, image is discarded

        Parameters
        ----------
        min : int, optional
            minimum number of sources, by default 4
        max : int, optional
            maximum number of sources, by default 10000
        """

        super().__init__(name=name)
        self.min = min
        self.max = max
        self._parallel_friendly = True

    def run(self, image):
        n = len(image.sources)
        if n < self.min or n > self.max:
            image.discard = True


class GetFluxes(Get):
    def __init__(self, *args, time: str = "jd", name: str = None, **kwargs):
        """A conveniant class to get fluxes and background from aperture and annulus blocks

        |read| :code:`Image.aperture`, :code:`Image.annulus` and :code:`Image.{time}`

        Parameters
        ----------
        time : str, optional
            The image property corresponding to time, by default 'jd'
        name: str, optional
            Name of the block
        *args, **kwargs:
            args and kwargs of :py:class:`prose.blocks.Get`
        """
        self._time_key = time
        get_fluxes = lambda im: im.aperture["fluxes"]

        def get_bkg(im):
            if "annulus" in im.computed.keys():
                return im.annulus["median"]
            else:
                return np.zeros(len(im.sources))

        def get_time(im):
            if self._time_key in im.computed.keys():
                return getattr(im, self._time_key)
            elif im.jd is not None:
                return im.jd
            else:
                return im.i

        def get_aperture(im):
            return im.aperture["radii"]

        super().__init__(
            *args,
            _time=get_time,
            _bkg=get_bkg,
            _fluxes=get_fluxes,
            _apertures=get_aperture,
            name=name,
            **kwargs,
        )
        self.fluxes = None
        self._parallel_friendly = True

    def terminate(self):
        super().terminate()
        area = np.pi * (self._apertures**2)
        raw_fluxes = (self._fluxes - self._bkg[:, :, None] * area[:, None, :]).T
        time = self._time
        data = {"bkg": np.mean(self._bkg, -1)}
        data.update({key: value for key, value in self.values.items() if key[0] != "_"})
        self.fluxes = Fluxes(
            time=time, fluxes=raw_fluxes, data=data, apertures=self._apertures
        )


class WriteTo(Block):
    def __init__(
        self, destination, label="processed", imtype=True, overwrite=False, name=None
    ):
        """Write image to FITS file

        Parameters
        ----------
        destination : str
            destination folder (folder and parents created if not existing)
        label : str, optional
            added at the end of filename as {original_path}_{label}.fits, by default "processed"
        imtype : bool, optional
            If bool, whether to set image imtype as label (`image.header["IMTYPE"] = label`). If a `str`, label to set for imtype (`image.header["IMTYPE"] = imtype`) , by default True
        overwrite : bool, optional
            whether to overwrite existing file, by default False
        name : str, optional
            name of the block, by default None
        """
        super().__init__(name=name)
        self.destination = Path(destination)
        self.label = label
        self.overwrite = overwrite
        if isinstance(imtype, bool):
            if imtype:
                self.imtype = self.label
            else:
                self.imtype = None
        else:
            assert isinstance(imtype, str), "imtype must be a bool or a str"
            self.imtype = imtype

        self.files = []

    def run(self, image):
        self.destination.mkdir(exist_ok=True, parents=True)

        new_hdu = fits.PrimaryHDU(image.data)
        new_hdu.header = image.fits_header

        if self.imtype is not None:
            image.fits_header[image.telescope.keyword_image_type] = self.imtype

        fits_new_path = self.destination / (
            Path(image.metadata["path"]).stem + f"_{self.label}.fits"
        )

        new_hdu.writeto(fits_new_path, overwrite=self.overwrite)
        self.files.append(fits_new_path)


class SelectiveStack(Block):
    def __init__(self, n=5, name=None):
        """Build a median stack image from the `n` best-FWHM images

        |read| :code:`Image.fwhm`

        Parameters
        ----------
        n : int, optional
            number of images to use, by default 5
        name : str, optional
            name of the blocks, by default None
        """
        super().__init__(name=name)
        self.n = n
        self._images = []
        self._sigmas = []

    def run(self, image: Image):
        sigma = image.fwhm
        if len(self._images) < self.n:
            self._images.append(image)
            self._sigmas.append(sigma)
        else:
            i = np.argmax(self._sigmas)
            if self._sigmas[i] > sigma:
                self._sigmas[i] = sigma
                self._images[i] = image

    def terminate(self):
        self.stack = Image(easy_median([im.data for im in self._images]))
