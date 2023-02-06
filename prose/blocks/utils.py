from ..core import Block, Image, Sources, FITSImage
from ..console_utils import info
from ..utils import easy_median
from ..fluxes import Fluxes
import numpy as np

__all__ = [
    "LimitSources",
    "Apply",
    "SortSources",
    "Get",
    "Calibration",
    "CleanBadPixels",
    "Del",
    "GetFluxes"
]

# TODO: document and test
class SortSources(Block):
    def __init__(self, name=None, verbose=False, key="cutout_sum"):
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
    def __init__(self, *names, name="get", arrays=True, **getters):
        super().__init__(name=name)
        getters.update({name: lambda image: getattr(image, name) for name in names})
        self.getters = getters
        self.values = {name: [] for name in getters.keys()}
        self.arrays = arrays

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
    """
    Flat, Bias and Dark calibration

    Parameters
    ----------
    darks : list
        list of dark files paths
    flats : list
        list of flat files paths
    bias : list
        list of bias files paths
    """

    def __init__(
        self,
        darks=None,
        flats=None,
        bias=None,
        loader=FITSImage,
        easy_ram=True,
        verbose=True,
        shared=True,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.loader = loader
        self.easy_ram = easy_ram

        self.shapes = {}

        self.master_bias = self._produce_master(bias, "bias")
        self.master_dark = self._produce_master(darks, "dark")
        self.master_flat = self._produce_master(flats, "flat")

        self._share()
        self.verbose = verbose

        self.calibration = self._calibration_shared if shared else self._calibration

    def _produce_master(self, images, image_type):
        if images is not None:
            assert isinstance(
                images, (list, np.ndarray, str)
            ), "images must be list or array or path"
            if len(images) == 0:
                images = None

        if isinstance(images, str):
            return self.loader(images).data

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

    def run(self, image):
        data = image.data
        calibrated_data = self.calibration(data, image.exposure.value)
        calibrated_data[calibrated_data < 0] = np.nan
        calibrated_data[~np.isfinite(calibrated_data)] = -1
        image.data = calibrated_data

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
        with np.errstate(divide='ignore', invalid='ignore'):
            return (image - (self.master_dark * exp_time + self.master_bias)) / self.master_flat


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

    def run(self, image):
        n = len(image.sources)
        if n < self.min or n > self.max:
            image.discard = True


class GetFluxes(Get):
    
    def __init__(self, data=None, time='jd'):
        self.data_keys = data if data is not None else []
        get_bkg = lambda im: im.aperture["fluxes"]
        get_fluxes = lambda im: im.annulus["median"][:, None] * np.pi*(im.aperture['radii']**2)
        self._time_key = time
        super().__init__(time, *self.data_keys, _bkg=get_bkg, _fluxes=get_fluxes)
        self.fluxes=None
        
    def terminate(self):
        super().terminate()
        raw_fluxes = (self._fluxes - self._bkg).T
        time = np.array(self.values[self._time_key])
        del self.values["_fluxes"]
        self.values["bkg"] = self.values["_bkg"]
        del self.values["_bkg"]
        del self.values[self._time_key]
        self.fluxes = Fluxes(time=time, fluxes=raw_fluxes, data=self.values)