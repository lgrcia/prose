from .. import Block
from astropy.io import fits
import numpy as np
from astropy.time import Time
from os import path
import imageio
from .. import viz
from astropy.stats import SigmaClip
from photutils import MedianBackground
from .psf import cutouts
from .. import utils
import matplotlib.pyplot as plt
import time
import xarray as xr
from ..utils import easy_median
from ..console_utils import info
from pathlib import Path
from . import Cutout2D
import matplotlib.patches as patches
from ..core.image import Image
from astropy.nddata import Cutout2D as astopy_Cutout2D
from astropy.units.quantity import Quantity
from astropy.stats import sigma_clipped_stats

__all__ = [
    "Stack",
    "StackStd",
    "SaveReduced",
    "RemoveBackground",
    "CleanCosmics",
    "WriteTo",
    "Pass",
    "ImageBuffer",
    "Set",
    "Flip",
    "Get",
    "XArray",
    "LocalInterpolation",
    "Trim",
    "Calibration",
    "CleanBadPixels",
    "MedianStack",
    "XArray2",
    "MPCalibration",
    "Del",
    "Function"
]

class DataBlock(Block):

    def __init__(self, name=None):
        super().__init__(name)

class Stack(DataBlock):
    """Build a FITS stack image of the observation

    The stack image is accessible through the ``stack`` attribute. It is built by accumulating images along creating a pixel weights map. This map allows to ignore bad pixels contributions to the stack, built through a weighted mean.
    
    .. note:
    
        Not using median stacking is done as to avoid storing a large number of images in the RAM

    The idea of weighting is stolen from https://github.com/lsst/meas_algorithms/blob/main/python/lsst/meas/algorithms/accumulator_mean_stack.py

    Parameters
    ----------
    destination : str, optional
        path of the stack image (must be a .fits file name), dfault is None and does not save
    header : dict, optional
        header base of the stack image to be saved, default is None for fresh header
    overwrite : bool, optional
        weather to overwrite file if exists, by default False
    """
    
    def __init__(self, ref=None, **kwargs):

        super(Stack, self).__init__(**kwargs)
        self._stack = None
        self._n_images = 0
        self._header = ref.header.copy() if ref else {}
        self.image = ref.copy() if ref else None

    @property
    def stack(self):
        return self.image

    def run(self, image):
        #TODO check that all images have same telescope?

        data = image.data.copy()

        if self._stack is None:
            #first run
            self._stack = data
            self.telescope = image.telescope
        else:
            self._stack += data

        self._n_images += 1

    def terminate(self):

        self._stack = self._stack/self._n_images

        self._header[self.telescope.keyword_image_type] = "stack"
        self._header["BZERO"] = 0
        self._header["REDDATE"] = Time.now().to_value("fits")
        self._header["NIMAGES"] = self._n_images

        if self.image is None:
            self.image = Image(data=self._stack, header=self._header)
        else:
            self.image.data = self._stack
            self.image.header = self._header

    def concat(self, block):
        if self._stack is not None:
            if block.stack is not None:
                self._stack += block._stack
            else:
                pass
        else:
            self._stack = block._stack
        self._n_images += block._n_images

class StackStd(DataBlock):
    
    def __init__(self, destination=None, overwrite=False, **kwargs):
        super(StackStd, self).__init__(**kwargs)
        self.images = []
        # self.stack_header = None
        # self.destination = destination
        self.overwrite = overwrite
        self.stack_std = None

    def run(self, image, **kwargs):
        self.images.append(image.data)

    def terminate(self):
        self.images = np.array(self.images)
        # shape_divisors = utils.divisors(self.images[0].shape[1])
        # n = shape_divisors[np.argmin(np.abs(50 - shape_divisors))]
        self.stack_std = np.std(self.images, axis=0) #concatenate([np.std(im, axis=0) for im in np.split(self.images, n, axis=1)])
        # stack_hdu = fits.PrimaryHDU(self.stack_std, header=self.stack_header)
        # stack_hdu.header["IMTYPE"] = "std"
        # stack_hdu.writeto(self.destination, overwrite=self.overwrite)


class SaveReduced(Block):
    """Save reduced FITS images.

    |write| ``Image.header``

    Parameters
    ----------
    destination : str
        folder path of the images. Orignial name is used with the addition of :code:`_reduced.fits`
    overwrite : bool, optional
        weather to overwrite file if exists, by default False
    """
    # TODO rename to SaveFITS and make destination a string like thing with the name of the image...
    
    def __init__(self, destination, overwrite=False, **kwargs):

        super().__init__(**kwargs)
        self.destination = Path(destination)
        self.destination.mkdir(exist_ok=True)
        self.overwrite = overwrite
        self.files = []

    def run(self, image, **kwargs):

        new_hdu = fits.PrimaryHDU(image.data)
        new_hdu.header = image.header
        
        # TODO: what the fuck?
        image.header["SEEING"] = image.get(image.telescope.keyword_seeing, "")
        image.header["BZERO"] = 0
        image.header["REDDATE"] = Time.now().to_value("fits")
        image.header[image.telescope.keyword_image_type] = "reduced"

        fits_new_path = path.join(
            self.destination,
            path.splitext(path.basename(image.path))[0] + "_reduced.fits"
        )

        new_hdu.writeto(fits_new_path, overwrite=self.overwrite)
        self.files.append(fits_new_path)
    
    def concat(self, block):
        self.files = [*self.files, *block.files]

class WriteTo(Block):
    
    def __init__(self, destination, label="processed", imtype=True, overwrite=False, name=None):
        """Write image to FITS file

        Parameters
        ----------
        destination : str
            destination folder (folder and parents created if not existing)
        label : str, optional
            added at the end of filename as {original_path}_{label}.fits, by default "processed"
        imtype : bool, optional
            If bool, wether to set image imtype as label (image.header["IMTYPE"] = label). If str imtype label to set (image.header["IMTYPE"] = imtype) , by default True
        overwrite : bool, optional
            wether to overwrite existing file, by default False
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
        new_hdu.header = image.header
        
        if self.imtype is not None:
            image.header[image.telescope.keyword_image_type] = self.imtype
        
        fits_new_path = self.destination / (Path(image.path).stem + f"_{self.label}.fits")

        new_hdu.writeto(fits_new_path, overwrite=self.overwrite)
        self.files.append(fits_new_path)

class RemoveBackground(Block):

    
    def __init__(self):
        super().__init__()
        self.stack_data = None

    def run(self, image, **kwargs):
        _, im_median, _ = sigma_clipped_stats(image.data, sigma=3.0)
        image.data = im_median


class CleanCosmics(Block):

    
    def __init__(self, threshold=2):
        super().__init__()
        self.stack_data = None
        self.threshold = threshold
        self.sigma_clip = SigmaClip(sigma=3.)
        self.bkg_estimator = MedianBackground()

    def initialize(self, fits_manager):
        if fits_manager.has_stack():
            self.stack_data = fits.getdata(fits_manager.get("stack")[0])
        self.std_stack = fits.getdata(path.join(fits_manager.folder, "test_std.fits"))

    def run(self, image, **kwargs):
        mask = image.data > (self.stack_data + self.std_stack * self.threshold)
        image.data[mask] = self.stack_data[mask]


class Pass(Block):
    """A Block that does nothing"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def run(self, image):
        pass


class ImageBuffer(DataBlock):
    """Stores the last Image
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image = None

    def run(self, image, **kwars):
        self.image = image.copy()


class Set(Block):
    """Sets specific attribute to every image

    For example to set attributes ``a`` with the value 2 on every image (i.e Image.a = 2):
    
    .. code-block:: python

        from prose import blocks

        set_block = blocks.Set(a=2)

    Parameters
    ----------
    kwargs : kwargs
        keywords argument and values to be set on every image
    """
    
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self.kwargs = kwargs

    def run(self, image):
        for name, value in self.kwargs.items():
            setattr(image, name, value)

class Flip(Block):
    """Flip an image according to a reference

    ``Telescope.keyword_flip`` is used. The image is fliped if its flip value differs from the reference one

    Parameters
    ----------
    reference_image : `Image`
        Image serving as a reference for the flip value
    """
    
    def __init__(self, reference_image, **kwargs):
        """[summary]

        Parameters
        ----------
        reference_image : [type]
            [description]
        """
        super().__init__(**kwargs)
        self.reference_image = reference_image
        self.reference_flip_value = None
        self.reference_flip_value = self.reference_image.flip

    def run(self, image, **kwargs):
        flip_value = image.flip
        if flip_value != self.reference_flip_value:
            image.data = image.data[::-1, ::-1]

class Function(Block):

    def __init__(self, f, name=None):
        super().__init__(name=name)
        self.f = f

    def run(self, image):
        self.f(image)

# TODO document
class Get(DataBlock):

    
    def __init__(self, *names, name="get"):
        super().__init__(name=name)
        self.names = names
        self.values = {name: [] for name in names}

    def run(self, image, **kwargs):
        for name in self.names:
            try:
                value = image.__getattribute__(name)
            except:
                try:
                    value = image.header[name]
                except:
                    raise AttributeError(f"'{name}' not in Image attributes or Image.header")

            self.values[name].append(value)
    
    def __getattr__(self, key):
        if key in self.names:
            return self.values[key]
        else:
            super().__getattribute__(key)


    def __call__(self, *names):
        if len(names) == 0:
            return self.values
        elif len(names) == 1:
            return self.values[names[0]]
        elif len(names) > 1:
            return [self.values[name] for name in names]

    def concat(self, block):
        for name in self.names:
            self.values[name] = [*self.values[name], *block.values[name]]

class XArray(DataBlock):

    
    def __init__(self, *names, name="xarray", raise_error=True, concat_dim="time", **kwargs):
        super().__init__(name=name)
        self.variables = {name: (dims, []) for dims, name in names}
        self.raise_error = raise_error
        self.xarray = xr.Dataset()
        self.concat_dim = concat_dim
        self.xarray.attrs.update(kwargs)

    def run(self, image):
        for name in self.variables:
            try:
                value = image.__getattribute__(name)
                if isinstance(image.__getattribute__(name), Quantity):
                    value = value.value
                self.variables[name][1].append(value)
            except AttributeError:
                if self.raise_error:
                    raise AttributeError()
                else:
                    pass

    def __call__(self):
        return self.xarray

    def terminate(self):
        for name, var in self.variables.items():
            self.xarray[name] = var

    def save(self, destination):
        self.xarray.to_netcdf(destination)

    def concat(self, block):
        if len(self.variables) > 0:
            if len(block.variables) > 0:
                for name, (dims, var) in self.variables.items():
                    if len(var) > 0 and len(block.variables[name][1]) > 0:
                        a = np.flatnonzero(np.array(dims) == self.concat_dim)
                        if len(a) > 0:
                            self.variables[name] = (dims, np.concatenate([var, block.variables[name][1]], axis=a[0]))
            else:
                pass
        else:
            self.variables = block.variables.copy()

    def to_observation(self, stack, sequence=None):
        if len(stack.stars_coords) != len(self.xarray.star):
            raise ValueError("stack stars_coords must be aligned to xarray stars_coords (stars_coords probably coming from the reference in the Stack block)")

        xarr = utils.image_in_xarray(stack, self.xarray, stars=True) # adding reference as a stack
        xarr = xarr.transpose("apertures", "star", "time", ...) # ... just needed

        if sequence is not None:
            xarr.attrs["photometry"] = [b.__class__.__name__ for b in sequence.blocks]

        # xarr.attrs["prose_version"] = __version__
        return xarr

    @property
    def citations(self):
        return {"xarray": """
        @article{hoyer2017xarray,
            title   = {xarray: {N-D} labeled arrays and datasets in {Python}},
            author  = {Hoyer, S. and J. Hamman},
            journal = {In revision, J. Open Res. Software},
            year    = {2017}
            }
        """} 

class LocalInterpolation(Block):
    
    def __init__(self, **kargs):
        super().__init__(**kargs)
    
    def run(self, image):
        image.data[image.data<0] = np.nan
        nans = np.array(np.where(np.isnan(image.data))).T 
        padded_data = np.pad(image.data.copy(), (1, 1), constant_values=np.nan)

        for i, j in nans + 1:
            mean = np.nanmean([
                padded_data[i, j-1],
                padded_data[i, j+1],
                padded_data[i-1, j],
                padded_data[i+1, j],
            ])
            padded_data[i, j] = mean
            image.data[i-1, j-1] = mean

class Trim(Block):
    """Image trimming. If trim is not specified, triming is taken from the telescope characteristics

    |write| ``Image.header``
    
    |modify|

    Parameters
    ----------
    skip_wcs : bool, optional
        whether to skip applying trim to WCS, by default False
    trim : tuple, int or flot, optional
        (x, y) trim values, by default None which uses the ``trim`` value from the image telescope definition. If an int or a float is provided trim will be be applied to both axes.
    

    Example
    -------

    In what follows we generate an example image and apply a trimming on it

    .. jupyter-execute::

        from prose.tutorials import example_image
        from prose.blocks import Trim

        # our example image
        image = example_image()

        # Creating and applying the Trim block
        trim = Trim(trim=100)
        trimmed_image = trim(image)

    We can now see the resulting trimmed image against its original shape

    .. jupyter-execute::

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        ax1 = plt.subplot(121)
        image.show(ax=ax1)
        trim.draw_cutout(image)
        plt.axis("off")
        _ = plt.title("original image (white = cutout)", loc="left")

        ax2 = plt.subplot(122)
        trimmed_image.show(ax=ax2)
        plt.axis("off")
        _ = plt.title("trimmed image", loc="left")

    """

    
    def __init__(self, skip_wcs=False, trim=None, **kwargs):

        super().__init__(**kwargs)
        self.skip_wcs = skip_wcs
        if isinstance(trim, (int, float)):
            trim = (trim, trim)
        self.trim = trim

    def run(self, image, **kwargs):
        shape = image.shape
        center = shape[::-1] / 2
        trim = self.trim if self.trim is not None else image.telescope.trimming[::-1]
        dimension = shape - 2 * np.array(trim)
        trim_image = astopy_Cutout2D(image.data, center, dimension, wcs=None if self.skip_wcs else image.wcs, )
        image.data = trim_image.data
        if not self.skip_wcs:
            image.header.update(trim_image.wcs.to_header())
            image.header["NAXIS2"], image.header["NAXIS1"] = trim_image.data.shape

    def draw_cutout(self, image, ax=None, lw=1, c="w"):
        w, h = image.shape - 2*np.array(self.trim)
        rect = patches.Rectangle(2*np.array(self.trim)/2, w, h, linewidth=lw, edgecolor=c, facecolor='none')
        if ax is None:
            ax = plt.gca()
        ax.add_patch(rect)


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

    
    def __init__(self, darks=None, flats=None, bias=None, loader=Image, easy_ram=True, verbose=True, **kwargs):

        super().__init__(**kwargs)
            
        self.loader = loader
        self.easy_ram = easy_ram

        self.master_bias = self._produce_master(bias, "bias")
        self.master_dark = self._produce_master(darks, "dark")
        self.master_flat = self._produce_master(flats, "flat")
        self.verbose = verbose

    def calibration(self, image, exp_time):
        with np.errstate(divide='ignore', invalid='ignore'):
            return (image - (self.master_dark * exp_time + self.master_bias)) / self.master_flat

    def _produce_master(self, images, image_type):
        if images is not None:
            assert isinstance(images, (list, np.ndarray, str)), "images must be list or array or path"
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
            if self.verbose: info(f"No {image_type} images set")
            if image_type == "dark":
                return 0
            elif image_type == "bias":
                return  0
            elif image_type == "flat":
                return 1
        else:
            if self.verbose: info(f"Building master {image_type}")

        for image_path in images:
            image = self.loader(image_path)
            if image_type == "dark":
                _dark = (image.data - self.master_bias) / image.exposure.value
                _master.append(_dark)
            elif image_type == "bias":
                _master.append(image.data)
            elif image_type == "flat":
                _flat = image.data - self.master_bias - self.master_dark*image.exposure.value
                _flat /= np.mean(_flat)
                _master.append(_flat)
                del image

        if len(_master) > 0:
            med = _median(_master)
            return med
        else:
            return None

    def show_masters(self, figsize=(20, 80)):
        plt.figure(figsize=figsize)
        plt.subplot(131)
        plt.title("Master bias")
        im = plt.imshow(utils.z_scale(self.master_bias), cmap="Greys_r", origin="lower")
        viz.add_colorbar(im)
        plt.subplot(132)
        plt.title("Master dark")
        im = plt.imshow(utils.z_scale(self.master_dark), cmap="Greys_r", origin="lower")
        viz.add_colorbar(im)
        plt.subplot(133)
        plt.title("Master flat")
        im = plt.imshow(utils.z_scale(self.master_flat), cmap="Greys_r", origin="lower")
        viz.add_colorbar(im)

    def show_bad_pixels(self):
        pass

    def run(self, image):
        data = image.data
        calibrated_image = self.calibration(data, image.exposure.value)
        calibrated_image[calibrated_image < 0] = np.nan
        calibrated_image[~np.isfinite(calibrated_image)] = -1
        image.data = calibrated_image

    @property
    def citations(self):
        return "astropy", "numpy"

    @property
    def shared(self):
        for imtype in ['bias', 'dark', 'flat']:
            data = self.__dict__[f"master_{imtype}"]
            m = np.memmap(f"__{imtype}.array", dtype='float32', mode='w+', shape=data.shape)
            m[:, :] = data[:, :]
        
        return MPCalibration()


class CleanBadPixels(Block):
    
    def __init__(self, bad_pixels_map=None, darks=None, flats=None, min_flat=0.6, loader=Image, **kwargs):
        super().__init__(**kwargs)
        
        self.loader = loader
        
        assert darks is not None or bad_pixels_map is not None, "bad_pixels_map or darks must be specified"
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

                theshold = 3*np.std(master_max_dark)
                median = np.median(master_max_dark)
                hots = np.abs(master_max_dark)-median > theshold
                deads = master_min_dark < median/2

                self.bad_pixels = np.where(hots | deads)
                self.bad_pixels_map = np.zeros_like(master_min_dark)

            if flats is not None:
                _flats = []
                for flat in flats:
                    data = self.loader(flat).data
                    _flats.append(data/np.mean(data))
                master_flat = easy_median(_flats)
                master_flat = self.clean(master_flat)
                bad_flats = np.where(master_flat < min_flat)
                if len(bad_flats) == 2:
                    self.bad_pixels = (
                        np.hstack([self.bad_pixels[0], bad_flats[0]]),
                        np.hstack([self.bad_pixels[1], bad_flats[1]])
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
        data[data<0] = np.nan
        nans = np.array(np.where(np.isnan(data))).T
        padded_data = np.pad(data.copy(), (1, 1), constant_values=np.nan)

        for i, j in nans + 1:
            mean = np.nanmean([
                padded_data[i, j-1],
                padded_data[i, j+1],
                padded_data[i-1, j],
                padded_data[i+1, j],
            ])
            padded_data[i, j] = mean
            data[i-1, j-1] = mean
            
        return data
    
    def run(self, image):
        image.data = self.clean(image.data.copy())


class MedianStack(DataBlock):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.images = []

    def run(self, image):
        self.images.append(image.data)
        
    def terminate(self):
        self.stack = Image(data=np.median(self.images, 0))


class XArray2(DataBlock):
    def __init__(self, names, name="xarray", raise_error=True, concat_dim="time", **kwargs):
        super().__init__(name=name)
        self.variables = names
        self.raise_error = raise_error
        self.xarray = xr.Dataset()
        self.concat_dim = concat_dim
        self.xarray.attrs.update(kwargs)
        self.xarray_init = False
    
    def _to_xarray(self, image):
        _x = xr.Dataset()
        for name, dims in self.variables.items():
            _x[name] = (dims, [self._get_value(image, name)])
        return _x
    
    def _get_value(self, image, key):
        try:
            value = image.__getattribute__(key)
            if isinstance(image.__getattribute__(key), Quantity):
                value = value.value
            return value
        except AttributeError:
            if self.raise_error:
                raise AttributeError()
            else:
                pass

    def run(self, image):
        if not self.xarray_init:
            self.xarray = self._to_xarray(image)
            self.xarray_init = True
        else:
            x = self._to_xarray(image)
            self.xarray = xr.concat([self.xarray, x], dim=self.concat_dim)

    def save(self, destination):
        self.xarray.to_netcdf(destination)            
        
    def concat(self, block):
        if len(block.xarray) == 0:
            pass
        elif len(self.xarray) == 0:
            self.xarray = block.xarray.copy()
            self.xarray_init = True
        else:
            self.xarray =  xr.concat([self.xarray, block.xarray], dim=self.concat_dim)


class MPCalibration(Block):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def calibration(self, image, exp_time):
        bias = np.memmap('__bias.array', dtype='float32', mode='r', shape=image.shape)
        dark = np.memmap('__dark.array', dtype='float32', mode='r', shape=image.shape)
        flat = np.memmap('__flat.array', dtype='float32', mode='r', shape=image.shape)
        with np.errstate(divide='ignore', invalid='ignore'):
            return (image - (dark * exp_time + bias)) / flat

    def run(self, image):
        data = image.data
        calibrated_image = self.calibration(data, image.exposure.value)
        calibrated_image[calibrated_image < 0] = np.nan
        calibrated_image[~np.isfinite(calibrated_image)] = -1
        image.data = calibrated_image

class Del(Block):

    def __init__(self, *args, name="del"):
        super().__init__(name=name)
        self.args = args

    def run(self, image):
        for arg in self.args:
            del image.__dict__[arg]

class Drizzle(Block):
    
    def __init__(self, reference, pixfrac=1., **kwargs):
        from drizzle import drizzle
        super().__init__(self, **kwargs)
        self.reference = reference
        self.pixfrac = pixfrac
        self.drizzle = drizzle.Drizzle(outwcs=reference.wcs, pixfrac=pixfrac)
        self.image = None
        
    def run(self, image):
        WCS = image.wcs
        self.drizzle.add_image(image.data, image.wcs)
    
    def terminate(self):
        data = self.drizzle.outsci
        header = self.reference.header.copy()
        header.update(self.drizzle.outwcs.to_header())
        self.image = Image(data=data, header=header)

    @property
    def citations(self):
        return {"drizzle": """
@ARTICLE{drizzle,
    author = {{Fruchter}, A.~S. and {Hook}, R.~N.},
    title = "{Drizzle: A Method for the Linear Reconstruction of Undersampled Images}",
    journal = {\pasp},
    keywords = {Methods: Data Analysis, Techniques: Photometric, Astrophysics},
    year = 2002,
    month = feb,
    volume = {114},
    number = {792},
    pages = {144-152},
    doi = {10.1086/338393},
    archivePrefix = {arXiv},
    eprint = {astro-ph/9808087},
    primaryClass = {astro-ph},
    adsurl = {https://ui.adsabs.harvard.edu/abs/2002PASP..114..144F},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}"""}