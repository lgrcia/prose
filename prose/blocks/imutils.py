from xarray.core import variable
from ..core import Block
from astropy.io import fits
import numpy as np
from astropy.time import Time
from os import path
import imageio
from .. import viz
from astropy.stats import SigmaClip
from photutils import MedianBackground
from .psf import cutouts
import os
from .. import utils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
import xarray as xr


class Stack(Block):
    """Build a FITS stack image of the observation

    Parameters
    ----------
    destination : str, optional
        path of the stack image (must be a .fits file name), dfault is None and does not save
    header : dict, optional
        header base of the stack image to be saved, default is None for fresh header
    overwrite : bool, optional
        weather to overwrite file if exists, by default False
    """

    def __init__(self, destination=None, header=None, overwrite=False, **kwargs):

        super(Stack, self).__init__(**kwargs)
        self.stack = None
        self.n_images = 0
        self.header = header if header else {}
        self.destination = destination
        self.fits_manager = None
        self.overwrite = overwrite
        self.telescope = None
        self.xarray = None

        self.reference_image_path = None

    def run(self, image, **kwargs):
        if self.stack is None:
            self.stack = image.data
            # telescope is assumed to be the one of first image
            self.telescope = image.telescope

        else:
            self.stack += image.data

        self.n_images += 1

    def terminate(self):

        self.stack = self.stack/self.n_images

        self.header[self.telescope.keyword_image_type] = "stack"
        self.header["BZERO"] = 0
        self.header["REDDATE"] = Time.now().to_value("fits")
        self.header["NIMAGES"] = self.n_images

        if self.destination is not None:
            stack_hdu = fits.PrimaryHDU(self.stack, header=self.header)
            stack_hdu.writeto(self.destination, overwrite=self.overwrite)

    def concat(self, block):
        if self.stack is not None:
            if block.stack is not None:
                self.stack += block.stack
            else:
                pass
        else:
            self.stack = block.stack
        self.n_images += block.n_images

class StackStd(Block):
    
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
        self.destination = destination
        if not path.exists(self.destination):
            os.mkdir(self.destination)
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


class Video(Block):
    """Build a video of all :code:`Image.data`.

    Can be either from raw image or a :code:`int8` rgb image.

    Parameters
    ----------
    destination : str
        path of the video which format depends on the extension (e.g. :code:`.mp4`, or :code:`.gif)
    overwrite : bool, optional
        weather to overwrite file if exists, by default False
    factor : float, optional
        subsampling factor of the image, by default 0.25
    fps : int, optional
        frames per second of the video, by default 10
    from_fits : bool, optional
        Wether :code:`Image.data` is a raw fits image, by default False. If True, a z scaling is applied as well as casting to `uint8`
    """

    def __init__(self, destination, overwrite=True, factor=0.25, fps=10, from_fits=False, **kwargs):

        super().__init__(**kwargs)
        self.destination = destination
        self.overwrite = overwrite
        self.images = []
        self.factor = factor
        self.fps = fps
        self.from_fits = from_fits

    def initialize(self, *args):
        # Check if writer is available (sometimes require extra packages)
        _ = imageio.get_writer(self.destination, mode="I")

    def run(self, image, **kwargs):
        if self.from_fits:
            self.images.append(viz.gif_image_array(image.data, factor=self.factor))
        else:
            self.images.append(image.data.copy())

    def terminate(self):
        imageio.mimsave(self.destination, self.images, fps=self.fps)

    def citations(self):
        return "imageio"


from astropy.stats import sigma_clipped_stats


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


class ImageBuffer(Block):
    """Store the last Image
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image = None

    def run(self, image, **kwars):
        self.image = image.copy()


class Set(Block):
    """Set specific attribute to every image

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
        image.__dict__.update(self.kwargs)


class Cutouts(Block):
    """Record all cutouts centered on Image.stars_coords

    Cutouts are sometimes called "imagette" and represent small portions of the image centered on cpecific points.

    Parameters
    ----------
    size : int, optional
        width and height of the cutout, by default 21
    """
    def __init__(self, size=21, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def run(self, image, **kwargs):
        image.cutouts_idxs, image.cutouts = cutouts(image.data, image.stars_coords, size=self.size)


class Flip(Block):
    """Flip an image according to a reference

    Telescope.keyword_flip is used. The image is fliped if its flip value differs from the reference

    Parameters
    ----------
    reference_image : `Image`
        Image serving as a reference
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

    def initialize(self, *args):
        self.reference_flip_value = self.reference_image.flip

    def run(self, image, **kwargs):
        flip_value = image.flip
        if flip_value != self.reference_flip_value:
            image.data = image.data[::-1, ::-1]


class Plot(Block):
    """Generate a video/gif given a plotting function

    Parameters
    ----------
    plot_function : functio,
        A plotting function taking an `Image` as argument and using pyplot
    destination : str path
        path of the image to be saved
    fps : int, optional
        frame per seconds, by default 10
    """

    def __init__(self, plot_function, destination, fps=10, **kwargs):
        super().__init__(**kwargs)
        self.plot_function = plot_function
        self.plots = []
        self.destination = destination
        self.fps = fps
        self._init_alias = plt.rcParams['text.antialiased']
        plt.rcParams['text.antialiased'] = False

    def initialize(self, *args):
        # Check if writer is available (sometimes require extra packages)
        _ = imageio.get_writer(self.destination, mode="I")

    def to_rbg(self):
        fig = plt.gcf()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        width, height = fig.canvas.get_width_height()
        returned = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        plt.imshow(returned)
        plt.close()
        return returned

    def run(self, image):
        self.plot_function(image)
        self.plots.append(self.to_rbg())

    def terminate(self):
        imageio.mimsave(self.destination, self.plots, fps=self.fps)
        plt.rcParams['text.antialiased'] = self._init_alias


class LivePlot(Block):
    def __init__(self, plot_function=None, sleep=0., size=None, **kwargs):
        super().__init__(**kwargs)
        if plot_function is None:
            plot_function = lambda im: viz.show_stars(
                im.data, im.stars_coords if hasattr(im, "stars_coords") else None,
                size=size
                )

        self.plot_function = plot_function
        self.sleep = sleep
        self.display = None
        self.size = size

    def initialize(self, *args):
        from IPython import display as disp
        self.display = disp
        if isinstance(self.size, tuple):
            plt.figure(figsize=self.size)

    def run(self, image):
        self.plot_function(image)
        self.display.clear_output(wait=True)
        self.display.display(plt.gcf())
        time.sleep(self.sleep)
        plt.cla()

    def terminate(self):
        plt.close()


class Get(Block):

    def __init__(self, *names, name="get"):
        super().__init__(name=name)
        self.names = names
        self.values = {name: [] for name in names}

    def run(self, image, **kwargs):
        for name in self.names:
            self.values[name].append(image.__dict__[name])

    def __call__(self, *names):
        if len(names) == 0:
            return self.values
        elif len(names) == 1:
            return self.values[names[0]]
        elif len(names) > 1:
            return [self.values[name] for name in names]


class XArray(Block):

    def __init__(self, *names, name="xarray", raise_error=True, concat_dim="time"):
        super().__init__(name=name)
        self.variables = {name: (dims, []) for dims, name in names}
        self.raise_error = raise_error
        self.xarray = xr.Dataset()
        self.concat_dim = concat_dim

    def run(self, image, **kwargs):
        for name in self.variables:
            try:
                self.variables[name][1].append(image.__getattribute__(name))
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
