from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.nddata import NDData
from photutils.psf import extract_stars
from astropy.stats import gaussian_sigma_to_fwhm
from .. import Block
import matplotlib.pyplot as plt
from collections import OrderedDict
from ..utils import fast_binning
from scipy.stats import shapiro
from scipy.interpolate import interp1d
from ..console_utils import info

__all__ = ["MedianPSF", "Cutouts"]

def cutouts(image, stars, size=15, same=True):
    """Custom version to extract stars cutouts

    Parameters
    ----------
    Parameters
    ----------
    image: np.ndarray or path
    stars: np.ndarray
        stars positions with shape (n,2)
    size: int
        size of the cuts around stars (in pixels), by default 15

    Returns
    -------
    np.ndarray of shape (size, size)
    
    """
    if isinstance(image, str):
        image = fits.getdata(image)

    warnings.simplefilter("ignore")
    if np.shape(stars) > (1,2):
        stars_tbl = Table(
            [stars[:, 0], stars[:, 1], np.arange(len(stars))],
            names=["x", "y", "id"])
        _stars = [None]*len(stars)
        stars = extract_stars(NDData(data=image), stars_tbl, size=size)
        idxs = np.array([s.id_label for s in stars])
        if same:
            for i, s in enumerate(stars):
                _stars[idxs[i]] = s
        else:
            _stars = stars
        return idxs, _stars
    else:
        stars_tbl = Table(
            data=np.array([stars[0][0], stars[0][1]]),
            names=["x", "y"])
        stars = ([0], [extract_stars(NDData(data=image), stars_tbl, size=size)])
        return stars


def good_cutouts(image, xy, r=30, upper=40000, lower=1000, trim=100):
    idxs, _cuts = cutouts(image, xy, r)
    cuts = OrderedDict(zip(idxs, _cuts))
    peaks = [cutout.data.max() for cutout in cuts.values()]

    for i, cutout in cuts.copy().items():
        if i in cuts:
            peak = cutout.data.max()
            center = cutout.center

            # removing saturated and faint stars
            if peak > upper or peak < lower:
                del cuts[i]

            # removing stars on borders
            elif np.any(center < [trim, trim]) or np.any(center > np.array(image.shape) - trim):
                del cuts[i]

            # removing close stars
            closest = idxs[np.nonzero(np.linalg.norm(center - xy[idxs], axis=1) < r)[0]]
            if len(closest) > 1:
                for j in closest:
                    if j in cuts:
                        del cuts[j]

    return cuts


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    height = data.max()
    background = data.min()
    data = data-np.min(data)
    total = data.sum()
    x, y = np.indices(data.shape)
    x = (x * data).sum() / total
    y = (y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    width_x /= gaussian_sigma_to_fwhm
    width_y /= gaussian_sigma_to_fwhm
    return height, x, y, width_x, width_y, 0.0, background

class Cutouts(Block):
    """Extract square image cutouts centered on ``Image.stars_coords``

    |read| ``Image.stars_coords``

    |write| 
    - ``Image.cutouts``: cutouts images
    - ``Image.cutouts_idxs``: indexes of stars_coords corresponding to each cutout 
    - ``Image.stars_coords`` if ``clean`` is ``True``

    Cutouts are sometimes called "imagette" and represent small square portions of the image centered on specific points.

    Parameters
    ----------
    size : int, optional
       square side length of the cutout in pixel, by default 21
    """
    
    def __init__(self, size=21, clean=True, name=None):
        super().__init__(name=name)
        self.size = size
        self.clean = clean

    def run(self, image):
        image.cutouts_idxs, image.cutouts = cutouts(image.data, image.stars_coords, size=self.size)
        if self.clean:
            if hasattr(image, "stars_coords"):
                image.sources = image.sources[image.cutouts_idxs]
            image.cutouts = [image.cutouts[i] for i in image.cutouts_idxs]
            image.cutouts_idxs = np.arange(len(image.cutouts))

    @property
    def citations(self):
        return "photutils"

class MedianPSF(Block):
    """Get median psf from image.

    |read| ``Image.cutouts``

    If ``Image.cutouts`` is not present (because a ``Cutouts`` block has not been included in the sequence), set cutout_size, which instantiate a ``Cutouts`` block within this one

    Parameters
    ----------
    cutout_size : int, optional
        size of the cutouts used to compute the global PSF, by default None which mean the Image.cutouts are used
    """
    
    def __init__(self, cutout_size=None, stars=None, n=None, **kwargs):
        super().__init__(**kwargs)
        self.cutout_block = None
        if cutout_size is not None:
            self.cutout_block = Cutouts(size=cutout_size)
        self.stars = stars
        self.n = n
        
    def run(self, image):
        if self.cutout_block is not None:
            image = self.cutout_block(image)
        normalized_cutouts = [c/np.sum(c) for c in image.cutouts if c is not None]
        if self.stars is None:
            image.psf =  np.median(normalized_cutouts, axis=0)
        elif self.n is not None:
            assert self.stars is None, "Either 'n' or 'stars' must be set, not both"
            image.psf =  np.median(normalized_cutouts, axis=0)
        else:
            image.psf =  np.median(normalized_cutouts[0:self.n], axis=0)


class _PSFModel(Block):

    def __init__(self, reference=None, **kwargs):
        super().__init__(**kwargs)
        self.x = self.y = None
        self.p0 = None
        self.reference = reference

        if reference is not None:
            if self.verbose:
                info(f"{self.__class__.__name__} optimizing reference PSF ...")
            self.from_cutouts(reference)
            self.p0 = self.optimize(reference.psf)

    def from_cutouts(self, image):
        self.cutout_size = image.cutouts[image.cutouts_idxs[0]].shape[0]
        self.x, self.y = np.indices((self.cutout_size, self.cutout_size))

    @property
    def optimized_model(self):
        return self.model(*self.optimized_params)

    def model(self, *args):
        raise NotImplementedError("")

    def nll(self, p, psf):
        ll = np.sum(np.power((self.model(*p) - psf), 2) * psf)
        return ll if np.isfinite(ll) else 1e25
    
    def optimize(self, psf):
        raise NotImplementedError("")

    def fwhm(self, params):
        raise NotImplementedError("")

    def sigma_to_fwhm(self, *args):
        return gaussian_sigma_to_fwhm

    def run(self, image):
        if self.x is None and hasattr(image, "cutouts"):
            self.from_cutouts(image)

        if self.reference is not None:
            assert np.all(image.shape == self.reference.shape), "reference shape differs from image shape"

        image.psf_models_params = self.optimize(image.psf)
        image.psf_model = self.model(*np.atleast_1d(image.psf_models_params))
        image.fwhmx, image.fwhmy, image.theta = self.fwhm(image.psf_models_params)
        image.fwhm = np.mean([image.fwhmx, image.fwhmy])
        image.psf_model_block = self.__class__.__name__

    def _minimize(self, psf, p0, bounds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.p0 is None:
                return differential_evolution(self.nll, bounds, x0=p0, args=(psf,)).x
            else:
                return minimize(self.nll, self.p0, bounds=bounds, args=(psf,)).x
    
    @property
    def citations(self):
        return "scipy"


class FWHM(_PSFModel):
    """
    Fast empirical FWHM

    To be used after a PSF building block
    
    |read| ``Image.psf``

    |write|

    - ``Image.psf_models_params``
    - ``Image.psf_model``
    - ``Image.fwhmx``
    - ``Image.fwhm``
    - ``Image.psf_model_block``
    
    (based on Arielle Bertrou-Cantou's idea)
    """
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def from_cutouts(self, image):
        super().from_cutouts(image)
        x = y = self.cutout_size/2
        self.radii = (np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)).flatten()
        self.interp_radii = np.linspace(0, np.max(self.radii), 500)

    def model(self, *args):
        return None

    def fwhm(self, param):
        return param, param, 0

    def optimize(self, psf):
        psf -= np.min(psf)
        pixels = psf.flatten()
        bin_radii, bin_pixels, _ = fast_binning(self.radii, pixels, bins=1)
        interp_bin_pixels = interp1d(bin_radii, bin_pixels,fill_value="extrapolate")(self.interp_radii)
        interp_bin_pixels -= np.min(interp_bin_pixels)
        fwhm = 2*self.interp_radii[np.flatnonzero(interp_bin_pixels > np.max(interp_bin_pixels)/2)[-1]]
        return fwhm

    def plot_radial_psf(self, psf):
        x, y = np.array(psf.shape)/2
        Y, X = np.indices(psf.shape)
        radii = (np.sqrt((X - x) ** 2 + (Y - y) ** 2)).flatten()
        idxs = np.argsort(radii)
        radii = radii[idxs]
        pixels = psf.flatten()
        pixels = pixels[idxs]

        binned_radii, binned_pixels, _ = fast_binning(radii, pixels, bins=1)

        fig = plt.figure(figsize=(9.5, 4))
        plt.plot(radii, pixels, "o", fillstyle='none', c="0.7", ms=4)
        plt.plot(binned_radii, binned_pixels, c="k")
        plt.xlabel("distance from center (pixels)")
        plt.ylabel("ADUs")
        f = 0

class FastGaussian(_PSFModel):
    """
    Fit a symetric 2D Gaussian model to an image effective PSF
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def model(self, height, s, m):
        dx = self.x - self.cutout_size/2
        dy = self.y -  self.cutout_size/2
        psf = height * np.exp(-((dx/(2*s))**2 + (dy/(2*s))**2))
        return psf + m

    def fwhm(self, params):
        return params[1]*self.sigma_to_fwhm(), params[1]*self.sigma_to_fwhm(), 0

    def optimize(self, psf):
        if self.reference is None:
            p0 = [np.max(psf), 4, np.min(psf)]
        else:
            p0 = self.p0
        min_sigma = 0.5
        bounds = [
            (0, 1.),
            (min_sigma, 1e20),
            (0, np.mean(psf)),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = minimize(self.nll, p0, bounds=bounds, args=(psf)).x
            return params

    @property
    def citations(self):
        return "scipy", "photutils"


class Gaussian2D(_PSFModel):
    r"""
    Fit an elliptical 2D Gaussian model to an image effective PSF

    To be used after a PSF building block
    
    |read| ``Image.psf``

    |write|

    - ``Image.psf_models_params``
    - ``Image.psf_model``
    - ``Image.fwhmx``
    - ``Image.fwhm``
    - ``Image.psf_model_block``
    
    
    PSF model is

    .. math::

        f(x, y|A, x_0, y_0, \sigma_x, \sigma_y, \theta, b) = - A \exp\left(\frac{(x'-x'_0)^2}{2\sigma_x^2} \frac{(y'-y'_0)^2}{2\sigma_y^2}\right) + b

    .. math::

        \text{with}\quad \begin{gather*}
        x' = xcos(\theta) + ysin(\theta) \\
        y' = -xsin(\theta) + ycos(\theta)
        \end{gather*}


    is fitted from an effective psf. :code:`scipy.optimize.minimize` is used to minimize :math:`\chi ^2` from data. Initial parameters are found using the moments of the `effective psf <https://photutils.readthedocs.io/en/stable/epsf.html>`_. This method is 4 times faster than :code:`photutils.centroids.fit_2dgaussian` and lead to similar results.

    Example
    -------

    We start by loading an example image and buidling its median psf

    .. jupyter-execute::

        from prose import blocks, Sequence
        from prose.tutorials import example_image

        # our example image
        image = example_image()

        # Sequence to build image PSF
        sequence = Sequence([
            blocks.SegmentedPeaks(),  # stars detection
            blocks.Cutouts(),
            blocks.MedianPSF(),       # building PSF
        ])

        sequence.run([image])   

    We can now apply the Gaussian2D block to the image in order to model its PSF

    .. jupyter-execute::

        import matplotlib.pyplot as plt

        block = blocks.psf.Gaussian2D()
        image = block(image)

    and vizualise the result

    .. jupyter-execute::

        from prose import viz

        print(f"model: {image.psf_model_block}")
        print("fwhmx, fwhmy, theta: " + ", ".join([f"{p:.2f}" for p in block.fwhm(image.psf_models_params)]))

        plt.figure(figsize=(12, 5))

        plt.subplot(131)
        plt.imshow(image.psf)
        plt.title("PSF")

        plt.subplot(132)
        plt.imshow(image.psf_model)
        plt.title(f"PSF model ({image.psf_model_block})")

        plt.subplot(133)
        residuals = image.psf - image.psf_model
        ax = plt.imshow(residuals)
        plt.title("residuals")
        viz.add_colorbar(ax)

        plt.tight_layout()

    """

    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def model(self, height, xo, yo, sx, sy, theta, m):
        dx = self.x - xo
        dy = self.y - yo
        a = (np.cos(theta)**2)/(2*sx**2) + (np.sin(theta)**2)/(2*sy**2)
        b = -(np.sin(2*theta))/(4*sx**2) + (np.sin(2*theta))/(4*sy**2)
        c = (np.sin(theta)**2)/(2*sx**2) + (np.cos(theta)**2)/(2*sy**2)
        psf = height * np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))
        return psf + m

    def fwhm(self, params):
        return params[3]*self.sigma_to_fwhm(), params[4]*self.sigma_to_fwhm(), params[-2]

    def optimize(self, psf):
        p0 = moments(psf)
        w = np.max(psf.shape)
        bounds = [
            (0, 1.),
            (0, w),
            (0, w),
            (0.5, w),
            (0.5, w),
            (-np.pi, np.pi),
            (0, np.mean(psf)),
        ]

        return self._minimize(psf, p0, bounds)

    @property
    def citations(self):
        return "scipy"


class Moffat2D(_PSFModel):
    r"""
    Fit an elliptical 2D Moffat model to an image effective PSF

    To be used after a PSF building block
    
    |read| ``Image.psf``

    |write|

    - ``Image.psf_models_params``
    - ``Image.psf_model``
    - ``Image.fwhmx``
    - ``Image.fwhm``
    - ``Image.psf_model_block``
    
    
    PSF model is

    .. math::   

        f(x, y|A, x_0, y_0, \sigma_x, \sigma_y, \theta, \beta, b) = \frac{A}{\left(1 + \frac{x'-x'_0}{\sigma_x^2} + \frac{y'-y'_0}{\sigma_y^2}\right)^\beta} + b

    .. math::

        \text{with}\quad \begin{gather*}
        x' = xcos(\theta) + ysin(\theta) \\
        y' = -xsin(\theta) + ycos(\theta)
        \end{gather*}

    is fitted from an effective psf. :code:`scipy.optimize.minimize` is used to minimize :math:`\chi ^2` from data. Initial parameters are found using the moments of the `effective psf <https://photutils.readthedocs.io/en/stable/epsf.html>`_. 


    Example
    -------

    We start by loading an example image and buidling its median psf

    .. jupyter-execute::

        from prose import blocks, Sequence
        from prose.tutorials import example_image

        # our example image
        image = example_image()

        # Sequence to build image PSF
        sequence = Sequence([
            blocks.SegmentedPeaks(),  # stars detection
            blocks.Cutouts(),
            blocks.MedianPSF(),       # building PSF
        ])

        sequence.run([image])   

    We can now apply the Moffat2D block to the image in order to model its PSF

    .. jupyter-execute::

        import matplotlib.pyplot as plt

        block = blocks.psf.Moffat2D()
        image = block(image)

    and vizualise the result

    .. jupyter-execute::

        from prose import viz

        print(f"model: {image.psf_model_block}")
        print("fwhmx, fwhmy, theta: " + ", ".join([f"{p:.2f}" for p in block.fwhm(image.psf_models_params)]))

        plt.figure(figsize=(12, 5))

        plt.subplot(131)
        plt.imshow(image.psf)
        plt.title("PSF")

        plt.subplot(132)
        plt.imshow(image.psf_model)
        plt.title(f"PSF model ({image.psf_model_block})")

        plt.subplot(133)
        residuals = image.psf - image.psf_model
        ax = plt.imshow(residuals)
        plt.title("residuals")
        viz.add_colorbar(ax)

        plt.tight_layout()

    """

    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def model(self, a, x0, y0, sx, sy, theta, b, beta):
    # https://pixinsight.com/doc/tools/DynamicPSF/DynamicPSF.html
        dx_ = self.x - x0
        dy_ = self.y - y0
        dx = dx_*np.cos(theta) + dy_*np.sin(theta)
        dy = -dx_*np.sin(theta) + dy_*np.cos(theta)
        
        return b + a / np.power(1 + (dx/sx)**2 + (dy/sy)**2, beta) 

    def fwhm(self, params):
        sm = self.sigma_to_fwhm(params[-1])
        return params[3]*sm, params[4]*sm, params[-2]

    def sigma_to_fwhm(self, beta):
        return 2*np.sqrt(np.power(2, 1/beta) - 1)
    
    def optimize(self, psf):
        p0 = list(moments(psf)) + [1]
        w = np.max(psf.shape)
        bounds = [
            (0, 1.),
            (0, w),
            (0, w),
            (0.5, w),
            (0.5, w),
            (-np.pi, np.pi),
            (0, np.mean(psf)),
            (1, 8),
        ]

        return self._minimize(psf, p0, bounds)

    @property
    def citations(self):
        return "scipy"


class KeepGoodStars(Block):

    
    def __init__(self, stat=0, n=-1, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        if isinstance(stat, int):
            if stat == 0:
                def stat(im):
                    return np.std(im) > 1000 
            elif stat == 1:
                def stat(im):
                    return shapiro(im).statistic > 0.03
        
        self.stat = stat

    def run(self, image, n=-1):
        i, stars = image.cutouts_idxs, image.cutouts
        good = np.array([self.stat(s.data) for s in stars])
        good_stars = image.stars_coords[i][np.argwhere(good).squeeze()][0:self.n]
        image.stars_coords = good_stars


class HFD(FWHM):

    # https://www.focusmax.org/Documents_V4/ITS%20Paper.pdf

    def __init__(self, order=4, **kwargs):
        super().__init__(**kwargs)
        self.sorted_idxs = None
        self.order = order

    def from_cutouts(self, image):
        super().from_cutouts(image)
        self.sorted_idxs = np.argsort(self.radii)
        self.sorted_radii = self.radii[self.sorted_idxs]
        self.cum_radii = np.arange(len(self.sorted_radii))
        self.X_radii_bkg = np.array([
            np.ones_like(self.sorted_radii),
            self.cum_radii
        ])

    def optimize(self, psf):
        psf -= np.percentile(psf, 10)
        pixels = psf.flatten()[self.sorted_idxs]
        cumsum = np.cumsum(pixels)
        bkg_idxs = np.flatnonzero(pixels < np.percentile(pixels, 5))
        # removing background in cumsum
        bkg_X = np.vstack([
            np.ones_like(bkg_idxs),
            bkg_idxs
        ])
        w = np.linalg.lstsq(bkg_X.T, cumsum[bkg_idxs])[0]
        cumsum -= w@self.X_radii_bkg
        cumsum = cumsum - cumsum[0]
        w = np.linalg.lstsq(self.X_radii, self.sorted_radii)[0]
        fwhm = self.sorted_radii[np.argmin(np.abs(cumsum - np.max(cumsum)/2))]
        return fwhm
