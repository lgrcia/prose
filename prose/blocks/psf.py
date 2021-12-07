from scipy.optimize import minimize
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.nddata import NDData
from photutils.psf import extract_stars
from astropy.stats import gaussian_sigma_to_fwhm
from ..core import Block
import matplotlib.pyplot as plt
from collections import OrderedDict
from ..utils import fast_binning

def image_psf(image, stars, size=15, normalize=False, return_cutouts=False):
    """
    Get global psf from image using photutils routines

    Parameters
    ----------
    image: np.ndarray or path
    stars: np.ndarray
        stars positions with shape (n,2)
    size: int
        size of the cuts around stars (in pixels)
    normalize: bool, optional
        weather to normalize the cutout, default is False

    Returns
    -------
    np.ndarray of shape (size, size)

    """
    _, cuts = cutouts(image, stars, size=size)
    cuts = cuts.data
    if normalize:
        cuts = [c/np.sum(c) for c in cuts]
    if return_cutouts:
        return np.median(cuts, axis=0), cuts
    else:
        return np.median(cuts, axis=0)


def cutouts(image, stars, size=15):
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
        stars = extract_stars(NDData(data=image), stars_tbl, size=size)
        idxs = np.array([s.id_label for s in stars])
        return idxs, stars
    else:
        stars_tbl = Table(
            data=np.array([stars[0][0], stars[0][1]]),
            names=["x", "y"])
        stars = extract_stars(NDData(data=image), stars_tbl, size=size)
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


class PSFModel(Block):

    def __init__(self, cutout_size=21, save_cutouts=False, **kwargs):
        super().__init__(**kwargs)
        self.cutout_size = cutout_size
        self.save_cutouts = save_cutouts
        self.x, self.y = np.indices((self.cutout_size, self.cutout_size))
        self.epsf = None

    @property
    def optimized_model(self):
        return self.model(*self.optimized_params)

    def build_epsf(self, image, stars):
        return image_psf(image, stars.copy(), size=self.cutout_size, return_cutouts=self.save_cutouts)

    def model(self):
        raise NotImplementedError("")

    def nll(self, p):
        ll = np.sum(np.power((self.model(*p) - self.epsf), 2) * self.epsf)
        return ll if np.isfinite(ll) else 1e25
    
    def optimize(self):
        raise NotImplementedError("")

    def sigma_to_fwhm(self, *args):
        return gaussian_sigma_to_fwhm

    def run(self, image):
        if self.save_cutouts:
            self.epsf, image.cutouts = self.build_epsf(image.data, image.stars_coords)
        else:
            self.epsf = self.build_epsf(image.data, image.stars_coords)
        image.fwhmx, image.fwhmy, image.theta = self.optimize()
        image.fwhm = np.mean([image.fwhmx, image.fwhmy])
        image.psf_sigma_x = image.fwhmx / self.sigma_to_fwhm()
        image.psf_sigma_y = image.fwhmy / self.sigma_to_fwhm()
        image.header["FWHM"] = image.fwhm
        image.header["FWHMX"] = image.fwhmx
        image.header["FWHMY"] = image.fwhmy
        image.header["PSFANGLE"] = image.theta
        image.header["FWHMALG"] = self.__class__.__name__

    def show_residuals(self):
        plt.imshow(self.epsf - self.optimized_model)
        plt.colorbar()
        ax = plt.gca()
        plt.text(0.05, 0.05, "$\Delta f=$ {:.2f}%".format(100*np.sum(np.abs(self.epsf - self.optimized_model))/np.sum(self.epsf)), 
        fontsize=14, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, c="w")

    def __call__(self, data):
        self.epsf = data
        return self.optimize()


class FWHM(PSFModel):
    """
    Fast empirical FWHM (based on Arielle Bertrou-Cantou's idea)
    """

    def __init__(self, cutout_size=51, **kwargs):
        super().__init__(cutout_size=cutout_size, **kwargs)
        Y, X = np.indices((self.cutout_size,self.cutout_size))
        x = y = self.cutout_size/2
        self.radii = (np.sqrt((X - x) ** 2 + (Y - y) ** 2)).flatten()

    def optimize(self):
        psf = self.epsf.copy()
        psf -= np.min(psf)
        pixels = psf.flatten()
        binned_radii, binned_pixels, _ = fast_binning(self.radii, pixels, bins=1)
        fwhm = 2*binned_radii[np.flatnonzero(binned_pixels > np.max(binned_pixels)/2)[-1]]
        return fwhm, fwhm, 0

class FastGaussian(PSFModel):
    """
    Fit a symetric 2D Gaussian model to an image effective PSF
    """
    def __init__(self, cutout_size=21, **kwargs):
        super().__init__(cutout_size=cutout_size, **kwargs)

    def model(self, height, s, m):
        dx = self.x - self.cutout_size/2
        dy = self.y - self.cutout_size/2
        psf = height * np.exp(-((dx/(2*s))**2 + (dy/(2*s))**2))
        return psf + m

    def optimize(self):
        p0 = [np.max(self.epsf), 4, np.min(self.epsf)]
        min_sigma = 0.5
        bounds = [
            (0, np.infty),
            (min_sigma, np.infty),
            (0, np.mean(self.epsf)),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = minimize(self.nll, p0, bounds=bounds).x
            self.optimized_params = params
            return params[1]*self.sigma_to_fwhm(), params[1]*self.sigma_to_fwhm(), 0

    def citations(self):
        return "scipy", "photutils"


class Gaussian2D(PSFModel):
    """
    Fit an elliptical 2D Gaussian model to an image effective PSF
    """
    def __init__(self, cutout_size=21, **kwargs):
        super().__init__(cutout_size=cutout_size, **kwargs)

    def model(self, height, xo, yo, sx, sy, theta, m):
        dx = self.x - xo
        dy = self.y - yo
        a = (np.cos(theta)**2)/(2*sx**2) + (np.sin(theta)**2)/(2*sy**2)
        b = -(np.sin(2*theta))/(4*sx**2) + (np.sin(2*theta))/(4*sy**2)
        c = (np.sin(theta)**2)/(2*sx**2) + (np.cos(theta)**2)/(2*sy**2)
        psf = height * np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))
        return psf + m

    def optimize(self):
        p0 = moments(self.epsf)
        x0, y0 = p0[1], p0[2]
        min_sigma = 0.5
        bounds = [
            (0, np.infty),
            (x0 - 3, x0 + 3),
            (y0 - 3, y0 + 3),
            (min_sigma, np.infty),
            (min_sigma, np.infty),
            (0, 4),
            (0, np.mean(self.epsf)),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = minimize(self.nll, p0, bounds=bounds).x
            self.optimized_params = params
            return params[3]*self.sigma_to_fwhm(), params[4]*self.sigma_to_fwhm(), params[-2]

    def citations(self):
        return "scipy", "photutils"


class Moffat2D(PSFModel):
    """
    Fit an elliptical 2D Moffat model to an image effective PSF
    """
    def __init__(self, cutout_size=21, **kwargs):
        super().__init__(cutout_size=cutout_size, **kwargs)

    def model(self, a, x0, y0, sx, sy, theta, b, beta):
    # https://pixinsight.com/doc/tools/DynamicPSF/DynamicPSF.html
        dx_ = self.x - x0
        dy_ = self.y - y0
        dx = dx_*np.cos(theta) + dy_*np.sin(theta)
        dy = -dx_*np.sin(theta) + dy_*np.cos(theta)
        
        return b + a / np.power(1 + (dx/sx)**2 + (dy/sy)**2, beta) 

    def sigma_to_fwhm(self):
        return 2*np.sqrt(np.power(2, 1/self.optimized_params[-1]) - 1)
    
    def optimize(self):
        p0 = list(moments(self.epsf))
        p0.append(1)
        x0, y0 = p0[1], p0[2]
        min_sigma = 0.5
        bounds = [
            (0, np.infty),
            (x0 - 3, x0 + 3),
            (y0 - 3, y0 + 3),
            (min_sigma, np.infty),
            (min_sigma, np.infty),
            (0, 4),
            (0, np.mean(self.epsf)),
            (1, 8),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = minimize(self.nll, p0, bounds=bounds).x
            self.optimized_params = params
            sm = self.sigma_to_fwhm()
            return params[3]*sm, params[4]*sm, params[-2]

    def citations(self):
        return "scipy", "photutils"


class KeepGoodStars(Block):

    def __init__(self, n=-1, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def run(self, image, n=-1):
        good_stars = self(image.data, image.stars_coords)
        image.stars_coords = good_stars

    def __call__(self, data, stars):
        i, _stars = cutouts(data, stars, size=21)
        #good = np.array([shapiro(s.data).statistic for s in _stars]) > 0.33
        good = np.array([np.std(s.data) for s in _stars]) > 1000
        return stars[i][np.argwhere(good).squeeze()][0:self.n]