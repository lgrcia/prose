from scipy.optimize import minimize
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.nddata import NDData
from photutils.psf import extract_stars
from astropy.stats import gaussian_sigma_to_fwhm
from prose._blocks.base import Block
from prose.console_utils import INFO_LABEL
import matplotlib.pyplot as plt


def image_psf(image, stars, size=15, normalize=False):
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

    stars_in = np.logical_and(
        np.all(stars < np.array(image.shape) - size, axis=1),
        np.all(stars > np.ones(2) * size, axis=1)
    )
    stars = stars[stars_in]

    # with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    stars_tbl = Table([stars[:, 0], stars[:, 1]], names=["x", "y"])
    stars = extract_stars(NDData(data=image), stars_tbl, size=size)
    
    return np.argwhere(stars_in).flatten(), stars

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

    def __init__(self, cutout_size=21, **kwargs):
        super().__init__(**kwargs)
        self.cutout_size = cutout_size
        self.x, self.y = None, None
        self.epsf = None

    @property
    def optimized_model(self):
        return self.model(*self.optimized_params)

    def build_epsf(self, image, stars):
        self.x, self.y = np.indices((self.cutout_size, self.cutout_size))
        return image_psf(image, stars.copy(), size=self.cutout_size)

    def model(self):
        raise NotImplementedError("")

    def nll(self, p):
        ll = np.sum(np.power((self.model(*p) - self.epsf), 2) * self.epsf)
        return ll if np.isfinite(ll) else 1e25
    
    def optimize(self):
        raise NotImplementedError("")

    def sigma_to_fwhm(self, *args):
        return gaussian_sigma_to_fwhm
    
    def stack_method(self, image):
        print("{} global psf FWHM: {:.2f} (pixels)".format(INFO_LABEL, np.mean(image.fwhm)))

    def run(self, image):
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


class Gaussian2D(PSFModel):
    """
    Fit an elliptical 2D Gaussian model to an image effective PSF
    """
    def __init__(self, cutout_size=21, **kwargs):
        super().__init__(cutout_size=21, **kwargs)

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

class Moffat2D(PSFModel):
    """
    Fit an elliptical 2D Moffat model to an image effective PSF
    """
    def __init__(self, cutout_size=21, **kwargs):
        super().__init__(cutout_size=21, **kwargs)
        self.cutout_size = cutout_size
        self.x, self.y = None, None

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
