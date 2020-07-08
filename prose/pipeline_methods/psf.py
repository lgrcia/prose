from scipy.optimize import minimize, leastsq
from scipy.optimize import curve_fit
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.nddata import NDData
from photutils.psf import extract_stars
from prose.characterization import Characterize
from astropy.stats import gaussian_sigma_to_fwhm


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
    cuts = cutouts(image, stars, size=size).data
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

    stars = stars[np.all(stars < np.array(image.shape) - size, axis=1)]
    stars = stars[np.all(stars > np.ones(2) * size, axis=1)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stars_tbl = Table([stars[:, 0], stars[:, 1]], names=["x", "y"])
        stars = extract_stars(NDData(data=image), stars_tbl, size=size)
    
    return stars

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


class NonLinearGaussian2D(Characterize):

    def __init__(self, cutout_size=21):
        self.cutout_size = cutout_size
        self.x, self.y = None, None

    def build_epsf(self, image, stars):
        self.x, self.y = np.indices((self.cutout_size, self.cutout_size))
        return image_psf(image, stars, size=self.cutout_size)

    def nll_gaussian_2d(self, p):
        ll = np.sum(np.power((self.model(*p) - self.epsf), 2) * self.epsf)
        return ll if np.isfinite(ll) else 1e25

    def model(self, height, xo, yo, sx, sy, theta, m):
        dx = self.x - xo
        dy = self.y - yo
        a = (np.cos(theta)**2)/(2*sx**2) + (np.sin(theta)**2)/(2*sy**2)
        b = -(np.sin(2*theta))/(4*sx**2) + (np.sin(2*theta))/(4*sy**2)
        c = (np.sin(theta)**2)/(2*sx**2) + (np.cos(theta)**2)/(2*sy**2)
        psf = height * np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))
        return psf + m

    def nll(self, p):
        ll = np.sum(np.power((self.model(*p) - self.epsf), 2) * self.epsf)
        return ll if np.isfinite(ll) else 1e25
    
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
            params = minimize(self.nll_gaussian_2d, p0, bounds=bounds).x
            self.optimized_params = params
            return params[3]*gaussian_sigma_to_fwhm, params[4]*gaussian_sigma_to_fwhm, params[-2]

    def run(self, image, stars):
        self.epsf = self.build_epsf(image, stars)
        return self.optimize()
