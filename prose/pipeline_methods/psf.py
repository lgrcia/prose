from scipy.optimize import minimize, leastsq
from scipy.optimize import curve_fit
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.nddata import NDData
from photutils.psf import extract_stars


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

    Returns
    -------
    np.ndarray of shape (size, size)

    """
    cuts = cutouts(image, stars, size=size).data
    if normalize:
        cuts = [c/np.sum(c) for c in cuts]
    return np.median(cuts, axis=0)

def cutouts(image, stars, size=15):
    if isinstance(image, str):
        image = fits.getdata(image)

    stars = stars[np.all(stars < np.array(image.shape) - size, axis=1)]
    stars = stars[np.all(stars > np.ones(2) * size, axis=1)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stars_tbl = Table([stars[:, 0], stars[:, 1]], names=["x", "y"])
        stars = extract_stars(NDData(data=image), stars_tbl, size=size)
    
    return stars

def gaussian_2d(x, y, height, xo, yo, sx, sy, theta, m):
    dx = x - xo
    dy = y - yo
    a = (np.cos(theta)**2)/(2*sx**2) + (np.sin(theta)**2)/(2*sy**2)
    b = -(np.sin(2*theta))/(4*sx**2) + (np.sin(2*theta))/(4*sy**2)
    c = (np.sin(theta)**2)/(2*sx**2) + (np.cos(theta)**2)/(2*sy**2)
    psf = height * np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))
    return psf + m


def nll_gaussian_2d(p, _im, x, y):
    ll = np.sum(np.power((gaussian_2d(x, y, *p) - _im), 2) * _im)
    return ll if np.isfinite(ll) else 1e25


def fit_gaussian2_nonlin(image, return_p0_bounds=False):
    x, y = np.indices(image.shape)
    p0 = moments(image)
    x0, y0 = p0[1], p0[2]
    min_sigma = 0.5
    bounds = [
        (0, np.infty),
        (x0 - 3, x0 + 3),
        (y0 - 3, y0 + 3),
        (min_sigma, np.infty),
        (min_sigma, np.infty),
        (0, 4),
        (0, np.mean(image)),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if return_p0_bounds:
            return (
                minimize(
                    nll_gaussian_2d, p0, args=(image, x, y), bounds=bounds
                ).x,
                p0,
                bounds,
            )
        else:
            return minimize(
                nll_gaussian_2d, p0, args=(image, x, y), bounds=bounds
            ).x


def gaussian(x, y, height, center_x, center_y, width_x, width_y, rotation, m):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)
    center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    xp = x * np.cos(rotation) - y * np.sin(rotation)
    yp = x * np.sin(rotation) + y * np.cos(rotation)
    g = height * np.exp(
        -(((center_x - xp) / width_x) ** 2 +
          ((center_y - yp) / width_y) ** 2) / 2.) + m

    return g


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    height = data.max()
    background = data.min()
    data = data-np.min(data)
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    width_x /= (2 * np.sqrt(2 * np.log(2)))
    width_y /= (2 * np.sqrt(2 * np.log(2)))
    return height, x, y, width_x, width_y, 0.0, background


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*np.indices(data.shape), *p) - data)
    p, success = leastsq(errorfunction, params)
    return p


def gaussian1d(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def photutil_epsf(image, stars_positions, extract_size=21, ndim=2):
    with warnings.catch_warnings():
        epsf = image_psf(image, stars_positions, extract_size)

        midpsfx = epsf[int(epsf.shape[0] / 2)]
        x = np.arange(len(midpsfx))
        poptx, _ = curve_fit(
            gaussian1d, x, midpsfx, p0=[np.max(midpsfx), int(epsf.shape[0] / 2), 10]
        )

        if ndim == 2:
            midpsfy = epsf[:, int(epsf.shape[1] / 2)]
            y = np.arange(len(midpsfy))
            popty, _ = curve_fit(
                gaussian1d, y, midpsfy, p0=[np.max(midpsfy), int(epsf.shape[1] / 2), 10]
            )
        else:
            popty = poptx

        return (
            2 * np.sqrt(2 * np.log(2)) * np.abs(poptx[2]),
            2 * np.sqrt(2 * np.log(2)) * np.abs(popty[2]),
        )


def fit_gaussian2d_linear(image, stars, size=21):
    cut = image_psf(image, stars, size)
    p = fitgaussian(cut)
    return p[3]*(2*np.sqrt(2*np.log(2))), p[4]*(2*np.sqrt(2*np.log(2))), p[-2]


def fit_gaussian2d(image, stars, size=21):
    cut = image_psf(image, stars, size)
    p = fit_gaussian2_nonlin(cut)
    return (2*np.sqrt(2*np.log(2)))*p[3], (2*np.sqrt(2*np.log(2)))*p[4], p[-2]