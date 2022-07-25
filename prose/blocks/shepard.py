from .. import Block
from .psf import good_cutouts, cutouts
from astropy.stats import gaussian_sigma_to_fwhm
from prose.blocks.psf import Gaussian2D
from itertools import product
import numpy as np
from scipy.special import hermite

def shapelet1d(x, n, b=1):
    _x = x / b
    s = (1 / np.sqrt((2 ** n) * np.sqrt(np.pi) * np.math.factorial(n))) * hermite(n)(_x) * np.exp(-(_x ** 2) / 2)
    return (1 / np.sqrt(b)) * s


def shapelet2d(x, y, n1, n2, b=1):
    X, Y = np.meshgrid(x, y)
    return shapelet1d(X, n1, b) * shapelet1d(Y, n2, b)


class Shepard(Block):

    
    def __init__(self, order=4, size=31, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.size = size

        # --------------

        self.Xs = None
        self.group_centers = None
        self.stars_groups = None
        self.stars_coords_groups = None

    def prepare(self, data, stars, fwhm):
        # # computing fwhm
        # good_cuts = good_cutouts(data, stars, r=self.size, upper=40000, lower=1000,
        #                          trim=100)  # TODO: adapt to telescope
        # good_cuts = [c.data / c.data.max() for c in list(good_cuts.values())]
        # fwhm = Gaussian2D(self.size)(np.mean(good_cuts, 0))[0] / gaussian_sigma_to_fwhm

        # building basis
        # --------------
        x = np.arange(self.size)

        def basis(x0, y0):
            if self.size == 0:
                ns = [(0, 0)]
            else:
                ns = product(range(self.order), repeat=2)
            return [*[shapelet2d(x - x0, x - y0, n1, n2, fwhm).flatten() for n1, n2 in ns]]

        stars_groups = []
        _xy_dict = dict(zip(np.arange(len(stars)), stars))

        for i, _xy in _xy_dict.copy().items():
            if i in _xy_dict:
                close = np.nonzero(np.linalg.norm(_xy - stars, axis=1) < self.size / 2)[0]
                if len(close) >= 1:
                    stars_groups.append(close)
                    for c in close:
                        if c in _xy_dict:
                            del _xy_dict[c]

        stars_coords_groups = np.array([stars[s] for s in stars_groups])
        group_centers = np.array([s.mean(0) for s in stars_coords_groups])
        idxs, groups_cutouts = cutouts(data, group_centers, self.size)

        self.stars_groups = [stars_groups[i] for i in idxs]
        self.group_centers = group_centers[idxs]
        self.stars_coords_groups = stars_coords_groups[idxs]

        self.Xs = []
        for i, c in enumerate(groups_cutouts):
            c = groups_cutouts[i]
            scg = self.stars_coords_groups[i] - (c.bbox.ixmin, c.bbox.iymin)
            X = np.vstack([np.ones((self.size, self.size)).flatten(), *[basis(*s) for s in scg]])
            self.Xs.append(X)

    def run(self, image, **kwargs):
        if self.Xs is None:
            self.prepare(image.data, image.stars_coords, image.fwhm)

        fluxes = {}
        idxs, groups_cutouts = cutouts(image.data, self.group_centers, self.size)
        for i, c in enumerate(groups_cutouts):
            w = np.linalg.lstsq(self.Xs[i].T, c.data.flatten())[0]
            scg = self.stars_coords_groups[i]
            for j, _X, _w in zip(self.stars_groups[i], np.split(self.Xs[i][1::], len(scg)), np.split(w[1::], len(scg))):
                fluxes[j] = (_w @ _X).sum()

        image.fluxes = np.array([np.array([fluxes.get(i, np.nan) for i in range(len(image.stars_coords))])])
        image.fluxes_errors = np.sqrt(image.fluxes)
