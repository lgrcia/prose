import numpy as np
from prose import FitsManager, Block
from prose.blocks.psf import moments
from prose import Unit
from prose.blocks import DAOFindStars
import warnings
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class CentroidCheck(Block):

    def __init__(self, star=0, cutout_size=21, **kwargs):
        super().__init__(**kwargs)
        self.ref_star = star
        self.star = star
        assert cutout_size % 2 == 1, "cutou_size must be odd"
        self.cutout_size = cutout_size
        self.stars_coords = []
        self.init_positions = []
        self.fitted_positions = []
        self.interpolated_positions = []
        self.x, self.y = np.indices((cutout_size, cutout_size))

    def initialize(self, fits_manager):
        pass

    def run(self, image):
        ref_x0_int, ref_y0_int = image.stars_coords[self.ref_star].astype(int)
        dx = dy = int(self.cutout_size / 2)
        cutout = image.data[ref_y0_int - dy:ref_y0_int + dy + 1, ref_x0_int - dx:ref_x0_int + dx + 1]
        dx0_ref, dy0_ref = self.optimize(cutout)

        x0_int, y0_int = image.stars_coords[self.star].astype(int)
        x0_init, y0_init = image.stars_coords[self.star]
        dx = dy = int(self.cutout_size / 2)
        cutout = image.data[y0_int - dy:y0_int + dy + 1, x0_int - dx:x0_int + dx + 1]
        dx0_fit, dy0_fit = self.optimize(cutout)

        self.init_positions.append([x0_int, y0_int])
        self.fitted_positions.append([x0_int - dx + dx0_fit, y0_int - dy + dy0_fit])

    def model(self, a, x0, y0, sx, sy, theta, b, beta):
        # https://pixinsight.com/doc/tools/DynamicPSF/DynamicPSF.html
        dx_ = self.x - x0
        dy_ = self.y - y0
        dx = dx_ * np.cos(theta) + dy_ * np.sin(theta)
        dy = -dx_ * np.sin(theta) + dy_ * np.cos(theta)

        return b + a / np.power(1 + (dx / sx) ** 2 + (dy / sy) ** 2, beta)

    def nll(self, p, image):
        ll = np.sum(np.power((self.model(*p) - image), 2) * image)
        return ll if np.isfinite(ll) else 1e25

    def optimize(self, image):
        p0 = list(moments(image))
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
            (0, np.mean(image)),
            (1, 8),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = minimize(self.nll, p0, bounds=bounds, args=(image)).x
            return params[1], params[2]

    def terminate(self):
        self.init_positions = np.array(self.init_positions)
        self.interpolated_positions = np.array(self.interpolated_positions)
        self.fitted_positions = np.array(self.fitted_positions)

    def plot(self, c="C0"):
        plt.plot(*(self.fitted_positions.T - self.init_positions.T), ".", c=c, label="inital positions error",
                 alpha=0.2)
        plt.legend()


class CentroidDiagnostic(Unit):
    """
    This tools performs accurate centroid estimate of a star against positions given in its stars_coords
    """

    def __init__(self, fits_manager, star, blocks):

        if isinstance(blocks, Block):
            blocks = [blocks]

        if isinstance(fits_manager, str):
            fits_manager = FitsManager(fits_manager, image_kw="reduced", verbose=False)

        default_methods = [
            #DAOFindStars(stack=True, name="detection")
            *blocks,
            CentroidCheck(star=star, name="centroid check")
        ]

        super().__init__(default_methods, fits_manager, "check", files="reduced", show_progress=True)

        self.run()

    def plot(self, **kwargs):
        self.blocks_dict["centroid check"].plot(**kwargs)