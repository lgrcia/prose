from skimage.measure import label, regionprops
import numpy as np
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from .registration import clean_stars_positions
from ..core import Block
from ..blocks.psf import cutouts
from ..utils import register_args

try:
    from sep import extract
except:
    raise AssertionError("Please install sep")


# TODO: when __call__, if data is prose.Image then run normally (like Block.run(Image)), if data is Image.data return products

class StarsDetection(Block):
    """Base class for stars detection.
    """
    def __init__(self, n_stars=None, sort=True, min_separation=None, check_nans=False, **kwargs):
        super().__init__(**kwargs)
        self.n_stars = n_stars
        self.sort = sort
        self.min_separation = min_separation
        self.last_coords = None

        self.check_nans = check_nans

    def single_detection(self, data):
        """
        Running detection on single or multiple images
        """
        raise NotImplementedError("method needs to be overidden")

    def run(self, image, **kwargs):
        data = np.nan_to_num(image.data, 0) if self.check_nans else image.data
        coordinates, peaks = self(data)

        if coordinates is not None:
            image.stars_coords = coordinates
            image.peaks = peaks
            self.last_coords = coordinates

        else:
            image.discard = True

    def __call__(self, data):
        coordinates, fluxes = self.single_detection(data)

        if len(coordinates) > 2:
            if self.sort:
                coordinates = coordinates[np.argsort(fluxes)[::-1]]
            if self.n_stars is not None:
                coordinates = coordinates[0:self.n_stars]
            if self.min_separation is not None:
                coordinates = clean_stars_positions(coordinates, tolerance=self.min_separation)

            return coordinates, fluxes

        else:
            return None, None


class DAOFindStars(StarsDetection):
    """
    DAOPHOT stars detection with :code:`photutils` implementation.

    |write| ``Image.stars_coords``
    
    Parameters
    ----------
    sigma_clip : float, optional
        sigma clipping factor used to evaluate background, by default 2.5
    lower_snr : int, optional
        minimum snr (as source_flux/background), by default 5
    fwhm : int, optional
        typical fwhm of image psf, by default 5
    n_stars : int, optional
        maximum number of stars to consider, by default None
    min_separation : float, optional
        minimum separation between sources, by default 5.0. If less than that, close sources are merged 
    sort : bool, optional
        wether to sort stars coordinates from the highest to the lowest intensity, by default True
    """
    @register_args
    def __init__(self, sigma_clip=2.5, lower_snr=5, fwhm=5, **kwargs):
        super().__init__(**kwargs)
        self.sigma_clip = sigma_clip
        self.lower_snr = lower_snr
        self.fwhm = fwhm

    def single_detection(self, data):
        mean, median, std = sigma_clipped_stats(data, sigma=self.sigma_clip)
        finder = DAOStarFinder(fwhm=self.fwhm, threshold=self.lower_snr * std)
        sources = finder(data - median)

        coordinates = np.transpose(np.array([sources["xcentroid"].data, sources["ycentroid"].data]))
        peaks = sources["peak"]

        return coordinates, peaks

    def citations(self):
        return "photutils", "numpy"

    @staticmethod
    def doc():
        return """photutils_ :code:`DAOStarFinder`."""


class SegmentedPeaks(StarsDetection):
    """
    Stars detection based on image segmentation.

    |write| ``Image.stars_coords``

    Parameters
    ----------
    threshold : float, optional
        threshold factor for which to consider pixel as potential sources, by default 1.5
    n_stars : int, optional
        maximum number of stars to consider, by default None
    min_separation : float, optional
        minimum separation between sources, by default 5.0. If less than that, close sources are merged 
    sort : bool, optional
        wether to sort stars coordinates from the highest to the lowest intensity, by default True
    """
    @register_args
    def __init__(self, threshold=2, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def single_detection(self, data):
        threshold = self.threshold*np.nanstd(data.flatten()) + np.median(data.flatten())
        regions = regionprops(label(data > threshold), data)
        coordinates = np.array([region.weighted_centroid[::-1] for region in regions])
        fluxes = np.array([np.sum(region.intensity_image) for region in regions])

        return coordinates, fluxes

    def citations(self):
        return "numpy", "skimage"
        

class SEDetection(StarsDetection):
    """
    Source Extractor detection

    |write| ``Image.stars_coords``

    Parameters
    ----------
    threshold : float, optional
        threshold factor for which to consider pixel as potential sources, by default 1.5
    n_stars : int, optional
        maximum number of stars to consider, by default None
    min_separation : float, optional
        minimum separation between sources, by default 5.0. If less than that, close sources are merged 
    sort : bool, optional
        wether to sort stars coordinates from the highest to the lowest intensity, by default True
    """
    @register_args
    def __init__(self, threshold=1.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def single_detection(self, data):
        data = data.byteswap().newbyteorder()
        sep_data = extract(data, self.threshold*np.median(data))
        coordinates = np.array([sep_data["x"], sep_data["y"]]).T
        fluxes = np.array(sep_data["flux"])

        return coordinates, fluxes

    def citations(self):
        return "source extractor", "sep"


class Peaks(Block):

    @register_args
    def __init__(self, cutout=21, **kwargs):
        super().__init__(**kwargs)
        self.cutout = cutout

    def run(self, image, **kwargs):
        idxs, cuts = cutouts(image.data, image.stars_coords, size=self.cutout)
        image.peaks = np.ones(len(image.stars_coords)) * -1
        for i, cut in zip(idxs, cuts):
            image.peaks[i] = np.max(cut.data)