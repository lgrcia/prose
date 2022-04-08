from skimage.measure import label, regionprops
import numpy as np
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from .registration import clean_stars_positions
from .. import Block
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
    def __init__(self, n_stars=None, sort=True, min_separation=None, **kwargs):
        super().__init__(**kwargs)
        self.n_stars = n_stars
        self.sort = sort
        self.min_separation = min_separation
        self.last_coords = None

    def clean(self, fluxes, coordinates, *args):

        if len(coordinates) > 2:
            if self.sort:
                idxs = np.argsort(fluxes)[::-1]
                coordinates = coordinates[idxs]
                for arg in args:
                    arg = [arg[i] for i in idxs]
            if self.n_stars is not None:
                coordinates = coordinates[0:self.n_stars]
            if self.min_separation is not None:
                coordinates = clean_stars_positions(coordinates, tolerance=self.min_separation)

            return [coordinates, fluxes, *args]

        else:
            return None, None


class DAOFindStars(StarsDetection):
    """
    DAOPHOT stars detection with :code:`photutils` implementation.

    |write| ``Image.stars_coords`` and ``Image.peaks``
    
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
        whether to sort stars coordinates from the highest to the lowest intensity, by default True
    """
    @register_args
    def __init__(self, sigma_clip=2.5, lower_snr=5, fwhm=5, **kwargs):
        super().__init__(**kwargs)
        self.sigma_clip = sigma_clip
        self.lower_snr = lower_snr
        self.fwhm = fwhm

    def run(self, image):
        _, median, std = sigma_clipped_stats(image.data, sigma=self.sigma_clip)
        finder = DAOStarFinder(fwhm=self.fwhm, threshold=self.lower_snr * std)
        sources = finder(image.data - median)

        coordinates = np.transpose(np.array([sources["xcentroid"].data, sources["ycentroid"].data]))
        peaks = sources["peak"]

        image.stars_coords, image.peaks =  self.clean(peaks, coordinates)

    def citations(self):
        return "photutils", "numpy"

class SegmentedPeaks(StarsDetection):
    """
    Stars detection based on image segmentation.

    |write| ``Image.stars_coords`` and ``Image.peaks``

    Parameters
    ----------
    threshold : float, optional
        threshold factor for which to consider pixel as potential sources, by default 1.5
    n_stars : int, optional
        maximum number of stars to consider, by default None
    min_separation : float, optional
        minimum separation between sources, by default 5.0. If less than that, close sources are merged 
    sort : bool, optional
        whether to sort stars coordinates from the highest to the lowest intensity, by default True
    """
    @register_args
    def __init__(self, unit_euler=False, threshold=4, min_area=3, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.unit_euler = unit_euler
        self.min_area = min_area

    def run(self, image):
        threshold = self.threshold*np.nanstd(image.data.flatten()) + np.median(image.data.flatten())
        regions = regionprops(label(image.data > threshold), image.data)
        if self.min_area is not None:
            regions = [r for r in regions if r.area > self.min_area]
        fluxes = np.array([np.sum(region.intensity_image) for region in regions])
        idxs = np.argsort(fluxes)[::-1][0:self.n_stars]
        regions = [regions[i] for i in idxs]
        fluxes = fluxes[idxs]

        if self.unit_euler:
            idxs = np.flatnonzero([r.euler_number == 1 for r in regions])
            regions = [regions[i] for i in idxs]
            fluxes = fluxes[idxs]
                    
        coordinates = np.array([region.weighted_centroid[::-1] for region in regions])

        image.stars_coords, image.fluxes =  self.clean(fluxes, coordinates)
        #image.regions =  regions

    def citations(self):
        return "numpy", "skimage"
        

class SEDetection(StarsDetection):
    """
    Source Extractor detection

    |write| ``Image.stars_coords`` and ``Image.peaks``

    Parameters
    ----------
    threshold : float, optional
        threshold factor for which to consider pixel as potential sources, by default 1.5
    n_stars : int, optional
        maximum number of stars to consider, by default None
    min_separation : float, optional
        minimum separation between sources, by default 5.0. If less than that, close sources are merged 
    sort : bool, optional
        whether to sort stars coordinates from the highest to the lowest intensity, by default True
    """
    @register_args
    def __init__(self, threshold=1.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def run(self, image):
        data = image.data.byteswap().newbyteorder()
        sep_data = extract(image.data, self.threshold*np.median(data))
        coordinates = np.array([sep_data["x"], sep_data["y"]]).T
        fluxes = np.array(sep_data["flux"])

        image.stars_coords, image.peaks =  self.clean(fluxes, coordinates)

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
        for j, i in enumerate(idxs):
            cut = cuts[j]
            if cut is not None:
                image.peaks[i] = np.max(cut.data)


class LimitStars(Block):

    def __init__(self, min=4, max=10000, **kwargs):
        super().__init__(**kwargs)
        self.min = min
        self.max = max
        
    def run(self, image):
        if image.stars_coords is None:
            image.discard = True
        else:
            n = len(image.stars_coords) 
            if n < self.min or n > self.max:
                image.discard = True