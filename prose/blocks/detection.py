from skimage.measure import label, regionprops
import numpy as np
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from .registration import clean_stars_positions
from .. import Block
from ..blocks.psf import cutouts
from ..console_utils import info
from scipy.interpolate import interp1d

try:
    from sep import extract
except:
    raise AssertionError("Please install sep")


# TODO: when __call__, if data is prose.Image then run normally (like Block.run(Image)), if data is Image.data return products
# TODO: min_separation delete stars too close, but do not leave one...


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

        if len(coordinates) > 0:
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
    unit_euler : bool, optional
        whether to impose the euler_number of the regions property to be one, by default False
    threshold : int, optional
        empirical stars detection threshold, by default 4
    min_area : int, optional
        minimum area (i.e. pixels above threshold) for a bright region to be considered a star, by default 3
    minor_length : int, optional
        minimum lenght (defined as the region axis_minor_length) for a bright region to be considered a star, by default 2
    reference : Image, optional
        a reference image on which to auto-compute the threshold, by default None. If provided, also provided n_stars that will serve as the target number of stars to detect
    auto : bool, optional
        auto-compute threshold for each image, by default False (This is very slow and should be True only outside of sequences)
    n_stars : int, optional
        number of stars to detect/keep, by default None (i.e. no constraint)
    sort : bool, optional
        whether to sort stars coordinates from the highest to the lowest intensity, by default True
    min_separation : float, optional
        minimum separation between sources, by default 5.0. If less than that, close sources are merged 
    verbose : bool, optional
        wether the block should be verbose, by default False


    Example
    -------

    .. jupyter-execute::

        from prose.tutorials import example_image
        
        image = example_image()

    The simplest way to run this detection block is

    .. jupyter-execute::

        from prose.blocks import SegmentedPeaks
        
        image = SegmentedPeaks()(image)
        image.show()


    The number of stars can be easily constrained with ``n_stars`` 

    .. jupyter-execute::

        image = SegmentedPeaks(n_stars=5)(image)
        image.show()


    The algorithm relies on the ``threshold`` parameter that can be
    auto-computed to reach a desired number of stars

    .. jupyter-execute::

        image = SegmentedPeaks(n_stars=50, auto=True, verbose=True)(image)
        image.show()


    however threshold optimisation is slow. When processing multiple images
    (in a ``Sequence`` for example) you can provide a reference image on
    which the threshold can be optimized once

    .. jupyter-execute::

        from tqdm.auto import tqdm
        
        print("threshold optimisation for multiple images")
        # -------------------------------------------------
        for _ in tqdm(range(3)):
            SegmentedPeaks(n_stars=15, auto=True)(image)
            
            
        print("threshold optimisation once")
        # ----------------------------------
        detection = SegmentedPeaks(n_stars=15, reference=image)
        
        for _ in tqdm(range(3)):
            detection(image)

    """
    
    def __init__(
        self, 
        unit_euler=False, 
        threshold=4, 
        min_area=3, 
        minor_length=2, 
        reference=None,
        auto=False,
        n_stars=None, 
        sort=True, 
        min_separation=None,
        verbose=False,
        **kwargs
    ):
        super().__init__(verbose=verbose, min_separation=min_separation, sort=sort, n_stars=n_stars, **kwargs)
        self.threshold = threshold
        self.unit_euler = unit_euler
        self.min_area = min_area
        self.minor_length = minor_length
        self.auto = auto
            
        if reference is not None:
            assert self.n_stars is not None, "n_stars must be provided when reference is provided"
            self._auto_threshold(reference)
            
    def _auto_threshold(self, image):
        if self.verbose:
            info("SegmentedPeaks threshold optimisation ...")
        thresholds = np.exp(np.linspace(np.log(1.5), np.log(500), 30))
        detected = [len(self._regions(image, t)) for t in thresholds]
        self.threshold = interp1d(detected, thresholds, fill_value="extrapolate")(self.n_stars)
        
        if self.verbose:
            info(f"threshold = {self.threshold:.2f}")
        
    def _regions(self, image, threshold=None):
        flat_data = image.data.flatten()
        median = np.median(flat_data)
        if threshold is None:
            threshold = self.threshold
                    
        flat_data = flat_data[np.abs(flat_data - median) < np.nanstd(flat_data)]
        threshold = threshold*np.nanstd(flat_data) + median
            
        regions = regionprops(label(image.data > threshold), image.data)
        regions = [r for r in regions if r.area > self.min_area and r.axis_minor_length > self.minor_length]
        
        return regions

    def run(self, image):
        if self.auto:
            self._auto_threshold(image)
            
        regions = self._regions(image)
        
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

    def citations(self):
        return "numpy", "skimage", "scipy"

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