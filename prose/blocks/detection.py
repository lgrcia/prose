from skimage.measure import label, regionprops
import numpy as np
from ..utils import clean_stars_positions
from .. import Block, Image
from ..console_utils import info
from scipy.interpolate import interp1d
from ..core.source import *
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder

__all__ = [
    "DAOFindStars", 
    "SEDetection", 
    "AutoSourceDetection", 
    "PointSourceDetection"
]

class _SourceDetection(Block):
    """Base class for sources detection.
    """
    def __init__(
        self, 
        threshold=4, 
        n=None, 
        sort=True, 
        min_separation=None, 
        name=None, 
        min_area=0, 
        minor_length=0
    ):
        super().__init__(name=name)
        self.n = n
        self.sort = sort
        self.min_separation = min_separation
        self.threshold = threshold
        self.min_area = min_area
        self.minor_length = minor_length
        
    def clean(self, sources, *args):
        peaks = np.array([s.peak for s in sources])
        coords = np.array([s.coords for s in sources])
        _sources = sources.copy()
        
        if len(sources) > 0:
            if self.sort:
                idxs = np.argsort(peaks)[::-1]
                _sources = sources[idxs]
            if self.n is not None:
                _sources = _sources[0:self.n]
            if self.min_separation:
                idxs = clean_stars_positions(_sources, tolerance=self.min_separation)
                _sources = _sources[idxs]

            for i, s in enumerate(_sources):
                s.i = i
                
            return _sources

        else:
            return None
    
    # TODO: obsolete, redo
    def auto_threshold(self, image):
        if self.verbose:
            info("SegmentedPeaks threshold optimisation ...")
        thresholds = np.exp(np.linspace(np.log(1.5), np.log(500), 30))
        detected = [len(self._regions(image, t)) for t in thresholds]
        self.threshold = interp1d(detected, thresholds, fill_value="extrapolate")(self.n_stars)
        
        if self.verbose:
            info(f"threshold = {self.threshold:.2f}")
        
    def regions(self, image: Image, threshold=None):
        flat_data = image.data.flatten()
        median = np.median(flat_data)
        if threshold is None:
            threshold = self.threshold
                    
        flat_data = flat_data[np.abs(flat_data - median) < np.nanstd(flat_data)]
        threshold = threshold*np.nanstd(flat_data) + median
            
        regions = regionprops(label(image.data > threshold), image.data)
        regions = [r for r in regions if r.area > self.min_area and r.axis_minor_length > self.minor_length]
        
        return regions
    
class AutoSourceDetection(_SourceDetection):
    """Blah

    Parameters
    ----------
    _SourceDetection : _type_
        _description_
    """
    def __init__(
        self, 
        threshold=4, 
        n=None, 
        sort=True, 
        min_separation=None, 
        name=None, 
        min_area=0, 
        minor_length=0
    ):
        super().__init__(
            threshold=threshold, 
            n=n, 
            sort=sort, 
            min_separation=min_separation, 
            name=name, 
            min_area=min_area,
            minor_length=minor_length
            )

    def run(self, image):
        regions = self.regions(image)
        sources = np.array([auto_source(region) for region in regions])
        image.sources = Sources(self.clean(sources))
    
class PointSourceDetection(_SourceDetection):

    def __init__(
        self, 
        unit_euler=False, 
        min_area=3, 
        minor_length=2,
        **kwargs
    ):

        super().__init__(min_area=min_area, minor_length=minor_length, **kwargs)
        self.unit_euler = unit_euler
        self.min_area = min_area
        self.minor_length = minor_length
    
    def run(self, image):
        regions = self.regions(image)
        if self.unit_euler:
            idxs = np.flatnonzero([r.euler_number == 1 for r in regions])
            regions = [regions[i] for i in idxs]
            
        sources = np.array([PointSource(region) for region in regions])
        image.sources = PointSources(self.clean(sources))

    @property
    def citations(self):
        return "scikit-image", "scipy"


class TraceDetection(_SourceDetection):

    def __init__(self, min_length=5, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
    
    def run(self, image):
        regions = self.regions(image)
        regions = [r for r in regions if r.axis_major_length > self.min_length]
            
        sources = np.array([TraceSource(region) for region in regions])
        image.sources = Sources(sources)

    @property
    def citations(self):
        return "scikit-image", "scipy"

class LimitStars(Block):

    def __init__(self, min=4, max=10000, **kwargs):
        super().__init__(**kwargs)
        self.min = min
        self.max = max
        
    def run(self, image):
        n = len(image.sources) 
        if n == 0:
            image.discard = True
        else:
            if n < self.min or n > self.max:
                image.discard = True

# backward compatibility
class SegmentedPeaks(PointSourceDetection):

    def __init__(
        self, 
        unit_euler=False, 
        min_area=3, 
        minor_length=2,
        n_stars=None,
        **kwargs
    ):

        super().__init__(n=n_stars, min_area=min_area, minor_length=minor_length, **kwargs)
        self.unit_euler = unit_euler
        self.min_area = min_area
        self.minor_length = minor_length

# TODO
class Peaks(Block):

    
    def __init__(self, cutout=11, **kwargs):
        super().__init__(**kwargs)
        self.cutout = cutout

    def run(self, image, **kwargs):
        idxs, cuts = cutouts(image.data, image.sources.coords, size=self.cutout)
        peaks = np.ones(len(image.stars_coords)) * -1
        for j, i in enumerate(idxs):
            cut = cuts[j]
            if cut is not None:
                peaks[i] = np.max(cut.data)
        image.peaks = peaks

    @property
    def citations(self):
        return "photutils"

class _SimplePointSourceDetection(_SourceDetection):

    def __init__(
        self, 
        n_stars=None,
        min_separation=None, 
        sort=True,
        name=None
    ):
        super().__init__(n=n_stars, sort=sort,  min_separation=min_separation, name=name)

    def run(self, image):
        coordinates, peaks = self.detect(image)
        sources = np.array([PointSource(coords=c, peak=p) for c, p in zip(coordinates, peaks)])
        image.sources = PointSources(self.clean(sources))


class DAOFindStars(_SimplePointSourceDetection):
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
    
    def __init__(
        self, 
        sigma_clip=2.5, 
        lower_snr=5, 
        fwhm=5, 
        n_stars=None,
        min_separation=None, 
        sort=True,
        name=None
    ):
        super().__init__(n_stars=n_stars, sort=sort,  min_separation=min_separation, name=name)
        self.sigma_clip = sigma_clip
        self.lower_snr = lower_snr
        self.fwhm = fwhm

    def detect(self, image):
        _, median, std = sigma_clipped_stats(image.data, sigma=self.sigma_clip)
        finder = DAOStarFinder(fwhm=self.fwhm, threshold=self.lower_snr * std)
        sources = finder(image.data - median)

        coordinates = np.transpose(np.array([sources["xcentroid"].data, sources["ycentroid"].data]))
        peaks = sources["peak"]

        return coordinates, peaks

    @property
    def citations(self):
        return "photutils"

try:
    from sep import extract
except:
    raise AssertionError("sep not installed")


class SEDetection(_SimplePointSourceDetection):
    
    def __init__(
        self, 
        threshold=2.5,
        n_stars=None,
        min_separation=None, 
        sort=True,
        name=None
    ):
        """
        Source Extractor detection.

        |write| ``Image.sources`` and ``Image.peaks``

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
        
        super().__init__(n_stars=n_stars, sort=sort,  min_separation=min_separation, name=name)
        self.threshold = threshold

    def detect(self, image):
        data = image.data.byteswap().newbyteorder()
        sep_data = extract(image.data, self.threshold*np.median(data))
        coordinates = np.array([sep_data["x"], sep_data["y"]]).T
        fluxes = np.array(sep_data["flux"])

        return coordinates, fluxes

    @property
    def citations(self):
        return "source extractor", "sep"

class AlignSources(Block):
    pass #TODO