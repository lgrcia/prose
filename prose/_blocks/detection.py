from skimage.measure import label, regionprops
import numpy as np
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from prose._blocks.registration import clean_stars_positions
from prose._blocks.base import Block
from sep import extract


class StarsDetection(Block):
    """Base class for stars detection.
    """
    def __init__(self, n_stars=None, sort=True, min_separation=None, **kwargs):
        super().__init__(**kwargs)
        self.n_stars = n_stars
        self.sort = sort
        self.min_separation = min_separation
        self.last_coords = None

    def single_detection(self, image):
        """
        Running detection on single or multiple images
        """
        raise NotImplementedError("method needs to be overidden")

    def run(self, image, return_coords=False):
        coordinates, fluxes = self.single_detection(image)
        if self.sort:
            coordinates = coordinates[np.argsort(fluxes)[::-1]]
        if self.n_stars is not None:
            coordinates = coordinates[0:self.n_stars]
        if self.min_separation is not None:
            coordinates = clean_stars_positions(coordinates, tolerance=self.min_separation)

        image.stars_coords = coordinates

        self.last_coords = coordinates


class DAOFindStars(StarsDetection):
    """
    DAOPHOT stars detection with :code:`photutils` implementation.

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
    def __init__(self, sigma_clip=2.5, lower_snr=5, fwhm=5, n_stars=None, min_separation=10, sort=True, **kwargs):
        super().__init__(n_stars=n_stars, sort=sort, **kwargs)
        self.sigma_clip = sigma_clip
        self.lower_snr = lower_snr
        self.fwhm = fwhm
        self.min_separation = min_separation
        self.sort = sort

    def single_detection(self, image):
        data = image.data
        mean, median, std = sigma_clipped_stats(data, sigma=self.sigma_clip)
        finder = DAOStarFinder(fwhm=self.fwhm, threshold=self.lower_snr * std)
        sources = finder(data - median)

        coordinates = np.transpose(np.array([sources["xcentroid"].data, sources["ycentroid"].data]))
        fluxes = sources["flux"]

        return coordinates, fluxes

    def citations(self):
        return "photutils", "numpy"

    @staticmethod
    def doc():
        return """photutils_ :code:`DAOStarFinder`."""


class SegmentedPeaks(StarsDetection):
    """
    Stars detection based on image segmentation.

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

    def __init__(self, threshold=2, min_separation=5.0, n_stars=None, sort=True, **kwargs):
        super().__init__(n_stars=n_stars, sort=sort, **kwargs)
        self.threshold = threshold
        self.min_separation = min_separation

    def single_detection(self, image):
        data = image.data
        threshold = self.threshold*np.median(data)
        regions = regionprops(label(data > threshold), data)
        coordinates = np.array([region.weighted_centroid[::-1] for region in regions])
        fluxes = np.array([np.sum(region.intensity_image) for region in regions])

        return coordinates, fluxes

    def citations(self):
        return "numpy", "skimage"

    @staticmethod
    def doc():
        return """A fast detection algorithm which:

- segment the image in blobs with pixels above a certain threshold
- compute the centroids of individual blobs as stars positions

.. image:: images/segmentation.png
   :align: center
   :height: 260px

This method should be used when speed is required over accuracy. Uses scikit-image_."""


class SEDetection(StarsDetection):
    """
    Source Extractor detection

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
    def __init__(self, threshold=1.5, n_stars=None, min_separation=5.0, sort=True, **kwargs):
        super().__init__(n_stars=n_stars, sort=True, min_separation=min_separation, **kwargs)
        self.threshold = threshold
        self.min_separation = min_separation

    def single_detection(self, image, return_coords=False):
        sep_data = extract(image.data, self.threshold*np.median(image.data))
        coordinates = np.array([sep_data["x"], sep_data["y"]])
        fluxes = np.array(sep_data["flux"])

        return coordinates, fluxes

    def citations(self):
        return "source extractor", "sep"

