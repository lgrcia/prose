from skimage.measure import label, regionprops
import numpy as np
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from prose.pipeline.registration import clean_stars_positions
from prose.pipeline.base import Block


class StarsDetection(Block):
    """Base class for stars detection
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def single_detection(self, image):
        """
        Running detection on single or multiple images
        """
        raise NotImplementedError("method needs to be overidden")

    def run(self, image, data, return_coords=False):
        return self.single_detection(image, return_coords=return_coords)


class DAOFindStars(StarsDetection):
    
    def __init__(self, sigma_clip=2.5, lower_snr=5, fwhm=5, n_stars=None, min_separation=10, sort=True, **kwargs):

        super().__init__(**kwargs)
        self.sigma_clip = sigma_clip
        self.lower_snr = lower_snr
        self.fwhm = fwhm
        self.n_stars = n_stars
        self.min_separation = min_separation
        self.sort = sort

    def single_detection(self, image, return_coords=False):
        data = image.data
        mean, median, std = sigma_clipped_stats(data, sigma=self.sigma_clip)
        finder = DAOStarFinder(fwhm=self.fwhm, threshold=self.lower_snr * std)
        sources = finder(data - median)

        if self.n_stars is not None:
            sources = sources[np.argsort(sources["flux"])[::-1][0:self.n_stars]]
        elif self.sort:
            sources = sources[np.argsort(sources["flux"])[::-1]]

        positions = np.transpose(
            np.array([sources["xcentroid"].data, sources["ycentroid"].data])
        )

        if type(self.min_separation) is int:
            stars_coords = clean_stars_positions(positions, tolerance=self.min_separation)
        else:
            stars_coords = positions

        if return_coords:
            return stars_coords
        else:
            image.stars_coords = stars_coords


class SegmentedPeaks(StarsDetection):

    def __init__(self, threshold=2, min_separation=10, n_stars=None):
        super().__init__()
        self.threshold = threshold
        self.min_separation = min_separation
        self.n_stars = n_stars

    def single_detection(self, image, return_coords=False):
        data = image.data
        threshold = self.threshold*np.median(data)
        regions = regionprops(label(data > threshold), data)
        coordinates = np.array([region.weighted_centroid[::-1] for region in regions])
        if self.n_stars is not None:
            sorted_idx = np.argsort([np.sum(region.intensity_image) for region in regions])[::-1]
            coordinates = coordinates[sorted_idx][0:self.n_stars]
        
        coordinates = clean_stars_positions(coordinates, tolerance=self.min_separation)

        if return_coords:
            return coordinates
        else:
            image.stars_coords = coordinates

