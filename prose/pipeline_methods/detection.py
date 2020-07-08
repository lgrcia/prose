from skimage.measure import label, regionprops
import os
import numpy as np
from astropy.io import fits
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from prose.pipeline_methods.alignment import clean_stars_positions


class StarsDetection:
    """Base class for stars detection
    """
    def __init__(self):
        pass

    def single_detection(self, image):
        """
        Running detection on single or multiple images
        """
        raise NotImplementedError("method needs to be overidden")

    def run(self, images):
        positions = []
        if isinstance(images, list):
            for image in images:
                positions.append(self.single_detection(image))
        else:
            positions = self.single_detection(images)

        return positions

class DAOFindStars(StarsDetection):
    
    def __init__(
        self,
        sigma_clip=2.5,
        lower_snr=5,
        fwhm=5,
        n_stars=None,
        min_separation=10,
        sort=True
    ):

        self.sigma_clip = sigma_clip
        self.lower_snr = lower_snr
        self.fwhm = fwhm
        self.n_stars = n_stars
        self.min_separation = min_separation
        self.sort = sort

    def single_detection(self, image):
        if isinstance(image, np.ndarray):
            data = image
        elif isinstance(image, str):
            if os.path.exists(image) and data.lower().endswith((".fts", ".fits")):
                data = fits.getdata(image)
        else:
            raise ValueError("{} should be a numpy array or a fits file")

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
            return clean_stars_positions(positions, tolerance=self.min_separation)
        else:
            return positions


class SegmentedPeaks(StarsDetection):

    def __init__(self, threshold=2, min_separation=10, n_stars=None):
        self.threshold = threshold
        self.min_separation = min_separation
        self.n_stars = n_stars

    def single_detection(self, image):
        if isinstance(image, np.ndarray):
            data = image
        elif isinstance(image, str):
            if os.path.exists(image) and image.lower().endswith((".fts", ".fits")):
                data = fits.getdata(image)
        else:
            raise ValueError("{} should be a numpy array or a fits file")
        threshold = self.threshold*np.median(data)
        regions = regionprops(label(data > threshold), data)
        coordinates = np.array([region.weighted_centroid[::-1] for region in regions])
        if self.n_stars is not None:
            sorted_idx = np.argsort([np.sum(region.intensity_image) for region in regions])[::-1]
            coordinates = coordinates[sorted_idx][0:self.n_stars]
        return clean_stars_positions(coordinates, tolerance=self.min_separation)

