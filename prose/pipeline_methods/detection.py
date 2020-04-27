from skimage.measure import label, regionprops
import os
import numpy as np
from astropy.io import fits
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from prose.pipeline_methods.alignment import clean_stars_positions


def segmented_peaks(data, threshold=2, min_separation=10, n_stars=None, **kwargs):
    if isinstance(data, np.ndarray):
        pass
    elif isinstance(data, str):
        if os.path.exists(data) and data.lower().endswith((".fts", ".fits")):
            data = fits.getdata(data)
    else:
        raise ValueError("{} should be a numpy array or a fits file")
    threshold = threshold*np.median(data)
    regions = regionprops(label(data > threshold), data)
    coordinates = np.array([region.weighted_centroid[::-1] for region in regions])
    if n_stars is not None:
        sorted_idx = np.argsort([np.sum(region.intensity_image) for region in regions])[::-1]
        coordinates = coordinates[sorted_idx][0:n_stars]
    return clean_stars_positions(coordinates, tolerance=min_separation)


def daofindstars(
    data,
    sigma_clip=2.5,
    lower_snr=5,
    fwhm=5,
    n_stars=None,
    min_separation=10,
    sort=True,
):
    if isinstance(data, np.ndarray):
        pass
    elif isinstance(data, str):
        if os.path.exists(data) and data.lower().endswith((".fts", ".fits")):
            data = fits.getdata(data)
    else:
        raise ValueError("{} should be a numpy array or a fits file")

    mean, median, std = sigma_clipped_stats(data, sigma=sigma_clip)
    finder = DAOStarFinder(fwhm=fwhm, threshold=lower_snr * std)
    sources = finder(data - median)

    if n_stars is not None:
        sources = sources[np.argsort(sources["flux"])[::-1][0:n_stars]]
    elif sort:
        sources = sources[np.argsort(sources["flux"])[::-1]]

    positions = np.transpose(
        np.array([sources["xcentroid"].data, sources["ycentroid"].data])
    )

    if type(min_separation) is int:
        return clean_stars_positions(positions, tolerance=min_separation)
    else:
        return positions
