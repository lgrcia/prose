import os
import numpy as np
from astropy.io import fits
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from prose.pipeline_methods.alignment import clean_stars_positions
from prose import Observation
import matplotlib.pyplot as plt
from prose import utils
from astropy.io import fits


data = fits.getdata(phot.stack_fits)
threshold = 1.3*np.median(data)
regions = regionprops(label(data > threshold), data)
coordinates = np.array([region.weighted_centroid[::-1] for region in regions])

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(utils.z_scale(data), cmap='Greys_r')
plt.xlim(950, 1150)
plt.ylim(950, 1150)
plt.title("original image", loc="left")
plt.subplot(122)
plt.imshow(utils.z_scale(data), cmap='Greys_r')
for i, region in enumerate(regions):
    im = region.filled_image
    plt.imshow(np.ma.masked_where(im == 0, im), extent=(
        region.bbox[1],
        region.bbox[3],
        region.bbox[0],
        region.bbox[2],
    ), cmap="viridis_r", alpha=0.5)
    plt.title("segmentation and centroid", loc="left")
    plt.plot(*coordinates[i], "x", c="k", ms=11, label="centroid")
plt.xlim(950, 1150)
plt.ylim(950, 1150)
plt.tight_layout()
plt.savefig("segmentation.png")

from skimage.measure import label, regionprops


def segmented_peaks(data, threshold=2, min_separation=10, n_stars=None):
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


