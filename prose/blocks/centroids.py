from photutils.centroids import centroid_sources, centroid_2dg, centroid_quadratic, centroid_com
from .. import Block
import numpy as np
from os import path
from prose import CONFIG
from .psf import cutouts
import warnings
from astropy.utils.exceptions import AstropyUserWarning


__all__ = [
    "COM",
    "Gaussian2D",
    "Quadratic",
    "BalletCentroid",
]

TF_LOADED = False

try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
except ModuleNotFoundError:
    TF_LOADED = True


class _PhotutilsCentroid(Block):

    
    def __init__(self, centroid_func, limit=None, cutout=21, **kwargs):
        """Photutils centroiding

        Parameters
        ----------
        centroid_func : function
            photutils.centroids function
        limit : int, optional
            maximum deviation from initial coordinate, by default `cutout/2`
        cutout : int, optional
            size of the cutout to be used for centroiding, by default 21
        """
        super().__init__(**kwargs)
        self.cutout = cutout
        self.centroid_func = centroid_func
        if limit is None:
            limit = cutout/2

        self.limit = limit

    def run(self, image):
        # *%+#@ photutils check (see photutils.centroids.core code...)
        in_image = np.all(image.stars_coords < image.shape[::-1] - (1, 1), axis=1)
        in_image = np.logical_and(in_image, np.all(image.stars_coords > (0, 0), axis=1))
        x, y = image.stars_coords[in_image].T.copy()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyUserWarning)
            centroid_stars_coords = np.array(centroid_sources(
                image.data, x, y, box_size=self.cutout, centroid_func=self.centroid_func
            )).T

        stars_coords = image.stars_coords.copy()
        stars_coords[in_image] = centroid_stars_coords
        in_limit = np.linalg.norm(image.stars_coords - stars_coords, axis=1) < self.limit
        image.stars_coords[in_limit] = stars_coords[in_limit]

    @property
    def citations(self):
        return "photutils"

class COM(_PhotutilsCentroid):
    """Centroiding using ``photutils.centroids.centroid_com``
    
    |read| ``Image.stars_coords``

    |write| ``Image.stars_coords``

    Parameters
    ----------
    limit : int, optional
        maximum deviation from initial coordinate, by default `cutout/2`
    cutout : int, optional
        size of the cutout to be used for centroiding, by default 21 
    """

    
    def __init__(self, **kwargs):
        super().__init__(centroid_func=centroid_com, **kwargs)

class Gaussian2D(_PhotutilsCentroid):
    """Centroiding using ``photutils.centroids.centroid_2dg``
    
    |read| ``Image.stars_coords``

    |write| ``Image.stars_coords``

    Parameters
    ----------
    limit : int, optional
        maximum deviation from initial coordinate, by default `cutout/2`
    cutout : int, optional
        size of the cutout to be used for centroiding, by default 21 
    """

    
    def __init__(self, **kwargs):
        super().__init__(centroid_func=centroid_2dg, **kwargs)

class Quadratic(_PhotutilsCentroid):
    """Centroiding using ``photutils.centroids.centroid_quadratic``
    
    |read| ``Image.stars_coords``
    
    |write| ``Image.stars_coords``

    Parameters
    ----------
    limit : int, optional
        maximum deviation from initial coordinate, by default `cutout/2`
    cutout : int, optional
        size of the cutout to be used for centroiding, by default 21 
    """
    
    
    def __init__(self, **kwargs):
        super().__init__(centroid_func=centroid_quadratic, **kwargs)


class _CNNCentroid(Block):

    def __init__(self, cutout=15, filename=None, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.model = None
        self.cutout = cutout
        self.x, self.y = np.indices((cutout, cutout))

    def import_and_check_model(self):
        model_file = path.join(CONFIG.folder_path, self.filename)

        if path.exists(model_file):
            self.build_model()
            self.model.load_weights(model_file)
        else:
            raise AssertionError("Still on dev, contact lgrcia")

    def build_model(self):
        raise NotImplementedError()

    def run(self, image, **kwargs):
        initial_positions = image.stars_coords.copy()
        stars_in, stars = cutouts(image.data.copy(), initial_positions.copy(), self.cutout)
        if len(stars_in) > 0:
            # Take normalised and reshaped data of stars that are not None
            stars_data_reshaped = []
            pos_int = []
            for i in stars_in:
                im = stars[i]
                stars_data_reshaped.append((im.data / np.max(im.data)).reshape(self.cutout, self.cutout, 1))
                pos_int.append([im.bbox.ixmin, im.bbox.iymin])
            stars_data_reshaped = np.array(stars_data_reshaped)
            pos_int = np.array(pos_int)
            current_stars_coords = image.stars_coords[stars_in].copy()
            # apply model
            aligned_stars_coords = pos_int + self.model(stars_data_reshaped, training=False).numpy()[:, ::-1]
            # if coords is nan (any of x, y), keep old coord
            nan_mask = np.any(np.isnan(aligned_stars_coords), 1)
            aligned_stars_coords[nan_mask] = current_stars_coords[nan_mask]
            # change image.stars_coords
            image.stars_coords[stars_in] = aligned_stars_coords

    @property
    def citations(self):
        return "tensorflow"

class BalletCentroid(_CNNCentroid):
    """Centroiding with  `ballet <https://github.com/lgrcia/ballet>`_.

    |write| ``Image.stars_coords``
    """
    
    def __init__(self, **kwargs):
        super().__init__(cutout=15, filename="centroid.h5", **kwargs)
        self.import_and_check_model()

    def build_model(self):
        self.model = Sequential([
            Conv2D(64, (3, 3), activation='relu', input_shape=(self.cutout, self.cutout, 1),
                                  use_bias=True, padding="same"),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(128, (3, 3), activation='relu', use_bias=True, padding="same"),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(256, (3, 3), activation='relu', use_bias=True, padding="same"),
            Flatten(),
            Dense(2048, activation="sigmoid", use_bias=True),
            Dense(512, activation="sigmoid", use_bias=True),
            Dense(2),
        ])


class _OldNNCentroid(_CNNCentroid):

    
    def __init__(self, **kwargs):
        super().__init__(cutout=21, filename="oldcentroid.h5", **kwargs)
        self.import_and_check_model()

    def build_model(self):
        self.model = self.tf_models.Sequential([
            self.tf_layers.Conv2D(self.cutout, (3, 3), activation='relu',
                          input_shape=(self.cutout, self.cutout, 1)),
            self.tf_layers.MaxPooling2D((2, 2)),
            self.tf_layers.Conv2D(64, (3, 3), activation='relu'),
            self.tf_layers.MaxPooling2D((2, 2)),
            self.tf_layers.Conv2D(124, (3, 3), activation='relu'),
            self.tf_layers.Flatten(),
            self.tf_layers.Dense(2048, activation='relu'),
            self.tf_layers.Dense(2),
        ])




