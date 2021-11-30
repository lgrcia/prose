from photutils.centroids import centroid_sources, centroid_2dg
from prose import Block
import numpy as np
from os import path
from prose import CONFIG
from .psf import cutouts


class Centroid2dg(Block):
    """Centroiding from  ``photutils.centroids.centroid_2dg``
    
    |write| ``Image.stars_coords``
    """

    def __init__(self, cutout=21, **kwargs):
        super().__init__(**kwargs)
        self.cutout = cutout

    def run(self, image, **kwargs):
        x, y = image.stars_coords.T

        image.stars_coords = np.array(centroid_sources(
            image.data, x, y, box_size=self.cutout, centroid_func=centroid_2dg
        )).T

    @staticmethod
    def citations():
        return "photutils", "numpy"


class CNNCentroid(Block):
    def __init__(self, cutout=15, filename=None, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.model = None
        self.cutout = cutout
        self.x, self.y = np.indices((cutout, cutout))

    def import_and_check_model(self):
        try:
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
            import tensorflow as tf
            from tensorflow.keras import models as models, layers
            self.tf_models = models
            self.tf_layers = layers
        except ModuleNotFoundError:
            raise ModuleNotFoundError("CNN centroid methods require tensorflow to be installed")

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
            stars_data_reshaped = np.array([
                (im.data / np.max(im.data)).reshape(self.cutout, self.cutout, 1) for im in stars
            ])
            pos_int = np.array([[st.bbox.ixmin, st.bbox.iymin] for st in stars])
            current_stars_coords = image.stars_coords[stars_in].copy()
            # apply model
            aligned_stars_coords = pos_int + self.model(stars_data_reshaped, training=False).numpy()[:, ::-1]
            # if coords is nan (any of x, y), keep old coord
            nan_mask = np.any(np.isnan(aligned_stars_coords), 1)
            aligned_stars_coords[nan_mask] = current_stars_coords[nan_mask]
            # change image.stars_coords
            image.stars_coords[stars_in] = aligned_stars_coords

    @staticmethod
    def citations():
        return "tensorflow", "numpy"


class BalletCentroid(CNNCentroid):
    """Centroiding with  `ballet <https://github.com/lgrcia/ballet>`_.

    |write| ``Image.stars_coords``
    """

    def __init__(self, **kwargs):
        super().__init__(cutout=15, filename="centroid.h5", **kwargs)
        self.import_and_check_model()

    def build_model(self):
        self.model = self.tf_models.Sequential([
            self.tf_layers.Conv2D(64, (3, 3), activation='relu', input_shape=(self.cutout, self.cutout, 1),
                                  use_bias=True, padding="same"),
            self.tf_layers.MaxPooling2D((2, 2), padding="same"),
            self.tf_layers.Conv2D(128, (3, 3), activation='relu', use_bias=True, padding="same"),
            self.tf_layers.MaxPooling2D((2, 2), padding="same"),
            self.tf_layers.Conv2D(256, (3, 3), activation='relu', use_bias=True, padding="same"),
            self.tf_layers.Flatten(),
            self.tf_layers.Dense(2048, activation="sigmoid", use_bias=True),
            self.tf_layers.Dense(512, activation="sigmoid", use_bias=True),
            self.tf_layers.Dense(2),
        ])


class OldNNCentroid(CNNCentroid):

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




