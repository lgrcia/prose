from photutils.centroids import centroid_sources, centroid_2dg
from prose import Block
import numpy as np
from os import path
from prose import CONFIG
from .psf import cutouts


class Centroid2dg(Block):

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


class BalletCentroid(Block):

    def __init__(self, cutout=15, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.cutout = cutout
        self.x, self.y = np.indices((cutout, cutout))
        self.import_and_check_model()

    def import_and_check_model(self):
        try:
            import tensorflow as tf
            from tensorflow.keras import models as models, layers
        except ModuleNotFoundError:
            raise ModuleNotFoundError("BalletCentroid requires tensorflow to be installed")

        model_file = path.join(CONFIG.folder_path, "centroid.h5")

        if path.exists(model_file):
            self.model = models.Sequential([
                layers.Conv2D(64, (3, 3), activation='relu', input_shape=(self.cutout, self.cutout, 1),
                              use_bias=True, padding="same"),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(128, (3, 3), activation='relu', use_bias=True, padding="same"),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(256, (3, 3), activation='relu', use_bias=True, padding="same"),
                layers.Flatten(),
                layers.Dense(2048, activation="sigmoid", use_bias=True),
                layers.Dense(512, activation="sigmoid", use_bias=True),
                layers.Dense(2),
            ])

            self.model.load_weights(model_file)
        else:
            raise AssertionError("Still on dev, contact lgrcia")

    def run(self, image, **kwargs):
        initial_positions = image.stars_coords.copy()
        stars_in, stars = cutouts(image.data.copy(), initial_positions.copy(), self.cutout)
        stars_data_reshaped = np.array([
            (im.data / np.max(im.data)).reshape(self.cutout, self.cutout, 1) for im in stars
        ])
        pos_int = np.array([[st.bbox.ixmin, st.bbox.iymin] for st in stars])
        image.stars_coords[stars_in] = pos_int + self.model(stars_data_reshaped, training=False).numpy()[:, ::-1]

    @staticmethod
    def citations():
        return "tensorflow", "numpy"
