from prose import Block
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from photutils.centroids import centroid_com, centroid_epsf, centroid_2dg
import time
from prose._blocks.psf import cutouts
from os import path


# TODO: create NNBlock

class NNCentroid(Block):
    """
    Centroiding of stars cutouts using a Convolutional Neural Network
    """

    def __init__(self, cutout_size=21, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.train_history = None
        self.cutout_size = cutout_size
        self.build_model()
        self.x, self.y = np.indices((cutout_size, cutout_size))

    def initialize(self, *args):
        self.load_model()

    def build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(self.cutout_size, (3, 3), activation='relu',
                                     input_shape=(self.cutout_size, self.cutout_size, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(self.cutout_size * 2, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(self.cutout_size * 4, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(2048, activation='relu'))
        self.model.add(layers.Dense(2))

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.Huber(),
                           metrics=['accuracy'])

    def load_model(self):
        self.model = tf.keras.models.load_model(path.join(__file__, '../trained_models/model_NNcentroid_prose'))

    def save_model(self):
        self.model.save_weights(path.join(__file__, '../trained_models/model_NNcentroid_prose'))

    def moffat2D_model(self, a, x0, y0, sx, sy, theta, b, beta):
        # https://pixinsight.com/doc/tools/DynamicPSF/DynamicPSF.html
        dx_ = self.x - x0
        dy_ = self.y - y0
        dx = dx_ * np.cos(theta) + dy_ * np.sin(theta)
        dy = -dx_ * np.sin(theta) + dy_ * np.cos(theta)

        return b + a / np.power(1 + (dx / sx) ** 2 + (dy / sy) ** 2, beta)

    def sigma_to_fwhm(self, beta):
        return 2 * np.sqrt(np.power(2, 1 / beta) - 1)

    def random_model_label(self, N=10000, flatten=False):

        images = []
        labels = []

        for i in range(N):
            a = np.random.uniform(800, 10000)
            x0, y0 = np.random.normal(self.cutout_size / 2, 3, 2)
            theta = np.random.normal(0, np.pi / 8)
            b = np.random.uniform(100, 400)
            beta = np.random.uniform(1, 8)
            sx = np.random.uniform(2.5, 7.5) / self.sigma_to_fwhm(beta)
            sy = np.random.normal(1, 0.3) * sx
            noise = np.random.normal(0, 20, (self.cutout_size, self.cutout_size))

            data = self.moffat2D_model(a, x0, y0, sx, sy, theta, b, beta) + noise
            data /= np.max(data)

            images.append(data.reshape(self.cutout_size, self.cutout_size, 1))
            labels.append([x0, y0])

        if N == 1 and flatten:
            return (np.array(images[0]), np.array(labels[0]))
        else:
            return (np.array(images), np.array(labels))

    def train_model(self, train=30000, test=10000, epochs=20):

        train_dataset = self.random_model_label(train)
        test_dataset = self.random_model_label(test)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)

        BATCH_SIZE = 100
        SHUFFLE_BUFFER_SIZE = 100

        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        test_dataset = test_dataset.batch(BATCH_SIZE)

        self.train_history = self.model.fit(train_dataset, epochs=epochs,
                                            validation_data=test_dataset)

    def show_example(self):
        data, label = self.random_model_label(1)
        label = label.flatten()
        _data = data[0].reshape(21, 21)
        pos = np.array(label[0]).flatten()
        plt.imshow(_data, cmap="Greys_r")
        t0_c2dg = time.time()
        pred_data = [data[0].reshape(1, 21, 21, 1)]
        t0_nn = time.time()
        pred = self.model(pred_data, training=False).numpy()
        tf_nn = time.time() - t0_nn
        plt.plot(*label[::-1], "x", c="k", label="true")
        plt.plot(*pred.T[::-1], "x", label="NNCentroid")
        t0_c2dg = time.time()
        c2dg = centroid_2dg(_data)
        tf_c2dg = time.time() - t0_c2dg
        plt.plot(*c2dg, "x", label="centroid_2dg")
        plt.legend()
        ax = plt.gca()
        plt.text(0.05, 0.05, "distance:\nNNCentroid: {:.1e} ({:.0f}x faster)\n2DGaussian: {:.1e}".format(
            np.sqrt(np.sum((label - pred) ** 2)),
            tf_c2dg / tf_nn,
            np.sqrt(np.sum((label - c2dg[::-1]) ** 2)),
        ), fontsize=11,
                 horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, c="w")

    def plot_history(self):
        assert self.train_history is not None, "No training history"
        plt.plot(self.train_history.history['accuracy'], label='accuracy')
        plt.plot(self.train_history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim()
        plt.legend(loc='lower right')

    def run(self, image):
        initial_positions = image.stars_coords.copy()
        stars_in, stars = cutouts(image.data.copy(), initial_positions.copy(), 21)
        # stars_data_reshaped = np.array(stars.data)
        # stars_data_reshaped /= np.max(stars.data, (1, 2))
        # stars_data_reshaped.reshape(1)
        stars_data_reshaped = np.array([(im.data / np.max(im.data)).reshape(21, 21, 1) for im in stars])
        pos_int = np.array([[st.bbox.ixmin, st.bbox.iymin] for st in stars])
        image.stars_coords[stars_in] = pos_int + self.model(stars_data_reshaped, training=False).numpy()[:, ::-1]

        # best = np.argsort(np.sqrt(np.sum((image.stars_coords[stars_in] - initial_positions[stars_in]) ** 2, 1)))
        # i = 0
        # plt.imshow(stars[best[i]].data, cmap="Greys_r")
        # plt.plot(*(initial_positions[stars_in] - pos_int)[::-1][best[i]], "x", label="initial position", c="k")
        # plt.plot(*(image.stars_coords[stars_in] - pos_int)[best[i]], "x", label="corrected_position", c="red")
        # plt.legend()
        # from prose import visualisation as viz
        # viz.show_stars(image.data, image.stars_coords)
        # t = 4
