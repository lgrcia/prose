from prose import Observation, Telescope, FitsManager
from prose.pipeline import Calibration, AperturePhotometry
import matplotlib.pyplot as plt
import os
import shutil
from os import path
import unittest
from prose.datasets import generate_prose_reduction_dataset
from prose.blocks.centroids import BalletCentroid, OldNNCentroid

RAW = "synthetic_dataset"
_REDUCED = "_synthetic_dataset"
REDUCED = "synthetic_dataset"

PHOT = "../test.phot"


class TestReduction(unittest.TestCase):

    def test_reduction(self):
        
        generate_prose_reduction_dataset(RAW, n_images=3)
        
        Telescope({
            "name": "fake_telescope",
            "trimming": (0, 0),
            "latlong": [24.6275, 70.4044]
        })

        fm = FitsManager(RAW, depth=2)
        destination = fm.obs_name
        # reduction = Calibration(images=fm.images, overwrite=True)
        reduction = Calibration(**fm.images_dict, overwrite=True)

        reduction.run(destination)

        photometry = AperturePhotometry(destination, overwrite=True, centroid=OldNNCentroid)
        photometry.run(destination=PHOT)

        shutil.rmtree(reduction.destination)
        shutil.rmtree(RAW)


class TestObservation(unittest.TestCase):

    def test_diff(self):
        obs = Observation(PHOT)
        obs.target = 1
        obs.broeg2005()

    def test_plot(self):
        obs = Observation(PHOT)
        obs.plot()
        plt.close()


if __name__ == "__main__":

    unittest.main()