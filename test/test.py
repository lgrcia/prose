from prose import Reduction, AperturePhotometry, Observation, Telescope, FitsManager
import matplotlib.pyplot as plt
import os
import shutil
from os import path
import unittest
from prose.datasets import generate_prose_reduction_dataset

RAW = "synthetic_dataset"
_REDUCED = "_synthetic_dataset"
REDUCED = "synthetic_dataset"

PHOT = "../docs/source/notes/fake_telescope_20200229_prose_I+z.phot"


class TestReduction(unittest.TestCase):

    def test_reduction(self):
        
        generate_prose_reduction_dataset(RAW, n_images=3)
        
        Telescope({
            "name": "fake_telescope",
            "trimming": (0, 0),
            "latlong": [24.6275, 70.4044]
        })

        fm = FitsManager(RAW, depth=2)
        reduction = Reduction(fm, overwrite=True)
        reduction.run()

        photometry = AperturePhotometry(reduction.destination, overwrite=True)
        photometry.run()

        shutil.rmtree(reduction.destination)
        shutil.rmtree(RAW)


class TestObservation(unittest.TestCase):

    def test_diff(self):
        obs = Observation(PHOT)
        obs.target = 1
        df = obs.broeg2005()

    def test_plot(self):
        obs = Observation(PHOT)
        obs.plot()
        plt.close()


if __name__ == "__main__":

    unittest.main()