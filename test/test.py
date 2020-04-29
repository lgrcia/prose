from prose import pipeline
from prose import Photometry
import matplotlib.pyplot as plt
import os
import shutil
from os import path
import unittest

RAW = "test_data/minimal_quatar2b_dataset"
_REDUCED = "test_data/_minimal_quatar2b_dataset_reduced"
REDUCED = "test_data/minimal_quatar2b_dataset_reduced"

class TestReduction(unittest.TestCase):

    def test_reduction(self):
        reduction = pipeline.Reduction(RAW, deepness=2)
        reduction.set_observation(0, check_calib_telescope=False)

        reduction.run(_REDUCED, save_gif=True, overwrite=True)

        photometry = pipeline.Photometry(_REDUCED)
        photometry.run(overwrite=True)

        shutil.rmtree(_REDUCED)

class TestPhotometry(unittest.TestCase):

    def test_diff_from_phots(self):
        phot = Photometry(REDUCED)
        phot.target["id"] = 0
        phot.Broeg2005()
        phot.save()

    def test_plot_lc(self):
        phot = Photometry(REDUCED)
        phot.lc.plot()
        plt.close()


if __name__ == "__main__":
    # check folder exist
    if path.exists(_REDUCED):
        shutil.rmtree(_REDUCED)

    unittest.main()