from prose import Reduction, Photometry, PhotProducts, Telescope, FitsManager
import matplotlib.pyplot as plt
import os
import shutil
from os import path
import unittest
from prose.datasets import generate_prose_reduction_datatset

RAW = "synthetic_dataset"
_REDUCED = "_synthetic_dataset"
REDUCED = "synthetic_dataset"


class TestReduction(unittest.TestCase):

    def test_reduction(self):
        
        generate_prose_reduction_datatset(RAW, n_images=5)
        
        Telescope({
            "name": "fake_telescope",
            "trimming": (0, 0),
            "latlong": [24.6275, 70.4044]
        })

        fm = FitsManager(RAW, depth=2)
        reduction = Reduction(fm, overwrite=True)
        reduction.run()

        photometry = Photometry(reduction.destination, overwrite=True)
        photometry.run()

        shutil.rmtree(reduction.destination)


class TestPhotometry(unittest.TestCase):

    def test_diff_from_phots(self):
        phot = PhotProducts(REDUCED)
        phot.target["id"] = 0
        phot.Broeg2005()
        phot.save()

    def test_plot_lc(self):
        phot = PhotProducts(REDUCED)
        phot.lc.plot()
        plt.close()


if __name__ == "__main__":
    # check folder exist
    if path.exists(_REDUCED):
        shutil.rmtree(_REDUCED)

    unittest.main()