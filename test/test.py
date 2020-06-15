from prose import pipeline, Photometry, Telescope, FitsManager
from prose.pipeline_methods.alignment import AstroAlignShift
import matplotlib.pyplot as plt
import os
import shutil
from os import path
import unittest

RAW = "test/minimal_quatar2b_dataset"
_REDUCED = "test/_minimal_quatar2b_dataset_reduced"
REDUCED = "test/minimal_quatar2b_dataset_reduced"


class TestReduction(unittest.TestCase):

    def test_reduction(self):

        _ = Telescope({
            "name": "test",
            "trimming": [40, 40],
            "pixel_scale": 0.66,
            "latlong": [31.2027, 7.8586]
        })

        fm = FitsManager(RAW, depth=2, )
        reduction = pipeline.Reduction(fits_manager=fm)
        reduction.set_observation(0, check_calib_telescope=False)

        reduction.run(_REDUCED, overwrite=True)

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