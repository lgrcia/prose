from prose import Observation, Telescope, FitsManager
from prose.pipeline import Calibration, AperturePhotometry
import matplotlib.pyplot as plt
import os
import shutil
from os import path
import unittest
from prose.datasets import generate_prose_reduction_dataset
from prose.blocks.centroids import BalletCentroid, OldNNCentroid
from astropy.time import Time
from astropy.io.fits import Header
import datetime
import os
from os import path
import shutil
from prose import viz
from prose import Telescope, blocks, Sequence
from prose.simulations import fits_image, ObservationSimulation

RAW = "synthetic_dataset"
_REDUCED = "_synthetic_dataset"
REDUCED = "synthetic_dataset"

PHOT = "../test2.phot"


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
        calib = Calibration(**fm.images_dict, overwrite=True)

        calib.run(destination)

        photometry = AperturePhotometry(
            files=calib.images,
            stack=calib.stack,
            overwrite=True
        )
        photometry.run(calib.phot_path)

        # shutil.rmtree(calib.destination)
        shutil.rmtree(RAW)

    def test_manual_reduction(self):

        try:
            shutil.rmtree(REDUCED)
        except:
            pass

        try:
            shutil.rmtree(RAW)
        except:
            pass


        import numpy as np
        import matplotlib.pyplot as plt

        # Generating data
        # ---------------

        n = 5
        N = 300
        time = np.linspace(0, 0.15, n)
        target_dflux = 1 + np.sin(time * 100) * 1e-2

        # Creating the observation
        np.random.seed(40)
        obs = ObservationSimulation(600, Telescope.from_name("A"))
        obs.set_psf((3.5, 3.5), 45, 4)
        obs.add_stars(N, time)
        obs.set_target(0, target_dflux)
        obs.positions += np.random.uniform(-10, 10, (2, n))[np.newaxis, :]

        # Cleaning the field
        obs.remove_stars(np.argwhere(obs.fluxes.mean(1) < 20).flatten())
        obs.clean_around_target(50)
        obs.save_fits(RAW, calibration=True)

        # Reduction
        # ---------

        from prose import FitsManager

        fm = FitsManager(RAW, depth=2)
        calibration = Calibration(**fm.images_dict)
        calibration.run(REDUCED)

        plt.ion()

        # photometry
        # ----------

        stack_detection = Sequence([
            blocks.SegmentedPeaks(n_stars=50, threshold=1.05),  # stars detection
            blocks.Gaussian2D(cutout_size=51, name="psf"),
            blocks.ImageBuffer(name="buffer"),
        ], calibration.stack)

        stack_detection.run(show_progress=False)
        stack = stack_detection.buffer.image

        # plotting stack and detected stars
        viz.show_stars(stack.data, stack.stars_coords, size=8)

        photometry = Sequence([
            blocks.Set(stars_coords=stack.stars_coords, fwhm=stack.fwhm),
            blocks.AffineTransform(),
            blocks.ImageBuffer(),
            blocks.PhotutilsAperturePhotometry(),
            blocks.Get("fluxes")
        ], calibration.images,name="Photometry")

        photometry.run()

        shutil.rmtree(REDUCED)
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