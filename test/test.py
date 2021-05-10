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
from pathlib import Path
from prose import viz
from prose import Telescope, blocks, Sequence, load
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

        o = load(calib.phot_path)
        o.target = 0
        o.broeg2005(cut=True)
        o.broeg2005(cut=False)

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
        REDUCED = Path(REDUCED)

        stack_path = REDUCED / "stack.fist"
        gif_path = REDUCED / "video.gif"
        reference = fm.images[len(fm.images)//2]

        calibration_block = blocks.Calibration(fm.darks, fm.flats, fm.bias),

        self.detection_s = Sequence([
            calibration_block,
            blocks.Trim(name="trimming"),
            blocks.SegmentedPeaks(n_stars=50, name="detection"),
            blocks.ImageBuffer(name="buffer")
        ], reference)

        self.detection_s.run(show_progress=False)

        reference = self.detection_s.buffer.image

        calibration = Sequence([
            calibration_block,
            blocks.Trim(skip_wcs=True),
            blocks.Flip(reference),
            blocks.SegmentedPeaks(n_stars=50),
            blocks._Twirl(reference.stars_coords, n=15),
            blocks.Moffat2D(),
            blocks.SaveReduced(REDUCED, overwrite=True),
            blocks.AffineTransform(stars=True, data=True),
            blocks.Stack(stack_path, header=reference.header, overwrite=True),
            blocks.Video(gif_path, name="video", from_fits=True),
            blocks.XArray(
                ("time", "jd_utc"),
                ("time", "bjd_tdb"),
                ("time", "flip"),
                ("time", "fwhm"),
                ("time", "fwhmx"),
                ("time", "fwhmy"),
                ("time", "dx"),
                ("time", "dy"),
                ("time", "airmass"),
                ("time", "exposure")
            )
        ], fm.images, name="Calibration")

        # photometry
        # ----------

        stack_detection = Sequence([
            blocks.SegmentedPeaks(n_stars=50, threshold=1.05),  # stars detection
            blocks.Moffat2D(cutout_size=51, name="psf"),
            blocks.ImageBuffer(name="buffer"),
        ], calibration.stack)

        stack_detection.run(show_progress=False)
        stack = stack_detection.buffer.image

        photometry = Sequence([
            blocks.Set(stars_coords=stack.stars_coords, fwhm=stack.fwhm),
            blocks.AffineTransform(),
            blocks.ImageBuffer(),
            blocks.Peaks(),
            blocks.PhotutilsAperturePhotometry(),
            blocks.XArray(
                (("time", "apertures", "star"), "fluxes"),
                (("time", "apertures", "star"), "errors"),
                (("time", "apertures", "star"), "apertures_area"),
                (("time", "apertures", "star"), "apertures_radii"),
                (("time", "star"), "sky"),
                (("time", "apertures"), "apertures_area"),
                (("time", "apertures"), "apertures_radii"),
                ("time", "annulus_rin"),
                ("time", "annulus_rout"),
                ("time", "annulus_area"),
                (("time", "star"), "peaks")
            )
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