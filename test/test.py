from prose import Observation, Telescope, FitsManager, viz
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
import shutil
from os import path
import shutil
from pathlib import Path
from prose import Telescope, blocks, Sequence, load
from prose.simulations import fits_image, ObservationSimulation
from prose.reports import Report, Summary


RAW = Path("synthetic_dataset")
REDUCED = "synthetic_dataset"
PHOT = "/Users/lgrcia/data/test_data_prose/Io_2021-11-28_TOI-4508.01_g'.phot"
OBS = Observation(PHOT)

# Removing existing folders
if RAW.exists():
    shutil.rmtree(RAW)

TEST_FODLER = Path("./test_results")
if TEST_FODLER.exists():
    shutil.rmtree(TEST_FODLER)

# Creating new folders
TEST_FODLER.mkdir(exist_ok=True)

class TestReport(unittest.TestCase):

    def test_manual_reduction(self):

        # The summary template
        summary = Summary(OBS)

        # The report
        report = Report([summary])
        report.make(TEST_FODLER / "report")
        report.compile()



class TestFitsManager(unittest.TestCase):

    def test_fits_manager(self):
        from prose import tutorials
        destination = TEST_FODLER / "fake_observations"
        tutorials.disorganised_folder(destination)

        fm = FitsManager(destination)
        result_file = TEST_FODLER / "test_fits_manager.txt"
        file = open(result_file, "w")
        file.write(fm.print(repr=True))
        file.close()

    def test_files(self):
        from prose import tutorials
        destination = TEST_FODLER / "fake_observations"
        tutorials.disorganised_folder(destination)

        fm = FitsManager(destination)
        result_file = TEST_FODLER / "test_fm_files.txt"
        file = open(result_file, "w")
        file.write("\n".join(fm.files(imtype="dark", exposure=8, exposure_tolerance=1)))
        file.close()

class TestReduction(unittest.TestCase):

    def test_reduction(self):
        
        # generate dataset
        import numpy as np
        from prose.tutorials import simulate_observation

        time = np.linspace(0, 0.15, 100) + 2450000
        target_dflux = 1 + np.sin(time*100)*1e-2
        simulate_observation(time, target_dflux, RAW)
        
        Telescope({
            "name": "fake_telescope",
            "trimming": (0, 0),
            "latlong": [24.6275, 70.4044]
        })

        fm = FitsManager(RAW, depth=2)
        destination = fm.obs_name
        calib = Calibration(**fm.observation_files(0), overwrite=True)
        calib.run(fm.images, TEST_FODLER / destination)

        photometry = AperturePhotometry(
            files=calib.images,
            stack=calib.stack,
            overwrite=True
        )
        photometry.run(calib.phot)

        o = load(calib.phot)
        o.target = 0
        o.broeg2005()
        o.plot()
        o.save()

        assert o.target == 0, "should be 0"

        plt.savefig(TEST_FODLER / "test_reduction.png")
        plt.close()

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

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_properties(self):
        result_file = TEST_FODLER / "properties.txt"
        file = open(result_file, "w")
        file.write(f"- simbad: {OBS.simbad}\n")
        file.write(f"- label: {OBS.label}\n")
        file.write(f"- date: {OBS.date}\n")
        file.write(f"- night_date: {OBS.night_date}\n")
        file.write(f"- telescope_name: {OBS.stack.telescope.name}\n")
        file.write(f"- xlabel: {OBS.xlabel}\n")

    def test_tess_ids(self):
        result_file = TEST_FODLER / "tess-properties.txt"
        file = open(result_file, "w")
        assert OBS.tic_id == '170789802', "wrong tic id"
        assert OBS.gaia_from_toi == "4877792060261482752", "Wrong gaia from TOI"
        file.write(f"- TIC-id: {OBS.tic_id }\n")
        file.write(f"- Gaia from TOI: {OBS.gaia_from_toi}\n")
        file.write(f"- TOI-label: {OBS.tfop_prefix}\n")

    def test_set_gaia_target(self):
        result_file = TEST_FODLER / "test_set_gaia_target.png"
        gaia = "Gaia DR2 4877792060261482752"
        OBS.set_catalog_target("gaia", gaia)
        OBS.target = 18
        OBS.stack.show_cutout(OBS.target)
        OBS.stack.plot_catalog('gaia', label=True)
        plt.title(f"expected {gaia}")
        plt.savefig(result_file)
        plt.close()

    def test_diff(self):
        result_file = TEST_FODLER / "test_diff.png"
        OBS.target = 18
        OBS.broeg2005()
        OBS.plot()
        plt.savefig(result_file)
        plt.close()

    def test_plot(self):
        result_file = TEST_FODLER / "test_plot.png"
        OBS.plot()
        plt.savefig(result_file)
        plt.close()

    def test_plot_systeamtics(self):
        result_file = TEST_FODLER / "test_systematics.png"
        OBS.plot_systematics()
        plt.savefig(result_file)
        plt.close()

    def test_plot_comps_lcs(self):
        result_file = TEST_FODLER / "test_plot_comps_lcs.png"
        OBS.plot_comps_lcs()
        plt.savefig(result_file)
        plt.close()

    def test_show_stars(self):
        result_file = TEST_FODLER / "test_show_stars.png"
        OBS.show_stars()
        plt.savefig(result_file)
        plt.close()

    def test_plot_psf_model(self):
        result_file = TEST_FODLER / "test_plot_psf_model_target.png"
        OBS.plot_psf_model()
        plt.savefig(result_file)
        plt.close()
        result_file = TEST_FODLER / "test_plot_psf_model_other_star.png"
        OBS.plot_psf_model(star=110, model=blocks.Moffat2D)
        plt.savefig(result_file)
        plt.close()

    def test_noise_stats(self):
        result_file = TEST_FODLER / "noise_stats.txt"
        file = open(result_file, "w")
        noise = OBS.noise_stats(verbose=False)
        file.write(f"white (pont2006)\t{noise['pont_white']:.3e}\n"
                   f"red (pont2006)\t{noise['pont_red']:.3e}\n"
                   f"white (binned)\t\t{noise['binned_white']:.3e}\n")

    def test_gaia_from_toi(self):
        assert OBS.gaia_from_toi == "4877792060261482752", "Wrong gaia from TOI"

    def test_plate_solve(self):
        result_file = TEST_FODLER / "test_plate_solve.png"
        OBS.plate_solve()
        OBS.query_catalog('gaia')
        gaias = OBS.stack.catalogs["gaia"][["x", "y"]].values
        OBS.stack.show(vmin=False, frame=True)
        viz.plot_marks(*gaias.T, color="y")
        plt.savefig(result_file)
        plt.close()

if __name__ == "__main__":
    unittest.main()