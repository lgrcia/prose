from prose import Observation, Telescope, FitsManager, viz
from prose.pipeline import Calibration, AperturePhotometry
import matplotlib.pyplot as plt
import shutil
import unittest
import os
import shutil
from os import path
import shutil
from pathlib import Path
from prose import Telescope, blocks, load
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

    def test_summary_report(self):

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

        time = np.linspace(0, 0.15, 10) + 2450000
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
        OBS.stack.show(frame=True)
        viz.plot_marks(*gaias.T, color="y")
        plt.savefig(result_file)
        plt.close()

if __name__ == "__main__":
    unittest.main()