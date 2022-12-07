from numpy import block
from prose import Observation, Telescope, FitsManager, viz, Sequence, blocks, tutorials
from prose.pipeline import Calibration, AperturePhotometry
import matplotlib.pyplot as plt
import shutil
import unittest
import shutil
import shutil
from pathlib import Path
from prose.reports import Report, Summary
from prose.tutorials import example_image
import numpy as np


RAW = Path("synthetic_dataset")
REDUCED = "test/synthetic_dataset"
PHOT = "/Users/lgrcia/data/test_data_prose/Io_2021-11-28_TOI-4508.01_g'.phot"

# Removing existing folders
if RAW.exists():
    shutil.rmtree(RAW)

TEST_FODLER = Path("./test_results")
if TEST_FODLER.exists():
    shutil.rmtree(TEST_FODLER)

# Creating new folders
TEST_FODLER.mkdir(exist_ok=True)

class TestSequence(unittest.TestCase):

    def test_pass_sequence(self):
        image = tutorials.example_image()
        s = Sequence([blocks.Pass()])
        s.run(image)

class TestReport(unittest.TestCase):

    OBS = Observation(PHOT)

    def test_summary_report(self):

        # The summary template
        summary = Summary(self.OBS)

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
        file.write(fm.__repr__())
        file.close()

    def test_files(self):
        from prose import tutorials
        destination = TEST_FODLER / "fake_observations"
        tutorials.disorganised_folder(destination)

        fm = FitsManager(destination)
        result_file = TEST_FODLER / "test_fm_files.txt"
        file = open(result_file, "w")
        file.write("\n".join(fm.files(type="dark", exposure=8, tolerance=1, path=True).path.values))
        file.close()

class TestReduction(unittest.TestCase):

    def test_pipeline_reduction(self):
        
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
        destination = fm.label(1)
        calib = Calibration(**fm.observation_files(1), overwrite=True)
        calib.run(fm.all_images, TEST_FODLER / destination)

        photometry = AperturePhotometry(
            files=calib.images,
            stack=calib.stack,
            overwrite=True
        )
        photometry.run(calib.phot)

        o = Observation(calib.phot)
        o.target = 0
        o.broeg2005()
        o.plot()
        o.save()

        assert o.target == 0, "should be 0"

        plt.savefig(TEST_FODLER / "test_reduction.png")
        plt.close()

        shutil.rmtree(RAW)

    def test_pipeline_empty_calibration(self):
    
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
        destination = fm.label(1)
        calib = Calibration(bias=[], darks=[], flats=[], overwrite=True)
        calib.run(fm.all_images, TEST_FODLER / destination)


    def test_master_calib_as_input(self):

        import numpy as np
        from prose.tutorials import simulate_observation
        from prose import Telescope

        time = np.linspace(0, 0.15, 100) + 2450000
        target_dflux = 1 + np.sin(time*100)*1e-2
        simulate_observation(time, target_dflux, RAW)

        _ = Telescope({
            "name": "A",
            "trimming": [40, 40],
            "pixel_scale": 0.66,
            "latlong": [31.2027, 7.8586],
            "keyword_light_images": "light"
        })

        # reduction
        # ---------
        from prose import FitsManager, Image, Sequence, blocks

        # ref
        fm = FitsManager(RAW, depth=2)

        calib1 = blocks.Calibration(darks=fm.all_darks, bias=fm.all_bias, flats=fm.all_flats)
        Image(data=calib1.master_bias).writeto(RAW/"bias")
        Image(data=calib1.master_dark).writeto(RAW/"darks")
        Image(data=calib1.master_flat).writeto(RAW/"flats")

        calib2 = blocks.Calibration(darks=str(RAW/"darks"), bias=str(RAW/"bias"), flats=str(RAW/"flats"))
        assert np.allclose(calib1.master_bias, calib2.master_bias)
        assert np.allclose(calib1.master_dark, calib2.master_dark)
        assert np.allclose(calib1.master_flat, calib2.master_flat)

    def test_sequence_reduction(self):

        # simulated data
        # --------------
        import numpy as np
        from prose.tutorials import simulate_observation
        from prose import Telescope

        time = np.linspace(0, 0.15, 100) + 2450000
        target_dflux = 1 + np.sin(time*100)*1e-2
        simulate_observation(time, target_dflux, RAW)

        _ = Telescope({
            "name": "A",
            "trimming": [40, 40],
            "pixel_scale": 0.66,
            "latlong": [31.2027, 7.8586],
            "keyword_light_images": "light"
        })

        # reduction
        # ---------
        from prose import FitsManager, Image, Sequence, blocks

        # ref
        fm = FitsManager(RAW, depth=2)
        ref = Image(fm.all_images[0])
        calibration = Sequence([
            blocks.Calibration(darks=fm.all_darks, bias=fm.all_bias, flats=fm.all_flats),
            blocks.Trim(),
            blocks.SegmentedPeaks(), # stars detection
            blocks.Cutouts(),                   # making stars cutouts
            blocks.MedianPSF(),                 # building PSF
            blocks.psf.Moffat2D(),              # modeling PSF
        ])

        calibration.run(ref, show_progress=False)

        # potometry
        photometry = Sequence([
            *calibration[0:-1],                
            blocks.psf.Moffat2D(reference=ref),
            blocks.detection.LimitStars(min=3),
            blocks.Twirl(ref.stars_coords),    
            blocks.Set(stars_coords=ref.stars_coords),
            blocks.AffineTransform(data=False, inverse=True),
            blocks.BalletCentroid(),                           
            blocks.PhotutilsAperturePhotometry(scale=ref.fwhm),
            blocks.Peaks(),
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
                ("time", "exposure"),
                ("time", "path"),
                ("time", "sky"),
                (("time", "apertures", "star"), "fluxes"),
                (("time", "apertures", "star"), "errors"),
                (("time", "apertures", "star"), "apertures_area"),
                (("time", "apertures", "star"), "apertures_radii"),
                (("time", "apertures"), "apertures_area"),
                (("time", "apertures"), "apertures_radii"),
                ("time", "annulus_rin"),
                ("time", "annulus_rout"),
                ("time", "annulus_area"),
                (("time", "star"), "peaks"),
                name="xarray"
            ),
            blocks.AffineTransform(stars=False, data=True),
            blocks.Stack(ref, name="stack"),
        ])

        photometry.run(fm.all_images)

        # diff flux
        obs = Observation(photometry.xarray.to_observation(photometry.stack.image, sequence=photometry))
        obs.target = 0
        obs.broeg2005()
        obs.plot()
        plt.savefig(TEST_FODLER / "sequence_reduction.png")
        obs.save(TEST_FODLER / "sequence_reduction.phot")



# class TestTFOPObservation(unittest.TestCase):

#     OBS = TFOPObservation(PHOT)

#     def __init__(self, *args, **kwargs):
#         unittest.TestCase.__init__(self, *args, **kwargs)

#     def test_tess_ids(self):
#         result_file = TEST_FODLER / "tess-properties.txt"
#         file = open(result_file, "w")
#         assert self.OBS.tic_id == '170789802', "wrong tic id"
#         assert self.OBS.gaia_from_toi == "4877792060261482752", "Wrong gaia from TOI"
#         file.write(f"- TIC-id: {self.OBS.tic_id }\n")
#         file.write(f"- Gaia from TOI: {self.OBS.gaia_from_toi}\n")
#         file.write(f"- TOI-label: {self.OBS.tfop_prefix}\n")

#     def test_gaia_from_toi(self):
#         assert self.OBS.gaia_from_toi == "4877792060261482752", "Wrong gaia from TOI"


class TestObservation(unittest.TestCase):

    OBS = Observation(PHOT)

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_properties(self):
        result_file = TEST_FODLER / "properties.txt"
        file = open(result_file, "w")
        file.write(f"- simbad: {self.OBS.simbad}\n")
        file.write(f"- label: {self.OBS.label}\n")
        file.write(f"- date: {self.OBS.date}\n")
        file.write(f"- night_date: {self.OBS.night_date}\n")
        file.write(f"- telescope_name: {self.OBS.stack.telescope.name}\n")
        file.write(f"- xlabel: {self.OBS.xlabel}\n")

    def test_set_gaia_target(self):
        result_file = TEST_FODLER / "test_set_gaia_target.png"
        gaia = "Gaia DR2 4877792060261482752"
        self.OBS.set_catalog_target("gaia", gaia)
        self.OBS.stack.show_cutout(self.OBS.target)
        self.OBS.stack.plot_catalog('gaia', label=True)
        plt.title(f"expected {gaia}")
        plt.savefig(result_file)
        plt.close()

    def test_diff(self):
        result_file = TEST_FODLER / "test_diff.png"
        self.OBS.target = 18
        self.OBS.broeg2005()
        self.OBS.plot()
        plt.savefig(result_file)
        plt.close()

    def test_plot(self):
        result_file = TEST_FODLER / "test_plot.png"
        self.OBS.plot()
        plt.savefig(result_file)
        plt.close()

    def test_plot_systeamtics(self):
        result_file = TEST_FODLER / "test_systematics.png"
        self.OBS.plot_systematics()
        plt.savefig(result_file)
        plt.close()

    def test_plot_comps_lcs(self):
        result_file = TEST_FODLER / "test_plot_comps_lcs.png"
        self.OBS.plot_comps_lcs()
        plt.savefig(result_file)
        plt.close()

    def test_show_stars(self):
        result_file = TEST_FODLER / "test_show_stars.png"
        self.OBS.show_stars()
        plt.savefig(result_file)
        plt.close()

    def test_plot_psf_model(self):
        result_file = TEST_FODLER / "test_plot_psf_model_target.png"
        self.OBS.plot_psf_model()
        plt.savefig(result_file)
        plt.close()
        result_file = TEST_FODLER / "test_plot_psf_model_other_star.png"
        self.OBS.plot_psf_model(star=125, model=blocks.psf.Moffat2D)
        plt.savefig(result_file)
        plt.close()

    def test_noise_stats(self):
        result_file = TEST_FODLER / "noise_stats.txt"
        file = open(result_file, "w")
        noise = self.OBS.noise_stats(verbose=False)
        file.write(f"white (pont2006)\t{noise['pont_white']:.3e}\n"
                   f"red (pont2006)\t{noise['pont_red']:.3e}\n"
                   f"white (binned)\t\t{noise['binned_white']:.3e}\n")

    def test_plate_solve(self):
        result_file = TEST_FODLER / "test_plate_solve.png"
        self.OBS.plate_solve()
        self.OBS.query_catalog('gaia')
        gaias = self.OBS.stack.catalogs["gaia"][["x", "y"]].values
        self.OBS.stack.show(frame=True)
        viz.plot_marks(*gaias.T, color="y")
        plt.savefig(result_file)
        plt.close()

class TestSourceAndDetection(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_sourcedetection(self):
        from prose.blocks.detection import AutoSourceDetection, PointSourceDetection, DAOFindStars, SEDetection, SegmentedPeaks
        import matplotlib.pyplot as plt
        from prose.tutorials import example_image

        im = example_image()

        classes = [AutoSourceDetection, PointSourceDetection, DAOFindStars, SEDetection, SegmentedPeaks]

        plt.figure(None, (len(classes)*5, 5))

        for i, c in enumerate(classes):
            ax = plt.subplot(1, len(classes), i+1)
            im2 = c()(im)
            im2.show(ax=ax)
            ax.set_title(c.__name__)

        result_file = TEST_FODLER / "test_detection_blocks.png"
        plt.tight_layout()
        plt.savefig(result_file)

    def test_source_orientation(self):
        from prose.blocks.detection import AutoSourceDetection
        import matplotlib.pyplot as plt
        from prose.tutorials import source_example

        im = source_example()
        im = AutoSourceDetection()(im)
        print(im.sources[2].orientation, np.arctan2(40, 30))
        computed = np.rad2deg(im.sources[2].orientation)
        expected = np.rad2deg(np.arctan2(40, 30))
        im.show(stars=False)
        for s in im.sources:
            plt.plot(*s.vertexes.T, c="k")
            plt.plot(*s.co_vertexes.T, c="k")
            s.aperture(1.1, True).plot(color="yellow")
            s.annulus(1, 1.2).plot(color="r")
        plt.tight_layout()
        plt.savefig(TEST_FODLER / "test_source_orientation.png")
        print(computed, expected)
        assert np.abs(computed - expected) < 1, ""

class TestArgsRegistration(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()