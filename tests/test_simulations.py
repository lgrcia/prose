import shutil

from prose import Telescope, FitsManager, FITSImage
from prose.simulations import ObservationSimulation


def test_additional_keywords(destination="./fits"):
    obs = ObservationSimulation(600, Telescope.from_name("A"))
    obs.set_psf((3.5, 3.5), 45, 4)
    obs.add_stars(10, [0, 1])
    obs.save_fits(destination, POINT="30, 40", SITE="112, 30")

    fm = FitsManager(destination)
    for image in fm.all_images:
        im = FITSImage(image)
        assert im.header["POINT"] == "30, 40"
        assert im.header["SITE"] == "112, 30"

    shutil.rmtree(destination)
