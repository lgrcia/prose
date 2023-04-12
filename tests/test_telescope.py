from prose import Telescope, Image, FITSImage
from prose import CONFIG
from datetime import datetime
from astropy.io.fits import Header


def test_creation(name="test_telescope"):
    Telescope(name=name, save=True)
    assert name in CONFIG.build_telescopes_dict().keys()


def test_custom_header_date(tmp_path):
    im = Image()
    im.header = Header()

    keyword = "OBSTIME"
    value = "2023:02:16:19:38:54.250"

    im.header[keyword] = value
    im.writeto(tmp_path / "test.fits")

    telescope = Telescope(keyword_observation_date=keyword)
    telescope.date_string_format = "%Y:%m:%d:%H:%M:%S.%f"

    im = FITSImage(tmp_path / "test.fits", load_data=False, telescope=telescope)

    assert im.date == datetime(2023, 2, 16, 19, 38, 54, 250000)
