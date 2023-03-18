from prose import Telescope
from prose import CONFIG
from datetime import datetime

def test_creation(name="test_telescope"):
    Telescope(name=name, save=True)
    assert name in CONFIG.build_telescopes_dict().keys()

def test_header_date(name="test_telescope"):
    telescope = Telescope(name=name, save=True)
    telescope.date_string_format = '%Y:%m:%d:%H:%M:%S.%f'
    date = telescope.date('2023:02:16:19:38:54.250')
    assert date == datetime(2023, 2, 16, 19, 38, 54, 250000)