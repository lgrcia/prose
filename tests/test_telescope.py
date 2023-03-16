from prose import Telescope
from prose import CONFIG


def test_creation(name="test_telescope"):
    Telescope(name=name, save=True)
    assert name in CONFIG.build_telescopes_dict().keys()
