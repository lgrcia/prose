from astropy.io.fits import Header

from prose import FitsManager, Image, Telescope


def test_empty_header(tmp_path):
    im = Image()
    im.writeto(tmp_path / "test.fits")

    FitsManager(tmp_path)


def test_custom_fm(tmp_path):
    im = Image()

    keyword = "FILT"
    value = "test_filter"

    im.header[keyword] = value
    im.writeto(tmp_path / "test.fits")

    telescope = Telescope(keyword_filter=keyword)

    fm = FitsManager(tmp_path, telescope=telescope)
    assert fm.observations().iloc[0]["filter"] == value
