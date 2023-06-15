import numpy as np
import pytest

from prose import FITSImage
from prose.core.image import Buffer, Image
from prose.core.source import PointSource, Sources
from prose.simulations import fits_image


def test_init_append(n=5):
    buffer = Buffer(n)
    init = np.random.randint(0, 20, size=20)
    buffer.init(init)
    np.testing.assert_equal(
        buffer.items[buffer.mid_index + 1 :], init[: buffer.mid_index]
    )
    buffer.append(4)
    assert buffer.items[-1] == 4


def test_buffer_iter():
    buffer = Buffer(5)
    data = np.random.randint(0, 20, 20)
    buffer.init(data)
    for i, buf in enumerate(buffer):
        assert buf.current == data[i]


def test_cutout(coords=(0, 0)):
    image = Image(data=np.random.rand(100, 100))
    im = image.cutout(coords, 5, wcs=False)
    assert im.data.shape == (5, 5)


def test_data_cutouts():
    image = Image(data=np.random.rand(100, 100))
    coords = np.random.rand(10, 2)
    cutouts = image.data_cutouts(coords, 5)


def test_plot_sources():
    image = Image(data=np.random.rand(100, 100))
    image.sources = Sources([PointSource(coords=(0, 0), i=i) for i in range(5)])
    image.show()
    image.sources[[0, 1, 3]].plot()
    image.sources[0].plot()
    # seen in a bug
    image.sources[np.int64(0)].plot()


def test_fitsimage(tmp_path):
    filename = tmp_path / "test.fits"
    fits_image(np.random.rand(100, 100), {}, filename)

    loaded_image = FITSImage(filename)
    assert "IMAGETYP" in dict(loaded_image.header)
