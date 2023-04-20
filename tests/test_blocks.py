import inspect
import sys

import numpy as np
import pytest

from prose import Sequence, blocks, example_image
from prose.blocks.centroids import _PhotutilsCentroid
from prose.blocks.detection import _SourceDetection
from prose.blocks.psf import _PSFModelBase

image = blocks.PointSourceDetection()(example_image())
image_psf = image.copy()

Sequence([blocks.Cutouts(), blocks.MedianEPSF()]).run(image_psf)


def classes(module, sublcasses):
    class_members = inspect.getmembers(sys.modules[module], inspect.isclass)

    def mask(n, c):
        return issubclass(c, sublcasses) and n[0] != "_"

    return [c for n, c in class_members if mask(n, c)]


@pytest.mark.parametrize("block", classes("prose.blocks.detection", _SourceDetection))
def test_detection_blocks(block):
    block().run(image)


@pytest.mark.parametrize("block", classes("prose.blocks.centroids", _PhotutilsCentroid))
def test_centroids_blocks(block):
    block().run(image)


def test_centroid_ballet():
    tf = pytest.importorskip("tensorflow")
    from prose.blocks.centroids import CentroidBallet

    CentroidBallet().run(image_psf)


@pytest.mark.parametrize("block", classes("prose.blocks.psf", _PSFModelBase))
def test_psf_blocks(block):
    if "JAX" in block.__name__:
        pytest.importorskip("jax")
    block().run(image_psf)


@pytest.mark.parametrize("d", [10, 50, 80, 100])
def test_sourcedetection_min_separation(d):
    from prose.blocks.detection import PointSourceDetection

    PointSourceDetection(min_separation=d).run(image)

    distances = np.linalg.norm(
        image.sources.coords - image.sources.coords[:, None], axis=-1
    )
    distances = np.where(np.eye(distances.shape[0]).astype(bool), np.nan, distances)
    distances = np.nanmin(distances, 0)
    np.testing.assert_allclose(distances > d, True)


def test_Trim():
    blocks.Trim(30).run(image.copy())


def test_Cutouts():
    im = blocks.Cutouts()(image)
    assert len(im._sources) == len(im.cutouts)


def test_ComputeTransform():
    from prose.blocks.geometry import ComputeTransform

    im = ComputeTransform(image.copy())(image.copy())
    assert np.allclose(im.transform, np.eye(3))


def test_MedianPSF():
    im = image.copy()
    blocks.Cutouts().run(im)
    blocks.MedianEPSF().run(im)


def test_AlignReferenceSources():
    blocks.AlignReferenceSources(image.copy())(image.copy())


def test_Get():
    image = example_image()
    image.a = 3
    image.b = 6
    image.fits_header = {"C": 42}

    g = blocks.Get("a", "b", "keyword:C", arrays=False)
    g(image)
    assert g.values == {"a": [3], "b": [6], "c": [42]}


def test_LimitSources():
    from prose.core.source import PointSource, Sources

    im = image.copy()
    im.sources = Sources([PointSource(0, 0) for _ in range(2)])
    blocks.LimitSources().run(im)
    assert im.discard == True


def test_Del():
    im = image.copy()
    im.a = 3

    blocks.Del("a", "data").run(im)
    assert not "a" in im.computed
    assert im.data is None


def test_Apply():
    im = image.copy()
    im.a = 3

    def f(im):
        im.a += 1

    blocks.Apply(f).run(im)
    assert im.a == 4


def test_Calibration_with_arrays():
    from prose.blocks import Calibration

    im = image.copy()

    bias = np.ones_like(im.data) * 1
    dark = np.ones_like(im.data)
    flat = np.ones_like(im.data) * 0.5
    flat /= np.mean(flat)

    observed_flat = flat + bias + dark
    observed_dark = dark + bias

    # None
    expected = im.data
    Calibration().run(im)
    np.testing.assert_allclose(im.data, expected)

    # bias only
    im = image.copy()
    im.data = im.data + bias
    expected = im.data - bias
    Calibration(bias=bias).run(im)
    np.testing.assert_allclose(im.data, expected)

    # dark and bias only
    im = image.copy()
    im.data = im.data + bias + dark
    expected = im.data - bias - dark
    Calibration(darks=observed_dark, bias=bias).run(im)
    np.testing.assert_allclose(im.data, expected)

    # flat only
    im = image.copy()
    im.data = im.data * flat
    expected = im.data / flat
    Calibration(flats=flat).run(im)
    np.testing.assert_allclose(im.data, expected)

    # flat and bias only
    im = image.copy()
    im.data = (im.data * flat) + bias
    expected = (im.data - bias) / flat
    Calibration(bias=bias, flats=observed_flat).run(im)
    np.testing.assert_allclose(im.data, expected)

    # flat, dark and bias
    im = image.copy()
    im.data = (im.data * flat) + bias + dark
    expected = (im.data - bias - dark) / flat
    Calibration(bias=bias, flats=observed_flat, darks=observed_dark).run(im)
    np.testing.assert_allclose(im.data, expected)

    # empty lists and ndarray
    # this reproduce an observed bug
    im = image.copy()
    im.data = im.data + dark
    expected = im.data - dark
    Calibration(bias=np.array([], dtype=object), flats=[], darks=observed_dark).run(im)


def test_Calibration_with_files(tmp_path):
    from prose.blocks import Calibration

    im = image.copy()
    calib = image.copy()
    calib_path = tmp_path / "calib.fits"
    calib.writeto(calib_path)
    Calibration(bias=calib_path).run(im)
    Calibration(bias=[calib_path]).run(im)
    Calibration(bias=np.array([calib_path])).run(im)


def test_SortSources():
    im = image_psf.copy()
    blocks.SortSources().run(im)
    peaks = [s.peak for s in im.sources]
    assert np.all(peaks[:-1] >= peaks[1:])
