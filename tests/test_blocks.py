import inspect
import sys
from prose import example_image, Sequence, blocks
import numpy as np
import pytest
from prose.blocks.detection import _SourceDetection
from prose.blocks.centroids import _PhotutilsCentroid
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


@pytest.mark.parametrize("block", classes("prose.blocks.psf", _PSFModelBase))
def test_psf_blocks(block):
    block().run(image_psf)


def test_detection_min_separation():
    from prose.blocks.detection import PointSourceDetection

    PointSourceDetection(min_separation=10.0)(image)


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
    from prose.core.source import Sources, PointSource

    im = image.copy()
    im.sources = Sources([PointSource(0, 0) for _ in range(2)])
    blocks.LimitSources().run(im)
    assert im.discard == True
