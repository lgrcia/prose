import inspect
import sys
import unittest
from prose import example_image, Sequence, Block, blocks
import numpy as np


def classes(module, sublcasses):
    class_members = inspect.getmembers(sys.modules[module], inspect.isclass)

    def mask(n, c):
        return issubclass(c, sublcasses) and n[0] != "_"

    return [c for n, c in class_members if mask(n, c)]


class TestBlocks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.blocks = []

    def load(self, module, subclasses):
        self.blocks = classes(module, subclasses)

    def test_all(self):
        for block in self.blocks:
            with self.subTest(block=block.__name__):
                block().run(self.image)


class TestBlocksDetection(TestBlocks):
    def __init__(self, *args, **kwargs):
        TestBlocks.__init__(self, *args, **kwargs)
        from prose.blocks.detection import _SourceDetection

        self.load("prose.blocks.detection", _SourceDetection)
        self.image = example_image()


class TestBlocksGeometry(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        from prose.blocks.detection import PointSourceDetection

        self.image = example_image()
        PointSourceDetection().run(self.image)

    def test_Trim(self):
        from prose.blocks.geometry import Trim

        im = Trim(30)(self.image)

    def test_Cutouts(self):
        from prose.blocks.geometry import Cutouts

        im = Cutouts()(self.image.copy())

        assert len(im._sources) == len(im.cutouts)

    def test_ComputeTransform(self):
        from prose.blocks.geometry import ComputeTransform

        im = ComputeTransform(self.image.copy())(self.image.copy())
        assert np.allclose(im.transform, np.eye(3))


class TestBlocksCentroids(TestBlocks):
    def __init__(self, *args, **kwargs):
        TestBlocks.__init__(self, *args, **kwargs)
        from prose.blocks.detection import PointSourceDetection
        from prose.blocks.centroids import _PhotutilsCentroid

        self.load("prose.blocks.centroids", _PhotutilsCentroid)
        self.image = example_image()
        self.image = PointSourceDetection()(self.image)

    def test_Balletentroid(self):
        from prose.blocks.centroids import CentroidBallet

        CentroidBallet()(self.image)


class TestBlocksPSF(TestBlocks):
    def __init__(self, *args, **kwargs):
        TestBlocks.__init__(self, *args, **kwargs)
        from prose.blocks.psf import _PSFModelBase

        self.load("prose.blocks.psf", _PSFModelBase)

        self.image = example_image()
        Sequence(
            [blocks.PointSourceDetection(), blocks.Cutouts(), blocks.MedianEPSF()]
        ).run(self.image)

    def test_MedianPSF(self):
        from prose.blocks.psf import MedianEPSF
        from prose.blocks.geometry import Cutouts

        im = Cutouts()(self.image)
        im = MedianEPSF()(im)
        
class TestObjectsCopy(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        im = example_image()
        self.image = blocks.PointSourceDetection()(im)
        self.image = blocks.Cutouts()(self.image)

    def test_copy_sources(self):
        from prose.core.source import Source, Sources
        sources = Sources([Source(coords=c) for c in np.random.rand(10, 2)])
        assert id(sources[0]) != id(sources.copy()[0])
        assert id(sources[0].coords) != id(sources.copy()[0].coords)

    def test_copy_image(self):
        self.image.a = 3
        image_copy = self.image.copy()
        image_copy.a = 5
        assert image_copy.a != self.image.a
        assert id(self.image.sources) != id(image_copy.sources)
        assert id(self.image.cutouts) != id(image_copy.cutouts)
        

class TestAlignment(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        from prose.blocks.detection import PointSourceDetection

        self.image = example_image()
        PointSourceDetection().run(self.image)

    def test_AlignReferenceSources(self):
        from prose.blocks.alignment import AlignReferenceSources

        im = AlignReferenceSources(self.image.copy())(self.image.copy())

class TestBlockGet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.image = example_image()
        self.image.a = 3
        self.image.b = 6
        self.image.fits_header = {"C": 42}
            
    def test_keyword(self):
        from prose.blocks import Get
        g = Get("a", "b", "keyword:C", arrays=False)
        g(self.image)
        assert g.values == {"a":[3], "b":[6], "c":[42]}