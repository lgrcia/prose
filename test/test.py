import inspect
import sys
import unittest
from prose import simulations
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

class TestAperturePhotometry(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        t = np.linspace(0, 1, 20)
        self.true_y = np.sin(2*np.pi*t/0.5) + 1.

        np.random.seed(5)

        fluxes = np.array([self.true_y, *[np.ones(len(t)) for _ in range(20)]])
        _coords = np.random.rand(len(fluxes), 2)
        shape = (100, 100)
        coords = np.array([_coords*shape for _ in range(len(t))])
        coords[:, 0, :] = np.array(shape)/2

        self.images = simulations.simple_images(fluxes, coords, 1., shape=shape)

    def test_photometry(self):
        ref = self.images[0]
        ref = blocks.PointSourceDetection(False, 0, 0)(ref)

        def set_sources(im):
            im.sources = ref.sources.copy()

        calibration = Sequence([
            blocks.Apply(set_sources),
            blocks.AperturePhotometry(radii=[2.], scale=False),
            blocks.AnnulusBackground(scale=False),
            blocks.GetFluxes()
        ])

        calibration.run(self.images)
        fluxes = calibration[-1].fluxes
        fluxes.aperture = 0
        fluxes.target = 14
        assert np.allclose(self.true_y, fluxes.flux)