import numpy as np

from prose import Sequence, blocks, example_image

image = example_image()
Sequence([blocks.PointSourceDetection(), blocks.Cutouts()]).run(image)


def test_copy_sources():
    from prose.core.source import Source, Sources

    sources = Sources([Source(coords=c) for c in np.random.rand(10, 2)])
    assert id(sources[0]) != id(sources.copy()[0])
    assert id(sources[0].coords) != id(sources.copy()[0].coords)


def test_copy_image():
    image.a = 3
    image_copy = image.copy()
    image_copy.a = 5
    assert image_copy.a != image.a
    assert id(image.sources) != id(image_copy.sources)
    assert id(image.cutouts) != id(image_copy.cutouts)
