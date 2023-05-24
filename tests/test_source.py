from prose.simulations import example_image
from prose.blocks import PointSourceDetection


def region_parser(source, region):
    source.width = region.bbox[3] - region.bbox[1]
    source.height = region.bbox[2] - region.bbox[0]


def test_region_parser():
    image = example_image()

    PointSourceDetection(parser=region_parser).run(image)

    assert image.sources[0].width == 17
    assert image.sources[0].height == 17
